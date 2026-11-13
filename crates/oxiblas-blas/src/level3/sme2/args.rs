//! The argument block of the SME2 macro-kernels and the kernel contract.
//!
//! # What one kernel call computes
//!
//! A macro-kernel computes one whole block `C[0..m, 0..n]` (column-major, an
//! `mc × nc` block of the blocked GEMM) from one packed A block (`m × kc`) and
//! one packed B block (`kc × n`), in a single streaming region: it loops over
//! column panels of `NR` columns (16 for `f32`, 8 for `f64`), inside them over
//! row panels of [`MR`] = 64 rows, runs the depth loop over `kc` for that
//! 64 × `NR` tile in ZA, and then applies the epilogue to the tile. The order
//! of the panel loops is free; every C element is computed independently.
//!
//! The blocked driver calls it once per `(pc, jc, ic)` step, in the order of
//! the NEON driver (`gemm.rs`: `pc` → `jc` → `ic`), with `beta_eff = beta` on
//! the first KC block and `1` afterwards.
//!
//! # Operand layout (all strides and sizes in ELEMENTS, never bytes)
//!
//! With `NR = T::NR` and `i < m`, `j < n`, `p < kc`:
//!
//! - **A** (`a`, `a_depth`, `a_panel`): `A(i, p) = a[(i / 64) * a_panel + p * a_depth + i % 64]`.
//!   The plain packed layout of `pack_a_trans_dispatch(.., mr = 64)` is
//!   `a_depth = 64`, `a_panel = 64 * kc` ([`MacroArgs::packed`]); it is
//!   zero-padded to whole 64-row panels. Wave 4 reads column-major A in place
//!   with `a_depth = lda`, `a_panel = 64`, where rows `≥ m` do not exist. So
//!   the kernel must never read `A(i, p)` for `i ≥ m` (predicate the loads,
//!   `whilelt … /z`) and must use both strides as given.
//! - **B** (`b`, `b_panel`): `B(p, j) = b[(j / NR) * b_panel + p * NR + j % NR]`,
//!   for `j < ceil(n / NR) * NR`. The packed layout of
//!   `pack_b_trans_dispatch(.., nr = NR)` is `b_panel = kc * NR`, zero-padded
//!   to whole panels: `B(p, j) = 0` for `n ≤ j < ceil(n / NR) * NR`. The kernel
//!   may read the padded columns (all-true B loads) but nothing beyond the
//!   last panel.
//! - **C** (`c`, `ldc`): `C(i, j) = c[i + j * ldc]`, `ldc ≥ m`. Rows `≥ m` and
//!   columns `≥ n` of the surrounding matrix are NEVER read or written.
//!
//! # The value of every C element (bitwise contract, rules R1–R4)
//!
//! `reference` (test builds) is the executable definition; `tests_contract.rs` proves it
//! equal to the NEON blocked path (`macro_panel_multiply_with` with the NEON
//! micro-kernels, edge tiles included) for every mode. Per element:
//!
//! ```text
//! acc = +0
//! for p in 0..kc:  acc = fma(A(i, p), B(p, j), acc)        one rounding each, p ascending
//! M0 (alpha == 1, beta == 0):  C = acc                      C is not read
//! M1 (alpha == 1, beta == 1):  C = acc + C
//! M2 (alpha != 1, beta == 0):  C = acc * alpha              C is not read
//! M3 (otherwise):              t = (alpha == 1) ? acc : acc * alpha
//!                              C = (beta == 1) ? t + C : fma(C, beta, t)
//! ```
//!
//! - **R1.** `kc` comes from the caller's blocking (the same KC partition as
//!   the NEON path), never from the kernel.
//! - **R2.** ZA starts at zero for every tile of every call: zero the tiles
//!   before each depth loop, never preload C into ZA, never split `p` across
//!   tiles or partial sums.
//! - **R3.** Combine like NEON: FMUL by alpha first, then store, FADD (`acc +
//!   C`) or FMLA (`t + C * beta`). An M3 kernel may multiply by alpha and use
//!   FMLA unconditionally: `acc * 1` is exact and `fma(C, 1, t)` equals
//!   `t + C`, so only NaN payloads can differ.
//! - **R4.** The mode is decided once in Rust ([`Epilogue::select`], with
//!   NEON's comparisons: `-0.0 == 0.0`, so `beta = -0` is M0/M2) and passed in
//!   `mode`; alpha and beta travel as bit patterns in `alpha_bits` /
//!   `beta_bits` (an `f32` zero-extended to 64 bits), because floating-point
//!   values never survive `smstart` in vector registers.
//!
//! FMOPA and the ZA-array FADD/FMLA force the default NaN, so NaN payloads may
//! differ from NEON; NaN positions do not. They also raise no floating-point
//! exception flags (the Arm ARM's `FPMulAdd_ZA` / `FPAdd_ZA` run with
//! exceptions off), so the flags a call leaves are those of its SVE vector
//! arithmetic only (see `asm.rs`), and can be fewer than NEON's. Flag parity
//! with NEON is not part of the promise; neither are flags raised by padded
//! lanes (for example `0 × ∞` against a zero-padded B column).
//!
//! # What the kernel may and may not assume
//!
//! It may assume: `m ≥ 1`, `n ≥ 1`, `kc ≥ 1`; SME2 with a 512-bit streaming
//! vector length (and FEAT_SME_F64F64 for `f64`) on the calling thread,
//! which `detect::thread_ready()` returned `true` for; every address the
//! layout above names is valid; C does not overlap A, B or the block;
//! `mode` matches `alpha_bits` / `beta_bits` as [`Epilogue::select`] gives it.
//!
//! It never writes the argument block: the block is read-only (a shared
//! `&MacroArgs`), so the kernel only loads from `x0` (the lint rejects a
//! store through `x0`).
//!
//! It must not assume: `m ≤ 64` or `n ≤ NR` or either being a multiple of
//! the panel size; `kc` being even or a multiple of anything; any pointer or
//! `ldc` alignment beyond the element's own; finite values in C (M0/M2 never
//! read it, so a NaN there must not leak); the plain-packing values of the
//! strides; anything about `a_pack` and `prefetch` other than their "off"
//! values below.
//!
//! # Fields reserved for Wave 4
//!
//! - `a_pack`: online packing of A. Null means off. When non-null (Wave 4
//!   only), the kernel stores the A panels it reads during column panel 0 to
//!   `a_pack` in the plain packed layout (zero-filled rows included) and reads
//!   them from there for the other column panels. Until Wave 4 every caller
//!   passes null and kernels do not read the field.
//! - `prefetch`: prefetch switches ([`PREFETCH_OFF`] = 0). Prefetching never
//!   changes results; until Wave 4 every caller passes 0 and kernels ignore
//!   the field.

// The block and its helpers are consumed by the kernels, the oracle, the
// self-test and the drivers of later waves; until the GEMM hook calls them
// (Wave 6) only tests use them. `expect` turns into a warning once that
// changes, so the attribute cannot outlive its reason.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the SME2 drivers once the GEMM hook lands"
    )
)]

use core::mem::offset_of;
use core::ops::{Add, Mul};

/// Rows of one row panel of A and of one ZA tile group: four 16-row `f32`
/// tiles or eight 8-row `f64` tiles.
pub(crate) const MR: usize = 64;

/// `prefetch` value that turns every prefetch off.
pub(crate) const PREFETCH_OFF: u64 = 0;

/// The element types of the SME2 backend: `f32` and `f64`.
pub(crate) trait SmeElement:
    Copy + PartialEq + Add<Output = Self> + Mul<Output = Self> + 'static
{
    /// Columns of one packed B panel: 16 (`f32`) or 8 (`f64`).
    const NR: usize;
    /// Rows of one ZA tile: 16 (`f32`) or 8 (`f64`).
    const TILE_ROWS: usize;
    /// `0`.
    const ZERO: Self;
    /// `1`.
    const ONE: Self;
    /// The bit pattern, zero-extended to 64 bits.
    fn bits(self) -> u64;
    /// The value of a bit pattern made by [`bits`](Self::bits) (an `f32`
    /// uses the low 32 bits).
    fn from_bits64(bits: u64) -> Self;
    /// `self * a + b` with one rounding.
    fn fused(self, a: Self, b: Self) -> Self;
    /// Whether the value is a NaN.
    fn nan(self) -> bool;
}

impl SmeElement for f32 {
    const NR: usize = 16;
    const TILE_ROWS: usize = 16;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bits64(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
    fn fused(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
}

impl SmeElement for f64 {
    const NR: usize = 8;
    const TILE_ROWS: usize = 8;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits64(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    fn fused(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
}

/// The epilogue of a kernel call, chosen in Rust from alpha and beta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub(crate) enum Epilogue {
    /// M0: `alpha == 1`, `beta == 0`: store the accumulator (C not read).
    Store = 0,
    /// M1: `alpha == 1`, `beta == 1`: `acc + C`.
    Add = 1,
    /// M2: `alpha != 1`, `beta == 0`: `acc * alpha` (C not read).
    Scale = 2,
    /// M3: everything else: `t = acc * alpha` (unless `alpha == 1`), then
    /// `t + C` when `beta == 1`, else `fma(C, beta, t)`.
    ScaleCombine = 3,
}

impl Epilogue {
    /// The mode for `alpha` and `beta`, with NEON's comparisons (`beta = -0`
    /// counts as zero; a NaN alpha or beta is "not one, not zero").
    pub(crate) fn select<T: SmeElement>(alpha: T, beta: T) -> Self {
        let alpha_one = alpha == T::ONE;
        if beta == T::ZERO {
            if alpha_one { Self::Store } else { Self::Scale }
        } else if alpha_one && beta == T::ONE {
            Self::Add
        } else {
            Self::ScaleCombine
        }
    }

    /// Whether the epilogue reads C.
    pub(crate) fn reads_c(self) -> bool {
        matches!(self, Self::Add | Self::ScaleCombine)
    }
}

/// The argument block of one macro-kernel call (see the module docs).
///
/// Every field is 8 bytes, so the offsets are the same for `f32` and `f64`
/// and a template reads any field with one `ldr x…, [x0, #{off}]` (`ldr w…`
/// for the low half of an `f32` bit pattern).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct MacroArgs<T> {
    /// A block: element `A(i, p)` is at `a + (i / 64) * a_panel + p * a_depth + i % 64`.
    pub(crate) a: *const T,
    /// Elements between depth steps `p` and `p + 1` of one A panel.
    pub(crate) a_depth: usize,
    /// Elements between consecutive 64-row A panels.
    pub(crate) a_panel: usize,
    /// Packed B block: `B(p, j)` is at `b + (j / NR) * b_panel + p * NR + j % NR`.
    pub(crate) b: *const T,
    /// Elements between consecutive NR-column B panels.
    pub(crate) b_panel: usize,
    /// `C(0, 0)` of the block; `C(i, j)` is at `c + i + j * ldc`.
    pub(crate) c: *mut T,
    /// Elements between consecutive columns of C (`ldc ≥ m`).
    pub(crate) ldc: usize,
    /// Rows of the block (`≥ 1`).
    pub(crate) m: usize,
    /// Columns of the block (`≥ 1`).
    pub(crate) n: usize,
    /// Depth of the block (`≥ 1`), the caller's KC step.
    pub(crate) kc: usize,
    /// Bit pattern of alpha (`f32`: zero-extended).
    pub(crate) alpha_bits: u64,
    /// Bit pattern of the effective beta (`f32`: zero-extended).
    pub(crate) beta_bits: u64,
    /// [`Epilogue`] as its discriminant (0–3).
    pub(crate) mode: u64,
    /// Online packing target for A (Wave 4); null = off.
    pub(crate) a_pack: *mut T,
    /// Prefetch switches (Wave 4); [`PREFETCH_OFF`] = off.
    pub(crate) prefetch: u64,
}

impl<T: SmeElement> MacroArgs<T> {
    /// A block in the plain packed layouts (`a_depth = 64`,
    /// `a_panel = 64 * kc`, `b_panel = kc * NR`), with the epilogue chosen
    /// from `alpha` and `beta` and the Wave-4 fields off.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn packed(
        a: *const T,
        b: *const T,
        c: *mut T,
        ldc: usize,
        m: usize,
        n: usize,
        kc: usize,
        alpha: T,
        beta: T,
    ) -> Self {
        Self {
            a,
            a_depth: MR,
            a_panel: MR * kc,
            b,
            b_panel: kc * T::NR,
            c,
            ldc,
            m,
            n,
            kc,
            alpha_bits: alpha.bits(),
            beta_bits: beta.bits(),
            mode: Epilogue::select(alpha, beta) as u64,
            a_pack: core::ptr::null_mut(),
            prefetch: PREFETCH_OFF,
        }
    }
}

/// Byte offset of `a`.
pub(crate) const OFF_A: usize = offset_of!(MacroArgs<f32>, a);
/// Byte offset of `a_depth`.
pub(crate) const OFF_A_DEPTH: usize = offset_of!(MacroArgs<f32>, a_depth);
/// Byte offset of `a_panel`.
pub(crate) const OFF_A_PANEL: usize = offset_of!(MacroArgs<f32>, a_panel);
/// Byte offset of `b`.
pub(crate) const OFF_B: usize = offset_of!(MacroArgs<f32>, b);
/// Byte offset of `b_panel`.
pub(crate) const OFF_B_PANEL: usize = offset_of!(MacroArgs<f32>, b_panel);
/// Byte offset of `c`.
pub(crate) const OFF_C: usize = offset_of!(MacroArgs<f32>, c);
/// Byte offset of `ldc`.
pub(crate) const OFF_LDC: usize = offset_of!(MacroArgs<f32>, ldc);
/// Byte offset of `m`.
pub(crate) const OFF_M: usize = offset_of!(MacroArgs<f32>, m);
/// Byte offset of `n`.
pub(crate) const OFF_N: usize = offset_of!(MacroArgs<f32>, n);
/// Byte offset of `kc`.
pub(crate) const OFF_KC: usize = offset_of!(MacroArgs<f32>, kc);
/// Byte offset of `alpha_bits`.
pub(crate) const OFF_ALPHA: usize = offset_of!(MacroArgs<f32>, alpha_bits);
/// Byte offset of `beta_bits`.
pub(crate) const OFF_BETA: usize = offset_of!(MacroArgs<f32>, beta_bits);
/// Byte offset of `mode`.
pub(crate) const OFF_MODE: usize = offset_of!(MacroArgs<f32>, mode);
/// Byte offset of `a_pack`.
pub(crate) const OFF_A_PACK: usize = offset_of!(MacroArgs<f32>, a_pack);
/// Byte offset of `prefetch`.
pub(crate) const OFF_PREFETCH: usize = offset_of!(MacroArgs<f32>, prefetch);

/// Writes the contract's value into every element of `C[0..m, 0..n]`: the
/// executable definition of what a macro-kernel computes (module docs).
///
/// # Safety
///
/// Every address that the layout of the module docs names for `args` must be
/// valid: A for `i < m`, `p < kc`; B for `p < kc`, `j < ceil(n / NR) * NR`;
/// C for `i < m`, `j < n` (C is read only when the mode reads it).
#[cfg(test)]
pub(crate) unsafe fn reference<T: SmeElement>(args: &MacroArgs<T>) {
    let nr = T::NR;
    let alpha_one = args.alpha_bits == T::ONE.bits();
    let (alpha, beta) = (
        T::from_bits64(args.alpha_bits),
        T::from_bits64(args.beta_bits),
    );
    for j in 0..args.n {
        for i in 0..args.m {
            let mut acc = T::ZERO;
            for p in 0..args.kc {
                // SAFETY: inside the A and B blocks by the caller's contract.
                let (a, b) = unsafe {
                    (
                        *args
                            .a
                            .add((i / MR) * args.a_panel + p * args.a_depth + i % MR),
                        *args.b.add((j / nr) * args.b_panel + p * nr + j % nr),
                    )
                };
                acc = a.fused(b, acc);
            }
            // SAFETY: inside the C block by the caller's contract.
            let c = unsafe { args.c.add(i + j * args.ldc) };
            let value = match args.mode {
                0 => acc,
                // SAFETY: as above; the mode reads C.
                1 => acc + unsafe { *c },
                2 => acc * alpha,
                _ => {
                    let t = if alpha_one { acc } else { acc * alpha };
                    // SAFETY: as above; the mode reads C.
                    let old = unsafe { *c };
                    if beta == T::ONE {
                        t + old
                    } else {
                        old.fused(beta, t)
                    }
                }
            };
            // SAFETY: inside the C block by the caller's contract.
            unsafe { *c = value };
        }
    }
}

/// The bitwise comparison of the promise: equal bits, or both NaN (payloads
/// may differ).
#[cfg(test)]
pub(crate) fn same_result<T: SmeElement>(got: T, want: T) -> bool {
    got.bits() == want.bits() || (got.nan() && want.nan())
}
