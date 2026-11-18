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
//!   zero-padded to whole 64-row panels. The driver reads column-major A in place
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
//! strides.
//!
//! # Online packing of A and prefetching
//!
//! - `a_pack`: online packing of A. Null means off. When it is non-null it
//!   points at a writable buffer of at least `ceil(m / 64) · 64 · kc`
//!   elements that overlaps neither A, B, C nor the argument block. The
//!   kernel then reads A with the given `a` / `a_depth` / `a_panel` while it
//!   computes column panel 0 and stores what it loads into the buffer in the
//!   plain packed layout; for every later column panel it reads A from the
//!   buffer (`a_depth = 64`, `a_panel = 64 · kc`). After the call the first
//!   `ceil(m / 64) · 64 · kc` elements of the buffer are, bit for bit, the
//!   plain packed block (`A(i, p)` at `(i / 64) · 64 · kc + p · 64 + i % 64`,
//!   `+0` for the rows `m ≤ i < ceil(m / 64) · 64` of the last panel),
//!   nothing beyond them is written, and C is exactly what the same call
//!   without `a_pack` computes. A call with `n ≤ NR` still fills the buffer.
//! - `prefetch`: prefetch switches ([`PREFETCH_OFF`] = 0). With
//!   [`PREFETCH_C`] set, the kernel issues prefetch hints for the live part
//!   of each tile's C columns before the tile's depth loop. Hints never
//!   change a result, a flag or any memory. The other bits are reserved and
//!   must be 0.
//!
//! Both macro-kernels honour both fields; with the "off" values they
//! compute a plain packed block.

use core::mem::offset_of;
use core::ops::{Add, Mul};

/// Rows of one row panel of A and of one ZA tile group: four 16-row `f32`
/// tiles or eight 8-row `f64` tiles.
pub(crate) const MR: usize = 64;

/// `prefetch` value that turns every prefetch off.
pub(crate) const PREFETCH_OFF: u64 = 0;

/// `prefetch` bit 0: prefetch the live part of each tile's C columns before
/// the tile's depth loop.
pub(crate) const PREFETCH_C: u64 = 1;

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
    /// Online packing target for A (module docs); null = off.
    pub(crate) a_pack: *mut T,
    /// Prefetch switches ([`PREFETCH_C`]); [`PREFETCH_OFF`] = off.
    pub(crate) prefetch: u64,
}

impl<T: SmeElement> MacroArgs<T> {
    /// A block in the plain packed layouts (`a_depth = 64`,
    /// `a_panel = 64 * kc`, `b_panel = kc * NR`), with the epilogue chosen
    /// from `alpha` and `beta`, online packing and prefetch off.
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

    /// The block with A read in place from a column-major matrix:
    /// `A(i, p) = a[i + p * lda]`, that is `a_depth = lda` and
    /// `a_panel = 64` (rows `≥ m` do not exist; the kernel never reads them).
    pub(crate) fn with_a_in_place(mut self, a: *const T, lda: usize) -> Self {
        self.a = a;
        self.a_depth = lda;
        self.a_panel = MR;
        self
    }

    /// The block with online packing of A into `a_pack` (module docs); a
    /// null pointer turns it off.
    pub(crate) fn with_online_pack(mut self, a_pack: *mut T) -> Self {
        self.a_pack = a_pack;
        self
    }

    /// The block with the prefetch switches `prefetch` ([`PREFETCH_C`] or
    /// [`PREFETCH_OFF`]).
    pub(crate) fn with_prefetch(mut self, prefetch: u64) -> Self {
        self.prefetch = prefetch;
        self
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
/// executable definition of what a macro-kernel computes (module docs). With
/// online packing on (`a_pack` non-null) it also writes the packed copy of
/// the A block, zero padding included, as the module docs define it.
///
/// # Safety
///
/// Every address that the layout of the module docs names for `args` must be
/// valid: A for `i < m`, `p < kc`; B for `p < kc`, `j < ceil(n / NR) * NR`;
/// C for `i < m`, `j < n` (C is read only when the mode reads it); and, when
/// `a_pack` is non-null, `ceil(m / 64) * 64 * kc` writable elements there
/// that overlap none of the others.
#[cfg(test)]
pub(crate) unsafe fn reference<T: SmeElement>(args: &MacroArgs<T>) {
    let nr = T::NR;
    if !args.a_pack.is_null() {
        for i in 0..args.m.div_ceil(MR) * MR {
            for p in 0..args.kc {
                let value = if i < args.m {
                    // SAFETY: inside the A block by the caller's contract.
                    unsafe {
                        *args
                            .a
                            .add((i / MR) * args.a_panel + p * args.a_depth + i % MR)
                    }
                } else {
                    T::ZERO
                };
                // SAFETY: inside the first `ceil(m / 64) * 64 * kc` elements
                // of the packing buffer, valid by the caller's contract.
                unsafe { *args.a_pack.add((i / MR) * MR * args.kc + p * MR + i % MR) = value };
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic, non-trivial value for element `index` of a buffer.
    fn value(index: usize, salt: u32) -> f32 {
        let mixed = (index as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(salt.wrapping_mul(40_503));
        (mixed >> 8) as f32 / 65_536.0 - 128.0
    }

    #[test]
    fn test_sme2_args_builders_set_only_their_fields() {
        let (a, b, mut c) = ([1.0f32; 4], [2.0f32; 16], [3.0f32; 4]);
        let plain =
            MacroArgs::<f32>::packed(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 4, 4, 1, 1, 1.0, 0.0);
        assert!(plain.a_pack.is_null());
        assert_eq!(plain.prefetch, PREFETCH_OFF);
        assert_eq!((plain.a_depth, plain.a_panel), (MR, MR));

        let mut copy = [0.0f32; 64];
        let block = plain
            .with_a_in_place(a.as_ptr(), 7)
            .with_online_pack(copy.as_mut_ptr())
            .with_prefetch(PREFETCH_C);
        assert_eq!(block.a, a.as_ptr());
        assert_eq!((block.a_depth, block.a_panel), (7, MR));
        assert_eq!(block.a_pack, copy.as_mut_ptr());
        assert_eq!(block.prefetch, PREFETCH_C);
        // Everything else is the plain block's.
        assert_eq!(
            (block.b, block.b_panel, block.c, block.ldc),
            (plain.b, plain.b_panel, plain.c, plain.ldc)
        );
        assert_eq!((block.m, block.n, block.kc), (plain.m, plain.n, plain.kc));
        assert_eq!(
            (block.alpha_bits, block.beta_bits, block.mode),
            (plain.alpha_bits, plain.beta_bits, plain.mode)
        );
        assert!(
            block
                .with_online_pack(core::ptr::null_mut())
                .a_pack
                .is_null()
        );
    }

    #[test]
    fn test_sme2_args_reference_online_pack_writes_plain_layout() {
        // 70 rows (two panels, the second with 6 live rows), depth 3, A read
        // in place with a leading dimension larger than the rows.
        let (m, n, kc, lda, ldc) = (70usize, 5usize, 3usize, 77usize, 71usize);
        let a: Vec<f32> = (0..lda * kc).map(|index| value(index, 1)).collect();
        let b: Vec<f32> = (0..kc * 16)
            .map(|index| if index % 16 < n { value(index, 2) } else { 0.0 })
            .collect();
        let panels = m.div_ceil(MR);
        let len = panels * MR * kc;
        let sentinel = f32::from_bits(0x7fa5_5a5a);
        let guard = 8;

        let mut c_plain = vec![0.5f32; ldc * n];
        let mut c_online = c_plain.clone();
        let mut copy = vec![sentinel; len + guard];

        let plain = MacroArgs::<f32>::packed(
            a.as_ptr(),
            b.as_ptr(),
            c_plain.as_mut_ptr(),
            ldc,
            m,
            n,
            kc,
            0.5,
            -0.25,
        )
        .with_a_in_place(a.as_ptr(), lda);
        let online = MacroArgs {
            c: c_online.as_mut_ptr(),
            ..plain
        }
        .with_online_pack(copy.as_mut_ptr());
        // SAFETY: A holds `lda * kc >= (kc - 1) * lda + m` elements, B one
        // whole zero-padded panel of depth `kc`, each C `ldc * n` elements
        // with `ldc >= m`, and the copy `len + guard` elements; none overlap.
        unsafe {
            reference(&plain);
            reference(&online);
        }

        // C does not depend on the packing.
        assert!(
            c_plain
                .iter()
                .zip(&c_online)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        // The copy is the plain packed block: A where the row exists, +0 in
        // the padding rows of the last panel, nothing beyond it.
        for i in 0..panels * MR {
            for p in 0..kc {
                let got = copy[(i / MR) * MR * kc + p * MR + i % MR];
                let want = if i < m { a[i + p * lda] } else { 0.0 };
                assert_eq!(got.to_bits(), want.to_bits(), "row {i}, depth {p}");
            }
        }
        assert!(
            copy[len..]
                .iter()
                .all(|x| x.to_bits() == sentinel.to_bits())
        );
    }
}
