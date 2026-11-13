//! First-use self-test of the SME2 macro-kernels.
//!
//! [`self_test_with`] runs one macro-kernel on a handful of odd blocks and a
//! systematic sweep of small ones, in every epilogue mode, and compares every result with the scalar oracle
//! (`oracle.rs`), bit for bit except NaN payloads. It exists because the
//! kernels are verified under QEMU only: if the first SME2 machine they meet
//! (an Apple M4, say) rounds, flushes subnormals or orders an epilogue
//! differently, the self-test fails and the backend must stay off for the
//! process, so GEMM keeps the NEON results.
//!
//! # What it checks
//!
//! The fixed blocks ([`CASES`]) cover several 64-row and `NR`-column panels, partial
//! tiles in both directions, full tiles only, a single element, `kc = 1`,
//! deep and odd `kc`, a `kc` that is a multiple of 8 (as the caller's `KC`
//! usually is), a last row panel of 17 to 32 rows (two active 16-row tiles),
//! three or more column panels with `kc > 1` for either panel width, the
//! plain packed A layout and the in-place A layout (column-major,
//! `a_depth = lda`, `a_panel = 64`) with more than one row panel and with
//! leading dimensions other than `m + 2`, a B panel stride larger than
//! `kc · NR` (the gap between panels holds NaN), and one block of special
//! values: subnormal inputs, products and results, signed zeros and a few
//! infinities.
//!
//! The sweep ([`sweep_cases`], generic over the tile height and the panel
//! width of the element type) takes every path through the kernel: every
//! row variant (1 to 4 live 16-row tiles for `f32`, 1 to 8 live 8-row tiles
//! for `f64`), each with `kc = 1`, an even and an odd `kc` (two passes of the
//! double-step loop, without and with the remainder step), as the only row
//! panel and as the last one after a full panel, packed and in place, the
//! last live tile full or holding one row, each over a full and a partial
//! column panel. `tests_selftest.rs` computes the paths of every block from
//! this list and requires all of them (the path table); a change of a
//! kernel's structure must be reflected there.
//!
//! Each block runs with the six alpha/beta pairs of [`SCALARS`]:
//! M0, M1, M2 (with `beta = -0`) and M3 in its three branches (`alpha == 1`,
//! `beta == 1`, neither), all with inexact scalars. For every run it checks:
//!
//! - every element of `C[0..m, 0..n]` against the oracle (`same_bits_or_nan`);
//! - that the guard border around C (a whole column before and after the
//!   block, one element before `C(0, 0)`, rows `m..ldc` of every column) still
//!   holds its sentinel bits exactly;
//! - that in M0 and M2 a C filled with NaN does not leak into the result;
//! - that A and B are unchanged.
//!
//! Cost (`test_sme2_selftest_cost_matches_the_docs` pins these figures):
//!
//! | | blocks | kernel calls | largest run | all runs together |
//! |---|---|---|---|---|
//! | `f32`, [`CASES`] only (before the sweep) | 7 | 42 | 72 848 B | 1 274 592 B |
//! | `f64`, [`CASES`] only (before the sweep) | 7 | 42 | 143 744 B | 2 469 600 B |
//! | `f32` with the sweep | 7 + 48 | 330 | 72 848 B | 8 160 288 B |
//! | `f64` with the sweep | 7 + 96 | 618 | 143 744 B | 19 579 872 B |
//!
//! Each kernel call has one oracle call beside it. A run holds A, B and
//! their copies, C (with its guard border) and its two copies; the largest
//! run is the `84 × 40 × 16` in-place block of [`CASES`] (the sweep's blocks
//! are at most `128 × (2 · NR - 1) × 5`). The buffers of one run are freed
//! before the next, so the peak is the largest run; the last column is the
//! sum of all allocations. Under QEMU (emulated, release build) the whole
//! [`run_self_test`] took 27–34 ms before the sweep and 120–170 ms with it
//! (both types); it has not been measured on hardware, where emulation's
//! cost per instruction is gone.
//!
//! The data are generated here (a fixed xorshift sequence, no clock, no
//! randomness source, no dependency), so the verdict is deterministic. The
//! test never panics: all indices are checked, a mismatch returns `false`.
//!
//! Floating-point flags: the oracle and the kernel raise the usual cumulative
//! flags (inexact, underflow, invalid from `0 · ∞`) on the calling thread. The
//! self-test cannot save FPSR (no assembly outside `sme_kernels!`); a caller
//! that must leave the flags untouched saves and restores them around it.
//!
//! # The real kernels: [`run_self_test`]
//!
//! [`run_self_test`] binds the self-test to the real macro-kernels: it runs
//! `self_test_with(sme2_gemm_f32)`, then `self_test_with(sme2_gemm_f64)` when
//! its argument says that the CPU has FEAT_SME_F64F64, and returns `true` only
//! when every kernel it ran passed (the verdicts are combined by
//! [`self_test_kernels`], which tests also run on plain-Rust stand-ins). It
//! makes [`self_test_calls`] kernel calls per type it runs (330 for `f32`,
//! 948 with `f64`; 42 and 84 before the sweep). For the detection (`detect.rs`), which calls it in Wave 6:
//!
//! - It is called once, after the operating-system and RDSVL checks and
//!   before the result is cached, and turns the whole backend off (or only
//!   `f64`, if the detection keeps the types apart) on `false`.
//! - It never calls `detect::hw()`, `detect::enabled()`, `detect::
//!   thread_ready()`, `sme2_active()` or any GEMM entry point, so it cannot
//!   re-enter the detection's `OnceLock`. For the same reason the detection
//!   cannot call `detect::thread_ready()` (which calls `hw()`) before it: it
//!   checks the thread state that function checks (SVCR = 0, TPIDR2_EL0 = 0,
//!   a 64-byte streaming vector length) with `probe_thread_state()` directly.
//! - A thread that is not in that state is not evidence of a broken kernel:
//!   the detection then does not run the self-test and records no mismatch
//!   (it declines without a verdict, or retries later).
//! - The self-test leaves its cumulative floating-point flags on the calling
//!   thread (see above). The detection saves FPSR before the call and
//!   restores it afterwards (Wave 6), so that the first GEMM call does not
//!   leave flags that its NEON fallback would not have raised.

// Nothing in the library runs the self-test before the GEMM hook (Wave 6)
// wires it into the detection; only tests use it until then. `expect` turns
// into a warning once that changes, so the attribute cannot outlive its
// reason.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the SME2 drivers once the GEMM hook lands"
    )
)]

use super::args::{Epilogue, MR, MacroArgs, PREFETCH_OFF, SmeElement};
use super::kernel_f32::sme2_gemm_f32;
use super::kernel_f64::sme2_gemm_f64;
use super::oracle::{oracle_block, same_bits_or_nan};

/// A macro-kernel, or anything with its signature (the oracle, a stand-in).
pub(super) type KernelFn<T> = unsafe fn(&MacroArgs<T>);

/// How the self-test lays out A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ALayout {
    /// The plain packed layout: `a_depth = 64`, `a_panel = 64 · kc`, rows
    /// `m..` of the last panel zero.
    Packed,
    /// Column-major in place: `a_depth = lda = m + lda_pad` ([`Case::lda_pad`]),
    /// `a_panel = 64`; the spare rows of every column hold NaN (a kernel must
    /// not use them).
    InPlace,
}

/// Which values the self-test generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Data {
    /// Full-mantissa values of mixed sign and magnitude, some exact zeros.
    Plain,
    /// Subnormal inputs and products, signed zeros, a few infinities.
    Special,
}

/// One block of the self-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Case {
    /// Rows.
    pub(super) m: usize,
    /// Columns.
    pub(super) n: usize,
    /// Depth.
    pub(super) kc: usize,
    /// Layout of A.
    pub(super) layout: ALayout,
    /// [`ALayout::InPlace`]: spare rows per column of A (`lda = m + lda_pad`).
    /// Must be 0 for [`ALayout::Packed`] (the run fails otherwise).
    pub(super) lda_pad: usize,
    /// Elements between consecutive B panels beyond `kc · NR`
    /// (`b_panel = kc · NR + b_pad`); the gaps hold NaN.
    pub(super) b_pad: usize,
    /// Values of A, B and C.
    pub(super) data: Data,
}

/// The blocks of [`self_test_with`] (see the module docs).
pub(super) const CASES: [Case; 7] = [
    // 2 row panels (64 + 6) and 2 (f32: 16 + 3) or 3 (f64: 8 + 8 + 3) column
    // panels: partial tiles in both directions; B panels 5 elements apart
    // beyond `kc · NR` (NaN in the gaps).
    Case {
        m: 70,
        n: 19,
        kc: 13,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 5,
        data: Data::Plain,
    },
    // kc = 1; 3 row panels (64 + 64 + 2), 3 (f32) or 5 (f64) column panels.
    Case {
        m: 130,
        n: 33,
        kc: 1,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Plain,
    },
    // Whole tiles only, A in place.
    Case {
        m: 64,
        n: 16,
        kc: 2,
        layout: ALayout::InPlace,
        lda_pad: 2,
        b_pad: 0,
        data: Data::Plain,
    },
    // A single element.
    Case {
        m: 1,
        n: 1,
        kc: 3,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Plain,
    },
    // Fewer rows than one tile, deep and odd kc, A in place.
    Case {
        m: 9,
        n: 7,
        kc: 67,
        layout: ALayout::InPlace,
        lda_pad: 2,
        b_pad: 0,
        data: Data::Plain,
    },
    // Subnormals and special values, partial tiles in both directions.
    Case {
        m: 33,
        n: 17,
        kc: 5,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Special,
    },
    // A in place over 2 row panels (64 + 20: the last one has two active
    // 16-row tiles) with `lda = m + 3`; kc = 16 (a multiple of 4 and 8);
    // 3 (f32: 16 + 16 + 8) or 5 (f64) column panels with kc > 1; B panels
    // 3 elements apart beyond `kc · NR`.
    Case {
        m: 84,
        n: 40,
        kc: 16,
        layout: ALayout::InPlace,
        lda_pad: 3,
        b_pad: 3,
        data: Data::Plain,
    },
];

/// The alpha/beta pairs of [`self_test_with`]: M0, M1, M2 (`beta = -0`),
/// M3 with `alpha == 1`, M3 with `beta == 1`, M3 with neither.
pub(super) const SCALARS: [(f64, f64); 6] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (0.3, -0.0),
    (1.0, -0.7),
    (-1.3, 1.0),
    (0.3, -0.7),
];

/// The depths of the sweep: `kc = 1` (no pass of the double-step loop, the
/// remainder step only), an even depth (two passes, no remainder) and an odd
/// depth (two passes and the remainder). Two passes take the loop's back
/// branch and its pointer steps, which one pass would not.
pub(super) const SWEEP_DEPTHS: [usize; 3] = [1, 4, 5];

/// Blocks of the sweep per row variant: two row-panel positions (only panel,
/// last panel after a full one) × two A layouts × [`SWEEP_DEPTHS`].
pub(super) const SWEEP_BLOCKS_PER_VARIANT: usize = 2 * 2 * SWEEP_DEPTHS.len();

/// The systematic sweep of [`self_test_with`] for a kernel with ZA tiles of
/// `tile_rows` rows and B panels of `nr` columns (`T::TILE_ROWS`, `T::NR`),
/// so that both element types are covered by construction.
///
/// For every row variant `L` (`1..=64 / tile_rows` live tiles in the last
/// row panel), every position of that panel (the only panel: `m ≤ 64`; the
/// last one after a full 64-row panel: `m = 64 + rows`), both A layouts and
/// every depth of [`SWEEP_DEPTHS`] it yields one block of `n = nr + c`
/// columns (`1 ≤ c < nr`: a full and a partial column panel). The last live
/// tile holds exactly one row (`rows = (L - 1) · tile_rows + 1`) for packed
/// A as the only panel and for in-place A as the last panel, and is full
/// (`rows = L · tile_rows`) otherwise, so each position and each layout meets
/// both. With the six pairs of [`SCALARS`] every block runs in every
/// epilogue mode. `tests_selftest.rs` proves from this list that every path
/// of both kernels is taken.
///
/// Empty when `tile_rows` is 0; a block that cannot be formed (`nr < 2`) is
/// left out.
pub(super) fn sweep_cases(tile_rows: usize, nr: usize) -> impl Iterator<Item = Case> {
    let variants = MR.checked_div(tile_rows).unwrap_or(0);
    (0..variants.saturating_mul(SWEEP_BLOCKS_PER_VARIANT))
        .filter_map(move |index| sweep_case(tile_rows, nr, index))
}

/// Block `index` of [`sweep_cases`]: the depth varies fastest, then the
/// layout, then the position, then the row variant.
fn sweep_case(tile_rows: usize, nr: usize, index: usize) -> Option<Case> {
    let depth_index = index % SWEEP_DEPTHS.len();
    let kc = *SWEEP_DEPTHS.get(depth_index)?;
    let rest = index / SWEEP_DEPTHS.len();
    let in_place = rest % 2 == 1;
    let later = (rest / 2) % 2 == 1;
    let tiles = rest / 4 + 1;
    let rows = if in_place == later {
        (tiles - 1).checked_mul(tile_rows)?.checked_add(1)?
    } else {
        tiles.checked_mul(tile_rows)?
    };
    let m = if later { rows.checked_add(MR)? } else { rows };
    // A partial last column panel of 1..nr - 1 columns, varied over blocks.
    let widths = nr.checked_sub(1).filter(|&widths| widths > 0)?;
    let partial = (5 * tiles + depth_index) % widths + 1;
    let (layout, lda_pad) = if in_place {
        (ALayout::InPlace, 1 + tiles % 3)
    } else {
        (ALayout::Packed, 0)
    };
    Some(Case {
        m,
        n: nr.checked_add(partial)?,
        kc,
        layout,
        lda_pad,
        b_pad: depth_index,
        data: Data::Plain,
    })
}

/// Every block [`self_test_with`] runs for a kernel with ZA tiles of
/// `tile_rows` rows and B panels of `nr` columns, in order: [`CASES`], then
/// [`sweep_cases`].
pub(super) fn self_test_blocks(tile_rows: usize, nr: usize) -> impl Iterator<Item = Case> {
    CASES.into_iter().chain(sweep_cases(tile_rows, nr))
}

/// The kernel calls of one [`self_test_with`] run for `T` (when every run
/// passes): every block of [`self_test_blocks`] with every pair of
/// [`SCALARS`].
pub(super) fn self_test_calls<T: SmeElement>() -> usize {
    self_test_blocks(T::TILE_ROWS, T::NR).count() * SCALARS.len()
}

/// The bytes one run of `case` allocates for `T`: A, B and C with its guard
/// border, a copy of A and of B, and two copies of C (the kernel's and the
/// oracle's). `None` for a block that [`self_test_with`] refuses.
pub(super) fn run_bytes<T: SmeElement>(case: &Case) -> Option<usize> {
    let shape = Shape::of::<T>(case)?;
    let elements = shape
        .a_len
        .checked_add(shape.b_len)?
        .checked_mul(2)?
        .checked_add(shape.c_len.checked_mul(3)?)?;
    elements.checked_mul(size_of::<T>())
}

/// Runs `kernel` on every block of [`CASES`] with every pair of [`SCALARS`]
/// and returns `true` only when every run matches the oracle (see the module
/// docs for what is compared). Never panics.
///
/// # Safety
///
/// The caller guarantees that `kernel` may run on this thread now, for
/// arguments that satisfy the kernel contract of `args.rs`: for a real SME2
/// kernel, `detect::thread_ready()` returned `true` on this thread (and
/// FEAT_SME_F64F64 is present for an `f64` kernel). The self-test supplies
/// valid blocks itself.
pub(super) unsafe fn self_test_with<T: SmeElement>(kernel: KernelFn<T>) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { run_cases(kernel, self_test_blocks(T::TILE_ROWS, T::NR), &SCALARS) }
}

/// Runs the first-use self-test on the real SME2 macro-kernels:
/// [`sme2_gemm_f32`] always, and [`sme2_gemm_f64`] when `f64_present` says
/// that the CPU has FEAT_SME_F64F64. Returns `true` only when every kernel it
/// ran matched the oracle on every block (see the module docs); a mismatch
/// returns `false`. Never panics.
///
/// It runs [`self_test_calls`] kernel calls per type (330 for `f32`, 618
/// for `f64`), and it calls nothing that runs or reads the detection
/// (`detect::hw()`, `detect::enabled()`, `detect::thread_ready()`,
/// `sme2_active()`) and no GEMM entry point, so the detection may call it
/// while it initialises its `OnceLock`. The calling thread keeps the
/// floating-point flags that the runs raise.
///
/// # Safety
///
/// The safety contract is the one of the kernels. The caller guarantees that
///
/// - the CPU has SME2 with a 512-bit streaming vector length, and
///   FEAT_SME_F64F64 when `f64_present` is `true`;
/// - just before the call, the calling thread was in the state that
///   `detect::thread_ready()` checks: not in streaming mode and no ZA storage
///   on (SVCR = 0), no pending lazy ZA save (TPIDR2_EL0 = 0), a 64-byte
///   streaming vector length. Outside the detection that is
///   `detect::thread_ready()` returning `true`; inside it (where calling
///   `thread_ready()` would re-enter the `OnceLock`) the caller reads the
///   state with `detect::probe_thread_state()`.
///
/// Every kernel call leaves the thread in that state again (the frame of
/// `asm.rs` exits streaming mode with ZA off and touches neither TPIDR2_EL0
/// nor the vector length), so the state checked once before this call holds
/// for every kernel call it makes.
///
/// A thread that is not in that state must not call this function. Such a
/// thread says nothing about the kernels: the caller declines the backend
/// without a verdict, or retries later, and does not record a mismatch.
pub(super) unsafe fn run_self_test(f64_present: bool) -> bool {
    let f32_kernel: KernelFn<f32> = sme2_gemm_f32;
    let f64_kernel: Option<KernelFn<f64>> = if f64_present {
        Some(sme2_gemm_f64)
    } else {
        None
    };
    // SAFETY: forwarded from this function's contract: SME2 is present and
    // the thread is ready for the `f32` kernel, and `f64_kernel` is `Some`
    // only when the caller guarantees FEAT_SME_F64F64 as well.
    unsafe { self_test_kernels(f32_kernel, f64_kernel) }
}

/// The verdict of [`run_self_test`] for given kernels: [`self_test_with`] on
/// `f32_kernel`, then on `f64_kernel` when it is `Some`; `true` only when
/// every kernel that ran passed. Stops at the first failure. Never panics.
///
/// # Safety
///
/// As for [`self_test_with`], for `f32_kernel` and, when it is `Some`,
/// `f64_kernel`.
pub(super) unsafe fn self_test_kernels(
    f32_kernel: KernelFn<f32>,
    f64_kernel: Option<KernelFn<f64>>,
) -> bool {
    // SAFETY: forwarded from this function's contract.
    if !unsafe { self_test_with(f32_kernel) } {
        return false;
    }
    match f64_kernel {
        // SAFETY: forwarded from this function's contract.
        Some(kernel) => unsafe { self_test_with(kernel) },
        None => true,
    }
}

/// [`self_test_with`] over chosen blocks and scalar pairs (used by tests to
/// run kernels that support only some shapes or modes). `true` only when
/// every run matches; an empty list is vacuously `true`.
///
/// # Safety
///
/// As for [`self_test_with`], for every block of `cases` and every pair of
/// `scalars`.
pub(super) unsafe fn self_test_cases<T: SmeElement>(
    kernel: KernelFn<T>,
    cases: &[Case],
    scalars: &[(f64, f64)],
) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { run_cases(kernel, cases.iter().copied(), scalars) }
}

/// Runs `kernel` on every block of `cases` with every pair of `scalars`, in
/// that order (block `b`, pair `s` gets the data seed `b · 256 + s`); `true`
/// only when every run matches. Stops at the first mismatch.
///
/// # Safety
///
/// As for [`self_test_with`], for every block of `cases` and every pair of
/// `scalars`.
unsafe fn run_cases<T: SmeElement>(
    kernel: KernelFn<T>,
    cases: impl Iterator<Item = Case>,
    scalars: &[(f64, f64)],
) -> bool {
    for (case_index, case) in cases.enumerate() {
        for (pair_index, &(alpha, beta)) in scalars.iter().enumerate() {
            let seed = ((case_index as u64) << 8) | pair_index as u64;
            // SAFETY: forwarded from this function's contract.
            if !unsafe { run_case(kernel, &case, to_element(alpha), to_element(beta), seed) } {
                return false;
            }
        }
    }
    true
}

/// `v` as a `T` (`f32` rounds to nearest). `SmeElement` is implemented by
/// `f32` and `f64` only, told apart by their size.
fn to_element<T: SmeElement>(v: f64) -> T {
    if size_of::<T>() == size_of::<f32>() {
        T::from_bits64(u64::from((v as f32).to_bits()))
    } else {
        T::from_bits64(v.to_bits())
    }
}

/// A NaN with the given payload bits for `T` (quiet; never the default NaN).
fn nan_with_payload<T: SmeElement>(payload: u32) -> T {
    if size_of::<T>() == size_of::<f32>() {
        T::from_bits64(u64::from(0x7fc0_0000 | (payload & 0x003f_ffff)))
    } else {
        T::from_bits64(0x7ff8_0000_0000_0000 | u64::from(payload))
    }
}

/// The guard sentinel around C: a NaN with a payload no arithmetic produces.
fn sentinel<T: SmeElement>() -> T {
    nan_with_payload(0x5e_1f7e)
}

/// The fill of C in modes that must not read it: another unusual NaN.
fn c_fill<T: SmeElement>() -> T {
    nan_with_payload(0x0f_111c)
}

/// A fixed xorshift sequence of test values.
struct Values {
    /// The generator state (never zero).
    state: u64,
    /// Which values to make.
    data: Data,
    /// `true` for `f32`.
    single: bool,
}

impl Values {
    /// A sequence for `seed` (operand `which` of one run).
    fn new(seed: u64, which: u64, data: Data, single: bool) -> Self {
        let state = (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ which.wrapping_mul(0xd1b5_4a32_d192_ed03))
            | 1;
        Self {
            state,
            data,
            single,
        }
    }

    /// The next raw 64 bits.
    fn bits(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// The next value as `f64` (rounded to `f32` by [`Values::next`] when
    /// `single`).
    fn next_f64(&mut self) -> f64 {
        let bits = self.bits();
        // A full-mantissa magnitude in [1, 2), with the sign from bit 0.
        let unit = f64::from_bits(0x3ff0_0000_0000_0000 | (bits >> 12));
        let signed = if bits & 1 == 0 { unit } else { -unit };
        let pick = (bits >> 1) & 63;
        match self.data {
            Data::Plain => match pick % 8 {
                0 => 0.0,
                1 | 2 => signed * 0.001_9,
                3 => signed * 37.0,
                _ => signed * 0.6,
            },
            Data::Special => {
                // Half the depth exponent of the smallest normal: products of
                // two such values are subnormal or underflow.
                let (half, sub) = if self.single {
                    (-64, -133)
                } else {
                    (-512, -1_035)
                };
                match pick {
                    0 => f64::INFINITY,
                    1 => f64::NEG_INFINITY,
                    2..=5 => -0.0,
                    6..=9 => 0.0,
                    10..=19 => signed * pow2(sub),
                    20..=49 => signed * pow2(half),
                    _ => signed * 0.75,
                }
            }
        }
    }

    /// The next value as a `T`.
    fn next<T: SmeElement>(&mut self) -> T {
        to_element(self.next_f64())
    }
}

/// `2^e` for `-1074 ≤ e ≤ 1023`, built from its bits (exact, subnormal
/// powers included; `powi` would overflow an intermediate `2^-e`).
fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1_u64 << (e + 1074).clamp(0, 51))
    }
}

/// Whether two slices hold the same bits, element by element.
fn same_bits<T: SmeElement>(got: &[T], want: &[T]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g.bits() == w.bits())
}

/// The sizes and strides of the operands of one run.
struct Shape {
    /// `a_depth` of the case.
    a_depth: usize,
    /// `a_panel` of the case.
    a_panel: usize,
    /// Elements of A.
    a_len: usize,
    /// `b_panel` of the case (`kc · NR + b_pad`).
    b_panel: usize,
    /// Elements of B (`ceil(n / NR)` panels).
    b_len: usize,
    /// Leading dimension of C (`m + 3`).
    ldc: usize,
    /// Elements of C with its guard border (`ldc · (n + 2)`).
    c_len: usize,
}

impl Shape {
    /// The shape of `case` for `T`; `None` for a packed case with an lda pad
    /// or a size that overflows.
    fn of<T: SmeElement>(case: &Case) -> Option<Self> {
        let Case {
            m,
            n,
            kc,
            layout,
            lda_pad,
            b_pad,
            ..
        } = *case;
        let (a_depth, a_panel, a_len) = match layout {
            ALayout::Packed => {
                if lda_pad != 0 {
                    return None;
                }
                let panel = MR.checked_mul(kc)?;
                (MR, panel, m.div_ceil(MR).checked_mul(panel)?)
            }
            // The allocation is sized as if the row panels were `lda · kc`
            // apart, plus 64: a stray read of rows `m..64` of the last column,
            // or of a kernel that steps row panels as in the packed layout
            // (by `a_depth · kc` or `64 · kc ≤ lda · kc` for `m ≥ 64`), stays
            // inside it and reads NaN (the contract forbids both, but the
            // self-test must not crash on a kernel that breaks it).
            ALayout::InPlace => {
                let lda = m.checked_add(lda_pad)?;
                let panels = m.div_ceil(MR).checked_mul(lda)?.checked_mul(kc)?;
                (lda, MR, panels.checked_add(MR)?)
            }
        };
        let b_panel = T::NR.checked_mul(kc)?.checked_add(b_pad)?;
        let b_len = n.div_ceil(T::NR).checked_mul(b_panel)?;
        let ldc = m.checked_add(3)?;
        let c_len = ldc.checked_mul(n.checked_add(2)?)?;
        Some(Self {
            a_depth,
            a_panel,
            a_len,
            b_panel,
            b_len,
            ldc,
            c_len,
        })
    }
}

/// The operands of one run, laid out as `args.rs` describes.
struct Operands<T> {
    /// A in the layout of the case.
    a: Vec<T>,
    /// `a_depth` of the case.
    a_depth: usize,
    /// `a_panel` of the case.
    a_panel: usize,
    /// B packed in `NR`-column panels, zero-padded, `b_panel` apart.
    b: Vec<T>,
    /// `b_panel` of the case (`kc · NR + b_pad`).
    b_panel: usize,
    /// C with its guard border.
    c: Vec<T>,
    /// Leading dimension of C (`m + 3`).
    ldc: usize,
    /// Index of `C(0, 0)` in `c` (`ldc + 1`).
    origin: usize,
}

impl<T: SmeElement> Operands<T> {
    /// Generates the operands of `case` for the mode `mode`.
    fn new(case: &Case, mode: Epilogue, seed: u64) -> Option<Self> {
        let Case {
            m,
            n,
            kc,
            layout,
            data,
            ..
        } = *case;
        let single = size_of::<T>() == size_of::<f32>();
        let nr = T::NR;
        let Shape {
            a_depth,
            a_panel,
            a_len,
            b_panel,
            b_len,
            ldc,
            c_len,
        } = Shape::of::<T>(case)?;

        let mut a = match layout {
            ALayout::Packed => vec![T::ZERO; a_len],
            ALayout::InPlace => vec![nan_with_payload(0x0a_0a0a); a_len],
        };
        let mut values = Values::new(seed, 1, data, single);
        for p in 0..kc {
            for i in 0..m {
                let index = (i / MR) * a_panel + p * a_depth + i % MR;
                *a.get_mut(index)? = values.next();
            }
        }

        // Each panel: `kc · NR` elements (columns `n..` of the last one zero),
        // then a gap of `b_pad` NaNs.
        let b_used = nr.checked_mul(kc)?;
        let mut b = vec![T::ZERO; b_len];
        for (index, value) in b.iter_mut().enumerate() {
            if index % b_panel >= b_used {
                *value = nan_with_payload(0x0b_0b0b);
            }
        }
        let mut values = Values::new(seed, 2, data, single);
        for p in 0..kc {
            for j in 0..n {
                let index = (j / nr) * b_panel + p * nr + j % nr;
                *b.get_mut(index)? = values.next();
            }
        }

        let origin = ldc.checked_add(1)?;
        let mut c = vec![sentinel(); c_len];
        let mut values = Values::new(seed, 3, data, single);
        for j in 0..n {
            for i in 0..m {
                let value = if mode.reads_c() {
                    values.next()
                } else {
                    c_fill()
                };
                *c.get_mut(origin + i + j * ldc)? = value;
            }
        }
        Some(Self {
            a,
            a_depth,
            a_panel,
            b,
            b_panel,
            c,
            ldc,
            origin,
        })
    }

    /// The argument block of this run, with C at `c` (a copy of `self.c`).
    ///
    /// The pointers into A and B come from `as_mut_ptr`, so a kernel that
    /// writes them (a contract violation) changes defined memory, and the
    /// self-test sees it.
    fn args(&mut self, case: &Case, c: &mut [T], alpha: T, beta: T) -> Option<MacroArgs<T>> {
        let origin = c.get_mut(self.origin..)?.as_mut_ptr();
        Some(MacroArgs {
            a: self.a.as_mut_ptr().cast_const(),
            a_depth: self.a_depth,
            a_panel: self.a_panel,
            b: self.b.as_mut_ptr().cast_const(),
            b_panel: self.b_panel,
            c: origin,
            ldc: self.ldc,
            m: case.m,
            n: case.n,
            kc: case.kc,
            alpha_bits: alpha.bits(),
            beta_bits: beta.bits(),
            mode: Epilogue::select(alpha, beta) as u64,
            a_pack: core::ptr::null_mut(),
            prefetch: PREFETCH_OFF,
        })
    }
}

/// One run: `kernel` and the oracle on the same block, then the comparisons
/// of the module docs. `false` on any mismatch or on an impossible size.
///
/// # Safety
///
/// As for [`self_test_with`].
unsafe fn run_case<T: SmeElement>(
    kernel: KernelFn<T>,
    case: &Case,
    alpha: T,
    beta: T,
    seed: u64,
) -> bool {
    if case.m == 0 || case.n == 0 || case.kc == 0 {
        return false;
    }
    let mode = Epilogue::select(alpha, beta);
    let Some(mut operands) = Operands::<T>::new(case, mode, seed) else {
        return false;
    };
    let (a_before, b_before) = (operands.a.clone(), operands.b.clone());
    let mut got = operands.c.clone();
    let mut want = operands.c.clone();
    let Some(got_args) = operands.args(case, &mut got, alpha, beta) else {
        return false;
    };
    // SAFETY: the block satisfies the kernel contract: A holds `A(i, p)` for
    // `i < m`, `p < kc` at the layout's offsets (in place: `i + p · lda` with
    // a spare tail), B holds `ceil(n / NR)` panels `b_panel = kc · NR + b_pad`
    // apart, each with `kc` zero-padded rows of `NR` elements, C has `ldc = m + 3 ≥ m` and `n + 2` columns from `origin = ldc + 1`, so
    // `C(i, j)` for `i < m`, `j < n` is inside `got`, which overlaps neither A
    // nor B; `m, n, kc ≥ 1`; the mode is `Epilogue::select(alpha, beta)`; the
    // Wave-4 fields are off. That the kernel may run on this thread is this
    // function's contract.
    unsafe { kernel(&got_args) };
    let Some(want_args) = operands.args(case, &mut want, alpha, beta) else {
        return false;
    };
    // SAFETY: the same block with C in `want`; the oracle needs no hardware.
    unsafe { oracle_block(&want_args) };

    if !same_bits(&operands.a, &a_before) || !same_bits(&operands.b, &b_before) {
        return false;
    }
    compare_c(&got, &want, operands.origin, operands.ldc, case.m, case.n)
}

/// The interior of C (`m × n` from `origin`, leading dimension `ldc`) must
/// match with [`same_bits_or_nan`]; every other element must hold the
/// sentinel's exact bits.
fn compare_c<T: SmeElement>(
    got: &[T],
    want: &[T],
    origin: usize,
    ldc: usize,
    m: usize,
    n: usize,
) -> bool {
    if got.len() != want.len() || ldc == 0 || ldc < m {
        return false;
    }
    let guard = sentinel::<T>().bits();
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        let inside = index
            .checked_sub(origin)
            .is_some_and(|offset| offset % ldc < m && offset / ldc < n);
        if !inside && (g.bits() != guard || w.bits() != guard) {
            return false;
        }
    }
    (0..n).all(|j| {
        let start = origin + j * ldc;
        match (got.get(start..start + m), want.get(start..start + m)) {
            (Some(g), Some(w)) => same_bits_or_nan(g, w),
            _ => false,
        }
    })
}
