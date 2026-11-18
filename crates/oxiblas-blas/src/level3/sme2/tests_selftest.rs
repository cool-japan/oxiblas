//! Tests of the first-use self-test (`selftest.rs`).
//!
//! - Its verdicts on the oracle and on wrong plain-Rust stand-in kernels, the
//!   claims its fixed blocks ([`CASES`]) make, and the verdict logic of the
//!   binder (`self_test_kernels`) with exact call counts. These run on every
//!   host.
//! - The guard border: its sentinel is a signalling NaN, and stand-ins that
//!   read, compute on and write back one element beyond the block in M1 or
//!   M3 (as a kernel does that addresses one row or column too many) are
//!   rejected.
//! - The mismatch report: where the self-test says it failed, for stand-ins
//!   whose defect is known.
//! - The real macro-kernels through `run_self_test` and `self_test_with`
//!   (their failure messages print the mismatch); these skip without SME2.
//!
//! The path table (every path class of both kernels is taken, and the model
//! matches the kernel text) is in `tests_selftest_paths.rs`, the edited
//! kernel texts that the model must reject in `tests_selftest_edits.rs`, and
//! the online cases (online packing, prefetch: the buffer's guards and
//! report, its containment limit) in `tests_selftest_online.rs`; the shape
//! of the sweep is at the end of this file.

use core::cell::Cell;
use core::hint::black_box;
use std::collections::BTreeSet;
use std::time::Instant;

use super::args::{Epilogue, MR, MacroArgs, SmeElement};
use super::kernel_f32::sme2_gemm_f32;
use super::kernel_f64::sme2_gemm_f64;
use super::oracle::oracle_block;
use super::selftest::{
    ALayout, CASES, CLayout, Case, Data, Difference, KernelFn, Mismatch, Online, SCALARS,
    SWEEP_BLOCKS_PER_VARIANT, SWEEP_DEPTHS, SWEEP_MIDDLE_BLOCKS, a_spare, b_gap, c_layout,
    run_buffers, run_bytes, run_self_test, run_self_test_detailed, self_test_blocks,
    self_test_calls, self_test_cases, self_test_cases_detailed, self_test_kernels, self_test_with,
    self_test_with_detailed, sentinel, sweep_cases,
};
use super::test_support::{
    Need, fpsr_read, fpsr_write, hardware_or_skip, kernel_entries_here, kernel_ready, sme_state,
};
use super::tests_oracle::TestFloat;

#[test]
fn test_sme2_oracle_special_data_reach_subnormals() {
    // The special-value case of the self-test does produce subnormal results
    // (so a kernel that flushes them is caught, see the stand-ins below).
    fn count<T: TestFloat>() -> usize {
        let case = CASES
            .iter()
            .find(|case| case.data == Data::Special)
            .copied();
        let Some(case) = case else {
            panic!("no special case");
        };
        SUBNORMALS.with(|cell| cell.set(0));
        let probe: KernelFn<T> = record_subnormals::<T>;
        // SAFETY: the probe is plain Rust and runs on any thread.
        let _ = unsafe { self_test_cases(probe, &[case], &SCALARS) };
        SUBNORMALS.with(|cell| cell.replace(0))
    }
    assert!(count::<f32>() > 0, "f32");
    assert!(count::<f64>() > 0, "f64");
}

std::thread_local! {
    /// Subnormal results seen by [`record_subnormals`] on this thread.
    static SUBNORMALS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Stand-in: the oracle, counting subnormal results.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn record_subnormals<T: TestFloat>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    for (i, j) in block_indices(args) {
        // SAFETY: `C(i, j)` of the block, valid by the contract.
        let value = unsafe { *args.c.add(i + j * args.ldc) };
        if value.subnormal() {
            SUBNORMALS.with(|cell| cell.set(cell.get() + 1));
        }
    }
}

// ---------------------------------------------------------------------------
// selftest.rs
// ---------------------------------------------------------------------------

/// All `(i, j)` of a block.
fn block_indices<T>(args: &MacroArgs<T>) -> impl Iterator<Item = (usize, usize)> {
    let (m, n) = (args.m, args.n);
    (0..n).flat_map(move |j| (0..m).map(move |i| (i, j)))
}

/// Adds one unit in the last place to a value (NaN stays NaN).
fn bump<T: SmeElement>(value: T) -> T {
    T::from_bits64(value.bits() + 1)
}

/// Stand-in: right everywhere except `C(0, 0)`, one ulp off.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_one_ulp<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    // SAFETY: `C(0, 0)` exists (`m, n ≥ 1`).
    unsafe { *args.c = bump(*args.c) };
}

/// The default NaN of the type (what the ZA instructions return).
fn default_nan<T: SmeElement>() -> T {
    T::from_bits64(if T::NR == 16 {
        0x7fc0_0000
    } else {
        0x7ff8_0000_0000_0000
    })
}

/// Stand-in: right inside the block, but writes the default NaN to `C(m, j)`
/// for every `j < n`, the row just below the block (a NaN over the NaN
/// sentinel: only an exact-bits border check sees it).
///
/// # Safety
///
/// As for `oracle_block`, and row `m` of every column must be writable (the
/// self-test's guard border).
unsafe fn wrong_writes_row_below<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    for j in 0..args.n {
        // SAFETY: the self-test keeps `ldc ≥ m + 3`, so `C(m, j)` is inside
        // its C buffer, in the guard rows of column `j`.
        unsafe { *args.c.add(args.m + j * args.ldc) = default_nan() };
    }
}

/// Stand-in: right inside the block, but writes the default NaN to the
/// element just before `C(0, 0)` (outside the block).
///
/// # Safety
///
/// As for `oracle_block`, and that element must be writable (the self-test's
/// guard column before the block).
unsafe fn wrong_writes_before_block<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    // SAFETY: the self-test keeps a guard column before the block, so the
    // element is inside its C buffer.
    unsafe { *args.c.sub(1) = default_nan() };
}

/// Stand-in: one ulp off in the last element `C(m - 1, n - 1)` only.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_last_element<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    // SAFETY: `C(m - 1, n - 1)` of the block (`m, n ≥ 1`).
    unsafe {
        let c = args.c.add(args.m - 1 + (args.n - 1) * args.ldc);
        *c = bump(*c);
    }
}

/// Stand-in: right in C, but writes `A(m, 0)` (the zero padding of the packed
/// layout, a spare row in place) when `m` is not a multiple of 64.
///
/// # Safety
///
/// As for `oracle_block`; the element must be writable (the self-test's A
/// comes from `as_mut_ptr` and holds it).
unsafe fn wrong_writes_a_padding<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    if !args.m.is_multiple_of(MR) {
        let offset = (args.m / MR) * args.a_panel + args.m % MR;
        // SAFETY: row `m` of the last panel exists in the self-test's
        // buffers (padding or spare row) and is writable.
        unsafe { *args.a.cast_mut().add(offset) = T::ONE };
    }
}

/// Stand-in: right in C, but writes `B(0, n)` (zero padding) when `n` is not
/// a multiple of `NR`.
///
/// # Safety
///
/// As for `oracle_block`; the element must be writable (the self-test's B
/// comes from `as_mut_ptr` and holds whole panels).
unsafe fn wrong_writes_b_padding<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    if !args.n.is_multiple_of(T::NR) {
        let offset = (args.n / T::NR) * args.b_panel + args.n % T::NR;
        // SAFETY: column `n` of the last panel is zero padding inside the
        // self-test's B, and writable.
        unsafe { *args.b.cast_mut().add(offset) = T::ONE };
    }
}

/// Stand-in: reads C in every mode and lets a NaN in C through (as a kernel
/// that adds a zero-scaled C would, with the sign of zero kept): exact for
/// any C that is not a NaN, so only the NaN fill of M0/M2 catches it.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_reads_c<T: SmeElement>(args: &MacroArgs<T>) {
    let old: Vec<T> = block_indices(args)
        // SAFETY: `C(i, j)` of the block.
        .map(|(i, j)| unsafe { *args.c.add(i + j * args.ldc) })
        .collect();
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    if Epilogue::select(
        T::from_bits64(args.alpha_bits),
        T::from_bits64(args.beta_bits),
    )
    .reads_c()
    {
        return;
    }
    for ((i, j), old) in block_indices(args).zip(old) {
        // SAFETY: `C(i, j)` of the block.
        unsafe {
            let c = args.c.add(i + j * args.ldc);
            if old.nan() {
                *c = old;
            }
        }
    }
}

/// Stand-in: computes M2 as M0 (forgets alpha).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_mode_scale_as_store<T: SmeElement>(args: &MacroArgs<T>) {
    let mode = if args.mode == Epilogue::Scale as u64 {
        Epilogue::Store as u64
    } else {
        args.mode
    };
    // SAFETY: forwarded; M0 reads less than M2.
    unsafe { oracle_block(&MacroArgs { mode, ..*args }) };
}

/// Stand-in: computes M3 as M1 (`acc + C`).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_mode_combine_as_add<T: SmeElement>(args: &MacroArgs<T>) {
    let mode = if args.mode == Epilogue::ScaleCombine as u64 {
        Epilogue::Add as u64
    } else {
        args.mode
    };
    // SAFETY: forwarded; M1 reads what M3 reads.
    unsafe { oracle_block(&MacroArgs { mode, ..*args }) };
}

/// Stand-in: one ulp off only in the last tile, and only when that tile is
/// partial in both directions.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_last_partial_tile<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    if args.m.is_multiple_of(MR) || args.n.is_multiple_of(T::NR) {
        return;
    }
    let (row0, col0) = (args.m / MR * MR, args.n / T::NR * T::NR);
    for (i, j) in block_indices(args) {
        if i >= row0 && j >= col0 {
            // SAFETY: `C(i, j)` of the block.
            unsafe {
                let c = args.c.add(i + j * args.ldc);
                *c = bump(*c);
            }
        }
    }
}

/// Stand-in: writes nothing.
///
/// # Safety
///
/// None needed; it touches nothing.
unsafe fn wrong_writes_nothing<T: SmeElement>(_args: &MacroArgs<T>) {}

/// Stand-in: rounds every product before adding it (no fused multiply-add).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_unfused<T: SmeElement>(args: &MacroArgs<T>) {
    let mode = Epilogue::select(
        T::from_bits64(args.alpha_bits),
        T::from_bits64(args.beta_bits),
    );
    let alpha = T::from_bits64(args.alpha_bits);
    let beta = T::from_bits64(args.beta_bits);
    for (i, j) in block_indices(args) {
        let mut acc = T::ZERO;
        for p in 0..args.kc {
            // SAFETY: `A(i, p)` and `B(p, j)` of the block.
            let (a, b) = unsafe {
                (
                    *args
                        .a
                        .add((i / MR) * args.a_panel + p * args.a_depth + i % MR),
                    *args
                        .b
                        .add((j / T::NR) * args.b_panel + p * T::NR + j % T::NR),
                )
            };
            acc = a * b + acc;
        }
        // SAFETY: `C(i, j)` of the block.
        let c = unsafe { args.c.add(i + j * args.ldc) };
        let old = if mode.reads_c() {
            // SAFETY: `C(i, j)` of the block, read only in the modes that
            // read C.
            unsafe { *c }
        } else {
            T::ZERO
        };
        let value = match mode {
            Epilogue::Store => acc,
            Epilogue::Add => acc + old,
            Epilogue::Scale => acc * alpha,
            Epilogue::ScaleCombine => {
                let t = if alpha == T::ONE { acc } else { acc * alpha };
                if beta == T::ONE {
                    t + old
                } else {
                    old.fused(beta, t)
                }
            }
        };
        // SAFETY: `C(i, j)` of the block.
        unsafe { *c = value };
    }
}

/// Stand-in: reads A with depth stride 64 whenever the given one is larger
/// (a kernel that hard-codes the packed `a_depth`).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_ignores_a_strides<T: SmeElement>(args: &MacroArgs<T>) {
    let narrowed = MacroArgs {
        a_depth: args.a_depth.min(MR),
        ..*args
    };
    // SAFETY: a smaller depth stride only lowers every A offset, so each read
    // stays inside the region the caller made valid for `args`.
    unsafe { oracle_block(&narrowed) };
}

/// Stand-in: flushes subnormal results to (signed) zero.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_flushes_subnormals<T: TestFloat>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    for (i, j) in block_indices(args) {
        // SAFETY: `C(i, j)` of the block.
        unsafe {
            let c = args.c.add(i + j * args.ldc);
            if (*c).subnormal() {
                *c = (*c) * T::ZERO;
            }
        }
    }
}

/// Stand-in: ignores `a_panel` and steps the 64-row panels of A by
/// `a_depth · kc` (right for the packed layout, wrong in place with more than
/// 64 rows).
///
/// # Safety
///
/// As for `oracle_block`, and A must hold `ceil(m / 64)` panels of
/// `a_depth · kc` elements plus 64 (the self-test sizes its A so).
unsafe fn wrong_ignores_a_panel<T: SmeElement>(args: &MacroArgs<T>) {
    let packed_step = MacroArgs {
        a_panel: args.a_depth * args.kc,
        ..*args
    };
    // SAFETY: every A offset is below `ceil(m / 64) · a_depth · kc + 64`,
    // inside the self-test's A (see `Operands::new`).
    unsafe { oracle_block(&packed_step) };
}

/// Stand-in: drops the last depth step when `kc` is a multiple of 4 (the
/// remainder-loop bug of a depth loop unrolled by 4).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_depth_multiple_of_4<T: SmeElement>(args: &MacroArgs<T>) {
    let kc = if args.kc.is_multiple_of(4) {
        args.kc - 1
    } else {
        args.kc
    };
    // SAFETY: fewer depth steps with the same strides read a subset of what
    // the caller made valid.
    unsafe { oracle_block(&MacroArgs { kc, ..*args }) };
}

/// Stand-in: one ulp off at `C(0, 0)` when the last 64-row panel holds 17 to
/// 32 rows (an `f32` kernel whose two-active-tile variant is broken).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_two_tile_tail<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    if (17..=32).contains(&(args.m % MR)) {
        // SAFETY: `C(0, 0)` exists (`m, n ≥ 1`).
        unsafe { *args.c = bump(*args.c) };
    }
}

/// Stand-in: with three or more column panels and `kc > 1`, the third
/// panel's B base forgets `kc` (`b + 2 · NR` instead of `b + 2 · b_panel`).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_third_panel_ignores_kc<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    let nr = T::NR;
    if args.n > 2 * nr && args.kc > 1 {
        let third = MacroArgs {
            b: args.b.wrapping_add(2 * nr),
            c: args.c.wrapping_add(2 * nr * args.ldc),
            n: (args.n - 2 * nr).min(nr),
            ..*args
        };
        // SAFETY: B reads stay below `2 · NR + kc · NR ≤ 3 · b_panel` (at
        // least three panels exist, `b_panel ≥ kc · NR`, `kc ≥ 1`); the C
        // block is columns `2 · NR..` of the caller's block.
        unsafe { oracle_block(&third) };
    }
}

/// Stand-in: ignores `b_panel` and steps the B panels by `kc · NR` (right
/// for the plain packed layout, wrong for a padded panel stride).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn wrong_ignores_b_panel<T: SmeElement>(args: &MacroArgs<T>) {
    let tight = MacroArgs {
        b_panel: args.kc * T::NR,
        ..*args
    };
    // SAFETY: `b_panel ≥ kc · NR`, so a smaller panel stride only lowers
    // every B offset.
    unsafe { oracle_block(&tight) };
}

/// The read-modify-write that the M1 and M3 epilogues of the kernels make
/// on one element, with a dead (zero) accumulator lane: `acc + C` in M1 and
/// `fma(C, beta, acc · alpha)` in M3 (FMUL, then FMLA), as real
/// floating-point operations at run time (`black_box` keeps the compiler
/// from folding the zero away). `None` in M0 and M2, which do not read C.
fn dead_lane_rmw<T: SmeElement>(args: &MacroArgs<T>, c: T) -> Option<T> {
    let acc = black_box(T::ZERO);
    let alpha = T::from_bits64(args.alpha_bits);
    let beta = T::from_bits64(args.beta_bits);
    match args.mode {
        1 => Some(acc + black_box(c)),
        3 => Some(black_box(c).fused(beta, acc * alpha)),
        _ => None,
    }
}

/// Stand-in: the oracle, and in M1 and M3 calls also the epilogue's
/// read-modify-write with a zero accumulator on the row just below the
/// block, `C(m, j)` for `j < n` (a kernel whose row predicate admits one row
/// too many). Right in M0 and M2.
///
/// # Safety
///
/// As for `oracle_block`, and row `m` of every column must be readable and
/// writable (the self-test's guard border).
unsafe fn wrong_rmw_row_below<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    for j in 0..args.n {
        // SAFETY: the self-test keeps `ldc ≥ m + 3`, so `C(m, j)` is inside
        // its C buffer, in the guard rows of column `j`.
        unsafe {
            let c = args.c.add(args.m + j * args.ldc);
            if let Some(value) = dead_lane_rmw(args, *c) {
                *c = value;
            }
        }
    }
}

/// Stand-in: the oracle, and in M1 and M3 calls also the epilogue's
/// read-modify-write with a zero accumulator on the column just beyond the
/// block, `C(i, n)` for `i < m` (a kernel whose column loop runs once too
/// often). Right in M0 and M2.
///
/// # Safety
///
/// As for `oracle_block`, and column `n` must be readable and writable for
/// rows `0..m` (the self-test's guard column after the block).
unsafe fn wrong_rmw_column_beyond<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    for i in 0..args.m {
        // SAFETY: the self-test's C reaches at least the end of column `n`
        // (`ldc - 1 ≥ m` guard elements of it, see `c_layout`), so `C(i, n)`
        // is inside its buffer.
        unsafe {
            let c = args.c.add(i + args.n * args.ldc);
            if let Some(value) = dead_lane_rmw(args, *c) {
                *c = value;
            }
        }
    }
}

/// What a kernel's dead lane stores to an element it must not touch: zero in
/// M0 and M2 (the ZA lane of a masked A row or a zero-padded B column holds
/// `+0`, and `0 · alpha` is zero), the epilogue's read-modify-write with a
/// zero accumulator in M1 and M3 ([`dead_lane_rmw`]).
fn dead_lane_store<T: SmeElement>(args: &MacroArgs<T>, c: T) -> T {
    dead_lane_rmw(args, c).unwrap_or(T::ZERO)
}

/// Which elements outside the block a whole-tile stand-in stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// All `NR` columns of a partial last column panel (rows `< m`).
    Columns,
    /// All 64 rows of a partial last row panel (columns `< n`).
    Rows,
    /// Both: every whole `64 × NR` tile of the block.
    Tiles,
    /// Only the farthest element a whole tile reaches,
    /// `C(64 · ceil(m / 64) - 1, NR · ceil(n / NR) - 1)`, when it lies
    /// outside the block.
    Farthest,
}

/// The oracle, and then the dead-lane stores of a kernel that stores whole
/// tiles in the direction `reach` ([`dead_lane_store`] on every element of
/// that reach outside the block).
///
/// # Safety
///
/// As for `oracle_block`, and every element of a whole `64 × NR` tile at
/// every tile position of the block must be readable and writable: the
/// self-test's C buffer covers them (`c_layout`, checked for every block by
/// `test_sme2_selftest_guard_covers_whole_tiles`).
unsafe fn store_whole_tiles<T: SmeElement>(args: &MacroArgs<T>, reach: Reach) {
    // SAFETY: forwarded.
    unsafe { oracle_block(args) };
    let (m, n) = (args.m, args.n);
    let rows_end = m.div_ceil(MR) * MR;
    let columns_end = n.div_ceil(T::NR) * T::NR;
    let (rows, columns) = match reach {
        Reach::Columns => (m, columns_end),
        Reach::Rows => (rows_end, n),
        Reach::Tiles | Reach::Farthest => (rows_end, columns_end),
    };
    for j in 0..columns {
        for i in 0..rows {
            let farthest = i + 1 == rows_end && j + 1 == columns_end;
            let outside = i >= m || j >= n;
            if outside && (reach != Reach::Farthest || farthest) {
                // SAFETY: `i < 64 · ceil(m / 64)`, `j < NR · ceil(n / NR)`:
                // an element of a whole tile, inside the self-test's C
                // (this function's contract).
                unsafe {
                    let c = args.c.add(i + j * args.ldc);
                    *c = dead_lane_store(args, *c);
                }
            }
        }
    }
}

/// Stand-in: stores all `NR` columns of a partial last column panel.
///
/// # Safety
///
/// As for [`store_whole_tiles`].
unsafe fn wrong_stores_whole_columns<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { store_whole_tiles(args, Reach::Columns) };
}

/// Stand-in: stores all 64 rows of a partial last row panel.
///
/// # Safety
///
/// As for [`store_whole_tiles`].
unsafe fn wrong_stores_whole_rows<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { store_whole_tiles(args, Reach::Rows) };
}

/// Stand-in: stores every whole `64 × NR` tile (rows and columns).
///
/// # Safety
///
/// As for [`store_whole_tiles`].
unsafe fn wrong_stores_whole_tiles<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { store_whole_tiles(args, Reach::Tiles) };
}

/// Stand-in: stores only the farthest element of the last whole tile.
///
/// # Safety
///
/// As for [`store_whole_tiles`].
unsafe fn wrong_stores_farthest<T: SmeElement>(args: &MacroArgs<T>) {
    // SAFETY: forwarded.
    unsafe { store_whole_tiles(args, Reach::Farthest) };
}

/// Every wrong stand-in for one type, with its name.
fn wrong_kernels<T: TestFloat>() -> [(&'static str, KernelFn<T>); 25] {
    [
        ("one ulp off", wrong_one_ulp::<T>),
        (
            "writes the row below the block",
            wrong_writes_row_below::<T>,
        ),
        ("writes before the block", wrong_writes_before_block::<T>),
        ("ignores a_panel", wrong_ignores_a_panel::<T>),
        ("kc multiple of 4", wrong_depth_multiple_of_4::<T>),
        ("17..32-row tail", wrong_two_tile_tail::<T>),
        ("third panel forgets kc", wrong_third_panel_ignores_kc::<T>),
        ("ignores b_panel", wrong_ignores_b_panel::<T>),
        ("C read in M0/M2", wrong_reads_c::<T>),
        ("M2 as M0", wrong_mode_scale_as_store::<T>),
        ("M3 as M1", wrong_mode_combine_as_add::<T>),
        ("last partial tile", wrong_last_partial_tile::<T>),
        ("writes nothing", wrong_writes_nothing::<T>),
        ("unfused depth loop", wrong_unfused::<T>),
        ("ignores A strides", wrong_ignores_a_strides::<T>),
        ("flushes subnormals", wrong_flushes_subnormals::<T>),
        ("last element", wrong_last_element::<T>),
        ("writes A padding", wrong_writes_a_padding::<T>),
        ("writes B padding", wrong_writes_b_padding::<T>),
        (
            "read-modify-write below the block",
            wrong_rmw_row_below::<T>,
        ),
        (
            "read-modify-write beyond the block",
            wrong_rmw_column_beyond::<T>,
        ),
        ("stores whole columns", wrong_stores_whole_columns::<T>),
        ("stores whole rows", wrong_stores_whole_rows::<T>),
        ("stores whole tiles", wrong_stores_whole_tiles::<T>),
        ("stores the farthest element", wrong_stores_farthest::<T>),
    ]
}

#[test]
fn test_sme2_selftest_accepts_the_oracle() {
    let f32_kernel: KernelFn<f32> = oracle_block::<f32>;
    let f64_kernel: KernelFn<f64> = oracle_block::<f64>;
    // SAFETY: the oracle is plain Rust and may run on any thread.
    assert!(unsafe { self_test_with(f32_kernel) });
    // SAFETY: as above.
    assert!(unsafe { self_test_with(f64_kernel) });
}

/// Every stand-in of [`wrong_kernels`] must be rejected.
fn rejects_all<T: TestFloat>() {
    for (name, kernel) in wrong_kernels::<T>() {
        // SAFETY: the stand-ins are plain Rust; each stays inside the
        // self-test's buffers (A, B, C with its guard border).
        let verdict = unsafe { self_test_with(kernel) };
        assert!(!verdict, "{} stand-in accepted: {name}", T::NR);
    }
}

#[test]
fn test_sme2_selftest_rejects_wrong_kernels_f32() {
    rejects_all::<f32>();
}

#[test]
fn test_sme2_selftest_rejects_wrong_kernels_f64() {
    rejects_all::<f64>();
}

/// A stand-in, the block it runs on and the scalar pairs.
type Check = (KernelFn<f32>, Case, &'static [(f64, f64)]);

#[test]
fn test_sme2_selftest_wrong_kernels_pass_where_they_are_right() {
    // Each stand-in is wrong only where its check looks: on a block that
    // avoids the defect, the self-test accepts it. This shows the verdicts
    // above come from the intended comparisons.
    let full = Case {
        m: 64,
        n: 16,
        kc: 1,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Plain,
        online: Online::Off,
    };
    // Packed A over 2 row panels (tail 6), odd kc, 2 (f32) column panels,
    // B panels `kc · NR` apart.
    let packed = Case {
        m: 70,
        n: 19,
        kc: 13,
        layout: ALayout::Packed,
        lda_pad: 0,
        b_pad: 0,
        data: Data::Plain,
        online: Online::Off,
    };
    let checks: [Check; 10] = [
        // C is NaN only in M0/M2; with M1 the reading stand-in is right.
        (wrong_reads_c::<f32>, full, &[(1.0, 1.0)]),
        (wrong_mode_scale_as_store::<f32>, full, &[(1.0, 0.0)]),
        (
            wrong_mode_combine_as_add::<f32>,
            full,
            &[(1.0, 1.0), (0.3, 0.0)],
        ),
        // Whole tiles only: the last tile is not partial.
        (wrong_last_partial_tile::<f32>, full, &SCALARS),
        // kc = 1: the product is the only rounding.
        (wrong_unfused::<f32>, full, &SCALARS),
        // Packed: `a_panel == a_depth · kc`.
        (wrong_ignores_a_panel::<f32>, packed, &SCALARS),
        // kc = 13.
        (wrong_depth_multiple_of_4::<f32>, packed, &SCALARS),
        // The last row panel holds 6 rows.
        (wrong_two_tile_tail::<f32>, packed, &SCALARS),
        // 2 column panels only.
        (wrong_third_panel_ignores_kc::<f32>, packed, &SCALARS),
        // `b_panel == kc · NR`.
        (wrong_ignores_b_panel::<f32>, packed, &SCALARS),
    ];
    for (index, (kernel, case, scalars)) in checks.into_iter().enumerate() {
        // SAFETY: plain-Rust stand-ins inside the self-test's buffers.
        let verdict = unsafe { self_test_cases(kernel, &[case], scalars) };
        assert!(verdict, "check {index}");
    }
    // Plain data: no subnormal results, so flushing them changes nothing.
    let flushing: KernelFn<f64> = wrong_flushes_subnormals::<f64>;
    // SAFETY: as above.
    assert!(unsafe { self_test_cases(flushing, &[full], &SCALARS) });
    let ignoring: KernelFn<f64> = wrong_ignores_a_strides::<f64>;
    // SAFETY: as above; with the packed layout its stride is the given one.
    assert!(unsafe { self_test_cases(ignoring, &[full], &SCALARS) });
    // M0 and M2 do not read C: the read-modify-write stand-ins leave the
    // guard alone there, on every kind of block.
    rmw_right_without_c::<f32>(&[full, packed]);
    rmw_right_without_c::<f64>(&[full, packed]);
    // The whole-tile stand-ins store nothing outside a block of whole tiles,
    // in every mode; whole rows only, or whole columns only, are right where
    // that dimension is whole.
    whole_tiles_right::<f32>();
    whole_tiles_right::<f64>();
}

/// The whole-tile stand-ins pass where their reach lies inside the block,
/// and fail where it does not.
fn whole_tiles_right<T: TestFloat>() {
    let block = |m: usize, n: usize| Case {
        m,
        n,
        kc: 3,
        layout: ALayout::InPlace,
        lda_pad: 1,
        b_pad: 1,
        data: Data::Plain,
        online: Online::Off,
    };
    // Whole tiles for both types (16 is a multiple of either NR).
    let whole = block(MR, 16);
    let whole_rows = block(MR, 19);
    let whole_columns = block(70, 16);
    let stand_ins: [(Reach, KernelFn<T>); 4] = [
        (Reach::Columns, wrong_stores_whole_columns::<T>),
        (Reach::Rows, wrong_stores_whole_rows::<T>),
        (Reach::Tiles, wrong_stores_whole_tiles::<T>),
        (Reach::Farthest, wrong_stores_farthest::<T>),
    ];
    for (reach, kernel) in stand_ins {
        for (case, right) in [
            (whole, true),
            (whole_rows, reach == Reach::Rows),
            (whole_columns, reach == Reach::Columns),
        ] {
            // SAFETY: plain-Rust stand-ins inside the self-test's buffers
            // (whole tiles are covered, see `store_whole_tiles`).
            let verdict = unsafe { self_test_cases(kernel, &[case], &SCALARS) };
            assert_eq!(verdict, right, "{} {reach:?} on {case}", T::NR);
        }
    }
}

/// The read-modify-write stand-ins pass on `cases` with the M0 and M2 pairs
/// of [`SCALARS`] (and fail with the M1 and M3 ones).
fn rmw_right_without_c<T: TestFloat>(cases: &[Case]) {
    let without_c: Vec<(f64, f64)> = SCALARS
        .iter()
        .copied()
        .filter(|&(alpha, beta)| !Epilogue::select(alpha, beta).reads_c())
        .collect();
    let with_c: Vec<(f64, f64)> = SCALARS
        .iter()
        .copied()
        .filter(|&(alpha, beta)| Epilogue::select(alpha, beta).reads_c())
        .collect();
    assert_eq!((without_c.len(), with_c.len()), (2, 4));
    let stand_ins: [KernelFn<T>; 2] = [wrong_rmw_row_below::<T>, wrong_rmw_column_beyond::<T>];
    for (index, kernel) in stand_ins.into_iter().enumerate() {
        // SAFETY: plain-Rust stand-ins inside the self-test's buffers.
        let right = unsafe { self_test_cases(kernel, cases, &without_c) };
        assert!(right, "{} read-modify-write stand-in {index}", T::NR);
        for &pair in &with_c {
            // SAFETY: as above.
            let wrong = unsafe { self_test_cases(kernel, cases, &[pair]) };
            assert!(!wrong, "{} stand-in {index} accepted with {pair:?}", T::NR);
        }
    }
}

#[test]
fn test_sme2_selftest_cases_cover_the_claims() {
    for nr in [16, 8] {
        let multi_panel = |case: &Case| case.m > MR && case.n > nr;
        let partial_both = |case: &Case| !case.m.is_multiple_of(MR) && !case.n.is_multiple_of(nr);
        let full_only = |case: &Case| case.m.is_multiple_of(MR) && case.n.is_multiple_of(nr);
        assert!(CASES.iter().any(multi_panel), "NR {nr}: several panels");
        assert!(
            CASES.iter().any(|c| multi_panel(c) && partial_both(c)),
            "NR {nr}: partial tiles in both directions"
        );
        assert!(CASES.iter().any(full_only), "NR {nr}: whole tiles");
        assert!(CASES.iter().any(|c| c.kc == 1 && multi_panel(c)), "kc = 1");
        let in_place = |case: &Case| case.layout == ALayout::InPlace;
        assert!(
            CASES.iter().any(|c| in_place(c) && c.m > MR),
            "NR {nr}: in-place A over several row panels"
        );
        assert!(
            CASES
                .iter()
                .any(|c| c.kc.is_multiple_of(8) && multi_panel(c)),
            "NR {nr}: kc a multiple of 4 and 8, several panels"
        );
        assert!(
            CASES
                .iter()
                .any(|c| !c.kc.is_multiple_of(2) && c.kc > 1 && multi_panel(c)),
            "NR {nr}: odd kc beside it"
        );
        assert!(
            CASES.iter().any(|c| (17..=32).contains(&(c.m % MR))),
            "NR {nr}: a last row panel of 17..32 rows"
        );
        assert!(
            CASES.iter().any(|c| c.n > 2 * nr && c.kc > 1),
            "NR {nr}: three or more column panels with kc > 1"
        );
        assert!(
            CASES.iter().any(|c| c.b_pad > 0 && c.n > nr && c.kc > 1),
            "NR {nr}: padded B panel stride over several panels"
        );
        assert!(
            CASES
                .iter()
                .any(|c| in_place(c) && c.lda_pad != 2 && c.lda_pad > 0),
            "NR {nr}: in-place lda other than m + 2"
        );
        assert!(
            CASES.iter().all(|c| in_place(c) || c.lda_pad == 0),
            "NR {nr}: packed cases carry no lda pad"
        );
        // What only CASES provides beside the sweep: three row panels
        // together with three or more column panels.
        assert!(
            CASES.iter().any(|c| c.m > 2 * MR && c.n > 2 * nr),
            "NR {nr}: three row panels and three column panels"
        );
    }
    // More than two row panels (the sweep has them as well, with the other
    // depths and A in place; see the path table).
    assert!(CASES.iter().any(|c| c.m > 2 * MR), "three row panels");
    assert!(
        CASES.iter().any(|c| c.kc > MR && c.kc % 2 == 1),
        "deep odd kc"
    );
    assert!(CASES.iter().any(|c| c.m == 1 && c.n == 1));
    assert!(CASES.iter().any(|c| c.layout == ALayout::InPlace));
    assert!(CASES.iter().any(|c| c.data == Data::Special && c.kc > 1));
    assert!(CASES.iter().any(|c| (c.m, c.n, c.kc) == (70, 19, 13)));
    let modes: Vec<Epilogue> = SCALARS
        .iter()
        .map(|&(alpha, beta)| Epilogue::select(alpha, beta))
        .collect();
    for mode in [
        Epilogue::Store,
        Epilogue::Add,
        Epilogue::Scale,
        Epilogue::ScaleCombine,
    ] {
        assert!(modes.contains(&mode), "{mode:?}");
    }
    // M3 in all three branches, with inexact scalars.
    assert!(SCALARS.contains(&(1.0, -0.7)) && SCALARS.contains(&(-1.3, 1.0)));
    assert!(SCALARS.contains(&(0.3, -0.7)));
    let empty: [Case; 0] = [];
    let probe: KernelFn<f32> = wrong_writes_nothing::<f32>;
    // SAFETY: the stand-in touches nothing.
    assert!(unsafe { self_test_cases(probe, &empty, &SCALARS) });
    // A degenerate block is refused, not run.
    let degenerate = Case { m: 0, ..CASES[0] };
    let oracle: KernelFn<f32> = oracle_block::<f32>;
    // SAFETY: the oracle is plain Rust.
    assert!(!unsafe { self_test_cases(oracle, &[degenerate], &SCALARS) });
    // So is a packed block with an lda pad (it would mean nothing).
    let packed_pad = Case {
        layout: ALayout::Packed,
        lda_pad: 1,
        ..CASES[0]
    };
    // SAFETY: the oracle is plain Rust.
    assert!(!unsafe { self_test_cases(oracle, &[packed_pad], &SCALARS) });
}

// ---------------------------------------------------------------------------
// The guard border and the mismatch report
// ---------------------------------------------------------------------------

/// Whether `bits` (of a `T`) is a signalling NaN: all exponent bits set, the
/// quiet bit clear, a non-zero payload.
fn signalling<T: SmeElement>(bits: u64) -> bool {
    let (exponent, quiet) = if T::NR == 16 {
        (0x7f80_0000, 0x0040_0000)
    } else {
        (0x7ff0_0000_0000_0000, 0x0008_0000_0000_0000)
    };
    let payload = bits & (quiet - 1);
    bits & exponent == exponent && bits & quiet == 0 && payload != 0
}

/// The bits of the guard after the kernels' dead-lane read-modify-write:
/// the sentinel with its quiet bit set.
fn quieted_guard<T: SmeElement>() -> u64 {
    let quiet = if T::NR == 16 {
        0x0040_0000
    } else {
        0x0008_0000_0000_0000
    };
    sentinel::<T>().bits() | quiet
}

/// The mechanism of the guard, for one type: the sentinel and the fills of
/// A and B are signalling NaNs, and arithmetic on the sentinel (with a zero,
/// as a dead accumulator lane gives) changes its bits.
fn guard_is_signalling<T: TestFloat>() {
    let guard = sentinel::<T>();
    assert!(
        signalling::<T>(guard.bits()),
        "{}: {:#x}",
        T::NR,
        guard.bits()
    );
    assert!(signalling::<T>(a_spare::<T>().bits()), "{}: A", T::NR);
    assert!(signalling::<T>(b_gap::<T>().bits()), "{}: B", T::NR);
    let fills = [guard.bits(), a_spare::<T>().bits(), b_gap::<T>().bits()];
    assert!(fills[0] != fills[1] && fills[1] != fills[2] && fills[0] != fills[2]);
    // `acc + C` (M1) and `fma(C, beta, acc · alpha)` (M3) with `acc = 0`.
    let sum = black_box(guard) + black_box(T::ZERO);
    assert_ne!(sum.bits(), guard.bits(), "{}: guard + 0", T::NR);
    assert_eq!(sum.bits(), quieted_guard::<T>(), "{}: guard + 0", T::NR);
    let fused = black_box(guard).fused(black_box(T::of(-0.7)), black_box(T::ZERO));
    assert_ne!(fused.bits(), guard.bits(), "{}: fma(guard, beta, 0)", T::NR);
    // A copy keeps the bits (the self-test only copies and compares them).
    let copied = [guard; 3].to_vec().clone();
    assert!(copied.iter().all(|value| value.bits() == guard.bits()));
}

#[test]
fn test_sme2_selftest_guard_is_signalling() {
    guard_is_signalling::<f32>();
    guard_is_signalling::<f64>();
}

/// The first mismatch of `kernel` over the whole list (`self_test_with`).
fn report<T: SmeElement>(kernel: KernelFn<T>) -> Mismatch {
    // SAFETY: the stand-ins are plain Rust and stay inside the self-test's
    // buffers.
    match unsafe { self_test_with_detailed(kernel) } {
        Ok(()) => panic!("{} stand-in accepted", T::NR),
        Err(mismatch) => {
            println!("{mismatch}");
            mismatch
        }
    }
}

/// The location each stand-in with a known defect is reported at, for one
/// type: the first block of [`CASES`] (`70 × 19 × 13`, packed, `ldc = 73`),
/// in the first pair that shows the defect.
fn reports_where<T: TestFloat>() {
    let bits = if T::NR == 16 { 32 } else { 64 };
    let case = CASES[0];
    let (m, n) = (case.m as i32, case.n as i32);
    let ldc = case.m + 3;
    let at = |pair: usize, difference: Difference| Mismatch {
        element_bits: bits,
        block: 0,
        case,
        pair,
        alpha: SCALARS[pair].0,
        beta: SCALARS[pair].1,
        difference,
    };
    // One ulp at C(0, 0): block 0, pair 0, element (0, 0).
    let one_ulp = report(wrong_one_ulp::<T>);
    let Difference::Element {
        row: 0,
        col: 0,
        found,
        wanted,
    } = one_ulp.difference
    else {
        panic!("{}: {one_ulp:?}", T::NR);
    };
    assert_eq!(found, wanted + 1, "{}", T::NR);
    assert_eq!(
        one_ulp,
        at(
            0,
            Difference::Element {
                row: 0,
                col: 0,
                found,
                wanted
            }
        )
    );
    let line = one_ulp.to_string();
    assert!(
        line.contains("block 0 (70x19x13") && line.contains("C(0, 0) is "),
        "{line}"
    );
    let guard = |row: i32, col: i32, found: u64| kernel_guard(row, col, ldc, found);
    // The default NaN written to the row below: guard element (m, 0).
    let below = guard(m, 0, default_nan::<T>().bits());
    assert_eq!(report(wrong_writes_row_below::<T>), at(0, below));
    // The element before C(0, 0): offset -1, split as row ldc - 1 of
    // column -1 (the same element as "one row above C(0, 0)").
    let before = guard(ldc as i32 - 1, -1, default_nan::<T>().bits());
    let before_report = report(wrong_writes_before_block::<T>);
    assert_eq!(before_report, at(0, before));
    assert!(
        before_report
            .to_string()
            .contains(&format!("(offset -1 from C(0, 0), ldc {ldc})")),
        "{before_report}"
    );
    // The read-modify-writes: first in M1 (pair 1), the quieted sentinel at
    // row m of column 0, and at row 0 of column n.
    let quiet = quieted_guard::<T>();
    let rmw_row = report(wrong_rmw_row_below::<T>);
    assert_eq!(rmw_row, at(1, guard(m, 0, quiet)));
    let rmw_column = report(wrong_rmw_column_beyond::<T>);
    assert_eq!(rmw_column, at(1, guard(0, n, quiet)));
    assert!(
        rmw_column.to_string().contains(&format!(
            "wrote C(0, {n}) outside the block (offset {} from C(0, 0), ldc {ldc})",
            n as usize * ldc
        )),
        "{rmw_column}"
    );
    // The whole-tile stores (M0, pair 0: a dead lane stores +0): all NR
    // columns of the partial last column panel first change C(0, n); all 64
    // rows of the partial last row panel, and whole tiles, C(m, 0).
    assert_eq!(
        report(wrong_stores_whole_columns::<T>),
        at(0, guard(0, n, 0))
    );
    assert_eq!(report(wrong_stores_whole_rows::<T>), at(0, guard(m, 0, 0)));
    assert_eq!(report(wrong_stores_whole_tiles::<T>), at(0, guard(m, 0, 0)));
    // The farthest element a whole tile reaches is inside the buffer and
    // checked: C(64 · ceil(m / 64) - 1, NR · ceil(n / NR) - 1) = C(127, ..),
    // reported by its offset and split with the row in 0..ldc (it aliases
    // an element of another column).
    let far_row = case.m.div_ceil(MR) * MR - 1;
    let far_col = case.n.div_ceil(T::NR) * T::NR - 1;
    let far = (far_row + far_col * ldc) as i64;
    let farthest = report(wrong_stores_farthest::<T>);
    let ldc_i64 = ldc as i64;
    let split = guard(
        far.rem_euclid(ldc_i64) as i32,
        far.div_euclid(ldc_i64) as i32,
        0,
    );
    assert_eq!(farthest, at(0, split));
    assert!(
        farthest
            .to_string()
            .contains(&format!("(offset {far} from C(0, 0), ldc {ldc})")),
        "{farthest}"
    );
    // A's padding A(m, 0) of the last row panel, B's padding B(0, n).
    let a_index = (case.m / MR) * MR * case.kc + case.m % MR;
    let b_index = (case.n / T::NR) * (T::NR * case.kc + case.b_pad) + case.n % T::NR;
    assert_eq!(
        report(wrong_writes_a_padding::<T>),
        at(0, Difference::AChanged { index: a_index })
    );
    assert_eq!(
        report(wrong_writes_b_padding::<T>),
        at(0, Difference::BChanged { index: b_index })
    );
    // A refused block, through the chosen-list entry point.
    let degenerate = Case { m: 0, ..case };
    let oracle: KernelFn<T> = oracle_block::<T>;
    // SAFETY: the oracle is plain Rust.
    let refused = unsafe { self_test_cases_detailed(oracle, &[case, degenerate], &SCALARS) };
    let Err(refused) = refused else {
        panic!("{}: a degenerate block passed", T::NR);
    };
    assert_eq!((refused.block, refused.pair), (1, 0));
    assert_eq!(refused.difference, Difference::Refused);
    assert!(refused.to_string().contains("refused"), "{refused}");
}

/// A guard difference of the kernel's C at (`row`, `col`) relative to
/// `C(0, 0)` with leading dimension `ldc`, the bits `found`; its offset is
/// `row + col · ldc`. `row` must be in `0..ldc` (the report's split).
fn kernel_guard(row: i32, col: i32, ldc: usize, found: u64) -> Difference {
    assert!((0..ldc as i32).contains(&row), "row {row} outside 0..{ldc}");
    Difference::KernelGuard {
        row,
        col,
        offset: i64::from(row) + i64::from(col) * ldc as i64,
        ldc: ldc as u32,
        found,
    }
}

#[test]
fn test_sme2_selftest_reports_where() {
    reports_where::<f32>();
    reports_where::<f64>();
}

#[test]
fn test_sme2_selftest_report_is_one_line() {
    // Every kind of difference prints as one line with the element type,
    // the block, the pair and the bits in full width.
    let case = CASES[5];
    let differences = [
        Difference::Refused,
        Difference::AChanged { index: 7 },
        Difference::BChanged { index: 8 },
        Difference::KernelGuard {
            row: 35,
            col: -1,
            offset: -1,
            ldc: 36,
            found: 0x7fde_1f7e,
        },
        Difference::OracleGuard {
            row: 0,
            col: 17,
            offset: 612,
            ldc: 36,
            found: 0x7fde_1f7e,
        },
        Difference::Element {
            row: 3,
            col: 4,
            found: 0x3f80_0001,
            wanted: 0x3f80_0000,
        },
    ];
    for difference in differences {
        let mismatch = Mismatch {
            element_bits: 32,
            block: 5,
            case,
            pair: 3,
            alpha: 1.0,
            beta: -0.7,
            difference,
        };
        let line = mismatch.to_string();
        println!("{line}");
        assert!(!line.contains('\n'), "{line}");
        assert!(
            line.starts_with("SME2 self-test mismatch, f32: block 5 (33x17x5, A packed")
                && line.contains("special data")
                && line.contains("pair 3 (alpha 1, beta -0.7)"),
            "{line}"
        );
        // A guard line carries the offset from C(0, 0) and ldc beside the
        // coordinates.
        if let Difference::KernelGuard { offset, .. } | Difference::OracleGuard { offset, .. } =
            difference
        {
            assert!(
                line.contains(&format!(
                    "outside the block (offset {offset} from C(0, 0), ldc 36)"
                )),
                "{line}"
            );
        }
    }
    let wide = Mismatch {
        element_bits: 64,
        block: 0,
        case,
        pair: 0,
        alpha: 1.0,
        beta: 0.0,
        difference: Difference::Element {
            row: 0,
            col: 0,
            found: 1,
            wanted: 2,
        },
    };
    assert!(
        wide.to_string()
            .ends_with("C(0, 0) is 0x0000000000000001, the oracle gives 0x0000000000000002"),
        "{wide}"
    );
}

// ---------------------------------------------------------------------------
// The real kernels: run_self_test and self_test_kernels
// ---------------------------------------------------------------------------

/// Kernel calls of one `self_test_with` run for `T`, as `u64`.
fn calls_of<T: SmeElement>() -> u64 {
    self_test_calls::<T>() as u64
}

thread_local! {
    /// Calls of [`counted_oracle`] on this thread: `[f32, f64]`.
    static COUNTED: Cell<[u64; 2]> = const { Cell::new([0, 0]) };
}

/// The oracle, counting its calls per type on this thread (exact even while
/// other tests run kernels).
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn counted_oracle<T: SmeElement>(args: &MacroArgs<T>) {
    COUNTED.with(|counted| {
        let mut calls = counted.get();
        calls[usize::from(T::NR != 16)] += 1;
        counted.set(calls);
    });
    // SAFETY: forwarded from this function's contract.
    unsafe { oracle_block(args) };
}

/// Resets [`COUNTED`] and returns the verdict of `self_test_kernels` on the
/// given kernels with the calls it made per type.
fn binder_verdict(
    f32_kernel: KernelFn<f32>,
    f64_kernel: Option<KernelFn<f64>>,
) -> (bool, [u64; 2]) {
    COUNTED.with(|counted| counted.set([0, 0]));
    // SAFETY: every kernel these tests pass is plain Rust (the oracle, a
    // counting oracle or a wrong stand-in that stays inside the buffers).
    let verdict = unsafe { self_test_kernels(f32_kernel, f64_kernel) };
    (verdict, COUNTED.with(Cell::get))
}

#[test]
fn test_sme2_selftest_binder_combines_verdicts() {
    // Plain Rust stand-ins: the binder's verdict logic runs on every host.
    let ok32: KernelFn<f32> = counted_oracle::<f32>;
    let ok64: KernelFn<f64> = counted_oracle::<f64>;
    let wrong32: KernelFn<f32> = wrong_one_ulp::<f32>;
    let wrong64: KernelFn<f64> = wrong_one_ulp::<f64>;
    // The fixed blocks: 7 × 6 calls per type; with the 4 online cases and
    // the two sweeps (4 row variants of f32, 8 of f64, 12 blocks each, and 6
    // blocks of three row panels, once plain and once with online packing)
    // 119 × 6 for f32 and 215 × 6 for f64.
    assert_eq!(CASES.len() * SCALARS.len(), 42);
    let (n32, n64) = (calls_of::<f32>(), calls_of::<f64>());
    assert_eq!((n32, n64), (714, 1_290));
    // Both pass: both types run, every block once.
    assert_eq!(binder_verdict(ok32, Some(ok64)), (true, [n32, n64]));
    // f32 alone: the f64 kernel is not run, so fewer kernel calls.
    assert_eq!(binder_verdict(ok32, None), (true, [n32, 0]));
    // A wrong f64 kernel fails the whole verdict, after f32 passed (the
    // wrong stand-in does not count its calls).
    assert_eq!(binder_verdict(ok32, Some(wrong64)), (false, [n32, 0]));
    // A wrong f32 kernel fails it and stops before f64.
    assert_eq!(binder_verdict(wrong32, Some(ok64)), (false, [0, 0]));
    assert_eq!(binder_verdict(wrong32, None), (false, [0, 0]));
}

/// Runs `run_self_test(f64_present)` on the real kernels: the thread is
/// ready before, the verdict is `true`, no SME state is left, and this
/// thread made exactly the expected kernel calls. Restores FPSR.
fn real_run_self_test(test: &str, f64_present: bool) {
    assert!(kernel_ready(), "{test}: thread not ready");
    let fpsr = fpsr_read();
    let entries = kernel_entries_here();
    let start = Instant::now();
    // SAFETY: the caller's `hardware_or_skip` returned true (SME2 with SVL
    // 512, and FEAT_SME_F64F64 when `f64_present`), and `kernel_ready()` was
    // true on this thread just above.
    let verdict = unsafe { run_self_test_detailed(f64_present) };
    let elapsed = start.elapsed();
    let state = sme_state();
    let grown = kernel_entries_here() - entries;
    fpsr_write(fpsr);
    println!("{test}: run_self_test({f64_present}) took {elapsed:?}");
    if let Err(mismatch) = verdict {
        panic!("{test}: run_self_test({f64_present}) failed: {mismatch}");
    }
    assert_eq!(state, Some((0, 0)), "{test}: SME state left");
    // Counted on this thread, where the self-test makes every call: one
    // entry per block, as on the stand-ins above.
    let want = calls_of::<f32>() + if f64_present { calls_of::<f64>() } else { 0 };
    assert_eq!(grown, want, "{test}: kernel calls on this thread");
    assert!(kernel_ready(), "{test}: thread not ready after");
    if !f64_present {
        // The `bool` binding agrees (it is `is_ok()` of the detailed one).
        let fpsr = fpsr_read();
        // SAFETY: as above; the thread is ready again (checked just above).
        let again = unsafe { run_self_test(false) };
        fpsr_write(fpsr);
        assert!(again, "{test}: run_self_test(false) failed");
    }
}

#[test]
fn test_sme2_selftest_real_kernels_pass() {
    let test = "test_sme2_selftest_real_kernels_pass";
    if hardware_or_skip(test, Need::F64) {
        real_run_self_test(test, true);
    }
}

#[test]
fn test_sme2_selftest_real_f32_alone_passes() {
    let test = "test_sme2_selftest_real_f32_alone_passes";
    if hardware_or_skip(test, Need::F32) {
        real_run_self_test(test, false);
    }
}

/// `self_test_with` on one real kernel, timed. Restores FPSR.
///
/// # Safety
///
/// `hardware_or_skip` returned true for the kernel's type.
unsafe fn real_kernel_passes<T: SmeElement>(test: &str, kernel: KernelFn<T>) {
    assert!(kernel_ready(), "{test}: thread not ready");
    let fpsr = fpsr_read();
    let entries = kernel_entries_here();
    let start = Instant::now();
    // SAFETY: SME2 (with FEAT_SME_F64F64 for `f64`) is present by this
    // function's contract, and `kernel_ready()` was true just above.
    let verdict = unsafe { self_test_with_detailed(kernel) };
    let elapsed = start.elapsed();
    let state = sme_state();
    let grown = kernel_entries_here() - entries;
    fpsr_write(fpsr);
    println!("{test}: self_test_with took {elapsed:?}");
    if let Err(mismatch) = verdict {
        panic!("{test}: the kernel differs from the oracle: {mismatch}");
    }
    assert_eq!(state, Some((0, 0)), "{test}: SME state left");
    let want = calls_of::<T>();
    assert_eq!(grown, want, "{test}: kernel calls on this thread");
}

#[test]
fn test_sme2_selftest_real_kernel_f32() {
    let test = "test_sme2_selftest_real_kernel_f32";
    if hardware_or_skip(test, Need::F32) {
        // SAFETY: `hardware_or_skip` returned true for `f32`.
        unsafe { real_kernel_passes::<f32>(test, sme2_gemm_f32) };
    }
}

#[test]
fn test_sme2_selftest_real_kernel_f64() {
    let test = "test_sme2_selftest_real_kernel_f64";
    if hardware_or_skip(test, Need::F64) {
        // SAFETY: `hardware_or_skip` returned true for `f64`
        // (FEAT_SME_F64F64).
        unsafe { real_kernel_passes::<f64>(test, sme2_gemm_f64) };
    }
}

/// The C buffer of every block covers, for each type, every element that a
/// whole `64 × NR` tile at any tile position of the block addresses, the
/// guard column before the block and the column after it (computed here from
/// the shape alone, independently of `c_layout`).
fn guard_covers_whole_tiles<T: SmeElement>() -> usize {
    let mut beyond_two_columns = 0;
    let mut inside_two_columns = Vec::new();
    for case in self_test_blocks(T::TILE_ROWS, T::NR) {
        let Some(CLayout { ldc, origin, len }) = c_layout::<T>(&case) else {
            panic!("{}: no layout for {case}", T::NR);
        };
        assert_eq!((ldc, origin), (case.m + 3, case.m + 4), "{case}");
        assert_eq!(
            run_buffers::<T>(&case).map(|[_, _, c, _]| c),
            Some(len),
            "{case}"
        );
        // Every tile position: its farthest element, `C(i0 + 63, j0 + NR - 1)`.
        let mut reach = 0;
        for j0 in (0..case.n).step_by(T::NR) {
            for i0 in (0..case.m).step_by(MR) {
                let far = (i0 + MR - 1) + (j0 + T::NR - 1) * ldc;
                assert!(origin + far < len, "{}: {case}: tile ({i0}, {j0})", T::NR);
                reach = reach.max(far);
            }
        }
        // The column after the block (rows 0..ldc - 1) and the one before.
        let column_after = case.n * ldc + ldc - 2;
        assert!(origin + column_after < len, "{}: {case}", T::NR);
        assert!(origin > ldc, "{case}");
        // Nothing more than the farther of the two.
        assert_eq!(len, origin + reach.max(column_after) + 1, "{case}");
        // A buffer of `ldc · (n + 2)` elements would end before it.
        if origin + reach >= ldc * (case.n + 2) {
            beyond_two_columns += 1;
        } else {
            inside_two_columns.push((case.m, case.n, case.online));
        }
    }
    println!(
        "{}: whole tiles inside ldc · (n + 2): {inside_two_columns:?}",
        T::NR
    );
    beyond_two_columns
}

#[test]
fn test_sme2_selftest_guard_covers_whole_tiles() {
    let f32_blocks = guard_covers_whole_tiles::<f32>();
    let f64_blocks = guard_covers_whole_tiles::<f64>();
    println!(
        "blocks whose whole tiles reach beyond ldc · (n + 2): f32 {f32_blocks}, f64 {f64_blocks}"
    );
    // Every block but a few has a whole tile beyond `ldc · (n + 2)`: all
    // but `64 × 16` of the 119 for f32; all but `64 × 16`, `84 × 40`, and
    // `64 × 15` and `128 × 15` of each sweep, of the 215 for f64 (tiles
    // whole in both directions, or rows that end before the column after
    // the block).
    assert_eq!((f32_blocks, f64_blocks), (118, 209));
}

/// Kernel calls and bytes of one `self_test_with` run for `T`: calls, the
/// largest run's bytes, the largest single buffer's bytes, all runs' bytes.
fn cost<T: SmeElement>(cases: impl Iterator<Item = Case>) -> (usize, usize, usize, usize) {
    let mut calls = 0;
    let mut largest = 0;
    let mut largest_buffer = 0;
    let mut total = 0;
    for case in cases {
        let (Some(bytes), Some(buffers)) = (run_bytes::<T>(&case), run_buffers::<T>(&case)) else {
            panic!("refused block {case:?}");
        };
        calls += SCALARS.len();
        largest = largest.max(bytes);
        let buffer = buffers.into_iter().max().unwrap_or(0) * size_of::<T>();
        largest_buffer = largest_buffer.max(buffer);
        total += bytes * SCALARS.len();
    }
    (calls, largest, largest_buffer, total)
}

#[test]
fn test_sme2_selftest_cost_matches_the_docs() {
    // The figures of the module docs of `selftest.rs` and `detect.rs`.
    let before_f32 = cost::<f32>(CASES.into_iter());
    let before_f64 = cost::<f64>(CASES.into_iter());
    let after_f32 = cost::<f32>(self_test_blocks(16, 16));
    let after_f64 = cost::<f64>(self_test_blocks(8, 8));
    println!("CASES only: f32 {before_f32:?}, f64 {before_f64:?}");
    println!("with the sweep: f32 {after_f32:?}, f64 {after_f64:?}");
    // (calls, bytes of the largest run, bytes of the largest single buffer,
    // bytes of all runs together)
    assert_eq!(before_f32, (42, 80_844, 26_308, 1_584_480));
    assert_eq!(before_f64, (42, 143_744, 44_104, 2_698_416));
    assert_eq!(after_f32, (714, 131_860, 26_308, 29_329_440));
    assert_eq!(after_f64, (1_290, 207_464, 51_200, 61_938_096));
    let largest = |tile: usize| {
        self_test_blocks(tile, tile)
            .max_by_key(|case| {
                if tile == 16 {
                    run_bytes::<f32>(case)
                } else {
                    run_bytes::<f64>(case)
                }
            })
            .map(|case| (case.m, case.n, case.kc, case.layout, case.online))
    };
    println!("largest runs: f32 {:?}, f64 {:?}", largest(16), largest(8));
    // The largest run: the third online case, `84 × (2 · NR + 3) × 16` in
    // place with online packing (three column panels, so its buffer's guard
    // holds two more copies), for either type.
    assert_eq!(
        largest(16),
        Some((84, 35, 16, ALayout::InPlace, Online::Pack))
    );
    assert_eq!(
        largest(8),
        Some((84, 19, 16, ALayout::InPlace, Online::Pack))
    );
    assert_eq!(after_f32.0, self_test_calls::<f32>());
    assert_eq!(after_f64.0, self_test_calls::<f64>());
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_selftest_sweep_shape() {
    // The sweep is generic: for a tile height and panel width it yields
    // `variants · SWEEP_BLOCKS_PER_VARIANT` valid blocks of the documented
    // shape (one full and one partial column panel, 1 or 64 + rows rows).
    for (tile_rows, nr) in [(16, 16), (8, 8), (32, 4), (4, 2)] {
        let sweep: Vec<Case> = sweep_cases(tile_rows, nr).collect();
        let per_variant = MR / tile_rows * SWEEP_BLOCKS_PER_VARIANT;
        assert_eq!(sweep.len(), per_variant + SWEEP_MIDDLE_BLOCKS);
        for case in &sweep {
            assert!(case.n > nr && case.n < 2 * nr, "{case:?}");
            assert!(SWEEP_DEPTHS.contains(&case.kc), "{case:?}");
            assert!(case.m >= 1 && case.m <= 3 * MR, "{case:?}");
            assert_eq!(case.data, Data::Plain);
            assert_eq!(case.layout == ALayout::Packed, case.lda_pad == 0);
        }
        // The first part has at most two row panels; the blocks of three
        // row panels (129 ≤ m ≤ 192) come last, with last panels of
        // different sizes, each depth once per layout.
        let (front, middle) = sweep.split_at(per_variant);
        assert!(front.iter().all(|case| case.m <= 2 * MR));
        assert!(
            middle
                .iter()
                .all(|case| case.m > 2 * MR && case.m <= 3 * MR),
            "{middle:?}"
        );
        let sizes: BTreeSet<usize> = middle.iter().map(|case| case.m).collect();
        assert_eq!(sizes.len(), SWEEP_MIDDLE_BLOCKS, "{middle:?}");
        for layout in [ALayout::Packed, ALayout::InPlace] {
            let depths: BTreeSet<usize> = middle
                .iter()
                .filter(|case| case.layout == layout)
                .map(|case| case.kc)
                .collect();
            assert_eq!(depths, SWEEP_DEPTHS.into_iter().collect(), "{layout:?}");
        }
    }
    // Degenerate shapes give no blocks rather than wrong ones.
    assert_eq!(sweep_cases(0, 16).count(), 0);
    assert_eq!(sweep_cases(16, 1).count(), 0);
    assert_eq!(sweep_cases(16, 0).count(), 0);
}
