//! Tests of the first-use self-test (`selftest.rs`).
//!
//! - Its verdicts on the oracle and on wrong plain-Rust stand-in kernels, the
//!   claims its fixed blocks ([`CASES`]) make, and the verdict logic of the
//!   binder (`self_test_kernels`) with exact call counts. These run on every
//!   host.
//! - The path table: from the self-test's own block list it computes, for the
//!   tile height and panel width of each element type, every path a block
//!   takes through that type's macro-kernel, and requires the set of paths
//!   that the kernels' structure defines (the [`KernelPaths`] descriptions).
//!   This runs on every host and is the permanent guard that the self-test
//!   reaches every part of the kernels.
//! - The real macro-kernels through `run_self_test` and `self_test_with`;
//!   these skip without SME2.

use core::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::time::Instant;

use super::args::{Epilogue, MR, MacroArgs, SmeElement};
use super::detect;
use super::kernel_f32::sme2_gemm_f32;
use super::kernel_f64::sme2_gemm_f64;
use super::oracle::oracle_block;
use super::selftest::{
    ALayout, CASES, Case, Data, KernelFn, SCALARS, SWEEP_BLOCKS_PER_VARIANT, SWEEP_DEPTHS,
    run_bytes, run_self_test, self_test_blocks, self_test_calls, self_test_cases,
    self_test_kernels, self_test_with, sweep_cases,
};
use super::test_support::{
    Need, fpsr_read, fpsr_write, hardware_or_skip, kernel_entries, sme_state,
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

/// Every wrong stand-in for one type, with its name.
fn wrong_kernels<T: TestFloat>() -> [(&'static str, KernelFn<T>); 19] {
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
    }
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
    // The fixed blocks: 7 × 6 calls per type; with the sweep (4 row variants
    // of f32, 8 of f64, 12 blocks each) 55 × 6 for f32 and 103 × 6 for f64.
    assert_eq!(CASES.len() * SCALARS.len(), 42);
    let (n32, n64) = (calls_of::<f32>(), calls_of::<f64>());
    assert_eq!((n32, n64), (330, 618));
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
/// ready before, the verdict is `true`, no SME state is left, and the process
/// made at least the expected kernel calls. Restores FPSR.
fn real_run_self_test(test: &str, f64_present: bool) {
    assert!(detect::thread_ready(), "{test}: thread not ready");
    let fpsr = fpsr_read();
    let entries = kernel_entries();
    let start = Instant::now();
    // SAFETY: the caller's `hardware_or_skip` returned true (SME2 with SVL
    // 512, and FEAT_SME_F64F64 when `f64_present`), and `thread_ready()` was
    // true on this thread just above.
    let verdict = unsafe { run_self_test(f64_present) };
    let elapsed = start.elapsed();
    let state = sme_state();
    let grown = kernel_entries() - entries;
    fpsr_write(fpsr);
    println!("{test}: run_self_test({f64_present}) took {elapsed:?}");
    assert!(verdict, "{test}: run_self_test({f64_present}) failed");
    assert_eq!(state, Some((0, 0)), "{test}: SME state left");
    // A lower bound only: other tests run kernels concurrently. The exact
    // counts are checked on stand-ins above.
    let want = calls_of::<f32>() + if f64_present { calls_of::<f64>() } else { 0 };
    assert!(grown >= want, "{test}: {grown} kernel calls, want {want}");
    assert!(detect::thread_ready(), "{test}: thread not ready after");
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
    assert!(detect::thread_ready(), "{test}: thread not ready");
    let fpsr = fpsr_read();
    let entries = kernel_entries();
    let start = Instant::now();
    // SAFETY: SME2 (with FEAT_SME_F64F64 for `f64`) is present by this
    // function's contract, and `thread_ready()` was true just above.
    let verdict = unsafe { self_test_with(kernel) };
    let elapsed = start.elapsed();
    let state = sme_state();
    let grown = kernel_entries() - entries;
    fpsr_write(fpsr);
    println!("{test}: self_test_with took {elapsed:?}");
    assert!(verdict, "{test}: the kernel differs from the oracle");
    assert_eq!(state, Some((0, 0)), "{test}: SME state left");
    let want = calls_of::<T>();
    assert!(grown >= want, "{test}: {grown} kernel calls, want {want}");
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

// ---------------------------------------------------------------------------
// The path table: every path of both kernels is taken by the self-test
// ---------------------------------------------------------------------------

/// The structure of one macro-kernel as data: what decides the path a block
/// takes through it.
///
/// Both kernels (`kernel_f32.rs`, `kernel_f64.rs`) walk column panels of
/// `nr` columns and, inside each, row panels of 64 rows. For one (row panel,
/// column panel) pair the path is:
///
/// - the row variant: the number of live tiles `L = ceil(rows / tile_rows)`,
///   `1..=variants`, where `rows = min(64, m - i0)`; each variant has its own
///   zeroing, depth loop and remainder, and the epilogue's tile chain stops
///   after tile `L`;
/// - the depth: the loop unrolled by `unroll` makes `kc / unroll` passes
///   (none, or two and more, which takes its back branch and pointer steps)
///   and then a remainder of `kc % unroll` steps;
/// - the epilogue mode, M0 to M3, and for M3 the branch of the contract
///   (`alpha == 1`, `beta == 1`, neither) that the scalars select;
/// - the column panel: full (`nr` columns) or the partial last one;
/// - the position of the row panel: the only one, a leading (always full)
///   one, or the last one after a full one;
/// - the A layout: packed, or in place;
/// - the last live tile of the row panel: full, or holding a single row (the
///   lower edge of the variant, where the row predicates and the tile chain
///   cut).
///
/// A change of a kernel's structure (another row variant, another unrolling
/// of the depth loop, another epilogue mode or branch, another edge that the
/// code treats on its own) must be reflected here and in [`paths_of_call`];
/// the table then fails until the sweep of `selftest.rs` takes the new paths.
#[derive(Clone, Copy, Debug)]
struct KernelPaths {
    /// The element type.
    name: &'static str,
    /// Rows of one ZA tile (`T::TILE_ROWS`).
    tile_rows: usize,
    /// Columns of one B panel (`T::NR`).
    nr: usize,
    /// Row variants: `1..=variants` live tiles.
    variants: usize,
    /// Depth steps per pass of the unrolled depth loop.
    unroll: usize,
}

/// `kernel_f32.rs`: 16-row tiles, 16-column panels, variants of 1 to 4
/// tiles (labels 41 to 48 and 61 to 64), a depth loop unrolled by two.
const F32_PATHS: KernelPaths = KernelPaths {
    name: "f32",
    tile_rows: 16,
    nr: 16,
    variants: 4,
    unroll: 2,
};

/// `kernel_f64.rs`: 8-row tiles, 8-column panels, variants of 1 to 8 tiles
/// (labels 41 to 48, 61 to 68 and 71 to 78), a depth loop unrolled by two.
const F64_PATHS: KernelPaths = KernelPaths {
    name: "f64",
    tile_rows: 8,
    nr: 8,
    variants: 8,
    unroll: 2,
};

/// The epilogue of a call: the mode and, for M3, the branch of the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EpilogueClass {
    /// `alpha == 1`, `beta == 0`.
    M0,
    /// `alpha == 1`, `beta == 1`.
    M1,
    /// `alpha != 1`, `beta == 0`.
    M2,
    /// M3 with `alpha == 1`.
    M3AlphaOne,
    /// M3 with `beta == 1`.
    M3BetaOne,
    /// M3 with neither.
    M3General,
}

/// Every [`EpilogueClass`].
const EPILOGUES: [EpilogueClass; 6] = [
    EpilogueClass::M0,
    EpilogueClass::M1,
    EpilogueClass::M2,
    EpilogueClass::M3AlphaOne,
    EpilogueClass::M3BetaOne,
    EpilogueClass::M3General,
];

/// The column panel of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Columns {
    /// `nr` columns.
    Full,
    /// The partial last panel.
    Partial,
}

/// The position of the row panel of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Position {
    /// The first and last row panel (`m ≤ 64`).
    Only,
    /// A row panel that is not the last one (always 64 full rows).
    Leading,
    /// The last row panel after a full one.
    Last,
}

/// The A layout of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layout {
    /// `a_depth = 64`, `a_panel = 64 · kc`.
    Packed,
    /// Any other strides (in place: `a_depth = lda`, `a_panel = 64`).
    InPlace,
}

/// The last live tile of the row panel of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LastTile {
    /// All `tile_rows` rows live.
    Full,
    /// One live row.
    OneRow,
    /// Between the two (never required).
    Other,
}

/// One path through a kernel: what one (row panel, column panel) pair of one
/// call takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Path {
    /// Live tiles (the row variant).
    tiles: usize,
    /// Passes of the unrolled depth loop: 0, 1, or 2 for two and more.
    passes: usize,
    /// Depth steps after the loop (`kc % unroll`).
    remainder: usize,
    /// Mode and M3 branch.
    epilogue: EpilogueClass,
    /// Full or partial column panel.
    columns: Columns,
    /// Position of the row panel.
    position: Position,
    /// A layout.
    layout: Layout,
    /// The last live tile.
    last_tile: LastTile,
}

/// A class of paths: a [`Path`] reduced to some of its dimensions (the
/// others are `None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Pattern {
    /// See [`Path`].
    tiles: Option<usize>,
    /// See [`Path`].
    passes: Option<usize>,
    /// See [`Path`].
    remainder: Option<usize>,
    /// See [`Path`].
    epilogue: Option<EpilogueClass>,
    /// See [`Path`].
    columns: Option<Columns>,
    /// See [`Path`].
    position: Option<Position>,
    /// See [`Path`].
    layout: Option<Layout>,
    /// See [`Path`].
    last_tile: Option<LastTile>,
}

/// A dimension of [`Path`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dim {
    /// [`Path::tiles`].
    Tiles,
    /// [`Path::passes`].
    Passes,
    /// [`Path::remainder`].
    Remainder,
    /// [`Path::epilogue`].
    Epilogue,
    /// [`Path::columns`].
    Columns,
    /// [`Path::position`].
    Position,
    /// [`Path::layout`].
    Layout,
    /// [`Path::last_tile`].
    LastTile,
}

/// The requirements: for each, every combination of the values of its
/// dimensions must be taken by some call (other dimensions free).
///
/// 1. Every row variant, with every depth (`kc = 1`, even, odd), in every
///    epilogue mode and M3 branch, with a full and a partial column panel, as
///    the only row panel and as the last one after a full one, packed and in
///    place.
/// 2. The same with the last live tile full and holding one row, per
///    position of the row panel ...
/// 3. ... and per A layout.
const REQUIREMENTS: [&[Dim]; 3] = [
    &[
        Dim::Tiles,
        Dim::Passes,
        Dim::Remainder,
        Dim::Epilogue,
        Dim::Columns,
        Dim::Position,
        Dim::Layout,
    ],
    &[
        Dim::Tiles,
        Dim::Passes,
        Dim::Remainder,
        Dim::Epilogue,
        Dim::Columns,
        Dim::Position,
        Dim::LastTile,
    ],
    &[
        Dim::Tiles,
        Dim::Passes,
        Dim::Remainder,
        Dim::Epilogue,
        Dim::Columns,
        Dim::Layout,
        Dim::LastTile,
    ],
];

/// Every path that the structure `desc` makes possible and that the table
/// requires: the passes are none or two and more (one pass never takes the
/// loop's back branch), the last tile full or holding one row.
fn possible_paths(desc: &KernelPaths) -> Vec<Path> {
    let mut paths = Vec::new();
    for tiles in 1..=desc.variants {
        for passes in [0, 2] {
            for remainder in 0..desc.unroll {
                // kc ≥ 1: without a pass there is a remainder.
                if passes == 0 && remainder == 0 {
                    continue;
                }
                for epilogue in EPILOGUES {
                    for columns in [Columns::Full, Columns::Partial] {
                        for position in [Position::Only, Position::Leading, Position::Last] {
                            for layout in [Layout::Packed, Layout::InPlace] {
                                for last_tile in [LastTile::Full, LastTile::OneRow] {
                                    // A leading row panel holds 64 rows.
                                    let leading_ok = position != Position::Leading
                                        || (tiles == desc.variants && last_tile == LastTile::Full);
                                    // One row is a full tile when tiles have one row.
                                    let edge_ok =
                                        last_tile != LastTile::OneRow || desc.tile_rows > 1;
                                    if leading_ok && edge_ok {
                                        paths.push(Path {
                                            tiles,
                                            passes,
                                            remainder,
                                            epilogue,
                                            columns,
                                            position,
                                            layout,
                                            last_tile,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    paths
}

/// `path` reduced to the dimensions `dims`.
fn project(path: &Path, dims: &[Dim]) -> Pattern {
    let keep = |dim: Dim| dims.contains(&dim);
    Pattern {
        tiles: keep(Dim::Tiles).then_some(path.tiles),
        passes: keep(Dim::Passes).then_some(path.passes),
        remainder: keep(Dim::Remainder).then_some(path.remainder),
        epilogue: keep(Dim::Epilogue).then_some(path.epilogue),
        columns: keep(Dim::Columns).then_some(path.columns),
        position: keep(Dim::Position).then_some(path.position),
        layout: keep(Dim::Layout).then_some(path.layout),
        last_tile: keep(Dim::LastTile).then_some(path.last_tile),
    }
}

/// Every class of paths the table requires for `desc`.
fn required(desc: &KernelPaths) -> BTreeSet<Pattern> {
    let paths = possible_paths(desc);
    REQUIREMENTS
        .iter()
        .flat_map(|dims| paths.iter().map(move |path| project(path, dims)))
        .collect()
}

/// What decides the path of a kernel call: the fields of its argument block
/// that the kernel branches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Call {
    /// Rows.
    m: usize,
    /// Columns.
    n: usize,
    /// Depth.
    kc: usize,
    /// A step between depth steps.
    a_depth: usize,
    /// A step between row panels.
    a_panel: usize,
    /// Bits of alpha.
    alpha_bits: u64,
    /// Bits of beta.
    beta_bits: u64,
}

impl Call {
    /// The call of a recorded argument block.
    fn of_args<T: SmeElement>(args: &MacroArgs<T>) -> Self {
        Self {
            m: args.m,
            n: args.n,
            kc: args.kc,
            a_depth: args.a_depth,
            a_panel: args.a_panel,
            alpha_bits: args.alpha_bits,
            beta_bits: args.beta_bits,
        }
    }

    /// The call that the self-test makes for `case` and the pair `scalars`
    /// (as `Operands::new` lays A out).
    fn of_case<T: TestFloat>(case: &Case, (alpha, beta): (f64, f64)) -> Self {
        let (a_depth, a_panel) = match case.layout {
            ALayout::Packed => (MR, MR * case.kc),
            ALayout::InPlace => (case.m + case.lda_pad, MR),
        };
        Self {
            m: case.m,
            n: case.n,
            kc: case.kc,
            a_depth,
            a_panel,
            alpha_bits: T::of(alpha).bits(),
            beta_bits: T::of(beta).bits(),
        }
    }
}

/// The epilogue class of a call of a `T` kernel.
fn epilogue_of<T: SmeElement>(call: &Call) -> EpilogueClass {
    let alpha = T::from_bits64(call.alpha_bits);
    let beta = T::from_bits64(call.beta_bits);
    match Epilogue::select(alpha, beta) {
        Epilogue::Store => EpilogueClass::M0,
        Epilogue::Add => EpilogueClass::M1,
        Epilogue::Scale => EpilogueClass::M2,
        Epilogue::ScaleCombine if alpha == T::ONE => EpilogueClass::M3AlphaOne,
        Epilogue::ScaleCombine if beta == T::ONE => EpilogueClass::M3BetaOne,
        Epilogue::ScaleCombine => EpilogueClass::M3General,
    }
}

/// Every path one call of a `T` kernel with structure `desc` takes: one per
/// (row panel, column panel) pair, in the kernel's order.
fn paths_of_call<T: SmeElement>(desc: &KernelPaths, call: &Call) -> Vec<Path> {
    let epilogue = epilogue_of::<T>(call);
    let layout = if call.a_depth == MR && call.a_panel == MR * call.kc {
        Layout::Packed
    } else {
        Layout::InPlace
    };
    let passes = (call.kc / desc.unroll).min(2);
    let remainder = call.kc % desc.unroll;
    let mut paths = Vec::new();
    for j0 in (0..call.n).step_by(desc.nr) {
        let columns = if call.n - j0 >= desc.nr {
            Columns::Full
        } else {
            Columns::Partial
        };
        for i0 in (0..call.m).step_by(MR) {
            let rows = (call.m - i0).min(MR);
            let position = if i0 + MR < call.m {
                Position::Leading
            } else if i0 == 0 {
                Position::Only
            } else {
                Position::Last
            };
            let last_tile = match rows % desc.tile_rows {
                0 => LastTile::Full,
                1 => LastTile::OneRow,
                _ => LastTile::Other,
            };
            paths.push(Path {
                tiles: rows.div_ceil(desc.tile_rows),
                passes,
                remainder,
                epilogue,
                columns,
                position,
                layout,
                last_tile,
            });
        }
    }
    paths
}

/// The calls the self-test makes for the blocks `cases`, in order.
fn listed_calls<T: TestFloat>(cases: impl Iterator<Item = Case>) -> Vec<Call> {
    cases
        .flat_map(|case| {
            SCALARS
                .iter()
                .map(move |&pair| Call::of_case::<T>(&case, pair))
        })
        .collect()
}

/// The classes of `desc` that no call of `calls` takes.
fn missing<T: SmeElement>(desc: &KernelPaths, calls: &[Call]) -> Vec<Pattern> {
    let taken: BTreeSet<Path> = calls
        .iter()
        .flat_map(|call| paths_of_call::<T>(desc, call))
        .collect();
    // Every class a taken path belongs to (a class is a projection).
    let classes: BTreeSet<Pattern> = REQUIREMENTS
        .iter()
        .flat_map(|dims| taken.iter().map(move |path| project(path, dims)))
        .collect();
    required(desc)
        .into_iter()
        .filter(|pattern| !classes.contains(pattern))
        .collect()
}

/// The table for one type: the description matches the type, and every
/// required class is taken by the self-test's own block list.
fn table_is_complete<T: TestFloat>(desc: &KernelPaths) {
    assert_eq!(desc.tile_rows, T::TILE_ROWS, "{}: tile rows", desc.name);
    assert_eq!(desc.nr, T::NR, "{}: panel width", desc.name);
    assert_eq!(desc.variants, MR / T::TILE_ROWS, "{}: variants", desc.name);
    let calls = listed_calls::<T>(self_test_blocks(T::TILE_ROWS, T::NR));
    let classes = required(desc).len();
    let gaps = missing::<T>(desc, &calls);
    println!(
        "{}: {} calls, {} required path classes, {} not taken",
        desc.name,
        calls.len(),
        classes,
        gaps.len()
    );
    assert!(
        gaps.is_empty(),
        "{}: {} of {classes} path classes not taken by the self-test, first: {:?}",
        desc.name,
        gaps.len(),
        gaps.first()
    );
}

#[test]
fn test_sme2_selftest_paths_all_taken_f32() {
    table_is_complete::<f32>(&F32_PATHS);
}

#[test]
fn test_sme2_selftest_paths_all_taken_f64() {
    table_is_complete::<f64>(&F64_PATHS);
}

/// The table on `desc` changed by `change` (a class the kernels do not
/// have) reports gaps.
fn new_class_is_a_gap<T: TestFloat>(desc: &KernelPaths) {
    let calls = listed_calls::<T>(self_test_blocks(T::TILE_ROWS, T::NR));
    assert!(missing::<T>(desc, &calls).is_empty(), "{}", desc.name);
    let more_variants = KernelPaths {
        variants: desc.variants + 1,
        ..*desc
    };
    assert!(
        !missing::<T>(&more_variants, &calls).is_empty(),
        "{}: another row variant",
        desc.name
    );
    let unrolled_by_4 = KernelPaths { unroll: 4, ..*desc };
    assert!(
        !missing::<T>(&unrolled_by_4, &calls).is_empty(),
        "{}: a depth loop unrolled by 4",
        desc.name
    );
    // No calls at all: every class is missing.
    assert_eq!(missing::<T>(desc, &[]).len(), required(desc).len());
}

#[test]
fn test_sme2_selftest_paths_new_class_is_a_gap() {
    new_class_is_a_gap::<f32>(&F32_PATHS);
    new_class_is_a_gap::<f64>(&F64_PATHS);
}

/// The sweep alone (without [`CASES`]) takes every required class, and
/// without any one of its blocks it has a gap: every block takes a class
/// that no other block of the sweep takes. (A class may also be taken by a
/// block of `CASES`, which tests other claims and may change.)
fn every_sweep_block_counts<T: TestFloat>(desc: &KernelPaths) {
    let sweep: Vec<Case> = sweep_cases(T::TILE_ROWS, T::NR).collect();
    assert_eq!(sweep.len(), desc.variants * SWEEP_BLOCKS_PER_VARIANT);
    let alone = listed_calls::<T>(sweep.iter().copied());
    assert!(
        missing::<T>(desc, &alone).is_empty(),
        "{}: the sweep alone",
        desc.name
    );
    for skip in 0..sweep.len() {
        let without = sweep
            .iter()
            .enumerate()
            .filter(|&(index, _)| index != skip)
            .map(|(_, &case)| case);
        let calls = listed_calls::<T>(without);
        assert!(
            !missing::<T>(desc, &calls).is_empty(),
            "{}: sweep block {skip} ({:?}) is not needed",
            desc.name,
            sweep.get(skip)
        );
    }
}

#[test]
fn test_sme2_selftest_paths_every_sweep_block_counts() {
    every_sweep_block_counts::<f32>(&F32_PATHS);
    every_sweep_block_counts::<f64>(&F64_PATHS);
}

thread_local! {
    /// Calls seen by [`recording_oracle`] on this thread.
    static RECORDED: RefCell<Vec<Call>> = const { RefCell::new(Vec::new()) };
}

/// Stand-in: the oracle, recording each call.
///
/// # Safety
///
/// As for `oracle_block`.
unsafe fn recording_oracle<T: SmeElement>(args: &MacroArgs<T>) {
    RECORDED.with(|recorded| recorded.borrow_mut().push(Call::of_args(args)));
    // SAFETY: forwarded from this function's contract.
    unsafe { oracle_block(args) };
}

/// `self_test_with` makes exactly the calls of the block list the table is
/// computed from, in order.
fn runs_its_list<T: TestFloat>() {
    RECORDED.with(|recorded| recorded.borrow_mut().clear());
    let kernel: KernelFn<T> = recording_oracle::<T>;
    // SAFETY: the stand-in is plain Rust.
    assert!(unsafe { self_test_with(kernel) });
    let recorded = RECORDED.with(|recorded| recorded.take());
    let listed = listed_calls::<T>(self_test_blocks(T::TILE_ROWS, T::NR));
    assert_eq!(recorded.len(), self_test_calls::<T>());
    assert!(
        recorded == listed,
        "{}: the run differs from the list",
        T::NR
    );
}

#[test]
fn test_sme2_selftest_paths_run_equals_list() {
    runs_its_list::<f32>();
    runs_its_list::<f64>();
}

#[test]
fn test_sme2_selftest_sweep_shape() {
    // The sweep is generic: for a tile height and panel width it yields
    // `variants · SWEEP_BLOCKS_PER_VARIANT` valid blocks of the documented
    // shape (one full and one partial column panel, 1 or 64 + rows rows).
    for (tile_rows, nr) in [(16, 16), (8, 8), (32, 4), (4, 2)] {
        let sweep: Vec<Case> = sweep_cases(tile_rows, nr).collect();
        assert_eq!(sweep.len(), MR / tile_rows * SWEEP_BLOCKS_PER_VARIANT);
        for case in &sweep {
            assert!(case.n > nr && case.n < 2 * nr, "{case:?}");
            assert!(SWEEP_DEPTHS.contains(&case.kc), "{case:?}");
            assert!(case.m >= 1 && case.m <= 2 * MR, "{case:?}");
            assert_eq!(case.data, Data::Plain);
            assert_eq!(case.layout == ALayout::Packed, case.lda_pad == 0);
        }
    }
    // Degenerate shapes give no blocks rather than wrong ones.
    assert_eq!(sweep_cases(0, 16).count(), 0);
    assert_eq!(sweep_cases(16, 1).count(), 0);
    assert_eq!(sweep_cases(16, 0).count(), 0);
}

/// Kernel calls and bytes of one `self_test_with` run for `T`: calls, the
/// largest run's bytes, all runs' bytes.
fn cost<T: SmeElement>(cases: impl Iterator<Item = Case>) -> (usize, usize, usize) {
    let mut calls = 0;
    let mut largest = 0;
    let mut total = 0;
    for case in cases {
        let Some(bytes) = run_bytes::<T>(&case) else {
            panic!("refused block {case:?}");
        };
        calls += SCALARS.len();
        largest = largest.max(bytes);
        total += bytes * SCALARS.len();
    }
    (calls, largest, total)
}

#[test]
fn test_sme2_selftest_cost_matches_the_docs() {
    // The figures of the module docs of `selftest.rs` (and the Wave-6 notes).
    let before_f32 = cost::<f32>(CASES.into_iter());
    let before_f64 = cost::<f64>(CASES.into_iter());
    let after_f32 = cost::<f32>(self_test_blocks(16, 16));
    let after_f64 = cost::<f64>(self_test_blocks(8, 8));
    println!("CASES only: f32 {before_f32:?}, f64 {before_f64:?}");
    println!("with the sweep: f32 {after_f32:?}, f64 {after_f64:?}");
    // (calls, bytes of the largest run, bytes of all runs together)
    assert_eq!(before_f32, (42, 72_848, 1_274_592));
    assert_eq!(before_f64, (42, 143_744, 2_469_600));
    assert_eq!(after_f32, (330, 72_848, 8_160_288));
    assert_eq!(after_f64, (618, 143_744, 19_579_872));
    assert_eq!(after_f32.0, self_test_calls::<f32>());
    assert_eq!(after_f64.0, self_test_calls::<f64>());
}
