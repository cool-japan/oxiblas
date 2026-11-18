//! Result tests of the few-rows kernels: every test here executes SME2 code
//! and starts with `hardware_or_skip`, except two plain-Rust tests (the
//! blocks of the first-use check and the independent model). The
//! cases run through the helpers of `tests_few_rows.rs` ([`check_case`]: the
//! kernel against the per-block reference and the macro-kernel path,
//! borders of signalling NaNs, exactly one streaming region and one kernel
//! entry per call on this thread, the thread's SME state afterwards).
//!
//! - The seams grid: every `m` of the type (1..48 / 1..32) × the column
//!   counts `1..NR+1`, `2NR-1..2NR+1`, `4NR+3` × the depths
//!   1, 7, 8, 9, 15, 16, 17, 63, 64, 65, in both orientations of B; the KC
//!   step (`k`, 32, 33, 64), the twelve alpha / beta pairs, the paddings and
//!   the first column of the span rotate with the case index, so every
//!   value of every axis meets every `m`.
//! - Every KC step and pair on boundary shapes (including `kc` = 1, 2, 7:
//!   a depth-block boundary after every few steps), the NEON blocked path
//!   itself, large strides, special values, an `Inf` right after a depth
//!   block (a depth loop that runs one step too far or a block that starts
//!   one step late shows), C NaN when it is not read, FPSR per mode, the
//!   declines, and the self-check.

use core::cell::RefCell;

use oxiblas_matrix::{MatMut, MatRef};

use super::args::{Epilogue, SmeElement, same_result};
use super::few_rows::{
    FewRowsArgs, few_rows_gemm, max_rows, oracle_few_rows, regions_on_this_thread, self_check,
    self_check_with,
};
use super::pack::{ColMajor, Span};
use super::test_support::{
    FPSR_DZC, FPSR_IOC, FPSR_IXC, FPSR_OFC, FPSR_UFC, Need, fpsr_read, fpsr_write,
    hardware_or_skip, sme_state,
};
use super::tests_few_rows::{
    Case, DEPTHS, Fill, FrElem, KCS, Operands, PAIRS, WIDE_GUARD, assert_c, check_case, check_ops,
    cols_of, grid_columns, neon_blocked, reference_blocks, run_kernel,
};
use super::tests_gemm_harness::Rng;
use crate::level3::trsm::Trans;

/// The seams grid for the rows `ms` and one orientation (see the module
/// docs).
fn grid<T: FrElem>(test: &str, ms: core::ops::RangeInclusive<usize>, trans: bool) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let columns = grid_columns::<T>();
    let mut index = 0usize;
    for m in ms {
        for &n in &columns {
            for &k in &DEPTHS {
                let kc = match KCS[index % KCS.len()] {
                    0 => k,
                    kc => kc,
                };
                let (alpha, beta) = PAIRS[(index / KCS.len()) % PAIRS.len()];
                let case = Case::<T>::new(m, n, k, kc, trans)
                    .scalars(alpha, beta)
                    .pads(index % 3, (index / 3) % 3, (index / 2) % 3)
                    .at_column(index % 2);
                check_case(&case, test);
                index += 1;
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_grid_f32_notrans_tile0() {
    grid::<f32>("test_sme2_few_rows_grid_f32_notrans_tile0", 1..=16, false);
}

#[test]
fn test_sme2_few_rows_grid_f32_notrans_tile1() {
    grid::<f32>("test_sme2_few_rows_grid_f32_notrans_tile1", 17..=32, false);
}

#[test]
fn test_sme2_few_rows_grid_f32_notrans_tile2() {
    grid::<f32>("test_sme2_few_rows_grid_f32_notrans_tile2", 33..=48, false);
}

#[test]
fn test_sme2_few_rows_grid_f32_trans_tile0() {
    grid::<f32>("test_sme2_few_rows_grid_f32_trans_tile0", 1..=16, true);
}

#[test]
fn test_sme2_few_rows_grid_f32_trans_tile1() {
    grid::<f32>("test_sme2_few_rows_grid_f32_trans_tile1", 17..=32, true);
}

#[test]
fn test_sme2_few_rows_grid_f32_trans_tile2() {
    grid::<f32>("test_sme2_few_rows_grid_f32_trans_tile2", 33..=48, true);
}

#[test]
fn test_sme2_few_rows_grid_f64_notrans_tiles01() {
    grid::<f64>("test_sme2_few_rows_grid_f64_notrans_tiles01", 1..=16, false);
}

#[test]
fn test_sme2_few_rows_grid_f64_notrans_tiles23() {
    grid::<f64>(
        "test_sme2_few_rows_grid_f64_notrans_tiles23",
        17..=32,
        false,
    );
}

#[test]
fn test_sme2_few_rows_grid_f64_trans_tiles01() {
    grid::<f64>("test_sme2_few_rows_grid_f64_trans_tiles01", 1..=16, true);
}

#[test]
fn test_sme2_few_rows_grid_f64_trans_tiles23() {
    grid::<f64>("test_sme2_few_rows_grid_f64_trans_tiles23", 17..=32, true);
}

/// Boundary shapes × every KC step × every pair × both orientations.
fn every_kc_and_pair<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let tile = T::TILE_ROWS;
    let nr = T::NR;
    let mut index = 0usize;
    for m in [1, tile, tile + 1, max_rows::<T>() - 1, max_rows::<T>()] {
        for (n, k) in [(nr, 65), (nr + 1, 40), (8 * nr + 1, 33), (2, 129)] {
            for kc in [k, 1, 2, 7, 32, 33, 64] {
                for trans in [false, true] {
                    for (alpha, beta) in PAIRS {
                        let case = Case::<T>::new(m, n, k, kc, trans)
                            .scalars(alpha, beta)
                            .at_column(index % 3);
                        check_case(&case, test);
                        index += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_every_kc_and_pair_f32() {
    every_kc_and_pair::<f32>("test_sme2_few_rows_every_kc_and_pair_f32");
}

#[test]
fn test_sme2_few_rows_every_kc_and_pair_f64() {
    every_kc_and_pair::<f64>("test_sme2_few_rows_every_kc_and_pair_f64");
}

/// The kernel against `gemm_planned` itself (blocked route, SME2 off), with
/// the KC step that call used.
///
/// `gemm_planned` runs the dispatched micro-kernel: NEON, except in a
/// `force-scalar` build (`--all-features`), where it is the portable kernel,
/// which this hardware test still runs against (the hardware gate ignores
/// the build's caps). The portable epilogue is `alpha · acc + beta · C` as
/// one `mul_add` of a rounded `beta · C` (`alpha · acc` alone when `beta`
/// is zero), the NEON one `C · beta + acc · alpha` with `acc · alpha`
/// rounded first. Every alpha and beta of [`PAIRS`] is zero, `±1` or
/// `±0.5`, so on this finite, normal data both products are exact and the
/// two epilogues round once, identically: the comparison holds in every
/// build. A pair whose products round would have to compare only where the
/// dispatched kernel is NEON (as `tests_kernel_f64.rs` does).
fn matches_neon<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let mut index = 0usize;
    for m in [1, 3, T::TILE_ROWS, T::TILE_ROWS + 1, max_rows::<T>()] {
        for n in [1, T::NR + 1, 4 * T::NR + 3] {
            for (k, kc) in [(7, 8), (65, 32), (130, 64), (257, 128)] {
                for trans in [false, true] {
                    let (alpha, beta) = PAIRS[index % PAIRS.len()];
                    index += 1;
                    let case = Case::<T>::new(m, n, k, kc, trans).scalars(alpha, beta);
                    let ops = Operands::new(&case);
                    let (neon, used_kc) = neon_blocked(&case, &ops);
                    let case = Case {
                        kc: used_kc,
                        ..case
                    };
                    let mut got = ops.c.clone();
                    run_kernel(&case, &ops, &mut got);
                    assert_c(&case, &ops, &got, &neon, test);
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_matches_neon_f32() {
    matches_neon::<f32>("test_sme2_few_rows_matches_neon_f32");
}

#[test]
fn test_sme2_few_rows_matches_neon_f64() {
    matches_neon::<f64>("test_sme2_few_rows_matches_neon_f64");
}

/// Leading dimensions far above the rows, the span deep inside B.
fn large_strides<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let mut index = 0usize;
    for m in [1, T::TILE_ROWS + 3, max_rows::<T>()] {
        for (n, k, kc) in [(2 * T::NR + 1, 70, 33), (3, 17, 64), (T::NR, 9, 4)] {
            for trans in [false, true] {
                let (alpha, beta) = PAIRS[index % PAIRS.len()];
                index += 1;
                let case = Case::<T>::new(m, n, k, kc, trans)
                    .scalars(alpha, beta)
                    .pads(1000 + index, 777, 513)
                    .at_column(37);
                check_case(&case, test);
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_large_strides_f32() {
    large_strides::<f32>("test_sme2_few_rows_large_strides_f32");
}

#[test]
fn test_sme2_few_rows_large_strides_f64() {
    large_strides::<f64>("test_sme2_few_rows_large_strides_f64");
}

/// Infinities and NaNs, subnormal data, signed zeros, and non-finite or
/// special scalars.
fn special_values<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let scalars = [
        (1.0, 0.0),
        (1.0, 1.0),
        (-1.0, -0.0),
        (0.5, -0.5),
        (2.0, 1.0),
        (-0.0, 1.0),
        (f64::INFINITY, 0.5),
        (1.0, f64::NAN),
        (f64::NAN, 0.0),
        (3.0, f64::NEG_INFINITY),
    ];
    let mut index = 0usize;
    for fill in [
        Fill::NonFinite,
        Fill::Subnormal,
        Fill::SignedZeros,
        Fill::Random,
    ] {
        for m in [1, T::TILE_ROWS + 1, max_rows::<T>()] {
            for (n, k, kc) in [(T::NR + 3, 40, 16), (1, 9, 9), (2 * T::NR, 33, 32)] {
                for trans in [false, true] {
                    let (alpha, beta) = scalars[index % scalars.len()];
                    index += 1;
                    if alpha == 0.0 {
                        continue;
                    }
                    let case = Case::<T>::new(m, n, k, kc, trans)
                        .scalars(alpha, beta)
                        .filled(fill);
                    check_case(&case, test);
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_special_values_f32() {
    special_values::<f32>("test_sme2_few_rows_special_values_f32");
}

#[test]
fn test_sme2_few_rows_special_values_f64() {
    special_values::<f64>("test_sme2_few_rows_special_values_f64");
}

/// `+Inf` in every row of A at the first depth step of the second block
/// (and at the last step of the first block, with `-Inf` right before it in
/// a separate case): a block that runs one step too far, or a second block
/// that starts one step late, changes finite results into infinities or
/// NaNs.
fn inf_at_block_edges<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let inf = T::of(f64::INFINITY);
    for m in [1, T::TILE_ROWS, max_rows::<T>()] {
        for (k, kc) in [(40, 16), (35, 17), (2 * T::TILE_ROWS + 1, T::TILE_ROWS)] {
            for trans in [false, true] {
                for edge in [kc, kc - 1] {
                    let case = Case::<T>::new(m, T::NR + 2, k, kc, trans).scalars(1.0, 0.0);
                    let mut ops = Operands::new(&case);
                    for i in 0..m {
                        ops.a[ops.a_off + i + edge * ops.lda] = inf;
                    }
                    check_ops(&case, &ops, test);
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_inf_at_block_edges_f32() {
    inf_at_block_edges::<f32>("test_sme2_few_rows_inf_at_block_edges_f32");
}

#[test]
fn test_sme2_few_rows_inf_at_block_edges_f64() {
    inf_at_block_edges::<f64>("test_sme2_few_rows_inf_at_block_edges_f64");
}

/// C full of signalling NaNs with beta = 0 / -0: the first block must not
/// read it, the later blocks read only what the first one wrote.
fn nan_c_not_read<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    for m in [1, T::TILE_ROWS - 1, max_rows::<T>()] {
        for (n, k, kc) in [(T::NR + 1, 20, 20), (3, 50, 16), (2 * T::NR + 1, 17, 8)] {
            for trans in [false, true] {
                for (alpha, beta) in [(1.0, 0.0), (1.0, -0.0), (-3.0, 0.0), (0.25, -0.0)] {
                    let case = Case::<T>::new(m, n, k, kc, trans).scalars(alpha, beta);
                    assert!(case.c_nan && !case.reads_c());
                    check_case(&case, test);
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_nan_c_not_read_f32() {
    nan_c_not_read::<f32>("test_sme2_few_rows_nan_c_not_read_f32");
}

#[test]
fn test_sme2_few_rows_nan_c_not_read_f64() {
    nan_c_not_read::<f64>("test_sme2_few_rows_nan_c_not_read_f64");
}

/// The flags a call leaves: the caller's, plus those of the epilogue
/// arithmetic of its blocks on live lanes only.
fn fpsr_per_mode<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let earlier = [0, FPSR_DZC, FPSR_IOC | FPSR_UFC];
    // (alpha, beta, data, one block?, exact: no flag may be added)
    let cases = [
        (1.0, 0.0, Fill::SignedZeros, true, true),
        (1.0, 0.0, Fill::SignedZeros, false, true),
        (0.5, 0.0, Fill::SignedZeros, true, true),
        (0.5, -0.5, Fill::SignedZeros, false, true),
        (1.0, 1.0, Fill::SignedZeros, false, true),
        (1.0 / 3.0, 0.0, Fill::SignedZeros, true, false),
        (1.0, 1.0, Fill::Random, true, false),
        (0.5, -0.75, Fill::Random, false, false),
    ];
    let saved = fpsr_read();
    for (alpha, beta, fill, one_block, exact) in cases {
        for trans in [false, true] {
            for flags in earlier {
                let (k, kc) = if one_block { (21, 21) } else { (21, 8) };
                let case = Case::<T>::new(max_rows::<T>() - 1, T::NR + 3, k, kc, trans)
                    .scalars(alpha, beta)
                    .filled(fill);
                let ops = Operands::new(&case);
                let mut want = ops.c.clone();
                reference_blocks(&case, &ops, &mut want);
                let mut got = ops.c.clone();
                fpsr_write(flags);
                run_kernel(&case, &ops, &mut got);
                let after = fpsr_read();
                fpsr_write(saved);
                assert_c(&case, &ops, &got, &want, test);
                let label = format!("{test}: {case:?}, flags before {flags:#x}, after {after:#x}");
                assert_eq!(after & flags, flags, "{label}: earlier flags lost");
                let added = after & !flags;
                if exact {
                    assert_eq!(added, 0, "{label}: an exact call raised flags");
                } else {
                    assert_ne!(
                        added & FPSR_IXC,
                        0,
                        "{label}: no IXC from inexact epilogues"
                    );
                    assert_eq!(
                        added & (FPSR_OFC | FPSR_DZC | FPSR_IOC),
                        0,
                        "{label}: unexpected flags"
                    );
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_fpsr_per_mode_f32() {
    fpsr_per_mode::<f32>("test_sme2_few_rows_fpsr_per_mode_f32");
}

#[test]
fn test_sme2_few_rows_fpsr_per_mode_f64() {
    fpsr_per_mode::<f64>("test_sme2_few_rows_fpsr_per_mode_f64");
}

/// Declines through `few_rows_gemm` on hardware: `false`, C bitwise
/// untouched, no streaming region.
fn declines<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let case = Case::<T>::new(4, 3, 5, 2, false);
    let ops = Operands::new(&case);
    let b = ops.b_op(&case);
    let attempts: [(Span, usize, f64); 5] = [
        (Span { start: 1, len: 0 }, 2, 1.0),
        (Span { start: 3, len: 3 }, 2, 1.0),
        (cols_of(&case), 0, 1.0),
        (cols_of(&case), 2, 0.0),
        (cols_of(&case), 2, -0.0),
    ];
    for (cols, kc, alpha) in attempts {
        let mut c = ops.c.clone();
        let regions = regions_on_this_thread();
        // SAFETY: the thread is ready (`hardware_or_skip`); the arguments
        // are declined before any kernel call.
        let done = unsafe {
            few_rows_gemm(
                ops.a_view(&case),
                &b,
                cols,
                T::of(alpha),
                case.beta,
                kc,
                MatViews::c(&ops, &case, &mut c, cols.len.max(1)),
            )
        };
        assert!(!done, "{test}: accepted {cols:?}, kc {kc}, alpha {alpha}");
        assert_eq!(regions_on_this_thread(), regions, "{test}: a region ran");
        assert!(
            c.iter().zip(&ops.c).all(|(x, y)| x.bits() == y.bits()),
            "{test}: C changed"
        );
    }
    // The control: the same operands are accepted.
    let mut c = ops.c.clone();
    run_kernel(&case, &ops, &mut c);
}

/// C views for the declines (the span length may differ from the case's).
struct MatViews;

impl MatViews {
    fn c<'c, T: FrElem>(
        ops: &Operands<T>,
        case: &Case<T>,
        c: &'c mut [T],
        n: usize,
    ) -> oxiblas_matrix::MatMut<'c, T> {
        oxiblas_matrix::MatMut::from_strided(&mut c[ops.c_off..], case.m, n, ops.ldc)
            .unwrap_or_else(|| panic!("C view {n}"))
    }
}

#[test]
fn test_sme2_few_rows_declines_f32() {
    declines::<f32>("test_sme2_few_rows_declines_f32");
}

#[test]
fn test_sme2_few_rows_declines_f64() {
    declines::<f64>("test_sme2_few_rows_declines_f64");
}

/// Many column panels and depth blocks are still one region (`run_kernel`
/// asserts the per-thread count moves by exactly one per call).
#[test]
fn test_sme2_few_rows_one_region_per_call() {
    let test = "test_sme2_few_rows_one_region_per_call";
    if !hardware_or_skip(test, Need::F64) {
        return;
    }
    for trans in [false, true] {
        let case = Case::<f32>::new(48, 16 * 9 + 5, 200, 32, trans);
        let before = regions_on_this_thread();
        check_case(&case, test);
        assert_eq!(regions_on_this_thread(), before + 1);
        let case = Case::<f64>::new(32, 8 * 9 + 3, 150, 33, trans);
        let before = regions_on_this_thread();
        check_case(&case, test);
        assert_eq!(regions_on_this_thread(), before + 1);
    }
    assert_eq!(sme_state(), Some((0, 0)));
}

#[test]
fn test_sme2_few_rows_self_check() {
    if !hardware_or_skip("test_sme2_few_rows_self_check", Need::F64) {
        return;
    }
    // SAFETY: the thread is ready and SME2 with FEAT_SME_F64F64 is present
    // (`hardware_or_skip` with `Need::F64`).
    unsafe {
        assert!(self_check(false), "f32 only");
        assert!(self_check(true), "f32 and f64");
    }
    assert_eq!(sme_state(), Some((0, 0)));
}

thread_local! {
    /// `[m, n, k, b_trans]` of every block that [`recording_model`] ran on
    /// this thread.
    static CHECKED_BLOCKS: RefCell<Vec<[usize; 4]>> = const { RefCell::new(Vec::new()) };
}

/// A stand-in for the kernel that records the shape of each block of the
/// first-use check and computes it with the scalar model, so that the check
/// passes.
///
/// # Safety
///
/// As `oracle_few_rows`.
unsafe fn recording_model<T: SmeElement>(args: &FewRowsArgs<T>) {
    let shape = [args.m, args.n, args.k, usize::from(args.b_trans != 0)];
    CHECKED_BLOCKS.with(|blocks| blocks.borrow_mut().push(shape));
    // SAFETY: forwarded.
    unsafe { oracle_few_rows(args) }
}

/// Plain Rust: the blocks of the first-use check reach both row bounds of
/// every depth-loop variant (`L` live tiles: `L` whole tiles and the lowest
/// count `(L - 1) · TILE_ROWS + 1`, at which a variant boundary that is off
/// by one row fails), every variant in both orientations, and the sizes
/// that `self_check` documents as its cost.
#[test]
fn test_sme2_few_rows_self_check_reaches_every_variant_boundary() {
    fn blocks_of<T: FrElem>() -> Vec<[usize; 4]> {
        CHECKED_BLOCKS.with(|blocks| blocks.borrow_mut().clear());
        // SAFETY: the stand-in is the scalar model, which every block of the
        // check is a valid argument block for.
        assert!(unsafe { self_check_with::<T>(recording_model::<T>) });
        CHECKED_BLOCKS.with(|blocks| blocks.borrow().clone())
    }
    fn check<T: FrElem>(name: &str, largest: [usize; 3]) {
        let blocks = blocks_of::<T>();
        let tile = T::TILE_ROWS;
        assert_eq!(blocks.len(), 16, "{name}: blocks of the check");
        for live in 1..=max_rows::<T>() / tile {
            for rows in [(live - 1) * tile + 1, live * tile] {
                assert!(
                    blocks.iter().any(|block| block[0] == rows),
                    "{name}: no block of {rows} rows ({live} live tiles)"
                );
            }
            for trans in [0, 1] {
                assert!(
                    blocks
                        .iter()
                        .any(|block| block[0].div_ceil(tile) == live && block[3] == trans),
                    "{name}: {live} live tiles never with b_trans {trans}"
                );
            }
        }
        for (axis, want) in largest.into_iter().enumerate() {
            let most = blocks.iter().map(|block| block[axis]).max();
            assert_eq!(most, Some(want), "{name}: largest of axis {axis}");
        }
    }
    check::<f32>("f32", [48, 35, 49]);
    check::<f64>("f64", [32, 19, 25]);
}

// ---------------------------------------------------------------------------
// Tester's additions: block boundaries, adjacent operands, degenerate views,
// stale transposer slices, NaN payloads, flags, the largest shapes
// ---------------------------------------------------------------------------

/// The value of the seams file for one element, computed directly from
/// element accessors, independently of `args::reference`, the packers and
/// the scalar model of `few_rows.rs`: per depth block `acc = +0`, the depth
/// steps in ascending order as fused multiply-adds, then the block's
/// epilogue (`select(alpha, beta)` for the first block, `select(alpha, 1)`
/// for the later ones).
fn seams_value<T: FrElem>(
    a: &dyn Fn(usize, usize) -> T,
    b: &dyn Fn(usize, usize) -> T,
    old: T,
    (i, j): (usize, usize),
    (k, kc): (usize, usize),
    (alpha, beta): (T, T),
) -> T {
    let mut c = old;
    let kc = kc.clamp(1, k);
    for (block, p0) in (0..k).step_by(kc).enumerate() {
        let d = kc.min(k - p0);
        let mut acc = T::ZERO;
        for p in p0..p0 + d {
            acc = a(i, p).fused(b(p, j), acc);
        }
        let beta_b = if block == 0 { beta } else { T::ONE };
        c = match Epilogue::select(alpha, beta_b) {
            Epilogue::Store => acc,
            Epilogue::Add => acc + c,
            Epilogue::Scale => acc * alpha,
            Epilogue::ScaleCombine => c.fused(beta_b, acc * alpha),
        };
    }
    c
}

/// Every KC step one below, at and one above the transposer's chunk, and
/// 32 / 33, with depths one below, at and one above two and three whole
/// blocks (the grid's KC rotation meets each depth with two KC steps only,
/// e.g. never `k = 64` with `kc = 32`).
fn block_boundaries<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let chunk = T::TILE_ROWS;
    let nr = T::NR;
    let mut index = 0usize;
    for kc in [chunk - 1, chunk, chunk + 1, 32, 33] {
        for blocks in [2, 3] {
            for k in [blocks * kc - 1, blocks * kc, blocks * kc + 1] {
                for (m, n) in [
                    (1, nr - 1),
                    (T::TILE_ROWS + 1, 2 * nr + 1),
                    (max_rows::<T>(), nr),
                ] {
                    for trans in [false, true] {
                        let (alpha, beta) = PAIRS[index % PAIRS.len()];
                        let case = Case::<T>::new(m, n, k, kc, trans)
                            .scalars(alpha, beta)
                            .pads(index % 2, (index / 2) % 2, index % 3)
                            .at_column(index % 3);
                        check_case(&case, test);
                        index += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_block_boundaries_f32() {
    block_boundaries::<f32>("test_sme2_few_rows_block_boundaries_f32");
}

#[test]
fn test_sme2_few_rows_block_boundaries_f64() {
    block_boundaries::<f64>("test_sme2_few_rows_block_boundaries_f64");
}

/// A, B and C cut out of ONE allocation, tightly packed (`lda = m`, the
/// stored B is exactly op(B), `ldc = m`), with 0 or 1 signalling NaNs
/// between them and `WIDE_GUARD` of them before and after the whole: a
/// write one element before or after C lands in an operand, and every
/// element outside the block of C (A, B, the gaps, the ends) must keep its
/// bits.
fn one_allocation<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let (tile, nr, max) = (T::TILE_ROWS, T::NR, max_rows::<T>());
    let shapes = [
        (max, 2 * nr + 1, 19, 8),
        (1, nr + 1, 9, 9),
        (tile + 1, 3, 33, 16),
        (max - 1, nr, 2 * tile + 1, tile),
    ];
    let mut index = 0usize;
    for (m, n, k, kc) in shapes {
        for trans in [false, true] {
            for gap in [0usize, 1] {
                for c_first in [false, true] {
                    let (alpha, beta) = PAIRS[index % PAIRS.len()];
                    index += 1;
                    adjacent_case::<T>(test, (m, n, k, kc), trans, gap, c_first, (alpha, beta));
                }
            }
        }
    }
}

/// One case of [`one_allocation`]: the layout is `[s][X][gap][B][gap][Y][s]`
/// with `X, Y = A, C` (or `C, A` when `c_first`), `s` `WIDE_GUARD`
/// signalling NaNs.
fn adjacent_case<T: FrElem>(
    test: &str,
    (m, n, k, kc): (usize, usize, usize, usize),
    trans: bool,
    gap: usize,
    c_first: bool,
    (alpha, beta): (f64, f64),
) {
    let (alpha, beta) = (T::of(alpha), T::of(beta));
    let (b_rows, b_cols) = if trans { (n, k) } else { (k, n) };
    let (a_len, b_len, c_len) = (m * k, b_rows * b_cols, m * n);
    let (first_len, second_len) = if c_first {
        (c_len, a_len)
    } else {
        (a_len, c_len)
    };
    let first_at = WIDE_GUARD;
    let b_at = first_at + first_len + gap;
    let second_at = b_at + b_len + gap;
    let total = second_at + second_len + WIDE_GUARD;
    let (a_at, c_at) = if c_first {
        (second_at, first_at)
    } else {
        (first_at, second_at)
    };
    let mut rng = Rng::new((m * 7919 + n * 104_729 + k * 31 + kc) as u64 + u64::from(trans));
    let mut buffer = vec![T::SENTINEL; total];
    let reads_c = Epilogue::select(alpha, beta).reads_c();
    for (at, slot) in buffer.iter_mut().enumerate() {
        let in_a = (a_at..a_at + a_len).contains(&at);
        let in_b = (b_at..b_at + b_len).contains(&at);
        let in_c = (c_at..c_at + c_len).contains(&at);
        if in_a || in_b || (in_c && reads_c) {
            *slot = rng.value::<T>();
        } else if in_c {
            *slot = T::CHECK_UNREAD;
        }
    }
    let before = buffer.clone();
    let a_of = |i: usize, p: usize| before[a_at + i + p * m];
    let b_of = |p: usize, j: usize| {
        if trans {
            before[b_at + j + p * n]
        } else {
            before[b_at + p + j * k]
        }
    };
    let mut want = before.clone();
    for j in 0..n {
        for i in 0..m {
            let old = before[c_at + i + j * m];
            want[c_at + i + j * m] = seams_value(&a_of, &b_of, old, (i, j), (k, kc), (alpha, beta));
        }
    }
    let label = format!(
        "{test}: m {m} n {n} k {k} kc {kc} trans {trans} gap {gap} c_first {c_first} \
         alpha {alpha:?} beta {beta:?}"
    );
    {
        let (head, rest) = buffer.split_at_mut(c_at);
        let (c_slice, tail) = rest.split_at_mut(c_len);
        let (a_slice, b_slice): (&[T], &[T]) = if c_first {
            (&tail[a_at - c_at - c_len..], &tail[b_at - c_at - c_len..])
        } else {
            (&head[a_at..], &head[b_at..])
        };
        let a = MatRef::from_strided(a_slice, m, k, m, 1).unwrap_or_else(|| panic!("A {label}"));
        let stored = MatRef::from_strided(b_slice, b_rows, b_cols, b_rows, 1)
            .unwrap_or_else(|| panic!("B {label}"));
        let b = ColMajor::normalize(stored, if trans { Trans::Trans } else { Trans::NoTrans })
            .unwrap_or_else(|| panic!("op(B) {label}"));
        let c = MatMut::from_strided(c_slice, m, n, m).unwrap_or_else(|| panic!("C {label}"));
        let regions = regions_on_this_thread();
        // SAFETY: the thread is ready (`hardware_or_skip`); A, B and C are
        // disjoint parts of one buffer (`split_at_mut`), so C overlaps
        // neither.
        let done = unsafe { few_rows_gemm(a, &b, Span { start: 0, len: n }, alpha, beta, kc, c) };
        assert!(done, "declined: {label}");
        assert_eq!(regions_on_this_thread(), regions + 1, "regions: {label}");
    }
    assert_eq!(sme_state(), Some((0, 0)), "SVCR / TPIDR2: {label}");
    for (at, (got, want)) in buffer.iter().zip(&want).enumerate() {
        let in_c = (c_at..c_at + c_len).contains(&at);
        if in_c {
            assert!(
                same_result(*got, *want),
                "{label}: C element {} = {got:?}, want {want:?}",
                at - c_at
            );
        } else {
            assert_eq!(
                got.bits(),
                want.bits(),
                "{label}: element {at} outside C changed"
            );
        }
    }
}

#[test]
fn test_sme2_few_rows_one_allocation_f32() {
    one_allocation::<f32>("test_sme2_few_rows_one_allocation_f32");
}

#[test]
fn test_sme2_few_rows_one_allocation_f64() {
    one_allocation::<f64>("test_sme2_few_rows_one_allocation_f64");
}

/// Stride of a dimension of length 1 that a correct implementation never
/// multiplies by a nonzero index (far beyond any buffer).
const HUGE: usize = 1 << 40;

/// Views whose unused strides are arbitrary, as the driver may pass them
/// (a row of A, a column of B or C, a depth of 1): `(rows, cols,
/// row_stride, col_stride)` of A and of the stored B, the orientation, and
/// `ldc` (any value when `n == 1`).
#[derive(Debug, Clone, Copy)]
struct OddViews {
    /// A: rows (`m`), columns (`k`), row stride, column stride.
    a: (usize, usize, usize, usize),
    /// The stored B: rows, columns, row stride, column stride.
    b: (usize, usize, usize, usize),
    /// op(B) = B^T.
    trans: bool,
    /// The row stride of the C view.
    ldc: usize,
}

/// The elements a view `(rows, cols, row_stride, col_stride)` addresses.
fn view_offsets((rows, cols, row_stride, col_stride): (usize, usize, usize, usize)) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            offsets.push(r * col_stride + c * row_stride);
        }
    }
    offsets
}

/// Runs `views` through `few_rows_gemm` with every KC step of `kcs` and the
/// four modes, against [`seams_value`] (only the addressed elements of the
/// buffers hold data; everything else of C must keep its bits).
fn odd_views_case<T: FrElem>(test: &str, views: OddViews, kcs: &[usize], seed: u64) {
    let (m, k) = (views.a.0, views.a.1);
    let n = if views.trans { views.b.0 } else { views.b.1 };
    let a_offsets = view_offsets(views.a);
    let b_offsets = view_offsets(views.b);
    let mut rng = Rng::new(seed);
    let a_len = a_offsets.iter().max().map_or(0, |at| at + 2);
    let b_len = b_offsets.iter().max().map_or(0, |at| at + 2);
    let mut a = vec![T::SENTINEL; a_len];
    let mut b = vec![T::SENTINEL; b_len];
    for &at in &a_offsets {
        a[at] = rng.value::<T>();
    }
    for &at in &b_offsets {
        b[at] = rng.value::<T>();
    }
    let a_at = |i: usize, p: usize| a[i * views.a.3 + p * views.a.2];
    let b_at = |p: usize, j: usize| {
        let (r, c) = if views.trans { (j, p) } else { (p, j) };
        b[r * views.b.3 + c * views.b.2]
    };
    let c_index = |i: usize, j: usize| 1 + i + j * views.ldc;
    let c_len = c_index(m - 1, n - 1) + 1 + WIDE_GUARD;
    for (case, (alpha, beta)) in [(1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (-1.0, -0.5)]
        .into_iter()
        .enumerate()
    {
        let kc = kcs[case % kcs.len()];
        let (alpha, beta) = (T::of(alpha), T::of(beta));
        let reads_c = Epilogue::select(alpha, beta).reads_c();
        let mut c = vec![T::SENTINEL; c_len];
        for j in 0..n {
            for i in 0..m {
                c[c_index(i, j)] = if reads_c {
                    rng.value::<T>()
                } else {
                    T::CHECK_UNREAD
                };
            }
        }
        let mut want = c.clone();
        for j in 0..n {
            for i in 0..m {
                let old = c[c_index(i, j)];
                want[c_index(i, j)] =
                    seams_value(&a_at, &b_at, old, (i, j), (k, kc), (alpha, beta));
            }
        }
        let label = format!("{test}: {views:?}, kc {kc}, alpha {alpha:?}, beta {beta:?}");
        let (ar, ac, ars, acs) = views.a;
        let a_view =
            MatRef::from_strided(&a, ar, ac, ars, acs).unwrap_or_else(|| panic!("A {label}"));
        let (br, bc, brs, bcs) = views.b;
        let stored =
            MatRef::from_strided(&b, br, bc, brs, bcs).unwrap_or_else(|| panic!("B {label}"));
        let trans = if views.trans {
            Trans::Trans
        } else {
            Trans::NoTrans
        };
        let b_op = ColMajor::normalize(stored, trans).unwrap_or_else(|| panic!("op(B) {label}"));
        let mut got = c.clone();
        // SAFETY: `got` holds every element `1 + i + j * ldc` of the view
        // (`c_len` covers the far corner); with `n == 1` only column 0 is
        // addressed, whatever the stride.
        let c_view = unsafe { MatMut::new(got.as_mut_ptr().wrapping_add(1), m, n, views.ldc) };
        let regions = regions_on_this_thread();
        // SAFETY: the thread is ready (`hardware_or_skip` of the caller); C
        // is its own buffer.
        let done = unsafe {
            few_rows_gemm(
                a_view,
                &b_op,
                Span { start: 0, len: n },
                alpha,
                beta,
                kc,
                c_view,
            )
        };
        assert!(done, "declined: {label}");
        assert_eq!(regions_on_this_thread(), regions + 1, "regions: {label}");
        assert_eq!(sme_state(), Some((0, 0)), "SVCR / TPIDR2: {label}");
        let in_block = |at: usize| {
            at >= 1 && {
                let offset = at - 1;
                if n == 1 {
                    offset < m
                } else {
                    offset % views.ldc < m && offset / views.ldc < n
                }
            }
        };
        for (at, (g, w)) in got.iter().zip(&want).enumerate() {
            if in_block(at) {
                assert!(
                    same_result(*g, *w),
                    "{label}: element {at} = {g:?}, want {w:?}"
                );
            } else {
                assert_eq!(
                    g.bits(),
                    c[at].bits(),
                    "{label}: element {at} outside C changed"
                );
            }
        }
    }
}

/// A single row of A with a non-unit row step, depths of 1 with arbitrary
/// depth strides, single columns of B and C with arbitrary strides (C with
/// stride 0), and 1 x 1 x 1: the strides `few_rows_args` normalises.
fn degenerate_views<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let (tile, nr, max) = (T::TILE_ROWS, T::NR, max_rows::<T>());
    let n = nr + 3;
    let cases = [
        // m = 1: a row of a larger matrix (row step 3), C a row (ldc 7).
        OddViews {
            a: (1, 37, 5, 3),
            b: (37, n, 40, 1),
            trans: false,
            ldc: 7,
        },
        OddViews {
            a: (1, 37, 5, 3),
            b: (n, 37, n + 5, 1),
            trans: true,
            ldc: 7,
        },
        // k = 1: the depth stride of A addresses nothing; a 1 x n stored B
        // with a column step of 9 (NoTrans) or an n x 1 one with a huge
        // depth stride (Trans).
        OddViews {
            a: (tile + 2, 1, HUGE, 1),
            b: (1, 2 * nr + 1, 2, 9),
            trans: false,
            ldc: tile + 3,
        },
        OddViews {
            a: (max, 1, HUGE, 1),
            b: (2 * nr + 1, 1, HUGE, 1),
            trans: true,
            ldc: max,
        },
        // n = 1: C with stride 0; B a k x 1 column with a huge stride
        // (NoTrans) or a 1 x k row with a column step of HUGE (Trans).
        OddViews {
            a: (max - 1, 2 * tile + 3, max, 1),
            b: (2 * tile + 3, 1, HUGE, 1),
            trans: false,
            ldc: 0,
        },
        OddViews {
            a: (tile, 2 * tile + 3, tile + 1, 1),
            b: (1, 2 * tile + 3, 3, HUGE),
            trans: true,
            ldc: 0,
        },
        // 1 x 1 x 1 with every stride huge.
        OddViews {
            a: (1, 1, HUGE, HUGE),
            b: (1, 1, HUGE, HUGE),
            trans: false,
            ldc: 0,
        },
        OddViews {
            a: (1, 1, HUGE, HUGE),
            b: (1, 1, HUGE, HUGE),
            trans: true,
            ldc: HUGE,
        },
    ];
    for (index, views) in cases.into_iter().enumerate() {
        odd_views_case::<T>(test, views, &[tile, 1, tile + 1, 64], index as u64 + 11);
    }
}

#[test]
fn test_sme2_few_rows_degenerate_views_f32() {
    degenerate_views::<f32>("test_sme2_few_rows_degenerate_views_f32");
}

#[test]
fn test_sme2_few_rows_degenerate_views_f64() {
    degenerate_views::<f64>("test_sme2_few_rows_degenerate_views_f64");
}

/// The transposer slices of the columns that are dead in the last, partial
/// column panel keep the B values of the previous panel (and, for the depth
/// chunks, of the previous chunk). Those columns of panel 0 hold +Inf and
/// NaN at every depth step, so a stale slice that reached a stored C slice
/// would show as Inf / NaN in the last panel, whose data is finite.
fn stale_transposer<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let (tile, nr, max) = (T::TILE_ROWS, T::NR, max_rows::<T>());
    let mut index = 0usize;
    for live in [1, nr / 2, nr - 1] {
        for (k, kc) in [(tile + 3, tile + 3), (2 * tile + 5, tile + 1), (3, 3)] {
            for m in [1, tile + 1, max] {
                for trans in [false, true] {
                    let (alpha, beta) =
                        [(1.0, 1.0), (0.5, -0.5), (1.0, 0.0), (-2.0, 0.0)][index % 4];
                    index += 1;
                    let case = Case::<T>::new(m, nr + live, k, kc, trans).scalars(alpha, beta);
                    let mut ops = Operands::new(&case);
                    for j in live..nr {
                        for p in 0..k {
                            let at = if trans {
                                (case.col0 + j) + p * ops.ldb
                            } else {
                                p + (case.col0 + j) * ops.ldb
                            };
                            ops.b[ops.b_off + at] = if p % 2 == 0 {
                                T::of(f64::INFINITY)
                            } else {
                                T::of(f64::NAN)
                            };
                        }
                    }
                    // The operands hold non-finite data now: `check_ops`
                    // must not expect the results free of NaN.
                    let case = Case {
                        fill: Fill::NonFinite,
                        ..case
                    };
                    check_ops(&case, &ops, test);
                    let mut got = ops.c.clone();
                    run_kernel(&case, &ops, &mut got);
                    for j in nr..nr + live {
                        for i in 0..m {
                            let value = got[ops.c_off + i + j * ops.ldc];
                            assert!(
                                value.to_f64().is_finite(),
                                "{test}: stale slice reached C({i}, {j}) = {value:?}; {case:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_stale_transposer_f32() {
    stale_transposer::<f32>("test_sme2_few_rows_stale_transposer_f32");
}

#[test]
fn test_sme2_few_rows_stale_transposer_f64() {
    stale_transposer::<f64>("test_sme2_few_rows_stale_transposer_f64");
}

/// Quiet NaNs with distinct payloads along one row of A, a signalling NaN
/// as live B data, infinities of both signs meeting zeros of both signs, in
/// every mode: the NaN positions and every other bit must be those of the
/// reference.
fn nan_payloads<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let (tile, nr, max) = (T::TILE_ROWS, T::NR, max_rows::<T>());
    let quiet = T::of(f64::NAN).bits();
    let mut index = 0usize;
    for m in [2, tile + 1, max] {
        for (n, k, kc) in [(nr + 1, tile + 5, tile + 5), (3, 2 * tile + 1, 7)] {
            for trans in [false, true] {
                let (alpha, beta) = PAIRS[index % PAIRS.len()];
                index += 1;
                let case = Case::<T>::new(m, n, k, kc, trans).scalars(alpha, beta);
                let mut ops = Operands::new(&case);
                let (a_off, lda, b_off, ldb, col0) =
                    (ops.a_off, ops.lda, ops.b_off, ops.ldb, case.col0);
                let a_at = |i: usize, p: usize| a_off + i + p * lda;
                for p in 0..k {
                    let payload = T::from_bits64(quiet | (p as u64 + 1));
                    let at = a_at(0, p);
                    ops.a[at] = payload;
                }
                let mid = m / 2;
                for p in 0..k {
                    let at = a_at(mid, p);
                    ops.a[at] = if p % 3 == 0 { T::of(-0.0) } else { T::of(0.0) };
                }
                let last = a_at(m - 1, k / 2);
                ops.a[last] = T::of(f64::NEG_INFINITY);
                let b_at = |p: usize, j: usize| {
                    b_off
                        + if trans {
                            (col0 + j) + p * ldb
                        } else {
                            p + (col0 + j) * ldb
                        }
                };
                let signalling = T::from_bits64(T::SENTINEL.bits() | 0x5);
                let at = b_at(k - 1, n - 1);
                ops.b[at] = signalling;
                let at = b_at(k / 2, 0);
                ops.b[at] = T::of(f64::INFINITY);
                // Non-finite data now: `check_ops` must not expect the
                // results free of NaN.
                let case = Case {
                    fill: Fill::NonFinite,
                    ..case
                };
                check_ops(&case, &ops, test);
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_nan_payloads_f32() {
    nan_payloads::<f32>("test_sme2_few_rows_nan_payloads_f32");
}

#[test]
fn test_sme2_few_rows_nan_payloads_f64() {
    nan_payloads::<f64>("test_sme2_few_rows_nan_payloads_f64");
}

/// The flags of the depth loop and of the epilogues, as the macro-kernel
/// FPSR tests establish them: FMOPA raises nothing even for inexact or
/// overflowing sums (M0 adds no flag), `inf + -inf` in M1 raises IOC, an
/// overflowing `acc * alpha` in M2 raises OFC and IXC.
fn fpsr_depth_and_overflow<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let big = if T::NR == 16 { 1.0e30 } else { 1.0e300 };
    // (label, alpha, beta, A, B, C (when read), flags added exactly)
    let cases: [(&str, f64, f64, f64, f64, f64, u64); 4] = [
        ("M0, inexact sums", 1.0, 0.0, 1.0 / 3.0, 1.0 / 7.0, 0.0, 0),
        ("M0, overflowing sums", 1.0, 0.0, big, big, 0.0, 0),
        (
            "M1, inf + -inf",
            1.0,
            1.0,
            f64::INFINITY,
            2.0,
            f64::NEG_INFINITY,
            FPSR_IOC,
        ),
        ("M2, overflow", big, 0.0, 3.0, big, 0.0, FPSR_OFC | FPSR_IXC),
    ];
    let saved = fpsr_read();
    for (label, alpha, beta, a_value, b_value, c_value, added) in cases {
        for trans in [false, true] {
            for m in [1, max_rows::<T>() - 1] {
                let case = Case::<T>::new(m, T::NR + 3, 21, 21, trans).scalars(alpha, beta);
                let mut ops = Operands::new(&case);
                for p in 0..case.k {
                    for i in 0..m {
                        ops.a[ops.a_off + i + p * ops.lda] = T::of(a_value);
                    }
                    for j in 0..case.n {
                        let at = if trans {
                            (case.col0 + j) + p * ops.ldb
                        } else {
                            p + (case.col0 + j) * ops.ldb
                        };
                        ops.b[ops.b_off + at] = T::of(b_value);
                    }
                }
                if case.reads_c() {
                    for j in 0..case.n {
                        for i in 0..m {
                            ops.c[ops.c_off + i + j * ops.ldc] = T::of(c_value);
                        }
                    }
                }
                let mut want = ops.c.clone();
                reference_blocks(&case, &ops, &mut want);
                let mut got = ops.c.clone();
                fpsr_write(0);
                run_kernel(&case, &ops, &mut got);
                let after = fpsr_read();
                fpsr_write(saved);
                assert_c(&case, &ops, &got, &want, test);
                assert_eq!(
                    after, added,
                    "{test}: {label}, m {m}, trans {trans}: flags {after:#x}, want {added:#x}"
                );
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_fpsr_depth_and_overflow_f32() {
    fpsr_depth_and_overflow::<f32>("test_sme2_few_rows_fpsr_depth_and_overflow_f32");
}

#[test]
fn test_sme2_few_rows_fpsr_depth_and_overflow_f64() {
    fpsr_depth_and_overflow::<f64>("test_sme2_few_rows_fpsr_depth_and_overflow_f64");
}

/// The largest shapes of the type: every row, many column panels ending in
/// a partial one, a deep depth in one block and in blocks of an odd KC step.
fn largest<T: FrElem>(test: &str) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let n = 7 * T::NR + 5;
    for (k, kc) in [(700, 700), (700, 255)] {
        for trans in [false, true] {
            for (alpha, beta) in [(0.5, -0.5), (1.0, 0.0)] {
                let case = Case::<T>::new(max_rows::<T>(), n, k, kc, trans).scalars(alpha, beta);
                check_case(&case, test);
            }
        }
    }
}

#[test]
fn test_sme2_few_rows_largest_f32() {
    largest::<f32>("test_sme2_few_rows_largest_f32");
}

#[test]
fn test_sme2_few_rows_largest_f64() {
    largest::<f64>("test_sme2_few_rows_largest_f64");
}

/// Plain Rust: the independent model [`seams_value`] equals the
/// per-block reference of the seams file on ordinary operands (so a failure
/// of the hardware tests that use the model is the kernel's).
#[test]
fn test_sme2_few_rows_seams_value_matches_reference() {
    fn compare<T: FrElem>() {
        let mut index = 0usize;
        for m in [1, T::TILE_ROWS + 1, max_rows::<T>()] {
            for (n, k, kc) in [
                (1, 1, 1),
                (T::NR + 1, 17, 8),
                (3, 40, 13),
                (2 * T::NR, 33, 33),
            ] {
                for trans in [false, true] {
                    let (alpha, beta) = PAIRS[index % PAIRS.len()];
                    index += 1;
                    for fill in [Fill::Random, Fill::NonFinite, Fill::SignedZeros] {
                        let case = Case::<T>::new(m, n, k, kc, trans)
                            .scalars(alpha, beta)
                            .filled(fill);
                        let ops = Operands::new(&case);
                        let mut want = ops.c.clone();
                        reference_blocks(&case, &ops, &mut want);
                        let a_of = |i: usize, p: usize| ops.a[ops.a_off + i + p * ops.lda];
                        let b_of = |p: usize, j: usize| {
                            let at = if trans {
                                (case.col0 + j) + p * ops.ldb
                            } else {
                                p + (case.col0 + j) * ops.ldb
                            };
                            ops.b[ops.b_off + at]
                        };
                        let mut got = ops.c.clone();
                        for j in 0..n {
                            for i in 0..m {
                                let at = ops.c_off + i + j * ops.ldc;
                                got[at] = seams_value(
                                    &a_of,
                                    &b_of,
                                    ops.c[at],
                                    (i, j),
                                    (k, kc),
                                    (case.alpha, case.beta),
                                );
                            }
                        }
                        assert_c(&case, &ops, &got, &want, "seams_value vs reference");
                    }
                }
            }
        }
    }
    compare::<f32>();
    compare::<f64>();
}
