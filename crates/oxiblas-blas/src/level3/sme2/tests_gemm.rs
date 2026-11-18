//! Tests of the SME2 GEMM driver of `driver.rs` and `parallel.rs` :
//! every accepted call against the NEON blocked result bit for bit, the
//! declining classes with C untouched, explicit KC, views, thread counts and
//! part counts, and the counters that prove SME2 ran.
//!
//! The cases, their NEON reference and the assertions (`run_case`,
//! `assert_same_bits`, `assert_matches_neon`, `assert_declined`,
//! `check_spec`) live in `tests_gemm_harness.rs`; its module docs say how a
//! case runs and what each assertion proves.
//!
//! Every test that executes SME2 code goes through `try_gemm` and so starts
//! with the backend gate `backend_or_skip` (it also skips in a build that
//! caps SIMD below SME2, where the backend is off). A decline test always
//! runs an accepted control call first, so on hardware it proves the class
//! declines and not merely that SME2 is missing. The two ungated
//! `declines_without_backend` tests are the counterpart: where the backend
//! is off for their type (no hardware, or a capped build) they call
//! `try_gemm` on shapes it accepts when it is on and prove that it declines
//! without touching anything.
//!
//! This file holds the decision, the split geometry, the block and decision
//! boundaries, views, transposes, explicit KC, parts, threads and the NEON
//! fallbacks; `tests_stress.rs` holds the size grid, special values,
//! adjacent operands, eight threads at once and the stress loops, on the
//! same harness.
//!
//! The thread-local test switches are set on the calling thread, which is
//! where the driver reads them: `detect::override_units`,
//! `driver::override_tuning` (MC target, B buffer size, parallel threshold,
//! depth blocks forced onto the NEON macro-panel) and
//! `parallel::mark_not_ready`.

use num_complex::Complex64;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::{MatMut, MatRef};

#[cfg(feature = "parallel")]
use super::args::same_result;
use super::detect::override_units;
use super::driver::{
    BlockCount, Class, MIN_KC, PAR_MIN_MACS_F32, PAR_MIN_MACS_F64, SMALL_CLASS_ON, SMALL_MACS,
    Shape, Tuning, TuningOverride, classify, kc_accepted, override_tuning, part_count, try_gemm,
};
use super::parallel::{Layout, Split, mark_not_ready, take_last_run};
use super::test_support::{
    Need, backend_or_skip, backend_present, decisions, force_off, hardware_present,
    kernel_entries_here, note_declined,
};
use super::tests_gemm_harness::{
    CBuf, Case, Operand, Outcome, Rng, SCALARS, Store, TRANSPOSES, TestElem, assert_declined,
    assert_kernel_calls_here, assert_matches_neon, assert_runs, assert_same_bits, check_spec,
    expected_part, run_case, spec_accepts, stored_shape,
};
#[cfg(feature = "parallel")]
use crate::level3::gemm::gemm_planned;
use crate::level3::gemm::{BlockingSource, call_blocking};
use crate::level3::gemm_packing::set_pack_cache_limit;
use crate::level3::gemm_plan::{DEFAULT_POLICY, GemmPlan, GemmRoute};
use crate::level3::trsm::Trans;

// ---------------------------------------------------------------------------
// The decision as pure functions (run everywhere)
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_gemm_classify() {
    assert_eq!(classify(1, 64, 64), Class::Thin);
    assert_eq!(classify(64, 1, 64), Class::Thin);
    assert_eq!(classify(1, 1, 1), Class::Thin);
    // Small: m·n·k <= 32³.
    assert_eq!(SMALL_MACS, 32 * 32 * 32);
    assert_eq!(classify(32, 32, 32), Class::Small);
    assert_eq!(classify(24, 8, 8), Class::Small);
    assert_eq!(classify(16, 16, 64), Class::Small);
    assert_eq!(classify(23, 8, 8), Class::SmallDeclined);
    assert_eq!(classify(24, 7, 8), Class::SmallDeclined);
    assert_eq!(classify(24, 8, 7), Class::SmallDeclined);
    assert_eq!(classify(16, 16, 16), Class::SmallDeclined);
    assert_eq!(classify(8, 64, 8), Class::SmallDeclined);
    assert_eq!(classify(16, 64, 16), Class::SmallDeclined);
    assert_eq!(classify(16, 16, 63), Class::SmallDeclined);
    assert_eq!(classify(15, 16, 64), Class::SmallDeclined);
    // Above 32³ every non-thin product is blocked, however narrow.
    assert_eq!(classify(2, 2, 8193), Class::Blocked);
    assert_eq!(classify(33, 32, 32), Class::Blocked);
    // Saturating work: huge dimensions do not wrap into the small class.
    assert_eq!(classify(usize::MAX, 2, 2), Class::Blocked);
    // The small class runs only when it is switched on, and it is
    // off by default (`SMALL_CLASS_ON`); the blocked class always runs.
    assert!(
        !Tuning::of::<f32>().small_class,
        "the small class is off by default"
    );
    assert_eq!(Tuning::of::<f64>().small_class, SMALL_CLASS_ON);
    assert!(Class::Small.accepted(true) && !Class::Small.accepted(false));
    assert!(Class::Blocked.accepted(true) && Class::Blocked.accepted(false));
    for small_on in [false, true] {
        assert!(!Class::Thin.accepted(small_on) && !Class::SmallDeclined.accepted(small_on));
    }
}

#[test]
fn test_sme2_gemm_kc_rule() {
    assert_eq!(MIN_KC, 32);
    assert!(kc_accepted(32, 1000));
    assert!(kc_accepted(256, 1000));
    assert!(
        !kc_accepted(31, 1000),
        "a degenerate step that splits the depth"
    );
    assert!(!kc_accepted(8, 9));
    // A short step that covers the whole depth is fine.
    assert!(kc_accepted(5, 5));
    assert!(kc_accepted(1, 1));
    assert!(!kc_accepted(0, 0));
    assert!(!kc_accepted(0, 10));
}

#[test]
fn test_sme2_gemm_part_count() {
    assert_eq!(PAR_MIN_MACS_F32, 1 << 27);
    assert_eq!(PAR_MIN_MACS_F64, 1 << 25);
    let big = 1 << 30;
    // Sequential, a single unit or too little work: one part.
    assert_eq!(part_count(big, Par::Seq, 4, 1 << 25), 1);
    #[cfg(feature = "parallel")]
    {
        assert_eq!(part_count(big, Par::RayonWith(8), 1, 1 << 25), 1);
        assert_eq!(part_count((1 << 25) - 1, Par::RayonWith(8), 4, 1 << 25), 1);
        assert_eq!(part_count(1 << 25, Par::RayonWith(8), 4, 1 << 25), 4);
        assert_eq!(part_count(big, Par::RayonWith(3), 4, 1 << 25), 3);
        assert_eq!(part_count(big, Par::RayonWith(2), 2, 1 << 25), 2);
        assert_eq!(part_count(big, Par::RayonWith(1), 2, 1 << 25), 1);
        assert_eq!(part_count(big, Par::RayonWith(0), 2, 1 << 25), 1);
    }
}

#[test]
fn test_sme2_gemm_layout_splits() {
    let tuning = Tuning::of::<f64>();
    // Rows (m >= n): pieces of whole 64-row panels, near equal, all columns.
    let layout = Layout::new::<f64>(&Shape::blocked(300, 40, 64), 3, &tuning).expect("layout");
    assert_eq!(layout.split, Split::Rows);
    let rows: Vec<(usize, usize)> = layout
        .parts
        .iter()
        .map(|part| (part.rows.start, part.rows.len))
        .collect();
    assert_eq!(rows, [(0, 64), (64, 128), (192, 108)]);
    for part in &layout.parts {
        assert_eq!((part.cols.start, part.cols.len), (0, 40));
        assert_eq!(part.mc, part.rows.len, "MC covers a piece of <= 512 rows");
        assert_eq!(part.nc, 40);
        assert_eq!(part.a_len, part.mc.div_ceil(64) * 64 * 64);
        assert_eq!(part.b_len, 40 * 64);
    }
    // Columns (m < n): pieces of whole NR-column panels, all rows.
    let layout =
        Layout::new::<f32>(&Shape::blocked(40, 200, 64), 2, &Tuning::of::<f32>()).expect("layout");
    assert_eq!(layout.split, Split::Cols);
    let cols: Vec<(usize, usize)> = layout
        .parts
        .iter()
        .map(|part| (part.cols.start, part.cols.len))
        .collect();
    assert_eq!(cols, [(0, 96), (96, 104)]);
    // A square C splits along its rows.
    let layout = Layout::new::<f64>(&Shape::blocked(128, 128, 64), 2, &tuning).expect("layout");
    assert_eq!(layout.split, Split::Rows);
    assert_eq!(layout.parts.len(), 2);
    let layout = Layout::new::<f32>(&Shape::blocked(64, 64, 64), 2, &tuning).expect("layout");
    assert_eq!((layout.split, layout.parts.len()), (Split::Rows, 1));
    // The part count is clamped to the panels on the split side.
    let layout = Layout::new::<f64>(&Shape::blocked(130, 100, 64), 7, &tuning).expect("layout");
    assert_eq!(layout.parts.len(), 3);
    let layout = Layout::new::<f64>(&Shape::blocked(10, 9, 64), 4, &tuning).expect("layout");
    assert_eq!(layout.parts.len(), 1);
    // Every count tiles C exactly with non-empty pieces at panel boundaries.
    for (m, n) in [(1000, 3), (3, 1000), (65, 64), (64, 65), (129, 7)] {
        for parts in 1..=9 {
            let layout =
                Layout::new::<f32>(&Shape::blocked(m, n, 8), parts, &tuning).expect("layout");
            let (len, unit) = match layout.split {
                Split::Rows => (m, 64),
                Split::Cols => (n, 16),
            };
            let mut next = 0;
            for part in &layout.parts {
                let piece = match layout.split {
                    Split::Rows => part.rows,
                    Split::Cols => part.cols,
                };
                assert_eq!(piece.start, next, "{m}x{n} in {parts}");
                assert!(piece.len > 0);
                assert_eq!(piece.start % unit, 0);
                next = piece.start + piece.len;
            }
            assert_eq!(next, len, "{m}x{n} in {parts}");
        }
    }
}

#[test]
fn test_sme2_gemm_layout_block_sizes() {
    // MC: balanced blocks of at most 512 rows in 64-row units.
    let tuning = Tuning::of::<f64>();
    let layout = Layout::new::<f64>(&Shape::blocked(1100, 8, 256), 1, &tuning).expect("layout");
    assert_eq!(layout.parts[0].mc, 384);
    // NC: the widest multiple of NR whose kc × NC block fits the 4 MiB cap,
    // for a part of several MC blocks (1100 rows: three blocks of 384).
    let layout = Layout::new::<f64>(&Shape::blocked(1100, 5000, 256), 1, &tuning).expect("layout");
    assert_eq!((layout.parts[0].b_cap, layout.parts[0].nc), (4 << 20, 1672));
    assert!(layout.parts[0].b_len * 8 <= 4 << 20);
    let layout = Layout::new::<f32>(&Shape::blocked(1100, 5000, 256), 1, &Tuning::of::<f32>())
        .expect("layout");
    assert_eq!((layout.parts[0].b_cap, layout.parts[0].nc), (4 << 20, 2512));
    // A part whose rows are one MC block (8 rows) with a packed B of the
    // depth block above 4 MiB (5000 × 256 elements) gets the 2 MiB cap.
    let layout = Layout::new::<f64>(&Shape::blocked(8, 5000, 256), 1, &tuning).expect("layout");
    assert_eq!((layout.parts[0].b_cap, layout.parts[0].nc), (2 << 20, 1000));
    assert!(layout.parts[0].b_len * 8 <= 2 << 20);
    let layout =
        Layout::new::<f32>(&Shape::blocked(8, 5000, 256), 1, &Tuning::of::<f32>()).expect("layout");
    assert_eq!((layout.parts[0].b_cap, layout.parts[0].nc), (2 << 20, 1680));
    // A depth so deep that not even NR columns fit: NC = NR. Its buffers
    // (64 MiB of B alone) exceed the default packing-cache limit of 8 MiB,
    // so the layout declines there; with a limit that admits them it is
    // built.
    assert!(
        Layout::new::<f64>(&Shape::blocked(8, 100, 1 << 20), 1, &tuning).is_none(),
        "buffers above the packing-cache limit"
    );
    {
        let _limit = set_pack_cache_limit(1 << 40);
        let roomy = Tuning::of::<f64>();
        let layout =
            Layout::new::<f64>(&Shape::blocked(8, 100, 1 << 20), 1, &roomy).expect("layout");
        assert_eq!(layout.parts[0].nc, 8);
    }
    // Overflowing sizes decline instead of wrapping.
    assert!(
        Layout::new::<f64>(
            &Shape::blocked(usize::MAX / 2, 8, usize::MAX / 4),
            1,
            &tuning
        )
        .is_none()
    );
    assert!(Layout::new::<f64>(&Shape::blocked(0, 8, 8), 1, &tuning).is_none());
}

#[test]
fn test_sme2_gemm_tuning_override_is_per_thread() {
    let defaults = Tuning::of::<f32>();
    assert_eq!(defaults.mc_target, 512);
    assert_eq!(defaults.b_bytes, 4 << 20);
    assert_eq!(defaults.par_min_macs, PAR_MIN_MACS_F32);
    assert_eq!(Tuning::of::<f64>().par_min_macs, PAR_MIN_MACS_F64);
    {
        let _guard = override_tuning(TuningOverride {
            mc_target: Some(64),
            par_min_macs: Some(0),
            ..TuningOverride::default()
        });
        let tuning = Tuning::of::<f32>();
        assert_eq!(tuning.mc_target, 64);
        assert_eq!(tuning.b_bytes, 4 << 20);
        assert_eq!(tuning.par_min_macs, 0);
        let elsewhere = std::thread::spawn(Tuning::of::<f32>).join();
        assert!(matches!(elsewhere, Ok(t) if t == defaults), "{elsewhere:?}");
    }
    assert_eq!(Tuning::of::<f32>(), defaults);
}

// ---------------------------------------------------------------------------
// Accepted calls against NEON (execute SME2)
// ---------------------------------------------------------------------------

/// Several row panels, column panels and depth blocks (explicit KC fitted to
/// 40 for k = 100), every epilogue mode, the default and the explicit
/// blocking, and small MC / B-buffer targets that make several row and
/// column blocks.
fn blocked_shapes<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let nr = T::NR;
    // Every case is sequential, so its one part runs on this thread: the
    // kernels' count of this thread moves by exactly the blocks of the runs.
    let entries = kernel_entries_here();
    let mut calls = 0;
    let mut count = |out: &Outcome<T>| {
        if let Some(report) = &out.report {
            calls += report
                .blocks
                .iter()
                .map(|blocks| blocks.kernel_calls)
                .sum::<u64>();
        }
    };
    // (65, 2 NR + 1, 33), not (65, NR + 1, 33), which for `f64`
    // (65 · 9 · 33 <= 32³) is the small class, declined by default.
    for (m, n, k) in [
        (130, 3 * nr - 5, 100),
        (64, 2 * nr, 64),
        (65, 2 * nr + 1, 33),
    ] {
        for (alpha, beta, nan) in SCALARS {
            let mut case = Case::<T>::new(m, n, k).scalars(alpha, beta).explicit_kc(48);
            if nan {
                case = case.nan_c();
            }
            let out = run_case(&case);
            if k == 100 {
                assert_eq!(out.kc, 40, "three depth blocks of 40, 40, 20");
            }
            assert_matches_neon(&out, &format!("{test} {case:?}"));
            count(&out);
            let default = run_case(&Case {
                source: BlockingSource::Default,
                ..case
            });
            assert_matches_neon(&default, &format!("{test} default {case:?}"));
            count(&default);
        }
    }
    // Several MC and NC blocks: MC = 64, NC = 2 NR (kc = 40).
    let _small = override_tuning(TuningOverride {
        mc_target: Some(64),
        b_bytes: Some(40 * core::mem::size_of::<T>() * 2 * nr),
        ..TuningOverride::default()
    });
    for (alpha, beta, nan) in SCALARS {
        let mut case = Case::<T>::new(200, 5 * nr + 3, 100)
            .scalars(alpha, beta)
            .explicit_kc(48);
        if nan {
            case = case.nan_c();
        }
        let out = run_case(&case);
        assert_eq!(out.kc, 40);
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.layout.parts[0].mc, 64);
        assert_eq!(report.layout.parts[0].nc, 2 * nr);
        assert_matches_neon(&out, &format!("{test} small blocks {case:?}"));
        count(&out);
    }
    assert!(calls > 0, "{test}: no kernel ran");
    assert_eq!(
        kernel_entries_here() - entries,
        calls,
        "{test}: kernel entries on this thread"
    );
}

#[test]
fn test_sme2_gemm_f32_blocked_shapes() {
    blocked_shapes::<f32>("test_sme2_gemm_f32_blocked_shapes");
}

#[test]
fn test_sme2_gemm_f64_blocked_shapes() {
    blocked_shapes::<f64>("test_sme2_gemm_f64_blocked_shapes");
}

/// Every transposition pair, with A and B stored column-major and
/// row-major (the latter normalised through the transpose), several depth
/// blocks.
fn transpositions<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let stores = [
        (Store::ColMajor { pad: 2 }, Store::ColMajor { pad: 0 }),
        (Store::RowMajor { pad: 3 }, Store::RowMajor { pad: 1 }),
    ];
    for trans_a in TRANSPOSES {
        for trans_b in TRANSPOSES {
            for (a_store, b_store) in stores {
                let case = Case::<T>::new(97, 2 * T::NR + 3, 75)
                    .trans(trans_a, trans_b)
                    .stores(a_store, b_store)
                    .scalars(-1.0, -0.5)
                    .explicit_kc(32);
                let out = run_case(&case);
                assert!(out.kc >= MIN_KC && out.kc < case.k, "{case:?}");
                assert_matches_neon(&out, &format!("{test} {case:?}"));
            }
        }
    }
}

#[test]
fn test_sme2_gemm_f32_transpositions() {
    transpositions::<f32>("test_sme2_gemm_f32_transpositions");
}

#[test]
fn test_sme2_gemm_f64_transpositions() {
    transpositions::<f64>("test_sme2_gemm_f64_transpositions");
}

/// The small class (accepted part), a large leading dimension of C, and
/// depth blocks forced onto the NEON macro-panel mixed with SME2 blocks.
fn small_views_and_block_fallback<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    // The small class is off by default (it declines), and runs
    // (on the few-rows route: m <= 32) when the test switches it on.
    for (m, n, k) in [(32, 32, 32), (24, 8, 8), (16, 16, 64)] {
        assert_declined(
            &run_case(&Case::<T>::new(m, n, k)),
            &format!("{test} small {m}x{n}x{k}, small class off"),
        );
        let _on = override_tuning(TuningOverride {
            small_class: Some(true),
            ..TuningOverride::default()
        });
        let out = run_case(&Case::<T>::new(m, n, k));
        assert_matches_neon(&out, &format!("{test} small {m}x{n}x{k}"));
    }
    let mut wide = Case::<T>::new(70, 3 * T::NR, 90).explicit_kc(32);
    wide.c_pad = 301;
    wide.a_store = Store::ColMajor { pad: 517 };
    assert_matches_neon(&run_case(&wide), &format!("{test} large ld"));

    // Depth block 1 of 3 on NEON, 0 and 2 on SME2.
    let mixed = override_tuning(TuningOverride {
        neon_depths: 0b010,
        ..TuningOverride::default()
    });
    let case = Case::<T>::new(100, 2 * T::NR + 1, 100).explicit_kc(48);
    let entries = kernel_entries_here();
    let out = run_case(&case);
    assert_eq!(out.kc, 40);
    let label = format!("{test} mixed depth blocks");
    assert_same_bits(&out, &label);
    // Depth block 1 of 3 ran on NEON for every column and row block.
    assert_runs(&out, &label, |_, part| {
        let per_depth = part.cols.len.div_ceil(part.nc) * part.rows.len.div_ceil(part.mc);
        (
            false,
            BlockCount {
                sme: 2 * per_depth,
                neon: per_depth,
                ..BlockCount::default()
            },
        )
    });
    // One sequential part on this thread: two thirds of its blocks entered
    // the kernel, here.
    let report = out.report.as_ref().expect("report");
    assert_eq!(
        kernel_entries_here() - entries,
        report.blocks[0].kernel_calls,
        "{label}: kernel entries on this thread"
    );
    drop(mixed);

    // The NEON block fallback away from the origin: several row and column
    // blocks (MC = 64, NC = 2 NR), transposed operands, and (with rayon) two
    // parts, so the fallback packs op(A) and op(B) at non-zero offsets of
    // both the part and the block. Depth blocks 0 and 2 on NEON.
    let _offsets = override_tuning(TuningOverride {
        mc_target: Some(64),
        b_bytes: Some(40 * core::mem::size_of::<T>() * 2 * T::NR),
        par_min_macs: Some(0),
        neon_depths: 0b101,
        ..TuningOverride::default()
    });
    #[cfg(feature = "parallel")]
    let (_units, par) = (override_units(2), Par::RayonWith(2));
    #[cfg(not(feature = "parallel"))]
    let par = Par::Seq;
    for (trans_a, trans_b) in [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::Trans),
    ] {
        for (m, n) in [(200, 5 * T::NR + 3), (70, 13 * T::NR + 1)] {
            let case = Case::<T>::new(m, n, 100)
                .explicit_kc(48)
                .trans(trans_a, trans_b)
                .par(par);
            let out = run_case(&case);
            let label = format!("{test} fallback offsets {case:?}");
            assert_eq!(out.kc, 40, "{label}");
            assert_same_bits(&out, &label);
            let report = out.report.as_ref().expect("report");
            assert!(report.layout.parts.len() > 1 || cfg!(not(feature = "parallel")));
            assert_runs(&out, &label, |_, part| {
                let per_depth = part.cols.len.div_ceil(part.nc) * part.rows.len.div_ceil(part.mc);
                assert!(per_depth > 1, "{label}: one block per depth");
                (
                    false,
                    BlockCount {
                        sme: per_depth,
                        neon: 2 * per_depth,
                        ..BlockCount::default()
                    },
                )
            });
        }
    }
}

#[test]
fn test_sme2_gemm_f32_small_views_and_block_fallback() {
    small_views_and_block_fallback::<f32>("test_sme2_gemm_f32_small_views_and_block_fallback");
}

#[test]
fn test_sme2_gemm_f64_small_views_and_block_fallback() {
    small_views_and_block_fallback::<f64>("test_sme2_gemm_f64_small_views_and_block_fallback");
}

#[test]
fn test_sme2_gemm_counters_move() {
    if !backend_or_skip("test_sme2_gemm_counters_move", Need::F64) {
        return;
    }
    let entries = kernel_entries_here();
    let before = decisions();
    let out = run_case(&Case::<f64>::new(64, 16, 40));
    assert_matches_neon(&out, "counters");
    assert_eq!(decisions(), before + 1);
    // One part, one depth block, one row block, two column panels in one
    // column block: exactly one macro-kernel call (A read in place) and one
    // ZA-transposer call that packs B (`op(B) = B`), on this thread
    // (the count is per thread, so other tests cannot add to it).
    assert_eq!(
        kernel_entries_here() - entries,
        2,
        "counters: kernel entries"
    );
    let declined = run_case(&Case::<f64>::new(1, 16, 40));
    assert_declined(&declined, "thin");
    assert_eq!(decisions(), before + 1);
}

// ---------------------------------------------------------------------------
// Declining classes (execute SME2 for the controls)
// ---------------------------------------------------------------------------

fn declines<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let control = Case::<T>::new(70, 3 * T::NR, 96).explicit_kc(32);
    assert_matches_neon(&run_case(&control), &format!("{test} control"));

    // Thin.
    assert_declined(&run_case(&Case::<T>::new(1, 64, 64)), "m == 1");
    assert_declined(&run_case(&Case::<T>::new(64, 1, 64)), "n == 1");
    // Small below the thresholds.
    for (m, n, k) in [(16, 16, 16), (8, 64, 8), (16, 64, 16), (23, 8, 8)] {
        assert_declined(
            &run_case(&Case::<T>::new(m, n, k)),
            &format!("small {m}x{n}x{k}"),
        );
    }
    // A degenerate explicit KC (fitted to 16 for k = 96); a short KC that
    // covers the whole depth is accepted.
    let degenerate = run_case(&control.explicit_kc(16));
    assert!(degenerate.kc < MIN_KC);
    assert_declined(&degenerate, "degenerate KC");
    let whole = run_case(&Case::<T>::new(64, 64, 16).explicit_kc(16));
    assert_matches_neon(&whole, "KC == k");
    // Views that cannot be normalised.
    for (a_store, b_store) in [
        (Store::Strided, Store::ColMajor { pad: 0 }),
        (Store::ColMajor { pad: 0 }, Store::Strided),
        (Store::Strided, Store::Strided),
    ] {
        for trans in [Trans::NoTrans, Trans::Trans] {
            let case = control.stores(a_store, b_store).trans(trans, trans);
            assert_declined(&run_case(&case), &format!("strided {case:?}"));
        }
    }
    // Calls the GEMM core never sends: alpha == 0, k == 0, empty C.
    assert_declined(&run_case(&control.scalars(0.0, 0.5)), "alpha == 0");
    assert_declined(&run_case(&control.scalars(-0.0, 0.5)), "alpha == -0");
    assert_declined(&run_case(&Case::<T>::new(64, 64, 0)), "k == 0");
    assert_declined(&run_case(&Case::<T>::new(0, 64, 64)), "m == 0");
    // SME2 forced off on the calling thread.
    {
        let _off = force_off();
        assert_declined(&run_case(&control), "forced off");
    }
    // The same control still runs afterwards.
    assert_matches_neon(&run_case(&control), &format!("{test} control again"));
}

#[test]
fn test_sme2_gemm_f32_declines() {
    declines::<f32>("test_sme2_gemm_f32_declines");
}

#[test]
fn test_sme2_gemm_f64_declines() {
    declines::<f64>("test_sme2_gemm_f64_declines");
}

#[test]
fn test_sme2_gemm_declines_other_types_and_overlapping_c() {
    if !backend_or_skip(
        "test_sme2_gemm_declines_other_types_and_overlapping_c",
        Need::F32,
    ) {
        return;
    }
    // The control: the same shape in f32 is accepted and runs its kernels,
    // so the declines below are the type's and the layout's, not SME2's.
    let control = run_case(&Case::<f32>::new(64, 64, 64));
    assert_matches_neon(&control, "f32 control");
    // A complex product of an accepted shape.
    let (m, n, k) = (64, 64, 64);
    let a: Vec<Complex64> = (0..m * k)
        .map(|i| Complex64::new(i as f64 * 0.25, 1.0))
        .collect();
    let b: Vec<Complex64> = (0..k * n)
        .map(|i| Complex64::new(1.0, i as f64 * -0.5))
        .collect();
    let mut c = vec![Complex64::new(f64::NAN, 3.0); m * n];
    let plan = GemmPlan::<Complex64>::new(DEFAULT_POLICY);
    let blocks = call_blocking(BlockingSource::Default, &plan, m, k, n);
    let a_view = MatRef::from_column_major(&a, m, k).expect("A");
    let b_view = MatRef::from_column_major(&b, k, n).expect("B");
    let before = decisions();
    let accepted = try_gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        Complex64::new(1.0, 0.0),
        &a_view,
        &b_view,
        Complex64::new(0.0, 0.0),
        &mut MatMut::from_column_major(&mut c, m, n).expect("C"),
        &blocks,
        Par::Seq,
    );
    assert!(!accepted, "complex product accepted");
    assert_eq!(decisions(), before);
    assert!(c.iter().all(|z| z.re.is_nan() && z.im == 3.0));

    // A C whose columns overlap (leading dimension below m, down to one
    // short of m) declines without writing.
    let (m, n, k) = (64, 64, 64);
    let a = vec![0.5_f32; m * k];
    let b = vec![0.25_f32; k * n];
    let plan = GemmPlan::<f32>::new(DEFAULT_POLICY);
    let blocks = call_blocking(BlockingSource::Default, &plan, m, k, n);
    for ld in [m - 8, m - 1] {
        let mut c = vec![f32::from_bits(0x7fa0_0000); (n - 1) * ld + m];
        // SAFETY: element `(i, j)` is at `i + j * ld`, below
        // `(n - 1) * ld + m = c.len()`; the view's elements overlap (the
        // decision must decline before writing any), and `c` is not used
        // while the view lives.
        let mut c_view = unsafe { MatMut::new(c.as_mut_ptr(), m, n, ld) };
        let accepted = try_gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &MatRef::from_column_major(&a, m, k).expect("A"),
            &MatRef::from_column_major(&b, k, n).expect("B"),
            0.0,
            &mut c_view,
            &blocks,
            Par::Seq,
        );
        assert!(!accepted, "overlapping C (ld {ld}) accepted");
        assert!(c.iter().all(|x| x.to_bits() == 0x7fa0_0000), "ld {ld}");
    }
}

/// Calls that the GEMM core rejects with its dimension checks (C one row
/// or one column too many or too few, op(B) with one row too many) decline
/// with C untouched, after an accepted control of the same shape.
fn declines_mismatched_shapes<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (70, 3 * T::NR, 64);
    let mut rng = Rng::new(17);
    let a = Operand::<T>::new(m, k, Store::ColMajor { pad: 0 }, &mut rng);
    let b = Operand::<T>::new(k, n, Store::ColMajor { pad: 0 }, &mut rng);
    let b_deep = Operand::<T>::new(k + 1, n, Store::ColMajor { pad: 0 }, &mut rng);
    let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
    let blocks = call_blocking(BlockingSource::Default, &plan, m, k, n);
    for (rows, cols, deep, accept) in [
        (m, n, false, true),
        (m + 1, n, false, false),
        (m - 1, n, false, false),
        (m, n + 1, false, false),
        (m, n - 1, false, false),
        (m, n, true, false),
    ] {
        let mut c = CBuf::<T>::new(rows, cols, 0, false, &mut rng);
        let before = c.data.clone();
        let decided = decisions();
        let entries = kernel_entries_here();
        let _ = take_last_run();
        let b_view = if deep { b_deep.view() } else { b.view() };
        let accepted = try_gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            T::of(0.5),
            &a.view(),
            &b_view,
            T::of(-1.0),
            &mut c.view(),
            &blocks,
            Par::Seq,
        );
        let label = format!("{test}: C {rows}x{cols}, op(B) deeper {deep}");
        assert_eq!(accepted, accept, "{label}");
        assert_eq!(decisions() - decided, u64::from(accept), "{label}");
        if accept {
            assert!(assert_kernel_calls_here(entries, &label) > 0, "{label}");
        } else {
            assert!(take_last_run().is_none(), "{label}: a part ran");
            assert_eq!(kernel_entries_here(), entries, "{label}: a kernel ran");
            for (index, (got, was)) in c.data.iter().zip(&before).enumerate() {
                assert_eq!(got.bits(), was.bits(), "{label}: element {index}");
            }
        }
    }
}

#[test]
fn test_sme2_gemm_f32_declines_mismatched_shapes() {
    declines_mismatched_shapes::<f32>("test_sme2_gemm_f32_declines_mismatched_shapes");
}

#[test]
fn test_sme2_gemm_f64_declines_mismatched_shapes() {
    declines_mismatched_shapes::<f64>("test_sme2_gemm_f64_declines_mismatched_shapes");
}

/// The parallel threshold of the `declines_without_backend` tests (through
/// the test tuning): their shapes lie on both sides of it.
const DECLINE_PAR_MIN_MACS: usize = 1 << 18;

/// Where the SME2 backend is off for `T` (no hardware for it, or a build
/// that caps SIMD below SME2), `try_gemm` declines calls that it accepts
/// when the backend is on: one below and one above the parallel threshold
/// of the test tuning, NoTrans and Trans, `beta = 0` with C full of a
/// signalling NaN and `beta = 1`. Each returns `false`, takes no decision,
/// runs no part and no kernel (the kernels' count of this thread), and
/// leaves C bitwise untouched. Then the test writes the marker
/// `sme2-gate: declined <test>`. Ungated: it runs natively, in the negative
/// QEMU pass and in capped builds; where the backend is on it has nothing
/// to check, prints one `note:` line and writes no marker (not a `skip:`
/// line: the positive pass of `tools/sme2_qemu.sh` counts every `skip:`
/// line as a failure when the output is not captured). It never panics
/// because of `OXIBLAS_TEST_REQUIRE_SME2` or `OXIBLAS_TEST_EXPECT_NO_SME2`.
fn declines_without_backend<T: TestElem>(test: &str) {
    if backend_present(T::NEED) {
        println!(
            "note: {test}: the SME2 backend is on for {:?}; this test checks the declines where it is off",
            T::NEED
        );
        return;
    }
    let seen = if hardware_present(T::NEED) {
        "capped build"
    } else {
        "no hardware"
    };
    #[cfg(feature = "parallel")]
    let (_units, par) = (override_units(2), Par::RayonWith(2));
    #[cfg(not(feature = "parallel"))]
    let par = Par::Seq;
    let _tuning = override_tuning(TuningOverride {
        par_min_macs: Some(DECLINE_PAR_MIN_MACS),
        ..TuningOverride::default()
    });
    let nr = T::NR;
    let mut rng = Rng::new(29);
    let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
    let mut calls = 0;
    for (m, n, k, above) in [(70, 2 * nr, 64, false), (200, 3 * nr, 96, true)] {
        assert_eq!(
            m * n * k >= DECLINE_PAR_MIN_MACS,
            above,
            "{test}: {m}x{n}x{k}"
        );
        for trans in [Trans::NoTrans, Trans::Trans] {
            let case = Case::<T>::new(m, n, k).trans(trans, trans);
            let blocks = call_blocking(case.source, &plan, m, k, n);
            let kc = blocks.sequential().kc.min(k);
            // A call the backend accepts when it is on (the spec's rule).
            assert!(spec_accepts(&case, kc), "{test}: {case:?}");
            let (ar, ac) = stored_shape(trans, m, k);
            let (br, bc) = stored_shape(trans, k, n);
            let a = Operand::<T>::new(ar, ac, case.a_store, &mut rng);
            let b = Operand::<T>::new(br, bc, case.b_store, &mut rng);
            for (beta, signalling) in [(0.0, true), (1.0, false)] {
                let mut c = CBuf::<T>::new(m, n, case.c_pad, false, &mut rng);
                if signalling {
                    c.data.fill(T::SENTINEL);
                }
                let before = c.data.clone();
                let label = format!("{test} ({seen}): {m}x{n}x{k} {trans:?} beta {beta}");
                let decided = decisions();
                let entries = kernel_entries_here();
                let _ = take_last_run();
                let accepted = try_gemm(
                    trans,
                    trans,
                    T::of(0.75),
                    &a.view(),
                    &b.view(),
                    T::of(beta),
                    &mut c.view(),
                    &blocks,
                    par,
                );
                assert!(!accepted, "{label}: accepted");
                assert_eq!(decisions(), decided, "{label}: decisions");
                assert!(take_last_run().is_none(), "{label}: a part ran");
                assert_eq!(kernel_entries_here(), entries, "{label}: a kernel ran");
                for (index, (got, was)) in c.data.iter().zip(&before).enumerate() {
                    assert_eq!(got.bits(), was.bits(), "{label}: element {index} of C");
                }
                calls += 1;
            }
        }
    }
    note_declined(test);
    println!("{test}: try_gemm declined all {calls} calls ({seen})");
}

#[test]
fn test_sme2_gemm_f32_declines_without_backend() {
    declines_without_backend::<f32>("test_sme2_gemm_f32_declines_without_backend");
}

#[test]
fn test_sme2_gemm_f64_declines_without_backend() {
    declines_without_backend::<f64>("test_sme2_gemm_f64_declines_without_backend");
}

// ---------------------------------------------------------------------------
// Parts and threads (execute SME2)
// ---------------------------------------------------------------------------

/// Runs `case` with `units` units and `par`, splitting from any work, and
/// checks the bits and the number and side of the parts.
#[cfg(feature = "parallel")]
fn parts_case<T: TestElem>(case: &Case<T>, units: usize, expect_parts: usize, label: &str) {
    let _units = override_units(units);
    let _split_all = override_tuning(TuningOverride {
        par_min_macs: Some(0),
        ..TuningOverride::default()
    });
    let out = run_case(case);
    assert_matches_neon(&out, label);
    // The promise is against the NEON result of the same `par`; the NEON
    // drivers are deterministic over `par`, which this re-checks.
    let mut neon_par = out.before.clone();
    {
        let _neon = force_off();
        let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
        gemm_planned(
            case.trans_a,
            case.trans_b,
            case.alpha,
            out.a.view(),
            out.b.view(),
            case.beta,
            neon_par.view(),
            case.par,
            case.source,
            &plan,
        );
    }
    for (index, (got, want)) in out.got.data.iter().zip(&neon_par.data).enumerate() {
        assert!(
            same_result(*got, *want),
            "{label}: element {index} differs from NEON with the same par"
        );
    }
    let report = out.report.as_ref().expect("report");
    assert_eq!(report.layout.parts.len(), expect_parts, "{label}: parts");
    let side = if case.m >= case.n {
        Split::Rows
    } else {
        Split::Cols
    };
    assert_eq!(report.layout.split, side, "{label}: split");
    assert_eq!(
        report.neon,
        vec![false; expect_parts],
        "{label}: NEON parts"
    );
}

#[cfg(feature = "parallel")]
fn parts_and_threads<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let rayon_threads = rayon::current_num_threads();
    // Tall (split along rows) and wide (along columns), several depth blocks.
    // Square C splits along rows (m >= n). With three panels on the split
    // side, the last piece is a single row or a single column.
    let tall = Case::<T>::new(300, 2 * T::NR + 3, 90).explicit_kc(32);
    let wide = Case::<T>::new(70, 25 * T::NR - 3, 90).explicit_kc(32);
    let square = Case::<T>::new(200, 200, 40).c_pad(0);
    let last_row = Case::<T>::new(129, 2 * T::NR, 90)
        .explicit_kc(32)
        .trans(Trans::Trans, Trans::NoTrans);
    let last_col = Case::<T>::new(16, 2 * T::NR + 1, 130)
        .explicit_kc(64)
        .trans(Trans::NoTrans, Trans::Trans)
        .c_pad(0);
    for base in [tall, wide, square, last_row, last_col] {
        let panels = if base.m >= base.n {
            base.m.div_ceil(64)
        } else {
            base.n.div_ceil(T::NR)
        };
        for units in [2, 3] {
            for (par, threads) in [
                (Par::Seq, 1),
                (Par::Rayon, rayon_threads),
                (Par::RayonWith(2), 2),
                (Par::RayonWith(3), 3),
                (Par::RayonWith(7), 7),
            ] {
                let expect = if par == Par::Seq {
                    1
                } else {
                    units.min(threads).clamp(1, panels)
                };
                let case = base.par(par);
                parts_case(
                    &case,
                    units,
                    expect,
                    &format!("{test} units {units} {case:?}"),
                );
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_f32_parts_and_threads() {
    parts_and_threads::<f32>("test_sme2_gemm_f32_parts_and_threads");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_f64_parts_and_threads() {
    parts_and_threads::<f64>("test_sme2_gemm_f64_parts_and_threads");
}

/// `(par, mask of "not ready" parts, expected part count)` for
/// [`test_sme2_gemm_f64_part_not_ready`] (three units).
fn not_ready_runs() -> Vec<(Par, u64, usize)> {
    let runs = vec![(Par::Seq, 0b1, 1)];
    #[cfg(feature = "parallel")]
    let runs = {
        let mut runs = runs;
        runs.extend([
            (Par::RayonWith(3), 0b010, 3),
            (Par::RayonWith(3), 0b101, 3),
            (Par::RayonWith(3), 0b111, 3),
            (Par::RayonWith(2), 0b001, 2),
        ]);
        runs
    };
    runs
}

#[test]
fn test_sme2_gemm_f64_part_not_ready() {
    let test = "test_sme2_gemm_f64_part_not_ready";
    if !backend_or_skip(test, Need::F64) {
        return;
    }
    let _units = override_units(3);
    let _split_all = override_tuning(TuningOverride {
        par_min_macs: Some(0),
        ..TuningOverride::default()
    });
    let runs = not_ready_runs();
    for base in [
        Case::<f64>::new(300, 19, 90).explicit_kc(32),
        Case::<f64>::new(70, 197, 90)
            .explicit_kc(32)
            .trans(Trans::Trans, Trans::Trans),
        Case::<f64>::new(200, 40, 90)
            .explicit_kc(32)
            .trans(Trans::Trans, Trans::NoTrans)
            .scalars(0.5, 0.0)
            .nan_c(),
    ] {
        for &(par, mask, parts) in &runs {
            let _marked = mark_not_ready(mask);
            let case = base.par(par);
            let out = run_case(&case);
            let label = format!("{test} mask {mask:#b} {case:?}");
            assert_same_bits(&out, &label);
            let report = out.report.as_ref().expect("report");
            assert_eq!(report.layout.parts.len(), parts, "{label}");
            // A marked part runs on the NEON driver (no blocks of its own),
            // every other part runs all its blocks on SME2.
            assert_runs(&out, &label, |index, part| {
                if mask & (1 << index) != 0 {
                    (true, BlockCount::default())
                } else {
                    (false, expected_part(&out, part))
                }
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_f64_parallel_threshold() {
    let test = "test_sme2_gemm_f64_parallel_threshold";
    if !backend_or_skip(test, Need::F64) {
        return;
    }
    // The real threshold, 2^25 multiply-adds: 256 × 256 × 511 stays on one
    // part, 256 × 256 × 512 splits.
    let _units = override_units(2);
    for (k, parts) in [(511, 1), (512, 2)] {
        let case = Case::<f64>::new(256, 256, k).par(Par::RayonWith(2));
        let out = run_case(&case);
        assert_matches_neon(&out, &format!("{test} k = {k}"));
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.layout.parts.len(), parts, "k = {k}");
    }
}

// ---------------------------------------------------------------------------
// Boundaries of the decision and of the blocks (execute SME2)
// ---------------------------------------------------------------------------

/// Every threshold of the decision at its exact value and one off, through
/// `try_gemm`, against the spec's verdict ([`spec_accepts`]); the KC rule
/// with the fitted KC asserted (explicit targets are fitted by
/// `balanced_block(k, target, 8)`); C with `ldc == m`.
fn decision_boundaries<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    // (m, n, k, accepted by default, accepted with the small class on):
    // thin, the small rule, the 32³ border. The small class is off
    // by default, so the small rule's accepted shapes decline unless the
    // test switches it on.
    for (m, n, k, accept, accept_small_on) in [
        (1, 40, 410, false, false),
        (2, 40, 410, true, true),
        (40, 1, 410, false, false),
        (40, 2, 410, true, true),
        (24, 8, 8, false, true),
        (23, 8, 8, false, false),
        (24, 7, 8, false, false),
        (24, 8, 7, false, false),
        (16, 16, 64, false, true),
        (15, 16, 64, false, false),
        (16, 15, 64, false, false),
        (16, 16, 63, false, false),
        (8, 64, 64, false, false),
        (8, 64, 65, true, true),
        (32, 32, 32, false, true),
        (2, 2, 8192, false, false),
        (2, 2, 8193, true, true),
        (2, 16, 1024, false, false),
        (2, 16, 1025, true, true),
    ] {
        for small_on in [false, true] {
            let _small = override_tuning(TuningOverride {
                small_class: Some(small_on),
                ..TuningOverride::default()
            });
            for c_pad in [0, 3] {
                let case = Case::<T>::new(m, n, k).c_pad(c_pad);
                let out = check_spec(&case, test);
                assert_eq!(
                    out.accepted,
                    if small_on { accept_small_on } else { accept },
                    "{test}: {m}x{n}x{k} ldc {}, small class on: {small_on}",
                    m + c_pad
                );
            }
        }
    }
    // (k, explicit KC target, fitted KC, accepted).
    for (k, target, kc, accept) in [
        (64, 32, 32, true),   // exactly MIN_KC, two depth blocks
        (65, 32, 24, false),  // fitted below MIN_KC while splitting
        (31, 32, 31, true),   // one block shorter than MIN_KC
        (32, 32, 32, true),   // one block of exactly MIN_KC
        (33, 32, 24, false),  // kc + 1 refits below MIN_KC
        (16, 16, 16, true),   // KC == k < MIN_KC
        (97, 32, 32, true),   // 32, 32, 32 and a single-step block
        (100, 48, 40, true),  // 40, 40, 20
        (100, 33, 32, true),  // 32, 32, 32, 4
        (100, 31, 24, false), // 24 × 4 + 4
        (129, 64, 48, true),  // 48, 48, 33
        (200, 199, 104, true),
        (200, 200, 200, true),
        (200, 201, 200, true),
        (131, 64, 48, true), // 2 · 64 + 3
    ] {
        // 4 NR + 1 columns (not 2 NR + 1) keep k = 16 out of the
        // small class (declined by default) for `f64` too.
        let case = Case::<T>::new(70, 4 * T::NR + 1, k)
            .explicit_kc(target)
            .scalars(-1.0, 1.0)
            .c_pad(k % 2);
        let out = check_spec(&case, test);
        assert_eq!(out.kc, kc, "{test}: k {k} target {target}");
        assert_eq!(out.accepted, accept, "{test}: k {k} target {target}");
    }
}

#[test]
fn test_sme2_gemm_f32_decision_boundaries() {
    decision_boundaries::<f32>("test_sme2_gemm_f32_decision_boundaries");
}

#[test]
fn test_sme2_gemm_f64_decision_boundaries() {
    decision_boundaries::<f64>("test_sme2_gemm_f64_decision_boundaries");
}

/// Rows and columns one below, at and one above a block (MC = 64 and
/// NC = NR through the tuning override; MC = 512 by default), and last
/// blocks of a single row, column and depth step. The number of kernel
/// calls is computed here from the dimensions, not from the driver's MC/NC.
fn block_edges<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let nr = T::NR;
    {
        // KC 32 for k = 97: depth blocks 32, 32, 32, 1.
        let _blocks = override_tuning(TuningOverride {
            mc_target: Some(64),
            b_bytes: Some(32 * core::mem::size_of::<T>() * nr),
            ..TuningOverride::default()
        });
        let mut index = 0;
        for m in [63, 64, 65, 127, 128, 129] {
            for n in [nr - 1, nr, nr + 1, 2 * nr + 1] {
                let (alpha, beta, nan) = SCALARS[index % SCALARS.len()];
                let mut case = Case::<T>::new(m, n, 97)
                    .explicit_kc(32)
                    .scalars(alpha, beta)
                    .trans(TRANSPOSES[index % 3], TRANSPOSES[(index / 3) % 3])
                    .c_pad(index % 2);
                if nan {
                    case = case.nan_c();
                }
                index += 1;
                let out = run_case(&case);
                let label = format!("{test} {case:?}");
                assert_eq!(out.kc, 32, "{label}");
                assert_matches_neon(&out, &label);
                let report = out.report.as_ref().expect("report");
                let blocks: usize = report.blocks.iter().map(|count| count.sme).sum();
                assert_eq!(blocks, 4 * m.div_ceil(64) * n.div_ceil(nr), "{label}");
            }
        }
    }
    // The default MC of 512 rows: one block up to 512, two from 513.
    for m in [511, 512, 513] {
        let case = Case::<T>::new(m, 2 * nr, 40).c_pad(0);
        let out = run_case(&case);
        let label = format!("{test} {case:?}");
        assert_matches_neon(&out, &label);
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.blocks[0].sme, m.div_ceil(512), "{label}");
    }
    // The default 4 MiB B buffer: one column block up to its width, two
    // beyond (KC 256), for a part of two MC blocks (MC = 64 through the
    // tuning, 65 rows: 16 rows would take the few-rows route, and one MC
    // block takes the 2 MiB cap beyond 4 MiB, which `tests_gemm_policy.rs`
    // covers).
    let width = (4 << 20) / (256 * core::mem::size_of::<T>());
    let _two_row_blocks = override_tuning(TuningOverride {
        mc_target: Some(64),
        ..TuningOverride::default()
    });
    for n in [width, width + 1] {
        let case = Case::<T>::new(65, n, 256)
            .explicit_kc(256)
            .scalars(1.0, 0.0);
        let out = run_case(&case);
        let label = format!("{test} wide {n}");
        assert_eq!(out.kc, 256, "{label}");
        assert_matches_neon(&out, &label);
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.layout.parts[0].b_cap, 4 << 20, "{label}");
        assert_eq!(report.blocks[0].sme, 2 * n.div_ceil(width), "{label}");
    }
}

#[test]
fn test_sme2_gemm_f32_block_edges() {
    block_edges::<f32>("test_sme2_gemm_f32_block_edges");
}

#[test]
fn test_sme2_gemm_f64_block_edges() {
    block_edges::<f64>("test_sme2_gemm_f64_block_edges");
}

/// Views: `k == 1` products (one depth step; a strided single row or
/// column normalises, so even "strided" operands are accepted there),
/// every store and transpose of each operand, `ldc == m` and large leading
/// dimensions, and a C block inside a bigger matrix.
fn views<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let stores = [
        Store::ColMajor { pad: 0 },
        Store::ColMajor { pad: 61 },
        Store::RowMajor { pad: 0 },
        Store::RowMajor { pad: 5 },
        Store::Strided,
    ];
    let mut index = 0;
    for a_store in stores {
        for b_store in stores {
            let trans = (TRANSPOSES[index % 3], TRANSPOSES[(index / 3) % 3]);
            index += 1;
            // k == 1: blocked (200 · 200 > 32³), every store normalises.
            let case = Case::<T>::new(200, 200, 1)
                .stores(a_store, b_store)
                .trans(trans.0, trans.1)
                .c_pad(index % 3);
            let out = check_spec(&case, test);
            assert!(out.accepted, "{test}: k == 1 {case:?}");
            // Deeper: a strided operand declines.
            let case = Case::<T>::new(66, 2 * T::NR + 2, 96)
                .stores(a_store, b_store)
                .trans(trans.0, trans.1)
                .explicit_kc(32)
                .c_pad(if index % 2 == 0 { 0 } else { 300 });
            let out = check_spec(&case, test);
            let strided = a_store == Store::Strided || b_store == Store::Strided;
            assert_eq!(out.accepted, !strided, "{test}: {case:?}");
        }
    }
}

#[test]
fn test_sme2_gemm_f32_views() {
    views::<f32>("test_sme2_gemm_f32_views");
}

#[test]
fn test_sme2_gemm_f64_views() {
    views::<f64>("test_sme2_gemm_f64_views");
}

/// The real `f32` threshold, `2^27` multiply-adds: 512 × 512 × 511 stays
/// on one part, 512 × 512 × 512 splits over two units.
#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_f32_parallel_threshold() {
    let test = "test_sme2_gemm_f32_parallel_threshold";
    if !backend_or_skip(test, Need::F32) {
        return;
    }
    let _units = override_units(2);
    for (k, parts) in [(511, 1), (512, 2)] {
        let case = Case::<f32>::new(512, 512, k).par(Par::RayonWith(2));
        let out = run_case(&case);
        assert_matches_neon(&out, &format!("{test} k = {k}"));
        let report = out.report.as_ref().expect("report");
        assert_eq!(report.layout.parts.len(), parts, "k = {k}");
    }
}
