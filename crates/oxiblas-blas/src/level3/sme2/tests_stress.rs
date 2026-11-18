//! Wide and long tests of the SME2 GEMM driver, on the harness of
//! `tests_gemm_harness.rs`:
//!
//! - the size grid of the plan (m in 1..17, 31..33, 47..49, 63..65,
//!   127..129; n in 1..17, 23..25; k in 1, 7..9, 63..65, kc ± 1, 2·kc + 3),
//!   every case checked against the spec's verdict (accepted and equal to
//!   NEON with every block on SME2, or declined with C untouched), the
//!   scalars, transposes, stores and C padding rotating with the case;
//! - special values (infinities, NaNs, subnormals) and operands that sit
//!   next to each other in one allocation;
//! - eight threads calling at once, each with its own test switches;
//! - stress (plan section 8 (d)): thousands of alternations of the SME2
//!   GEMM with NEON GEMV, dot products, NEON GEMM and plain `f64` work,
//!   with eight `f64` values kept live across every SME2 call (in optimised
//!   builds in the callee-saved registers d8 to d15, which leaving streaming
//!   mode would otherwise destroy), `sme_state() == Some((0, 0))` after every
//!   call and
//!   every result compared bitwise every round; and a multi-threaded
//!   variant whose GEMM is itself split over rayon workers.
//!
//! Every test that executes SME2 code goes through `try_gemm` and so starts
//! with the backend gate `backend_or_skip`, once per test. Every accepted
//! call proves that its kernels ran with the kernels' own per-thread count
//! (`kernel_calls` of each part, and the count of the calling thread).

use std::hint::black_box;

use oxiblas_core::parallel::Par;
use oxiblas_matrix::{MatMut, MatRef};

use super::args::same_result;
#[cfg(feature = "parallel")]
use super::detect::override_units;
#[cfg(feature = "parallel")]
use super::driver::BlockCount;
use super::driver::{TuningOverride, override_tuning, try_gemm};
#[cfg(feature = "parallel")]
use super::parallel::mark_not_ready;
use super::parallel::take_last_run;
use super::test_support::{
    Need, backend_or_skip, decisions, force_off, kernel_entries_here, sme_state,
};
use super::tests_gemm_harness::{
    Case, Rng, SCALARS, Special, Store, TRANSPOSES, TestElem, assert_kernel_calls_here,
    assert_matches_neon, check_spec, run_case, stored_shape,
};
#[cfg(feature = "parallel")]
use super::tests_gemm_harness::{assert_runs, assert_same_bits, expected_part};
use crate::level1::dot;
use crate::level2::{GemvTrans, gemv};
use crate::level3::gemm::{BlockingSource, GemmBlocking, call_blocking, gemm_planned};
use crate::level3::gemm_plan::{DEFAULT_POLICY, GemmPlan, GemmRoute};
use crate::level3::trsm::Trans;

// ---------------------------------------------------------------------------
// The size grid
// ---------------------------------------------------------------------------

/// Rows of the grid.
const M_GRID: [usize; 29] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
    127, 128, 129,
];

/// Columns of the grid.
const N_GRID: [usize; 20] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25,
];

/// Operand stores the grid rotates through.
const GRID_STORES: [(Store, Store); 4] = [
    (Store::ColMajor { pad: 0 }, Store::ColMajor { pad: 2 }),
    (Store::RowMajor { pad: 1 }, Store::ColMajor { pad: 0 }),
    (Store::ColMajor { pad: 3 }, Store::RowMajor { pad: 0 }),
    (Store::RowMajor { pad: 0 }, Store::RowMajor { pad: 2 }),
];

/// Case `index` of a grid: the scalars, transposes, stores and C padding
/// rotate with the index (with periods 6, 3 · 3, 4 and 3, staggered so that
/// they combine differently along the grid).
fn grid_case<T: TestElem>(m: usize, n: usize, k: usize, index: usize) -> Case<T> {
    let (alpha, beta, nan) = SCALARS[(index / 9) % SCALARS.len()];
    let (a_store, b_store) = GRID_STORES[(index / 2) % GRID_STORES.len()];
    let case = Case::<T>::new(m, n, k)
        .trans(TRANSPOSES[index % 3], TRANSPOSES[(index / 3) % 3])
        .scalars(alpha, beta)
        .stores(a_store, b_store)
        .c_pad([0, 1, 7][(index / 5) % 3])
        .seed(index as u64 * 7919 + k as u64);
    if nan { case.nan_c() } else { case }
}

/// Runs the grid `m × n × ks` (every pair, or one `n` per `m` when
/// `diagonal`), with the explicit KC target `explicit` on every other case
/// when given, and checks every case against the spec's verdict (one pass
/// of the gate: [`grid_cases`]).
fn grid<T: TestElem>(test: &str, ks: &[usize], explicit: Option<usize>, diagonal: bool) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    grid_cases::<T>(test, ks, explicit, diagonal);
}

/// [`grid`] without the gate (the caller has passed it). Every case is
/// sequential, so its part runs on this thread: the kernels' count of this
/// thread moves by exactly the kernel calls of the accepted cases.
fn grid_cases<T: TestElem>(test: &str, ks: &[usize], explicit: Option<usize>, diagonal: bool) {
    let entries = kernel_entries_here();
    let mut calls = 0;
    let (mut accepted, mut declined, mut index) = (0, 0, 0);
    for (row, &m) in M_GRID.iter().enumerate() {
        for (col, &n) in N_GRID.iter().enumerate() {
            if diagonal && col != (row * 7) % N_GRID.len() {
                continue;
            }
            for &k in ks {
                let mut case = grid_case::<T>(m, n, k, index);
                if let Some(kc) = explicit
                    && index % 2 == 1
                {
                    case = case.explicit_kc(kc);
                }
                index += 1;
                let out = check_spec(&case, test);
                if out.accepted {
                    accepted += 1;
                } else {
                    declined += 1;
                }
                if let Some(report) = &out.report {
                    calls += report
                        .blocks
                        .iter()
                        .map(|blocks| blocks.kernel_calls)
                        .sum::<u64>();
                }
            }
        }
    }
    assert!(
        accepted > 0 && declined > 0,
        "{test}: {accepted} accepted, {declined} declined"
    );
    assert!(calls > 0, "{test}: no kernel ran");
    assert_eq!(
        kernel_entries_here() - entries,
        calls,
        "{test}: kernel entries on this thread"
    );
}

/// The small class switched on for the shallow grids: every product of
/// `M_GRID × N_GRID × {1, 7, 8, 9}` is at most 32³ multiply-adds, so with
/// the small class off by default (`driver::SMALL_CLASS_ON`) the
/// whole grid would decline. With it on, the grid keeps running the shapes
/// of the small rule on SME2 (few rows and blocked), and `spec_accepts`
/// still declines the rest; the other grids run with the default (off).
fn small_class_on() -> super::driver::TuningGuard {
    override_tuning(TuningOverride {
        small_class: Some(true),
        ..TuningOverride::default()
    })
}

#[test]
fn test_sme2_gemm_f32_grid_shallow() {
    let _small = small_class_on();
    grid::<f32>(
        "test_sme2_gemm_f32_grid_shallow",
        &[1, 7, 8, 9],
        None,
        false,
    );
}

#[test]
fn test_sme2_gemm_f64_grid_shallow() {
    let _small = small_class_on();
    grid::<f64>(
        "test_sme2_gemm_f64_grid_shallow",
        &[1, 7, 8, 9],
        None,
        false,
    );
}

#[test]
fn test_sme2_gemm_f32_grid_mid() {
    grid::<f32>(
        "test_sme2_gemm_f32_grid_mid",
        &[63, 64, 65],
        Some(32),
        false,
    );
}

#[test]
fn test_sme2_gemm_f64_grid_mid() {
    grid::<f64>(
        "test_sme2_gemm_f64_grid_mid",
        &[63, 64, 65],
        Some(32),
        false,
    );
}

/// The deep end of the grid, one `n` per `m`: k = kc ± 1, kc and
/// 2·kc + 3 for an explicit KC target of 64 on every other case (fitted
/// to 63, 64, 40 and 48; the default blocking on the others), and for the
/// default KC target of the type (448 for `f64`, 896 for `f32`; from
/// k = 512 on the default blocking adapts KC to the caches, and each case
/// checks the fitted KC it got against the spec).
fn grid_deep<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    grid_cases::<T>(test, &[63, 64, 65, 131], Some(64), true);
    let kc = if core::mem::size_of::<T>() == 8 {
        448
    } else {
        896
    };
    grid_cases::<T>(test, &[kc - 1, kc, kc + 1, 2 * kc + 3], None, true);
}

#[test]
fn test_sme2_gemm_f32_grid_deep() {
    grid_deep::<f32>("test_sme2_gemm_f32_grid_deep");
}

#[test]
fn test_sme2_gemm_f64_grid_deep() {
    grid_deep::<f64>("test_sme2_gemm_f64_grid_deep");
}

// ---------------------------------------------------------------------------
// Special values and adjacent operands
// ---------------------------------------------------------------------------

/// Infinities and NaNs in A, B and C, NaN and infinite alpha and beta, and
/// subnormal data: NaN positions and every other bit match NEON.
fn special_values<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let base = Case::<T>::new(70, 2 * T::NR + 1, 100).explicit_kc(40);
    let mut runs = Vec::new();
    for (alpha, beta, nan) in SCALARS {
        for special in [Special::NonFinite, Special::Subnormal] {
            runs.push((alpha, beta, nan, special));
        }
    }
    for (alpha, beta) in [
        (f64::NAN, 0.5),
        (f64::INFINITY, -0.5),
        (0.5, f64::NAN),
        (1.0, f64::NEG_INFINITY),
        (f64::NAN, 0.0),
    ] {
        runs.push((alpha, beta, false, Special::None));
        runs.push((alpha, beta, false, Special::NonFinite));
    }
    for (alpha, beta, nan, special) in runs {
        let mut case = base.scalars(alpha, beta).special(special);
        if nan {
            case = case.nan_c();
        }
        let out = run_case(&case);
        assert_matches_neon(&out, &format!("{test} {case:?}"));
        if special == Special::NonFinite {
            assert!(
                (0..case.n).any(|j| (0..case.m).any(|i| out.got.at(i, j).to_f64().is_nan())),
                "{test}: no NaN reached C ({case:?})"
            );
        }
    }
}

#[test]
fn test_sme2_gemm_f32_special_values() {
    special_values::<f32>("test_sme2_gemm_f32_special_values");
}

#[test]
fn test_sme2_gemm_f64_special_values() {
    special_values::<f64>("test_sme2_gemm_f64_special_values");
}

/// Operands that alias nothing but sit next to each other in one
/// allocation, with no gap (`ldc == m`), C first, between and last: C is
/// computed as NEON computes it, and A, B and everything around C keep
/// their bits.
fn adjacent_operands<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (70, 3 * T::NR + 1, 75);
    let (a_len, b_len, c_len) = (m * k, k * n, m * n);
    for (order, (trans_a, trans_b)) in [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
    ]
    .into_iter()
    .enumerate()
    {
        let mut rng = Rng::new(order as u64 + 7);
        let mut buf: Vec<T> = (0..a_len + b_len + c_len).map(|_| rng.value()).collect();
        let original = buf.clone();
        // [C | A | B], [A | C | B], [A | B | C].
        let c_at = [0, a_len, a_len + b_len][order];
        let a_at = if order == 0 { c_len } else { 0 };
        let b_at = if order == 2 { a_len } else { a_len + c_len };
        let (ar, ac) = stored_shape(trans_a, m, k);
        let (br, bc) = stored_shape(trans_b, k, n);
        let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
        let source = BlockingSource::Explicit(GemmBlocking {
            mc: 256,
            kc: 32,
            nc: 4096,
        });
        let blocks = call_blocking(source, &plan, m, k, n);
        let mut want = original[c_at..c_at + c_len].to_vec();
        {
            let _neon = force_off();
            gemm_planned(
                trans_a,
                trans_b,
                T::of(-0.5),
                MatRef::from_column_major(&original[a_at..a_at + a_len], ar, ac).expect("A"),
                MatRef::from_column_major(&original[b_at..b_at + b_len], br, bc).expect("B"),
                T::of(0.75),
                MatMut::from_column_major(&mut want, m, n).expect("C"),
                Par::Seq,
                source,
                &plan,
            );
        }
        let (head, rest) = buf.split_at_mut(c_at);
        let (c_slice, tail) = rest.split_at_mut(c_len);
        let (head, tail): (&[T], &[T]) = (head, tail);
        let operand = |at: usize, len: usize| outside_c(head, tail, c_at + c_len, at, len);
        let entries = kernel_entries_here();
        let _ = take_last_run();
        let accepted = try_gemm(
            trans_a,
            trans_b,
            T::of(-0.5),
            &MatRef::from_column_major(operand(a_at, a_len), ar, ac).expect("A"),
            &MatRef::from_column_major(operand(b_at, b_len), br, bc).expect("B"),
            T::of(0.75),
            &mut MatMut::from_column_major(c_slice, m, n).expect("C"),
            &blocks,
            Par::Seq,
        );
        assert!(accepted, "{test}: layout {order} declined");
        assert!(
            assert_kernel_calls_here(entries, &format!("{test}: layout {order}")) > 0,
            "{test}: layout {order}"
        );
        for (index, (got, was)) in buf.iter().zip(&original).enumerate() {
            if (c_at..c_at + c_len).contains(&index) {
                let want = want[index - c_at];
                assert!(
                    same_result(*got, want),
                    "{test}: layout {order}: C element {} = {got:?}, NEON {want:?}",
                    index - c_at
                );
            } else {
                assert_eq!(
                    got.bits(),
                    was.bits(),
                    "{test}: layout {order}: element {index}"
                );
            }
        }
        assert_eq!(sme_state(), Some((0, 0)), "{test}");
    }
}

/// The `len` elements at `at` of a buffer split around C into `head`
/// (before C) and `tail` (from `tail_at` on).
fn outside_c<'s, T>(
    head: &'s [T],
    tail: &'s [T],
    tail_at: usize,
    at: usize,
    len: usize,
) -> &'s [T] {
    if at < tail_at {
        &head[at..at + len]
    } else {
        &tail[at - tail_at..at - tail_at + len]
    }
}

#[test]
fn test_sme2_gemm_f32_adjacent_operands() {
    adjacent_operands::<f32>("test_sme2_gemm_f32_adjacent_operands");
}

#[test]
fn test_sme2_gemm_f64_adjacent_operands() {
    adjacent_operands::<f64>("test_sme2_gemm_f64_adjacent_operands");
}

// ---------------------------------------------------------------------------
// Eight threads at once
// ---------------------------------------------------------------------------

/// `Par::RayonWith(n)` where the build has rayon, `Par::Seq` otherwise.
fn par_with(threads: usize) -> Par {
    #[cfg(feature = "parallel")]
    {
        Par::RayonWith(threads)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = threads;
        Par::Seq
    }
}

/// The work of thread `index` of [`test_sme2_gemm_eight_threads_at_once`]:
/// a few products of its own kind, each checked in full, and the exact
/// number of decisions on this thread. Threads 1 and 2 set test switches
/// (units, tuning, a part marked "not ready"); the others set none and
/// check that they see the defaults (the layout check of
/// `assert_matches_neon` reads the tuning in force on its own thread).
fn thread_work(index: usize) {
    let test = format!("test_sme2_gemm_eight_threads_at_once thread {index}");
    let decided = decisions();
    let mut calls = 0;
    for round in 0..6 {
        let m = 64 + 9 * round + index;
        match index % 4 {
            0 => {
                let case = Case::<f32>::new(m, 40 + index, 70).seed((index * 100 + round) as u64);
                assert_matches_neon(&run_case(&case), &test);
            }
            1 => {
                #[cfg(feature = "parallel")]
                let _switches = (
                    override_units(3),
                    override_tuning(TuningOverride {
                        par_min_macs: Some(0),
                        mc_target: Some(64),
                        ..TuningOverride::default()
                    }),
                );
                let case = Case::<f64>::new(2 * m, 30, 96)
                    .explicit_kc(32)
                    .par(par_with(3))
                    .seed((index * 100 + round) as u64);
                assert_matches_neon(&run_case(&case), &test);
            }
            2 => {
                #[cfg(feature = "parallel")]
                {
                    let _switches = (
                        override_units(2),
                        override_tuning(TuningOverride {
                            par_min_macs: Some(0),
                            ..TuningOverride::default()
                        }),
                        mark_not_ready(0b10),
                    );
                    let case = Case::<f64>::new(2 * m, 20, 90)
                        .trans(Trans::Trans, Trans::Trans)
                        .par(par_with(2))
                        .seed((index * 100 + round) as u64);
                    let out = run_case(&case);
                    assert_same_bits(&out, &test);
                    assert_runs(&out, &test, |part_index, part| {
                        if part_index == 1 {
                            (true, BlockCount::default())
                        } else {
                            (false, expected_part(&out, part))
                        }
                    });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let case = Case::<f64>::new(m, 20, 90).seed((index * 100 + round) as u64);
                    assert_matches_neon(&run_case(&case), &test);
                }
            }
            _ => {
                let case = Case::<f32>::new(m, 3 * 16 + index, 33)
                    .trans(Trans::NoTrans, Trans::Trans)
                    .scalars(0.5, 0.0)
                    .nan_c()
                    .seed((index * 100 + round) as u64);
                assert_matches_neon(&run_case(&case), &test);
            }
        }
        calls += 1;
    }
    assert_eq!(decisions() - decided, calls, "{test}: decisions");
    assert_eq!(sme_state(), Some((0, 0)), "{test}");
}

#[test]
fn test_sme2_gemm_eight_threads_at_once() {
    if !backend_or_skip("test_sme2_gemm_eight_threads_at_once", Need::F64) {
        return;
    }
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|index| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    thread_work(index);
                })
            })
            .collect();
        for handle in handles {
            if let Err(panic) = handle.join() {
                std::panic::resume_unwind(panic);
            }
        }
    });
}

/// Calls from inside rayon pools: on a worker of a one-thread pool (the
/// parts of a split run inline on that worker) and of a three-thread pool
/// (the parts run on its workers), with three units and a split from any
/// work; afterwards every worker of both pools is out of streaming mode
/// with no ZA state.
#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_inside_rayon_pools() {
    let test = "test_sme2_gemm_inside_rayon_pools";
    if !backend_or_skip(test, Need::F64) {
        return;
    }
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        pool.install(|| {
            let _switches = (
                override_units(3),
                override_tuning(TuningOverride {
                    par_min_macs: Some(0),
                    ..TuningOverride::default()
                }),
            );
            for (par, parts) in [(Par::RayonWith(3), 3), (Par::Rayon, threads.min(3))] {
                let case = Case::<f64>::new(200, 40, 96)
                    .explicit_kc(32)
                    .par(par)
                    .seed(threads as u64);
                let out = run_case(&case);
                let label = format!("{test}: pool of {threads}, {par:?}");
                assert_matches_neon(&out, &label);
                let report = out.report.as_ref().expect("report");
                assert_eq!(report.layout.parts.len(), parts, "{label}");
            }
        });
        let states = pool.broadcast(|_| sme_state());
        assert!(
            states.iter().all(|state| *state == Some((0, 0))),
            "{test}: pool of {threads}: {states:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Stress: alternations with NEON and f64 work
// ---------------------------------------------------------------------------

/// Eight `f64` values derived from `seed`, each needed after the call.
#[inline(never)]
fn live_values(seed: f64) -> [f64; 8] {
    let s = black_box(seed);
    let d0 = s * 1.5;
    let d1 = s + 0.25;
    let d2 = s * s;
    let d3 = d0 - d1;
    let d4 = d2.mul_add(0.75, s);
    let d5 = d1 * d3;
    let d6 = d4 - 2.0;
    let d7 = d5 + d6;
    [d0, d1, d2, d3, d4, d5, d6, d7]
}

/// The combination of [`live_values`] that [`call_keeping_f64_live`]
/// returns.
#[inline(never)]
fn combine(d: [f64; 8]) -> f64 {
    ((((d[0] + d[1]) * d[2] - d[3]) + d[4]) * d[5] - d[6]) / d[7]
}

/// Calls `call` (opaque: through `black_box`, so it is a real call) while
/// eight `f64` values are live across it, then combines them. In optimised
/// builds (`--release`, as `tools/sme2_qemu.sh` builds) the values live in
/// the callee-saved registers d8 to d15 (the low halves of v8 to v15),
/// which leaving streaming mode zeroes unless the SME2 frame saved them; the
/// combination shows any change. In a debug build they are kept on the
/// stack across the call, so there the check proves nothing about d8 to
/// d15. Each value is scaled after the call by a factor of one the compiler
/// cannot see, so none of the eight can be folded into a partial result
/// before the call.
#[inline(never)]
fn call_keeping_f64_live(call: &mut dyn FnMut() -> bool, seed: f64) -> (bool, f64) {
    let s = black_box(seed);
    let d0 = s * 1.5;
    let d1 = s + 0.25;
    let d2 = s * s;
    let d3 = d0 - d1;
    let d4 = d2.mul_add(0.75, s);
    let d5 = d1 * d3;
    let d6 = d4 - 2.0;
    let d7 = d5 + d6;
    let call = black_box(call);
    let accepted = call();
    let g = black_box(1.0_f64);
    let d = [
        d0 * g,
        d1 * g,
        d2 * g,
        d3 * g,
        d4 * g,
        d5 * g,
        d6 * g,
        d7 * g,
    ];
    (accepted, combine(d))
}

/// A rounding-sensitive `f64` sum (it changes with the rounding mode or a
/// flush-to-zero setting), after the SME2 calls.
#[inline(never)]
fn f64_work(terms: usize) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..terms {
        let x = black_box(i as f64 + 1.0);
        sum += 1.0 / x + f64::MIN_POSITIVE * 0.25 / x;
    }
    sum
}

/// One round's inputs and expected results for the stress loops.
struct StressData<T> {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<T>,
    b: Vec<T>,
    c: Vec<T>,
    x: Vec<T>,
    z: Vec<T>,
    y: Vec<T>,
    /// The NEON GEMM result `0.75 · A · B - 0.5 · C`.
    c_want: Vec<T>,
    /// The NEON GEMV result `-1 · A · z + 0.5 · y`.
    y_want: Vec<T>,
    /// `dot(x, x)`.
    dot_want: T,
    /// The fitted blocking of the GEMM.
    blocks: crate::level3::gemm::CallBlocking,
    plan: GemmPlan<T>,
}

impl<T: TestElem> StressData<T> {
    /// `m × n × k` (one kernel call per GEMM), data from `seed`.
    fn new(m: usize, n: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut values = |len: usize| -> Vec<T> { (0..len).map(|_| rng.value()).collect() };
        let (a, b, c, x, z, y) = (
            values(m * k),
            values(k * n),
            values(m * n),
            values(1000),
            values(k),
            values(m),
        );
        let plan = GemmPlan::<T>::new(DEFAULT_POLICY).with_forced_route(GemmRoute::Blocked);
        let blocks = call_blocking(BlockingSource::Default, &plan, m, k, n);
        let mut data = Self {
            m,
            n,
            k,
            a,
            b,
            c,
            x,
            z,
            y,
            c_want: Vec::new(),
            y_want: Vec::new(),
            dot_want: T::of(0.0),
            blocks,
            plan,
        };
        data.c_want = data.neon_gemm();
        data.y_want = data.gemv();
        data.dot_want = dot(&data.x, &data.x);
        data
    }

    fn a(&self) -> MatRef<'_, T> {
        MatRef::from_column_major(&self.a, self.m, self.k).expect("A")
    }

    fn b(&self) -> MatRef<'_, T> {
        MatRef::from_column_major(&self.b, self.k, self.n).expect("B")
    }

    /// The NEON blocked GEMM (SME2 forced off on this thread).
    fn neon_gemm(&self) -> Vec<T> {
        let mut c = self.c.clone();
        let _neon = force_off();
        gemm_planned(
            Trans::NoTrans,
            Trans::NoTrans,
            T::of(0.75),
            self.a(),
            self.b(),
            T::of(-0.5),
            MatMut::from_column_major(&mut c, self.m, self.n).expect("C"),
            Par::Seq,
            BlockingSource::Default,
            &self.plan,
        );
        c
    }

    /// The NEON GEMV.
    fn gemv(&self) -> Vec<T> {
        let mut y = self.y.clone();
        gemv(
            GemvTrans::NoTrans,
            T::of(-1.0),
            self.a(),
            &self.z,
            T::of(0.5),
            &mut y,
        );
        y
    }

    /// One SME2 GEMM with `par`, eight `f64` values live across it.
    fn sme2_gemm(&self, par: Par, seed: f64) -> (Vec<T>, bool, f64) {
        let mut c = self.c.clone();
        let (a, b) = (self.a(), self.b());
        let mut view = MatMut::from_column_major(&mut c, self.m, self.n).expect("C");
        let blocks = &self.blocks;
        let mut call = || {
            try_gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                T::of(0.75),
                &a,
                &b,
                T::of(-0.5),
                &mut view,
                blocks,
                par,
            )
        };
        let (accepted, live) = call_keeping_f64_live(&mut call, seed);
        (c, accepted, live)
    }
}

/// Asserts `got` equals `want` element by element (bitwise, NaN payloads
/// aside).
#[track_caller]
fn assert_all_same<T: TestElem>(got: &[T], want: &[T], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            same_result(*got, *want),
            "{label}: element {index}: {got:?} vs {want:?}"
        );
    }
}

/// `rounds` alternations on this thread: SME2 GEMM (with `par`) holding
/// eight live `f64` values, the streaming state, NEON GEMV, a dot product,
/// NEON GEMM and plain `f64` work, every result compared every round.
fn alternate<T: TestElem>(data: &StressData<T>, par: Par, rounds: usize, label: &str) {
    let work_want = f64_work(64);
    let decided = decisions();
    let entries = kernel_entries_here();
    let mut calls = 0;
    for round in 0..rounds {
        let seed = 0.5 + round as f64 / 1024.0;
        let before = kernel_entries_here();
        let _ = take_last_run();
        let (c, accepted, live) = data.sme2_gemm(par, seed);
        assert!(accepted, "{label}: round {round}: declined");
        calls += assert_kernel_calls_here(before, &format!("{label}: round {round}"));
        assert_eq!(
            live.to_bits(),
            combine(live_values(seed)).to_bits(),
            "{label}: round {round}: f64 values live across the call changed"
        );
        assert_eq!(
            sme_state(),
            Some((0, 0)),
            "{label}: round {round}: SME state"
        );
        assert_all_same(
            &c,
            &data.c_want,
            &format!("{label}: round {round}: SME2 GEMM"),
        );
        let y = data.gemv();
        assert_all_same(&y, &data.y_want, &format!("{label}: round {round}: GEMV"));
        let d = dot(&data.x, &data.x);
        assert_eq!(
            d.bits(),
            data.dot_want.bits(),
            "{label}: round {round}: dot"
        );
        if round % 8 == 0 {
            let neon = data.neon_gemm();
            assert_all_same(
                &neon,
                &data.c_want,
                &format!("{label}: round {round}: NEON GEMM"),
            );
        }
        assert_eq!(
            f64_work(64).to_bits(),
            work_want.to_bits(),
            "{label}: round {round}: f64 work"
        );
    }
    assert_eq!(decisions() - decided, rounds as u64, "{label}: decisions");
    if par == Par::Seq {
        // Sequential: every kernel call of the loop ran on this thread.
        assert_eq!(
            kernel_entries_here() - entries,
            calls,
            "{label}: kernel entries on this thread"
        );
    }
}

/// Two kernel entries per GEMM, 64 × 2·NR × 33 (the blocked class for
/// both types: `m · n · k > 32³`): one ZA transposer call that packs B
/// (`op(B) = B`) and one macro-kernel call (A read in place). The depth is
/// 33 because 32 would make the `f64` call the small class, which is
/// declined by default (`driver::SMALL_CLASS_ON`).
fn stress_data<T: TestElem>(seed: u64) -> StressData<T> {
    StressData::<T>::new(64, 2 * T::NR, 33, seed)
}

/// Kernel entries of one GEMM of [`stress_data`]: the B transposer and the
/// macro-kernel.
const STRESS_ENTRIES_PER_GEMM: u64 = 2;

#[test]
fn test_sme2_stress_f32_alternation() {
    if !backend_or_skip("test_sme2_stress_f32_alternation", Need::F32) {
        return;
    }
    let entries = kernel_entries_here();
    alternate(&stress_data::<f32>(1), Par::Seq, 3000, "f32 alternation");
    // Two kernel entries per GEMM (B transposer, macro-kernel), all on this
    // thread.
    assert_eq!(
        kernel_entries_here() - entries,
        3000 * STRESS_ENTRIES_PER_GEMM,
        "f32 alternation"
    );
}

#[test]
fn test_sme2_stress_f64_alternation() {
    if !backend_or_skip("test_sme2_stress_f64_alternation", Need::F64) {
        return;
    }
    let entries = kernel_entries_here();
    alternate(&stress_data::<f64>(2), Par::Seq, 3000, "f64 alternation");
    // Two kernel entries per GEMM (B transposer, macro-kernel), all on this
    // thread.
    assert_eq!(
        kernel_entries_here() - entries,
        3000 * STRESS_ENTRIES_PER_GEMM,
        "f64 alternation"
    );
}

/// Four threads alternate at once (two per type); with rayon, two more run
/// GEMMs split over two units on the rayon workers, and afterwards every
/// rayon worker is out of streaming mode with no ZA state.
#[test]
fn test_sme2_stress_threads() {
    if !backend_or_skip("test_sme2_stress_threads", Need::F64) {
        return;
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for index in 0..4_u64 {
            handles.push(scope.spawn(move || {
                let label = format!("thread {index}");
                if index % 2 == 0 {
                    alternate(&stress_data::<f32>(10 + index), Par::Seq, 600, &label);
                } else {
                    alternate(&stress_data::<f64>(10 + index), Par::Seq, 600, &label);
                }
            }));
        }
        #[cfg(feature = "parallel")]
        for index in 4..6_u64 {
            handles.push(scope.spawn(move || {
                let _switches = (
                    override_units(2),
                    override_tuning(TuningOverride {
                        par_min_macs: Some(0),
                        ..TuningOverride::default()
                    }),
                );
                let label = format!("split thread {index}");
                let data = StressData::<f64>::new(130, 24, 40, 10 + index);
                alternate(&data, Par::RayonWith(2), 300, &label);
            }));
        }
        for handle in handles {
            if let Err(panic) = handle.join() {
                std::panic::resume_unwind(panic);
            }
        }
    });
    #[cfg(feature = "parallel")]
    {
        let states = rayon::broadcast(|_| sme_state());
        assert!(
            states.iter().all(|state| *state == Some((0, 0))),
            "rayon workers: {states:?}"
        );
    }
}
