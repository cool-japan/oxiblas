//! Tests of the SME2 GEMV (`sme2.rs`), part 2: the tests that execute SME2
//! code (under QEMU here; they skip without SME2), on the harness of
//! `sme2_tests.rs`.
//!
//! - **Grid**: `try_gemv` against the NEON kernels bit for bit for both
//!   transposes and types over `m` in [`M_LIST`] and `n` in [`N_LIST`] as
//!   far as the crossover accepts them, alpha in {1, -1, 0.5, 1/3}, `y`
//!   pre-filled with plain values, NaN and `-0`, `lda` padded by 0–2 rows of
//!   signalling NaN; `Trans` calls of several kernel calls (more than 1024
//!   columns).
//! - **Special values** in A and `x` (NaN, infinities, subnormals, `-0`,
//!   the largest finite value), NaN and infinite alpha.
//! - **Views**: large `lda`, a sub-matrix inside signalling-NaN borders, `x`
//!   and `y` as sub-slices with signalling-NaN neighbours.
//! - **Parts**: units overridden to 2 and 3, `Par::Rayon` and
//!   `RayonWith(2, 3, 7)`, the 8 MiB threshold itself, parts on rayon
//!   workers; per-part records with exact kernel counts.
//! - **Threads**: a pool of three workers with the rendezvous, chosen
//!   workers "not ready" through the per-thread flag
//!   (`detect::set_pretend_not_ready`): only the caller ready, the caller
//!   and one more, all ready; a calling thread that is not ready declines;
//!   eight threads calling at once.
//! - **Kernels directly**: the FPSR frame and the SME state after a call,
//!   the self-check of the real kernels.
//! - **Edges** (at the end of the file): the raw
//!   `Trans` lane partials against the definition for every residue of `m`,
//!   including columns whose every product underflows to `-0` (a dead
//!   tail lane must keep a `-0` lane); the same through `try_gemv`; special
//!   values for every residue of `m` and every last column block; every
//!   row-block and column-block boundary one below, at and one above;
//!   A, `x` and `y` side by side in one allocation between signalling NaNs;
//!   the `Trans` scratch poisoned with signalling NaN before every call
//!   (also on pool workers), so a partial that is not stored shows.
//!
//! Every test of an accepted call checks exact kernel-entry counts on the
//! thread that ran each part (see `sme2_tests.rs`).

#[cfg(feature = "parallel")]
use std::marker::PhantomData;

use oxiblas_core::parallel::Par;

use super::GemvTrans;
use super::sme2::{
    CHECK_NOTRANS, CHECK_TRANS, GemvArgs, GemvKernelFn, Switches, TRANS_COLS_PER_CALL, crossover,
    override_switches, reference_lanes, scalar_start, self_check, self_check_with,
    sme2_gemv_notrans_f32, sme2_gemv_notrans_f64, sme2_gemv_trans_f32, sme2_gemv_trans_f64,
    take_last_run, try_gemv,
};
#[cfg(feature = "parallel")]
use super::sme2_tests::assert_parts;
use super::sme2_tests::{
    Data, GvElem, Layout, PretendHere, Problem, Rng, YFill, assert_declined, assert_same_bits,
    assert_sequential, last_column, run_problem, tail_lanes,
};
use crate::level3::sme2::args::{SmeElement, same_result};
use crate::level3::sme2::detect::override_units;
#[cfg(feature = "parallel")]
use crate::level3::sme2::detect::{pretends_not_ready, set_pretend_not_ready};
use crate::level3::sme2::test_support::{
    FPSR_DZC, FPSR_IXC, Need, backend_or_skip, decisions, fpsr_read, fpsr_write, hardware_or_skip,
    kernel_entries_here, sme_state,
};

/// Stored row counts of the grid.
const M_LIST: [usize; 19] = [
    64, 65, 71, 79, 80, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 1100, 2049,
];

/// Stored column counts of the grid (`n = 1` only reaches `NoTrans`).
const N_LIST: [usize; 17] = [
    1, 2, 3, 4, 5, 15, 16, 17, 31, 33, 63, 64, 65, 255, 257, 1025, 1030,
];

/// The alphas of the grid.
const ALPHAS: [f64; 4] = [1.0, -1.0, 0.5, 1.0 / 3.0];

/// The `y` fills of the grid.
const FILLS: [YFill; 3] = [YFill::Plain, YFill::Nan, YFill::NegZero];

/// The grid for one type and transpose: every accepted `(m, n)`, each with
/// the next alpha and `y` fill in turn (so every alpha meets every fill),
/// sequential, with one kernel call per `NoTrans` call and one per started
/// 1024 columns for `Trans`, all on this thread.
fn grid<T: GvElem>(test: &str, trans: GemvTrans) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let transposed = trans != GemvTrans::NoTrans;
    let mut rng = Rng::new(3 + u64::from(transposed));
    let mut seen = [[false; 3]; 4];
    let mut cases = 0;
    for &m in &M_LIST {
        for &n in &N_LIST {
            if !crossover(transposed, m, n) {
                continue;
            }
            let (a, f) = (cases % 4, (cases / 4) % 3);
            seen[a][f] = true;
            let layout = Layout {
                lda_pad: cases % 3,
                ..Layout::TIGHT
            };
            let p = Problem::new(
                trans,
                m,
                n,
                T::of(ALPHAS[a]),
                layout,
                Data::Plain,
                FILLS[f],
                &mut rng,
            );
            let out = run_problem(&p, Par::Seq);
            let label = format!("{test}: {m}x{n} alpha {} {:?}", ALPHAS[a], FILLS[f]);
            assert_sequential(&p, &out, &label);
            cases += 1;
        }
    }
    assert!(seen.iter().flatten().all(|s| *s), "{test}: alpha x fill");
    println!("{test}: {cases} shapes");
}

#[test]
fn test_sme2_gemv_f32_notrans_grid() {
    grid::<f32>("test_sme2_gemv_f32_notrans_grid", GemvTrans::NoTrans);
}

#[test]
fn test_sme2_gemv_f32_trans_grid() {
    grid::<f32>("test_sme2_gemv_f32_trans_grid", GemvTrans::Trans);
}

#[test]
fn test_sme2_gemv_f64_notrans_grid() {
    grid::<f64>("test_sme2_gemv_f64_notrans_grid", GemvTrans::NoTrans);
}

#[test]
fn test_sme2_gemv_f64_trans_grid() {
    grid::<f64>("test_sme2_gemv_f64_trans_grid", GemvTrans::ConjTrans);
}

/// `NoTrans` with few columns, which the grid cannot reach (the crossover
/// needs `m · n >= 128²`): `n` from 1 to 7 (no full column group up to 3,
/// one plus every remainder above), 16 and more row blocks with a partial
/// last group, every alpha and `y` fill.
fn notrans_narrow<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(71);
    let mut index = 0;
    for m in [16384, 16385, 16399] {
        for n in [1, 2, 3, 4, 5, 7] {
            assert!(crossover(false, m, n), "{test}: {m}x{n}");
            let (a, f) = (index % 4, (index / 4) % 3);
            index += 1;
            let p = Problem::new(
                GemvTrans::NoTrans,
                m,
                n,
                T::of(ALPHAS[a]),
                Layout::TIGHT,
                Data::Plain,
                FILLS[f],
                &mut rng,
            );
            let out = run_problem(&p, Par::Seq);
            let label = format!("{test}: {m}x{n} alpha {} {:?}", ALPHAS[a], FILLS[f]);
            assert_sequential(&p, &out, &label);
        }
    }
    assert_eq!(index, 18, "{test}: shapes");
}

#[test]
fn test_sme2_gemv_f32_notrans_narrow() {
    notrans_narrow::<f32>("test_sme2_gemv_f32_notrans_narrow");
}

#[test]
fn test_sme2_gemv_f64_notrans_narrow() {
    notrans_narrow::<f64>("test_sme2_gemv_f64_notrans_narrow");
}

/// `Trans` calls of more than one kernel call (more than 1024 columns), and
/// column counts around the 64-column block, with every vector-tail case.
fn trans_many_calls<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(11);
    for (m, n) in [
        (64, 1024),
        (79, 1025),
        (69, 2048),
        (100, 2049),
        (90, 3000),
        (86, 191),
        (94, 193),
    ] {
        let p = Problem::new(
            GemvTrans::Trans,
            m,
            n,
            T::of(-0.75),
            Layout::TIGHT,
            Data::Plain,
            YFill::Plain,
            &mut rng,
        );
        let out = run_problem(&p, Par::Seq);
        assert_sequential(&p, &out, &format!("{test}: {m}x{n}"));
    }
}

#[test]
fn test_sme2_gemv_f32_trans_many_calls() {
    trans_many_calls::<f32>("test_sme2_gemv_f32_trans_many_calls");
}

#[test]
fn test_sme2_gemv_f64_trans_many_calls() {
    trans_many_calls::<f64>("test_sme2_gemv_f64_trans_many_calls");
}

/// Special values in A and `x`, NaN and infinite alpha, every `y` fill.
fn special_values<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(23);
    let alphas = [
        T::of(1.0),
        T::of(-1.0 / 3.0),
        T::qnan(),
        T::inf(),
        T::tiny(),
    ];
    let mut index = 0;
    for (m, n) in [(130, 130), (77, 300), (1100, 19), (2049, 20), (64, 1030)] {
        for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
            for fill in FILLS {
                let alpha = alphas[index % alphas.len()];
                index += 1;
                let p = Problem::new(
                    trans,
                    m,
                    n,
                    alpha,
                    Layout::TIGHT,
                    Data::Special,
                    fill,
                    &mut rng,
                );
                let out = run_problem(&p, Par::Seq);
                let label = format!("{test}: {trans:?} {m}x{n} alpha {alpha:?} {fill:?}");
                assert_sequential(&p, &out, &label);
            }
        }
    }
}

#[test]
fn test_sme2_gemv_f32_special_values() {
    special_values::<f32>("test_sme2_gemv_f32_special_values");
}

#[test]
fn test_sme2_gemv_f64_special_values() {
    special_values::<f64>("test_sme2_gemv_f64_special_values");
}

/// Large `lda` (A a sub-matrix inside signalling-NaN rows and columns),
/// `x` and `y` as sub-slices with signalling-NaN neighbours: the result
/// equals NEON and every border element keeps its bits.
fn views<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(31);
    for (m, n) in [(100, 300), (1030, 20), (65, 257), (513, 33)] {
        for lda_pad in [37, m + 1] {
            for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
                let layout = Layout {
                    lda_pad,
                    a_border: 3 * m + 5,
                    x_border: 9,
                    y_border: 17,
                };
                let p = Problem::new(
                    trans,
                    m,
                    n,
                    T::of(0.5),
                    layout,
                    Data::Plain,
                    YFill::Plain,
                    &mut rng,
                );
                let out = run_problem(&p, Par::Seq);
                let label = format!("{test}: {trans:?} {m}x{n} lda {}", p.lda);
                assert_sequential(&p, &out, &label);
            }
        }
    }
}

#[test]
fn test_sme2_gemv_f32_views() {
    views::<f32>("test_sme2_gemv_f32_views");
}

#[test]
fn test_sme2_gemv_f64_views() {
    views::<f64>("test_sme2_gemv_f64_views");
}

// ---------------------------------------------------------------------------
// Parts
// ---------------------------------------------------------------------------

/// The shapes of the parallel tests: rows and columns split into whole
/// vector steps and column groups, short last pieces, more units than
/// steps.
#[cfg(feature = "parallel")]
const PART_SHAPES: [(GemvTrans, usize, usize); 7] = [
    (GemvTrans::NoTrans, 1100, 300),
    (GemvTrans::Trans, 300, 1100),
    (GemvTrans::NoTrans, 80, 300),
    (GemvTrans::Trans, 600, 30),
    (GemvTrans::NoTrans, 2049, 20),
    (GemvTrans::ConjTrans, 64, 2100),
    (GemvTrans::Trans, 70, 257),
];

/// Units 2 and 3, `Par::Rayon` and `RayonWith(2, 3, 7)`, every work split
/// (threshold 0): the parts follow the spec, every part runs on SME2 with
/// its exact kernel count on its own thread, the result is the NEON one;
/// a call from outside any pool runs every part on a rayon worker. Then
/// the 8 MiB threshold itself, and `Par::Seq`.
#[cfg(feature = "parallel")]
fn parallel<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    assert_eq!(rayon::current_thread_index(), None, "{test}: inside a pool");
    let me = std::thread::current().id();
    let mut rng = Rng::new(43);
    {
        let _split = override_switches(Switches {
            par_min_bytes: Some(0),
            rendezvous: false,
        });
        for units in [2, 3] {
            let _units = override_units(units);
            for par in [
                Par::Rayon,
                Par::RayonWith(2),
                Par::RayonWith(3),
                Par::RayonWith(7),
                Par::Seq,
            ] {
                let parts = if par.is_sequential() {
                    1
                } else {
                    units.min(par.num_threads())
                };
                for (trans, m, n) in PART_SHAPES {
                    let p = Problem::new(
                        trans,
                        m,
                        n,
                        T::of(1.0 / 3.0),
                        Layout::TIGHT,
                        Data::Plain,
                        YFill::Plain,
                        &mut rng,
                    );
                    let out = run_problem(&p, par);
                    let label = format!("{test}: units {units} {par:?} {trans:?} {m}x{n}");
                    assert_same_bits(&p, &out, &label);
                    assert_parts(&p, &out, parts, |_, _| true, &label);
                    let run = out.run.as_ref().expect("run report");
                    if run.parts.len() > 1 {
                        assert!(
                            run.parts
                                .iter()
                                .all(|part| part.worker.is_some() && part.thread != me),
                            "{label}: a part ran outside rayon: {:?}",
                            run.parts
                        );
                        assert_eq!(out.entries, 0, "{label}: kernels on the caller");
                    }
                }
            }
        }
    }
    // The threshold itself: 8 MiB splits, one column less does not.
    let _units = override_units(3);
    let rows = (8 << 20) / (1024 * core::mem::size_of::<T>());
    for (n, parts) in [(1024, 3), (1023, 1)] {
        for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
            let p = Problem::new(
                trans,
                rows,
                n,
                T::of(-1.0),
                Layout::TIGHT,
                Data::Plain,
                YFill::Plain,
                &mut rng,
            );
            let out = run_problem(&p, Par::RayonWith(3));
            let label = format!("{test}: threshold {trans:?} {rows}x{n}");
            assert_same_bits(&p, &out, &label);
            assert_parts(&p, &out, parts, |_, _| true, &label);
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f32_parallel() {
    parallel::<f32>("test_sme2_gemv_f32_parallel");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f64_parallel() {
    parallel::<f64>("test_sme2_gemv_f64_parallel");
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

/// Workers of the dedicated pool, and parts of each call made in it.
#[cfg(feature = "parallel")]
const WORKERS: usize = 3;

/// Clears the "pretend not ready" flag of every worker of a pool when
/// dropped, so that a failing assertion cannot leave a worker pretending.
#[cfg(feature = "parallel")]
struct ClearFlags<'p> {
    /// The pool whose workers are cleared.
    pool: &'p rayon::ThreadPool,
    /// Keeps the guard on its thread.
    _not_send: PhantomData<*const ()>,
}

#[cfg(feature = "parallel")]
impl<'p> ClearFlags<'p> {
    /// The guard of `pool`.
    fn new(pool: &'p rayon::ThreadPool) -> Self {
        Self {
            pool,
            _not_send: PhantomData,
        }
    }
}

#[cfg(feature = "parallel")]
impl Drop for ClearFlags<'_> {
    fn drop(&mut self) {
        self.pool.broadcast(|_| {
            set_pretend_not_ready(false);
        });
    }
}

/// A rayon pool of exactly `threads` threads.
#[cfg(feature = "parallel")]
fn pool_of(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool")
}

/// The flag of every worker of `pool`, by worker index.
#[cfg(feature = "parallel")]
fn flags(pool: &rayon::ThreadPool) -> Vec<bool> {
    pool.broadcast(|_| pretends_not_ready())
}

/// The calls of the per-part tests: a row and a column split into three
/// parts each.
#[cfg(feature = "parallel")]
const THREAD_SHAPES: [(GemvTrans, usize, usize); 3] = [
    (GemvTrans::NoTrans, 300, 200),
    (GemvTrans::Trans, 200, 300),
    (GemvTrans::ConjTrans, 90, 1100),
];

/// The per-part readiness tests: inside a pool of three workers, the
/// calling worker and `others_ready` of the other workers ready, the rest
/// pretending not to be (their own per-thread flag); three units,
/// `RayonWith(3)`, every work split, the rendezvous on, so the three parts
/// run on the three workers whatever the scheduler does. A part runs on
/// NEON exactly when its own worker pretends, with no kernel entry on its
/// thread; an SME2 part shows its exact kernel count on its thread; the
/// result is the NEON one (the same bits in every case). A driver that
/// checked readiness once on the calling thread and handed the verdict to
/// the parts fails the first two on every run.
#[cfg(feature = "parallel")]
fn ready_workers<T: GvElem>(test: &str, others_ready: usize) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let pool = pool_of(WORKERS);
    let clear = ClearFlags::new(&pool);
    let mut rng = Rng::new(51 + others_ready as u64);
    for (trans, m, n) in THREAD_SHAPES {
        let p = Problem::new(
            trans,
            m,
            n,
            T::of(0.5),
            Layout::TIGHT,
            Data::Plain,
            YFill::Plain,
            &mut rng,
        );
        let (out, caller, flagged) = pool.install(|| {
            let caller = rayon::current_thread_index().expect("install runs on a worker");
            let flagged: Vec<usize> = (0..WORKERS)
                .filter(|&worker| worker != caller)
                .skip(others_ready)
                .collect();
            rayon::broadcast(|ctx| {
                set_pretend_not_ready(flagged.contains(&ctx.index()));
            });
            let _switches = (
                override_units(WORKERS),
                override_switches(Switches {
                    par_min_bytes: Some(0),
                    rendezvous: true,
                }),
            );
            (run_problem(&p, Par::RayonWith(WORKERS)), caller, flagged)
        });
        let label = format!("{test}: {trans:?} {m}x{n}, caller {caller}, pretending {flagged:?}");
        assert_eq!(flagged.len(), WORKERS - 1 - others_ready, "{label}");
        assert_same_bits(&p, &out, &label);
        let run = out.run.as_ref().expect("run report");
        assert_eq!(run.parts.len(), WORKERS, "{label}: parts");
        for (index, part) in run.parts.iter().enumerate() {
            assert!(
                !run.parts[..index]
                    .iter()
                    .any(|other| other.thread == part.thread),
                "{label}: parts share a thread: {:?}",
                run.parts
            );
        }
        let mut workers: Vec<Option<usize>> = run.parts.iter().map(|part| part.worker).collect();
        workers.sort_unstable();
        assert_eq!(
            workers,
            (0..WORKERS).map(Some).collect::<Vec<_>>(),
            "{label}: worker indices"
        );
        assert_parts(
            &p,
            &out,
            WORKERS,
            |_, part| !part.worker.is_some_and(|worker| flagged.contains(&worker)),
            &label,
        );
        assert_eq!(
            run.parts.iter().filter(|part| part.sme2).count(),
            WORKERS - flagged.len(),
            "{label}: SME2 parts"
        );
    }
    drop(clear);
    assert_eq!(flags(&pool), vec![false; WORKERS], "{test}: flags left set");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f32_only_caller_ready() {
    ready_workers::<f32>("test_sme2_gemv_f32_only_caller_ready", 0);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f64_only_caller_ready() {
    ready_workers::<f64>("test_sme2_gemv_f64_only_caller_ready", 0);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f32_caller_and_one_ready() {
    ready_workers::<f32>("test_sme2_gemv_f32_caller_and_one_ready", 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f64_caller_and_one_ready() {
    ready_workers::<f64>("test_sme2_gemv_f64_caller_and_one_ready", 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f32_all_ready() {
    ready_workers::<f32>("test_sme2_gemv_f32_all_ready", WORKERS - 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f64_all_ready() {
    ready_workers::<f64>("test_sme2_gemv_f64_all_ready", WORKERS - 1);
}

/// The parallel threshold of [`caller_not_ready`]: its shapes lie on both
/// sides of it.
const CALLER_PAR_MIN_BYTES: usize = 1 << 18;

/// A calling thread that is not ready declines the call, after a control
/// call from a ready thread that is accepted: `false`, no decision, no run
/// report, no kernel entry, `y` bitwise untouched; below and above the
/// parallel threshold, both transposes, from a plain thread and from a
/// pool worker whose fellow workers are ready.
fn caller_not_ready<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let _units = override_units(3);
    let _switches = override_switches(Switches {
        par_min_bytes: Some(CALLER_PAR_MIN_BYTES),
        rendezvous: false,
    });
    #[cfg(feature = "parallel")]
    let par = Par::RayonWith(3);
    #[cfg(not(feature = "parallel"))]
    let par = Par::Seq;
    let mut rng = Rng::new(61);
    let mut problems = Vec::new();
    let mut sides = [false; 2];
    for (m, n) in [(130, 140), (300, 400)] {
        sides[usize::from(m * n * core::mem::size_of::<T>() >= CALLER_PAR_MIN_BYTES)] = true;
        for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
            let p = Problem::new(
                trans,
                m,
                n,
                T::of(0.25),
                Layout::TIGHT,
                Data::Plain,
                YFill::Plain,
                &mut rng,
            );
            let control = run_problem(&p, par);
            assert_same_bits(&p, &control, &format!("{test}: control {trans:?} {m}x{n}"));
            let parts = control.run.as_ref().map_or(0, |run| run.parts.len());
            assert!(parts >= 1, "{test}: control parts");
            let _pretend = PretendHere::new();
            let out = run_problem(&p, par);
            assert_declined(&out, &format!("{test}: plain thread {trans:?} {m}x{n}"));
            problems.push(p);
        }
    }
    assert_eq!(sides, [true, true], "{test}: both sides of the threshold");
    #[cfg(feature = "parallel")]
    {
        let pool = pool_of(WORKERS);
        let clear = ClearFlags::new(&pool);
        pool.broadcast(|_| {
            set_pretend_not_ready(false);
        });
        pool.install(|| {
            let _units = override_units(3);
            let _switches = override_switches(Switches {
                par_min_bytes: Some(CALLER_PAR_MIN_BYTES),
                rendezvous: false,
            });
            let _pretend = PretendHere::new();
            for p in &problems {
                let out = run_problem(p, par);
                assert_declined(
                    &out,
                    &format!("{test}: pool worker {:?} {}x{}", p.trans, p.m, p.n),
                );
            }
        });
        drop(clear);
        assert_eq!(flags(&pool), vec![false; WORKERS], "{test}: flags left set");
    }
}

#[test]
fn test_sme2_gemv_f32_caller_not_ready() {
    caller_not_ready::<f32>("test_sme2_gemv_f32_caller_not_ready");
}

#[test]
fn test_sme2_gemv_f64_caller_not_ready() {
    caller_not_ready::<f64>("test_sme2_gemv_f64_caller_not_ready");
}

/// The calls of one thread of [`test_sme2_gemv_eight_threads_at_once`].
fn thread_calls<T: GvElem>(seed: u64, label: &str) {
    let mut rng = Rng::new(seed);
    for round in 0..3 {
        for (trans, m, n) in [
            (GemvTrans::NoTrans, 130 + round, 140),
            (GemvTrans::Trans, 140, 130 + round),
            (GemvTrans::NoTrans, 1030, 17 + round),
        ] {
            let p = Problem::new(
                trans,
                m,
                n,
                T::of(-0.5),
                Layout::TIGHT,
                Data::Plain,
                YFill::Plain,
                &mut rng,
            );
            let out = run_problem(&p, Par::Seq);
            assert_sequential(
                &p,
                &out,
                &format!("{label}: round {round} {trans:?} {m}x{n}"),
            );
        }
    }
}

/// Eight threads calling `try_gemv` at once, `f32` and `f64` alternating:
/// every call gives the NEON bits with its exact kernel count on its own
/// thread.
#[test]
fn test_sme2_gemv_eight_threads_at_once() {
    let test = "test_sme2_gemv_eight_threads_at_once";
    if !backend_or_skip(test, Need::F64) {
        return;
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8u64)
            .map(|index| {
                scope.spawn(move || {
                    let label = format!("{test}: thread {index}");
                    if index % 2 == 0 {
                        thread_calls::<f32>(100 + index, &label);
                    } else {
                        thread_calls::<f64>(100 + index, &label);
                    }
                })
            })
            .collect();
        for handle in handles {
            assert!(handle.join().is_ok(), "{test}: a thread failed");
        }
    });
}

// ---------------------------------------------------------------------------
// Kernels directly
// ---------------------------------------------------------------------------

/// Runs `kernel` on `args` with FPSR set to `before` and returns FPSR after
/// it, restoring the thread's FPSR; asserts one kernel entry and the SME
/// state after the call.
///
/// # Safety
///
/// `args` satisfies the kernel's contract and the thread is ready.
unsafe fn fpsr_after<T: SmeElement>(
    kernel: GemvKernelFn<T>,
    args: &GemvArgs<T>,
    before: u64,
) -> u64 {
    let saved = fpsr_read();
    let entries = kernel_entries_here();
    fpsr_write(before);
    // SAFETY: forwarded from this function's contract.
    unsafe { kernel(args) };
    let after = fpsr_read();
    fpsr_write(saved);
    assert_eq!(kernel_entries_here() - entries, 1, "kernel entries");
    assert_eq!(sme_state(), Some((0, 0)), "SME state after the kernel");
    after
}

/// The FPSR frame on the real kernels: flags set before a call survive it;
/// a `NoTrans` call with an exact `alpha · x` adds nothing, one with an
/// inexact one adds exactly IXC (the SVE `fmul`); a `Trans` call adds
/// nothing (its arithmetic is ZA-array FMLA only).
fn fpsr_frame<T: GvElem>(test: &str, notrans: GemvKernelFn<T>, trans: GemvKernelFn<T>) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    let (m, n) = (100, 7);
    let a: Vec<T> = (0..m * n).map(|i| T::of((i % 7) as f64 - 3.0)).collect();
    for (alpha, flags) in [(0.5, 0), (1.0 / 3.0, FPSR_IXC)] {
        // 0.5 · 3 is exact, (1/3) · 3 is not.
        let x = vec![T::of(3.0); n];
        let mut y = vec![T::of(2.0); m];
        let args = GemvArgs {
            a: a.as_ptr(),
            lda: m,
            x: x.as_ptr(),
            out: y.as_mut_ptr(),
            m,
            n,
            alpha_bits: T::of(alpha).bits(),
        };
        // SAFETY: the gate found the hardware; A is m x n with lda = m, x
        // has n and y has m elements.
        let after = unsafe { fpsr_after(notrans, &args, FPSR_DZC) };
        assert_eq!(
            after,
            FPSR_DZC | flags,
            "{test}: NoTrans alpha {alpha}: {after:#x}"
        );
    }
    let x = vec![T::of(1.0 / 3.0); m];
    let mut partials = vec![T::ZERO; n * T::LANES];
    let args = GemvArgs {
        a: a.as_ptr(),
        lda: m,
        x: x.as_ptr(),
        out: partials.as_mut_ptr(),
        m,
        n,
        alpha_bits: 0,
    };
    // SAFETY: as above; the partials buffer holds n · LANES elements.
    let after = unsafe { fpsr_after(trans, &args, FPSR_DZC) };
    assert_eq!(after, FPSR_DZC, "{test}: Trans: {after:#x}");
    // SAFETY: as above.
    let after = unsafe { fpsr_after(trans, &args, 0) };
    assert_eq!(after, 0, "{test}: Trans from clear flags: {after:#x}");
}

#[test]
fn test_sme2_gemv_f32_fpsr_frame() {
    fpsr_frame::<f32>(
        "test_sme2_gemv_f32_fpsr_frame",
        sme2_gemv_notrans_f32,
        sme2_gemv_trans_f32,
    );
}

#[test]
fn test_sme2_gemv_f64_fpsr_frame() {
    fpsr_frame::<f64>(
        "test_sme2_gemv_f64_fpsr_frame",
        sme2_gemv_notrans_f64,
        sme2_gemv_trans_f64,
    );
}

/// The self-check passes on the real kernels (with and without `f64`),
/// with one kernel entry per checked shape and kernel, leaves the SME state
/// clear, and fails when one real kernel is replaced by a wrong stand-in.
#[test]
fn test_sme2_gemv_self_check_passes() {
    let test = "test_sme2_gemv_self_check_passes";
    if !hardware_or_skip(test, Need::F64) {
        return;
    }
    let saved = fpsr_read();
    let shapes = (CHECK_NOTRANS.len() + CHECK_TRANS.len()) as u64;
    for (f64_present, calls) in [(true, 2 * shapes), (false, shapes)] {
        let entries = kernel_entries_here();
        // SAFETY: the gate found SME2 with FEAT_SME_F64F64 on this thread.
        let passed = unsafe { self_check(f64_present) };
        assert!(passed, "{test}: f64 {f64_present}");
        assert_eq!(
            kernel_entries_here() - entries,
            calls,
            "{test}: kernel calls"
        );
        assert_eq!(sme_state(), Some((0, 0)), "{test}: SME state");
    }
    let real32: (GemvKernelFn<f32>, GemvKernelFn<f32>) =
        (sme2_gemv_notrans_f32, sme2_gemv_trans_f32);
    let real64: (GemvKernelFn<f64>, GemvKernelFn<f64>) =
        (sme2_gemv_notrans_f64, sme2_gemv_trans_f64);
    // SAFETY: as above; the stand-ins are plain Rust that keep to the
    // self-check's buffers.
    unsafe {
        assert!(
            self_check_with(real32, Some(real64)),
            "{test}: real kernels"
        );
        assert!(
            !self_check_with(real32, Some((last_column::<f64>, real64.1))),
            "{test}: wrong f64 NoTrans stand-in accepted"
        );
        assert!(
            !self_check_with((real32.0, tail_lanes::<f32>), Some(real64)),
            "{test}: wrong f32 Trans stand-in accepted"
        );
    }
    fpsr_write(saved);
}

// ---------------------------------------------------------------------------
// Edges
// ---------------------------------------------------------------------------

/// Fills this thread's `Trans` scratch buffer with the signalling NaN, so a
/// lane partial that the kernel does not store reaches `y` as a NaN instead
/// of a stale partial that may happen to be right.
fn poison_scratch<T: GvElem>() {
    T::with_scratch(TRANS_COLS_PER_CALL * T::LANES, |scratch| {
        scratch.fill(T::snan());
    });
}

/// The power of two whose square underflows to zero: `2^-80` (`f32`),
/// `2^-600` (`f64`). `fma(-u, u, ±0)` is `-0`.
fn underflow_unit<T: GvElem>() -> T {
    let exponent = if core::mem::size_of::<T>() == 4 {
        -80
    } else {
        -600
    };
    T::of(2.0f64.powi(exponent))
}

/// `-0`.
fn neg_zero<T: GvElem>() -> T {
    T::ZERO * T::of(-1.0)
}

/// Whether `got` has the bits of `want`, or both are NaN.
fn same_lane<T: SmeElement>(got: T, want: T) -> bool {
    got.bits() == want.bits() || (got.nan() && want.nan())
}

/// The `Trans` kernel on its own, for every residue of `m` (64..=95) and
/// several column counts (one, a partial and a full group, a full block
/// and a partial second one): every lane partial equals the definition
/// (`reference_lanes`) bit for bit (NaN positions), every partial is
/// written (the buffer starts as signalling NaN), nothing beyond them, A
/// and `x` unchanged, one kernel entry, SME state clear. Three data sets:
/// columns whose every product underflows to `-0` (every lane of the
/// definition is `-0`, also the lanes that the vector-tail step does not
/// own: a dead lane must add an exact `-0`), special values, plain values.
fn trans_lanes<T: GvElem>(test: &str, kernel: GemvKernelFn<T>) {
    if !hardware_or_skip(test, T::NEED) {
        return;
    }
    const GUARD: usize = 8;
    let lanes = T::LANES;
    let u = underflow_unit::<T>();
    let mut rng = Rng::new(97);
    let mut underflow_columns = 0;
    for m in 64..=95 {
        for n in [1, 3, 4, 5, 64, 66] {
            let lda = m + 1 + n % 2;
            for mode in 0..3 {
                let mut x = vec![T::snan(); m + 2 * GUARD];
                for value in &mut x[GUARD..GUARD + m] {
                    *value = match mode {
                        0 => u,
                        1 => rng.special(),
                        _ => rng.plain(),
                    };
                }
                let mut a = vec![T::snan(); lda * n + 2 * GUARD];
                for j in 0..n {
                    for i in 0..m {
                        a[GUARD + i + j * lda] = match (mode, j % 2) {
                            (0, 0) => T::ZERO - u,
                            (0, _) | (1, _) => rng.special(),
                            _ => rng.plain(),
                        };
                    }
                }
                let (a_before, x_before) = (a.clone(), x.clone());
                let mut out = vec![T::snan(); n * lanes + 2 * GUARD];
                let args = GemvArgs {
                    a: a[GUARD..].as_ptr(),
                    lda,
                    x: x[GUARD..].as_ptr(),
                    out: out[GUARD..].as_mut_ptr(),
                    m,
                    n,
                    alpha_bits: 0,
                };
                let entries = kernel_entries_here();
                // SAFETY: the gate found the hardware on this thread; A
                // holds `lda · n >= lda · (n - 1) + m` elements after the
                // guard, x has m, the partials buffer n · lanes, m >= 16,
                // and the three buffers are distinct.
                unsafe { kernel(&args) };
                let label = format!("{test}: {m}x{n} lda {lda} mode {mode}");
                assert_eq!(kernel_entries_here() - entries, 1, "{label}: entries");
                assert_eq!(sme_state(), Some((0, 0)), "{label}: SME state");
                assert!(
                    super::sme2_tests::identical(&a, &a_before)
                        && super::sme2_tests::identical(&x, &x_before),
                    "{label}: A or x changed"
                );
                let snan = T::snan().bits();
                assert!(
                    out[..GUARD]
                        .iter()
                        .chain(&out[GUARD + n * lanes..])
                        .all(|g| g.bits() == snan),
                    "{label}: a partial was written outside 0..n·lanes"
                );
                for j in 0..n {
                    let col = &a[GUARD + j * lda..GUARD + j * lda + m];
                    let want = reference_lanes(col, &x[GUARD..GUARD + m], lanes);
                    if mode == 0 && j % 2 == 0 {
                        assert!(
                            want.iter().all(|w| w.bits() == neg_zero::<T>().bits()),
                            "{label}: the underflow column is not all -0"
                        );
                        underflow_columns += 1;
                    }
                    let got = &out[GUARD + j * lanes..GUARD + (j + 1) * lanes];
                    for (l, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            same_lane(*g, *w),
                            "{label}: column {j} lane {l}: {g:?} ({:#x}), definition {w:?} ({:#x})",
                            g.bits(),
                            w.bits()
                        );
                    }
                }
            }
        }
    }
    assert!(underflow_columns > 1000, "{test}: {underflow_columns}");
}

#[test]
fn test_sme2_gemv_f32_trans_lanes_every_residue() {
    trans_lanes::<f32>(
        "test_sme2_gemv_f32_trans_lanes_every_residue",
        sme2_gemv_trans_f32,
    );
}

#[test]
fn test_sme2_gemv_f64_trans_lanes_every_residue() {
    trans_lanes::<f64>(
        "test_sme2_gemv_f64_trans_lanes_every_residue",
        sme2_gemv_trans_f64,
    );
}

/// `Trans` through `try_gemv` on columns whose dot product is `-0` only
/// because every product of the body and the vector tail underflows to
/// `-0` and every scalar-tail product is an exact `-0`: NEON keeps `-0` in
/// every lane, so `y = -0 + alpha · s` is `-0` for `alpha = 1` and `+0` for
/// `alpha = -1`. That is checked first on the NEON kernels (natively too,
/// so the data cannot be vacuous); then the SME2 call must give the same
/// bits, for every residue of `m`.
fn trans_negative_zero<T: GvElem>(test: &str) {
    let u = underflow_unit::<T>();
    let n = 256;
    let mut rng = Rng::new(113);
    let mut problems = Vec::new();
    for m in 64..=79 {
        let start = scalar_start(m, T::LANES);
        for alpha in [1.0, -1.0] {
            let mut p = Problem::new(
                GemvTrans::Trans,
                m,
                n,
                T::of(alpha),
                Layout::TIGHT,
                Data::Plain,
                YFill::NegZero,
                &mut rng,
            );
            for j in 0..n {
                for i in 0..m {
                    p.a[p.a_off + i + j * p.lda] = if i < start { T::ZERO - u } else { neg_zero() };
                }
            }
            for i in 0..m {
                p.x[p.x_off + i] = if i < start { u } else { T::ONE };
            }
            let want = if alpha > 0.0 {
                neg_zero::<T>()
            } else {
                T::ZERO
            };
            let neon = p.neon();
            assert!(
                neon[p.y_off..p.y_off + n]
                    .iter()
                    .all(|v| v.bits() == want.bits()),
                "{test}: m {m} alpha {alpha}: the NEON result is not {want:?}"
            );
            problems.push(p);
        }
    }
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for p in &problems {
        poison_scratch::<T>();
        let out = run_problem(p, Par::Seq);
        assert_sequential(p, &out, &format!("{test}: m {} alpha {:?}", p.m, p.alpha));
    }
}

#[test]
fn test_sme2_gemv_f32_trans_negative_zero_lanes() {
    trans_negative_zero::<f32>("test_sme2_gemv_f32_trans_negative_zero_lanes");
}

#[test]
fn test_sme2_gemv_f64_trans_negative_zero_lanes() {
    trans_negative_zero::<f64>("test_sme2_gemv_f64_trans_negative_zero_lanes");
}

/// Special values in A and `x` for every residue of `m` (64..=95, and two
/// multi-block row counts), sparse ([`Rng::sparse`]: most results stay
/// finite, so a wrong product shows as a number) and dense ([`Rng::special`]),
/// both transposes, with column counts that end
/// in a full block, a single column, three columns and a full last group,
/// every alpha (NaN and infinite included) with every `y` fill, `lda`
/// padded by 0–2 signalling NaNs, the scratch poisoned before every call.
fn special_every_residue<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(83);
    let alphas = [
        T::of(1.0),
        T::of(-1.0 / 3.0),
        T::qnan(),
        T::inf(),
        T::tiny(),
        T::of(0.5),
    ];
    let mut index = 0;
    let mut seen = [[false; 3]; 6];
    for m in (64..=95).chain([1029, 1043]) {
        for trans in [GemvTrans::NoTrans, GemvTrans::Trans] {
            let ns: [usize; 4] = if m > 1000 {
                [16, 17, 19, 20]
            } else {
                [256, 257, 259, 300]
            };
            for n in ns {
                assert!(crossover(trans != GemvTrans::NoTrans, m, n), "{test}");
                let (a, f) = (index % 6, (index / 6) % 3);
                seen[a][f] = true;
                let layout = Layout {
                    lda_pad: index % 3,
                    ..Layout::TIGHT
                };
                let data = if index % 2 == 0 {
                    Data::Sparse
                } else {
                    Data::Special
                };
                index += 1;
                let p = Problem::new(trans, m, n, alphas[a], layout, data, FILLS[f], &mut rng);
                poison_scratch::<T>();
                let out = run_problem(&p, Par::Seq);
                let label = format!(
                    "{test}: {trans:?} {m}x{n} {data:?} alpha {:?} {:?}",
                    alphas[a], FILLS[f]
                );
                assert_sequential(&p, &out, &label);
            }
        }
    }
    assert!(seen.iter().flatten().all(|s| *s), "{test}: alpha x fill");
}

#[test]
fn test_sme2_gemv_f32_special_every_residue() {
    special_every_residue::<f32>("test_sme2_gemv_f32_special_every_residue");
}

#[test]
fn test_sme2_gemv_f64_special_every_residue() {
    special_every_residue::<f64>("test_sme2_gemv_f64_special_every_residue");
}

/// Every boundary one below, at and one above: the `NoTrans` row block in
/// ZA (`64 · LANES` rows: 1024 `f32`, 512 `f64`) and twice it, with an odd
/// and an even column count; the `Trans` column block (64), twice it, and
/// the 1024-column kernel call and twice it, each with the smallest row
/// count the crossover accepts plus a varying residue. Plain, sparse and
/// dense special data rotate; the scratch is poisoned before every call.
fn block_boundaries<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(101);
    let block = 64 * T::LANES;
    let mut shapes = Vec::new();
    for base in [block, 2 * block] {
        for m in [base - 1, base, base + 1] {
            for n in [33, 36] {
                shapes.push((GemvTrans::NoTrans, m, n));
            }
        }
    }
    for (index, n) in [
        63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2047, 2048, 2049,
    ]
    .into_iter()
    .enumerate()
    {
        let m = (128 * 128usize).div_ceil(n).max(64) + index % 16;
        shapes.push((GemvTrans::Trans, m, n));
    }
    for (index, (trans, m, n)) in shapes.into_iter().enumerate() {
        assert!(crossover(trans != GemvTrans::NoTrans, m, n), "{test}");
        let data = [Data::Plain, Data::Sparse, Data::Special][(index / 3) % 3];
        let p = Problem::new(
            trans,
            m,
            n,
            T::of(ALPHAS[index % 4]),
            Layout {
                lda_pad: index % 3,
                ..Layout::TIGHT
            },
            data,
            FILLS[index % 3],
            &mut rng,
        );
        poison_scratch::<T>();
        let out = run_problem(&p, Par::Seq);
        assert_sequential(&p, &out, &format!("{test}: {trans:?} {m}x{n} {data:?}"));
    }
}

#[test]
fn test_sme2_gemv_f32_block_boundaries() {
    block_boundaries::<f32>("test_sme2_gemv_f32_block_boundaries");
}

#[test]
fn test_sme2_gemv_f64_block_boundaries() {
    block_boundaries::<f64>("test_sme2_gemv_f64_block_boundaries");
}

/// A, `x` and `y` of one call as disjoint pieces of ONE allocation, each
/// between signalling NaNs and next to the others (`[g A g x g y g]` or
/// `[g y g x g A g]`, A ending at its last element, `lda > m`): the call
/// gives the NEON bits in `y`, every other element of the allocation keeps
/// its bits, one decision, the exact kernel calls on this thread.
fn one_allocation<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let mut rng = Rng::new(89);
    let g = 5;
    for (trans, m, n, y_first) in [
        (GemvTrans::NoTrans, 65, 257, false),
        (GemvTrans::NoTrans, 1025, 17, true),
        (GemvTrans::Trans, 77, 1025, false),
        (GemvTrans::Trans, 64, 256, true),
        (GemvTrans::ConjTrans, 1030, 19, true),
        (GemvTrans::NoTrans, 513, 35, false),
    ] {
        let lda = m + 2;
        let transposed = trans != GemvTrans::NoTrans;
        let (x_len, y_len) = if transposed { (m, n) } else { (n, m) };
        let a_len = lda * (n - 1) + m;
        let total = a_len + x_len + y_len + 4 * g;
        let (a_at, x_at, y_at) = if y_first {
            (3 * g + y_len + x_len, 2 * g + y_len, g)
        } else {
            (g, 2 * g + a_len, 3 * g + a_len + x_len)
        };
        let mut buf = vec![T::snan(); total];
        for j in 0..n {
            for i in 0..m {
                buf[a_at + i + j * lda] = rng.special();
            }
        }
        for value in &mut buf[x_at..x_at + x_len] {
            *value = rng.special();
        }
        for value in &mut buf[y_at..y_at + y_len] {
            *value = rng.plain();
        }
        let alpha = T::of(-0.75);
        let before = buf.clone();
        let mut want = buf.clone();
        {
            let (a, x, y) = pieces(&mut want, (a_at, a_len), (x_at, x_len), (y_at, y_len));
            let view = oxiblas_matrix::MatRef::from_strided(a, m, n, lda, 1).expect("A view");
            if transposed {
                T::neon_trans(alpha, view, x, y);
            } else {
                T::neon_notrans(alpha, view, x, y);
            }
        }
        poison_scratch::<T>();
        let _ = take_last_run();
        let (decided, entries) = (decisions(), kernel_entries_here());
        let accepted = {
            let (a, x, y) = pieces(&mut buf, (a_at, a_len), (x_at, x_len), (y_at, y_len));
            let view = oxiblas_matrix::MatRef::from_strided(a, m, n, lda, 1).expect("A view");
            try_gemv(trans, alpha, view, x, y, Par::Seq)
        };
        let (decided, entries) = (decisions() - decided, kernel_entries_here() - entries);
        let label = format!("{test}: {trans:?} {m}x{n} y first {y_first}");
        assert!(accepted, "{label}: declined");
        assert_eq!(decided, 1, "{label}: decisions");
        let calls = if transposed {
            n.div_ceil(1024) as u64
        } else {
            1
        };
        assert_eq!(entries, calls, "{label}: kernel calls on this thread");
        let run = take_last_run().expect("run report");
        assert_eq!(run.parts.len(), 1, "{label}: parts");
        assert!(
            run.parts[0].sme2 && run.parts[0].kernel_calls == calls,
            "{label}: {run:?}"
        );
        assert_eq!(sme_state(), Some((0, 0)), "{label}: SME state");
        for (k, ((got, w), b)) in buf.iter().zip(&want).zip(&before).enumerate() {
            if (y_at..y_at + y_len).contains(&k) {
                assert!(
                    same_result(*got, *w),
                    "{label}: y[{}] = {got:?}, NEON {w:?}",
                    k - y_at
                );
            } else {
                assert_eq!(
                    got.bits(),
                    b.bits(),
                    "{label}: element {k} outside y changed"
                );
            }
        }
    }
}

/// `(A, x, y)` of [`one_allocation`]: `y` is the only mutable piece; the
/// pieces must not overlap (`(start, len)` each).
fn pieces<T>(
    buf: &mut [T],
    a: (usize, usize),
    x: (usize, usize),
    y: (usize, usize),
) -> (&[T], &[T], &mut [T]) {
    if y.0 >= a.0 + a.1 && y.0 >= x.0 + x.1 {
        let (head, tail) = buf.split_at_mut(y.0);
        let head: &[T] = head;
        (
            &head[a.0..a.0 + a.1],
            &head[x.0..x.0 + x.1],
            &mut tail[..y.1],
        )
    } else {
        assert!(a.0 >= y.0 + y.1 && x.0 >= y.0 + y.1, "pieces overlap");
        let (head, tail) = buf.split_at_mut(y.0 + y.1);
        let tail: &[T] = tail;
        let off = y.0 + y.1;
        (
            &tail[a.0 - off..a.0 - off + a.1],
            &tail[x.0 - off..x.0 - off + x.1],
            &mut head[y.0..],
        )
    }
}

#[test]
fn test_sme2_gemv_f32_operands_in_one_allocation() {
    one_allocation::<f32>("test_sme2_gemv_f32_operands_in_one_allocation");
}

#[test]
fn test_sme2_gemv_f64_operands_in_one_allocation() {
    one_allocation::<f64>("test_sme2_gemv_f64_operands_in_one_allocation");
}

/// Parallel calls whose workers' scratch buffers were poisoned with
/// signalling NaN first (through `broadcast`), three parts on the three
/// workers of a pool (rendezvous), special values: the NEON bits, every
/// part on SME2 with its exact kernel count on its own thread.
#[cfg(feature = "parallel")]
fn parallel_poisoned<T: GvElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let pool = pool_of(WORKERS);
    let mut rng = Rng::new(107);
    for (trans, m, n) in [
        (GemvTrans::Trans, 77, 1030),
        (GemvTrans::ConjTrans, 64, 2100),
        (GemvTrans::Trans, 1029, 37),
        (GemvTrans::NoTrans, 1029, 37),
    ] {
        let p = Problem::new(
            trans,
            m,
            n,
            T::of(-0.75),
            Layout::TIGHT,
            Data::Special,
            YFill::Plain,
            &mut rng,
        );
        let out = pool.install(|| {
            rayon::broadcast(|_| poison_scratch::<T>());
            let _switches = (
                override_units(WORKERS),
                override_switches(Switches {
                    par_min_bytes: Some(0),
                    rendezvous: true,
                }),
            );
            run_problem(&p, Par::RayonWith(WORKERS))
        });
        let label = format!("{test}: {trans:?} {m}x{n}");
        assert_same_bits(&p, &out, &label);
        assert_parts(&p, &out, WORKERS, |_, _| true, &label);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f32_parallel_poisoned_scratch() {
    parallel_poisoned::<f32>("test_sme2_gemv_f32_parallel_poisoned_scratch");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemv_f64_parallel_poisoned_scratch() {
    parallel_poisoned::<f64>("test_sme2_gemv_f64_parallel_poisoned_scratch");
}
