//! Tests of the driver policy (`driver.rs`, "The policy"): every
//! branch reached and proven by the per-part policy record of the run
//! report, every accepted call bit for bit the NEON blocked result.
//!
//! - **Pure functions** (run everywhere): the constants, the A-source rule
//!   ([`driver::a_source`], [`driver::in_place_fits`]), the reuse rule
//!   ([`driver::a_reuse_fits`]), the B cap ([`driver::b_cap`]), the route
//!   ([`driver::route_for`]), the small class and its prefetch switches, the
//!   layout of every route and A mode with the packing-cache limit, the
//!   argument-block setters against the builders of `args.rs`, and the
//!   harness's spec of the layout against the driver's, each at the real
//!   thresholds: one below, at and one above 2 MiB, 4 MiB, the few-rows
//!   limit and 32³.
//! - **Through `try_gemm`** (SME2 code: `backend_or_skip`): shapes that
//!   reach each branch, at the real thresholds where a call of tolerable
//!   size can (the 2 MiB in-place span, the 4 MiB reuse limit, the 4 MiB B
//!   cap, the few-rows limit, 32³) and through the tuning override
//!   elsewhere; every call goes through `assert_matches_neon`, which checks
//!   the bits against the NEON blocked route, the layout against the
//!   harness's spec and the policy record of every part against the spec's
//!   own walk over the blocks (`tests_gemm_harness.rs`), and the tests add
//!   the branch they are about explicitly (a count of that branch above
//!   zero), so a test cannot pass by reaching another branch. Transposed
//!   and padded (row-major and column-major) operands, every part count
//!   (one to three units), wide guards around C, stale packing buffers,
//!   special values, and the readiness rule (every SME2 entry of a part
//!   after that part's own check; one check in the decision).
//!
//! Every counter is read after the gate call of the test, on the test's
//! own thread (the first-use checks of the detection may run kernels on
//! the first gated call of a process).

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::Par;

use super::args::{PREFETCH_C, PREFETCH_OFF};
#[cfg(feature = "parallel")]
use super::detect::override_units;
use super::detect::set_ready_countdown;
use super::driver::{
    A_REUSE_BYTES, AMode, ASource, B_BUFFER_BYTES, B_ONE_BLOCK_BYTES, BlockCount, Class,
    IN_PLACE_BYTES, MC_TARGET, PolicyCount, Route, SMALL_CLASS_ON, SMALL_MACS, Shape, Tuning,
    TuningOverride, a_reuse_fits, a_source, b_cap, classify, in_place_fits, override_tuning,
    route_for,
};
use super::pack::Span;
#[cfg(feature = "parallel")]
use super::parallel::mark_not_ready;
use super::parallel::{Layout, Split};
use super::test_support::{Need, backend_or_skip, kernel_entries_here};
use super::tests_gemm_harness::{
    Case, Outcome, SCALARS, Special, Store, TestElem, assert_declined, assert_matches_neon,
    assert_runs, assert_same_bits, run_case, run_case_around, spec_accepts, spec_part_of,
};
use crate::level3::gemm_packing::{
    PackBuffer, PackSlot, drain_pack_cache, pack_cache_allocations, set_pack_cache_limit,
};
use crate::level3::trsm::Trans;

/// Sentinels around C in the wide-guard cases (a store that runs a vector
/// or a panel too far changes one of them instead of corrupting the heap).
const WIDE: usize = 256;

/// The policy records of all parts of `out`, summed.
fn policy_sum<T>(out: &Outcome<T>) -> PolicyCount {
    let report = out.report.as_ref().expect("run report");
    let mut sum = PolicyCount::default();
    for blocks in &report.blocks {
        let p = blocks.policy;
        sum.a_in_place += p.a_in_place;
        sum.a_online_fill += p.a_online_fill;
        sum.a_online_rebuilt += p.a_online_rebuilt;
        sum.a_reused += p.a_reused;
        sum.a_za_reads += p.a_za_reads;
        sum.a_za += p.a_za;
        sum.b_za += p.b_za;
        sum.b_plain += p.b_plain;
        sum.prefetch += p.prefetch;
        sum.few_rows += p.few_rows;
    }
    sum
}

/// The SME2 calls that compute C over all parts of `out`.
fn sme_calls<T>(out: &Outcome<T>) -> usize {
    out.report
        .as_ref()
        .expect("run report")
        .blocks
        .iter()
        .map(|blocks| blocks.sme)
        .sum()
}

/// A readiness countdown on the calling thread while it lives (the next
/// `calls` calls of `thread_ready()` answer as usual, later ones `false`).
struct Countdown {
    /// The countdown to restore.
    previous: Option<u32>,
    /// Keeps the guard on its thread.
    _not_send: core::marker::PhantomData<*const ()>,
}

impl Countdown {
    /// Starts a countdown of `calls` calls.
    fn new(calls: u32) -> Self {
        Self {
            previous: set_ready_countdown(Some(calls)),
            _not_send: core::marker::PhantomData,
        }
    }
}

impl Drop for Countdown {
    fn drop(&mut self) {
        set_ready_countdown(self.previous);
    }
}

/// Empties this thread's packing caches, then parks A and B buffers of `T`
/// of exactly `a_len` / `b_len` elements filled with the signalling-NaN
/// sentinel, so that a call that reads a buffer element it did not write
/// in the same call sees a value that changes its bits. (Emptied first: a
/// cached buffer only grows, and buffers above the packing-cache limit are
/// not parked.)
fn stale_caches<T: TestElem>(a_len: usize, b_len: usize) {
    drain_pack_cache();
    for (slot, len) in [(PackSlot::A, a_len), (PackSlot::B, b_len)] {
        let mut buf = PackBuffer::<T>::take(slot, len);
        buf.fill(T::SENTINEL);
    }
}

/// Runs `case` with this thread's packing caches made stale just before
/// the SME2 call (after the NEON reference, which uses the same caches),
/// and asserts that the call allocated no packing buffer: the stale buffers
/// (at least `a_len` / `b_len` elements, parked within the packing-cache
/// limit) are the ones it used.
fn run_stale<T: TestElem>(case: &Case<T>, a_len: usize, b_len: usize) -> Outcome<T> {
    run_case_around(case, |call| {
        stale_caches::<T>(a_len, b_len);
        let allocations = pack_cache_allocations();
        let accepted = call();
        assert_eq!(
            pack_cache_allocations(),
            allocations,
            "{case:?}: the call allocated a packing buffer instead of using the stale ones"
        );
        accepted
    })
}

// ---------------------------------------------------------------------------
// Pure functions (run everywhere)
// ---------------------------------------------------------------------------

#[test]
fn test_sme2_gemm_policy_constants() {
    assert_eq!(IN_PLACE_BYTES, 2 << 20);
    assert_eq!(A_REUSE_BYTES, 4 << 20);
    assert_eq!(B_BUFFER_BYTES, 4 << 20);
    assert_eq!(B_ONE_BLOCK_BYTES, 2 << 20);
    assert_eq!(MC_TARGET, 512);
    assert_eq!(SMALL_MACS, 32 * 32 * 32);
    // The small class is off by default, through the tuning a call reads.
    let tuning = Tuning::of::<f32>();
    assert!(!tuning.small_class, "small class on by default");
    assert_eq!(tuning.small_class, SMALL_CLASS_ON);
    assert_eq!(
        (
            tuning.in_place_bytes,
            tuning.a_reuse_bytes,
            tuning.b_bytes,
            tuning.b_one_block_bytes
        ),
        (2 << 20, 4 << 20, 4 << 20, 2 << 20)
    );
    assert_eq!(
        tuning.pack_limit,
        crate::level3::gemm_packing::PACK_CACHE_MAX_BYTES
    );
    {
        let _limit = set_pack_cache_limit(12345);
        assert_eq!(Tuning::of::<f64>().pack_limit, 12345, "the limit in force");
    }
}

#[test]
fn test_sme2_gemm_policy_route_at_max_rows() {
    for (m, route) in [
        (2, Route::FewRows { copied: false }),
        (47, Route::FewRows { copied: false }),
        (48, Route::FewRows { copied: false }),
        (49, Route::Blocked),
    ] {
        assert_eq!(route_for::<f32>(m), route, "f32 m = {m}");
    }
    for (m, route) in [
        (2, Route::FewRows { copied: false }),
        (31, Route::FewRows { copied: false }),
        (32, Route::FewRows { copied: false }),
        (33, Route::Blocked),
    ] {
        assert_eq!(route_for::<f64>(m), route, "f64 m = {m}");
    }
}

/// The in-place span `((depth - 1) · lda + rows) · elem` one element below,
/// at and one above 2 MiB, for both element sizes, and the A source of a
/// block around it.
#[test]
fn test_sme2_gemm_policy_in_place_threshold() {
    // f64: 2 MiB = 262144 elements = 63 · 4160 + 64.
    for (rows, fits) in [(63, true), (64, true), (65, false)] {
        assert_eq!(
            in_place_fits(4160, rows, 64, 8, IN_PLACE_BYTES),
            fits,
            "f64 rows {rows}"
        );
        let want = if fits {
            ASource::InPlace
        } else {
            ASource::Online
        };
        assert_eq!(
            a_source(Trans::NoTrans, 4160, rows, 64, 8, IN_PLACE_BYTES),
            want
        );
        // A transposed A always takes the ZA transposer, whatever its span.
        assert_eq!(
            a_source(Trans::Trans, 4160, rows, 64, 8, IN_PLACE_BYTES),
            ASource::Za
        );
        assert_eq!(
            a_source(Trans::ConjTrans, 1, 1, 1, 8, IN_PLACE_BYTES),
            ASource::Za
        );
    }
    // f32: 2 MiB = 524288 elements = 128 · 4095 + 128.
    for (rows, fits) in [(127, true), (128, true), (129, false)] {
        assert_eq!(
            in_place_fits(4095, rows, 129, 4, IN_PLACE_BYTES),
            fits,
            "f32 rows {rows}"
        );
    }
    // The depth counts from the first step: one depth step spans `rows`.
    assert!(in_place_fits(usize::MAX, 64, 1, 8, IN_PLACE_BYTES));
    // Overflow is "does not fit", never a wrap into a small span.
    assert!(!in_place_fits(usize::MAX, 64, 3, 8, IN_PLACE_BYTES));
    assert!(!in_place_fits(1, 1, 0, 8, IN_PLACE_BYTES));
}

/// The reuse limit `round_up(rows, 64) · kc · elem <= 4 MiB` one panel
/// below, at and one above.
#[test]
fn test_sme2_gemm_policy_reuse_threshold() {
    // f64 at kc = 2048: 256 rows are 4 MiB.
    for (rows, fits) in [(192, true), (255, true), (256, true), (257, false)] {
        assert_eq!(
            a_reuse_fits(rows, 2048, 8, A_REUSE_BYTES),
            fits,
            "f64 rows {rows}"
        );
    }
    // f32 at kc = 2048: 512 rows are 4 MiB.
    for (rows, fits) in [(448, true), (512, true), (513, false)] {
        assert_eq!(
            a_reuse_fits(rows, 2048, 4, A_REUSE_BYTES),
            fits,
            "f32 rows {rows}"
        );
    }
    assert!(!a_reuse_fits(usize::MAX, 2, 8, A_REUSE_BYTES));
}

/// The B cap one NR panel below, at and one above 4 MiB of packed B in a
/// depth block, for a part of one MC block (2 MiB beyond) and of two
/// (always 4 MiB).
#[test]
fn test_sme2_gemm_policy_b_cap_threshold() {
    let tuning = Tuning::of::<f64>();
    // f64, kc = 512: 1024 columns are 4 MiB.
    for (cols, one_block_cap) in [(1016, 4 << 20), (1024, 4 << 20), (1025, 2 << 20)] {
        assert_eq!(
            b_cap::<f64>(40, 512, cols, 512, &tuning),
            Some(one_block_cap),
            "f64 one MC block, {cols} columns"
        );
        assert_eq!(
            b_cap::<f64>(513, 512, cols, 512, &tuning),
            Some(4 << 20),
            "f64 two MC blocks, {cols} columns"
        );
        let layout =
            Layout::new::<f64>(&Shape::blocked(40, cols, 512), 1, &tuning).expect("layout");
        let part = layout.parts[0];
        assert_eq!(part.b_cap, one_block_cap, "f64 layout {cols}");
        let blocks = if one_block_cap == 4 << 20 { 1 } else { 3 };
        assert_eq!(
            cols.div_ceil(part.nc),
            blocks,
            "f64 layout {cols}: column blocks"
        );
    }
    // f32, kc = 512: 2048 columns are 4 MiB (NR = 16).
    let tuning = Tuning::of::<f32>();
    for (cols, cap) in [(2032, 4 << 20), (2048, 4 << 20), (2049, 2 << 20)] {
        assert_eq!(b_cap::<f32>(49, 512, cols, 512, &tuning), Some(cap));
        assert_eq!(b_cap::<f32>(600, 512, cols, 512, &tuning), Some(4 << 20));
    }
    // Overflowing sizes give no cap (the layout then declines).
    assert_eq!(b_cap::<f64>(1, 64, usize::MAX, 2, &tuning), None);
    // The one-block cap never exceeds a (test-lowered) general cap.
    let low = Tuning {
        b_bytes: 1 << 20,
        ..tuning
    };
    assert_eq!(b_cap::<f32>(64, 64, 4096, 512, &low), Some(1 << 20));
}

/// The small class with the switch off and on: the classes, their prefetch
/// switches, and the harness's spec of the decision.
#[test]
fn test_sme2_gemm_policy_small_class_switch() {
    for (m, n, k, class) in [
        (32, 32, 32, Class::Small),
        (31, 32, 32, Class::Small),
        (33, 32, 32, Class::Blocked),
        (32, 32, 33, Class::Blocked),
        (64, 16, 32, Class::Small),
        (16, 16, 16, Class::SmallDeclined),
    ] {
        assert_eq!(classify(m, n, k), class, "{m}x{n}x{k}");
    }
    assert!(!Class::Small.accepted(false) && Class::Small.accepted(true));
    assert_eq!(Class::Blocked.prefetch(), PREFETCH_C);
    assert_eq!(Class::Small.prefetch(), PREFETCH_OFF);
    for small_on in [false, true] {
        let _small = override_tuning(TuningOverride {
            small_class: Some(small_on),
            ..TuningOverride::default()
        });
        for (m, n, k, small) in [(32, 32, 32, true), (64, 16, 32, true), (33, 32, 32, false)] {
            let case = Case::<f32>::new(m, n, k);
            assert_eq!(
                spec_accepts(&case, k),
                !small || small_on,
                "{m}x{n}x{k} small class {small_on}"
            );
        }
    }
}

/// The layout of every route and A mode, with the packing-cache limit.
#[test]
fn test_sme2_gemm_policy_layout_modes() {
    let tuning = Tuning::of::<f64>();
    let no_trans = |m: usize, n: usize, kc: usize, lda: usize| Shape {
        m,
        n,
        kc,
        a_trans: Trans::NoTrans,
        lda,
        route: Route::Blocked,
    };
    // In place: no A buffer.
    let layout = Layout::new::<f64>(&no_trans(300, 40, 64, 300), 1, &tuning).expect("layout");
    assert_eq!(
        (layout.parts[0].a_mode, layout.parts[0].a_len),
        (AMode::InPlace, 0)
    );
    // Online, reused across jc: a copy of all rows (two row blocks).
    let layout = Layout::new::<f64>(&no_trans(600, 40, 512, 600), 1, &tuning).expect("layout");
    let part = layout.parts[0];
    assert_eq!(part.mc, 320);
    assert_eq!((part.a_mode, part.a_len), (AMode::OnlineReuse, 640 * 512));
    // Online, rebuilt: the copy of all rows (1088 · 512 · 8 > 4 MiB).
    let layout = Layout::new::<f64>(&no_trans(1030, 40, 512, 1030), 1, &tuning).expect("layout");
    let part = layout.parts[0];
    assert_eq!(part.a_mode, AMode::OnlineRebuild);
    assert_eq!(part.a_len, part.mc.div_ceil(64) * 64 * 512);
    // The reused copy fits the reuse limit but not the packing cache next
    // to B: rebuilt (the plain buffers still fit).
    {
        let _limit = set_pack_cache_limit(5 << 19);
        let tight = Tuning::of::<f64>();
        let layout = Layout::new::<f64>(&no_trans(600, 40, 512, 600), 1, &tight).expect("layout");
        let part = layout.parts[0];
        assert_eq!(part.a_mode, AMode::OnlineRebuild);
        assert!((part.a_len + part.b_len) * 8 <= 5 << 19);
        // Not even the plain buffers fit: the layout (and so the call)
        // declines.
        let _limit = set_pack_cache_limit(1 << 20);
        let tighter = Tuning::of::<f64>();
        assert!(Layout::new::<f64>(&no_trans(600, 40, 512, 600), 1, &tighter).is_none());
        // ... but an in-place part needs only B.
        let layout = Layout::new::<f64>(&no_trans(300, 40, 64, 300), 1, &tighter).expect("layout");
        assert_eq!(layout.parts[0].a_len, 0);
    }
    // Transposed A: the ZA transposer into an MC-row buffer.
    let layout = Layout::new::<f64>(&Shape::blocked(600, 40, 512), 1, &tuning).expect("layout");
    assert_eq!(
        (layout.parts[0].a_mode, layout.parts[0].a_len),
        (AMode::Za, 320 * 512)
    );
    // Few rows: split along the columns whatever the shape, no buffer.
    for copied in [false, true] {
        let shape = Shape {
            route: Route::FewRows { copied },
            ..no_trans(30, 20, 64, 30)
        };
        let layout = Layout::new::<f64>(&shape, 3, &tuning).expect("layout");
        assert_eq!(layout.split, Split::Cols);
        let cols: Vec<(usize, usize)> = layout
            .parts
            .iter()
            .map(|part| (part.cols.start, part.cols.len))
            .collect();
        assert_eq!(cols, [(0, 8), (8, 8), (16, 4)]);
        for part in &layout.parts {
            assert_eq!(part.route, Route::FewRows { copied });
            assert_eq!((part.rows.start, part.rows.len), (0, 30));
            assert_eq!((part.a_len, part.b_len, part.b_cap), (0, 0, 0));
        }
    }
}

/// The harness's spec of a part (written on its own) equals the
/// driver's layout over a grid of shapes, routes, transposes, tunings and
/// limits, so the layout checks of the hardware tests compare two
/// independent derivations.
#[test]
fn test_sme2_gemm_policy_spec_matches_layout() {
    fn check<T: TestElem>() {
        let shapes = [
            (49, 40, 100),
            (100, 2 * T::NR + 1, 100),
            (300, 40, 96),
            (600, 3 * T::NR, 512),
            (1030, 40, 512),
            (40, 1025, 512),
            (8, 9, 64),
            (130, 2049, 256),
        ];
        let tunings = [
            TuningOverride::default(),
            TuningOverride {
                in_place_bytes: Some(0),
                ..TuningOverride::default()
            },
            TuningOverride {
                in_place_bytes: Some(0),
                a_reuse_bytes: Some(0),
                mc_target: Some(64),
                ..TuningOverride::default()
            },
            TuningOverride {
                b_bytes: Some(64 * 1024),
                b_one_block_bytes: Some(16 * 1024),
                ..TuningOverride::default()
            },
        ];
        let mut checked = 0;
        for (m, n, k) in shapes {
            for over in tunings {
                let _tuning = override_tuning(over);
                for (trans, store) in [
                    (Trans::NoTrans, Store::ColMajor { pad: 3 }),
                    (Trans::Trans, Store::ColMajor { pad: 0 }),
                    (Trans::NoTrans, Store::RowMajor { pad: 1 }),
                    (Trans::Trans, Store::RowMajor { pad: 5 }),
                ] {
                    let case = Case::<T>::new(m, n, k)
                        .trans(trans, Trans::NoTrans)
                        .stores(store, Store::ColMajor { pad: 0 });
                    let tuning = Tuning::of::<T>();
                    let flipped = matches!(store, Store::RowMajor { .. });
                    let a_trans = if (trans == Trans::NoTrans) == flipped {
                        Trans::Trans
                    } else {
                        Trans::NoTrans
                    };
                    let lda = match (store, trans) {
                        (Store::ColMajor { pad }, Trans::NoTrans) => m + pad,
                        (Store::ColMajor { pad }, _) => k + pad,
                        (Store::RowMajor { pad }, Trans::NoTrans) => k + pad,
                        (Store::RowMajor { pad }, _) => m + pad,
                        (Store::Strided, _) => 0,
                    };
                    let route = super::tests_gemm_harness::spec_route(&case, &tuning);
                    let shape = Shape {
                        m,
                        n,
                        kc: k,
                        a_trans,
                        lda,
                        route,
                    };
                    let layout = Layout::new::<T>(&shape, 1, &tuning);
                    let spec = spec_part_of(
                        &case,
                        k,
                        Span { start: 0, len: m },
                        Span { start: 0, len: n },
                        &tuning,
                    );
                    assert_eq!(
                        layout.map(|layout| layout.parts[0]),
                        spec,
                        "{case:?} {over:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 8 * 4 * 4);
    }
    check::<f32>();
    check::<f64>();
}

// ---------------------------------------------------------------------------
// Through try_gemm (execute SME2)
// ---------------------------------------------------------------------------

/// Every A source and both B packers, reached through the tuning override
/// with small shapes: in place, online packing reused across several column
/// blocks (two row blocks), online packing rebuilt per block, the
/// transposer for `Aᵀ` (from a transposed column-major A and from a
/// row-major A read as `A`), mixed per-block sources within one part (the
/// last row block and the last depth block in place, the others online),
/// every epilogue mode, several depth blocks.
fn a_sources<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let nr = T::NR;
    let elem = core::mem::size_of::<T>();
    // MC = 64, NC = NR (kc = 40): several row and column blocks.
    let small_blocks = TuningOverride {
        mc_target: Some(64),
        b_bytes: Some(40 * elem * nr),
        b_one_block_bytes: Some(40 * elem * nr),
        ..TuningOverride::default()
    };
    let base = Case::<T>::new(130, 3 * nr + 1, 100).explicit_kc(48);
    for (index, (alpha, beta, nan)) in SCALARS.into_iter().enumerate() {
        let mut case = base.scalars(alpha, beta).seed(index as u64);
        if nan {
            case = case.nan_c();
        }
        // In place (the default thresholds: a span of a few KiB).
        {
            let _t = override_tuning(small_blocks);
            let out = run_case(&case);
            let label = format!("{test} in place {case:?}");
            assert_eq!(out.kc, 40, "{label}");
            assert_matches_neon(&out, &label);
            let p = policy_sum(&out);
            assert_eq!(p.a_in_place, sme_calls(&out), "{label}");
            assert!(p.b_za > 0 && p.b_plain == 0, "{label}: {p:?}");
            assert_eq!(p.prefetch, sme_calls(&out), "{label}: prefetch");
        }
        // Online, reused across jc.
        {
            let _t = override_tuning(TuningOverride {
                in_place_bytes: Some(0),
                ..small_blocks
            });
            let out = run_case(&case);
            let label = format!("{test} online reuse {case:?}");
            assert_matches_neon(&out, &label);
            let report = out.report.as_ref().expect("report");
            assert_eq!(report.layout.parts[0].a_mode, AMode::OnlineReuse);
            let p = policy_sum(&out);
            assert!(p.a_online_fill > 0 && p.a_reused > 0, "{label}: {p:?}");
            assert_eq!(p.a_in_place + p.a_online_rebuilt + p.a_za, 0, "{label}");
        }
        // Online, rebuilt per (jc, ic).
        {
            let _t = override_tuning(TuningOverride {
                in_place_bytes: Some(0),
                a_reuse_bytes: Some(0),
                ..small_blocks
            });
            let out = run_case(&case);
            let label = format!("{test} online rebuilt {case:?}");
            assert_matches_neon(&out, &label);
            let report = out.report.as_ref().expect("report");
            assert_eq!(report.layout.parts[0].a_mode, AMode::OnlineRebuild);
            let p = policy_sum(&out);
            assert_eq!(p.a_online_rebuilt, sme_calls(&out), "{label}: {p:?}");
            assert_eq!(p.a_reused + p.a_online_fill, 0, "{label}");
        }
        // The transposer for Aᵀ, plain B for Bᵀ.
        {
            let _t = override_tuning(small_blocks);
            let trans = case.trans(Trans::Trans, Trans::Trans);
            let out = run_case(&trans);
            let label = format!("{test} transposed {trans:?}");
            assert_matches_neon(&out, &label);
            let p = policy_sum(&out);
            assert_eq!(p.a_za, sme_calls(&out), "{label}: {p:?}");
            assert_eq!(p.a_za_reads, sme_calls(&out), "{label}");
            assert!(p.b_plain > 0 && p.b_za == 0, "{label}: {p:?}");
        }
    }
    // Row-major operands: A row-major read as `A` is `Aᵀ` stored
    // column-major (the transposer); `Aᵀ` of a row-major A is
    // column-contiguous (in place); B likewise.
    for (trans, store, za) in [
        (Trans::NoTrans, Store::RowMajor { pad: 3 }, true),
        (Trans::Trans, Store::RowMajor { pad: 0 }, false),
        (Trans::ConjTrans, Store::RowMajor { pad: 2 }, false),
    ] {
        let case = base
            .trans(trans, trans)
            .stores(store, Store::RowMajor { pad: 1 });
        let out = run_case(&case);
        let label = format!("{test} row-major {case:?}");
        assert_matches_neon(&out, &label);
        let p = policy_sum(&out);
        assert_eq!(p.a_za > 0, za, "{label}: {p:?}");
        assert_eq!(p.a_in_place > 0, !za, "{label}: {p:?}");
        assert_eq!(p.b_za > 0, !za, "{label}: B {p:?}");
    }
    // Mixed sources in one part: the in-place limit admits the last row
    // block (2 rows) of the full depth and the last depth block (20 steps)
    // of every row block, not a 64 × 40 block. lda = 133.
    {
        let lda = 133;
        let limit = (39 * lda + 2).max(19 * lda + 64) * elem;
        let _t = override_tuning(TuningOverride {
            in_place_bytes: Some(limit),
            ..small_blocks
        });
        let case = base.stores(Store::ColMajor { pad: 3 }, Store::ColMajor { pad: 0 });
        let out = run_case(&case);
        let label = format!("{test} mixed sources");
        assert_matches_neon(&out, &label);
        let p = policy_sum(&out);
        assert!(
            p.a_in_place > 0 && p.a_online_fill > 0 && p.a_reused > 0,
            "{label}: {p:?}"
        );
    }
}

#[test]
fn test_sme2_gemm_policy_f32_a_sources() {
    a_sources::<f32>("test_sme2_gemm_policy_f32_a_sources");
}

#[test]
fn test_sme2_gemm_policy_f64_a_sources() {
    a_sources::<f64>("test_sme2_gemm_policy_f64_a_sources");
}

/// The real 2 MiB in-place span, one element below, at and one above, with
/// several column panels in one column block (`f64`: 63 · 4160 + m
/// elements at kc = 64; `f32`: 128 · 4095 + m at kc = 129).
fn in_place_real_threshold<T: TestElem>(test: &str, lda: usize, kc: usize, ms: [usize; 3]) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for (index, m) in ms.into_iter().enumerate() {
        let case = Case::<T>::new(m, 2 * T::NR + 1, kc)
            .explicit_kc(kc)
            .stores(Store::ColMajor { pad: lda - m }, Store::ColMajor { pad: 1 })
            .c_guard(WIDE);
        let out = run_case(&case);
        let label = format!("{test} m {m}");
        assert_eq!(out.kc, kc, "{label}");
        assert_matches_neon(&out, &label);
        let p = policy_sum(&out);
        let span = ((kc - 1) * lda + m) * core::mem::size_of::<T>();
        if index < 2 {
            assert!(span <= 2 << 20, "{label}: {span}");
            assert_eq!(p.a_in_place, sme_calls(&out), "{label}: {p:?}");
        } else {
            assert!(span > 2 << 20, "{label}: {span}");
            assert_eq!(p.a_in_place, 0, "{label}: {p:?}");
            assert_eq!(p.a_online_fill, sme_calls(&out), "{label}: {p:?}");
        }
    }
}

#[test]
fn test_sme2_gemm_policy_f64_in_place_real_threshold() {
    in_place_real_threshold::<f64>(
        "test_sme2_gemm_policy_f64_in_place_real_threshold",
        4160,
        64,
        [63, 64, 65],
    );
}

#[test]
fn test_sme2_gemm_policy_f32_in_place_real_threshold() {
    in_place_real_threshold::<f32>(
        "test_sme2_gemm_policy_f32_in_place_real_threshold",
        4095,
        129,
        [127, 128, 129],
    );
}

/// The real 4 MiB reuse limit, one panel below, at and one above, at
/// kc = 2048 (`f64`: 256 rows; `f32`: 512 rows), with NC = NR through the
/// tuning (several column blocks: the reused copy is read), and stale
/// packing buffers.
fn reuse_real_threshold<T: TestElem>(test: &str, ms: [usize; 3]) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let elem = core::mem::size_of::<T>();
    let kc = 2048;
    let _t = override_tuning(TuningOverride {
        b_bytes: Some(kc * elem * T::NR),
        b_one_block_bytes: Some(kc * elem * T::NR),
        ..TuningOverride::default()
    });
    for (index, m) in ms.into_iter().enumerate() {
        let case = Case::<T>::new(m, T::NR + 1, kc)
            .explicit_kc(kc)
            .scalars(1.0, 1.0);
        // Stale buffers as large as the part's A buffer can be (the copy
        // of all rows) and two B blocks.
        let out = run_stale(&case, m.div_ceil(64) * 64 * kc, 2 * T::NR * kc);
        let label = format!("{test} m {m}");
        assert_eq!(out.kc, kc, "{label}");
        assert_matches_neon(&out, &label);
        let report = out.report.as_ref().expect("report");
        let p = policy_sum(&out);
        if index < 2 {
            assert_eq!(report.layout.parts[0].a_mode, AMode::OnlineReuse, "{label}");
            assert!(p.a_online_fill > 0 && p.a_reused > 0, "{label}: {p:?}");
        } else {
            assert_eq!(
                report.layout.parts[0].a_mode,
                AMode::OnlineRebuild,
                "{label}"
            );
            assert_eq!(p.a_online_rebuilt, sme_calls(&out), "{label}: {p:?}");
        }
    }
}

/// A copy of all rows within the reuse limit that does not fit the
/// packing cache next to B (the limit lowered to one byte below the sum):
/// rebuilt per block in an MC-row buffer, which fits.
fn reuse_over_cache_limit<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let elem = core::mem::size_of::<T>();
    let (m, n, kc): (usize, usize, usize) = (600, 40, 512);
    let whole = m.div_ceil(64) * 64 * kc;
    let b_len = n.div_ceil(T::NR) * T::NR * kc;
    assert!(whole * elem <= 4 << 20, "{test}: within the reuse limit");
    let limit = (whole + b_len) * elem - 1;
    let _limit = set_pack_cache_limit(limit);
    // lda = 1200: the 320 × 512 blocks span more than 2 MiB (online).
    let case = Case::<T>::new(m, n, kc)
        .explicit_kc(kc)
        .stores(Store::ColMajor { pad: 600 }, Store::ColMajor { pad: 0 })
        .c_guard(WIDE);
    let out = run_case(&case);
    let label = format!("{test} limit {limit}");
    assert_matches_neon(&out, &label);
    let part = out.report.as_ref().expect("report").layout.parts[0];
    assert_eq!(part.a_mode, AMode::OnlineRebuild, "{label}");
    assert!((part.a_len + part.b_len) * elem <= limit, "{label}");
    let p = policy_sum(&out);
    assert_eq!(p.a_online_rebuilt, sme_calls(&out), "{label}: {p:?}");
    // One byte more: the copy is reused.
    let _roomy = set_pack_cache_limit(limit + 1);
    let out = run_case(&case);
    assert_matches_neon(&out, &format!("{label} + 1"));
    let part = out.report.as_ref().expect("report").layout.parts[0];
    assert_eq!(part.a_mode, AMode::OnlineReuse, "{label} + 1");
}

#[test]
fn test_sme2_gemm_policy_f32_reuse_over_cache_limit() {
    reuse_over_cache_limit::<f32>("test_sme2_gemm_policy_f32_reuse_over_cache_limit");
}

#[test]
fn test_sme2_gemm_policy_f64_reuse_over_cache_limit() {
    reuse_over_cache_limit::<f64>("test_sme2_gemm_policy_f64_reuse_over_cache_limit");
}

#[test]
fn test_sme2_gemm_policy_f64_reuse_real_threshold() {
    reuse_real_threshold::<f64>(
        "test_sme2_gemm_policy_f64_reuse_real_threshold",
        [192, 256, 257],
    );
}

#[test]
fn test_sme2_gemm_policy_f32_reuse_real_threshold() {
    reuse_real_threshold::<f32>(
        "test_sme2_gemm_policy_f32_reuse_real_threshold",
        [448, 512, 513],
    );
}

/// The real 4 MiB B cap of a part of one MC block, one NR panel below, at
/// and one above (kc = 512): 4 MiB up to the width, 2 MiB beyond.
fn b_cap_real_threshold<T: TestElem>(test: &str, m: usize) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let kc = 512;
    let width = (4 << 20) / (kc * core::mem::size_of::<T>());
    for (n, cap) in [
        (width - T::NR, 4 << 20),
        (width, 4 << 20),
        (width + 1, 2 << 20),
    ] {
        let case = Case::<T>::new(m, n, kc)
            .explicit_kc(kc)
            .trans(Trans::NoTrans, Trans::Trans)
            .scalars(0.5, 0.0)
            .nan_c();
        let out = run_case(&case);
        let label = format!("{test} n {n}");
        assert_matches_neon(&out, &label);
        let part = out.report.as_ref().expect("report").layout.parts[0];
        assert_eq!(part.b_cap, cap, "{label}");
        let blocks = if cap == 4 << 20 { 1 } else { 3 };
        assert_eq!(n.div_ceil(part.nc), blocks, "{label}: column blocks");
    }
}

#[test]
fn test_sme2_gemm_policy_f64_b_cap_real_threshold() {
    b_cap_real_threshold::<f64>("test_sme2_gemm_policy_f64_b_cap_real_threshold", 40);
}

#[test]
fn test_sme2_gemm_policy_f32_b_cap_real_threshold() {
    b_cap_real_threshold::<f32>("test_sme2_gemm_policy_f32_b_cap_real_threshold", 49);
}

/// The few-rows limit one below, at and one above, for `op(A) = A` (in
/// place) and `op(A) = Aᵀ` (the copy), with several depth blocks (k = 100,
/// KC 40: few rows must take the blocked path's KC), every epilogue mode,
/// wide guards around C.
fn few_rows_limit<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let limit = T::FEW_ROWS;
    let mut index = 0;
    for m in [limit - 1, limit, limit + 1] {
        for (trans_a, store) in [
            (Trans::NoTrans, Store::ColMajor { pad: 5 }),
            (Trans::Trans, Store::ColMajor { pad: 0 }),
            (Trans::NoTrans, Store::RowMajor { pad: 2 }),
            (Trans::ConjTrans, Store::RowMajor { pad: 0 }),
        ] {
            for trans_b in [Trans::NoTrans, Trans::Trans] {
                let (alpha, beta, nan) = SCALARS[index % SCALARS.len()];
                index += 1;
                let mut case = Case::<T>::new(m, 3 * T::NR + 1, 100)
                    .explicit_kc(48)
                    .trans(trans_a, trans_b)
                    .stores(store, Store::ColMajor { pad: 3 })
                    .scalars(alpha, beta)
                    .c_guard(WIDE);
                if nan {
                    case = case.nan_c();
                }
                let out = run_case(&case);
                let label = format!("{test} {case:?}");
                assert_eq!(out.kc, 40, "{label}");
                assert_matches_neon(&out, &label);
                let report = out.report.as_ref().expect("report");
                let copied = (trans_a == Trans::NoTrans) == matches!(store, Store::RowMajor { .. });
                if m <= limit {
                    assert_eq!(report.route, Route::FewRows { copied }, "{label}");
                    assert_eq!(policy_sum(&out).few_rows, 1, "{label}");
                } else {
                    assert_eq!(report.route, Route::Blocked, "{label}");
                    assert_eq!(policy_sum(&out).few_rows, 0, "{label}");
                }
            }
        }
    }
}

#[test]
fn test_sme2_gemm_policy_f32_few_rows_limit() {
    few_rows_limit::<f32>("test_sme2_gemm_policy_f32_few_rows_limit");
}

#[test]
fn test_sme2_gemm_policy_f64_few_rows_limit() {
    few_rows_limit::<f64>("test_sme2_gemm_policy_f64_few_rows_limit");
}

/// Few rows beyond the limit tests: special values, a copy of `Aᵀ` that
/// does not fit the packing cache (the blocked route instead), stale
/// buffers, and 32³ one below, at and one above with the small class off
/// and on.
fn few_rows_more<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    for special in [Special::NonFinite, Special::Subnormal] {
        for trans_a in [Trans::NoTrans, Trans::Trans] {
            let case = Case::<T>::new(T::FEW_ROWS - 3, 2 * T::NR + 5, 70)
                .explicit_kc(40)
                .trans(trans_a, Trans::NoTrans)
                .special(special)
                .c_guard(WIDE);
            assert_matches_neon(&run_case(&case), &format!("{test} {case:?}"));
        }
    }
    // The copy of Aᵀ (m · k elements) at and one byte above the
    // packing-cache limit: the few-rows route, then the blocked route (its
    // buffers, 64 × KC of A and 2 NR × KC of B with KC 48, still fit).
    let (m, k) = (T::FEW_ROWS - 8, 300);
    let bytes = m * k * core::mem::size_of::<T>();
    for (limit, route) in [
        (bytes, Route::FewRows { copied: true }),
        (bytes - 1, Route::Blocked),
    ] {
        let _limit = set_pack_cache_limit(limit);
        let case = Case::<T>::new(m, T::NR + 3, k)
            .explicit_kc(48)
            .trans(Trans::Trans, Trans::NoTrans)
            .c_guard(WIDE);
        // Stale buffers of exactly what the route needs, so that they park
        // under the lowered limit: the copy, or the blocked A and B blocks.
        let (a_len, b_len) = match route {
            Route::Blocked => (64 * 48, 2 * T::NR * 48),
            Route::FewRows { .. } => (m * k, 0),
        };
        let out = run_stale(&case, a_len, b_len);
        let label = format!("{test} copy limit {limit}");
        assert_eq!(out.kc, 48, "{label}");
        assert_matches_neon(&out, &label);
        assert_eq!(out.report.as_ref().expect("report").route, route, "{label}");
    }
    // Not even the blocked buffers fit: the call declines, C untouched.
    {
        let _limit = set_pack_cache_limit(1024);
        let case = Case::<T>::new(100, T::NR + 3, k).explicit_kc(48);
        let out = run_case(&case);
        assert!(!spec_accepts(&case, out.kc), "{test}: spec");
        assert_declined(&out, &format!("{test} buffers above the limit"));
    }
    // 32³: (32, 32, 32) and (31, 32, 32) are the small class, (32, 32, 33)
    // and (33, 32, 32) the blocked class.
    for small_on in [false, true] {
        let _small = override_tuning(TuningOverride {
            small_class: Some(small_on),
            ..TuningOverride::default()
        });
        for (m, n, k, small) in [
            (31, 32, 32, true),
            (32, 32, 32, true),
            (32, 32, 33, false),
            (33, 32, 32, false),
        ] {
            for trans in [Trans::NoTrans, Trans::Trans] {
                let case = Case::<T>::new(m, n, k).trans(trans, trans).c_guard(WIDE);
                let out = run_case(&case);
                let label = format!("{test} 32³ {case:?} small class {small_on}");
                if small && !small_on {
                    assert_declined(&out, &label);
                } else {
                    assert_matches_neon(&out, &label);
                    let report = out.report.as_ref().expect("report");
                    assert_eq!(report.prefetch, if small { 0 } else { 1 }, "{label}");
                }
            }
        }
    }
    // The small class on the blocked route (64 rows): no prefetch.
    let _small = override_tuning(TuningOverride {
        small_class: Some(true),
        ..TuningOverride::default()
    });
    let out = run_case(&Case::<T>::new(64, 16, 32));
    assert_matches_neon(&out, &format!("{test} small blocked"));
    assert_eq!(out.report.as_ref().expect("report").route, Route::Blocked);
    assert_eq!(policy_sum(&out).prefetch, 0, "{test} small blocked");
}

#[test]
fn test_sme2_gemm_policy_f32_few_rows_more() {
    few_rows_more::<f32>("test_sme2_gemm_policy_f32_few_rows_more");
}

#[test]
fn test_sme2_gemm_policy_f64_few_rows_more() {
    few_rows_more::<f64>("test_sme2_gemm_policy_f64_few_rows_more");
}

/// Stale packing buffers and wide guards for every A mode: a call that
/// read a buffer element it did not write in the same call, or wrote
/// outside C, would change bits.
fn stale_and_guards<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let elem = core::mem::size_of::<T>();
    let small_blocks = TuningOverride {
        mc_target: Some(64),
        b_bytes: Some(40 * elem * T::NR),
        b_one_block_bytes: Some(40 * elem * T::NR),
        ..TuningOverride::default()
    };
    for (over, trans) in [
        (small_blocks, Trans::NoTrans),
        (
            TuningOverride {
                in_place_bytes: Some(0),
                ..small_blocks
            },
            Trans::NoTrans,
        ),
        (
            TuningOverride {
                in_place_bytes: Some(0),
                a_reuse_bytes: Some(0),
                ..small_blocks
            },
            Trans::NoTrans,
        ),
        (small_blocks, Trans::Trans),
    ] {
        let _t = override_tuning(over);
        for (m, n) in [(129, 3 * T::NR + 2), (65, T::NR - 1), (200, 2 * T::NR)] {
            let case = Case::<T>::new(m, n, 100)
                .explicit_kc(48)
                .trans(trans, trans)
                .c_pad(0)
                .c_guard(WIDE)
                .scalars(1.0, 1.0);
            let out = run_stale(&case, 1 << 14, 1 << 12);
            assert_matches_neon(&out, &format!("{test} {over:?} {case:?}"));
        }
    }
}

#[test]
fn test_sme2_gemm_policy_f32_stale_and_guards() {
    stale_and_guards::<f32>("test_sme2_gemm_policy_f32_stale_and_guards");
}

#[test]
fn test_sme2_gemm_policy_f64_stale_and_guards() {
    stale_and_guards::<f64>("test_sme2_gemm_policy_f64_stale_and_guards");
}

/// Every route and A mode on one, two and three parts (three units,
/// splitting from any work).
#[cfg(feature = "parallel")]
fn part_counts<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let _units = override_units(3);
    let elem = core::mem::size_of::<T>();
    let modes = [
        TuningOverride {
            par_min_macs: Some(0),
            ..TuningOverride::default()
        },
        TuningOverride {
            par_min_macs: Some(0),
            in_place_bytes: Some(0),
            mc_target: Some(64),
            b_bytes: Some(32 * elem * T::NR),
            b_one_block_bytes: Some(32 * elem * T::NR),
            ..TuningOverride::default()
        },
        TuningOverride {
            par_min_macs: Some(0),
            in_place_bytes: Some(0),
            a_reuse_bytes: Some(0),
            ..TuningOverride::default()
        },
    ];
    for over in modes {
        let _t = override_tuning(over);
        for (m, n, trans) in [
            (200, 2 * T::NR + 1, Trans::NoTrans),
            (70, 7 * T::NR - 3, Trans::Trans),
            (T::FEW_ROWS, 5 * T::NR + 2, Trans::NoTrans),
            (T::FEW_ROWS - 1, 3 * T::NR, Trans::Trans),
        ] {
            // Panels on the split side: columns for few rows and for
            // m < n, rows otherwise.
            let panels = if m <= T::FEW_ROWS || m < n {
                n.div_ceil(T::NR)
            } else {
                m.div_ceil(64)
            };
            for (par, threads) in [
                (Par::Seq, 1),
                (Par::RayonWith(2), 2),
                (Par::RayonWith(3), 3),
            ] {
                let parts = threads.min(panels);
                let case = Case::<T>::new(m, n, 96)
                    .explicit_kc(32)
                    .trans(trans, trans)
                    .par(par)
                    .c_guard(WIDE);
                let out = run_case(&case);
                let label = format!("{test} {over:?} {case:?}");
                assert_matches_neon(&out, &label);
                let report = out.report.as_ref().expect("report");
                assert_eq!(report.layout.parts.len(), parts, "{label}");
                if m <= T::FEW_ROWS {
                    assert_eq!(report.layout.split, Split::Cols, "{label}");
                    assert_eq!(policy_sum(&out).few_rows, parts, "{label}");
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_f32_part_counts() {
    part_counts::<f32>("test_sme2_gemm_policy_f32_part_counts");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_gemm_policy_f64_part_counts() {
    part_counts::<f64>("test_sme2_gemm_policy_f64_part_counts");
}

/// The readiness rule for every SME2 entry the policy uses: the decision
/// checks the calling thread once (a countdown of one call passes it), the
/// one part then checks again on the same thread, answers `false` and runs
/// on NEON. No kernel (macro-kernel, ZA transposer, few rows) may run on
/// this thread, and the bits are NEON's.
fn readiness<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let online = TuningOverride {
        in_place_bytes: Some(0),
        ..TuningOverride::default()
    };
    for (over, case) in [
        // Few rows, A in place.
        (
            TuningOverride::default(),
            Case::<T>::new(T::FEW_ROWS, 2 * T::NR + 1, 70),
        ),
        // Few rows, Aᵀ copied.
        (
            TuningOverride::default(),
            Case::<T>::new(T::FEW_ROWS - 2, 2 * T::NR, 70).trans(Trans::Trans, Trans::NoTrans),
        ),
        // Blocked, online packing, B through the transposer.
        (online, Case::<T>::new(100, 2 * T::NR + 1, 70)),
        // Blocked, Aᵀ through the transposer, plain B.
        (
            TuningOverride::default(),
            Case::<T>::new(100, 2 * T::NR + 1, 70).trans(Trans::Trans, Trans::Trans),
        ),
    ] {
        let _t = override_tuning(over);
        let label = format!("{test} {over:?} {case:?}");
        let control = run_case(&case);
        assert_matches_neon(&control, &format!("{label}: control"));
        let entries = kernel_entries_here();
        let out = run_case_around(&case, |call| {
            let _countdown = Countdown::new(1);
            call()
        });
        assert_same_bits(&out, &label);
        assert_runs(&out, &label, |_, _| (true, BlockCount::default()));
        assert_eq!(kernel_entries_here(), entries, "{label}: a kernel ran");
    }
    // With rayon: a marked part of a few-rows call runs on NEON, the others
    // on SME2.
    #[cfg(feature = "parallel")]
    {
        let _units = override_units(3);
        let _t = override_tuning(TuningOverride {
            par_min_macs: Some(0),
            ..TuningOverride::default()
        });
        for trans in [Trans::NoTrans, Trans::Trans] {
            let _marked = mark_not_ready(0b010);
            let case = Case::<T>::new(T::FEW_ROWS - 1, 6 * T::NR + 1, 70)
                .trans(trans, Trans::NoTrans)
                .par(Par::RayonWith(3));
            let out = run_case(&case);
            let label = format!("{test} marked part {case:?}");
            assert_same_bits(&out, &label);
            assert_runs(&out, &label, |index, part| {
                if index == 1 {
                    (true, BlockCount::default())
                } else {
                    (false, super::tests_gemm_harness::expected_part(&out, part))
                }
            });
        }
    }
}

#[test]
fn test_sme2_gemm_policy_f32_readiness() {
    readiness::<f32>("test_sme2_gemm_policy_f32_readiness");
}

#[test]
fn test_sme2_gemm_policy_f64_readiness() {
    readiness::<f64>("test_sme2_gemm_policy_f64_readiness");
}

/// The prefetch switches reach every macro-kernel call of the blocked
/// class, for every A source, and nothing else of the record moves with
/// them.
#[test]
fn test_sme2_gemm_policy_prefetch_on_blocked_calls() {
    let test = "test_sme2_gemm_policy_prefetch_on_blocked_calls";
    if !backend_or_skip(test, Need::F64) {
        return;
    }
    for (over, trans) in [
        (TuningOverride::default(), Trans::NoTrans),
        (
            TuningOverride {
                in_place_bytes: Some(0),
                mc_target: Some(64),
                ..TuningOverride::default()
            },
            Trans::NoTrans,
        ),
        (TuningOverride::default(), Trans::Trans),
    ] {
        let _t = override_tuning(over);
        let case = Case::<f64>::new(150, 3 * 8 + 1, 80)
            .explicit_kc(40)
            .trans(trans, trans);
        let out = run_case(&case);
        let label = format!("{test} {over:?} {trans:?}");
        assert_matches_neon(&out, &label);
        assert_eq!(out.report.as_ref().expect("report").prefetch, PREFETCH_C);
        assert_eq!(policy_sum(&out).prefetch, sme_calls(&out), "{label}");
    }
}
