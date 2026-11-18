//! Tests of the GEMM hook in `gemm_core` through the public entry points:
//! the calls that the NEON build runs on the blocked route run on SME2, with
//! the bits of the same public call with the backend off.
//!
//! Every test that executes SME2 code goes through
//! `test_support::backend_or_skip` (the call goes through the dispatch
//! decision of the hook). The reference of every comparison is the same
//! public call, with the same operands and the same `Par`, made while the
//! calling thread holds `test_support::force_off`: the hook then declines
//! on that thread and the NEON blocked route computes it. The proof that
//! SME2 ran is the run record of the driver (`parallel::take_last_run`:
//! every part's kernel entries counted on the thread that ran it, and what
//! it ran) and the decision counter of the calling thread, never the
//! process-wide kernel counter.
//!
//! The helpers ([`Entry`], [`Call`], [`run_public`], [`assert_same_c`],
//! [`assert_sme2_ran`]) are shared with `tests_hook_declines.rs`,
//! `tests_hook_thin.rs` and `tests_hook_first_use.rs`.

use oxiblas_core::parallel::Par;

#[cfg(feature = "parallel")]
use super::detect::override_units;
use super::detect::{queries_here, set_ready_countdown};
#[cfg(feature = "parallel")]
use super::driver::{TuningOverride, override_tuning};
use super::parallel::{RunReport, take_last_run};
use super::test_support::{backend_or_skip, decisions, force_off, sme_state};
use super::tests_gemm_harness::{
    CBuf, Operand, Rng, Store, TestElem, expected_blocks, stored_shape,
};
use crate::level3::gemm::{
    BlockingSource, GemmBlocking, call_blocking, gemm, gemm_asymmetric_with_par,
    gemm_auto_with_par, gemm_transposed, gemm_transposed_with_blocking, gemm_transposed_with_par,
    gemm_with_blocking, gemm_with_par,
};
use crate::level3::gemm_kernel::GemmKernel;
use crate::level3::gemm_plan::{DEFAULT_POLICY, GemmPlan};
use crate::level3::trsm::Trans;

/// A public GEMM entry point (and, where it takes them, its blocking).
#[derive(Debug, Clone, Copy)]
pub(super) enum Entry {
    /// `gemm` (`NoTrans` / `NoTrans`, `Par::Seq`).
    Gemm,
    /// `gemm_with_par` (`NoTrans` / `NoTrans`).
    GemmWithPar,
    /// `gemm_transposed` (`Par::Seq`).
    Transposed,
    /// `gemm_transposed_with_par`.
    TransposedWithPar,
    /// `gemm_with_blocking` (`NoTrans` / `NoTrans`, explicit blocking).
    WithBlocking(GemmBlocking),
    /// `gemm_transposed_with_blocking` (explicit blocking).
    TransposedWithBlocking(GemmBlocking),
    /// `gemm_auto_with_par` (`NoTrans` / `NoTrans`, heuristic blocking).
    Auto,
    /// `gemm_asymmetric_with_par` (`NoTrans` / `NoTrans`, heuristic
    /// blocking).
    Asymmetric,
}

impl Entry {
    /// Whether the entry point takes transposes.
    pub(super) fn transposes(self) -> bool {
        matches!(
            self,
            Self::Transposed | Self::TransposedWithPar | Self::TransposedWithBlocking(_)
        )
    }

    /// Whether the entry point takes a `Par`.
    pub(super) fn takes_par(self) -> bool {
        !matches!(self, Self::Gemm | Self::Transposed)
    }

    /// Where the blocks of the entry point's call of `m × k` by `k × n` come
    /// from (the same source the entry point passes to `gemm_planned`).
    pub(super) fn source<T: GemmKernel>(self, m: usize, n: usize, k: usize) -> BlockingSource {
        let shape = T::micro_kernel_shape();
        match self {
            Self::WithBlocking(blocking) | Self::TransposedWithBlocking(blocking) => {
                BlockingSource::Explicit(blocking)
            }
            Self::Auto => BlockingSource::Heuristic(GemmBlocking::auto_tuned::<T>(m, k, n, &shape)),
            Self::Asymmetric => {
                BlockingSource::Heuristic(GemmBlocking::asymmetric::<T>(m, k, n, &shape))
            }
            _ => BlockingSource::Default,
        }
    }

    /// The KC step of the entry point's call: the sequential KC of the
    /// call's fitted blocking, at most `k`.
    pub(super) fn kc<T: GemmKernel>(self, m: usize, n: usize, k: usize) -> usize {
        let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
        call_blocking(self.source::<T>(m, n, k), &plan, m, k, n)
            .sequential()
            .kc
            .min(k)
    }
}

/// The entry points that take transposes, and those that do not.
pub(super) fn entries() -> [Entry; 8] {
    let explicit = GemmBlocking {
        mc: 96,
        kc: 48,
        nc: 64,
    };
    [
        Entry::Gemm,
        Entry::GemmWithPar,
        Entry::Transposed,
        Entry::TransposedWithPar,
        Entry::WithBlocking(explicit),
        Entry::TransposedWithBlocking(explicit),
        Entry::Auto,
        Entry::Asymmetric,
    ]
}

/// The four transpose combinations.
pub(super) const TRANS_PAIRS: [(Trans, Trans); 4] = [
    (Trans::NoTrans, Trans::NoTrans),
    (Trans::NoTrans, Trans::Trans),
    (Trans::Trans, Trans::NoTrans),
    (Trans::Trans, Trans::Trans),
];

/// One public call: the entry point, the problem, the operands' storage and
/// the scalars.
#[derive(Debug, Clone, Copy)]
pub(super) struct Call {
    /// The entry point.
    pub(super) entry: Entry,
    /// Transposes (both `NoTrans` for an entry point without them).
    pub(super) trans: (Trans, Trans),
    /// `C` is `m × n`, `op(A)` is `m × k`.
    pub(super) dims: (usize, usize, usize),
    /// alpha and beta.
    pub(super) scalars: (f64, f64),
    /// Storage of A and B.
    pub(super) stores: (Store, Store),
    /// Rows between the columns of C beyond `m`.
    pub(super) c_pad: usize,
    /// C starts as quiet NaNs (with `beta == 0` it must not be read).
    pub(super) nan_c: bool,
    /// Plant `+Inf` and NaN in A, `-Inf` in B.
    pub(super) non_finite: bool,
    /// The `Par` of entry points that take one.
    pub(super) par: Par,
    /// The seed of the data.
    pub(super) seed: u64,
}

impl Call {
    /// `entry` on an `m × n × k` problem: `NoTrans` / `NoTrans`, alpha 0.75,
    /// beta 0.5, column-major operands, C padded by 3 rows, finite data,
    /// `Par::Seq`.
    pub(super) fn new(entry: Entry, m: usize, n: usize, k: usize) -> Self {
        Self {
            entry,
            trans: (Trans::NoTrans, Trans::NoTrans),
            dims: (m, n, k),
            scalars: (0.75, 0.5),
            stores: (Store::ColMajor { pad: 1 }, Store::ColMajor { pad: 2 }),
            c_pad: 3,
            nan_c: false,
            non_finite: false,
            par: Par::Seq,
            seed: 7,
        }
    }

    /// With transposes.
    pub(super) fn trans(self, trans_a: Trans, trans_b: Trans) -> Self {
        Self {
            trans: (trans_a, trans_b),
            ..self
        }
    }

    /// With alpha and beta.
    pub(super) fn scalars(self, alpha: f64, beta: f64) -> Self {
        Self {
            scalars: (alpha, beta),
            ..self
        }
    }

    /// With the storage of A and B.
    pub(super) fn stores(self, a: Store, b: Store) -> Self {
        Self {
            stores: (a, b),
            ..self
        }
    }

    /// With `Par`.
    pub(super) fn par(self, par: Par) -> Self {
        Self { par, ..self }
    }

    /// C full of NaN.
    pub(super) fn nan_c(self) -> Self {
        Self {
            nan_c: true,
            ..self
        }
    }

    /// Non-finite values in A and B.
    pub(super) fn non_finite(self) -> Self {
        Self {
            non_finite: true,
            ..self
        }
    }

    /// With another seed.
    pub(super) fn seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }
}

/// What one public call left and recorded.
pub(super) struct Run<T> {
    /// C with its border, after the call.
    pub(super) c: CBuf<T>,
    /// SME2 dispatch decisions of the calling thread during the call.
    pub(super) decisions: u64,
    /// Detection calls of the calling thread during the call.
    pub(super) queries: u64,
    /// The driver's run record of the call (`None`: no SME2 run).
    pub(super) report: Option<RunReport>,
}

/// The operands of `call`: A and B as stored, and C with its border.
pub(super) fn operands<T: TestElem>(call: &Call) -> (Operand<T>, Operand<T>, CBuf<T>) {
    let (m, n, k) = call.dims;
    let mut rng = Rng::new(call.seed);
    let (ar, ac) = stored_shape(call.trans.0, m, k);
    let (br, bc) = stored_shape(call.trans.1, k, n);
    let mut a = Operand::<T>::new(ar, ac, call.stores.0, &mut rng);
    let mut b = Operand::<T>::new(br, bc, call.stores.1, &mut rng);
    if call.non_finite {
        // +Inf and a NaN in the first stored column of A, -Inf in the last
        // stored column of B: whole rows and columns of C become non-finite.
        let first = 0;
        a.data[first] = T::of(f64::INFINITY);
        a.data[(ar - 1) * a.col_stride + first] = T::of(f64::NAN);
        b.data[(bc - 1) * b.row_stride] = T::of(f64::NEG_INFINITY);
    }
    let c = CBuf::new(m, n, call.c_pad, call.nan_c, &mut rng);
    (a, b, c)
}

/// Makes `call` through its public entry point on this thread (no state
/// changed) and returns what it left and recorded.
pub(super) fn run_public<T: TestElem>(call: &Call) -> Run<T> {
    let (a, b, mut c) = operands::<T>(call);
    let (alpha, beta) = (T::of(call.scalars.0), T::of(call.scalars.1));
    let (trans_a, trans_b) = call.trans;
    assert!(
        call.entry.transposes() || call.trans == (Trans::NoTrans, Trans::NoTrans),
        "{call:?}: the entry point takes no transposes"
    );
    let par = if call.entry.takes_par() {
        call.par
    } else {
        Par::Seq
    };
    let _ = take_last_run();
    let (decided, asked) = (decisions(), queries_here());
    {
        let (a, b, c) = (a.view(), b.view(), c.view());
        match call.entry {
            Entry::Gemm => gemm(alpha, a, b, beta, c),
            Entry::GemmWithPar => gemm_with_par(alpha, a, b, beta, c, par),
            Entry::Transposed => gemm_transposed(trans_a, trans_b, alpha, a, b, beta, c),
            Entry::TransposedWithPar => {
                gemm_transposed_with_par(trans_a, trans_b, alpha, a, b, beta, c, par);
            }
            Entry::WithBlocking(blocking) => {
                gemm_with_blocking(alpha, a, b, beta, c, par, &blocking);
            }
            Entry::TransposedWithBlocking(blocking) => gemm_transposed_with_blocking(
                trans_a, trans_b, alpha, a, b, beta, c, par, &blocking,
            ),
            Entry::Auto => gemm_auto_with_par(alpha, a, b, beta, c, par),
            Entry::Asymmetric => gemm_asymmetric_with_par(alpha, a, b, beta, c, par),
        }
    }
    Run {
        c,
        decisions: decisions() - decided,
        queries: queries_here() - asked,
        report: take_last_run(),
    }
}

/// [`run_public`] with the backend forced off on this thread: the NEON
/// result of the public call.
pub(super) fn run_reference<T: TestElem>(call: &Call) -> Run<T> {
    let _off = force_off();
    let run = run_public::<T>(call);
    assert_eq!(run.decisions, 0, "{call:?}: a decision under force_off");
    assert!(run.report.is_none(), "{call:?}: SME2 ran under force_off");
    run
}

/// Every element of the C block is the reference's (bitwise, or NaN in
/// both), and every element of the border is bitwise untouched.
pub(super) fn assert_same_c<T: TestElem>(got: &Run<T>, want: &Run<T>, label: &str) {
    assert_eq!(
        got.c.data.len(),
        want.c.data.len(),
        "{label}: buffer length"
    );
    for (index, (g, w)) in got.c.data.iter().zip(&want.c.data).enumerate() {
        let same = if got.c.in_block(index) {
            super::args::same_result(*g, *w)
        } else {
            g.bits() == w.bits() && g.bits() == T::SENTINEL.bits()
        };
        assert!(
            same,
            "{label}: element {index} (in block: {}) is {:#x}, the NEON call gives {:#x}",
            got.c.in_block(index),
            g.bits(),
            w.bits()
        );
    }
}

/// The call ran on SME2: one decision, a run record whose every part ran on
/// SME2 with the number of SME2 calls of its blocks for depth `k` in steps
/// of `kc` and whose kernel entries (counted on the part's own thread) are
/// exactly those calls plus the ZA transposers it recorded; the thread is
/// out of streaming mode with no ZA state afterwards. Returns the kernel
/// entries of all parts.
pub(super) fn assert_sme2_ran<T: TestElem>(run: &Run<T>, k: usize, kc: usize, label: &str) -> u64 {
    assert_eq!(run.decisions, 1, "{label}: decisions");
    let Some(report) = run.report.as_ref() else {
        panic!("{label}: no SME2 run recorded");
    };
    assert_eq!(report.neon.len(), report.layout.parts.len(), "{label}");
    let mut total = 0;
    for (index, part) in report.layout.parts.iter().enumerate() {
        let blocks = &report.blocks[index];
        assert!(!report.neon[index], "{label}: part {index} ran on NEON");
        assert_eq!(blocks.neon, 0, "{label}: part {index} ran a NEON block");
        assert_eq!(
            blocks.sme,
            expected_blocks(part, k, kc),
            "{label}: SME2 calls of part {index} (kc {kc})"
        );
        let expected_entries = blocks.sme + blocks.policy.a_za + blocks.policy.b_za;
        assert_eq!(
            blocks.kernel_calls, expected_entries as u64,
            "{label}: kernel entries of part {index}"
        );
        total += blocks.kernel_calls;
    }
    assert_eq!(
        sme_state(),
        Some((0, 0)),
        "{label}: SME state after the call"
    );
    total
}

/// `call` through its entry point with the backend on and off: same C, and
/// SME2 ran (see [`assert_sme2_ran`]). Returns the kernel entries.
pub(super) fn check_runs_on_sme2<T: TestElem>(call: &Call, label: &str) -> u64 {
    let want = run_reference::<T>(call);
    let got = run_public::<T>(call);
    let (m, n, k) = call.dims;
    let kc = call.entry.kc::<T>(m, n, k);
    let entries = assert_sme2_ran(&got, k, kc, label);
    assert_same_c(&got, &want, label);
    entries
}

/// alpha / beta pairs: the four epilogue modes, a negative zero alpha
/// (not 0 for the quick return: `-0 == 0`, so that one is a quick return
/// too) and NaN and infinite scalars.
const SCALARS: [(f64, f64); 7] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (-0.5, 0.0),
    (0.75, -1.25),
    (1.0, -0.0),
    (2.0, f64::INFINITY),
    (f64::NAN, 0.5),
];

/// Every entry point, every transpose combination it takes, every scalar
/// pair, NaN in C with beta 0, non-finite data and padded (strided) C:
/// SME2 runs, with the NEON bits.
fn public_entries_run_on_sme2<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let nr = T::NR;
    // 70 x (3·NR + 1) x 65: above 32³, blocked, one block per part.
    let (m, n, k) = (70, 3 * nr + 1, 65);
    for entry in entries() {
        let pairs: &[(Trans, Trans)] = if entry.transposes() {
            &TRANS_PAIRS
        } else {
            &TRANS_PAIRS[..1]
        };
        for &(trans_a, trans_b) in pairs {
            for (index, &(alpha, beta)) in SCALARS.iter().enumerate() {
                let mut call = Call::new(entry, m, n, k)
                    .trans(trans_a, trans_b)
                    .scalars(alpha, beta)
                    .seed(index as u64 + 11);
                if beta == 0.0 {
                    call = call.nan_c();
                }
                if index % 2 == 1 {
                    call = call.non_finite();
                }
                let label = format!("{test}: {entry:?} {trans_a:?}/{trans_b:?} a {alpha} b {beta}");
                check_runs_on_sme2::<T>(&call, &label);
            }
        }
    }
}

#[test]
fn test_sme2_hook_public_entries_run_on_sme2_f32() {
    public_entries_run_on_sme2::<f32>("test_sme2_hook_public_entries_run_on_sme2_f32");
}

#[test]
fn test_sme2_hook_public_entries_run_on_sme2_f64() {
    public_entries_run_on_sme2::<f64>("test_sme2_hook_public_entries_run_on_sme2_f64");
}

/// The exact kernel entries of a one-block call (one depth, column and row
/// block, `Par::Seq`): the macro-kernel, plus the ZA transposer of B for
/// `op(B) = B` (the plain packer for `op(B) = Bᵀ` runs no kernel), plus the
/// ZA transposer of A for `op(A) = Aᵀ` (`op(A) = A` is read in place).
fn one_block_kernel_entries<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (70, 2 * T::NR + 3, 65);
    for ((trans_a, trans_b), expected) in TRANS_PAIRS.into_iter().zip([2_u64, 1, 3, 2]) {
        let call = Call::new(Entry::Transposed, m, n, k).trans(trans_a, trans_b);
        let label = format!("{test}: {trans_a:?}/{trans_b:?}");
        let entries = check_runs_on_sme2::<T>(&call, &label);
        assert_eq!(entries, expected, "{label}: kernel entries");
        let report = run_public::<T>(&call).report;
        let parts = report
            .as_ref()
            .map_or(0, |report| report.layout.parts.len());
        assert_eq!(parts, 1, "{label}: parts");
    }
}

#[test]
fn test_sme2_hook_one_block_kernel_entries_f32() {
    one_block_kernel_entries::<f32>("test_sme2_hook_one_block_kernel_entries_f32");
}

#[test]
fn test_sme2_hook_one_block_kernel_entries_f64() {
    one_block_kernel_entries::<f64>("test_sme2_hook_one_block_kernel_entries_f64");
}

/// A call split over 2 and 3 SME units (the unit count overridden on the
/// calling thread, the parallel threshold lowered), on rayon's global pool
/// and inside a pool of its own: every part runs on SME2, on as many
/// distinct threads as parts, with the NEON bits of the same parallel call.
#[cfg(feature = "parallel")]
fn split_calls_run_on_sme2<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let pool = match rayon::ThreadPoolBuilder::new().num_threads(3).build() {
        Ok(pool) => pool,
        Err(error) => panic!("{test}: no pool: {error}"),
    };
    let (m, n, k) = (200, 6 * T::NR + 5, 96);
    for units in [2, 3] {
        for &(trans_a, trans_b) in &TRANS_PAIRS {
            for (in_pool, par) in [(false, Par::RayonWith(units)), (true, Par::Rayon)] {
                let label = format!("{test}: {units} units {trans_a:?}/{trans_b:?} pool {in_pool}");
                let call = Call::new(Entry::TransposedWithPar, m, n, k)
                    .trans(trans_a, trans_b)
                    .par(par)
                    .seed(units as u64);
                let check = || {
                    // Thread-local switches: set on the thread that calls.
                    let _units = override_units(units);
                    let _tuning = override_tuning(TuningOverride {
                        par_min_macs: Some(1 << 12),
                        ..TuningOverride::default()
                    });
                    let want = run_reference::<T>(&call);
                    let got = run_public::<T>(&call);
                    let kc = call.entry.kc::<T>(m, n, k);
                    assert_sme2_ran(&got, k, kc, &label);
                    assert_same_c(&got, &want, &label);
                    got.report
                };
                let report = if in_pool {
                    pool.install(check)
                } else {
                    check()
                };
                let Some(report) = report else {
                    panic!("{label}: no report");
                };
                assert_eq!(report.layout.parts.len(), units, "{label}: parts");
                if in_pool {
                    // Every part ran on a worker of the pool it was called in.
                    for (index, worker) in report.workers.iter().enumerate() {
                        assert!(
                            worker.is_some_and(|worker| worker < 3),
                            "{label}: part {index} ran on {worker:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_hook_split_calls_run_on_sme2_f32() {
    split_calls_run_on_sme2::<f32>("test_sme2_hook_split_calls_run_on_sme2_f32");
}

#[cfg(feature = "parallel")]
#[test]
fn test_sme2_hook_split_calls_run_on_sme2_f64() {
    split_calls_run_on_sme2::<f64>("test_sme2_hook_split_calls_run_on_sme2_f64");
}

/// Exactly one `thread_ready()` in the decision of a public call: with a
/// readiness countdown of one call, the decision passes and the single part,
/// run on the calling thread, finds the countdown run out and falls back to
/// NEON (same bits); with a countdown of zero calls the decision declines.
fn one_readiness_check_per_decision<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let call = Call::new(Entry::GemmWithPar, 90, 2 * T::NR + 1, 70);
    let want = run_reference::<T>(&call);
    let previous = set_ready_countdown(Some(1));
    let got = run_public::<T>(&call);
    let left = set_ready_countdown(Some(0));
    let declined = run_public::<T>(&call);
    set_ready_countdown(previous);
    assert_eq!(left, Some(0), "{test}: the call did not use its one check");
    assert_eq!(
        got.decisions, 1,
        "{test}: the decision used more than one check"
    );
    let neon = got.report.as_ref().map(|report| report.neon.clone());
    assert_eq!(
        neon,
        Some(vec![true]),
        "{test}: the part did not check its thread"
    );
    assert_same_c(&got, &want, test);
    assert_eq!(
        declined.decisions, 0,
        "{test}: decided without a ready thread"
    );
    assert!(
        declined.report.is_none(),
        "{test}: SME2 ran without a ready thread"
    );
    assert_same_c(&declined, &want, test);
}

#[test]
fn test_sme2_hook_one_readiness_check_per_decision_f32() {
    one_readiness_check_per_decision::<f32>("test_sme2_hook_one_readiness_check_per_decision_f32");
}

#[test]
fn test_sme2_hook_one_readiness_check_per_decision_f64() {
    one_readiness_check_per_decision::<f64>("test_sme2_hook_one_readiness_check_per_decision_f64");
}

/// The explicit-blocking entry points: the caller's KC (fitted to the
/// depth) is the depth step of the SME2 call (its block counts and its bits
/// equal the NEON call with that blocking, and differ from the default
/// blocking's bits), and a degenerate step (fitted KC below 32 while it
/// splits the depth) declines.
fn explicit_blocking_is_the_depth_step<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (90, 3 * T::NR, 300);
    let caller = GemmBlocking {
        mc: 128,
        kc: 64,
        nc: 256,
    };
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        let label = format!("{test}: {trans_a:?}/{trans_b:?}");
        let call =
            Call::new(Entry::TransposedWithBlocking(caller), m, n, k).trans(trans_a, trans_b);
        let kc = call.entry.kc::<T>(m, n, k);
        assert!(kc < k && kc >= 32, "{label}: the fitted KC {kc}");
        check_runs_on_sme2::<T>(&call, &label);
        let got = run_public::<T>(&call);
        let depth_blocks = got
            .report
            .as_ref()
            .and_then(|report| report.layout.parts.first().map(|part| (report, part)))
            .map(|(report, part)| (report.blocks[0].sme, expected_blocks(part, k, kc)));
        let Some((sme, expected)) = depth_blocks else {
            panic!("{label}: no report");
        };
        assert_eq!(sme, expected, "{label}: blocks for KC {kc}");
        assert!(
            sme >= k.div_ceil(kc),
            "{label}: fewer blocks than depth steps"
        );
        // Another KC partition rounds differently: the default blocking's
        // result is not the explicit one's.
        let default = run_public::<T>(&Call {
            entry: Entry::Transposed,
            ..call
        });
        let differs = got
            .c
            .data
            .iter()
            .zip(&default.c.data)
            .any(|(x, y)| x.bits() != y.bits());
        assert!(
            differs,
            "{label}: the KC partition left no trace in the bits"
        );

        // A degenerate step declines: KC 8 against a depth of 300.
        let tiny = GemmBlocking { kc: 8, ..caller };
        let degenerate = Call {
            entry: Entry::TransposedWithBlocking(tiny),
            ..call
        };
        let fitted = degenerate.entry.kc::<T>(m, n, k);
        assert!(fitted < 32 && fitted < k, "{label}: fitted {fitted}");
        let want = run_reference::<T>(&degenerate);
        let got = run_public::<T>(&degenerate);
        assert_eq!(got.decisions, 0, "{label}: decided with KC {fitted}");
        assert!(got.report.is_none(), "{label}: SME2 ran with KC {fitted}");
        assert_same_c(&got, &want, &label);
    }
}

#[test]
fn test_sme2_hook_explicit_blocking_is_the_depth_step_f32() {
    explicit_blocking_is_the_depth_step::<f32>(
        "test_sme2_hook_explicit_blocking_is_the_depth_step_f32",
    );
}

#[test]
fn test_sme2_hook_explicit_blocking_is_the_depth_step_f64() {
    explicit_blocking_is_the_depth_step::<f64>(
        "test_sme2_hook_explicit_blocking_is_the_depth_step_f64",
    );
}

/// The few-rows route of the driver through the public entry points (`m`
/// at most 48 for `f32`, 32 for `f64`, above 32³): SME2 runs with the NEON
/// bits, `op(A) = Aᵀ` included (the driver's column-major copy).
fn few_rows_through_the_hook<T: TestElem>(test: &str) {
    if !backend_or_skip(test, T::NEED) {
        return;
    }
    let (m, n, k) = (T::FEW_ROWS, 8 * T::NR + 3, 100);
    for &(trans_a, trans_b) in &TRANS_PAIRS {
        let label = format!("{test}: {trans_a:?}/{trans_b:?}");
        let call = Call::new(Entry::TransposedWithPar, m, n, k).trans(trans_a, trans_b);
        check_runs_on_sme2::<T>(&call, &label);
    }
}

#[test]
fn test_sme2_hook_few_rows_through_the_hook_f32() {
    few_rows_through_the_hook::<f32>("test_sme2_hook_few_rows_through_the_hook_f32");
}

#[test]
fn test_sme2_hook_few_rows_through_the_hook_f64() {
    few_rows_through_the_hook::<f64>("test_sme2_hook_few_rows_through_the_hook_f64");
}
