//! The real engine: the GEMM of one case, with the variants' plans, limits
//! and counters.

use super::super::gemm::{BlockingSource, call_blocking, gemm_planned};
use super::super::gemm_kernel::MicroKernelShape;
use super::super::gemm_packing::{
    PACK_CACHE_MAX_BYTES, PackCacheLimitGuard, drain_pack_cache, pack_cache_allocations,
    pack_cache_bytes, pack_cache_limit, set_pack_cache_limit,
};
use super::super::gemm_plan::{
    GemmContext, GemmPlan, GemmPolicy, GemmRoute, LEGACY_POLICY, PackScope, PlanKernelKey,
    RouteInputs,
};
use super::super::gemm_thin::{ThinSplits, thin_split_expected, thin_splits};
use super::check::{
    AbReal, Operands, Reference, SEED_A, SEED_B, check_bitwise, check_reference, check_streamed,
    initial_c, operand, reference,
};
use super::engage::{CacheUse, DriverRuns, Engage, Facts, Fingerprint, Path, path_on};
use super::groups::{Case, Dims, GroupSpec, MIB, Variant};
use super::sampler::{CallCount, CheckKind, Engine, Entry, Slot, count_calls};
use super::specs::Beta;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;
use std::cell::RefCell;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use super::super::{
    gemm::PARALLEL_GATE,
    gemm_parallel::{scope_declines, scope_runs},
};

/// Calls of the cache probe after the warm-up (the correctness call).
const PROBE_CALLS: usize = 2;

thread_local! {
    /// The packing-cache limit guard this thread holds for the A/B run.
    static HELD_LIMIT: RefCell<Option<PackCacheLimitGuard>> = const { RefCell::new(None) };
}

/// Sets this thread's packing-cache limit to `limit` until the next call
/// (`None`: the limit from before the run). The previous guard is dropped
/// first, so it restores the earlier limit.
pub(super) fn hold_cache_limit(limit: Option<usize>) {
    HELD_LIMIT.with(|held| {
        let mut held = held.borrow_mut();
        *held = None;
        *held = limit.map(set_pack_cache_limit);
    });
}

/// Where a group run executes: the test thread, or an installed pool.
#[derive(Default)]
pub(super) struct Env<'p> {
    #[cfg(feature = "parallel")]
    pub(super) pool: Option<&'p rayon::ThreadPool>,
    #[cfg(not(feature = "parallel"))]
    pub(super) _pool: core::marker::PhantomData<&'p ()>,
}

impl Env<'_> {
    /// Runs `f` where the GEMM calls run (inside the pool when there is one).
    pub(super) fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            return pool.install(f);
        }
        f()
    }

    /// `f` on the calling thread, then on every worker of the pool.
    pub(super) fn on_every_thread<R: Send>(&self, f: impl Fn() -> R + Sync) -> Vec<R> {
        let first = f();
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            let mut out = vec![first];
            out.extend(pool.broadcast(|_| f()));
            return out;
        }
        vec![first]
    }

    /// Sets the packing-cache limit on the calling thread and on every
    /// worker of the pool.
    pub(super) fn hold(&self, limit: Option<usize>) {
        hold_cache_limit(limit);
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            pool.broadcast(|_| hold_cache_limit(limit));
        }
    }
}

/// What one call was handed, as far as it decides what the call executes:
/// the plan's context (tier and policy), kernel shape and forced route and
/// scope, and the `Par`. Calls with equal keys made on one thread under one
/// packing-cache limit have equal fingerprints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlanKey {
    ctx: GemmContext,
    kernel: PlanKernelKey,
    forced_route: Option<GemmRoute>,
    forced_scope: Option<PackScope>,
    par: Par,
}

impl PlanKey {
    /// Whether `plan` and `par` are what this key describes (field by field,
    /// without building a key).
    #[inline]
    fn describes<T>(&self, plan: &GemmPlan<T>, par: Par) -> bool {
        self.par == par
            && self.kernel == plan.kernel_key()
            && self.forced_route == plan.forced_route
            && self.forced_scope == plan.forced_scope
            && self.ctx == plan.ctx
    }
}

/// A plan and the `Par` one or more calls were handed.
struct Handed<T> {
    key: PlanKey,
    plan: GemmPlan<T>,
}

/// The distinct plans a run of calls was handed, with how many calls had
/// each (normally one entry).
struct HandedTally<T> {
    seen: Vec<(Handed<T>, usize)>,
}

impl<T: AbReal> HandedTally<T> {
    /// An empty tally with room for one plan, so that adding the first one
    /// does not allocate.
    fn new() -> Self {
        Self {
            seen: Vec::with_capacity(1),
        }
    }

    /// Counts one call that was handed `plan` and `par`. A call handed what
    /// an earlier one was handed only compares with it (in place: inside a
    /// timed loop this is the whole cost of the record); a new plan is
    /// copied.
    #[inline]
    fn add(&mut self, plan: &GemmPlan<T>, par: Par) {
        let known = self.seen.iter_mut().find(|s| s.0.key.describes(plan, par));
        match known {
            Some(seen) => seen.1 += 1,
            None => {
                let key = PlanKey {
                    ctx: plan.ctx,
                    kernel: plan.kernel_key(),
                    forced_route: plan.forced_route,
                    forced_scope: plan.forced_scope,
                    par,
                };
                self.seen.push((Handed { key, plan: *plan }, 1));
            }
        }
    }
}

/// The operands and the `Par` of one case.
struct Problem<'a, T: AbReal> {
    ops: Operands<'a, T>,
    dims: Dims,
    beta: T,
    par: Par,
}

impl<T: AbReal> Problem<'_, T> {
    /// One call of `variant`, the only place the harness calls the GEMM: the
    /// plan is built for the call as the public entry points build theirs
    /// ([`Variant::plan`]), then `C = op(A)·op(B) + beta·C` runs with it and
    /// this problem's `Par` (alpha 1, default blocking). The plan and the
    /// `Par` the call was handed (the same values, bound once) are counted
    /// in `tally`.
    #[inline]
    fn call(&self, variant: &Variant, c: &mut Mat<T>, tally: &mut HandedTally<T>) {
        let plan = variant.plan::<T>(self.dims);
        let par = self.par;
        let ops = &self.ops;
        let (a, b) = (ops.a.as_ref(), ops.b.as_ref());
        let blocking = BlockingSource::Default;
        let (ta, tb) = (ops.trans_a, ops.trans_b);
        gemm_planned(
            ta,
            tb,
            T::one(),
            a,
            b,
            self.beta,
            c.as_mut(),
            par,
            blocking,
            &plan,
        );
        tally.add(&plan, par);
    }

    /// The fingerprint of a call of this problem that was handed `handed`,
    /// on the calling thread (its pool, its packing-cache limit, which the
    /// B-block cap reads): the route the plan takes for these operands, the
    /// blocks `call_blocking` fits, the scope at the `Par`'s thread count.
    fn fingerprint(&self, handed: &Handed<T>) -> Fingerprint {
        let (plan, par) = (&handed.plan, handed.key.par);
        let Dims { m, k, n } = self.dims;
        let ops = &self.ops;
        let (a, b) = (ops.a.as_ref(), ops.b.as_ref());
        let tier = plan.ctx.tier();
        let inp = RouteInputs::new(ops.trans_a, ops.trans_b, &a, &b, m, n, k, tier);
        let blocks = call_blocking(BlockingSource::Default, plan, m, k, n);
        let seq = *blocks.sequential();
        #[cfg(feature = "parallel")]
        let par_nc = blocks.parallel().nc;
        #[cfg(not(feature = "parallel"))]
        let par_nc = seq.nc;
        Fingerprint {
            path: path_on(plan.route(&inp), &inp, &plan.ctx.policy),
            kernel: plan.micro_kernel().shape,
            kc: seq.kc,
            seq_nc: seq.nc,
            par_nc,
            scope: plan.scope(par.num_threads()),
            par,
            pool: current_pool_threads(),
        }
    }

    /// The fingerprints of the calls in `tally` (made on this thread).
    fn fingerprints(&self, tally: &HandedTally<T>) -> Vec<(Fingerprint, usize)> {
        let seen = tally.seen.iter();
        seen.map(|(handed, calls)| (self.fingerprint(handed), *calls))
            .collect()
    }
}

/// The problem of `case` on the stored operands, called with `par`.
fn problem_of<'x, T: AbReal>(
    case: &Case,
    par: Par,
    a: &'x Mat<T>,
    b: &'x Mat<T>,
    c0: Option<&'x Mat<T>>,
) -> Problem<'x, T> {
    let ops = Operands {
        trans_a: case.trans.a,
        trans_b: case.trans.b,
        a,
        b,
        c0,
    };
    let beta = match case.beta {
        Beta::Zero => T::zero(),
        Beta::One => T::one(),
    };
    Problem {
        ops,
        dims: case.dims,
        beta,
        par,
    }
}

/// The 2D driver's runs of this thread so far.
fn driver_runs() -> DriverRuns {
    #[cfg(feature = "parallel")]
    {
        let (shared, per_tile) = scope_runs();
        DriverRuns {
            shared,
            per_tile,
            declined: scope_declines(),
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        DriverRuns::default()
    }
}

/// `after − before`, counter by counter.
fn runs_between(before: DriverRuns, after: DriverRuns) -> DriverRuns {
    DriverRuns {
        shared: after.shared - before.shared,
        per_tile: after.per_tile - before.per_tile,
        declined: after.declined - before.declined,
    }
}

/// What the correctness call saw on the thread that made it.
#[derive(Debug, Clone, Copy)]
struct CallSeen {
    /// What the call executed.
    ran: Fingerprint,
    runs: DriverRuns,
    thin: ThinSplits,
    limit: usize,
}

/// The pool the calling code runs in, if any.
fn current_pool_threads() -> Option<usize> {
    #[cfg(feature = "parallel")]
    {
        rayon::current_thread_index().map(|_| rayon::current_num_threads())
    }
    #[cfg(not(feature = "parallel"))]
    {
        None
    }
}

/// The real engine of one case of `spec`.
pub(super) struct GemmEngine<'a, 'p, T: AbReal> {
    spec: &'a GroupSpec,
    case: Case,
    env: &'a Env<'p>,
    a: Mat<T>,
    b: Mat<T>,
    c0: Option<Mat<T>>,
    reference: Option<Reference>,
    base_result: Option<Mat<T>>,
    timed_c: Mat<T>,
    /// The calls made since the last `take_calls`, per variant name and
    /// fingerprint (this replaces a count per name). Every call goes through
    /// `Problem::call`, which records the plan and the `Par` it handed to
    /// `gemm_planned`; the fingerprint is computed from those on the thread
    /// of the call. So a warm-up, timed or probe call that ran with another
    /// route, kernel shape, NC, pack scope, `Par` or pool than its variant's
    /// correctness call is recorded with another fingerprint, and the runner
    /// fails the run (`runner::check_calls`). Not seen: a plan of another
    /// policy that agrees with the checked one in all of these; the
    /// candidates of every group differ from A in fields that show in them
    /// (the packing-cache limit, which is no plan field, is checked on every
    /// thread before every timed sample).
    calls: Vec<CallCount>,
}

impl<'a, 'p, T: AbReal> GemmEngine<'a, 'p, T> {
    /// The engine of `case`: its operands built, nothing computed yet.
    pub(super) fn new(spec: &'a GroupSpec, case: &Case, env: &'a Env<'p>) -> Self {
        let Dims { m, k, n } = case.dims;
        let (ta, tb) = (case.trans.a, case.trans.b);
        let c0 = (case.beta == Beta::One).then(|| initial_c::<T>(m, n, Beta::One));
        Self {
            spec,
            case: case.clone(),
            env,
            a: operand::<T>(ta, m, k, SEED_A),
            b: operand::<T>(tb, k, n, SEED_B),
            timed_c: initial_c::<T>(m, n, case.beta),
            c0,
            reference: None,
            base_result: None,
            calls: Vec::new(),
        }
    }

    fn problem(&self) -> Problem<'_, T> {
        problem_of(
            &self.case,
            self.spec.par,
            &self.a,
            &self.b,
            self.c0.as_ref(),
        )
    }

    /// Records the calls of `variant` with their fingerprints.
    fn record(&mut self, variant: &Variant, calls: &[(Fingerprint, usize)]) {
        for &(ran, count) in calls {
            count_calls(&mut self.calls, variant.name, ran, count);
        }
    }

    /// The correctness call of `variant` into a fresh C, on the thread that
    /// runs the products, with what it saw.
    fn correctness_call(&self, variant: &Variant) -> Result<(Mat<T>, CallSeen), String> {
        let Dims { m, n, .. } = self.case.dims;
        let mut out = match &self.c0 {
            Some(c0) => c0.clone(),
            None => initial_c::<T>(m, n, Beta::Zero),
        };
        let problem = self.problem();
        let seen = {
            let out = &mut out;
            self.env.run(move || {
                let mut tally = HandedTally::new();
                let (runs_before, thin_before) = (driver_runs(), thin_splits());
                problem.call(variant, out, &mut tally);
                let (runs_after, thin_after) = (driver_runs(), thin_splits());
                let ran = match problem.fingerprints(&tally).as_slice() {
                    &[(ran, 1)] => ran,
                    other => return Err(format!("the correctness call recorded {other:?}")),
                };
                Ok(CallSeen {
                    ran,
                    runs: runs_between(runs_before, runs_after),
                    thin: ThinSplits {
                        gated: thin_after.gated - thin_before.gated,
                        split: thin_after.split - thin_before.split,
                    },
                    limit: pack_cache_limit(),
                })
            })
        };
        Ok((out, seen?))
    }

    /// The parallel checks of a correctness call: the pool size, and that
    /// the call took the parallel path exactly where the gates say (the
    /// expectation comes from the group's thread count, not from the `Par`
    /// of the call).
    fn check_parallel(&self, seen: &CallSeen) -> Result<(), String> {
        let spec = self.spec;
        let want_pool = spec.parallel.then_some(spec.threads);
        if seen.ran.pool != want_pool {
            return Err(format!(
                "ran in a pool of {:?} threads, expected {want_pool:?}",
                seen.ran.pool
            ));
        }
        let work = usize::try_from(self.case.dims.work()).unwrap_or(usize::MAX);
        let expected = spec.expected_par();
        let runs = seen.runs;
        let blocked = matches!(seen.ran.path.route, GemmRoute::Blocked);
        let want_2d = blocked && blocked_gate(work, expected);
        let attempts = runs.shared + runs.per_tile + runs.declined;
        if attempts != usize::from(want_2d) {
            return Err(format!(
                "2D driver runs {runs:?}, expected {} (gate on {expected:?})",
                usize::from(want_2d)
            ));
        }
        let scope_ok = match seen.ran.scope {
            PackScope::Shared => runs.per_tile == 0,
            PackScope::PerTile => runs.shared == 0,
        };
        if !scope_ok {
            return Err(format!(
                "2D driver runs {runs:?} against the scope {:?}",
                seen.ran.scope
            ));
        }
        let Dims { m, n, .. } = self.case.dims;
        let thin = seen.ran.path.route == GemmRoute::Thin;
        let policy = self.thin_gate_policy();
        let gated = thin
            && policy
                .thin_par_threshold()
                .should_parallelize(work, expected);
        let y_len = if n == 1 { m } else { n };
        let split = gated && thin_split_expected(y_len, expected.num_threads());
        let want = ThinSplits {
            gated: usize::from(gated),
            split: usize::from(split),
        };
        if seen.thin != want {
            return Err(format!("thin splits {:?}, expected {want:?}", seen.thin));
        }
        Ok(())
    }

    /// The policy whose thin gate applies (every variant shares it).
    fn thin_gate_policy(&self) -> GemmPolicy {
        self.spec
            .variants
            .first()
            .map_or(LEGACY_POLICY, |v| v.policy)
    }

    /// Packing-buffer use in the steady state of `variant`: after the
    /// correctness call (the warm-up), the buffers allocated during
    /// [`PROBE_CALLS`] more calls and the bytes parked after them, on the
    /// calling thread and every worker together. The probe calls are
    /// recorded with their fingerprints like any other call.
    fn cache_probe(&mut self, variant: &Variant) -> CacheUse {
        let count = |env: &Env<'_>| {
            env.on_every_thread(pack_cache_allocations)
                .iter()
                .sum::<usize>()
        };
        let before = count(self.env);
        let problem = self.problem();
        let mut c = self.timed_c.clone();
        let calls = {
            let (c, problem) = (&mut c, &problem);
            self.env.run(move || {
                let mut tally = HandedTally::new();
                for _ in 0..PROBE_CALLS {
                    problem.call(variant, c, &mut tally);
                }
                problem.fingerprints(&tally)
            })
        };
        let after = count(self.env);
        let parked = self.env.on_every_thread(pack_cache_bytes).iter().sum();
        self.record(variant, &calls);
        CacheUse {
            allocations: after - before,
            parked,
        }
    }

    /// The result check of `entry`'s correctness call, `out`: against the
    /// reference ([`CheckKind::Reference`]: route groups keep the reference
    /// for every variant, the bitwise groups check only A and stream it) or
    /// bit for bit against A's result. Returns the largest
    /// `|error| / tolerance` of a reference check.
    pub(super) fn check_result(
        &mut self,
        entry: &Entry,
        out: &Mat<T>,
    ) -> Result<Option<f64>, String> {
        let (case, par) = (&self.case, self.spec.par);
        let ops = problem_of(case, par, &self.a, &self.b, self.c0.as_ref()).ops;
        let mkn = case.dims.mkn();
        match entry.check {
            CheckKind::Reference if self.spec.engage == Engage::Route => {
                let built = self.reference.get_or_insert_with(|| reference(&ops, mkn));
                check_reference(out, built).map(Some)
            }
            CheckKind::Reference => check_streamed(&ops, mkn, out).map(Some),
            CheckKind::Bitwise => match &self.base_result {
                Some(base) => check_bitwise(out, base).map(|()| None),
                None => Err("no result of A to compare with bit for bit".to_string()),
            },
        }
    }
}

/// The gate of the 2D parallel driver for `work` multiply-adds under `par`.
fn blocked_gate(work: usize, par: Par) -> bool {
    #[cfg(feature = "parallel")]
    {
        PARALLEL_GATE.should_parallelize(work, par)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (work, par);
        false
    }
}

/// The `detail` text of a correctness call.
fn detail_of(seen: &CallSeen, cache: Option<CacheUse>) -> String {
    let ran = &seen.ran;
    let Path { route, simd } = ran.path;
    let simd = if simd { "+simd" } else { "" };
    let MicroKernelShape { mr, nr } = ran.kernel;
    let mut text = format!(
        "route={route:?}{simd} kernel={mr}x{nr} kc={} nc={}/{} scope={:?} 2d={} thin={}/{} \
         limit={}MiB",
        ran.kc,
        ran.seq_nc,
        ran.par_nc,
        ran.scope,
        seen.runs.text(),
        seen.thin.gated,
        seen.thin.split,
        seen.limit / MIB,
    );
    if let Some(use_) = cache {
        text += &format!(" allocs={} parked={}", use_.allocations, use_.parked);
    }
    if let Some(threads) = ran.pool {
        text += &format!(" pool={threads}");
    }
    text
}

impl<T: AbReal> Engine for GemmEngine<'_, '_, T> {
    fn start_clean(&mut self) -> Result<(), String> {
        let parked = self.env.on_every_thread(|| {
            drain_pack_cache();
            pack_cache_bytes()
        });
        if parked.iter().any(|&bytes| bytes != 0) {
            return Err(format!(
                "packing caches not empty after the drain: {parked:?}"
            ));
        }
        Ok(())
    }

    fn check(&mut self, entry: &Entry) -> Result<Facts, String> {
        let variant = entry.variant;
        self.env.hold(variant.limit);
        let (out, seen) = self.correctness_call(&variant)?;
        self.record(&variant, &[(seen.ran, 1)]);
        self.check_parallel(&seen)?;
        let want_limit = variant.limit.unwrap_or(PACK_CACHE_MAX_BYTES);
        if seen.limit != want_limit {
            return Err(format!(
                "limit {} in force, expected {want_limit}",
                seen.limit
            ));
        }
        let same_bits_as_base = match &self.base_result {
            Some(base) => check_bitwise(&out, base).is_ok(),
            None => true,
        };
        let worst = self.check_result(entry, &out)?;
        if entry.slot == Slot::Base {
            self.base_result = Some(out);
        }
        let cache = (self.spec.engage == Engage::CacheLimit).then(|| self.cache_probe(&variant));
        let nc = if seen.runs.ran() {
            seen.ran.par_nc
        } else {
            seen.ran.seq_nc
        };
        Ok(Facts {
            ran: seen.ran,
            runs: seen.runs,
            thin: seen.thin,
            nc,
            cache,
            same_bits_as_base,
            worst,
            detail: detail_of(&seen, cache),
        })
    }

    fn apply_limit(&mut self, limit: Option<usize>) {
        self.env.hold(limit);
    }

    fn take_calls(&mut self) -> Vec<CallCount> {
        std::mem::take(&mut self.calls)
    }

    fn verify_limit(&mut self, limit: Option<usize>) -> Result<(), String> {
        let want = limit.unwrap_or(PACK_CACHE_MAX_BYTES);
        let seen = self.env.on_every_thread(pack_cache_limit);
        if seen.iter().any(|&l| l != want) {
            return Err(format!(
                "limits in force {seen:?}, expected {want} everywhere"
            ));
        }
        Ok(())
    }

    fn call(&mut self, variant: &Variant) {
        let problem = problem_of(
            &self.case,
            self.spec.par,
            &self.a,
            &self.b,
            self.c0.as_ref(),
        );
        let c = &mut self.timed_c;
        let calls = self.env.run(move || {
            let mut tally = HandedTally::new();
            problem.call(variant, c, &mut tally);
            problem.fingerprints(&tally)
        });
        self.record(variant, &calls);
    }

    fn time(&mut self, variant: &Variant, reps: usize) -> Duration {
        let problem = problem_of(
            &self.case,
            self.spec.par,
            &self.a,
            &self.b,
            self.c0.as_ref(),
        );
        let c = &mut self.timed_c;
        let (elapsed, calls) = self.env.run(move || {
            // Per call the plan and `Par` are only compared in place with
            // the previous call's (the same work for every variant); the
            // fingerprint of each distinct plan is computed after the clock
            // stopped, on this thread.
            let mut tally = HandedTally::new();
            let start = Instant::now();
            for _ in 0..reps {
                problem.call(variant, c, &mut tally);
            }
            let elapsed = start.elapsed();
            std::hint::black_box(&*c);
            (elapsed, problem.fingerprints(&tally))
        });
        self.record(variant, &calls);
        elapsed
    }
}
