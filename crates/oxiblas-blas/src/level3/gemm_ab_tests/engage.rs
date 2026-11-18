//! Engagement: whether a candidate changes what runs on a case.
//!
//! The engine gathers [`Facts`] about every variant's correctness call on the
//! machine; [`engages`] compares a candidate's facts with A's for the
//! group's [`Engage`] kind. The `predicted_*` functions give the same answer
//! without running a product, for the dry run and for the intended targets.

use super::super::gemm::GemmBlocking;
use super::super::gemm_kernel::MicroKernelShape;
use super::super::gemm_plan::{GemmPolicy, GemmRoute, PackScope, RouteInputs, choose_route};
use super::super::gemm_thin::ThinSplits;
use oxiblas_core::parallel::Par;

/// What a group's candidates change, which decides how engagement is seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Engage {
    /// The router's path (route, SIMD small kernel or not) at the host tier.
    Route,
    /// The plan's micro-kernel shape.
    Kernel,
    /// The plan's pack scope, and the 2D driver's runs show it.
    Scope,
    /// NC of the driver that runs the case.
    BCap,
    /// The packing-buffer allocations or the parked bytes after a warm-up.
    CacheLimit,
}

/// The route of a call and whether the SIMD small kernels compute it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Path {
    pub(super) route: GemmRoute,
    pub(super) simd: bool,
}

/// The path `policy` takes for the problem `inp` (see [`choose_route`]).
pub(super) fn path_of(inp: &RouteInputs, policy: &GemmPolicy) -> Path {
    path_on(choose_route(inp, policy), inp, policy)
}

/// The path of a call that takes `route` for the problem `inp` under
/// `policy` (whether the SIMD small kernels compute it).
pub(super) fn path_on(route: GemmRoute, inp: &RouteInputs, policy: &GemmPolicy) -> Path {
    let col_major = inp.a_unit_col_stride && inp.b_unit_col_stride;
    let simd = match route {
        GemmRoute::Small => inp.simd_small(policy) && col_major,
        GemmRoute::SmallTransCopy => true,
        GemmRoute::Thin | GemmRoute::SmallTrans | GemmRoute::Blocked => false,
    };
    Path { route, simd }
}

/// Runs of the 2D parallel driver during one call (counted on the thread
/// that called it).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DriverRuns {
    /// Runs with the shared pack scope.
    pub(super) shared: usize,
    /// Runs with the per-tile pack scope.
    pub(super) per_tile: usize,
    /// Calls the 2D driver declined (fewer than two tiles).
    pub(super) declined: usize,
}

impl DriverRuns {
    /// The 2D driver ran (with either scope).
    pub(super) const fn ran(&self) -> bool {
        self.shared + self.per_tile > 0
    }

    /// The text of the `2d=` detail item.
    pub(super) fn text(&self) -> &'static str {
        match (self.shared > 0, self.per_tile > 0, self.declined > 0) {
            (true, false, false) => "shared",
            (false, true, false) => "per_tile",
            (false, false, true) => "declined",
            (false, false, false) => "none",
            _ => "mixed",
        }
    }
}

/// Packing-buffer use in the steady state (calling thread and every pool
/// worker together).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CacheUse {
    /// Buffers allocated during the probe calls after a warm-up.
    pub(super) allocations: usize,
    /// Bytes left parked after the last probe call.
    pub(super) parked: usize,
}

/// What one GEMM call executed, derived from the plan and the `Par` handed
/// to `gemm_planned` and from the thread the call ran on: the path (route
/// and SIMD small kernel), the micro-kernel shape, KC and the fitted NC of
/// both drivers (with the policy's B-block cap under the limit in force),
/// the pack scope at the `Par`'s thread count, the `Par` and the size of the
/// pool the call ran in.
///
/// Every call of a case records one (see `engine::Problem::call`); the
/// runner requires every warm-up, timed and probe call of a variant to have
/// the fingerprint of that variant's correctness call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Fingerprint {
    pub(super) path: Path,
    pub(super) kernel: MicroKernelShape,
    pub(super) kc: usize,
    /// NC of the sequential driver.
    pub(super) seq_nc: usize,
    /// NC of the 2D parallel driver (the sequential one's without the
    /// `parallel` feature).
    pub(super) par_nc: usize,
    pub(super) scope: PackScope,
    pub(super) par: Par,
    /// Threads of the pool the call ran in (`None`: not in a pool).
    pub(super) pool: Option<usize>,
}

impl Fingerprint {
    /// The fingerprint as text, for error messages.
    pub(super) fn text(&self) -> String {
        let Path { route, simd } = self.path;
        let simd = if simd { "+simd" } else { "" };
        let MicroKernelShape { mr, nr } = self.kernel;
        format!(
            "[route={route:?}{simd} kernel={mr}x{nr} kc={} nc={}/{} scope={:?} par={:?} \
             pool={:?}]",
            self.kc, self.seq_nc, self.par_nc, self.scope, self.par, self.pool
        )
    }
}

/// What one variant's correctness call showed.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Facts {
    /// What the correctness call executed (the router's path on the actual
    /// operands at the host tier, the plan's kernel shape, its pack scope at
    /// the call's thread count, ...): every other call of the variant must
    /// show the same.
    pub(super) ran: Fingerprint,
    /// The 2D driver's runs during the call.
    pub(super) runs: DriverRuns,
    /// The thin route's parallel splits during the call.
    pub(super) thin: ThinSplits,
    /// NC of the driver that ran (the sequential one unless the 2D driver
    /// ran).
    pub(super) nc: usize,
    /// Steady-state packing-buffer use (cache-limit groups only).
    pub(super) cache: Option<CacheUse>,
    /// The result has the bits of A's result.
    pub(super) same_bits_as_base: bool,
    /// The largest `|error| / tolerance` against the reference (route
    /// groups), `None` where the result is compared bit for bit only.
    pub(super) worst: Option<f64>,
    /// Route, kernel, blocking and counters, for the `detail` column.
    pub(super) detail: String,
}

/// Whether a candidate with `cand` facts changes what runs against A's
/// `base` facts, for the group kind `kind`.
pub(super) fn engages(kind: Engage, base: &Facts, cand: &Facts) -> bool {
    match kind {
        Engage::Route => cand.ran.path != base.ran.path,
        Engage::Kernel => cand.ran.kernel != base.ran.kernel,
        Engage::Scope => cand.ran.scope != base.ran.scope && cand.runs != base.runs,
        Engage::BCap => cand.nc != base.nc,
        Engage::CacheLimit => cand.cache != base.cache,
    }
}

/// What the engagement test compared, for the `detail` column:
/// `seen=<A's value>-><candidate's value>`.
pub(super) fn seen(kind: Engage, base: &Facts, cand: &Facts) -> String {
    let pair = |a: String, b: String| format!("seen={a}->{b}");
    let path = |p: &Path| format!("{:?}{}", p.route, if p.simd { "+simd" } else { "" });
    let kernel = |s: &MicroKernelShape| format!("{}x{}", s.mr, s.nr);
    let cache = |c: &Option<CacheUse>| match c {
        Some(use_) => format!("allocs:{},parked:{}", use_.allocations, use_.parked),
        None => "none".to_string(),
    };
    match kind {
        Engage::Route => pair(path(&base.ran.path), path(&cand.ran.path)),
        Engage::Kernel => pair(kernel(&base.ran.kernel), kernel(&cand.ran.kernel)),
        Engage::Scope => pair(
            format!("{:?}:{}", base.ran.scope, base.runs.text()),
            format!("{:?}:{}", cand.ran.scope, cand.runs.text()),
        ),
        Engage::BCap => pair(base.nc.to_string(), cand.nc.to_string()),
        Engage::CacheLimit => pair(cache(&base.cache), cache(&cand.cache)),
    }
}

/// The steady-state packing-buffer use of the sequential blocked driver
/// with the fitted blocks `blocks` (for `m × k` by `k × n`, kernel `shape`,
/// `elem` bytes per element) under the cache limit `limit`, as the cache
/// policy gives it: B is put back before A, a buffer is parked when the
/// bytes parked with it stay within the limit, and every call allocates the
/// buffers that were not parked.
pub(super) fn predicted_cache_use(
    blocks: &GemmBlocking,
    [m, k, n]: [usize; 3],
    shape: &MicroKernelShape,
    elem: usize,
    limit: usize,
) -> CacheUse {
    let (mc, kc, nc) = (blocks.mc.min(m), blocks.kc.min(k), blocks.nc.min(n));
    let a_bytes = mc.div_ceil(shape.mr) * shape.mr * kc * elem;
    let b_bytes = kc * nc.div_ceil(shape.nr) * shape.nr * elem;
    let b_parked = b_bytes <= limit;
    let b_kept = if b_parked { b_bytes } else { 0 };
    let a_parked = a_bytes + b_kept <= limit;
    let a_kept = if a_parked { a_bytes } else { 0 };
    CacheUse {
        allocations: usize::from(!a_parked) + usize::from(!b_parked),
        parked: a_kept + b_kept,
    }
}
