//! Per-call GEMM plan: SIMD tier, routing policy, micro-kernel and route.
//!
//! Every real GEMM entry point ([`gemm_transposed_with_par`],
//! [`gemm_transposed_with_blocking`] and everything built on them) builds one
//! [`GemmPlan`] per call and hands it to the internal `gemm_planned` funnel:
//!
//! - [`SimdTier`] is derived from the *capped* micro-kernel kind
//!   (`selected_kernel_kind`), so every SIMD path keyed off the tier honours
//!   the `force-scalar`, `max-simd-128` and `max-simd-256` features exactly
//!   like the micro-kernels do.
//! - [`GemmPolicy`] says which optional paths a call may take.
//!   [`LEGACY_POLICY`] reproduces 0.2.3 exactly; the per-architecture
//!   candidates hold the new behaviour. [`DEFAULT_POLICY`] is the legacy
//!   policy unless the crate is built with `--cfg oxiblas_gemm_candidate`,
//!   which selects the candidate of the target architecture (for A/B runs of
//!   the same source). `--cfg oxiblas_gemm_bcap_all` additionally makes the
//!   candidates cap the B block in both blocked drivers ([`BCapScope::All`]).
//! - [`GemmContext`] is the non-generic part (tier and policy) that the route
//!   implementations receive.
//! - [`choose_route`] is a pure function of the problem description
//!   ([`RouteInputs`]) and the policy.
//!
//! [`gemm_transposed_with_par`]: super::gemm::gemm_transposed_with_par
//! [`gemm_transposed_with_blocking`]: super::gemm::gemm_transposed_with_blocking

use super::autotune::{BBlockCaps, ISSUE_23_B_CAPS};
use super::gemm_kernel::{GemmKernel, MicroKernel, kernel_for_problem, selected_kernel_kind};
use super::gemm_small::SMALL_THRESHOLD;
use super::trsm::Trans;
use core::any::TypeId;
use oxiblas_core::parallel::ParThreshold;
use oxiblas_core::scalar::Field;
use oxiblas_core::simd::dispatch::{GemmKernelKind, has_avx2_fma};
use oxiblas_matrix::MatRef;

/// SIMD tier of one GEMM call, derived from the capped micro-kernel kind.
///
/// Every variant exists on every target (the routing tables are pure and
/// testable everywhere); the SIMD code behind a tier is compiled only for the
/// architecture that has it. [`SimdTier::Avx2Fma`] is produced only when the
/// CPU reports AVX2 and FMA and the build allows 256-bit vectors, so code
/// keyed off it may rely on both features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimdTier {
    /// No SIMD kernels (`force-scalar`, or no supported SIMD).
    Portable,
    /// x86-64 with SSE4.2 but without AVX2 + FMA (or capped to 128 bits).
    Sse42,
    /// aarch64 NEON.
    Neon,
    /// x86-64 with AVX2 and FMA (also on AVX-512 hosts).
    Avx2Fma,
}

impl SimdTier {
    /// The tier of this build and CPU.
    pub(crate) fn current() -> Self {
        Self::from_kind(selected_kernel_kind(), has_avx2_fma())
    }

    /// Maps a (capped) micro-kernel kind to a tier.
    ///
    /// `avx2_fma` is whether the CPU reports both AVX2 and FMA under the
    /// build's SIMD ceiling; the AVX2 and AVX-512 kinds map to
    /// [`SimdTier::Avx2Fma`] only when it holds (otherwise to
    /// [`SimdTier::Sse42`], which uses no AVX instruction).
    pub(crate) const fn from_kind(kind: GemmKernelKind, avx2_fma: bool) -> Self {
        match kind {
            GemmKernelKind::Avx512 | GemmKernelKind::Avx2 => {
                if avx2_fma {
                    Self::Avx2Fma
                } else {
                    Self::Sse42
                }
            }
            GemmKernelKind::Sse42 => Self::Sse42,
            GemmKernelKind::Neon => Self::Neon,
            GemmKernelKind::Scalar => Self::Portable,
        }
    }
}

/// Test gate for code that runs only on the [`SimdTier::Avx2Fma`] tier.
///
/// Returns `true` when this build and CPU are on that tier. Otherwise it
/// prints one `SKIPPED` line and returns `false`, so the calling test returns
/// early. On x86-64, when the environment variable `OXIBLAS_REQUIRE_AVX2` is
/// set (to anything but an empty string or `0`) and the build does not cap
/// SIMD below 256 bits, a missing tier is a failure instead of a skip: a run
/// that is meant to cover the AVX2 kernels cannot pass by skipping them.
#[cfg(test)]
pub(crate) fn avx2_fma_tier_or_skip(test: &str) -> bool {
    if SimdTier::current() == SimdTier::Avx2Fma {
        return true;
    }
    let capped = cfg!(any(feature = "force-scalar", feature = "max-simd-128"));
    let required = cfg!(target_arch = "x86_64")
        && !capped
        && std::env::var("OXIBLAS_REQUIRE_AVX2").is_ok_and(|v| !v.is_empty() && v != "0");
    assert!(
        !required,
        "{test}: OXIBLAS_REQUIRE_AVX2 is set, but this run is not on the AVX2+FMA tier"
    );
    println!("{test}: SKIPPED (not on the AVX2+FMA tier)");
    false
}

/// Where the issue #23 B-block cap ([`BBlockCaps`]) applies.
///
/// Calls with explicit (caller-chosen) blocking are never capped, whatever
/// the policy says; see `gemm::BlockingSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BCapScope {
    /// Nowhere (0.2.3).
    Off,
    /// In the sequential blocked driver; the 2D parallel driver keeps the
    /// fitted NC.
    Sequential,
    /// In both blocked drivers: the cap is applied before the driver is
    /// chosen, so the 2D parallel driver gets the capped NC too. The
    /// candidates use it only when built with `--cfg oxiblas_gemm_bcap_all`
    /// (A/B runs; see [`CANDIDATE_B_CAP`]).
    All,
}

/// Which f32 micro-kernel a plan carries on the AVX2 + FMA tier.
///
/// The canonical f32 kernel of that tier (`f32::micro_kernel_dispatch()`)
/// is the 8×8 kernel under every policy; the policy only chooses the kernel
/// of one call's plan (`gemm_kernel::kernel_for_problem`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum F32Avx2Policy {
    /// Always the canonical 8×8 kernel (0.2.3).
    Legacy8x8,
    /// The kernel shape chosen per problem (16×6 where it pays off).
    Adaptive,
}

/// How the 2D parallel driver keeps its packed panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackScope {
    /// One packed B block per KC step, shared by every tile (0.2.3).
    Shared,
    /// Every tile packs its own part of B (and A) on the worker that runs it.
    PerTile,
}

/// Which optional GEMM paths a call may take.
///
/// Each field is one switch or threshold; a policy never changes what is
/// computed, only the route (and with it the rounding of the last bits) and
/// the block sizes. The three policies below list the fields in the same
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmPolicy {
    /// `m == 1` or `n == 1` (`f64`/`f32`, large operand with a unit stride)
    /// runs as one GEMV (`gemm_thin`).
    pub(crate) thin: bool,
    /// The work gate of the parallel thin path, as the two numbers of a
    /// [`ParThreshold`]: the least total work (`m·n·k`) and the least work
    /// per thread (see [`GemmPolicy::thin_par_threshold`]).
    pub(crate) thin_par_gate: (usize, usize),
    /// The AVX2 + FMA tier has SIMD small kernels (`m >= 5`, `n >= 4`).
    pub(crate) small_simd_x86: bool,
    /// Largest `m·n·k` that `NoTrans`/`NoTrans` problems with column-major
    /// operands send to the tier's SIMD small kernels. Problems up to
    /// [`SMALL_THRESHOLD`] take the small path in any case.
    pub(crate) small_simd_max_work: usize,
    /// `[min, max]` of `m·n·k` for which transposed or strided problems of a
    /// tier with SIMD small kernels are copied to column-major scratch and run
    /// on those kernels (`gemm_small_trans_copy`); `None` never copies.
    pub(crate) small_trans_copy: Option<(usize, usize)>,
    /// The f32 micro-kernel of a plan on the AVX2 + FMA tier.
    pub(crate) f32_avx2: F32Avx2Policy,
    /// Thread count from which the 2D parallel driver uses
    /// [`PackScope::PerTile`]; `usize::MAX` never does.
    pub(crate) per_tile_min_threads: usize,
    /// Where the B-block cap applies.
    pub(crate) b_cap: BCapScope,
    /// The B-block cap rule.
    pub(crate) b_caps: BBlockCaps,
}

impl GemmPolicy {
    /// The work gate of the parallel thin path ([`GemmPolicy::thin_par_gate`]).
    pub(crate) const fn thin_par_threshold(&self) -> ParThreshold {
        ParThreshold::new(self.thin_par_gate.0, self.thin_par_gate.1)
    }
}

/// Work gate of the parallel thin path in every policy: at least 512²
/// multiply-adds in total and 128² per thread.
const THIN_PAR_GATE: (usize, usize) = (512 * 512, 128 * 128);

/// The policy that reproduces 0.2.3 exactly: no thin route, the small path up
/// to [`SMALL_THRESHOLD`] (with the NEON small kernels on aarch64) and no
/// copy, the canonical f32 kernel, the shared pack scope, everything else
/// blocked with the fitted NC.
pub(crate) const LEGACY_POLICY: GemmPolicy = GemmPolicy {
    thin: false,
    thin_par_gate: THIN_PAR_GATE,
    small_simd_x86: false,
    small_simd_max_work: SMALL_THRESHOLD,
    small_trans_copy: None,
    f32_avx2: F32Avx2Policy::Legacy8x8,
    per_tile_min_threads: usize::MAX,
    b_cap: BCapScope::Off,
    b_caps: ISSUE_23_B_CAPS,
};

/// `m·n·k` range of the small transposed copy in the candidates: from 8³ up to
/// the small threshold.
const SMALL_TRANS_COPY_RANGE: (usize, usize) = (8 * 8 * 8, SMALL_THRESHOLD);

/// The B-block cap scope of both candidates: [`BCapScope::Sequential`], or
/// [`BCapScope::All`] when built with `--cfg oxiblas_gemm_bcap_all` (so the
/// `All` variant can be measured on its own, from the same source).
const CANDIDATE_B_CAP: BCapScope = if cfg!(oxiblas_gemm_bcap_all) {
    BCapScope::All
} else {
    BCapScope::Sequential
};

/// Candidate policy for x86-64 (enabled field by field after A/B
/// measurement).
pub(crate) const CANDIDATE_X86: GemmPolicy = GemmPolicy {
    thin: true,
    thin_par_gate: THIN_PAR_GATE,
    small_simd_x86: true,
    small_simd_max_work: 80 * 80 * 80,
    small_trans_copy: Some(SMALL_TRANS_COPY_RANGE),
    f32_avx2: F32Avx2Policy::Adaptive,
    per_tile_min_threads: 5,
    b_cap: CANDIDATE_B_CAP,
    b_caps: ISSUE_23_B_CAPS,
};

/// Candidate policy for aarch64 (enabled field by field after A/B
/// measurement).
pub(crate) const CANDIDATE_AARCH64: GemmPolicy = GemmPolicy {
    thin: true,
    thin_par_gate: THIN_PAR_GATE,
    small_simd_x86: false,
    small_simd_max_work: SMALL_THRESHOLD,
    small_trans_copy: Some(SMALL_TRANS_COPY_RANGE),
    f32_avx2: F32Avx2Policy::Legacy8x8,
    per_tile_min_threads: usize::MAX,
    b_cap: CANDIDATE_B_CAP,
    b_caps: ISSUE_23_B_CAPS,
};

/// The candidate policy of the target architecture. Targets without a
/// candidate keep [`LEGACY_POLICY`]: nothing has been measured there.
pub(crate) const CANDIDATE_POLICY: GemmPolicy = if cfg!(target_arch = "x86_64") {
    CANDIDATE_X86
} else if cfg!(target_arch = "aarch64") {
    CANDIDATE_AARCH64
} else {
    LEGACY_POLICY
};

/// The policy of the public GEMM entry points: [`LEGACY_POLICY`], or
/// [`CANDIDATE_POLICY`] when built with `--cfg oxiblas_gemm_candidate`.
pub(crate) const DEFAULT_POLICY: GemmPolicy = if cfg!(oxiblas_gemm_candidate) {
    CANDIDATE_POLICY
} else {
    LEGACY_POLICY
};

/// The non-generic part of a GEMM plan: the SIMD tier and the policy.
///
/// The tier can only be the detected one or [`SimdTier::Portable`], so a
/// route implementation may use the tier's SIMD kernels without further
/// checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmContext {
    /// The SIMD tier (detected, or lowered to portable).
    tier: SimdTier,
    /// The routing policy.
    pub(crate) policy: GemmPolicy,
}

impl GemmContext {
    /// The context of this build and CPU for `policy`.
    pub(crate) fn detect(policy: GemmPolicy) -> Self {
        Self {
            tier: SimdTier::current(),
            policy,
        }
    }

    /// The SIMD tier.
    pub(crate) const fn tier(&self) -> SimdTier {
        self.tier
    }

    /// This context with the portable tier (no SIMD kernels outside the
    /// micro-kernel).
    #[cfg(test)]
    pub(crate) const fn with_portable_tier(mut self) -> Self {
        self.tier = SimdTier::Portable;
        self
    }
}

/// The implementation that computes one GEMM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmRoute {
    /// `gemm_thin`: `m == 1` or `n == 1` as one GEMV.
    Thin,
    /// `gemm_small_with`: `NoTrans`/`NoTrans` without packing (SIMD small
    /// kernels where the tier has them).
    Small,
    /// `gemm_small_trans_copy`: `op(A)` / `op(B)` copied to column-major
    /// scratch, then `gemm_small_with`.
    SmallTransCopy,
    /// `gemm_small_trans`: transpose-aware loops without packing.
    SmallTrans,
    /// The packed, blocked driver (sequential or 2D parallel).
    Blocked,
}

impl GemmRoute {
    /// Whether this route computes the problem described by `inp` correctly.
    ///
    /// `Thin` needs a thin `f64`/`f32` problem whose large operand has a unit
    /// stride, and `Small` reads both operands as stored, so it needs
    /// `NoTrans` on both; every other route handles any operand.
    pub(crate) fn accepts(self, inp: &RouteInputs) -> bool {
        match self {
            Self::Thin => inp.thin_ok(),
            Self::Small => inp.both_notrans(),
            Self::SmallTransCopy | Self::SmallTrans | Self::Blocked => true,
        }
    }
}

/// What the router needs to know about one GEMM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouteInputs {
    /// Rows of `op(A)` and C.
    pub(crate) m: usize,
    /// Columns of `op(B)` and C.
    pub(crate) n: usize,
    /// Columns of `op(A)`, rows of `op(B)`.
    pub(crate) k: usize,
    /// Transpose of A.
    pub(crate) trans_a: Trans,
    /// Transpose of B.
    pub(crate) trans_b: Trans,
    /// The stored A has adjacent rows (`col_stride() == 1`).
    pub(crate) a_unit_col_stride: bool,
    /// The stored A has adjacent columns (`row_stride() == 1`).
    pub(crate) a_unit_row_stride: bool,
    /// The stored B has adjacent rows (`col_stride() == 1`).
    pub(crate) b_unit_col_stride: bool,
    /// The stored B has adjacent columns (`row_stride() == 1`).
    pub(crate) b_unit_row_stride: bool,
    /// The element type is `f64` or `f32`.
    pub(crate) simd_real: bool,
    /// The SIMD tier of the call.
    pub(crate) tier: SimdTier,
}

impl RouteInputs {
    /// Describes `C = alpha·op(A)·op(B) + beta·C` with `op(A)` of `m × k` and
    /// `op(B)` of `k × n`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<T: Field>(
        trans_a: Trans,
        trans_b: Trans,
        a: &MatRef<'_, T>,
        b: &MatRef<'_, T>,
        m: usize,
        n: usize,
        k: usize,
        tier: SimdTier,
    ) -> Self {
        let id = TypeId::of::<T>();
        Self {
            m,
            n,
            k,
            trans_a,
            trans_b,
            a_unit_col_stride: a.col_stride() == 1,
            a_unit_row_stride: a.row_stride() == 1,
            b_unit_col_stride: b.col_stride() == 1,
            b_unit_row_stride: b.row_stride() == 1,
            simd_real: id == TypeId::of::<f64>() || id == TypeId::of::<f32>(),
            tier,
        }
    }

    /// `m·n·k`, saturating.
    pub(crate) const fn work(&self) -> usize {
        self.m.saturating_mul(self.n).saturating_mul(self.k)
    }

    /// Both operands are `NoTrans`.
    pub(crate) fn both_notrans(&self) -> bool {
        self.trans_a == Trans::NoTrans && self.trans_b == Trans::NoTrans
    }

    /// A thin problem the GEMV path can take: `f64`/`f32`, `m == 1` or
    /// `n == 1`, and the large operand (A when `n == 1`, otherwise B) has a
    /// unit stride, which is what `gemm_thin` needs to read it in place.
    pub(crate) fn thin_ok(&self) -> bool {
        let large_ok = if self.n == 1 {
            self.a_unit_col_stride || self.a_unit_row_stride
        } else {
            self.b_unit_col_stride || self.b_unit_row_stride
        };
        self.simd_real && (self.m == 1 || self.n == 1) && large_ok
    }

    /// The tier has SIMD small kernels for this element type and shape
    /// under `policy`: NEON from 4×4, AVX2 + FMA (with
    /// `policy.small_simd_x86`) from `m >= 5`, `n >= 4`.
    pub(crate) fn simd_small(&self, policy: &GemmPolicy) -> bool {
        self.simd_real
            && match self.tier {
                SimdTier::Neon => self.m >= 4 && self.n >= 4,
                SimdTier::Avx2Fma => policy.small_simd_x86 && self.m >= 5 && self.n >= 4,
                SimdTier::Sse42 | SimdTier::Portable => false,
            }
    }
}

/// Chooses the route for one GEMM call (a pure function).
///
/// In order:
/// 1. With `policy.thin`, a thin problem the GEMV path can take
///    ([`RouteInputs::thin_ok`]): [`GemmRoute::Thin`].
/// 2. `NoTrans`/`NoTrans` with column-major operands whose tier has SIMD
///    small kernels, up to `policy.small_simd_max_work`: [`GemmRoute::Small`].
/// 3. Other operands of such a tier with `m·n·k` in
///    `policy.small_trans_copy`: [`GemmRoute::SmallTransCopy`].
/// 4. Up to [`SMALL_THRESHOLD`]: [`GemmRoute::Small`] for `NoTrans`/`NoTrans`,
///    [`GemmRoute::SmallTrans`] otherwise.
/// 5. Everything else: [`GemmRoute::Blocked`].
pub(crate) fn choose_route(inp: &RouteInputs, policy: &GemmPolicy) -> GemmRoute {
    if policy.thin && inp.thin_ok() {
        return GemmRoute::Thin;
    }
    let work = inp.work();
    let nn = inp.both_notrans();
    let col_major = nn && inp.a_unit_col_stride && inp.b_unit_col_stride;
    let simd_small = inp.simd_small(policy);
    if simd_small && col_major && work <= policy.small_simd_max_work {
        return GemmRoute::Small;
    }
    let copy = matches!(policy.small_trans_copy, Some((lo, hi)) if lo <= work && work <= hi);
    if simd_small && !col_major && copy {
        return GemmRoute::SmallTransCopy;
    }
    if work <= SMALL_THRESHOLD {
        return if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        };
    }
    GemmRoute::Blocked
}

/// Everything one GEMM call needs to decide how it runs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmPlan<T> {
    /// Tier and policy.
    pub(crate) ctx: GemmContext,
    /// The micro-kernel of the blocked route; the packed panels use its shape.
    ///
    /// The default blocking takes KC from the *canonical* kernel of `T`
    /// (`T::micro_kernel_dispatch()`) whatever this kernel is, so the result
    /// bits do not depend on it, except NaN payloads: NaN positions are the
    /// same, the payload a NaN carries can differ between two kernels (see
    /// `gemm::default_blocking` and `gemm_kernel::kernel_for_problem`).
    pub(crate) kernel: MicroKernel<T>,
    /// A route that overrides [`choose_route`] when it accepts the problem
    /// (tests and A/B runs).
    pub(crate) forced_route: Option<GemmRoute>,
    /// A pack scope that overrides the policy's choice (tests and A/B runs).
    pub(crate) forced_scope: Option<PackScope>,
}

impl<T: GemmKernel> GemmPlan<T> {
    /// The plan of this build and CPU for `policy`, with the canonical
    /// micro-kernel of `T`.
    pub(crate) fn new(policy: GemmPolicy) -> Self {
        Self {
            ctx: GemmContext::detect(policy),
            kernel: T::micro_kernel_dispatch(),
            forced_route: None,
            forced_scope: None,
        }
    }

    /// The plan of this build and CPU for `policy` and an `m × k` by `k × n`
    /// product: [`GemmPlan::new`] with the micro-kernel that
    /// `gemm_kernel::kernel_for_problem` picks for the tier, the policy and
    /// the dimensions.
    pub(crate) fn for_problem(policy: GemmPolicy, m: usize, n: usize, k: usize) -> Self {
        let mut plan = Self::new(policy);
        plan.kernel = kernel_for_problem::<T>(plan.ctx.tier(), policy.f32_avx2, m, n, k);
        plan
    }
}

impl<T> GemmPlan<T> {
    /// The route for the problem `inp`: the forced route when it accepts the
    /// problem, otherwise [`choose_route`].
    pub(crate) fn route(&self, inp: &RouteInputs) -> GemmRoute {
        match self.forced_route {
            Some(route) if route.accepts(inp) => route,
            _ => choose_route(inp, &self.ctx.policy),
        }
    }

    /// The pack scope of the 2D parallel driver on `threads` threads: the
    /// forced scope, otherwise [`PackScope::PerTile`] from
    /// `policy.per_tile_min_threads` threads on and [`PackScope::Shared`]
    /// below (`usize::MAX`: always shared).
    pub(crate) fn scope(&self, threads: usize) -> PackScope {
        match self.forced_scope {
            Some(scope) => scope,
            None if self.ctx.policy.per_tile_min_threads != usize::MAX
                && threads >= self.ctx.policy.per_tile_min_threads =>
            {
                PackScope::PerTile
            }
            None => PackScope::Shared,
        }
    }

    /// This plan with `route` forced (when it accepts the problem).
    #[cfg(test)]
    pub(crate) fn with_forced_route(mut self, route: GemmRoute) -> Self {
        self.forced_route = Some(route);
        self
    }

    /// This plan with the pack scope of the 2D parallel driver forced.
    #[cfg(test)]
    pub(crate) fn with_forced_scope(mut self, scope: PackScope) -> Self {
        self.forced_scope = Some(scope);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_TIERS: [SimdTier; 4] = [
        SimdTier::Portable,
        SimdTier::Sse42,
        SimdTier::Neon,
        SimdTier::Avx2Fma,
    ];

    const ALL_TRANS: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];

    fn inputs(
        (m, k, n): (usize, usize, usize),
        (trans_a, trans_b): (Trans, Trans),
        (a_unit, b_unit): (bool, bool),
        simd_real: bool,
        tier: SimdTier,
    ) -> RouteInputs {
        RouteInputs {
            m,
            n,
            k,
            trans_a,
            trans_b,
            a_unit_col_stride: a_unit,
            a_unit_row_stride: !a_unit,
            b_unit_col_stride: b_unit,
            b_unit_row_stride: !b_unit,
            simd_real,
            tier,
        }
    }

    /// The 0.2.3 routing rule: small path up to 32³ (`gemm_small` for
    /// `NoTrans`/`NoTrans`, `gemm_small_trans` otherwise), blocked above.
    fn route_0_2_3(inp: &RouteInputs) -> GemmRoute {
        if inp.m * inp.n * inp.k <= SMALL_THRESHOLD {
            if inp.both_notrans() {
                GemmRoute::Small
            } else {
                GemmRoute::SmallTrans
            }
        } else {
            GemmRoute::Blocked
        }
    }

    const SHAPES: [(usize, usize, usize); 22] = [
        (1, 1, 1),
        (1, 1000, 1),
        (1, 64, 512),
        (512, 64, 1),
        (3, 5, 4),
        (4, 4, 4),
        (4, 2048, 4),
        (8, 8, 8),
        (5, 3, 7),
        (32, 32, 32),
        (33, 32, 32),
        (32, 33, 32),
        (64, 64, 64),
        (2, 16_384, 1),
        // AVX2 small-kernel boundaries: m 4 / 5, n 3 / 4.
        (4, 8, 4),
        (5, 8, 4),
        (5, 8, 3),
        (5, 1024, 4),
        // Around the x86 candidate ceiling of 80³.
        (80, 80, 80),
        (81, 80, 80),
        (5, 25_600, 4),
        (5, 25_601, 4),
    ];

    #[test]
    fn test_gemm_plan_tier_from_kind_table() {
        use GemmKernelKind as K;
        let table = [
            (K::Avx512, true, SimdTier::Avx2Fma),
            (K::Avx512, false, SimdTier::Sse42),
            (K::Avx2, true, SimdTier::Avx2Fma),
            (K::Avx2, false, SimdTier::Sse42),
            (K::Sse42, true, SimdTier::Sse42),
            (K::Sse42, false, SimdTier::Sse42),
            (K::Neon, false, SimdTier::Neon),
            (K::Neon, true, SimdTier::Neon),
            (K::Scalar, false, SimdTier::Portable),
            (K::Scalar, true, SimdTier::Portable),
        ];
        for (kind, avx2_fma, tier) in table {
            assert_eq!(
                SimdTier::from_kind(kind, avx2_fma),
                tier,
                "{kind:?} avx2_fma={avx2_fma}"
            );
        }
    }

    #[test]
    fn test_gemm_plan_host_tier_consistent() {
        let tier = SimdTier::current();
        let kind = selected_kernel_kind();
        assert_eq!(tier, SimdTier::from_kind(kind, has_avx2_fma()));
        if cfg!(feature = "force-scalar") {
            assert_eq!(tier, SimdTier::Portable);
        }
        if cfg!(all(target_arch = "aarch64", not(feature = "force-scalar"))) {
            assert_eq!(tier, SimdTier::Neon);
        }
        if cfg!(not(target_arch = "aarch64")) {
            assert_ne!(tier, SimdTier::Neon);
        }
        if cfg!(not(target_arch = "x86_64")) {
            assert_ne!(tier, SimdTier::Sse42);
            assert_ne!(tier, SimdTier::Avx2Fma);
        }
        if cfg!(feature = "max-simd-128") {
            assert_ne!(tier, SimdTier::Avx2Fma);
        }
        if tier == SimdTier::Avx2Fma {
            assert!(has_avx2_fma());
        }
        // The shared AVX2 test gate opens exactly on the AVX2+FMA tier.
        assert_eq!(
            avx2_fma_tier_or_skip("test_gemm_plan_host_tier_consistent"),
            tier == SimdTier::Avx2Fma
        );
        // The plan carries exactly this tier and the dispatched kernel.
        let plan = GemmPlan::<f64>::new(LEGACY_POLICY);
        assert_eq!(plan.ctx.tier(), tier);
        assert_eq!(plan.kernel.shape, f64::micro_kernel_shape());
        assert_eq!(plan.forced_route, None);
    }

    #[test]
    fn test_gemm_plan_legacy_route_matches_0_2_3() {
        for &shape in &SHAPES {
            for &ta in &ALL_TRANS {
                for &tb in &ALL_TRANS {
                    for units in [(true, true), (false, true), (true, false), (false, false)] {
                        for simd_real in [true, false] {
                            for &tier in &ALL_TIERS {
                                let inp = inputs(shape, (ta, tb), units, simd_real, tier);
                                assert_eq!(
                                    choose_route(&inp, &LEGACY_POLICY),
                                    route_0_2_3(&inp),
                                    "{inp:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_plan_small_simd_ceiling() {
        // A larger ceiling sends column-major NoTrans/NoTrans problems of a
        // tier with SIMD small kernels to the small path; nothing else moves.
        let policy = GemmPolicy {
            small_simd_max_work: 64 * 64 * 64,
            ..LEGACY_POLICY
        };
        let nn = (Trans::NoTrans, Trans::NoTrans);
        let neon = |shape, units, real| inputs(shape, nn, units, real, SimdTier::Neon);
        let big = (64, 64, 64);
        assert_eq!(
            choose_route(&neon(big, (true, true), true), &policy),
            GemmRoute::Small
        );
        assert_eq!(
            choose_route(&neon(big, (false, true), true), &policy),
            GemmRoute::Blocked
        );
        assert_eq!(
            choose_route(&neon(big, (true, true), false), &policy),
            GemmRoute::Blocked
        );
        // m < 4: no SIMD tile, and above 32³ (3·128·128).
        assert_eq!(
            choose_route(&neon((3, 128, 128), (true, true), true), &policy),
            GemmRoute::Blocked
        );
        assert_eq!(
            choose_route(&neon((65, 64, 64), (true, true), true), &policy),
            GemmRoute::Blocked
        );
        for tier in [SimdTier::Portable, SimdTier::Sse42, SimdTier::Avx2Fma] {
            let inp = inputs(big, nn, (true, true), true, tier);
            assert_eq!(choose_route(&inp, &policy), GemmRoute::Blocked, "{tier:?}");
        }
        // With `small_simd_x86` the AVX2 + FMA tier has small kernels too
        // (from 5 × 4), under the same ceiling.
        let x86 = GemmPolicy {
            small_simd_x86: true,
            ..policy
        };
        let avx2 = |shape, units, real| inputs(shape, nn, units, real, SimdTier::Avx2Fma);
        assert_eq!(
            choose_route(&avx2(big, (true, true), true), &x86),
            GemmRoute::Small
        );
        assert_eq!(
            choose_route(&avx2(big, (false, true), true), &x86),
            GemmRoute::Blocked
        );
        assert_eq!(
            choose_route(&avx2((4, 128, 128), (true, true), true), &x86),
            GemmRoute::Blocked
        );
        assert_eq!(
            choose_route(&avx2((65, 64, 64), (true, true), true), &x86),
            GemmRoute::Blocked
        );
        for tier in [SimdTier::Portable, SimdTier::Sse42] {
            let inp = inputs(big, nn, (true, true), true, tier);
            assert_eq!(choose_route(&inp, &x86), GemmRoute::Blocked, "{tier:?}");
        }
        let tn = inputs(
            big,
            (Trans::Trans, Trans::NoTrans),
            (true, true),
            true,
            SimdTier::Neon,
        );
        assert_eq!(choose_route(&tn, &policy), GemmRoute::Blocked);
    }

    #[test]
    fn test_gemm_plan_forced_route_validated() {
        let plan = GemmPlan::<f64>::new(LEGACY_POLICY);
        let small = inputs(
            (8, 8, 8),
            (Trans::NoTrans, Trans::NoTrans),
            (true, true),
            true,
            SimdTier::Portable,
        );
        let small_t = RouteInputs {
            trans_b: Trans::Trans,
            ..small
        };
        let large = RouteInputs {
            m: 100,
            n: 100,
            k: 100,
            ..small
        };
        // Blocked and SmallTrans accept everything.
        for route in [GemmRoute::Blocked, GemmRoute::SmallTrans] {
            let forced = plan.with_forced_route(route);
            for inp in [small, small_t, large] {
                assert_eq!(forced.route(&inp), route, "{route:?} {inp:?}");
            }
        }
        // Small must not read a transposed operand as stored.
        let forced = plan.with_forced_route(GemmRoute::Small);
        assert_eq!(forced.route(&large), GemmRoute::Small);
        assert_eq!(forced.route(&small_t), GemmRoute::SmallTrans);
        let large_t = RouteInputs {
            trans_a: Trans::ConjTrans,
            ..large
        };
        assert_eq!(forced.route(&large_t), GemmRoute::Blocked);
        // No forced route: the router decides.
        assert_eq!(plan.route(&large), GemmRoute::Blocked);
        assert_eq!(plan.route(&small_t), GemmRoute::SmallTrans);
    }

    #[test]
    fn test_gemm_plan_work_saturates() {
        let inp = inputs(
            (usize::MAX / 2, 3, 5),
            (Trans::NoTrans, Trans::NoTrans),
            (true, true),
            true,
            SimdTier::Neon,
        );
        assert_eq!(inp.work(), usize::MAX);
        assert_eq!(choose_route(&inp, &LEGACY_POLICY), GemmRoute::Blocked);
        assert_eq!(choose_route(&inp, &CANDIDATE_POLICY), GemmRoute::Blocked);
    }

    /// The candidate routing rule written out per policy: `x86` selects the
    /// x86 candidate's values (AVX2 small kernels from 5 × 4 up to 80³), the
    /// aarch64 candidate otherwise (NEON small kernels from 4 × 4 up to 32³).
    fn candidate_route(inp: &RouteInputs, x86: bool) -> GemmRoute {
        let (m, n) = (inp.m, inp.n);
        let work = m * n * inp.k;
        let nn = inp.both_notrans();
        let col_major = nn && inp.a_unit_col_stride && inp.b_unit_col_stride;
        let tile = inp.simd_real
            && match inp.tier {
                SimdTier::Neon => m >= 4 && n >= 4,
                SimdTier::Avx2Fma => x86 && m >= 5 && n >= 4,
                SimdTier::Sse42 | SimdTier::Portable => false,
            };
        let ceiling = if x86 { 80 * 80 * 80 } else { 32 * 32 * 32 };
        // Every large operand here has a unit stride (rows or columns).
        if inp.simd_real && (m == 1 || n == 1) {
            GemmRoute::Thin
        } else if tile && col_major && work <= ceiling {
            GemmRoute::Small
        } else if tile && !col_major && (512..=32_768).contains(&work) {
            GemmRoute::SmallTransCopy
        } else {
            route_0_2_3(inp)
        }
    }

    #[test]
    fn test_gemm_plan_candidate_route_table() {
        // The candidate routes, tier by tier; `LEGACY_POLICY` keeps 0.2.3.
        for (policy, x86) in [(CANDIDATE_X86, true), (CANDIDATE_AARCH64, false)] {
            for &shape in &SHAPES {
                for &ta in &ALL_TRANS {
                    for &tb in &ALL_TRANS {
                        for units in [(true, true), (false, true), (true, false), (false, false)] {
                            for simd_real in [true, false] {
                                for &tier in &ALL_TIERS {
                                    let inp = inputs(shape, (ta, tb), units, simd_real, tier);
                                    assert_eq!(
                                        choose_route(&inp, &policy),
                                        candidate_route(&inp, x86),
                                        "x86={x86} {inp:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        // Below 8³ the transposed products keep the loops.
        let tiny = inputs(
            (4, 4, 4),
            (Trans::Trans, Trans::NoTrans),
            (true, true),
            true,
            SimdTier::Neon,
        );
        assert_eq!(
            choose_route(&tiny, &CANDIDATE_AARCH64),
            GemmRoute::SmallTrans
        );
        let at_min = inputs(
            (8, 8, 8),
            (Trans::Trans, Trans::NoTrans),
            (true, true),
            true,
            SimdTier::Neon,
        );
        assert_eq!(
            choose_route(&at_min, &CANDIDATE_AARCH64),
            GemmRoute::SmallTransCopy
        );
        // The AVX2 small route of the x86 candidate, at its edges.
        let nn = (Trans::NoTrans, Trans::NoTrans);
        let avx2 = |shape| inputs(shape, nn, (true, true), true, SimdTier::Avx2Fma);
        let x86 = &CANDIDATE_X86;
        assert_eq!(choose_route(&avx2((64, 64, 64)), x86), GemmRoute::Small);
        assert_eq!(choose_route(&avx2((80, 80, 80)), x86), GemmRoute::Small);
        assert_eq!(choose_route(&avx2((81, 80, 80)), x86), GemmRoute::Blocked);
        assert_eq!(choose_route(&avx2((5, 1024, 4)), x86), GemmRoute::Small);
        // m = 4 or n = 3 has no AVX2 tile: above 32³ it stays blocked.
        assert_eq!(choose_route(&avx2((4, 4096, 4)), x86), GemmRoute::Blocked);
        assert_eq!(choose_route(&avx2((5, 4096, 3)), x86), GemmRoute::Blocked);
        // The aarch64 candidate has no AVX2 small kernels, nor has the legacy
        // policy.
        for policy in [CANDIDATE_AARCH64, LEGACY_POLICY] {
            assert_eq!(
                choose_route(&avx2((64, 64, 64)), &policy),
                GemmRoute::Blocked
            );
        }
        // Transposed AVX2 problems in the copy range are copied.
        let tn = inputs(
            (16, 16, 16),
            (Trans::Trans, Trans::NoTrans),
            (true, true),
            true,
            SimdTier::Avx2Fma,
        );
        assert_eq!(choose_route(&tn, x86), GemmRoute::SmallTransCopy);
    }

    #[test]
    fn test_gemm_plan_policy_field_table() {
        // X-1 values, field by field.
        let gate = (512 * 512, 128 * 128);
        for policy in [LEGACY_POLICY, CANDIDATE_X86, CANDIDATE_AARCH64] {
            assert_eq!(policy.thin_par_gate, gate);
            assert_eq!(policy.b_caps, ISSUE_23_B_CAPS);
            let threshold = policy.thin_par_threshold();
            assert_eq!(
                (threshold.min_elements, threshold.min_work_per_thread),
                gate
            );
        }
        assert_eq!(
            [
                LEGACY_POLICY.thin,
                CANDIDATE_X86.thin,
                CANDIDATE_AARCH64.thin
            ],
            [false, true, true]
        );
        assert_eq!(
            [
                LEGACY_POLICY.small_simd_x86,
                CANDIDATE_X86.small_simd_x86,
                CANDIDATE_AARCH64.small_simd_x86
            ],
            [false, true, false]
        );
        assert_eq!(LEGACY_POLICY.small_simd_max_work, SMALL_THRESHOLD);
        assert_eq!(CANDIDATE_X86.small_simd_max_work, 80 * 80 * 80);
        assert_eq!(CANDIDATE_AARCH64.small_simd_max_work, SMALL_THRESHOLD);
        assert_eq!(LEGACY_POLICY.small_trans_copy, None);
        for policy in [CANDIDATE_X86, CANDIDATE_AARCH64] {
            assert_eq!(policy.small_trans_copy, Some((8 * 8 * 8, 32 * 32 * 32)));
            assert_eq!(policy.b_cap, CANDIDATE_B_CAP);
        }
        let expected_cap = if cfg!(oxiblas_gemm_bcap_all) {
            BCapScope::All
        } else {
            BCapScope::Sequential
        };
        assert_eq!(CANDIDATE_B_CAP, expected_cap);
        assert_eq!(LEGACY_POLICY.b_cap, BCapScope::Off);
        assert_eq!(
            [
                LEGACY_POLICY.f32_avx2,
                CANDIDATE_X86.f32_avx2,
                CANDIDATE_AARCH64.f32_avx2
            ],
            [
                F32Avx2Policy::Legacy8x8,
                F32Avx2Policy::Adaptive,
                F32Avx2Policy::Legacy8x8
            ]
        );
        assert_eq!(
            [
                LEGACY_POLICY.per_tile_min_threads,
                CANDIDATE_X86.per_tile_min_threads,
                CANDIDATE_AARCH64.per_tile_min_threads
            ],
            [usize::MAX, 5, usize::MAX]
        );
    }

    #[test]
    fn test_gemm_plan_scope_selection() {
        let legacy = GemmPlan::<f64>::new(LEGACY_POLICY);
        let x86 = GemmPlan::<f64>::new(CANDIDATE_X86);
        let aarch64 = GemmPlan::<f64>::new(CANDIDATE_AARCH64);
        for threads in [0, 1, 2, 4, 5, 8, 64, usize::MAX - 1, usize::MAX] {
            // `usize::MAX` means never, even for `usize::MAX` threads.
            assert_eq!(legacy.scope(threads), PackScope::Shared, "{threads}");
            assert_eq!(aarch64.scope(threads), PackScope::Shared, "{threads}");
            let expected = if threads >= 5 {
                PackScope::PerTile
            } else {
                PackScope::Shared
            };
            assert_eq!(x86.scope(threads), expected, "{threads}");
            // A forced scope wins at every thread count.
            for forced in [PackScope::Shared, PackScope::PerTile] {
                assert_eq!(legacy.with_forced_scope(forced).scope(threads), forced);
                assert_eq!(x86.with_forced_scope(forced).scope(threads), forced);
            }
        }
        assert_eq!(legacy.forced_scope, None);
    }

    #[test]
    fn test_gemm_plan_for_problem_delegates_the_kernel_choice() {
        // `for_problem` is `new` with the kernel that `kernel_for_problem`
        // picks for the plan's tier, the policy's f32 AVX2 choice and the
        // shape. Which kernel that is (per tier, type and shape) is tested
        // with `kernel_for_problem` in `gemm_kernel/tests.rs`; here only the
        // part that never changes: `Legacy8x8` always gives the canonical
        // kernel of the type.
        fn check<T: GemmKernel>() {
            let canonical = T::micro_kernel_shape();
            for policy in [LEGACY_POLICY, CANDIDATE_X86, CANDIDATE_AARCH64] {
                for (m, n, k) in [
                    (1, 1, 1),
                    (8, 4096, 4096),
                    (4096, 8, 4096),
                    (12, 4096, 4096),
                    (16, 16, 16),
                    (77, 61, 3),
                ] {
                    let plan = GemmPlan::<T>::for_problem(policy, m, n, k);
                    let base = GemmPlan::<T>::new(policy);
                    let chosen = kernel_for_problem::<T>(plan.ctx.tier(), policy.f32_avx2, m, n, k);
                    assert_eq!(plan.kernel.shape, chosen.shape, "{m}x{n}x{k}");
                    if policy.f32_avx2 == F32Avx2Policy::Legacy8x8 {
                        assert_eq!(plan.kernel.shape, canonical, "{m}x{n}x{k}");
                    }
                    assert_eq!(base.kernel.shape, canonical);
                    assert_eq!(plan.ctx, base.ctx);
                    assert_eq!(plan.forced_route, None);
                    assert_eq!(plan.forced_scope, None);
                }
            }
            for &tier in &ALL_TIERS {
                let kernel =
                    kernel_for_problem::<T>(tier, F32Avx2Policy::Legacy8x8, 16, 4096, 4096);
                assert_eq!(kernel.shape, canonical, "{tier:?}");
            }
        }
        check::<f64>();
        check::<f32>();
        check::<num_complex::Complex64>();
    }

    #[test]
    fn test_gemm_plan_default_policy_selection() {
        if cfg!(oxiblas_gemm_candidate) {
            assert_eq!(DEFAULT_POLICY, CANDIDATE_POLICY);
        } else {
            assert_eq!(DEFAULT_POLICY, LEGACY_POLICY);
        }
        if cfg!(target_arch = "x86_64") {
            assert_eq!(CANDIDATE_POLICY, CANDIDATE_X86);
        } else if cfg!(target_arch = "aarch64") {
            assert_eq!(CANDIDATE_POLICY, CANDIDATE_AARCH64);
        } else {
            assert_eq!(CANDIDATE_POLICY, LEGACY_POLICY);
        }
        // The portable tier is the only one a context can be lowered to.
        let ctx = GemmContext::detect(LEGACY_POLICY).with_portable_tier();
        assert_eq!(ctx.tier(), SimdTier::Portable);
        assert_eq!(ctx.policy, LEGACY_POLICY);
    }
}
