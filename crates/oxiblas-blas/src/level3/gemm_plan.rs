//! Per-call GEMM plan: SIMD tier, routing policy and problem, from which the
//! route of the call follows; the micro-kernel is resolved only when the
//! blocked route starts.
//!
//! Every real GEMM entry point ([`gemm_transposed_with_par`],
//! [`gemm_transposed_with_blocking`] and everything built on them) builds one
//! [`GemmPlan`] per call and hands it to the internal `gemm_planned` funnel:
//!
//! - [`SimdTier`] is derived from the *capped* micro-kernel kind
//!   (`selected_kernel_kind`), so every SIMD path keyed off the tier honours
//!   the `force-scalar`, `max-simd-128` and `max-simd-256` features exactly
//!   like the micro-kernels do.
//! - [`GemmPolicy`] says which optional paths a call may take; the policies
//!   are defined in `gemm_policy` and re-exported here.
//!   [`LEGACY_POLICY`] reproduces 0.2.3: every result that is not a NaN has
//!   the bits of 0.2.3 and every NaN result sits where 0.2.3 has one; only
//!   which NaN a result carries where an operation meets two NaNs may differ
//!   (the promise and how it is checked: [`LEGACY_POLICY`]).
//!   [`DEFAULT_POLICY`], the policy of the public entry points, is the policy
//!   adopted on aarch64 after A/B measurement (the scoped thin route and the
//!   small transposed copy of a transposed A, both only for operands stored
//!   column-major, the thin route only on a tier with SIMD thin kernels) in
//!   aarch64 builds without `force-scalar`, and [`LEGACY_POLICY`] on every other
//!   target and with `force-scalar`. `--cfg oxiblas_gemm_legacy` makes it
//!   [`LEGACY_POLICY`] everywhere (the routes and bits of 0.2.3);
//!   `--cfg oxiblas_gemm_candidate` selects the candidate of the target
//!   architecture (for A/B runs of the same source), and
//!   `--cfg oxiblas_gemm_bcap_all` additionally makes the candidates cap the
//!   B block in both blocked drivers ([`BCapScope::All`]). Which calls of
//!   the adopted default give other bits than 0.2.3: the module docs of
//!   `gemm_policy`.
//! - [`GemmContext`] is the non-generic part (tier and policy) that the route
//!   implementations receive.
//! - [`choose_route`] is a pure function of the problem description
//!   ([`RouteInputs`]) and the policy.
//! - The plan holds the problem `(m, n, k)` and, only when one is given
//!   (tests, A/B runs), a micro-kernel. The blocked route resolves the
//!   kernel once, with [`GemmPlan::micro_kernel`]
//!   (`gemm_kernel::kernel_for_problem` for the tier, the policy and the
//!   problem); the other routes run no micro-kernel and never resolve one,
//!   so building a plan reads the tier and nothing else.
//!
//! [`gemm_transposed_with_par`]: super::gemm::gemm_transposed_with_par
//! [`LEGACY_POLICY`]: super::gemm_policy::LEGACY_POLICY
//! [`gemm_transposed_with_blocking`]: super::gemm::gemm_transposed_with_blocking

#[cfg(test)]
use super::gemm_kernel::MicroKernelShape;
use super::gemm_kernel::{GemmKernel, MicroKernel, kernel_for_problem, selected_kernel_kind};
// The policies live in `gemm_policy`; they are re-exported here so that the
// plan, its route and its policy are imported from one place.
#[cfg(test)]
pub(crate) use super::gemm_policy::{
    ADOPTED_AARCH64, CANDIDATE_AARCH64, CANDIDATE_POLICY, CANDIDATE_X86, LEGACY_POLICY,
};
pub(crate) use super::gemm_policy::{
    BCapScope, DEFAULT_POLICY, F32Avx2Policy, GemmPolicy, PackScope, THIN_COL_AXPY, THIN_COL_DOT,
    THIN_MIN_WORK, THIN_ROW_AXPY, THIN_ROW_DOT, ThinPolicy,
};
use super::gemm_small::SMALL_THRESHOLD;
use super::trsm::Trans;
use core::any::TypeId;
use core::sync::atomic::{AtomicU8, Ordering};
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
///
/// The discriminants are the values of the tier cache behind
/// [`SimdTier::current`] (`0` there means "not detected yet").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SimdTier {
    /// No SIMD kernels (`force-scalar`, or no supported SIMD).
    Portable = 1,
    /// x86-64 with SSE4.2 but without AVX2 + FMA (or capped to 128 bits).
    Sse42 = 2,
    /// aarch64 NEON.
    Neon = 3,
    /// x86-64 with AVX2 and FMA (also on AVX-512 hosts).
    Avx2Fma = 4,
}

/// The tier of this build and CPU once [`SimdTier::current`] has detected it
/// (`tier as u8`), `0` before.
///
/// One byte holds everything the tier depends on (the capped kernel kind
/// and, on x86-64, the AVX2 + FMA report), so a GEMM call reads the tier with
/// one load on every architecture. Both inputs are fixed for the life of the
/// process, so every thread that detects the tier stores the same value, and
/// relaxed ordering is enough.
static CURRENT_TIER: AtomicU8 = AtomicU8::new(0);

impl SimdTier {
    /// The tier of this build and CPU: one relaxed load of the cached tier
    /// ([`CURRENT_TIER`]); only the first call of the process (or calls that
    /// race it) detect it ([`SimdTier::detect`]).
    #[inline]
    pub(crate) fn current() -> Self {
        #[cfg(test)]
        call_probe::note(call_probe::Event::TierRead);
        match Self::from_cache(CURRENT_TIER.load(Ordering::Relaxed)) {
            Some(tier) => tier,
            None => Self::detect_and_cache(),
        }
    }

    /// [`SimdTier::detect`], stored in [`CURRENT_TIER`].
    #[cold]
    #[inline(never)]
    fn detect_and_cache() -> Self {
        let tier = Self::detect();
        CURRENT_TIER.store(tier as u8, Ordering::Relaxed);
        tier
    }

    /// The tier of this build and CPU, detected without the cache: the
    /// capped micro-kernel kind and the CPU's AVX2 + FMA report.
    pub(crate) fn detect() -> Self {
        Self::from_kind(selected_kernel_kind(), has_avx2_fma())
    }

    /// The tier a value of [`CURRENT_TIER`] stands for (`None` for `0`, not
    /// detected yet).
    #[inline]
    const fn from_cache(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Portable),
            2 => Some(Self::Sse42),
            3 => Some(Self::Neon),
            4 => Some(Self::Avx2Fma),
            _ => None,
        }
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
///
/// When that variable is set (by the same rule) and the gate returns `true`,
/// it prints one line `<test>: AVX2+FMA tier confirmed`. Run with
/// `--show-output` (or `--nocapture`), these lines are the evidence that the
/// gated AVX2 code ran: step 4 of `tools/x86_gemm_runbook.sh` requires at
/// least one of them, and no `SKIPPED` or requirement-failure line, in the
/// log of each policy's parallel test run. At start-up the runbook reads the
/// body of this function (from `fn avx2_fma_tier_or_skip(` to its closing
/// brace) and stops with exit status 2 unless it contains the three format
/// strings `"{test}: AVX2+FMA tier confirmed"`, `"{test}: SKIPPED (not on the
/// AVX2+FMA tier)"` and `"{test}: OXIBLAS_REQUIRE_AVX2 is set, but"`: change
/// them together with the runbook.
#[cfg(test)]
pub(crate) fn avx2_fma_tier_or_skip(test: &str) -> bool {
    let require_set =
        std::env::var("OXIBLAS_REQUIRE_AVX2").is_ok_and(|v| !v.is_empty() && v != "0");
    if SimdTier::current() == SimdTier::Avx2Fma {
        if require_set {
            println!("{test}: AVX2+FMA tier confirmed");
        }
        return true;
    }
    let capped = cfg!(any(feature = "force-scalar", feature = "max-simd-128"));
    let required = cfg!(target_arch = "x86_64") && !capped && require_set;
    assert!(
        !required,
        "{test}: OXIBLAS_REQUIRE_AVX2 is set, but this run is not on the AVX2+FMA tier"
    );
    println!("{test}: SKIPPED (not on the AVX2+FMA tier)");
    false
}

/// Whether oxiblas-core's kernel selection is capped to scalar code, which
/// on aarch64 only its `force-scalar` feature does. Cargo unifies features
/// over a build, so another crate can turn that feature on in oxiblas-core
/// without this crate's own `force-scalar`: such a build keeps this crate's
/// default policy of aarch64 ([`DEFAULT_POLICY`]) while every call has the
/// tier [`SimdTier::Portable`].
#[cfg(test)]
pub(crate) fn core_forces_scalar() -> bool {
    use oxiblas_core::simd::{SimdLevel, detect_simd_level};
    detect_simd_level() == SimdLevel::Scalar
}

/// Skip check for tests of code that runs only on the [`SimdTier::Neon`]
/// tier, compiled into aarch64 builds without this crate's `force-scalar`.
/// Returns `true` on that tier. Otherwise the only admitted cause is
/// oxiblas-core's `force-scalar` ([`core_forces_scalar`]): it prints one
/// `SKIPPED` line and returns `false`; any other tier fails.
#[cfg(all(test, target_arch = "aarch64", not(feature = "force-scalar")))]
pub(crate) fn neon_tier_or_skip(test: &str) -> bool {
    let tier = SimdTier::current();
    if tier == SimdTier::Neon {
        return true;
    }
    assert!(
        core_forces_scalar() && tier == SimdTier::Portable,
        "{test}: tier {tier:?}, and oxiblas-core does not force scalar code"
    );
    println!("{test}: SKIPPED (oxiblas-core's force-scalar: the portable tier)");
    false
}

/// Below this `m·n·k` the router knows only the two small routes
/// ([`GemmRoute::Small`], [`GemmRoute::SmallTrans`]) and decides with one
/// comparison of the size: no thin scope starts below it
/// ([`THIN_MIN_WORK`]) and no policy of this crate copies from below it (a
/// policy that does, or [`ThinPolicy::All`], takes the full decision). 8³,
/// the least size from which a small transposed copy was ever a candidate.
pub(crate) const ROUTE_TINY_WORK: usize = 8 * 8 * 8;

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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub(crate) const fn work(&self) -> usize {
        self.m.saturating_mul(self.n).saturating_mul(self.k)
    }

    /// Both operands are `NoTrans`.
    #[inline]
    pub(crate) fn both_notrans(&self) -> bool {
        self.trans_a == Trans::NoTrans && self.trans_b == Trans::NoTrans
    }

    /// A is transposed (`Trans`, or `ConjTrans`, which is `Trans` for a real
    /// matrix) and both operands are stored column-major (`col_stride() ==
    /// 1`): the products that the small transposed copy with
    /// `small_trans_copy_a_only` copies (the measured class: `TN` and `TT`
    /// of column-major operands).
    #[inline]
    pub(crate) fn a_transposed_col_major(&self) -> bool {
        self.trans_a != Trans::NoTrans && self.a_unit_col_stride && self.b_unit_col_stride
    }

    /// Whether this thin problem ([`RouteInputs::thin_ok`]) of
    /// `work = m·n·k >= THIN_MIN_WORK` (which the router has checked) is in
    /// the scope of [`ThinPolicy::Scoped`].
    ///
    /// The scope is what the deciding measurement timed: both operands
    /// stored column-major (`col_stride() == 1`) on a tier with SIMD thin
    /// kernels ([`SimdTier::Neon`], [`SimdTier::Avx2Fma`]), at every size. A
    /// product with another storage of A or B (a transposed or strided
    /// view), and every product on [`SimdTier::Portable`] or
    /// [`SimdTier::Sse42`], is outside it and keeps the route of 0.2.3.
    ///
    /// A thin product is one GEMV on the large operand, in one of two forms
    /// (`gemm_thin`): the axpy form updates the vector of C over the columns
    /// of a column-major matrix (one fused multiply-add chain per element of
    /// C), the dot form computes every element of C as one dot product of
    /// length `k`. What the thin route has to beat depends on the size and
    /// the transposes:
    ///
    /// - above [`SMALL_THRESHOLD`], 0.2.3 runs the blocked driver, which
    ///   packs both operands into panels of the micro-kernel's width for one
    ///   row or column of C: every such product is in the scope (all four
    ///   transpose combinations);
    /// - up to [`SMALL_THRESHOLD`], 0.2.3 runs its small loops. In the scope
    ///   are the four layouts with the vector operand not transposed, each
    ///   from its own least sizes: a column of C from a column of B
    ///   ([`THIN_COL_AXPY`] for `op(A) = A`, [`THIN_COL_DOT`] for
    ///   `op(A) = Aᵀ`) and a row of C from a row of A ([`THIN_ROW_DOT`] for
    ///   `op(B) = B`, [`THIN_ROW_AXPY`] for `op(B) = Bᵀ`). A row of C costs
    ///   the thin route more than a column (the row is gathered into a
    ///   buffer and scattered back), so the two families have limits of
    ///   their own. A dot product (`m == n == 1`) has a vector of length 1,
    ///   below every least length. Every other small thin product keeps the
    ///   route of 0.2.3.
    #[inline]
    pub(crate) fn thin_in_scope(&self, work: usize) -> bool {
        let simd_thin_tier = matches!(self.tier, SimdTier::Neon | SimdTier::Avx2Fma);
        if !(self.a_unit_col_stride && self.b_unit_col_stride && simd_thin_tier) {
            return false;
        }
        if work > SMALL_THRESHOLD {
            return true;
        }
        let a_notrans = self.trans_a == Trans::NoTrans;
        let b_notrans = self.trans_b == Trans::NoTrans;
        let limits = if self.n == 1 {
            if !b_notrans {
                return false;
            }
            if a_notrans {
                THIN_COL_AXPY
            } else {
                THIN_COL_DOT
            }
        } else {
            if !a_notrans {
                return false;
            }
            if b_notrans {
                THIN_ROW_DOT
            } else {
                THIN_ROW_AXPY
            }
        };
        self.m.max(self.n) >= limits.min_vec && self.k >= limits.min_depth
    }

    /// A thin problem the GEMV path can take: `f64`/`f32`, `m == 1` or
    /// `n == 1`, and the large operand (A when `n == 1`, otherwise B) has a
    /// unit stride, which is what `gemm_thin` needs to read it in place.
    #[inline]
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
    #[inline]
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
/// 1. Below [`ROUTE_TINY_WORK`] (for every policy but [`ThinPolicy::All`] and
///    policies that copy from a smaller size): [`GemmRoute::Small`] for
///    `NoTrans`/`NoTrans`, [`GemmRoute::SmallTrans`] otherwise.
/// 2. `NoTrans`/`NoTrans` with column-major operands whose tier has SIMD
///    small kernels, up to `policy.small_simd_max_work`: [`GemmRoute::Small`].
/// 3. A thin problem the GEMV path can take ([`RouteInputs::thin_ok`]) that
///    `policy.thin` sends there ([`ThinPolicy`]): [`GemmRoute::Thin`].
/// 4. Other operands of a tier with SIMD small kernels with `m·n·k` in
///    `policy.small_trans_copy` (and, with `small_trans_copy_a_only`, only
///    a transposed A with both operands stored column-major,
///    [`RouteInputs::a_transposed_col_major`]):
///    [`GemmRoute::SmallTransCopy`].
/// 5. Up to [`SMALL_THRESHOLD`]: [`GemmRoute::Small`] for `NoTrans`/`NoTrans`,
///    [`GemmRoute::SmallTrans`] otherwise.
/// 6. Everything else: [`GemmRoute::Blocked`].
///
/// The order is the cheapest one for the calls that are cheapest: a product
/// up to 8³ takes 8 to 50 ns, and one more comparison on its way is visible.
///
/// `#[inline(always)]`: it runs on every call, and the generic GEMM core
/// that calls it is instantiated in the caller's crate; out of line, the
/// policy would have to be written to memory on every call, which tiny
/// products pay for measurably. A plain `#[inline]` is not enough: with a
/// policy that has the thin route and the small transposed copy, the
/// compiler kept this function out of line, and with it the small routes of
/// the GEMM core (see `gemm_thin_route` for the other half of the remedy).
#[inline(always)]
pub(crate) fn choose_route(inp: &RouteInputs, policy: &GemmPolicy) -> GemmRoute {
    let work = inp.work();
    let nn = inp.both_notrans();
    // 1. The smallest products: below `ROUTE_TINY_WORK` only the two small
    //    routes exist (unless the policy sends every thin product to the
    //    thin route, or copies from a smaller size), so one comparison of the
    //    size settles them, the same under every such policy.
    let tiny_routes_only = policy.thin != ThinPolicy::All
        && policy
            .small_trans_copy
            .is_none_or(|(lo, _)| lo >= ROUTE_TINY_WORK);
    if work < ROUTE_TINY_WORK && tiny_routes_only {
        return if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        };
    }
    // 2. The SIMD small kernels. They need at least four rows and columns,
    //    so no thin product is among theirs: tested before the thin route,
    //    they cost a `NoTrans`/`NoTrans` product nothing more than in 0.2.3.
    let col_major = nn && inp.a_unit_col_stride && inp.b_unit_col_stride;
    let simd_small = inp.simd_small(policy);
    if simd_small && col_major && work <= policy.small_simd_max_work {
        return GemmRoute::Small;
    }
    // 3. The thin route: the size, then the scope, then the policy. Below
    //    `THIN_MIN_WORK`, and for every product outside the scope, a call
    //    takes the same decisions in the same order under `ThinPolicy::Off`
    //    and `ThinPolicy::Scoped` (only `ThinPolicy::All` differs there), so
    //    the scope cannot change what such a call costs. Where the policy is
    //    a constant (every public entry point), a policy without the thin
    //    route drops these tests. (The nesting is that order, so it stays
    //    as it is.)
    #[allow(clippy::collapsible_if)]
    if work >= THIN_MIN_WORK || policy.thin == ThinPolicy::All {
        if inp.thin_ok() {
            if work >= THIN_MIN_WORK && inp.thin_in_scope(work) {
                if policy.thin != ThinPolicy::Off {
                    return GemmRoute::Thin;
                }
            } else if policy.thin == ThinPolicy::All {
                return GemmRoute::Thin;
            }
        }
    }
    // 4. The small transposed copy.
    let copy = matches!(policy.small_trans_copy, Some((lo, hi)) if lo <= work && work <= hi)
        && (!policy.small_trans_copy_a_only || inp.a_transposed_col_major());
    if simd_small && !col_major && copy {
        return GemmRoute::SmallTransCopy;
    }
    // 5. The small loops up to the small threshold, the blocked driver above.
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
    /// `(m, n, k)` of the `m × k` by `k × n` product the plan is for: the
    /// blocked route runs the micro-kernel that
    /// `gemm_kernel::kernel_for_problem` picks for them and the plan's tier
    /// and policy ([`GemmPlan::micro_kernel`]), unless one is given.
    pub(crate) problem: (usize, usize, usize),
    /// A micro-kernel given explicitly (the canonical one of
    /// `GemmPlan::new`, tests and A/B runs), which the blocked route runs
    /// instead of the one picked for `problem`.
    ///
    /// Whatever kernel runs, the default blocking takes KC from the
    /// *canonical* kernel of `T` (`T::micro_kernel_dispatch()`), so the
    /// result bits do not depend on it, except NaN payloads: NaN positions
    /// are the same, the payload a NaN carries can differ between two kernels
    /// (see `gemm::default_blocking` and `gemm_kernel::kernel_for_problem`).
    pub(crate) given_kernel: Option<MicroKernel<T>>,
    /// A route that overrides [`choose_route`] when it accepts the problem
    /// (tests and A/B runs).
    pub(crate) forced_route: Option<GemmRoute>,
    /// A pack scope that overrides the policy's choice (tests and A/B runs).
    pub(crate) forced_scope: Option<PackScope>,
}

/// What decides the micro-kernel of a plan, comparable without resolving it
/// (tests: the A/B harness compares the plans of its timed calls with it).
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlanKernelKey {
    /// [`GemmPlan::problem`].
    pub(crate) problem: (usize, usize, usize),
    /// The shape of [`GemmPlan::given_kernel`].
    pub(crate) given: Option<MicroKernelShape>,
}

impl<T: GemmKernel> GemmPlan<T> {
    /// The plan of this build and CPU for `policy`, with the canonical
    /// micro-kernel of `T` given explicitly (and no problem: `(0, 0, 0)`).
    #[cfg(test)]
    pub(crate) fn new(policy: GemmPolicy) -> Self {
        Self::for_problem(policy, 0, 0, 0).with_kernel(T::micro_kernel_dispatch())
    }

    /// The plan of this build and CPU for `policy` and an `m × k` by `k × n`
    /// product: its blocked route runs the micro-kernel that
    /// `gemm_kernel::kernel_for_problem` picks for the tier, the policy and
    /// the dimensions.
    ///
    /// Building it reads the tier (one load) and nothing else: the
    /// micro-kernel is resolved by [`GemmPlan::micro_kernel`] where the
    /// blocked route starts, so the other routes never dispatch one.
    #[inline]
    pub(crate) fn for_problem(policy: GemmPolicy, m: usize, n: usize, k: usize) -> Self {
        Self {
            ctx: GemmContext::detect(policy),
            problem: (m, n, k),
            given_kernel: None,
            forced_route: None,
            forced_scope: None,
        }
    }

    /// The micro-kernel of the blocked route: the given kernel, or the one
    /// `gemm_kernel::kernel_for_problem` picks for the plan's tier, policy
    /// and problem. The blocked route resolves it once per call.
    #[inline]
    pub(crate) fn micro_kernel(&self) -> MicroKernel<T> {
        match self.given_kernel {
            Some(kernel) => kernel,
            None => {
                let (m, n, k) = self.problem;
                let kernel =
                    kernel_for_problem::<T>(self.ctx.tier(), self.ctx.policy.f32_avx2, m, n, k);
                #[cfg(test)]
                call_probe::note_resolved(kernel.shape);
                kernel
            }
        }
    }

    /// This plan with the micro-kernel `kernel` given explicitly.
    #[cfg(test)]
    pub(crate) fn with_kernel(mut self, kernel: MicroKernel<T>) -> Self {
        self.given_kernel = Some(kernel);
        self
    }
}

impl<T> GemmPlan<T> {
    /// What decides this plan's micro-kernel ([`PlanKernelKey`]).
    #[cfg(test)]
    pub(crate) fn kernel_key(&self) -> PlanKernelKey {
        PlanKernelKey {
            problem: self.problem,
            given: self.given_kernel.as_ref().map(|kernel| kernel.shape),
        }
    }

    /// The route for the problem `inp`: the forced route when it accepts the
    /// problem, otherwise [`choose_route`]. (Test builds record it in
    /// `call_probe`, so a test can tell which route a public call took.)
    #[inline]
    pub(crate) fn route(&self, inp: &RouteInputs) -> GemmRoute {
        let route = match self.forced_route {
            Some(route) if route.accepts(inp) => route,
            _ => choose_route(inp, &self.ctx.policy),
        };
        #[cfg(test)]
        call_probe::note_route(route);
        route
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

    /// Whether this is a plan of the public entry points: no given
    /// micro-kernel, no forced route, no forced pack scope, and the tier
    /// [`SimdTier::Neon`].
    ///
    /// Only such a plan lets the SME2 backend take its blocked route (the
    /// GEMM hook in `gemm_core`) or its thin route (`gemm_thin`). The other
    /// plans are made by tests of the NEON internals and by A/B runs, which
    /// force a kernel, a route, a scope or a lower tier to measure exactly
    /// that code: a forced call must run what it forced, so the backend
    /// leaves it to the kernels it asked for. The tier matters because the
    /// backend's promise is about NEON (an SME2 result equals the NEON
    /// result bit for bit): a context lowered to [`SimdTier::Portable`]
    /// (`GemmContext::with_portable_tier`, test builds) asks for the portable
    /// kernels, whose bits SME2 does not reproduce. Every public entry point
    /// builds its plan with [`GemmPlan::for_problem`], which forces none of
    /// the three and takes the detected tier, `Neon` on every `aarch64` build
    /// in which the backend can run (only `force-scalar` gives `Portable`
    /// there: this crate's keeps the backend off; oxiblas-core's alone, which
    /// another crate of the build can turn on, gives every plan the portable
    /// tier, so no plan of such a build is a production plan).
    #[cfg(all(feature = "sme2", target_arch = "aarch64"))]
    #[inline]
    pub(crate) const fn is_production(&self) -> bool {
        self.given_kernel.is_none()
            && self.forced_route.is_none()
            && self.forced_scope.is_none()
            && matches!(self.ctx.tier(), SimdTier::Neon)
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

/// Per-thread counts of the per-call work that tiny products must not pay
/// for (tests only): the tests of the plan (`test_gemm_plan_*_resolves_*`)
/// require that a call on the small, small-transposed, small-copy or thin
/// route reads the tier once and resolves no micro-kernel, and that a
/// blocked call resolves exactly one.
///
/// The counts are per thread because tests run in parallel; a test reads the
/// counts of its own thread before and after one call ([`snapshot`]). The
/// probe also keeps the shape of the last micro-kernel a plan resolved
/// ([`last_resolved`]), so a test can tell which kernel a blocked call ran,
/// and the last route a plan chose ([`last_route`]), so a test can tell
/// which route a public call took.
#[cfg(test)]
pub(crate) mod call_probe {
    use super::{GemmRoute, MicroKernelShape};
    use std::cell::Cell;

    /// One counted event.
    #[derive(Debug, Clone, Copy)]
    pub(crate) enum Event {
        /// [`super::SimdTier::current`] (the tier read of a plan).
        TierRead,
        /// `gemm_kernel::kernel_for_problem`: the micro-kernel decision of a
        /// plan's blocked route.
        KernelResolution,
        /// `gemm_kernel::selected_kernel_kind`: the CPU-feature dispatch
        /// behind every canonical micro-kernel (and the uncached tier).
        KernelDispatch,
    }

    /// The counts of this thread so far.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub(crate) struct Counts {
        /// [`Event::TierRead`] events.
        pub(crate) tier_reads: usize,
        /// [`Event::KernelResolution`] events.
        pub(crate) kernel_resolutions: usize,
        /// [`Event::KernelDispatch`] events.
        pub(crate) kernel_dispatches: usize,
    }

    impl Counts {
        /// The events between `earlier` and `self` (both of this thread).
        pub(crate) fn since(self, earlier: Self) -> Self {
            Self {
                tier_reads: self.tier_reads - earlier.tier_reads,
                kernel_resolutions: self.kernel_resolutions - earlier.kernel_resolutions,
                kernel_dispatches: self.kernel_dispatches - earlier.kernel_dispatches,
            }
        }
    }

    thread_local! {
        static COUNTS: Cell<Counts> = const {
            Cell::new(Counts {
                tier_reads: 0,
                kernel_resolutions: 0,
                kernel_dispatches: 0,
            })
        };
        /// The shape of the micro-kernel that the last resolution of a plan
        /// ([`super::GemmPlan::micro_kernel`] without a given kernel) on this
        /// thread produced.
        static LAST_RESOLVED: Cell<Option<MicroKernelShape>> = const { Cell::new(None) };
        /// The route that the last route decision of a plan
        /// ([`super::GemmPlan::route`]) on this thread produced.
        static LAST_ROUTE: Cell<Option<GemmRoute>> = const { Cell::new(None) };
    }

    /// Records the route a plan just chose.
    pub(crate) fn note_route(route: GemmRoute) {
        // `try_with`: a call during thread teardown is simply not recorded.
        let _ = LAST_ROUTE.try_with(|last| last.set(Some(route)));
    }

    /// The route the last route decision of a plan on this thread produced
    /// (`None` before the first).
    pub(crate) fn last_route() -> Option<GemmRoute> {
        LAST_ROUTE.try_with(Cell::get).ok().flatten()
    }

    /// Forgets the last route of this thread, so that a call that returns
    /// before its route decision (`k == 0`, `alpha == 0`) leaves `None`.
    pub(crate) fn clear_route() {
        let _ = LAST_ROUTE.try_with(|last| last.set(None));
    }

    /// Records the shape of the micro-kernel a plan just resolved.
    pub(crate) fn note_resolved(shape: MicroKernelShape) {
        // `try_with`: a call during thread teardown is simply not recorded.
        let _ = LAST_RESOLVED.try_with(|last| last.set(Some(shape)));
    }

    /// The shape of the micro-kernel the last resolution of a plan on this
    /// thread produced (`None` before the first).
    pub(crate) fn last_resolved() -> Option<MicroKernelShape> {
        LAST_RESOLVED.try_with(Cell::get).ok().flatten()
    }

    /// Counts `event` on this thread.
    pub(crate) fn note(event: Event) {
        // `try_with`: a call during thread teardown is simply not counted.
        let _ = COUNTS.try_with(|counts| {
            let mut c = counts.get();
            match event {
                Event::TierRead => c.tier_reads += 1,
                Event::KernelResolution => c.kernel_resolutions += 1,
                Event::KernelDispatch => c.kernel_dispatches += 1,
            }
            counts.set(c);
        });
    }

    /// The counts of this thread so far.
    pub(crate) fn snapshot() -> Counts {
        COUNTS.try_with(Cell::get).unwrap_or_default()
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
            // NEON, unless oxiblas-core alone has `force-scalar` (another
            // crate of the build can turn it on): then the portable tier.
            let expected = if core_forces_scalar() {
                SimdTier::Portable
            } else {
                SimdTier::Neon
            };
            assert_eq!(tier, expected);
        }
        if cfg!(feature = "force-scalar") {
            assert!(core_forces_scalar());
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
        // A plan built with `new` holds exactly this tier, and the
        // dispatched kernel as its given kernel.
        let plan = GemmPlan::<f64>::new(LEGACY_POLICY);
        assert_eq!(plan.ctx.tier(), tier);
        assert_eq!(plan.micro_kernel().shape, f64::micro_kernel_shape());
        assert_eq!(plan.forced_route, None);
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

    /// Only a plan as the public entry points build it (`for_problem`: no
    /// given kernel, no forced route, no forced scope, the detected tier,
    /// which is `Neon` unless `force-scalar`) counts as the production plan
    /// that the SME2 hooks serve; each of the three markers alone, and a
    /// context lowered to the portable tier alone, makes it a forced plan.
    #[cfg(all(feature = "sme2", target_arch = "aarch64"))]
    #[test]
    fn test_gemm_plan_is_production_only_without_forcing() {
        fn check<T: GemmKernel>() {
            const ROUTES: [GemmRoute; 5] = [
                GemmRoute::Thin,
                GemmRoute::Small,
                GemmRoute::SmallTransCopy,
                GemmRoute::SmallTrans,
                GemmRoute::Blocked,
            ];
            for policy in [LEGACY_POLICY, CANDIDATE_X86, CANDIDATE_AARCH64] {
                for (m, n, k) in [(0, 0, 0), (1, 1, 1), (100, 100, 100), (4096, 1, 7)] {
                    let plan = GemmPlan::<T>::for_problem(policy, m, n, k);
                    let tier = plan.ctx.tier();
                    assert_eq!(
                        tier,
                        if cfg!(feature = "force-scalar") {
                            SimdTier::Portable
                        } else {
                            SimdTier::Neon
                        },
                        "the detected tier {m}x{n}x{k}"
                    );
                    assert_eq!(
                        plan.is_production(),
                        tier == SimdTier::Neon,
                        "{m}x{n}x{k} on the tier {tier:?}"
                    );
                    let mut lowered = plan;
                    lowered.ctx = lowered.ctx.with_portable_tier();
                    assert!(
                        !lowered.is_production(),
                        "portable tier {m}x{n}x{k}: a lowered tier counts as production"
                    );
                    let given = plan.with_kernel(T::micro_kernel_dispatch());
                    assert!(!given.is_production(), "given kernel {m}x{n}x{k}");
                    for route in ROUTES {
                        assert!(!plan.with_forced_route(route).is_production(), "{route:?}");
                    }
                    for scope in [PackScope::Shared, PackScope::PerTile] {
                        assert!(!plan.with_forced_scope(scope).is_production(), "{scope:?}");
                    }
                }
                assert!(!GemmPlan::<T>::new(policy).is_production(), "GemmPlan::new");
            }
        }
        check::<f64>();
        check::<f32>();
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
                    assert_eq!(plan.micro_kernel().shape, chosen.shape, "{m}x{n}x{k}");
                    if policy.f32_avx2 == F32Avx2Policy::Legacy8x8 {
                        assert_eq!(plan.micro_kernel().shape, canonical, "{m}x{n}x{k}");
                    }
                    assert_eq!(base.micro_kernel().shape, canonical);
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

    // The per-call cost of the plan, by structure (timing is measured by
    // `tools/gemm_tiny_call_check.sh`): with the counts of `call_probe`, a
    // call on a route without a micro-kernel reads the tier once and
    // dispatches nothing; a blocked call resolves exactly one micro-kernel.

    use super::call_probe::{Counts, snapshot};
    use crate::level3::gemm::{
        BlockingSource, GemmBlocking, gemm, gemm_auto, gemm_planned, gemm_transposed,
        gemm_with_blocking,
    };
    use oxiblas_core::parallel::Par;
    use oxiblas_matrix::Mat;

    /// What `call` did on this thread, after the tier cache is warm (the
    /// first detection of the process may happen on any thread).
    fn counts_of(call: impl FnOnce()) -> Counts {
        let _ = SimdTier::current();
        let before = snapshot();
        call();
        snapshot().since(before)
    }

    /// The counts a call on a route without a micro-kernel must show.
    const TINY: Counts = Counts {
        tier_reads: 1,
        kernel_resolutions: 0,
        kernel_dispatches: 0,
    };

    /// `k` as a `T` (a sum of ones): every element of `ones · ones`.
    fn sum_of_ones<T: Field>(k: usize) -> T {
        (0..k).fold(T::zero(), |acc, _| acc + T::one())
    }

    /// Every element of `c` is `x`.
    fn all_equal<T: Field>(c: &Mat<T>, x: T) -> bool {
        (0..c.ncols()).all(|j| (0..c.nrows()).all(|i| c[(i, j)] == x))
    }

    /// Operands of all ones for `op(A)` of `m × k` and `op(B)` of `k × n`.
    fn ones<T: Field + bytemuck::Zeroable>(
        (m, k, n): (usize, usize, usize),
        (ta, tb): (Trans, Trans),
    ) -> (Mat<T>, Mat<T>, Mat<T>) {
        let a_dims = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
        let b_dims = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
        (
            Mat::filled(a_dims.0, a_dims.1, T::one()),
            Mat::filled(b_dims.0, b_dims.1, T::one()),
            Mat::zeros(m, n),
        )
    }

    /// The route the default policy takes for this call of the public API.
    fn default_route<T: Field>(
        a: &Mat<T>,
        b: &Mat<T>,
        (m, k, n): (usize, usize, usize),
        (ta, tb): (Trans, Trans),
    ) -> GemmRoute {
        let inp = RouteInputs::new(
            ta,
            tb,
            &a.as_ref(),
            &b.as_ref(),
            m,
            n,
            k,
            SimdTier::current(),
        );
        choose_route(&inp, &DEFAULT_POLICY)
    }

    /// One public call (`gemm` for `NoTrans`/`NoTrans`, `gemm_transposed`
    /// otherwise): its route under the default policy and its counts. The
    /// result is checked, so a route that skipped the product would fail.
    fn public_call<T: GemmKernel + bytemuck::Zeroable>(
        dims: (usize, usize, usize),
        trans: (Trans, Trans),
    ) -> (GemmRoute, Counts) {
        let (a, b, mut c) = ones::<T>(dims, trans);
        let route = default_route(&a, &b, dims, trans);
        let counts = counts_of(|| {
            if trans == (Trans::NoTrans, Trans::NoTrans) {
                gemm(T::one(), a.as_ref(), b.as_ref(), T::zero(), c.as_mut());
            } else {
                let (ta, tb) = trans;
                gemm_transposed(
                    ta,
                    tb,
                    T::one(),
                    a.as_ref(),
                    b.as_ref(),
                    T::zero(),
                    c.as_mut(),
                );
            }
        });
        let expected = sum_of_ones::<T>(dims.1);
        assert!(all_equal(&c, expected), "{dims:?} {trans:?}");
        // The route the call took is the one the default policy gives.
        assert_eq!(
            super::call_probe::last_route(),
            Some(route),
            "{dims:?} {trans:?}"
        );
        (route, counts)
    }

    /// One call through `gemm_planned` with a plan of the default policy
    /// whose route is forced to `route` (the routes the default policy
    /// reaches only in some builds), and its counts.
    fn forced_call<T: GemmKernel + bytemuck::Zeroable>(
        dims: (usize, usize, usize),
        (ta, tb): (Trans, Trans),
        route: GemmRoute,
    ) -> Counts {
        let (a, b, mut c) = ones::<T>(dims, (ta, tb));
        let (m, k, n) = dims;
        let counts = counts_of(|| {
            let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k).with_forced_route(route);
            gemm_planned(
                ta,
                tb,
                T::one(),
                a.as_ref(),
                b.as_ref(),
                T::zero(),
                c.as_mut(),
                Par::Seq,
                BlockingSource::Default,
                &plan,
            );
        });
        let expected = sum_of_ones::<T>(k);
        assert!(all_equal(&c, expected), "{dims:?} {route:?}");
        counts
    }

    const NN: (Trans, Trans) = (Trans::NoTrans, Trans::NoTrans);
    const TRANSPOSED: [(Trans, Trans); 3] = [
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
        (Trans::ConjTrans, Trans::Trans),
    ];

    #[test]
    fn test_gemm_plan_tier_cache_one_read() {
        // The cached tier is the detected one, and every tier survives the
        // cache encoding; 0 means "not detected yet". The first read of a
        // process detects (each test may run in a process of its own), so
        // the cache is warmed first and the second read is the cached one.
        let _ = SimdTier::current();
        assert_eq!(SimdTier::current(), SimdTier::detect());
        for tier in ALL_TIERS {
            assert_eq!(SimdTier::from_cache(tier as u8), Some(tier));
        }
        assert_eq!(SimdTier::from_cache(0), None);
        assert_eq!(SimdTier::from_cache(5), None);
        // A warm read is one read and no CPU-feature dispatch.
        let counts = counts_of(|| {
            let _ = SimdTier::current();
        });
        assert_eq!(counts, TINY);
    }

    #[test]
    fn test_gemm_plan_for_problem_resolves_no_kernel() {
        fn check<T: GemmKernel>() {
            for policy in [LEGACY_POLICY, CANDIDATE_X86, CANDIDATE_AARCH64] {
                let mut plan = None;
                let counts = counts_of(|| plan = Some(GemmPlan::<T>::for_problem(policy, 7, 5, 3)));
                assert_eq!(counts, TINY, "{policy:?}");
                let plan = plan.unwrap_or_else(|| GemmPlan::<T>::for_problem(policy, 7, 5, 3));
                assert_eq!(plan.problem, (7, 5, 3));
                assert!(plan.given_kernel.is_none());
                // Resolving it is one resolution of the problem's kernel.
                let mut shape = None;
                let counts = counts_of(|| shape = Some(plan.micro_kernel().shape));
                assert_eq!(counts.kernel_resolutions, 1, "{policy:?}");
                let chosen = kernel_for_problem::<T>(plan.ctx.tier(), policy.f32_avx2, 7, 5, 3);
                assert_eq!(shape, Some(chosen.shape));
            }
            // A given kernel is used as it is, without a resolution.
            let given = GemmPlan::<T>::new(LEGACY_POLICY);
            let counts = counts_of(|| {
                let _ = given.micro_kernel();
            });
            assert_eq!(counts.kernel_resolutions, 0);
            assert_eq!(counts.kernel_dispatches, 0);
        }
        check::<f64>();
        check::<f32>();
        check::<num_complex::Complex64>();
    }

    #[test]
    fn test_gemm_plan_small_call_resolves_no_kernel() {
        fn check<T: GemmKernel + bytemuck::Zeroable>() {
            for s in [2, 3, 4, 5, 8, 12, 16, 32] {
                let (route, counts) = public_call::<T>((s, s, s), NN);
                assert_eq!(route, GemmRoute::Small, "{s}");
                assert_eq!(counts, TINY, "{s}^3");
            }
            // 1 × 1 × 1 and other thin shapes: the small route in the legacy
            // policy, the thin route in the candidates.
            for dims in [(1, 1, 1), (1, 8, 4), (4, 8, 1)] {
                let (route, counts) = public_call::<T>(dims, NN);
                assert!(
                    matches!(route, GemmRoute::Small | GemmRoute::Thin),
                    "{dims:?} {route:?}"
                );
                assert_eq!(counts, TINY, "{dims:?}");
            }
        }
        check::<f64>();
        check::<f32>();
        check::<num_complex::Complex64>();
    }

    #[test]
    fn test_gemm_plan_small_trans_call_resolves_no_kernel() {
        fn check<T: GemmKernel + bytemuck::Zeroable>() {
            for trans in TRANSPOSED {
                // Below 8³ the transposed products keep the loops in every
                // policy; from 8³ the candidates may copy to column-major.
                for s in [2, 3, 4] {
                    let (route, counts) = public_call::<T>((s, s, s), trans);
                    assert_eq!(route, GemmRoute::SmallTrans, "{s} {trans:?}");
                    assert_eq!(counts, TINY, "{s}^3 {trans:?}");
                }
                for s in [8, 16, 32] {
                    let (route, counts) = public_call::<T>((s, s, s), trans);
                    assert!(
                        matches!(route, GemmRoute::SmallTrans | GemmRoute::SmallTransCopy),
                        "{s} {trans:?} {route:?}"
                    );
                    assert_eq!(counts, TINY, "{s}^3 {trans:?}");
                }
                // The copy route with the default policy's plan, in every
                // build (the legacy policy never routes there itself).
                for s in [2, 8, 16] {
                    let counts = forced_call::<T>((s, s, s), trans, GemmRoute::SmallTransCopy);
                    assert_eq!(counts, TINY, "copy {s}^3 {trans:?}");
                }
            }
        }
        check::<f64>();
        check::<f32>();
        check::<num_complex::Complex64>();
    }

    #[test]
    fn test_gemm_plan_thin_call_resolves_no_kernel() {
        // The thin route with the default policy's plan (the legacy policy
        // reaches it only when it is forced); `gemm_thin` takes these
        // operands, so a declined call (which would fall back to the blocked
        // route and resolve a kernel) fails here too.
        fn check<T: GemmKernel + bytemuck::Zeroable>() {
            for dims in [
                (1, 64, 32),
                (32, 64, 1),
                (1, 1, 1),
                (1, 300, 200),
                (200, 300, 1),
            ] {
                for trans in [NN, (Trans::Trans, Trans::NoTrans)] {
                    let counts = forced_call::<T>(dims, trans, GemmRoute::Thin);
                    assert_eq!(counts, TINY, "{dims:?} {trans:?}");
                }
            }
        }
        check::<f64>();
        check::<f32>();
    }

    #[test]
    fn test_gemm_plan_blocked_call_resolves_one_kernel() {
        fn check<T: GemmKernel + bytemuck::Zeroable>() {
            let one_resolution = |counts: Counts, what: &str| {
                assert_eq!(counts.kernel_resolutions, 1, "{what}: {counts:?}");
                assert_eq!(counts.tier_reads, 1, "{what}: {counts:?}");
            };
            // Above `SMALL_THRESHOLD` the legacy policy blocks. A policy with
            // SIMD small kernels keeps column-major problems on a small route
            // up to its `small_simd_max_work` (the x86-64 candidate: 80³),
            // and such a call resolves nothing, like every small call.
            // Above every policy's ceiling the route is blocked everywhere.
            let small_ceiling = [LEGACY_POLICY, CANDIDATE_X86, CANDIDATE_AARCH64]
                .iter()
                .map(|policy| policy.small_simd_max_work)
                .fold(SMALL_THRESHOLD, usize::max);
            for dims in [
                (33, 32, 32),
                (48, 48, 48),
                (64, 40, 72),
                (81, 80, 80),
                (96, 64, 96),
            ] {
                let (m, k, n) = dims;
                for trans in [
                    NN,
                    (Trans::Trans, Trans::NoTrans),
                    (Trans::Trans, Trans::Trans),
                ] {
                    let (route, counts) = public_call::<T>(dims, trans);
                    let what = format!("public {dims:?} {trans:?}");
                    if m * n * k > small_ceiling || DEFAULT_POLICY == LEGACY_POLICY {
                        assert_eq!(route, GemmRoute::Blocked, "{what}");
                    }
                    if route == GemmRoute::Blocked {
                        one_resolution(counts, &what);
                    } else {
                        assert!(
                            matches!(route, GemmRoute::Small | GemmRoute::SmallTransCopy),
                            "{what}: {route:?}"
                        );
                        assert_eq!(counts, TINY, "{what}");
                    }
                }
            }
            // The other entry points that build a plan: heuristic targets
            // and caller-chosen blocking. They route like `gemm`, so the
            // first problem is blocked in the legacy policy only and the
            // second one in every policy.
            for dims in [(48, 40, 56), (88, 72, 96)] {
                let (m, k, n) = dims;
                let (a, b, mut c) = ones::<T>(dims, NN);
                let route = default_route(&a, &b, dims, NN);
                if m * n * k > small_ceiling || DEFAULT_POLICY == LEGACY_POLICY {
                    assert_eq!(route, GemmRoute::Blocked, "{dims:?}");
                }
                let counts = counts_of(|| {
                    gemm_auto(T::one(), a.as_ref(), b.as_ref(), T::zero(), c.as_mut());
                });
                let what = format!("gemm_auto {dims:?}");
                if route == GemmRoute::Blocked {
                    one_resolution(counts, &what);
                } else {
                    // The heuristic targets need the kernel's shape before
                    // the route is known (one dispatch), but a small route
                    // resolves no micro-kernel.
                    assert_eq!(route, GemmRoute::Small, "{what}");
                    assert_eq!(counts.kernel_resolutions, 0, "{what}: {counts:?}");
                    assert_eq!(counts.tier_reads, 1, "{what}: {counts:?}");
                }
                let expected = sum_of_ones::<T>(k);
                assert!(all_equal(&c, expected), "{what}");

                let (a, b, mut c) = ones::<T>(dims, NN);
                let blocking = GemmBlocking::default();
                let counts = counts_of(|| {
                    gemm_with_blocking(
                        T::one(),
                        a.as_ref(),
                        b.as_ref(),
                        T::zero(),
                        c.as_mut(),
                        Par::Seq,
                        &blocking,
                    );
                });
                let what = format!("gemm_with_blocking {dims:?}");
                if route == GemmRoute::Blocked {
                    one_resolution(counts, &what);
                } else {
                    assert_eq!(counts, TINY, "{what}");
                }
                assert!(all_equal(&c, expected), "{what}");
            }
            // A tiny problem forced onto the blocked route resolves one too.
            let counts = forced_call::<T>((2, 2, 2), NN, GemmRoute::Blocked);
            one_resolution(counts, "forced blocked 2^3");
        }
        check::<f64>();
        check::<f32>();
        check::<num_complex::Complex64>();
    }
}

#[cfg(test)]
#[path = "gemm_plan_blocked_tests.rs"]
mod blocked_tests;

#[cfg(test)]
#[path = "gemm_route_tests.rs"]
pub(crate) mod route_tests;

#[cfg(test)]
#[path = "gemm_route_public_tests.rs"]
mod route_public_tests;

#[cfg(test)]
#[path = "gemm_route_model_tests.rs"]
pub(crate) mod route_model_tests;

#[cfg(test)]
#[path = "gemm_route_public_check_tests.rs"]
mod route_public_check_tests;

#[cfg(test)]
#[path = "gemm_route_scope_tests.rs"]
mod route_scope_tests;

#[cfg(test)]
#[path = "gemm_route_storage_tests.rs"]
mod route_storage_tests;
