//! Variants, cases and the group runs built from the case tables.

use super::super::autotune::{ISSUE_23_B_CAPS, ISSUE_23_B_CAPS_SHALLOW_SINGLE_PASS};
use super::super::gemm_plan::{
    BCapScope, GemmPlan, GemmPolicy, LEGACY_POLICY, RouteInputs, SimdTier, ThinPolicy,
};
use super::check::AbReal;
use super::engage::{Engage, path_of};
use super::specs::{self, BOTH, Beta, Dtype, PACK_CACHE, PACK_CACHE_CAP_OFF, Shape, TransPair};
use super::stats::Threshold;
use oxiblas_core::parallel::Par;

#[cfg(feature = "parallel")]
use super::super::gemm::PARALLEL_GATE;
#[cfg(target_arch = "x86_64")]
use super::super::gemm_plan::CANDIDATE_X86;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::super::gemm_policy::{SMALL_TRANS_COPY_A_ONLY_RANGE, SMALL_TRANS_COPY_FULL_RANGE};
#[cfg(target_arch = "x86_64")]
use super::{
    super::gemm_kernel::{MicroKernel, MicroKernelShape, f32_avx2_picks_16x6, kernel_for_problem},
    super::gemm_plan::F32Avx2Policy,
    super::gemm_small::SMALL_THRESHOLD,
    specs::F32_ONLY,
};

/// One mebibyte.
pub(super) const MIB: usize = 1 << 20;

/// The micro-kernel of a variant's plans: the plan's own choice, or (x86-64)
/// the f32 16×6 AVX2 + FMA kernel on every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KernelPick {
    Plan,
    #[cfg(target_arch = "x86_64")]
    F32x16x6,
}

/// One side of an A/B comparison: a policy, the kernel pick, and the
/// packing-cache limit of the calling thread and of every pool worker while
/// it runs (`None`: the default, 8 MiB).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Variant {
    pub(super) name: &'static str,
    pub(super) policy: GemmPolicy,
    pub(super) kernel: KernelPick,
    pub(super) limit: Option<usize>,
    /// Its verdict is informational and no partial range is computed for
    /// it: no policy field can express this candidate (a forced kernel), or
    /// its group is an information group (`ab_seq_thin_grid`).
    pub(super) informational: bool,
}

/// [`LEGACY_POLICY`] changed by `edit`.
fn legacy_with(edit: impl FnOnce(&mut GemmPolicy)) -> GemmPolicy {
    let mut policy = LEGACY_POLICY;
    edit(&mut policy);
    policy
}

impl Variant {
    /// The variant `name` with `policy`.
    pub(super) const fn with(name: &'static str, policy: GemmPolicy) -> Self {
        Self {
            name,
            policy,
            kernel: KernelPick::Plan,
            limit: None,
            informational: false,
        }
    }

    /// [`LEGACY_POLICY`] as it ships.
    pub(super) const fn legacy(name: &'static str) -> Self {
        Self::with(name, LEGACY_POLICY)
    }

    /// This variant renamed, with a packing-cache limit of `mib` MiB.
    pub(super) const fn limited(mut self, name: &'static str, mib: usize) -> Self {
        self.name = name;
        self.limit = Some(mib * MIB);
        self
    }

    /// The plan of this variant for an `m × k` by `k × n` product, built as
    /// the public entry points build theirs (`GemmPlan::for_problem`), with
    /// the forced kernel where the variant forces one. The engine builds it
    /// for every call it makes, timed or not.
    pub(super) fn plan<T: AbReal>(&self, dims: Dims) -> GemmPlan<T> {
        let plan = GemmPlan::<T>::for_problem(self.policy, dims.m, dims.n, dims.k);
        match self.kernel {
            KernelPick::Plan => plan,
            #[cfg(target_arch = "x86_64")]
            KernelPick::F32x16x6 => plan.with_kernel(forced_f32_16x6::<T>()),
        }
    }
}

/// The f32 16×6 kernel (AVX2 + FMA code: call only on that tier).
#[cfg(target_arch = "x86_64")]
fn forced_f32_16x6<T: AbReal>() -> MicroKernel<T> {
    // A large square: the padding model always picks 16×6 for it.
    let (tier, adaptive) = (SimdTier::Avx2Fma, F32Avx2Policy::Adaptive);
    let kernel = kernel_for_problem::<T>(tier, adaptive, 4096, 4096, 4096);
    let shape = MicroKernelShape { mr: 16, nr: 6 };
    assert_eq!(kernel.shape, shape, "16x6 exists only for f32 (8x8 tier)");
    kernel
}

/// `op(A)` is `m × k`, `op(B)` is `k × n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Dims {
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
}

impl Dims {
    /// `m·n·k`.
    pub(super) const fn work(&self) -> u128 {
        self.m as u128 * self.n as u128 * self.k as u128
    }

    /// `[m, k, n]`.
    pub(super) const fn mkn(&self) -> [usize; 3] {
        [self.m, self.k, self.n]
    }
}

/// One problem of a group run, with the targets it intends for each
/// candidate.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Case {
    pub(super) dtype: Dtype,
    pub(super) trans: TransPair,
    pub(super) dims: Dims,
    pub(super) beta: Beta,
    pub(super) quick: bool,
    /// The table names it a target (B-cap and cache-limit groups).
    pub(super) named_target: bool,
    /// Per candidate (`variants[1..]`): whether the group intends it as a
    /// target (see [`GroupSpec::intended_for`]).
    pub(super) intended: Vec<bool>,
}

impl Case {
    /// The case of `shape` for `dtype`.
    fn from_shape(shape: &Shape, dtype: Dtype) -> Self {
        Self {
            dtype,
            trans: shape.trans,
            dims: Dims {
                m: shape.m,
                k: shape.k,
                n: shape.n,
            },
            beta: shape.beta,
            quick: shape.quick.includes(dtype),
            named_target: shape.target,
            intended: Vec::new(),
        }
    }

    /// `TT_MxKxN_dtype`, with `_beta1` for `beta = 1`.
    pub(super) fn label(&self) -> String {
        let Dims { m, k, n } = self.dims;
        let (trans, dtype) = (self.trans.name(), self.dtype.name());
        let beta = match self.beta {
            Beta::Zero => "",
            Beta::One => "_beta1",
        };
        format!("{trans}_{m}x{k}x{n}_{dtype}{beta}")
    }

    /// Router inputs for plain column-major operands at `tier`.
    pub(super) fn route_inputs(&self, tier: SimdTier) -> RouteInputs {
        let Dims { m, k, n } = self.dims;
        RouteInputs {
            m,
            n,
            k,
            trans_a: self.trans.a,
            trans_b: self.trans.b,
            a_unit_col_stride: true,
            a_unit_row_stride: false,
            b_unit_col_stride: true,
            b_unit_row_stride: false,
            simd_real: true,
            tier,
        }
    }
}

/// The cases of `shapes` for each of `dtypes` (all shapes of the first type
/// first).
fn cases(dtypes: &[Dtype], shapes: &[Shape]) -> Vec<Case> {
    let per_type = |dtype: Dtype| shapes.iter().map(move |s| Case::from_shape(s, dtype));
    dtypes.iter().flat_map(|&d| per_type(d)).collect()
}

/// The SIMD tier the candidates of this architecture are designed for.
pub(super) const DESIGN_TIER: SimdTier = if cfg!(target_arch = "x86_64") {
    SimdTier::Avx2Fma
} else if cfg!(target_arch = "aarch64") {
    SimdTier::Neon
} else {
    SimdTier::Portable
};

/// The cases and variants of one group run. `run` is `seq`, `t<threads>` or
/// the cache group's cap setting; `variants[0]` is A (the control A′ is added
/// when a case runs).
#[derive(Debug, Clone)]
pub(super) struct GroupSpec {
    pub(super) name: &'static str,
    pub(super) run: String,
    pub(super) threads: usize,
    /// A parallel group (runs in an installed pool of `threads` threads).
    pub(super) parallel: bool,
    /// The `Par` the products are called with.
    pub(super) par: Par,
    pub(super) engage: Engage,
    pub(super) threshold: Threshold,
    pub(super) variants: Vec<Variant>,
    pub(super) cases: Vec<Case>,
}

/// A sequential group run.
fn group(
    name: &'static str,
    (engage, threshold): (Engage, Threshold),
    variants: Vec<Variant>,
    cases: Vec<Case>,
) -> GroupSpec {
    GroupSpec {
        name,
        run: "seq".to_string(),
        threads: 1,
        parallel: false,
        par: Par::Seq,
        engage,
        threshold,
        variants,
        cases,
    }
    .with_intended()
}

impl GroupSpec {
    /// The candidates (every variant but A).
    pub(super) fn candidates(&self) -> &[Variant] {
        self.variants.get(1..).unwrap_or(&[])
    }

    /// The `Par` a product of this run is expected to see: computed from the
    /// thread count, never from [`GroupSpec::par`], so that the parallel
    /// checks catch a call made with another `Par`.
    pub(super) fn expected_par(&self) -> Par {
        #[cfg(feature = "parallel")]
        if self.parallel {
            return Par::RayonWith(self.threads);
        }
        Par::Seq
    }

    /// The variants differ in their packing-cache limit.
    pub(super) fn limits_differ(&self) -> bool {
        let first = self.variants.first().map(|v| v.limit);
        self.variants.iter().any(|v| Some(v.limit) != first)
    }

    /// This group run on `threads` threads (parallel groups).
    #[cfg(feature = "parallel")]
    fn on(mut self, threads: usize) -> Self {
        (self.run, self.threads, self.parallel) = (format!("t{threads}"), threads, true);
        self.par = Par::RayonWith(threads);
        self.with_intended()
    }

    /// This group run named `run`.
    fn named(mut self, run: &str) -> Self {
        self.run = run.to_string();
        self
    }

    /// The cases of this run (all, or the quick subset).
    pub(super) fn selected(&self, quick: bool) -> impl Iterator<Item = &Case> {
        self.cases.iter().filter(move |c| c.quick || !quick)
    }

    /// Whether the group intends `case` as a target of `cand`: where the
    /// design says the candidate changes what runs (the router at the design
    /// tier, the padding model of the f32 kernel, the parallel gate of the
    /// pack scope), or where the table names a target (B caps, cache limit).
    pub(super) fn intended_for(&self, case: &Case, cand: &Variant) -> bool {
        let base = self.variants.first().copied().unwrap_or(*cand);
        match self.engage {
            Engage::Route => {
                let inp = case.route_inputs(DESIGN_TIER);
                path_of(&inp, &cand.policy) != path_of(&inp, &base.policy)
            }
            Engage::Kernel => intended_kernel(case, cand),
            Engage::Scope => self.intended_scope(case, &base, cand),
            Engage::BCap | Engage::CacheLimit => case.named_target,
        }
    }

    /// Scope groups: the candidate plans another scope and the gate passes.
    fn intended_scope(&self, case: &Case, base: &Variant, cand: &Variant) -> bool {
        #[cfg(feature = "parallel")]
        {
            let work = usize::try_from(case.dims.work()).unwrap_or(usize::MAX);
            let gate = PARALLEL_GATE.should_parallelize(work, self.expected_par());
            let scope = |v: &Variant| GemmPlan::<f64>::new(v.policy).scope(self.threads);
            scope(base) != scope(cand) && gate
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = (case, base, cand);
            false
        }
    }

    /// Sets every case's intended targets.
    fn with_intended(mut self) -> Self {
        let intended: Vec<Vec<bool>> = self
            .cases
            .iter()
            .map(|c| {
                self.candidates()
                    .iter()
                    .map(|v| self.intended_for(c, v))
                    .collect()
            })
            .collect();
        for (case, flags) in self.cases.iter_mut().zip(intended) {
            case.intended = flags;
        }
        self
    }
}

/// The f32 kernel group: the padding model picks 16×6 at the AVX2 tier, or
/// the variant forces it.
#[cfg(target_arch = "x86_64")]
fn intended_kernel(case: &Case, cand: &Variant) -> bool {
    let Dims { m, k, n } = case.dims;
    let (tier, canonical) = (SimdTier::Avx2Fma, MicroKernelShape { mr: 8, nr: 8 });
    let picks = f32_avx2_picks_16x6(tier, cand.policy.f32_avx2, canonical, m, n, k);
    picks || cand.kernel == KernelPick::F32x16x6
}

/// No f32 kernel group off x86-64.
#[cfg(not(target_arch = "x86_64"))]
fn intended_kernel(_case: &Case, _cand: &Variant) -> bool {
    false
}

/// `ab_seq_f32_kernel`: the f32 AVX2 kernel of the plan, and 16×6 forced.
#[cfg(target_arch = "x86_64")]
pub(super) fn spec_seq_f32_kernel() -> GroupSpec {
    let adaptive = legacy_with(|p| p.f32_avx2 = F32Avx2Policy::Adaptive);
    // The forced kernel is no policy field: informational only.
    let mut forced = Variant::legacy("forced16x6");
    forced.kernel = KernelPick::F32x16x6;
    forced.informational = true;
    let variants = vec![
        Variant::legacy("legacy8x8"),
        Variant::with("adaptive", adaptive),
        forced,
    ];
    let kind = (Engage::Kernel, Threshold::LowerWork);
    let cases = cases(&F32_ONLY, &specs::F32_KERNEL);
    group("ab_seq_f32_kernel", kind, variants, cases)
}

/// `ab_seq_small_simd`: the AVX2 + FMA small kernels up to two ceilings.
#[cfg(target_arch = "x86_64")]
pub(super) fn spec_seq_small_simd() -> GroupSpec {
    let up_to = |name, max: usize| {
        let policy = legacy_with(|p| {
            p.small_simd_x86 = true;
            p.small_simd_max_work = max;
        });
        Variant::with(name, policy)
    };
    let variants = vec![
        Variant::legacy("off"),
        up_to("simd<=32^3", SMALL_THRESHOLD),
        up_to("simd<=80^3", CANDIDATE_X86.small_simd_max_work),
    ];
    let kind = (Engage::Route, Threshold::UpperWork);
    let cases = cases(&BOTH, &specs::SMALL_SIMD);
    group("ab_seq_small_simd", kind, variants, cases)
}

/// `ab_seq_small_trans_copy`: transposed small problems copied to
/// column-major for the SIMD small kernels: every transposed or strided
/// product from 8³ (`copy<=32^3`; on x86-64 also up to the 80³ ceiling of
/// the AVX2 small kernels, `copy<=80^3`), and only the products with A
/// transposed and both operands stored column-major, from 12³
/// (`copy_a_only>=12^3`, the range adopted on aarch64; the cases are all
/// stored column-major).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) fn spec_seq_small_trans_copy() -> GroupSpec {
    // The AVX2 + FMA tier has small kernels only with `small_simd_x86`.
    let x86 = cfg!(target_arch = "x86_64");
    let copy = |name, range: (usize, usize), a_only: bool| {
        let policy = legacy_with(|p| {
            p.small_simd_x86 = x86;
            p.small_trans_copy = Some(range);
            p.small_trans_copy_a_only = a_only;
        });
        Variant::with(name, policy)
    };
    let mut variants = vec![
        Variant::legacy("off"),
        copy("copy<=32^3", SMALL_TRANS_COPY_FULL_RANGE, false),
    ];
    #[cfg(target_arch = "aarch64")]
    let shapes = specs::SMALL_TRANS_COPY.to_vec();
    #[cfg(target_arch = "x86_64")]
    let shapes = [&specs::SMALL_TRANS_COPY[..], &specs::SMALL_TRANS_COPY_X86].concat();
    #[cfg(target_arch = "x86_64")]
    {
        let (lo, hi) = SMALL_TRANS_COPY_FULL_RANGE;
        let wide = (lo, hi.max(CANDIDATE_X86.small_simd_max_work));
        variants.push(copy("copy<=80^3", wide, false));
    }
    variants.push(copy(
        "copy_a_only>=12^3",
        SMALL_TRANS_COPY_A_ONLY_RANGE,
        true,
    ));
    let kind = (Engage::Route, Threshold::WorkInterval);
    let cases = cases(&BOTH, &shapes);
    group("ab_seq_small_trans_copy", kind, variants, cases)
}

/// A thin group named `name` on `sizes`: the whole thin route (`thin`) and
/// the thin route within the measured scope (`thin_scoped`, the field
/// adopted on aarch64). With `informational`, the candidates' verdicts only
/// inform (an information group).
fn spec_thin(name: &'static str, sizes: &specs::ThinSizes, informational: bool) -> GroupSpec {
    let mut thin = Variant::with("thin", legacy_with(|p| p.thin = ThinPolicy::All));
    let mut scoped = Variant::with("thin_scoped", legacy_with(|p| p.thin = ThinPolicy::Scoped));
    thin.informational = informational;
    scoped.informational = informational;
    let variants = vec![Variant::legacy("off"), thin, scoped];
    let kind = (Engage::Route, Threshold::None);
    group(
        name,
        kind,
        variants,
        cases(&BOTH, &specs::thin_shapes(sizes)),
    )
}

/// `ab_seq_thin`: the sequential thin route, the decision group (the keep
/// rule judges it).
pub(super) fn spec_seq_thin() -> GroupSpec {
    spec_thin("ab_seq_thin", &specs::SEQ_THIN, false)
}

/// `ab_seq_thin_grid`: the sequential thin route on the dense grid, the
/// information group (its verdicts are informational).
pub(super) fn spec_seq_thin_grid() -> GroupSpec {
    spec_thin("ab_seq_thin_grid", &specs::SEQ_THIN_GRID, true)
}

/// The names of a B-cap group's variants.
#[derive(Debug, Clone, Copy)]
struct BCapNames {
    off: &'static str,
    capped: &'static str,
    alt: &'static str,
}

/// A B-cap group with `scope`: both readings of the rule.
fn spec_b_cap(name: &'static str, scope: BCapScope, names: BCapNames) -> GroupSpec {
    let capped = legacy_with(|p| {
        p.b_cap = scope;
        p.b_caps = ISSUE_23_B_CAPS;
    });
    let alt = GemmPolicy {
        b_caps: ISSUE_23_B_CAPS_SHALLOW_SINGLE_PASS,
        ..capped
    };
    let variants = vec![
        Variant::legacy(names.off),
        Variant::with(names.capped, capped),
        Variant::with(names.alt, alt),
    ];
    let kind = (Engage::BCap, Threshold::None);
    group(name, kind, variants, cases(&BOTH, &specs::B_CAP))
}

/// `ab_seq_b_cap`: the issue #23 B-block caps in the sequential driver.
pub(super) fn spec_seq_b_cap() -> GroupSpec {
    let names = BCapNames {
        off: "cap_off",
        capped: "cap_seq",
        alt: "cap_seq_alt",
    };
    spec_b_cap("ab_seq_b_cap", BCapScope::Sequential, names)
}

/// `ab_seq_pack_cache` with the B cap `cap` (`Off`: run `cap_off`; otherwise
/// `cap_on`): the per-thread cache limit at 8 (A), 16 and 32 MiB.
pub(super) fn spec_seq_pack_cache(cap: BCapScope) -> GroupSpec {
    let base = Variant::with("limit8MiB", legacy_with(|p| p.b_cap = cap));
    let variants = vec![
        base,
        base.limited("limit16MiB", 16),
        base.limited("limit32MiB", 32),
    ];
    let mut shapes = PACK_CACHE.to_vec();
    let run = if cap == BCapScope::Off {
        shapes.extend(PACK_CACHE_CAP_OFF);
        "cap_off"
    } else {
        "cap_on"
    };
    let kind = (Engage::CacheLimit, Threshold::None);
    group("ab_seq_pack_cache", kind, variants, cases(&BOTH, &shapes)).named(run)
}

/// `ab_par_pack_scope`: shared against per-tile packing on `threads`.
#[cfg(feature = "parallel")]
pub(super) fn spec_par_pack_scope(threads: usize) -> GroupSpec {
    let per_tile = Variant::with("per_tile", legacy_with(|p| p.per_tile_min_threads = 1));
    // Per tile with room for A and B on every worker (with 8 MiB, A is freed
    // after every task for some shapes).
    let roomy = per_tile.limited("per_tile_16MiB", 16);
    let variants = vec![Variant::legacy("shared"), per_tile, roomy];
    let kind = (Engage::Scope, Threshold::None);
    let cases = cases(&BOTH, &specs::PACK_SCOPE);
    group("ab_par_pack_scope", kind, variants, cases).on(threads)
}

/// `ab_par_thin`: the parallel thin split on `threads`.
#[cfg(feature = "parallel")]
pub(super) fn spec_par_thin(threads: usize) -> GroupSpec {
    spec_thin("ab_par_thin", &specs::PAR_THIN, false).on(threads)
}

/// `ab_par_b_cap`: the B caps in both drivers (`BCapScope::All`) on
/// `threads`.
#[cfg(feature = "parallel")]
pub(super) fn spec_par_b_cap(threads: usize) -> GroupSpec {
    let names = BCapNames {
        off: "cap_off",
        capped: "cap_all",
        alt: "cap_all_alt",
    };
    spec_b_cap("ab_par_b_cap", BCapScope::All, names).on(threads)
}

/// Every group run this build has (the parallel ones for each of
/// `threads`), in the order of the module documentation.
pub(super) fn all_specs(threads: &[usize]) -> Vec<GroupSpec> {
    let mut specs = Vec::new();
    #[cfg(target_arch = "x86_64")]
    specs.extend([spec_seq_f32_kernel(), spec_seq_small_simd()]);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    specs.push(spec_seq_small_trans_copy());
    specs.extend([spec_seq_thin(), spec_seq_thin_grid(), spec_seq_b_cap()]);
    specs.extend([BCapScope::Off, BCapScope::Sequential].map(spec_seq_pack_cache));
    #[cfg(feature = "parallel")]
    for &t in threads {
        specs.extend([spec_par_pack_scope(t), spec_par_thin(t), spec_par_b_cap(t)]);
    }
    let _ = threads;
    specs
}
