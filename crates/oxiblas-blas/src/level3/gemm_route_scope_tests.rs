//! The measured scope of the two adopted routes: operands stored
//! column-major, and for the thin route a tier with SIMD thin kernels.
//!
//! - [`wide_route`] is the router without the limits to the measured
//!   storage and tier, with its two predicates written out
//!   ([`wide_thin_in_scope`]: every thin product above 32³ whatever its
//!   storage and tier; [`wide_a_in_place`]: the copy of every product whose
//!   `op(A)` is not `NoTrans` with adjacent rows). Over the exhaustive grid
//!   the router may only narrow it: under every policy a route is either
//!   the wider route or the route of 0.2.3 (`LEGACY_POLICY`), and for
//!   operands stored column-major on a tier with SIMD thin kernels it is
//!   the wider route.
//! - Public calls on both sides of the scope: a thin product above 32³ and
//!   one from 1024 multiply-adds to 32³ take the thin route exactly when
//!   the build's policy has the scoped thin route and the call's tier has
//!   SIMD thin kernels, and otherwise the route and the bits of 0.2.3;
//!   operands not stored column-major take the route and the bits of 0.2.3
//!   on every tier. The same calls through a plan lowered to the portable
//!   tier show the tier condition in every build.

use super::call_probe::{clear_route, last_route};
use super::route_tests::{
    ALL_LAYOUTS, ALL_TIERS, ALL_TRANS, GRID, Layout, route_inputs, simd_thin_tier,
};
use super::*;
use crate::level3::gemm::{BlockingSource, gemm_planned, gemm_transposed, read_op};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::{MatMut, MatRef};

/// The scope without the limits to the measured storage and tier: above
/// 32³ every thin product; from 1024 to 32³ the four layouts
/// with both operands stored column-major and the vector operand not
/// transposed, from their least sizes.
fn wide_thin_in_scope(inp: &RouteInputs, work: usize) -> bool {
    if work > SMALL_THRESHOLD {
        return true;
    }
    if !(inp.a_unit_col_stride && inp.b_unit_col_stride) {
        return false;
    }
    let a_notrans = inp.trans_a == Trans::NoTrans;
    let b_notrans = inp.trans_b == Trans::NoTrans;
    let (min_vec, min_depth) = if inp.n == 1 {
        if !b_notrans {
            return false;
        }
        if a_notrans { (2, 2) } else { (2, 16) }
    } else {
        if !a_notrans {
            return false;
        }
        if b_notrans { (4, 2) } else { (4, 4) }
    };
    inp.m.max(inp.n) >= min_vec && inp.k >= min_depth
}

/// The copy's `op(A)` rule without the limit to the measured storage:
/// `op(A)` is `NoTrans` with adjacent rows (such a product is not copied,
/// every other one is).
fn wide_a_in_place(inp: &RouteInputs) -> bool {
    inp.trans_a == Trans::NoTrans && inp.a_unit_col_stride
}

/// The router without the limits to the measured storage and tier,
/// decision for decision.
fn wide_route(inp: &RouteInputs, policy: &GemmPolicy) -> GemmRoute {
    let work = inp.m.saturating_mul(inp.n).saturating_mul(inp.k);
    let nn = inp.trans_a == Trans::NoTrans && inp.trans_b == Trans::NoTrans;
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
    let col_major = nn && inp.a_unit_col_stride && inp.b_unit_col_stride;
    let simd_small = inp.simd_real
        && match inp.tier {
            SimdTier::Neon => inp.m >= 4 && inp.n >= 4,
            SimdTier::Avx2Fma => policy.small_simd_x86 && inp.m >= 5 && inp.n >= 4,
            SimdTier::Sse42 | SimdTier::Portable => false,
        };
    if simd_small && col_major && work <= policy.small_simd_max_work {
        return GemmRoute::Small;
    }
    let large_ok = if inp.n == 1 {
        inp.a_unit_col_stride || inp.a_unit_row_stride
    } else {
        inp.b_unit_col_stride || inp.b_unit_row_stride
    };
    let thin_ok = inp.simd_real && (inp.m == 1 || inp.n == 1) && large_ok;
    if (work >= THIN_MIN_WORK || policy.thin == ThinPolicy::All) && thin_ok {
        if work >= THIN_MIN_WORK && wide_thin_in_scope(inp, work) {
            if policy.thin != ThinPolicy::Off {
                return GemmRoute::Thin;
            }
        } else if policy.thin == ThinPolicy::All {
            return GemmRoute::Thin;
        }
    }
    let copy = matches!(policy.small_trans_copy, Some((lo, hi)) if lo <= work && work <= hi)
        && !(policy.small_trans_copy_a_only && wide_a_in_place(inp));
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

#[test]
fn test_gemm_route_scope_narrowing_only() {
    let policies = [
        ("adopted", ADOPTED_AARCH64),
        ("aarch64 candidate", CANDIDATE_AARCH64),
        ("x86-64 candidate", CANDIDATE_X86),
        ("legacy", LEGACY_POLICY),
        (
            "legacy + whole thin route",
            GemmPolicy {
                thin: ThinPolicy::All,
                ..LEGACY_POLICY
            },
        ),
        (
            "adopted + whole thin route",
            GemmPolicy {
                thin: ThinPolicy::All,
                ..ADOPTED_AARCH64
            },
        ),
    ];
    let mut kinds: Vec<(bool, SimdTier)> = ALL_TIERS.iter().map(|&t| (true, t)).collect();
    kinds.push((false, SimdTier::Neon));
    // Products the adopted policy moves back to 0.2.3's route, by the wider
    // route: [thin by storage, thin by tier, copy].
    let mut moved = [0usize; 3];
    for &m in &GRID {
        for &k in &GRID {
            for &n in &GRID {
                for ta in ALL_TRANS {
                    for tb in ALL_TRANS {
                        for la in ALL_LAYOUTS {
                            for lb in ALL_LAYOUTS {
                                for &(real, tier) in &kinds {
                                    let inp =
                                        route_inputs((m, k, n), (ta, tb), (la, lb), real, tier);
                                    let legacy = choose_route(&inp, &LEGACY_POLICY);
                                    let both_col = la.units().0 && lb.units().0;
                                    for (name, policy) in &policies {
                                        let new = choose_route(&inp, policy);
                                        let wide = wide_route(&inp, policy);
                                        assert!(
                                            new == wide || new == legacy,
                                            "{name} {inp:?}: {new:?}, wider {wide:?}, legacy {legacy:?}"
                                        );
                                        if both_col && simd_thin_tier(tier) {
                                            assert_eq!(new, wide, "{name} {inp:?}: column-major");
                                        }
                                        if both_col && new != wide {
                                            // Column-major operands change
                                            // only off the SIMD thin tiers,
                                            // and only off the thin route.
                                            assert_eq!(wide, GemmRoute::Thin, "{name} {inp:?}");
                                            assert!(!simd_thin_tier(tier), "{name} {inp:?}");
                                        }
                                        if policy.thin != ThinPolicy::Scoped
                                            && !policy.small_trans_copy_a_only
                                        {
                                            assert_eq!(
                                                new, wide,
                                                "{name} {inp:?}: unscoped policy"
                                            );
                                        }
                                        if *name == "adopted" && new != wide {
                                            let slot = match wide {
                                                GemmRoute::Thin if both_col => 1,
                                                GemmRoute::Thin => 0,
                                                _ => 2,
                                            };
                                            moved[slot] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(moved.iter().all(|&c| c > 0), "{moved:?}");
}

/// `f64` / `f32`.
trait Elem: GemmKernel + bytemuck::Zeroable + core::fmt::Debug {
    const EPS: f64;
    fn of(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
}

impl Elem for f64 {
    const EPS: f64 = f64::EPSILON;
    fn of(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl Elem for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn of(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// `(row_stride, col_stride)` of a stored `rows × cols` operand in `layout`
/// (`Col`: column-major with padding; `Row`: a transposed view of padded
/// storage; `Neither`: no unit stride).
fn strides(rows: usize, cols: usize, layout: Layout) -> (usize, usize) {
    match layout {
        Layout::Col | Layout::Both => (rows + 3, 1),
        Layout::Row => (1, cols + 2),
        Layout::Neither => (2 * rows + 1, 2),
    }
}

/// The storage of a `rows × cols` operand in `layout`.
fn stored<T: Elem>(rows: usize, cols: usize, layout: Layout, seed: usize) -> Vec<T> {
    let (rs, cs) = strides(rows, cols, layout);
    let len = (rows - 1) * cs + (cols - 1) * rs + 1;
    let mut data = vec![T::of(1.0e6); len];
    for j in 0..cols {
        for i in 0..rows {
            let v = ((i * 7 + j * 13 + seed * 5) % 23) as f64 / 7.0 - 1.1;
            data[i * cs + j * rs] = T::of(v);
        }
    }
    data
}

fn view<T: Elem>(data: &[T], rows: usize, cols: usize, layout: Layout) -> MatRef<'_, T> {
    let (rs, cs) = strides(rows, cols, layout);
    match MatRef::from_strided(data, rows, cols, rs, cs) {
        Some(v) => v,
        None => panic!("view out of bounds"),
    }
}

fn c_view<T: Elem>(data: &mut [T], m: usize, n: usize) -> MatMut<'_, T> {
    match MatMut::from_strided(data, m, n, m) {
        Some(v) => v,
        None => panic!("C out of bounds"),
    }
}

/// What the measured scope says about one case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// A thin product in the scope (both operands column-major): the thin
    /// route on a SIMD thin tier under a scoped policy.
    Thin,
    /// TN / TT of column-major operands in the copy's range: the copy on
    /// the NEON tier under a policy with `small_trans_copy_a_only`.
    Copy,
    /// An operand not stored column-major: the route of 0.2.3.
    Outside,
}

/// `(m, k, n)`, transposes, storage of A and B, the scope, 0.2.3's route.
type Case = (
    (usize, usize, usize),
    (Trans, Trans),
    (Layout, Layout),
    Scope,
    GemmRoute,
);

const N: Trans = Trans::NoTrans;
const T: Trans = Trans::Trans;
const C: Trans = Trans::ConjTrans;
const COL: Layout = Layout::Col;
const ROW: Layout = Layout::Row;
const NEITHER: Layout = Layout::Neither;

const CASES: [Case; 16] = [
    // Thin, above 32³ and from 1024 to 32³, column-major.
    (
        (4096, 9, 1),
        (N, N),
        (COL, COL),
        Scope::Thin,
        GemmRoute::Blocked,
    ),
    (
        (1, 9, 4096),
        (T, T),
        (COL, COL),
        Scope::Thin,
        GemmRoute::Blocked,
    ),
    (
        (2, 512, 1),
        (N, N),
        (COL, COL),
        Scope::Thin,
        GemmRoute::Small,
    ),
    (
        (1, 256, 4),
        (N, T),
        (COL, COL),
        Scope::Thin,
        GemmRoute::SmallTrans,
    ),
    // Thin with a large operand stored row-major, or a strided vector.
    (
        (4096, 9, 1),
        (N, N),
        (ROW, COL),
        Scope::Outside,
        GemmRoute::Blocked,
    ),
    (
        (4096, 9, 1),
        (T, N),
        (ROW, COL),
        Scope::Outside,
        GemmRoute::Blocked,
    ),
    (
        (4096, 9, 1),
        (N, N),
        (COL, ROW),
        Scope::Outside,
        GemmRoute::Blocked,
    ),
    (
        (1, 9, 4096),
        (N, N),
        (NEITHER, COL),
        Scope::Outside,
        GemmRoute::Blocked,
    ),
    (
        (2, 512, 1),
        (N, N),
        (ROW, COL),
        Scope::Outside,
        GemmRoute::Small,
    ),
    (
        (1, 256, 4),
        (N, N),
        (COL, ROW),
        Scope::Outside,
        GemmRoute::Small,
    ),
    // The copy's range: TN / TT column-major, then other storage.
    (
        (16, 16, 16),
        (T, N),
        (COL, COL),
        Scope::Copy,
        GemmRoute::SmallTrans,
    ),
    (
        (12, 12, 12),
        (C, T),
        (COL, COL),
        Scope::Copy,
        GemmRoute::SmallTrans,
    ),
    (
        (16, 16, 16),
        (N, N),
        (ROW, COL),
        Scope::Outside,
        GemmRoute::Small,
    ),
    (
        (16, 16, 16),
        (N, T),
        (NEITHER, COL),
        Scope::Outside,
        GemmRoute::SmallTrans,
    ),
    (
        (16, 16, 16),
        (T, N),
        (COL, ROW),
        Scope::Outside,
        GemmRoute::SmallTrans,
    ),
    (
        (16, 16, 16),
        (T, T),
        (ROW, COL),
        Scope::Outside,
        GemmRoute::SmallTrans,
    ),
];

/// The route the measured scope gives `scope` under `policy` on `tier`.
fn scoped_route(scope: Scope, legacy: GemmRoute, policy: &GemmPolicy, tier: SimdTier) -> GemmRoute {
    match scope {
        Scope::Thin if policy.thin != ThinPolicy::Off && simd_thin_tier(tier) => GemmRoute::Thin,
        Scope::Copy if policy.small_trans_copy_a_only && tier == SimdTier::Neon => {
            GemmRoute::SmallTransCopy
        }
        Scope::Thin | Scope::Copy | Scope::Outside => legacy,
    }
}

/// One call of `case` for `T`: through the public entry point (`plan ==
/// None`) or through `plan`. Returns the route and C.
fn run_case<T: Elem>(case: &Case, plan: Option<&GemmPlan<T>>) -> (Option<GemmRoute>, Vec<T>) {
    let ((m, k, n), (ta, tb), (la, lb), _, _) = *case;
    let (ar, ac) = if ta == N { (m, k) } else { (k, m) };
    let (br, bc) = if tb == N { (k, n) } else { (n, k) };
    let a = stored::<T>(ar, ac, la, 1 + m + 3 * k);
    let b = stored::<T>(br, bc, lb, 2 + n + 5 * k);
    let (av, bv) = (view(&a, ar, ac, la), view(&b, br, bc, lb));
    let mut c: Vec<T> = (0..m * n)
        .map(|i| T::of((i % 11) as f64 / 4.0 - 1.0))
        .collect();
    let (alpha, beta) = (T::of(-0.75), T::of(0.5));
    let c0: Vec<f64> = c.iter().map(|x| x.to_f64()).collect();
    clear_route();
    match plan {
        None => gemm_transposed(ta, tb, alpha, av, bv, beta, c_view(&mut c, m, n)),
        Some(plan) => gemm_planned(
            ta,
            tb,
            alpha,
            av,
            bv,
            beta,
            c_view(&mut c, m, n),
            Par::Seq,
            BlockingSource::Default,
            plan,
        ),
    }
    let route = last_route();
    // The value, within 2·(k + 2)·eps of the magnitudes.
    for j in 0..n {
        for i in 0..m {
            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
            for p in 0..k {
                let prod = read_op(&av, ta, i, p).to_f64() * read_op(&bv, tb, p, j).to_f64();
                exact += prod;
                magnitude += prod.abs();
            }
            let reference = -0.75 * exact + 0.5 * c0[i + j * m];
            let bound = 0.75 * magnitude + (0.5 * c0[i + j * m]).abs();
            let tol = 2.0 * (k as f64 + 2.0) * T::EPS * bound + f64::MIN_POSITIVE;
            let got = c[i + j * m].to_f64();
            assert!(
                (got - reference).abs() <= tol,
                "{case:?}: ({i},{j}) {got} vs {reference}"
            );
        }
    }
    (route, c)
}

/// Every case through the public entry point on the call's tier, and
/// through plans lowered to the portable tier; a route of 0.2.3 must give
/// the bits of a `LEGACY_POLICY` plan on the same tier.
fn check_cases<T: Elem>() -> usize {
    let tier = SimdTier::current();
    let mut legacy_calls = 0usize;
    for case in &CASES {
        let ((m, k, n), _, _, scope, legacy) = *case;
        let ctx = format!("{} {case:?} {tier:?}", core::any::type_name::<T>());
        // The public call, on the tier of this build and CPU.
        let want = scoped_route(scope, legacy, &DEFAULT_POLICY, tier);
        let (route, c) = run_case::<T>(case, None);
        assert_eq!(route, Some(want), "{ctx}: public call");
        if want == legacy {
            let plan = GemmPlan::<T>::for_problem(LEGACY_POLICY, m, n, k);
            let (_, c_legacy) = run_case::<T>(case, Some(&plan));
            let same = c.iter().zip(&c_legacy).all(|(x, y)| x.bits() == y.bits());
            assert!(same, "{ctx}: bits differ from the legacy plan");
            legacy_calls += 1;
        }
        // The default policy on the portable tier: no thin route and no
        // copy, the route and the bits of 0.2.3 on that tier.
        let mut portable = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
        portable.ctx = portable.ctx.with_portable_tier();
        let mut portable_legacy = GemmPlan::<T>::for_problem(LEGACY_POLICY, m, n, k);
        portable_legacy.ctx = portable_legacy.ctx.with_portable_tier();
        let (route, c) = run_case::<T>(case, Some(&portable));
        assert_eq!(route, Some(legacy), "{ctx}: portable tier");
        let (_, c_legacy) = run_case::<T>(case, Some(&portable_legacy));
        let same = c.iter().zip(&c_legacy).all(|(x, y)| x.bits() == y.bits());
        assert!(same, "{ctx}: portable bits differ from the legacy plan");
    }
    legacy_calls
}

#[test]
fn test_gemm_route_scope_public_calls_follow_storage_and_tier() {
    let tier = SimdTier::current();
    println!(
        "tier {tier:?}, default policy: {}",
        if DEFAULT_POLICY == ADOPTED_AARCH64 {
            "ADOPTED_AARCH64"
        } else if DEFAULT_POLICY == LEGACY_POLICY {
            "LEGACY_POLICY"
        } else {
            "a candidate"
        }
    );
    // The build of this crate without `force-scalar` on aarch64 has the
    // adopted policy (without a cfg); its tier is NEON unless oxiblas-core
    // alone has `force-scalar`, which makes it portable.
    if cfg!(all(target_arch = "aarch64", not(feature = "force-scalar")))
        && !cfg!(oxiblas_gemm_legacy)
        && !cfg!(oxiblas_gemm_candidate)
    {
        assert_eq!(DEFAULT_POLICY, ADOPTED_AARCH64);
        let expected = if core_forces_scalar() {
            SimdTier::Portable
        } else {
            SimdTier::Neon
        };
        assert_eq!(tier, expected);
    }
    // The x86-64 candidate copies every product in its range: its routes are
    // stated by the route tables, not by these cases.
    if DEFAULT_POLICY == CANDIDATE_X86 {
        println!("x86-64 candidate: covered by the route tables");
        return;
    }
    let legacy_calls = check_cases::<f64>() + check_cases::<f32>();
    // Off a SIMD thin tier, or under the legacy policy, every case keeps
    // the route of 0.2.3; on the NEON tier under the adopted policy only
    // the cases outside the scope do.
    let outside = CASES.iter().filter(|c| c.3 == Scope::Outside).count();
    assert_eq!(outside, 10);
    let adopted_neon = DEFAULT_POLICY.thin == ThinPolicy::Scoped && tier == SimdTier::Neon;
    let expected = if adopted_neon { outside } else { CASES.len() };
    assert_eq!(legacy_calls, 2 * expected, "{tier:?}");
}
