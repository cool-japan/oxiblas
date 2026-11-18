//! The router against two oracles that do not share its code.
//!
//! - [`stated_route`]: the routing rules restated from the measured outcome
//!   that decided the defaults (the scope table of the thin route, the
//!   small transposed copy and its `op(A)` rule, both limited to what was
//!   measured: operands stored column-major, and for the thin route a tier
//!   with SIMD thin kernels; the SIMD small route, the small threshold),
//!   with every number written out. It has no order of
//!   tests: each route's set of products is stated on its own, and the sets
//!   are disjoint (a thin product has `m == 1` or `n == 1`, the SIMD small
//!   kernels and the copy need `m, n >= 4`), so any order of the router's
//!   tests that gives the same routes passes.
//! - [`plain_route`]: the router in its plain form, written out with its
//!   helpers inlined (a boolean thin switch tested first, no scope, no tiny
//!   shortcut, no `op(A)` rule). For every policy without the scope and
//!   without the `op(A)` rule the current router must give its routes:
//!   `ThinPolicy::Off` is its `thin = false`, `ThinPolicy::All` its
//!   `thin = true`. With the legacy policy's fields this is the router whose
//!   routes reproduce 0.2.3.
//!
//! The grids: every `m, n, k` in `0..=40`; thin shapes (`m == 1` or
//! `n == 1`) whose other two sizes run over `1..=40`, the powers of two up
//! to 131072 with their neighbours and the neighbours of every limit; and
//! every ordered factorisation of the work values next to a limit (511 to
//! 513, 1023 to 1025, 1727 to 1729, 32767 to 32769, and 511999 to 512001
//! around the x86-64 candidate's 80³). Each shape is checked with the nine
//! transpose pairs, the 16 combinations of the four stride flags, and
//! `f64`/`f32` and another element type on each of the four tiers.

use super::super::gemm_policy::ADOPTED_POLICY;
use super::*;

/// The route-relevant fields of a policy, as the stated rules give them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rules {
    thin: ThinPolicy,
    x86_small: bool,
    simd_max: usize,
    copy: Option<(usize, usize)>,
    a_only: bool,
}

/// 0.2.3: no thin route, no copy, the small loops up to 32³.
pub(crate) const RULES_LEGACY: Rules = Rules {
    thin: ThinPolicy::Off,
    x86_small: false,
    simd_max: 32_768,
    copy: None,
    a_only: false,
};

/// Adopted on aarch64: the scoped thin route, the copy from 12³ to 32³ of
/// a transposed A with both operands stored column-major.
pub(crate) const RULES_ADOPTED: Rules = Rules {
    thin: ThinPolicy::Scoped,
    x86_small: false,
    simd_max: 32_768,
    copy: Some((1728, 32_768)),
    a_only: true,
};

/// The x86-64 candidate: the scoped thin route, the AVX2 small kernels up
/// to 80³, the copy of every transposed or strided product from 8³.
pub(crate) const RULES_CANDIDATE_X86: Rules = Rules {
    thin: ThinPolicy::Scoped,
    x86_small: true,
    simd_max: 512_000,
    copy: Some((512, 32_768)),
    a_only: false,
};

/// The route-relevant fields of `p`.
fn rules_of(p: &GemmPolicy) -> Rules {
    Rules {
        thin: p.thin,
        x86_small: p.small_simd_x86,
        simd_max: p.small_simd_max_work,
        copy: p.small_trans_copy,
        a_only: p.small_trans_copy_a_only,
    }
}

/// A policy with the route-relevant fields `r` (the rest legacy).
fn policy_of(r: Rules) -> GemmPolicy {
    GemmPolicy {
        thin: r.thin,
        small_simd_x86: r.x86_small,
        small_simd_max_work: r.simd_max,
        small_trans_copy: r.copy,
        small_trans_copy_a_only: r.a_only,
        ..LEGACY_POLICY
    }
}

/// The route of `inp` under `r`, from the stated rules alone.
pub(crate) fn stated_route(inp: &RouteInputs, r: Rules) -> GemmRoute {
    let (m, n, k) = (inp.m as u128, inp.n as u128, inp.k as u128);
    let work = m * n * k;
    let ta = inp.trans_a != Trans::NoTrans; // ConjTrans of a real matrix is Trans
    let tb = inp.trans_b != Trans::NoTrans;
    let (a_col, a_row) = (inp.a_unit_col_stride, inp.a_unit_row_stride);
    let (b_col, b_row) = (inp.b_unit_col_stride, inp.b_unit_row_stride);

    // A thin product: f64/f32, a column (n == 1, also m == n == 1, the
    // large operand A) or a row (m == 1, n > 1, the large operand B) of C,
    // the large operand stored with adjacent rows or adjacent columns.
    let thin_product = inp.simd_real
        && if n == 1 {
            a_col || a_row
        } else {
            m == 1 && (b_col || b_row)
        };
    if thin_product {
        let thin = match r.thin {
            ThinPolicy::Off => false,
            ThinPolicy::All => true,
            ThinPolicy::Scoped => {
                // What was measured: both operands column-major, on the NEON
                // or the AVX2 + FMA tier. The scope table: above 32³ every
                // transpose pair; from 1024 to 32³ four layouts, by least
                // max(m, n) and least k.
                let both_col = a_col && b_col;
                let simd_thin_tier = matches!(inp.tier, SimdTier::Neon | SimdTier::Avx2Fma);
                let v = m.max(n);
                let row = if n == 1 {
                    match (ta, tb) {
                        (false, false) => v >= 2 && k >= 2, // THIN_COL_AXPY
                        (true, false) => v >= 2 && k >= 16, // THIN_COL_DOT
                        _ => false,
                    }
                } else {
                    match (ta, tb) {
                        (false, false) => v >= 4 && k >= 2, // THIN_ROW_DOT
                        (false, true) => v >= 4 && k >= 4,  // THIN_ROW_AXPY
                        _ => false,
                    }
                };
                both_col && simd_thin_tier && (work > 32_768 || (work >= 1024 && row))
            }
        };
        if thin {
            return GemmRoute::Thin;
        }
    }

    // A tier with SIMD small kernels for this shape: NEON from 4 × 4, the
    // AVX2 + FMA tier from 5 × 4 where the policy has them.
    let kernels = inp.simd_real
        && match inp.tier {
            SimdTier::Neon => m >= 4 && n >= 4,
            SimdTier::Avx2Fma => r.x86_small && m >= 5 && n >= 4,
            SimdTier::Portable | SimdTier::Sse42 => false,
        };
    let nn = !ta && !tb;
    let col_major = nn && a_col && b_col;
    if kernels && col_major && work <= r.simd_max as u128 {
        return GemmRoute::Small;
    }
    // The small transposed copy: every other product of such a tier in the
    // range, or with the op(A) rule only those with A transposed and both
    // operands stored column-major (TN and TT of column-major operands).
    if let Some((lo, hi)) = r.copy {
        let in_range = lo as u128 <= work && work <= hi as u128;
        let measured_class = ta && a_col && b_col;
        if kernels && !col_major && in_range && (!r.a_only || measured_class) {
            return GemmRoute::SmallTransCopy;
        }
    }
    // 0.2.3's routes.
    if work <= 32_768 {
        if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        }
    } else {
        GemmRoute::Blocked
    }
}

/// The router in its plain form (its helpers inlined): the boolean thin
/// switch first, then the SIMD small route, the copy, the small threshold.
fn plain_route(
    inp: &RouteInputs,
    thin: bool,
    small_simd_x86: bool,
    small_simd_max_work: usize,
    small_trans_copy: Option<(usize, usize)>,
) -> GemmRoute {
    let thin_ok = {
        let large_ok = if inp.n == 1 {
            inp.a_unit_col_stride || inp.a_unit_row_stride
        } else {
            inp.b_unit_col_stride || inp.b_unit_row_stride
        };
        inp.simd_real && (inp.m == 1 || inp.n == 1) && large_ok
    };
    if thin && thin_ok {
        return GemmRoute::Thin;
    }
    let work = inp.m.saturating_mul(inp.n).saturating_mul(inp.k);
    let nn = inp.trans_a == Trans::NoTrans && inp.trans_b == Trans::NoTrans;
    let col_major = nn && inp.a_unit_col_stride && inp.b_unit_col_stride;
    let simd_small = inp.simd_real
        && match inp.tier {
            SimdTier::Neon => inp.m >= 4 && inp.n >= 4,
            SimdTier::Avx2Fma => small_simd_x86 && inp.m >= 5 && inp.n >= 4,
            SimdTier::Sse42 | SimdTier::Portable => false,
        };
    if simd_small && col_major && work <= small_simd_max_work {
        return GemmRoute::Small;
    }
    let copy = matches!(small_trans_copy, Some((lo, hi)) if lo <= work && work <= hi);
    if simd_small && !col_major && copy {
        return GemmRoute::SmallTransCopy;
    }
    if work <= 32_768 {
        return if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        };
    }
    GemmRoute::Blocked
}

const TRANS: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];

/// `(simd_real, tier)`: f64/f32 and another element type on every tier.
const KINDS: [(bool, SimdTier); 8] = [
    (true, SimdTier::Portable),
    (true, SimdTier::Sse42),
    (true, SimdTier::Neon),
    (true, SimdTier::Avx2Fma),
    (false, SimdTier::Portable),
    (false, SimdTier::Sse42),
    (false, SimdTier::Neon),
    (false, SimdTier::Avx2Fma),
];

/// Every input of one shape: 9 transpose pairs, 16 stride flag sets, 8 kinds.
fn for_inputs((m, k, n): (usize, usize, usize), mut check: impl FnMut(&RouteInputs)) {
    for trans_a in TRANS {
        for trans_b in TRANS {
            for flags in 0u8..16 {
                for (simd_real, tier) in KINDS {
                    check(&RouteInputs {
                        m,
                        n,
                        k,
                        trans_a,
                        trans_b,
                        a_unit_col_stride: flags & 1 != 0,
                        a_unit_row_stride: flags & 2 != 0,
                        b_unit_col_stride: flags & 4 != 0,
                        b_unit_row_stride: flags & 8 != 0,
                        simd_real,
                        tier,
                    });
                }
            }
        }
    }
}

/// The sizes next to the limits of a thin product's other two sizes.
fn thin_sizes() -> Vec<usize> {
    let mut v: Vec<usize> = (1..=40).collect();
    let mut p = 64usize;
    while p <= 131_072 {
        v.extend([p - 1, p, p + 1]);
        p *= 2;
    }
    // Neighbours of the limits: 1024 / 32³ divided by small sizes.
    v.extend([
        255, 257, 341, 342, 511, 512, 513, 1023, 1025, 2047, 2048, 2049, 2184, 2185, 8191, 8192,
        8193, 10_922, 10_923, 16_383, 16_384, 16_385, 32_767, 32_768, 32_769,
    ]);
    v.sort_unstable();
    v.dedup();
    v
}

/// The thin shapes: `(1, k, n)` and `(m, k, 1)` over [`thin_sizes`].
fn thin_shapes() -> Vec<(usize, usize, usize)> {
    let sizes = thin_sizes();
    let mut shapes = Vec::new();
    for &x in &sizes {
        for &k in &sizes {
            shapes.push((1, k, x));
            if x != 1 {
                shapes.push((x, k, 1));
            }
        }
    }
    shapes
}

/// Every ordered factorisation `(m, k, n)` of the work values next to a
/// limit (the tiny shortcut, the floor of the scope, the adopted copy's
/// lower end, the small threshold).
fn boundary_shapes() -> Vec<(usize, usize, usize)> {
    factorisations(&[
        511, 512, 513, 1023, 1024, 1025, 1727, 1728, 1729, 32_767, 32_768, 32_769,
    ])
}

/// Every ordered factorisation `(m, k, n)` of the work values next to the
/// x86-64 candidate's ceiling of the SIMD small route (80³ = 512000).
fn ceiling_shapes() -> Vec<(usize, usize, usize)> {
    factorisations(&[511_999, 512_000, 512_001])
}

/// Every ordered factorisation `(m, k, n)` of each value of `works`.
fn factorisations(works: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut shapes = Vec::new();
    for &w in works {
        for m in (1..=w).filter(|d| w % d == 0) {
            let rest = w / m;
            for k in (1..=rest).filter(|d| rest % d == 0) {
                shapes.push((m, k, rest / k));
            }
        }
    }
    shapes
}

/// The dense cube `m, n, k in 0..=40` with `m` in `ms`.
fn cube(ms: core::ops::RangeInclusive<usize>) -> Vec<(usize, usize, usize)> {
    let mut shapes = Vec::new();
    for m in ms {
        for k in 0..=40 {
            for n in 0..=40 {
                shapes.push((m, k, n));
            }
        }
    }
    shapes
}

/// Checks the shipped policies on `shapes`: each against the stated rule,
/// the legacy policy and the policies without the scope also against the
/// plain router. Returns the number of routes by kind under the adopted
/// policy.
fn check_shipped(shapes: &[(usize, usize, usize)]) -> [usize; 5] {
    let all = GemmPolicy {
        thin: ThinPolicy::All,
        ..LEGACY_POLICY
    };
    let mut adopted_routes = [0usize; 5];
    for &shape in shapes {
        for_inputs(shape, |inp| {
            let legacy = choose_route(inp, &LEGACY_POLICY);
            assert_eq!(legacy, stated_route(inp, RULES_LEGACY), "legacy {inp:?}");
            assert_eq!(
                legacy,
                plain_route(inp, false, false, 32_768, None),
                "legacy vs the plain router {inp:?}"
            );
            let adopted = choose_route(inp, &ADOPTED_AARCH64);
            assert_eq!(adopted, stated_route(inp, RULES_ADOPTED), "adopted {inp:?}");
            adopted_routes[adopted as usize] += 1;
            assert_eq!(
                choose_route(inp, &CANDIDATE_AARCH64),
                adopted,
                "aarch64 candidate {inp:?}"
            );
            assert_eq!(
                choose_route(inp, &CANDIDATE_X86),
                stated_route(inp, RULES_CANDIDATE_X86),
                "x86 candidate {inp:?}"
            );
            let whole = choose_route(inp, &all);
            assert_eq!(
                whole,
                stated_route(
                    inp,
                    Rules {
                        thin: ThinPolicy::All,
                        ..RULES_LEGACY
                    }
                ),
                "legacy + All {inp:?}"
            );
            assert_eq!(
                whole,
                plain_route(inp, true, false, 32_768, None),
                "legacy + All vs the plain router {inp:?}"
            );
        });
    }
    adopted_routes
}

/// Synthetic policies: every copy range start next to the tiny shortcut and
/// the adopted lower end, with and without the AVX2 small kernels and their
/// higher ceiling, under every thin policy and with and without the `op(A)`
/// rule; against the stated rule, and (without the scope and the `op(A)`
/// rule) against the plain router.
fn check_synthetic(shapes: &[(usize, usize, usize)]) {
    let copies = [
        None,
        Some((1, 32_768)),
        Some((64, 32_768)),
        Some((511, 32_768)),
        Some((512, 32_768)),
        Some((513, 32_768)),
        Some((1728, 32_768)),
        Some((1728, 1728)),
    ];
    let mut rules = Vec::new();
    for copy in copies {
        for x86_small in [false, true] {
            for simd_max in [32_768usize, 512_000] {
                for thin in [ThinPolicy::Off, ThinPolicy::Scoped, ThinPolicy::All] {
                    for a_only in [false, true] {
                        rules.push(Rules {
                            thin,
                            x86_small,
                            simd_max,
                            copy,
                            a_only,
                        });
                    }
                }
            }
        }
    }
    let policies: Vec<(Rules, GemmPolicy)> = rules.iter().map(|&r| (r, policy_of(r))).collect();
    for &shape in shapes {
        for_inputs(shape, |inp| {
            for (r, policy) in &policies {
                let route = choose_route(inp, policy);
                assert_eq!(route, stated_route(inp, *r), "{r:?} {inp:?}");
                if r.thin != ThinPolicy::Scoped && !r.a_only {
                    let plain = plain_route(
                        inp,
                        r.thin == ThinPolicy::All,
                        r.x86_small,
                        r.simd_max,
                        r.copy,
                    );
                    assert_eq!(route, plain, "the plain router {r:?} {inp:?}");
                }
            }
        });
    }
}

#[test]
fn test_gemm_plan_route_rules_policy_fields() {
    // The shipped policies have the route fields of the stated rules.
    assert_eq!(rules_of(&LEGACY_POLICY), RULES_LEGACY);
    assert_eq!(rules_of(&ADOPTED_AARCH64), RULES_ADOPTED);
    assert_eq!(rules_of(&CANDIDATE_AARCH64), RULES_ADOPTED);
    assert_eq!(rules_of(&CANDIDATE_X86), RULES_CANDIDATE_X86);
    // The tiny shortcut is 8³.
    assert_eq!(ROUTE_TINY_WORK, 512);
}

#[test]
fn test_gemm_plan_route_rules_cube_low() {
    // Up to 13³ every route but the blocked driver is reached.
    let routes = check_shipped(&cube(0..=13));
    assert_eq!(routes[GemmRoute::Blocked as usize], 0, "{routes:?}");
    let reached = routes.iter().filter(|&&c| c > 0).count();
    assert_eq!(reached, 4, "{routes:?}");
}

#[test]
fn test_gemm_plan_route_rules_cube_mid() {
    check_shipped(&cube(14..=27));
}

#[test]
fn test_gemm_plan_route_rules_cube_high() {
    check_shipped(&cube(28..=40));
}

#[test]
fn test_gemm_plan_route_rules_thin_shapes() {
    let routes = check_shipped(&thin_shapes());
    // Thin and the loops and the blocked driver are reached.
    for route in [GemmRoute::Thin, GemmRoute::Small, GemmRoute::SmallTrans] {
        assert!(routes[route as usize] > 0, "{route:?} {routes:?}");
    }
    assert!(routes[GemmRoute::Blocked as usize] > 0, "{routes:?}");
}

#[test]
fn test_gemm_plan_route_rules_boundary_shapes() {
    let shapes = boundary_shapes();
    let routes = check_shipped(&shapes);
    assert!(routes.iter().all(|&c| c > 0), "{routes:?}");
    check_synthetic(&shapes);
}

#[test]
fn test_gemm_plan_route_rules_ceiling_shapes() {
    let shapes = ceiling_shapes();
    let routes = check_shipped(&shapes);
    // The x86-64 candidate's small route on both sides of 80³.
    let at = |m, k, n| RouteInputs {
        m,
        n,
        k,
        trans_a: Trans::NoTrans,
        trans_b: Trans::NoTrans,
        a_unit_col_stride: true,
        a_unit_row_stride: false,
        b_unit_col_stride: true,
        b_unit_row_stride: false,
        simd_real: true,
        tier: SimdTier::Avx2Fma,
    };
    assert_eq!(
        choose_route(&at(80, 80, 80), &CANDIDATE_X86),
        GemmRoute::Small
    );
    assert_eq!(
        choose_route(&at(80, 80, 80), &LEGACY_POLICY),
        GemmRoute::Blocked
    );
    // 512001 = 3⁵·7²·43: (21, 1, 24381) is just above the ceiling.
    assert_eq!(
        choose_route(&at(21, 1, 24_381), &CANDIDATE_X86),
        GemmRoute::Blocked
    );
    assert_eq!(
        choose_route(&at(20, 1, 25_600), &CANDIDATE_X86),
        GemmRoute::Small
    );
    assert!(routes[GemmRoute::Thin as usize] > 0, "{routes:?}");
    check_synthetic(&shapes);
}

#[test]
fn test_gemm_plan_route_rules_synthetic_cube() {
    // The synthetic policies on the cube up to 12 (the tiny shortcut's
    // range and the adopted copy's lower end) and on thin shapes around the
    // floor of the scope.
    let mut shapes = Vec::new();
    for m in 0..=12 {
        for k in 0..=12 {
            for n in 0..=12 {
                shapes.push((m, k, n));
            }
        }
    }
    for x in [1usize, 2, 3, 4, 5, 64, 128, 256, 512, 1024] {
        for k in [1usize, 2, 3, 4, 5, 15, 16, 17, 255, 256, 257, 511, 512, 513] {
            shapes.push((1, k, x));
            shapes.push((x, k, 1));
        }
    }
    check_synthetic(&shapes);
}

/// The adopted policy is active only in aarch64 builds without
/// `force-scalar` (checked at compile time, so a compile-only build for
/// x86-64 fails if it were active there).
const fn is_legacy_routes(p: &GemmPolicy) -> bool {
    matches!(p.thin, ThinPolicy::Off)
        && !p.small_simd_x86
        && p.small_simd_max_work == LEGACY_POLICY.small_simd_max_work
        && p.small_trans_copy.is_none()
        && !p.small_trans_copy_a_only
}

const _: () = assert!(
    cfg!(all(target_arch = "aarch64", not(feature = "force-scalar")))
        || is_legacy_routes(&ADOPTED_POLICY)
);
const _: () = assert!(
    cfg!(all(target_arch = "aarch64", not(feature = "force-scalar")))
        || cfg!(oxiblas_gemm_candidate)
        || is_legacy_routes(&DEFAULT_POLICY)
);
const _: () = assert!(!cfg!(oxiblas_gemm_legacy) || is_legacy_routes(&DEFAULT_POLICY));
