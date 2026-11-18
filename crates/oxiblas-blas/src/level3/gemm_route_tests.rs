//! The route rules of the policies, against independent models of them.
//!
//! The models below restate the routing rules in a few lines each (0.2.3's
//! rule, the adopted aarch64 rule, the candidates' rules); they do not call
//! the router's helpers (`thin_in_scope`, `thin_ok`, `simd_small`). The
//! route tables are pure, so every test here runs on every host.

use super::*;

/// The transposes of a real matrix (`ConjTrans` counts as `Trans`).
pub(crate) const ALL_TRANS: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];

/// Every tier.
pub(crate) const ALL_TIERS: [SimdTier; 4] = [
    SimdTier::Portable,
    SimdTier::Sse42,
    SimdTier::Neon,
    SimdTier::Avx2Fma,
];

/// How one stored operand is laid out: which of its strides are 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// Adjacent rows (`col_stride() == 1`), the other stride is not 1.
    Col,
    /// Adjacent columns (`row_stride() == 1`), e.g. a `transpose()` view.
    Row,
    /// Neither stride is 1 (a strided view).
    Neither,
    /// Both strides are 1 (a single row or column).
    Both,
}

pub(crate) const ALL_LAYOUTS: [Layout; 4] =
    [Layout::Col, Layout::Row, Layout::Neither, Layout::Both];

impl Layout {
    /// `(col_stride() == 1, row_stride() == 1)`.
    pub(super) const fn units(self) -> (bool, bool) {
        match self {
            Self::Col => (true, false),
            Self::Row => (false, true),
            Self::Neither => (false, false),
            Self::Both => (true, true),
        }
    }
}

/// The router inputs of `op(A)` of `m × k` by `op(B)` of `k × n`.
pub(crate) fn route_inputs(
    (m, k, n): (usize, usize, usize),
    (trans_a, trans_b): (Trans, Trans),
    (a, b): (Layout, Layout),
    simd_real: bool,
    tier: SimdTier,
) -> RouteInputs {
    let (a_col, a_row) = a.units();
    let (b_col, b_row) = b.units();
    RouteInputs {
        m,
        n,
        k,
        trans_a,
        trans_b,
        a_unit_col_stride: a_col,
        a_unit_row_stride: a_row,
        b_unit_col_stride: b_col,
        b_unit_row_stride: b_row,
        simd_real,
        tier,
    }
}

/// Which rule a model follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rule {
    /// 0.2.3 (`LEGACY_POLICY`).
    V0_2_3,
    /// The policy adopted on aarch64 (`ADOPTED_AARCH64`, and the aarch64
    /// candidate, whose extra field changes no route).
    AdoptedAarch64,
    /// The x86-64 candidate.
    CandidateX86,
}

/// The route of `inp` under `rule`, restated from the policy docs.
pub(crate) fn model_route(inp: &RouteInputs, rule: Rule) -> GemmRoute {
    let (m, n, k) = (inp.m as u128, inp.n as u128, inp.k as u128);
    let work = m * n * k;
    let a_t = inp.trans_a != Trans::NoTrans;
    let b_t = inp.trans_b != Trans::NoTrans;
    let nn = !a_t && !b_t;
    // 0.2.3: the small loops up to 32³, the blocked driver above.
    let legacy = if work <= 32_768 {
        if nn {
            GemmRoute::Small
        } else {
            GemmRoute::SmallTrans
        }
    } else {
        GemmRoute::Blocked
    };
    if rule == Rule::V0_2_3 {
        return legacy;
    }
    // The thin route: f64/f32, a row or a column of C, the large operand
    // (A for a column, also for m == n == 1; B for a row) stored with
    // adjacent rows or columns. In the scope: both operands stored
    // column-major, on the NEON or the AVX2 + FMA tier; above 32³ every
    // such product, from 1024 up to 32³ the four layouts of the scope table.
    let large_unit = if n == 1 {
        inp.a_unit_col_stride || inp.a_unit_row_stride
    } else {
        inp.b_unit_col_stride || inp.b_unit_row_stride
    };
    if inp.simd_real && (m == 1 || n == 1) && large_unit {
        let both_col = inp.a_unit_col_stride && inp.b_unit_col_stride;
        let vec_len = m.max(n);
        let small_scope = match (n == 1, a_t, b_t) {
            // Columns: NN (axpy) from 2 rows and k >= 2, TN (dots) from 2
            // dots of length 16.
            (true, false, false) => vec_len >= 2 && k >= 2,
            (true, true, false) => vec_len >= 2 && k >= 16,
            // Rows (n > 1): NN (dots) from 4 columns and k >= 2, NT (axpy)
            // from 4 columns and k >= 4.
            (false, false, false) => vec_len >= 4 && k >= 2,
            (false, false, true) => vec_len >= 4 && k >= 4,
            _ => false,
        };
        let simd_thin_tier = matches!(inp.tier, SimdTier::Neon | SimdTier::Avx2Fma);
        let in_scope =
            both_col && simd_thin_tier && (work > 32_768 || (work >= 1024 && small_scope));
        if in_scope {
            return GemmRoute::Thin;
        }
    }
    // The SIMD small kernels: NEON from 4 × 4; with the x86-64 candidate
    // also AVX2 + FMA from 5 × 4.
    let x86 = rule == Rule::CandidateX86;
    let tile = inp.simd_real
        && match inp.tier {
            SimdTier::Neon => m >= 4 && n >= 4,
            SimdTier::Avx2Fma => x86 && m >= 5 && n >= 4,
            SimdTier::Sse42 | SimdTier::Portable => false,
        };
    let col_major = nn && inp.a_unit_col_stride && inp.b_unit_col_stride;
    let ceiling = if x86 { 512_000 } else { 32_768 };
    if tile && col_major && work <= ceiling {
        return GemmRoute::Small;
    }
    // The small transposed copy: the adopted rule copies only a transposed
    // A with both operands stored column-major (TN and TT), from 12³; the
    // x86-64 candidate every other product from 8³.
    let a_trans_col_major = a_t && inp.a_unit_col_stride && inp.b_unit_col_stride;
    let copy = if x86 {
        (512..=32_768).contains(&work)
    } else {
        (1728..=32_768).contains(&work) && a_trans_col_major
    };
    if tile && !col_major && copy {
        return GemmRoute::SmallTransCopy;
    }
    legacy
}

/// The sizes of the exhaustive grids.
pub(crate) const GRID: [usize; 17] = [
    1, 2, 3, 4, 7, 8, 12, 16, 31, 32, 33, 64, 127, 128, 4096, 32_768, 65_536,
];

/// Every input of the exhaustive grid: `GRID³`, the nine transpose pairs,
/// every layout of both operands, every tier with `f64`/`f32` and the
/// portable tier with another element type.
fn for_grid(mut check: impl FnMut(&RouteInputs)) {
    let mut kinds: Vec<(bool, SimdTier)> = ALL_TIERS.iter().map(|&t| (true, t)).collect();
    kinds.push((false, SimdTier::Neon));
    for &m in &GRID {
        for &k in &GRID {
            for &n in &GRID {
                for ta in ALL_TRANS {
                    for tb in ALL_TRANS {
                        for la in ALL_LAYOUTS {
                            for lb in ALL_LAYOUTS {
                                for &(real, tier) in &kinds {
                                    let shape = (m, k, n);
                                    let inp = route_inputs(shape, (ta, tb), (la, lb), real, tier);
                                    check(&inp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The shapes the 0.2.3 test has always covered (small-kernel and ceiling
/// boundaries), next to the grid.
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

/// Every input over [`SHAPES`] (the same dimensions as the grid's).
fn for_shapes(mut check: impl FnMut(&RouteInputs)) {
    for shape in SHAPES {
        for ta in ALL_TRANS {
            for tb in ALL_TRANS {
                for la in ALL_LAYOUTS {
                    for lb in ALL_LAYOUTS {
                        for real in [true, false] {
                            for tier in ALL_TIERS {
                                check(&route_inputs(shape, (ta, tb), (la, lb), real, tier));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_plan_legacy_route_matches_0_2_3() {
    // The 0.2.3 rule, written out without the model: the small loops up to
    // 32³ (`gemm_small` for NoTrans/NoTrans, `gemm_small_trans` otherwise),
    // the blocked driver above, for every input.
    let route_0_2_3 = |inp: &RouteInputs| {
        if inp.m * inp.n * inp.k <= SMALL_THRESHOLD {
            if inp.both_notrans() {
                GemmRoute::Small
            } else {
                GemmRoute::SmallTrans
            }
        } else {
            GemmRoute::Blocked
        }
    };
    let mut count = 0usize;
    let mut check = |inp: &RouteInputs| {
        let route = choose_route(inp, &LEGACY_POLICY);
        assert_eq!(route, route_0_2_3(inp), "{inp:?}");
        assert_eq!(route, model_route(inp, Rule::V0_2_3), "{inp:?}");
        count += 1;
    };
    for_shapes(&mut check);
    for_grid(&mut check);
    assert_eq!(count, 22 * 9 * 16 * 2 * 4 + 17 * 17 * 17 * 9 * 16 * 5);
}

#[test]
fn test_gemm_plan_adopted_route_matches_model() {
    let mut routes = [0usize; 5];
    let mut check = |inp: &RouteInputs| {
        let route = choose_route(inp, &ADOPTED_AARCH64);
        assert_eq!(route, model_route(inp, Rule::AdoptedAarch64), "{inp:?}");
        // The aarch64 candidate adds only the B cap, which moves no route.
        assert_eq!(choose_route(inp, &CANDIDATE_AARCH64), route, "{inp:?}");
        routes[route as usize] += 1;
    };
    for_shapes(&mut check);
    for_grid(&mut check);
    // Every route is reached on the grid.
    assert!(routes.iter().all(|&c| c > 0), "{routes:?}");
}

#[test]
fn test_gemm_plan_candidate_route_table() {
    let mut check = |inp: &RouteInputs| {
        assert_eq!(
            choose_route(inp, &CANDIDATE_X86),
            model_route(inp, Rule::CandidateX86),
            "{inp:?}"
        );
    };
    for_shapes(&mut check);
    for_grid(&mut check);
    // The AVX2 small route of the x86 candidate, at its edges.
    let nn = (Trans::NoTrans, Trans::NoTrans);
    let cols = (Layout::Col, Layout::Col);
    let avx2 = |shape| route_inputs(shape, nn, cols, true, SimdTier::Avx2Fma);
    let x86 = &CANDIDATE_X86;
    assert_eq!(choose_route(&avx2((80, 80, 80)), x86), GemmRoute::Small);
    assert_eq!(choose_route(&avx2((81, 80, 80)), x86), GemmRoute::Blocked);
    assert_eq!(choose_route(&avx2((4, 4096, 4)), x86), GemmRoute::Blocked);
    // Transposed problems from 8³ are copied by the x86 candidate, whose
    // op(A) is read in place or not.
    for (ta, tb) in [
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
    ] {
        let at = |shape| route_inputs(shape, (ta, tb), cols, true, SimdTier::Avx2Fma);
        assert_eq!(choose_route(&at((8, 8, 8)), x86), GemmRoute::SmallTransCopy);
        assert_eq!(choose_route(&at((7, 8, 8)), x86), GemmRoute::SmallTrans);
    }
}

/// One row of the scope table test: `(m, k, n)`, the transposes, the
/// layouts of A and B, and whether the adopted policy takes the thin route
/// on a tier with SIMD thin kernels (NEON, AVX2 + FMA; on the other two
/// tiers no row is thin).
pub(crate) type ScopeRow = (
    (usize, usize, usize),
    (Trans, Trans),
    (Layout, Layout),
    bool,
);

const N: Trans = Trans::NoTrans;
const T: Trans = Trans::Trans;
const C: Trans = Trans::ConjTrans;
const COLS: (Layout, Layout) = (Layout::Col, Layout::Col);

/// The scope table of the adopted thin route, row by row, on both sides of
/// every limit (`THIN_MIN_WORK` = 1024, the least vector length and depth
/// of each layout, `SMALL_THRESHOLD` = 32³). The expected answers are
/// written out.
pub(crate) const SCOPE_ROWS: [ScopeRow; 86] = [
    // Above 32³: every thin product with both operands stored column-major,
    // every transpose pair. Just below, the same products keep 0.2.3's
    // route.
    ((64, 512, 1), (T, T), COLS, false),
    ((64, 513, 1), (T, T), COLS, true),
    ((1, 512, 64), (T, T), COLS, false),
    ((1, 513, 64), (T, T), COLS, true),
    ((64, 512, 1), (N, T), COLS, false),
    ((64, 513, 1), (N, T), COLS, true),
    ((1, 512, 64), (T, N), COLS, false),
    ((1, 513, 64), (T, N), COLS, true),
    ((64, 513, 1), (N, N), COLS, true),
    ((1, 513, 64), (N, N), COLS, true),
    // Above 32³ with an operand not stored column-major: the route of
    // 0.2.3 (the blocked driver). A large operand stored row-major (a
    // transposed view), in several transpose pairs, ...
    ((64, 512, 1), (N, N), (Layout::Row, Layout::Col), false),
    ((64, 513, 1), (N, N), (Layout::Row, Layout::Col), false),
    ((64, 513, 1), (T, N), (Layout::Row, Layout::Col), false),
    ((1, 512, 64), (N, N), (Layout::Col, Layout::Row), false),
    ((1, 513, 64), (N, N), (Layout::Col, Layout::Row), false),
    ((1, 513, 64), (N, T), (Layout::Col, Layout::Row), false),
    ((4096, 9, 1), (T, T), (Layout::Row, Layout::Col), false),
    ((1, 9, 4096), (C, C), (Layout::Col, Layout::Row), false),
    // ... a strided vector operand (the large operand column-major) ...
    ((64, 513, 1), (N, N), (Layout::Col, Layout::Neither), false),
    ((64, 513, 1), (N, N), (Layout::Col, Layout::Row), false),
    ((1, 513, 64), (N, N), (Layout::Neither, Layout::Col), false),
    ((1, 513, 64), (N, N), (Layout::Row, Layout::Col), false),
    // ... while an operand with both unit strides is column-major.
    ((64, 513, 1), (N, N), (Layout::Col, Layout::Both), true),
    ((1, 513, 64), (N, N), (Layout::Both, Layout::Col), true),
    ((1, 32_768, 1), (N, N), COLS, false),
    ((1, 32_769, 1), (N, N), COLS, true),
    ((1, 32_768, 1), (T, T), COLS, false),
    ((1, 32_769, 1), (T, T), COLS, true),
    ((32_768, 1, 1), (T, N), COLS, false),
    ((32_769, 1, 1), (T, N), COLS, true),
    ((1, 1, 32_768), (N, N), COLS, false),
    ((1, 1, 32_769), (N, N), COLS, true),
    ((1, 2, 16_384), (N, T), COLS, false),
    ((1, 3, 10_922), (N, T), COLS, false),
    ((1, 3, 10_923), (N, T), COLS, true),
    ((2184, 15, 1), (T, N), COLS, false),
    ((2185, 15, 1), (T, N), COLS, true),
    ((32_768, 1, 1), (N, N), COLS, false),
    ((32_769, 1, 1), (N, N), COLS, true),
    // A large operand without a unit stride is never thin.
    ((64, 513, 1), (N, N), (Layout::Neither, Layout::Col), false),
    ((1, 513, 64), (N, N), (Layout::Col, Layout::Neither), false),
    // n == 1, op(A) = A, op(B) = B, both column-major (THIN_COL_AXPY:
    // m >= 2, k >= 2, from 1024 multiply-adds).
    ((2, 511, 1), (N, N), COLS, false),
    ((2, 512, 1), (N, N), COLS, true),
    ((512, 2, 1), (N, N), COLS, true),
    ((1024, 1, 1), (N, N), COLS, false),
    ((2, 16_384, 1), (N, N), COLS, true),
    ((2, 512, 1), (N, N), (Layout::Row, Layout::Col), false),
    ((2, 512, 1), (N, N), (Layout::Col, Layout::Row), false),
    // n == 1, op(A) = Aᵀ (THIN_COL_DOT: m >= 2, k >= 16).
    ((64, 16, 1), (T, N), COLS, true),
    ((64, 15, 1), (T, N), COLS, false),
    ((128, 15, 1), (T, N), COLS, false),
    ((128, 16, 1), (T, N), COLS, true),
    ((128, 16, 1), (C, N), COLS, true),
    ((2, 512, 1), (T, N), COLS, true),
    ((2, 511, 1), (T, N), COLS, false),
    ((128, 16, 1), (T, N), (Layout::Row, Layout::Col), false),
    // m == 1, n > 1, op(A) = A, op(B) = B (THIN_ROW_DOT: n >= 4, k >= 2).
    ((1, 256, 4), (N, N), COLS, true),
    ((1, 255, 4), (N, N), COLS, false),
    ((1, 512, 3), (N, N), COLS, false),
    ((1, 512, 2), (N, N), COLS, false),
    ((1, 2, 512), (N, N), COLS, true),
    ((1, 1, 1024), (N, N), COLS, false),
    ((1, 256, 4), (N, N), (Layout::Row, Layout::Col), false),
    ((1, 256, 4), (N, N), (Layout::Col, Layout::Row), false),
    // m == 1, n > 1, op(B) = Bᵀ (THIN_ROW_AXPY: n >= 4, k >= 4).
    ((1, 256, 4), (N, T), COLS, true),
    ((1, 4, 256), (N, T), COLS, true),
    ((1, 4, 255), (N, T), COLS, false),
    ((1, 3, 512), (N, T), COLS, false),
    ((1, 512, 3), (N, T), COLS, false),
    ((1, 4, 256), (N, C), COLS, true),
    ((1, 4, 256), (N, T), (Layout::Col, Layout::Row), false),
    // Every other small thin product keeps 0.2.3's route: the vector
    // operand transposed (NT and TT columns, TN and TT rows), below 1024
    // multiply-adds, and the dot products (m == n == 1).
    ((64, 16, 1), (N, T), COLS, false),
    ((64, 16, 1), (T, T), COLS, false),
    ((64, 16, 1), (C, C), COLS, false),
    ((1, 16, 64), (T, N), COLS, false),
    ((1, 16, 64), (T, T), COLS, false),
    ((1024, 32, 1), (N, T), COLS, false),
    ((1, 32, 1024), (T, N), COLS, false),
    ((32, 31, 1), (N, N), COLS, false),
    ((1, 31, 32), (N, N), COLS, false),
    ((1, 1023, 1), (N, N), COLS, false),
    ((1, 1024, 1), (N, N), COLS, false),
    ((1, 1024, 1), (T, N), COLS, false),
    ((1, 1024, 1), (N, T), COLS, false),
    ((1, 2048, 1), (T, T), COLS, false),
    ((1, 32_768, 1), (C, C), COLS, false),
];

/// The tiers with SIMD thin kernels, the only ones on which a scoped
/// policy takes the thin route.
pub(crate) const fn simd_thin_tier(tier: SimdTier) -> bool {
    match tier {
        SimdTier::Neon | SimdTier::Avx2Fma => true,
        SimdTier::Portable | SimdTier::Sse42 => false,
    }
}

#[test]
fn test_gemm_plan_thin_scope_table() {
    let mut thin_rows = [0usize; 4];
    for (shape, trans, layouts, thin) in SCOPE_ROWS {
        for (t, tier) in ALL_TIERS.into_iter().enumerate() {
            let inp = route_inputs(shape, trans, layouts, true, tier);
            let ctx = format!("{shape:?} {trans:?} {layouts:?} {tier:?}");
            // Every scoped policy: thin exactly on the rows of the table
            // and on a tier with SIMD thin kernels.
            let want = thin && simd_thin_tier(tier);
            for (name, policy) in [
                ("adopted", &ADOPTED_AARCH64),
                ("aarch64 candidate", &CANDIDATE_AARCH64),
                ("x86-64 candidate", &CANDIDATE_X86),
            ] {
                assert_eq!(policy.thin, ThinPolicy::Scoped, "{name}");
                let route = choose_route(&inp, policy);
                assert_eq!(route == GemmRoute::Thin, want, "{name} {ctx}: {route:?}");
                // Off the thin route, the route of 0.2.3 (a thin product is
                // never a SIMD small or copy product).
                if !want {
                    assert_eq!(route, choose_route(&inp, &LEGACY_POLICY), "{name} {ctx}");
                }
            }
            thin_rows[t] += usize::from(want);
            // The scope itself (the router has checked the floor).
            let work = inp.work();
            if work >= THIN_MIN_WORK && inp.thin_ok() {
                assert_eq!(inp.thin_in_scope(work), want, "{ctx}");
            }
            // The whole thin route takes every thin product with a unit
            // stride; no policy without it ever does.
            let all = GemmPolicy {
                thin: ThinPolicy::All,
                ..LEGACY_POLICY
            };
            assert_eq!(
                choose_route(&inp, &all) == GemmRoute::Thin,
                inp.thin_ok(),
                "{ctx}"
            );
            assert_ne!(choose_route(&inp, &LEGACY_POLICY), GemmRoute::Thin);
            // Another element type never takes it.
            let other = route_inputs(shape, trans, layouts, false, tier);
            assert_ne!(choose_route(&other, &ADOPTED_AARCH64), GemmRoute::Thin);
        }
        for tier in ALL_TIERS {
            assert_eq!(
                model_route(
                    &route_inputs(shape, trans, layouts, true, tier),
                    Rule::AdoptedAarch64
                ) == GemmRoute::Thin,
                thin && simd_thin_tier(tier),
                "model {shape:?} {trans:?} {layouts:?} {tier:?}"
            );
        }
    }
    // The thin rows of the table on each tier (Portable, Sse42, Neon,
    // Avx2Fma).
    let rows = SCOPE_ROWS.iter().filter(|row| row.3).count();
    assert_eq!(rows, 27);
    assert_eq!(thin_rows, [0, 0, rows, rows]);
}

#[test]
fn test_gemm_plan_small_trans_copy_a_only_rule() {
    let dims = [3usize, 4, 5, 8, 11, 12, 16, 31, 32, 33, 64, 157, 432];
    let mut copies = 0usize;
    for &m in &dims {
        for &k in &[1usize, 2, 8, 11, 12, 16, 32, 33] {
            for &n in &dims {
                for ta in ALL_TRANS {
                    for tb in ALL_TRANS {
                        for la in ALL_LAYOUTS {
                            for lb in ALL_LAYOUTS {
                                let inp = route_inputs(
                                    (m, k, n),
                                    (ta, tb),
                                    (la, lb),
                                    true,
                                    SimdTier::Neon,
                                );
                                let route = choose_route(&inp, &ADOPTED_AARCH64);
                                let ctx = format!("{inp:?}");
                                let work = m * n * k;
                                // The measured class: A transposed, both
                                // operands stored column-major.
                                let class = ta != Trans::NoTrans && la.units().0 && lb.units().0;
                                if !class {
                                    // NoTrans A, or an operand not stored
                                    // column-major: never copied, the route
                                    // of 0.2.3.
                                    assert_ne!(route, GemmRoute::SmallTransCopy, "{ctx}");
                                    assert_eq!(route, choose_route(&inp, &LEGACY_POLICY), "{ctx}");
                                } else if m >= 4 && n >= 4 && (1728..=32_768).contains(&work) {
                                    assert_eq!(route, GemmRoute::SmallTransCopy, "{ctx}");
                                    copies += 1;
                                } else {
                                    assert_eq!(route, choose_route(&inp, &LEGACY_POLICY), "{ctx}");
                                }
                                // No other tier copies under this policy.
                                for tier in [SimdTier::Portable, SimdTier::Sse42, SimdTier::Avx2Fma]
                                {
                                    let other = RouteInputs { tier, ..inp };
                                    assert_eq!(
                                        choose_route(&other, &ADOPTED_AARCH64),
                                        choose_route(&other, &LEGACY_POLICY),
                                        "{ctx} {tier:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(copies > 0);
    // The range's two ends are exact (A transposed, both column-major).
    let tn = |shape| route_inputs(shape, (T, N), COLS, true, SimdTier::Neon);
    let route = |shape| choose_route(&tn(shape), &ADOPTED_AARCH64);
    assert_eq!(route((12, 12, 12)), GemmRoute::SmallTransCopy);
    assert_eq!(route((4, 1, 432)), GemmRoute::SmallTransCopy);
    assert_eq!(route((11, 1, 157)), GemmRoute::SmallTrans); // 1727
    assert_eq!(route((32, 32, 32)), GemmRoute::SmallTransCopy);
    assert_eq!(route((9, 11, 331)), GemmRoute::Blocked); // 32769
    // m or n below the NEON kernels' 4: the loops.
    assert_eq!(route((3, 32, 32)), GemmRoute::SmallTrans);
    assert_eq!(route((32, 32, 3)), GemmRoute::SmallTrans);
    // TN and TT of column-major operands are copied (an operand with both
    // unit strides is column-major); NT of column-major operands is not.
    let at = |trans, layouts| route_inputs((16, 16, 16), trans, layouts, true, SimdTier::Neon);
    let adopted = |trans, layouts| choose_route(&at(trans, layouts), &ADOPTED_AARCH64);
    let copied = GemmRoute::SmallTransCopy;
    assert_eq!(adopted((T, N), COLS), copied);
    assert_eq!(adopted((T, T), COLS), copied);
    assert_eq!(adopted((C, T), COLS), copied);
    assert_eq!(adopted((T, N), (Layout::Both, Layout::Col)), copied);
    assert_eq!(adopted((T, N), (Layout::Col, Layout::Both)), copied);
    assert_eq!(adopted((N, T), COLS), GemmRoute::SmallTrans);
    // An operand not stored column-major takes the route of 0.2.3: NoTrans
    // A without adjacent rows (NN: the small loops; NT: the transposed
    // loops), a transposed A stored row-major or strided, and TN / TT with
    // a strided B.
    let (row, neither, col) = (Layout::Row, Layout::Neither, Layout::Col);
    for (trans, layouts, legacy) in [
        ((N, N), (row, col), GemmRoute::Small),
        ((N, N), (neither, col), GemmRoute::Small),
        ((N, N), (col, row), GemmRoute::Small),
        ((N, T), (row, col), GemmRoute::SmallTrans),
        ((N, T), (neither, col), GemmRoute::SmallTrans),
        ((T, N), (row, col), GemmRoute::SmallTrans),
        ((T, T), (neither, col), GemmRoute::SmallTrans),
        ((T, N), (col, row), GemmRoute::SmallTrans),
        ((T, N), (col, neither), GemmRoute::SmallTrans),
        ((T, T), (col, row), GemmRoute::SmallTrans),
        ((C, C), (row, row), GemmRoute::SmallTrans),
    ] {
        let ctx = format!("{trans:?} {layouts:?}");
        assert_eq!(adopted(trans, layouts), legacy, "{ctx}");
        assert_eq!(
            choose_route(&at(trans, layouts), &LEGACY_POLICY),
            legacy,
            "{ctx}"
        );
        // The x86-64 candidate's copy of every product in its range (no
        // `op(A)` rule) is unchanged: it copies each of them on a tier with
        // SIMD small kernels.
        assert_eq!(
            choose_route(&at(trans, layouts), &CANDIDATE_X86),
            copied,
            "x86-64 candidate {ctx}"
        );
    }
    assert_eq!(choose_route(&at((N, T), COLS), &CANDIDATE_X86), copied);
}
