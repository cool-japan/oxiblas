//! The routes of public GEMM calls under this build's default policy.
//!
//! A public call (`gemm` or `gemm_transposed`) must take the route the
//! model of the build's rule says (the route a plan chose is recorded by
//! `call_probe::last_route`), compute the product within a stated error
//! bound of a triple-loop reference, and give exactly the bits of its
//! route's contract: the thin route those of its contract on the call's
//! tier (the oracle of `gemm_thin`'s tests; on the NEON tier also those of
//! `level2::gemv` on the column-major large operand), the small transposed
//! copy those of the `NoTrans`/`NoTrans`
//! small kernels of the policy on column-major copies of `op(A)` and
//! `op(B)` (`gemm_small_trans_copy`), and every other route those of
//! [`LEGACY_POLICY`]. The thin and copy cases
//! are chosen so that their bits differ from the legacy route's, so a call
//! that took the legacy route instead fails here. The routes the model
//! predicts follow the tier of the call (`SimdTier::current()`), so the
//! tests hold in every build, also one whose tier is not the one its
//! policy was chosen for.

use super::call_probe::{last_route, snapshot};
use super::route_tests::{Layout, Rule, model_route, simd_thin_tier};
use super::*;
use crate::level2::{GemvTrans, gemv};
use crate::level3::gemm::{BlockingSource, gemm, gemm_planned, gemm_transposed, read_op};
use crate::level3::gemm_small::gemm_small_with;
use crate::level3::gemm_thin::tests::{contract_f32, contract_f64};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::{Mat, MatRef};

/// `f64` / `f32` for these tests.
trait Real: GemmKernel + bytemuck::Zeroable + core::fmt::Debug {
    const EPS: f64;
    fn of(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
    /// The thin route's contract on `tier` (`gemm_thin`'s test oracle).
    #[allow(clippy::too_many_arguments)]
    fn thin_oracle(
        tier: SimdTier,
        trans: (Trans, Trans),
        alpha: Self,
        a: &MatRef<'_, Self>,
        b: &MatRef<'_, Self>,
        beta: Self,
        c0: &Mat<Self>,
    ) -> Mat<Self>;
}

impl Real for f64 {
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
    fn thin_oracle(
        tier: SimdTier,
        trans: (Trans, Trans),
        alpha: Self,
        a: &MatRef<'_, Self>,
        b: &MatRef<'_, Self>,
        beta: Self,
        c0: &Mat<Self>,
    ) -> Mat<Self> {
        contract_f64(tier, trans, alpha, a, b, beta, c0)
    }
}

impl Real for f32 {
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
    fn thin_oracle(
        tier: SimdTier,
        trans: (Trans, Trans),
        alpha: Self,
        a: &MatRef<'_, Self>,
        b: &MatRef<'_, Self>,
        beta: Self,
        c0: &Mat<Self>,
    ) -> Mat<Self> {
        contract_f32(tier, trans, alpha, a, b, beta, c0)
    }
}

/// The rule of this build's default policy, restated from the policy docs
/// (not read from `DEFAULT_POLICY`): the legacy cfg wins, then the
/// candidate cfg; without either, aarch64 without `force-scalar` has the
/// adopted rule and every other build 0.2.3's.
fn rule_of_build() -> Rule {
    let aarch64 = cfg!(target_arch = "aarch64");
    if cfg!(oxiblas_gemm_legacy) {
        Rule::V0_2_3
    } else if cfg!(oxiblas_gemm_candidate) {
        if cfg!(target_arch = "x86_64") {
            Rule::CandidateX86
        } else if aarch64 {
            Rule::AdoptedAarch64
        } else {
            Rule::V0_2_3
        }
    } else if aarch64 && !cfg!(feature = "force-scalar") {
        Rule::AdoptedAarch64
    } else {
        Rule::V0_2_3
    }
}

/// The policy of `rule` (the copy's small kernels depend on it: the AVX2 +
/// FMA tier has them only with `small_simd_x86`).
fn policy_of(rule: Rule) -> GemmPolicy {
    match rule {
        Rule::V0_2_3 => LEGACY_POLICY,
        Rule::AdoptedAarch64 => ADOPTED_AARCH64,
        Rule::CandidateX86 => CANDIDATE_X86,
    }
}

/// A `rows × cols` column-major matrix of values that round in sums.
fn values<T: Real>(rows: usize, cols: usize, seed: usize) -> Mat<T> {
    let mut m = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            let v = ((i * 7 + j * 13 + seed * 5) % 23) as f64 / 7.0 - 1.1;
            m[(i, j)] = T::of(v);
        }
    }
    m
}

/// The stored operand of `op(X)` of `rows × cols` (`trans`) in `layout`
/// (`Col`: as stored; `Row`: a `transpose()` view of column-major storage):
/// returns the storage and whether the operand is a view of its transpose.
fn operand<T: Real>(
    (rows, cols): (usize, usize),
    trans: Trans,
    layout: Layout,
    seed: usize,
) -> (Mat<T>, bool) {
    let (sr, sc) = if trans == Trans::NoTrans {
        (rows, cols)
    } else {
        (cols, rows)
    };
    match layout {
        Layout::Row => (values::<T>(sc, sr, seed), true),
        Layout::Col | Layout::Neither | Layout::Both => (values::<T>(sr, sc, seed), false),
    }
}

/// The view of an operand made by [`operand`].
fn view<T: Real>((storage, transposed): &(Mat<T>, bool)) -> MatRef<'_, T> {
    if *transposed {
        storage.as_ref().transpose()
    } else {
        storage.as_ref()
    }
}

/// `C = alpha·op(A)·op(B) + beta·C0` through a plan of `policy`.
#[allow(clippy::too_many_arguments)]
fn planned<T: Real>(
    policy: GemmPolicy,
    (ta, tb): (Trans, Trans),
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c0: &Mat<T>,
    (m, k, n): (usize, usize, usize),
) -> Mat<T> {
    let mut c = c0.clone();
    let plan = GemmPlan::<T>::for_problem(policy, m, n, k);
    gemm_planned(
        ta,
        tb,
        alpha,
        a,
        b,
        beta,
        c.as_mut(),
        Par::Seq,
        BlockingSource::Default,
        &plan,
    );
    c
}

/// The thin route's contract: one `level2::gemv` on the column-major
/// storage of the large operand (A for `n == 1`, B for `m == 1`).
#[allow(clippy::too_many_arguments)]
fn thin_contract<T: Real>(
    (ta, tb): (Trans, Trans),
    alpha: T,
    a: &(Mat<T>, bool),
    b: &(Mat<T>, bool),
    beta: T,
    c0: &Mat<T>,
    (m, k, n): (usize, usize, usize),
) -> Mat<T> {
    let (av, bv) = (view(a), view(b));
    let mut c = c0.clone();
    let flag = |t: bool| {
        if t {
            GemvTrans::Trans
        } else {
            GemvTrans::NoTrans
        }
    };
    if n == 1 {
        // op(A) = S^(view xor ta): the dot form where that is a transpose.
        let x: Vec<T> = (0..k).map(|p| read_op(&bv, tb, p, 0)).collect();
        let mut y: Vec<T> = (0..m).map(|i| c0[(i, 0)]).collect();
        let gt = flag(a.1 ^ (ta != Trans::NoTrans));
        gemv(gt, alpha, a.0.as_ref(), &x, beta, &mut y);
        for (i, yi) in y.into_iter().enumerate() {
            c[(i, 0)] = yi;
        }
    } else {
        // C[0, :]ᵀ = op(B)ᵀ·op(A)[0, :]ᵀ with op(B)ᵀ = S^(1 xor view xor tb).
        let x: Vec<T> = (0..k).map(|p| read_op(&av, ta, 0, p)).collect();
        let mut y: Vec<T> = (0..n).map(|j| c0[(0, j)]).collect();
        let gt = flag(!(b.1 ^ (tb != Trans::NoTrans)));
        gemv(gt, alpha, b.0.as_ref(), &x, beta, &mut y);
        for (j, yj) in y.into_iter().enumerate() {
            c[(0, j)] = yj;
        }
    }
    c
}

/// The copy's contract: the `NoTrans`/`NoTrans` small path of this tier and
/// `policy` on column-major copies of `op(A)` and `op(B)`.
#[allow(clippy::too_many_arguments)]
fn copy_contract<T: Real>(
    policy: GemmPolicy,
    (ta, tb): (Trans, Trans),
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c0: &Mat<T>,
    (m, k, n): (usize, usize, usize),
) -> Mat<T> {
    let mut opa = Mat::<T>::zeros(m, k);
    let mut opb = Mat::<T>::zeros(k, n);
    for p in 0..k {
        for i in 0..m {
            opa[(i, p)] = read_op(&a, ta, i, p);
        }
        for j in 0..n {
            opb[(p, j)] = read_op(&b, tb, p, j);
        }
    }
    let mut c = c0.clone();
    let ctx = GemmContext::detect(policy);
    gemm_small_with(
        ctx,
        alpha,
        &opa.as_ref(),
        &opb.as_ref(),
        beta,
        &mut c.as_mut(),
    );
    c
}

/// Whether two results have the same bits everywhere.
fn same_bits<T: Real>(x: &Mat<T>, y: &Mat<T>) -> bool {
    (0..x.ncols()).all(|j| (0..x.nrows()).all(|i| x[(i, j)].bits() == y[(i, j)].bits()))
}

/// One public call of `shape`: its route against the model of the build's
/// rule, its value against a triple-loop reference, its bits against its
/// route's contract. Returns the route and whether its bits differ from
/// the legacy route's.
fn check_public<T: Real>(
    (m, k, n): (usize, usize, usize),
    (ta, tb): (Trans, Trans),
    (la, lb): (Layout, Layout),
) -> (GemmRoute, bool) {
    let shape = (m, k, n);
    let ctx = format!(
        "{} {shape:?} {ta:?}/{tb:?} {la:?}/{lb:?}",
        core::any::type_name::<T>()
    );
    let a = operand::<T>((m, k), ta, la, 1 + m + 3 * k);
    let b = operand::<T>((k, n), tb, lb, 2 + n + 5 * k);
    let (av, bv) = (view(&a), view(&b));
    let (alpha, beta) = (T::of(-0.75), T::of(0.5));
    let c0 = values::<T>(m, n, 7 + k);
    // The route the model of this build's rule gives these operands, and
    // the layout flags the test means.
    let inp = RouteInputs::new(ta, tb, &av, &bv, m, n, k, SimdTier::current());
    assert_eq!(inp.a_unit_col_stride, la.units().0, "{ctx}: A layout");
    assert_eq!(inp.b_unit_col_stride, lb.units().0, "{ctx}: B layout");
    let rule = rule_of_build();
    let expected = model_route(&inp, rule);

    let mut c = c0.clone();
    let before = snapshot();
    if (ta, tb) == (Trans::NoTrans, Trans::NoTrans) {
        gemm(alpha, av, bv, beta, c.as_mut());
    } else {
        gemm_transposed(ta, tb, alpha, av, bv, beta, c.as_mut());
    }
    let counts = snapshot().since(before);
    let route = last_route();
    assert_eq!(route, Some(expected), "{ctx}");

    // The value: within 2·(k + 2)·eps of the magnitudes.
    for j in 0..n {
        for i in 0..m {
            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
            for p in 0..k {
                let prod = read_op(&av, ta, i, p).to_f64() * read_op(&bv, tb, p, j).to_f64();
                exact += prod;
                magnitude += prod.abs();
            }
            let reference = -0.75 * exact + 0.5 * c0[(i, j)].to_f64();
            let bound = 0.75 * magnitude + (0.5 * c0[(i, j)].to_f64()).abs();
            let tol = 2.0 * (k as f64 + 2.0) * T::EPS * bound + f64::MIN_POSITIVE;
            let got = c[(i, j)].to_f64();
            assert!(
                (got - reference).abs() <= tol,
                "{ctx}: ({i},{j}) {got} vs {reference} (tol {tol})"
            );
        }
    }

    // The bits: the route's contract.
    let legacy = planned(LEGACY_POLICY, (ta, tb), alpha, av, bv, beta, &c0, shape);
    let contract = match expected {
        GemmRoute::Thin => {
            // A thin call runs no micro-kernel (`gemm_thin` did not decline).
            assert_eq!(counts.kernel_resolutions, 0, "{ctx}: {counts:?}");
            // On the NEON tier the contract is also the bits of
            // `level2::gemv` (which runs the NEON GEMV kernels there).
            if SimdTier::current() == SimdTier::Neon {
                let gemv = thin_contract((ta, tb), alpha, &a, &b, beta, &c0, shape);
                assert!(same_bits(&c, &gemv), "{ctx}: bits of gemv");
            }
            let tier = SimdTier::current();
            T::thin_oracle(tier, (ta, tb), alpha, &av, &bv, beta, &c0)
        }
        GemmRoute::SmallTransCopy => {
            assert_eq!(counts.kernel_resolutions, 0, "{ctx}: {counts:?}");
            let policy = policy_of(rule);
            copy_contract(policy, (ta, tb), alpha, av, bv, beta, &c0, shape)
        }
        GemmRoute::Small | GemmRoute::SmallTrans | GemmRoute::Blocked => legacy.clone(),
    };
    assert!(same_bits(&c, &contract), "{ctx}: bits of {expected:?}");
    (expected, !same_bits(&c, &legacy))
}

/// One sample: `(m, k, n)`, the transposes and the layouts of A and B.
type Sample = ((usize, usize, usize), (Trans, Trans), (Layout, Layout));

/// The thin shapes: every row of the scope table (both sides of every
/// limit) whose operands can be built as column-major storage or as a
/// `transpose()` view of it.
fn thin_shapes() -> Vec<Sample> {
    super::route_tests::SCOPE_ROWS
        .iter()
        .filter(|row| {
            let ok = |l: Layout| matches!(l, Layout::Col | Layout::Row);
            ok(row.2.0) && ok(row.2.1)
        })
        .map(|row| (row.0, row.1, row.2))
        .collect()
}

/// The copy shapes: both ends of the adopted range, A transposed or not,
/// operands stored column-major or not, and the NEON kernels' least size.
const COPY_SHAPES: [Sample; 16] = {
    use Layout::{Col, Row};
    use Trans::{ConjTrans as C, NoTrans as N, Trans as T};
    [
        ((12, 12, 12), (T, N), (Col, Col)),
        ((11, 1, 157), (T, N), (Col, Col)),
        ((4, 1, 432), (T, N), (Col, Col)),
        ((32, 32, 32), (T, T), (Col, Col)),
        ((9, 11, 331), (T, N), (Col, Col)),
        ((16, 16, 16), (C, C), (Col, Col)),
        ((16, 16, 16), (N, N), (Row, Col)),
        ((16, 16, 16), (N, T), (Row, Col)),
        ((16, 16, 16), (N, T), (Col, Col)),
        ((16, 16, 16), (N, N), (Col, Row)),
        ((8, 8, 8), (T, N), (Col, Col)),
        ((3, 32, 32), (T, N), (Col, Col)),
        ((32, 32, 3), (T, T), (Col, Col)),
        ((24, 16, 24), (T, N), (Row, Row)),
        ((16, 16, 16), (T, N), (Col, Row)),
        ((16, 16, 16), (T, T), (Row, Col)),
    ]
};

/// Every sample shape for `T`: what it is, its route, whether its bits
/// differ from the legacy route's, and its depth `k`.
fn check_all<T: Real>() -> Vec<(String, GemmRoute, bool, usize)> {
    let mut seen = Vec::new();
    for (shape, trans, layouts) in thin_shapes().into_iter().chain(COPY_SHAPES) {
        let (route, differs) = check_public::<T>(shape, trans, layouts);
        let what = format!("{shape:?} {trans:?} {layouts:?}");
        seen.push((what, route, differs, shape.1));
    }
    seen
}

#[test]
fn test_gemm_plan_public_calls_take_the_default_routes() {
    let f64s = check_all::<f64>();
    let f32s = check_all::<f32>();
    let rule = rule_of_build();
    let mut new_routes = 0usize;
    for ((what, route, d64, k), (_, _, d32, _)) in f64s.iter().zip(&f32s) {
        // Every thin or copy case of depth 2 or more gives other bits than
        // the legacy route in at least one element type: a call that took
        // the legacy route would have failed above. (With `k == 1` every
        // element is one product, which both orders round alike; the route
        // record alone shows the route there.)
        if matches!(route, GemmRoute::Thin | GemmRoute::SmallTransCopy) {
            assert!(
                *d64 || *d32 || *k == 1,
                "{what}: {route:?} has the legacy bits"
            );
            new_routes += 1;
        } else {
            assert!(!d64 && !d32, "{what}: {route:?}");
        }
    }
    // The adopted rule takes the new routes on this sample on the tier it
    // was measured on; 0.2.3's never.
    let ok = |l: Layout| matches!(l, Layout::Col | Layout::Row);
    let thin_rows = super::route_tests::SCOPE_ROWS
        .iter()
        .filter(|row| row.3 && ok(row.2.0) && ok(row.2.1))
        .count();
    assert_eq!(thin_rows, 25);
    let tier = SimdTier::current();
    match rule {
        // The 25 thin rows of the scope table on a tier with SIMD thin
        // kernels, and 4 of the copy shapes (12³ TN, 4×1×432 TN, 32³ TT,
        // 16³ CC: A transposed, both operands column-major) on the NEON
        // tier. Another tier (the portable tier of `force-scalar`, also
        // when only oxiblas-core has it) takes neither.
        Rule::AdoptedAarch64 => {
            let thin = if simd_thin_tier(tier) { thin_rows } else { 0 };
            let copies = if tier == SimdTier::Neon { 4 } else { 0 };
            assert_eq!(new_routes, thin + copies, "{tier:?}");
        }
        Rule::CandidateX86 => {
            if simd_thin_tier(tier) {
                assert!(new_routes > 0, "{new_routes}");
            }
        }
        Rule::V0_2_3 => assert_eq!(new_routes, 0),
    }
}
