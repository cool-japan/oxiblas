//! Tests for the thin GEMM path: `m == 1` or `n == 1` computed as one GEMV
//! (the contract in the module docs of `gemm_thin`).

use super::*;
use crate::level3::gemm::{BlockingSource, gemm_planned, op_dims, read_op};
use crate::level3::gemm_plan::{
    CANDIDATE_POLICY, GemmPlan, GemmPolicy, GemmRoute, LEGACY_POLICY, RouteInputs, SimdTier,
    avx2_fma_tier_or_skip, choose_route,
};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

/// Element helpers for the checks.
trait Elem: ThinElem + GemmKernel + bytemuck::Zeroable + core::fmt::Display {
    /// Lanes of one NEON vector (the dot-form geometry).
    const LANES: usize;
    /// Unit roundoff.
    const EPS: f64;
    /// The type's name in kernel names (`f64` / `f32`).
    const NAME: &'static str;
    /// `self·b + c` with one rounding (`mul_add`), on every target.
    fn fused(self, b: Self, c: Self) -> Self;
    fn bits(self) -> u64;
    fn of(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn nan() -> Self;
}

impl Elem for f64 {
    const LANES: usize = 2;
    const EPS: f64 = f64::EPSILON;
    const NAME: &'static str = "f64";
    fn fused(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn of(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn nan() -> Self {
        f64::NAN
    }
}

impl Elem for f32 {
    const LANES: usize = 4;
    const EPS: f64 = f32::EPSILON as f64;
    const NAME: &'static str = "f32";
    fn fused(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn of(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn nan() -> Self {
        f32::NAN
    }
}

fn unit(seed: usize) -> f64 {
    ((seed.wrapping_mul(2_654_435_761) >> 7) % 2003) as f64 / 1001.5 - 1.0
}

fn fill<T: Elem>(rows: usize, cols: usize, seed: usize) -> Mat<T> {
    let mut out: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            out[(i, j)] = T::of(unit(i * 7_919 + j * 104_729 + seed * 1_299_709));
        }
    }
    out
}

/// The transpose of `m` as a new column-major matrix.
fn transposed<T: Elem>(m: &Mat<T>) -> Mat<T> {
    let mut out: Mat<T> = Mat::zeros(m.ncols(), m.nrows());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out[(j, i)] = m[(i, j)];
        }
    }
    out
}

/// A plan whose route is forced to `Thin`, on the portable kernels.
fn thin_plan<T: Elem>() -> GemmPlan<T> {
    let mut plan = GemmPlan::<T>::new(CANDIDATE_POLICY).with_forced_route(GemmRoute::Thin);
    plan.ctx = plan.ctx.with_portable_tier();
    plan
}

/// A plan whose route is forced to `Thin`, on the tier of this build and CPU
/// (NEON on aarch64, AVX2 + FMA on such x86-64 hosts).
fn native_plan<T: Elem>() -> GemmPlan<T> {
    let mut plan = GemmPlan::<T>::new(CANDIDATE_POLICY).with_forced_route(GemmRoute::Thin);
    plan.ctx = GemmContext::detect(CANDIDATE_POLICY);
    assert_eq!(plan.ctx.tier(), SimdTier::current());
    plan
}

/// How the oracle forms the multiply-adds of the contract's chains (every
/// one except `t_j = alpha·x[j]` and the final `y[j] + alpha·d_j` of the dot
/// form, which are a plain product and sum on every tier).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Madd {
    /// One rounding (`mul_add`).
    Fused,
    /// A rounded product, then a rounded sum.
    Split,
}

/// `a·b + c` the way `mode` says.
fn mac<T: Elem>(mode: Madd, a: T, b: T, c: T) -> T {
    match mode {
        Madd::Fused => a.fused(b, c),
        Madd::Split => a * b + c,
    }
}

/// The multiply-add of `tier`'s kernels, as the contract states it: fused on
/// the NEON and AVX2 + FMA tiers; for the portable kernels fused exactly when
/// the target has a native FMA.
fn madd_of(tier: SimdTier) -> Madd {
    let native_fma = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
    match tier {
        SimdTier::Neon | SimdTier::Avx2Fma => Madd::Fused,
        SimdTier::Sse42 | SimdTier::Portable if native_fma => Madd::Fused,
        SimdTier::Sse42 | SimdTier::Portable => Madd::Split,
    }
}

/// The dot product of the contract (NEON order with `T::LANES` lanes),
/// written out with indices: `pair(i)` is (column element, vector element).
fn dot_ref<T: Elem>(mode: Madd, len: usize, pair: impl Fn(usize) -> (T, T)) -> T {
    let l = T::LANES;
    let mut acc = vec![vec![T::zero(); l]; 4];
    let mut i = 0;
    while i + 4 * l <= len {
        for (q, accq) in acc.iter_mut().enumerate() {
            for (lane, a) in accq.iter_mut().enumerate() {
                let (cv, xv) = pair(i + q * l + lane);
                *a = mac(mode, cv, xv, *a);
            }
        }
        i += 4 * l;
    }
    if len - i >= 2 * l {
        for (q, accq) in acc.iter_mut().take(2).enumerate() {
            for (lane, a) in accq.iter_mut().enumerate() {
                let (cv, xv) = pair(i + q * l + lane);
                *a = mac(mode, cv, xv, *a);
            }
        }
        i += 2 * l;
    }
    if len - i >= l {
        for (lane, a) in acc[2].iter_mut().enumerate() {
            let (cv, xv) = pair(i + lane);
            *a = mac(mode, cv, xv, *a);
        }
        i += l;
    }
    let mut v: Vec<T> = (0..l)
        .map(|lane| (acc[0][lane] + acc[1][lane]) + (acc[2][lane] + acc[3][lane]))
        .collect();
    while v.len() > 1 {
        v = v.chunks(2).map(|p| p[0] + p[1]).collect();
    }
    let mut s = v[0];
    while i < len {
        let (cv, xv) = pair(i);
        s = mac(mode, cv, xv, s);
        i += 1;
    }
    s
}

/// The contract computed from its definition with plain indexing: the
/// result `C` (`m × n`) for the stored operands `a`, `b` and the initial `c0`,
/// with the multiply-adds formed as `mode` says.
#[allow(clippy::too_many_arguments)]
fn oracle<T: Elem>(
    mode: Madd,
    ta: Trans,
    tb: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c0: &Mat<T>,
) -> Mat<T> {
    let (m, k) = op_dims(a, ta);
    let n = op_dims(b, tb).1;
    let op_a = |i: usize, p: usize| read_op(a, ta, i, p);
    let op_b = |p: usize, j: usize| read_op(b, tb, p, j);
    let scaled = |v: T| {
        if beta == T::zero() {
            T::zero()
        } else if beta == T::one() {
            v
        } else {
            beta * v
        }
    };
    let mut out: Mat<T> = Mat::zeros(m, n);
    if n == 1 {
        // GEMV on A: axpy over the columns of op(A) when they are the
        // columns of the column-major storage, dot over its rows otherwise.
        let axpy = if a.col_stride() == 1 {
            ta == Trans::NoTrans
        } else {
            ta != Trans::NoTrans
        };
        let mut y: Vec<T> = (0..m).map(|i| scaled(c0[(i, 0)])).collect();
        if axpy {
            for p in 0..k {
                let t = alpha * op_b(p, 0);
                for (i, yi) in y.iter_mut().enumerate() {
                    *yi = mac(mode, op_a(i, p), t, *yi);
                }
            }
        } else {
            for (i, yi) in y.iter_mut().enumerate() {
                let d = dot_ref(mode, k, |p| (op_a(i, p), op_b(p, 0)));
                *yi = *yi + alpha * d;
            }
        }
        for (i, &yi) in y.iter().enumerate() {
            out[(i, 0)] = yi;
        }
    } else {
        // m == 1: GEMV on B, axpy over the rows of op(B) when they are the
        // columns of the column-major storage, dot over its columns otherwise.
        let axpy = if b.col_stride() == 1 {
            tb != Trans::NoTrans
        } else {
            tb == Trans::NoTrans
        };
        let mut y: Vec<T> = (0..n).map(|j| scaled(c0[(0, j)])).collect();
        if axpy {
            for p in 0..k {
                let t = alpha * op_a(0, p);
                for (j, yj) in y.iter_mut().enumerate() {
                    *yj = mac(mode, op_b(p, j), t, *yj);
                }
            }
        } else {
            for (j, yj) in y.iter_mut().enumerate() {
                let d = dot_ref(mode, k, |p| (op_b(p, j), op_a(0, p)));
                *yj = *yj + alpha * d;
            }
        }
        for (j, &yj) in y.iter().enumerate() {
            out[(0, j)] = yj;
        }
    }
    out
}

/// One layout of the two operands: as stored, or as `transpose()` views of a
/// stored transpose (unit row stride).
#[derive(Debug, Clone, Copy)]
struct Views {
    a: bool,
    b: bool,
}

const ALL_VIEWS: [Views; 4] = [
    Views { a: false, b: false },
    Views { a: true, b: false },
    Views { a: false, b: true },
    Views { a: true, b: true },
];

const ALL_TRANS: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];

/// Runs every thin case of `(m, k, n)` through `plan` (route forced to thin)
/// and checks the untouched NaN border of C, the bits against the oracle with
/// the multiply-add of the plan's tier, and the value against a float
/// reference.
fn check_shape<T: Elem>(plan: &GemmPlan<T>, (m, k, n): (usize, usize, usize)) {
    let mode = madd_of(plan.ctx.tier());
    for ta in ALL_TRANS {
        for tb in ALL_TRANS {
            let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
            let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
            let a_plain = fill::<T>(ar, ac, 1 + m + 3 * k);
            let b_plain = fill::<T>(br, bc, 2 + n + 5 * k);
            let a_t = transposed(&a_plain);
            let b_t = transposed(&b_plain);
            for views in ALL_VIEWS {
                let a = if views.a {
                    a_t.as_ref().transpose()
                } else {
                    a_plain.as_ref()
                };
                let b = if views.b {
                    b_t.as_ref().transpose()
                } else {
                    b_plain.as_ref()
                };
                let scalars = [1.0, -0.5]
                    .into_iter()
                    .flat_map(|alpha| [0.0, 1.0, 0.25].map(|beta| (alpha, beta)));
                for (alpha, beta) in scalars {
                    let (alpha, beta) = (T::of(alpha), T::of(beta));
                    let ctx = format!(
                        "{} {m}x{k}x{n} {ta:?}/{tb:?} {views:?} alpha={alpha} beta={beta}",
                        core::any::type_name::<T>()
                    );
                    let inputs = RouteInputs::new(ta, tb, &a, &b, m, n, k, SimdTier::Portable);
                    assert_eq!(plan.route(&inputs), GemmRoute::Thin, "{ctx}");
                    assert_eq!(
                        choose_route(&inputs, &CANDIDATE_POLICY),
                        if CANDIDATE_POLICY.thin {
                            GemmRoute::Thin
                        } else {
                            choose_route(&inputs, &LEGACY_POLICY)
                        },
                        "{ctx}"
                    );

                    // C inside a NaN-bordered store; NaN inside too when
                    // beta == 0 (it must not be read).
                    let mut big: Mat<T> = Mat::filled(m + 3, n + 2, T::nan());
                    let c0 = if beta == T::zero() {
                        Mat::filled(m, n, T::nan())
                    } else {
                        fill::<T>(m, n, 7 + k)
                    };
                    for j in 0..n {
                        for i in 0..m {
                            big[(i + 2, j + 1)] = c0[(i, j)];
                        }
                    }
                    let before = big.clone();
                    gemm_planned(
                        ta,
                        tb,
                        alpha,
                        a,
                        b,
                        beta,
                        big.as_mut().submatrix(2, 1, m, n),
                        Par::Seq,
                        BlockingSource::Default,
                        plan,
                    );
                    let expected = oracle(mode, ta, tb, alpha, &a, &b, beta, &c0);
                    for j in 0..n + 2 {
                        for i in 0..m + 3 {
                            let inside = (2..2 + m).contains(&i) && (1..1 + n).contains(&j);
                            if !inside {
                                assert_eq!(
                                    big[(i, j)].bits(),
                                    before[(i, j)].bits(),
                                    "{ctx}: border ({i},{j}) written"
                                );
                                continue;
                            }
                            let (ci, cj) = (i - 2, j - 1);
                            let got = big[(i, j)];
                            assert_eq!(
                                got.bits(),
                                expected[(ci, cj)].bits(),
                                "{ctx}: ({ci},{cj}) {got} vs contract {}",
                                expected[(ci, cj)]
                            );
                            // Float reference with a componentwise bound.
                            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
                            for p in 0..k {
                                let prod = read_op(&a, ta, ci, p).to_f64()
                                    * read_op(&b, tb, p, cj).to_f64();
                                exact += prod;
                                magnitude += prod.abs();
                            }
                            let (al, be) = (alpha.to_f64(), beta.to_f64());
                            let mut reference = al * exact;
                            let mut bound = al.abs() * magnitude;
                            if beta != T::zero() {
                                let c = c0[(ci, cj)].to_f64();
                                reference += be * c;
                                bound += (be * c).abs();
                            }
                            let tol = 2.0 * (k as f64 + 2.0) * T::EPS * bound + f64::MIN_POSITIVE;
                            assert!(
                                got.to_f64().is_finite() && (got.to_f64() - reference).abs() <= tol,
                                "{ctx}: ({ci},{cj}) {got} vs reference {reference} (tol {tol})"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Every shape of the reference test on the forced portable tier and on the
/// native tier (NEON, AVX2 + FMA, or portable again on other hosts). The
/// lengths cover every vector step and tail of every kernel: 37 = 32 + 4 + 1
/// rows for the f64 AVX2 axpy strips, 45 = 32 + 8 + 5 for f32; k = 17, 64 and
/// 1000 reach every partial-sum step of the dot form.
fn check_matches_reference<T: Elem>() {
    for plan in [thin_plan::<T>(), native_plan::<T>()] {
        for k in [1usize, 3, 17, 64] {
            for len in [2usize, 5, 37, 45] {
                check_shape::<T>(&plan, (1, k, len));
                check_shape::<T>(&plan, (len, k, 1));
            }
            check_shape::<T>(&plan, (1, k, 1));
        }
        for len in [1usize, 9] {
            check_shape::<T>(&plan, (1, 1000, len));
            check_shape::<T>(&plan, (len, 1000, 1));
        }
    }
}

#[test]
fn test_gemm_thin_matches_reference_f64() {
    check_matches_reference::<f64>();
}

#[test]
fn test_gemm_thin_matches_reference_f32() {
    check_matches_reference::<f32>();
}

#[test]
fn test_gemm_thin_nan_in_a_propagates_with_zero_b() {
    // No column is skipped for a zero x[j]: NaN * 0 and Inf * 0 are NaN, as
    // in the reference product (and in the blocked route).
    fn check<T: Elem>(poison: T) {
        let thin = thin_plan::<T>();
        let blocked = GemmPlan::<T>::new(LEGACY_POLICY).with_forced_route(GemmRoute::Blocked);
        for ta in [Trans::NoTrans, Trans::Trans] {
            for (m, k, n) in [(7usize, 9usize, 1usize), (1, 9, 6), (1, 9, 1)] {
                let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
                let mut a = fill::<T>(ar, ac, 3);
                let (i0, p0) = (m / 2, 4);
                if ta == Trans::NoTrans {
                    a[(i0, p0)] = poison;
                } else {
                    a[(p0, i0)] = poison;
                }
                let b: Mat<T> = Mat::zeros(k, n);
                let mut results = Vec::new();
                for plan in [&thin, &blocked] {
                    let mut c: Mat<T> = Mat::filled(m, n, T::nan());
                    gemm_planned(
                        ta,
                        Trans::NoTrans,
                        T::one(),
                        a.as_ref(),
                        b.as_ref(),
                        T::zero(),
                        c.as_mut(),
                        Par::Seq,
                        BlockingSource::Default,
                        plan,
                    );
                    results.push(c);
                }
                for (route, c) in ["thin", "blocked"].iter().zip(&results) {
                    for j in 0..n {
                        for i in 0..m {
                            // Row i0 of op(A) meets every column of B.
                            let expect_nan = i == i0;
                            assert_eq!(
                                c[(i, j)].to_f64().is_nan(),
                                expect_nan,
                                "{route} {ta:?} {m}x{k}x{n} ({i},{j}) = {}",
                                c[(i, j)]
                            );
                        }
                    }
                }
            }
        }
        // m == 1: the NaN sits in x = op(A)[0, :] and reaches every column.
        let mut a = fill::<T>(1, 9, 5);
        a[(0, 4)] = poison;
        let b: Mat<T> = Mat::zeros(9, 6);
        let mut c: Mat<T> = Mat::filled(1, 6, T::nan());
        gemm_planned(
            Trans::NoTrans,
            Trans::NoTrans,
            T::one(),
            a.as_ref(),
            b.as_ref(),
            T::zero(),
            c.as_mut(),
            Par::Seq,
            BlockingSource::Default,
            &thin,
        );
        for j in 0..6 {
            assert!(c[(0, j)].to_f64().is_nan(), "row x NaN, column {j}");
        }
    }
    check::<f64>(f64::NAN);
    check::<f64>(f64::INFINITY);
    check::<f32>(f32::NAN);
    check::<f32>(f32::NEG_INFINITY);
}

#[test]
fn test_gemm_thin_route_table() {
    let base = RouteInputs {
        m: 1,
        n: 4096,
        k: 4096,
        trans_a: Trans::NoTrans,
        trans_b: Trans::NoTrans,
        a_unit_col_stride: true,
        a_unit_row_stride: false,
        b_unit_col_stride: true,
        b_unit_row_stride: false,
        simd_real: true,
        tier: SimdTier::Neon,
    };
    let thin_policy = GemmPolicy {
        thin: true,
        ..LEGACY_POLICY
    };
    let tiers = [
        SimdTier::Portable,
        SimdTier::Sse42,
        SimdTier::Neon,
        SimdTier::Avx2Fma,
    ];
    // Row (m == 1), column (n == 1) and dot (m == n == 1) shapes, every
    // tier, both transposes, any size.
    for tier in tiers {
        for (m, k, n) in [
            (1usize, 4096usize, 4096usize),
            (4096, 4096, 1),
            (1, 1, 1),
            (1, 3, 2),
            (2, 3, 1),
        ] {
            for (trans_a, trans_b) in [
                (Trans::NoTrans, Trans::NoTrans),
                (Trans::Trans, Trans::ConjTrans),
            ] {
                let inp = RouteInputs {
                    m,
                    n,
                    k,
                    trans_a,
                    trans_b,
                    tier,
                    ..base
                };
                assert!(inp.thin_ok(), "{inp:?}");
                assert_eq!(choose_route(&inp, &thin_policy), GemmRoute::Thin, "{inp:?}");
                assert_ne!(
                    choose_route(&inp, &LEGACY_POLICY),
                    GemmRoute::Thin,
                    "{inp:?}"
                );
            }
        }
    }
    // The large operand decides: B for m == 1, A for n == 1 (and for
    // m == n == 1); a unit row stride (transposed view) is enough.
    let row = base;
    let col = RouteInputs {
        m: 4096,
        n: 1,
        ..base
    };
    let dot = RouteInputs { m: 1, n: 1, ..base };
    let no_b = |inp: RouteInputs| RouteInputs {
        b_unit_col_stride: false,
        b_unit_row_stride: false,
        ..inp
    };
    let no_a = |inp: RouteInputs| RouteInputs {
        a_unit_col_stride: false,
        a_unit_row_stride: false,
        ..inp
    };
    let b_view = |inp: RouteInputs| RouteInputs {
        b_unit_col_stride: false,
        b_unit_row_stride: true,
        ..inp
    };
    let a_view = |inp: RouteInputs| RouteInputs {
        a_unit_col_stride: false,
        a_unit_row_stride: true,
        ..inp
    };
    let table = [
        (row, true),
        (no_b(row), false),
        (no_a(row), true),
        (b_view(row), true),
        (col, true),
        (no_a(col), false),
        (no_b(col), true),
        (a_view(col), true),
        (dot, true),
        (no_a(dot), false),
        (no_b(dot), true),
        (a_view(dot), true),
        // Not f64/f32, or not thin.
        (
            RouteInputs {
                simd_real: false,
                ..row
            },
            false,
        ),
        (RouteInputs { m: 2, n: 2, ..row }, false),
    ];
    for (inp, thin) in table {
        assert_eq!(inp.thin_ok(), thin, "{inp:?}");
        assert_eq!(
            choose_route(&inp, &thin_policy) == GemmRoute::Thin,
            thin,
            "{inp:?}"
        );
        assert_eq!(GemmRoute::Thin.accepts(&inp), thin, "{inp:?}");
    }
    // A forced thin route on a problem it cannot take falls back to the
    // router.
    let forced = thin_plan::<f64>();
    let wide = RouteInputs { m: 2, n: 2, ..row };
    assert_eq!(forced.route(&wide), choose_route(&wide, &CANDIDATE_POLICY));
    assert_eq!(
        forced.route(&no_b(row)),
        choose_route(&no_b(row), &CANDIDATE_POLICY)
    );
}

/// Checks that `plan`'s thin path gives exactly the bits of `level2::gemv`
/// (the NEON GEMV kernels on aarch64) on the same column-major operands, for
/// every vector step and tail of both forms.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
fn check_plan_equals_gemv<T: Elem>(plan: &GemmPlan<T>) {
    use crate::level2::{GemvTrans, gemv};
    {
        let lens = [1usize, 2, 3, 5, 8, 9, 16, 17, 31, 33, 64, 100];
        let depths = [1usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 33, 100, 257];
        for &len in &lens {
            for &k in &depths {
                for (alpha, beta) in [(1.0, 0.0), (-0.75, 0.5), (0.5, 1.0)] {
                    let (alpha, beta) = (T::of(alpha), T::of(beta));
                    // n == 1: gemv(op_A, A, op(B)[:, 0], C[:, 0]).
                    for ta in [Trans::NoTrans, Trans::Trans] {
                        let a = if ta == Trans::NoTrans {
                            fill::<T>(len, k, 11)
                        } else {
                            fill::<T>(k, len, 11)
                        };
                        let b = fill::<T>(k, 1, 12);
                        let c0 = fill::<T>(len, 1, 13);
                        let mut c = c0.clone();
                        gemm_planned(
                            ta,
                            Trans::NoTrans,
                            alpha,
                            a.as_ref(),
                            b.as_ref(),
                            beta,
                            c.as_mut(),
                            Par::Seq,
                            BlockingSource::Default,
                            plan,
                        );
                        let x: Vec<T> = (0..k).map(|p| b[(p, 0)]).collect();
                        let mut y: Vec<T> = (0..len).map(|i| c0[(i, 0)]).collect();
                        let gt = if ta == Trans::NoTrans {
                            GemvTrans::NoTrans
                        } else {
                            GemvTrans::Trans
                        };
                        gemv(gt, alpha, a.as_ref(), &x, beta, &mut y);
                        for (i, yi) in y.iter().enumerate() {
                            assert_eq!(
                                c[(i, 0)].bits(),
                                yi.bits(),
                                "n=1 {ta:?} len={len} k={k} ({i})"
                            );
                        }
                    }
                    // m == 1 (n > 1; m == n == 1 follows the n == 1 rule):
                    // gemv(op_B, B, op(A)[0, :], C[0, :]).
                    if len == 1 {
                        continue;
                    }
                    for tb in [Trans::NoTrans, Trans::Trans] {
                        let a = fill::<T>(1, k, 21);
                        let b = if tb == Trans::NoTrans {
                            fill::<T>(k, len, 22)
                        } else {
                            fill::<T>(len, k, 22)
                        };
                        let c0 = fill::<T>(1, len, 23);
                        let mut c = c0.clone();
                        gemm_planned(
                            Trans::NoTrans,
                            tb,
                            alpha,
                            a.as_ref(),
                            b.as_ref(),
                            beta,
                            c.as_mut(),
                            Par::Seq,
                            BlockingSource::Default,
                            plan,
                        );
                        let x: Vec<T> = (0..k).map(|p| a[(0, p)]).collect();
                        let mut y: Vec<T> = (0..len).map(|j| c0[(0, j)]).collect();
                        let gt = if tb == Trans::NoTrans {
                            GemvTrans::Trans
                        } else {
                            GemvTrans::NoTrans
                        };
                        gemv(gt, alpha, b.as_ref(), &x, beta, &mut y);
                        for (j, yj) in y.iter().enumerate() {
                            assert_eq!(
                                c[(0, j)].bits(),
                                yj.bits(),
                                "m=1 {tb:?} len={len} k={k} ({j})"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// On aarch64 the portable kernels fuse every multiply-add of the contract,
/// so the portable tier gives exactly the bits of `level2::gemv` (the NEON
/// GEMV kernels) on the same column-major operands.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[test]
fn test_gemm_thin_portable_equals_gemv_bitwise() {
    check_plan_equals_gemv(&thin_plan::<f64>());
    check_plan_equals_gemv(&thin_plan::<f32>());
}

#[test]
fn test_gemm_thin_threads_follow_the_policy_gate() {
    // The thin work gate of every policy: 512² multiply-adds in total and
    // 128² per thread.
    let ctx = GemmContext::detect(CANDIDATE_POLICY);
    let a = fill::<f64>(600, 500, 1);
    assert_eq!(thin_threads(&ctx, &a.as_ref(), Par::Seq), 1);
    let small = fill::<f64>(100, 100, 2);
    #[cfg(feature = "parallel")]
    {
        assert_eq!(thin_threads(&ctx, &small.as_ref(), Par::RayonWith(4)), 1);
        assert_eq!(thin_threads(&ctx, &a.as_ref(), Par::RayonWith(4)), 4);
        // 300000 / 32 < 128²: the per-thread floor closes the gate.
        assert_eq!(thin_threads(&ctx, &a.as_ref(), Par::RayonWith(32)), 1);
    }
    #[cfg(not(feature = "parallel"))]
    assert_eq!(thin_threads(&ctx, &small.as_ref(), Par::Seq), 1);
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemm_thin_parallel_bitwise_equals_sequential() {
    // Every split of y (uneven parts, parts shorter than a vector step at the
    // end) gives the sequential bits, for both forms and both thin shapes,
    // on the portable and the native tier.
    fn check<T: Elem>() {
        let portable = thin_plan::<T>();
        let native = native_plan::<T>();
        // (len of y, k): every case passes the gate for every thread count
        // below; the lengths sit around multiples of the 64-element split
        // unit (129 = 2·64 + 1, 193 = 3·64 + 1, 321 = 5·64 + 1).
        let cases = [
            (1_000usize, 300usize),
            (777, 513),
            (130, 2_100),
            (129, 2_100),
            (193, 1_400),
            (321, 900),
            (4_099, 70),
        ];
        for plan in [&portable, &native] {
            for &(len, k) in &cases {
                for (alpha, beta) in [(1.0, 0.0), (-0.75, 0.5)] {
                    let (alpha, beta) = (T::of(alpha), T::of(beta));
                    let run = |ta: Trans, tb: Trans, a: &Mat<T>, b: &Mat<T>, c0: &Mat<T>, par| {
                        let mut c = c0.clone();
                        gemm_planned(
                            ta,
                            tb,
                            alpha,
                            a.as_ref(),
                            b.as_ref(),
                            beta,
                            c.as_mut(),
                            par,
                            BlockingSource::Default,
                            plan,
                        );
                        c
                    };
                    for t in [Trans::NoTrans, Trans::Trans] {
                        // n == 1 (op(A) = A: axpy form; Aᵀ: dot form).
                        let a = if t == Trans::NoTrans {
                            fill::<T>(len, k, 31)
                        } else {
                            fill::<T>(k, len, 31)
                        };
                        let b = fill::<T>(k, 1, 32);
                        let c0 = fill::<T>(len, 1, 33);
                        // m == 1 (op(B) = B: dot form; Bᵀ: axpy form).
                        let x = fill::<T>(1, k, 34);
                        let s = if t == Trans::NoTrans {
                            fill::<T>(k, len, 35)
                        } else {
                            fill::<T>(len, k, 35)
                        };
                        let r0 = fill::<T>(1, len, 36);
                        let seq_col = run(t, Trans::NoTrans, &a, &b, &c0, Par::Seq);
                        let seq_row = run(Trans::NoTrans, t, &x, &s, &r0, Par::Seq);
                        for threads in THREADS {
                            let par = Par::RayonWith(threads);
                            let ctx = format!("len={len} k={k} {t:?} t={threads}");
                            let col = run(t, Trans::NoTrans, &a, &b, &c0, par);
                            let row = run(Trans::NoTrans, t, &x, &s, &r0, par);
                            for i in 0..len {
                                assert_eq!(
                                    col[(i, 0)].bits(),
                                    seq_col[(i, 0)].bits(),
                                    "n=1 {ctx} ({i})"
                                );
                                assert_eq!(
                                    row[(0, i)].bits(),
                                    seq_row[(0, i)].bits(),
                                    "m=1 {ctx} ({i})"
                                );
                            }
                        }
                    }
                }
            }
        }
        // The cases do open the gate.
        let ctx = GemmContext::detect(CANDIDATE_POLICY);
        for &(len, k) in &cases {
            let s = fill::<T>(len, k, 1);
            for threads in THREADS {
                assert_eq!(
                    thin_threads(&ctx, &s.as_ref(), Par::RayonWith(threads)),
                    threads
                );
            }
        }
    }
    const THREADS: [usize; 4] = [2, 3, 5, 8];
    check::<f64>();
    check::<f32>();
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemm_thin_parallel_parts_cover_y() {
    // Parts are whole 64-element runs (the last one shorter), at most one
    // per thread.
    for len in [1usize, 63, 64, 65, 1_000, 4_099] {
        for threads in [1usize, 2, 3, 8, 64] {
            let part = part_len(len, threads);
            assert_eq!(part % SPLIT_ALIGN, 0, "{len} {threads}");
            assert!(len.div_ceil(part) <= threads.max(1), "{len} {threads}");
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel tables and the kernels themselves
// ---------------------------------------------------------------------------

/// The message of the panic that `kernel` raises for operands that do not
/// match. Every thin kernel checks its operands first and names itself in
/// that message, so the message identifies the kernel without comparing
/// function pointers (which is not reliable).
fn kernel_name<T: Elem>(kernel: ThinKernel<T>) -> String {
    let s: Mat<T> = Mat::zeros(2, 2);
    let mut y = [T::zero(); 1];
    // Two assumptions: panics unwind (the test profile's `panic = "unwind"`,
    // so `catch_unwind` sees them), and every kernel checks its operands
    // before anything else (before the AVX2 kernels' CPU check, too).
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernel(T::one(), s.as_ref(), &[], &mut y);
    }));
    match outcome {
        Ok(()) => panic!("a thin kernel accepted mismatched operands"),
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|m| (*m).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default(),
    }
}

/// The names (message prefixes) of the axpy and dot kernels that `tier` must
/// select for `T` in this build (module docs, "Kernels per tier").
fn expected_kernel_names<T: Elem>(tier: SimdTier) -> (String, String) {
    let ty = T::NAME;
    let neon_built = cfg!(all(target_arch = "aarch64", not(feature = "force-scalar")));
    if neon_built && tier == SimdTier::Neon {
        (format!("notrans_{ty}:"), format!("trans_{ty}:"))
    } else if cfg!(target_arch = "x86_64") && tier == SimdTier::Avx2Fma {
        (format!("axpy_{ty}_avx2:"), format!("dot_{ty}_avx2:"))
    } else {
        ("axpy_portable:".to_string(), "dot_portable:".to_string())
    }
}

const ALL_TIERS: [SimdTier; 4] = [
    SimdTier::Portable,
    SimdTier::Sse42,
    SimdTier::Neon,
    SimdTier::Avx2Fma,
];

#[test]
fn test_gemm_thin_kernel_table_follows_tier() {
    // A pure rule, checked on every target and feature set: NEON tier -> the
    // level-2 NEON GEMV kernels (where they are built), AVX2 + FMA tier -> the
    // AVX2 kernels (x86-64), everything else -> portable. The AVX2 kernels
    // check their operands before the CPU, so this runs without AVX2 too.
    fn check<T: Elem>() {
        for tier in ALL_TIERS {
            let Some(kernels) = thin_kernels::<T>(tier) else {
                panic!("no thin kernels for {} on {tier:?}", T::NAME);
            };
            let (axpy, dot) = expected_kernel_names::<T>(tier);
            let (got_axpy, got_dot) = (kernel_name(kernels.axpy), kernel_name(kernels.dot));
            assert!(
                got_axpy.starts_with(&axpy),
                "{} {tier:?}: axpy kernel `{got_axpy}`, expected `{axpy}`",
                T::NAME
            );
            assert!(
                got_dot.starts_with(&dot),
                "{} {tier:?}: dot kernel `{got_dot}`, expected `{dot}`",
                T::NAME
            );
        }
    }
    check::<f64>();
    check::<f32>();
    for tier in ALL_TIERS {
        assert!(thin_kernels::<num_complex::Complex64>(tier).is_none());
        assert!(thin_kernels::<num_complex::Complex32>(tier).is_none());
    }
}

/// The kernel-level contract: `y` after the kernel of `form` with the
/// multiply-adds formed as `mode` says, written out with indices.
fn kernel_oracle<T: Elem>(
    mode: Madd,
    form: Form,
    alpha: T,
    s: &MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
) {
    match form {
        Form::Axpy => {
            for (j, &xj) in x.iter().enumerate() {
                let t = alpha * xj;
                for (i, yi) in y.iter_mut().enumerate() {
                    *yi = mac(mode, s[(i, j)], t, *yi);
                }
            }
        }
        Form::Dot => {
            for (j, yj) in y.iter_mut().enumerate() {
                let d = dot_ref(mode, s.nrows(), |p| (s[(p, j)], x[p]));
                *yj = *yj + alpha * d;
            }
        }
    }
}

/// Runs both kernels of `kernels` on every `m × n` view with `m <= 41` and
/// `n <= 13` (every row remainder of every vector step, every column group
/// of 8, 4 and 1) and compares them bit for bit with the oracle in `mode`.
///
/// `S` sits inside a NaN store (a NaN row above, NaN rows below up to the
/// padded leading dimension, a NaN column after), and `x` and `y` are exact
/// sub-slices of NaN-bordered buffers, so a read outside the operands turns
/// a result into NaN and a write outside `y` changes a border.
fn check_kernels_against_oracle<T: Elem>(kernels: ThinKernels<T>, mode: Madd, what: &str) {
    for m in 0..=41usize {
        for n in 0..=13usize {
            let mut store: Mat<T> = Mat::filled(m + 4, n + 1, T::nan());
            for j in 0..n {
                for i in 0..m {
                    store[(i + 1, j)] = T::of(unit(i * 31 + j * 1_009 + m * 7 + n));
                }
            }
            let s = store.as_ref().submatrix(1, 0, m, n);
            for form in [Form::Axpy, Form::Dot] {
                let (xl, yl) = match form {
                    Form::Axpy => (n, m),
                    Form::Dot => (m, n),
                };
                for alpha in [1.0, -0.75] {
                    let alpha = T::of(alpha);
                    let mut xs = vec![T::nan(); xl + 2];
                    for (p, xp) in xs[1..=xl].iter_mut().enumerate() {
                        *xp = T::of(unit(p * 13 + 5));
                    }
                    let mut ys = vec![T::nan(); yl + 2];
                    for (i, yi) in ys[1..=yl].iter_mut().enumerate() {
                        *yi = T::of(unit(i * 17 + 3));
                    }
                    let mut expected = ys[1..=yl].to_vec();
                    kernel_oracle(mode, form, alpha, &s, &xs[1..=xl], &mut expected);
                    let before = ys.clone();
                    run(kernels, form, alpha, s, &xs[1..=xl], &mut ys[1..=yl]);
                    let ctx = format!("{what} {form:?} m={m} n={n} alpha={alpha}");
                    assert_eq!(ys[0].bits(), before[0].bits(), "{ctx}: y[-1] written");
                    assert_eq!(
                        ys[yl + 1].bits(),
                        before[yl + 1].bits(),
                        "{ctx}: y[len] written"
                    );
                    for (i, (got, want)) in ys[1..=yl].iter().zip(&expected).enumerate() {
                        assert_eq!(
                            got.bits(),
                            want.bits(),
                            "{ctx}: y[{i}] = {got}, contract {want}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_thin_portable_kernels_equal_contract_bitwise() {
    // Bitwise: the oracle forms the multiply-adds exactly as this build's
    // portable kernels (fused with a native FMA, split otherwise).
    let mode = madd_of(SimdTier::Portable);
    check_kernels_against_oracle(PORTABLE_F64, mode, "portable f64");
    check_kernels_against_oracle(PORTABLE_F32, mode, "portable f32");
}

#[test]
fn test_gemm_thin_native_kernels_equal_contract_bitwise() {
    // Bitwise: the kernels of this host's tier (NEON, AVX2 + FMA, or
    // portable) follow the contract's per-element sequence exactly.
    let tier = SimdTier::current();
    let mode = madd_of(tier);
    check_kernels_against_oracle(tier_kernels_f64(tier), mode, "native f64");
    check_kernels_against_oracle(tier_kernels_f32(tier), mode, "native f32");
}

#[test]
fn test_gemm_thin_avx2_kernels_equal_fused_oracle_bitwise() {
    // Bitwise: every multiply-add that the contract fuses is a `vfmadd` in
    // the AVX2 kernels and a `mul_add` in the oracle (`t_j = alpha * x[j]` and
    // the last step of the dot form are unfused in both); the partial sums,
    // their reduction tree and the row tails follow the same order.
    if !avx2_fma_tier_or_skip("test_gemm_thin_avx2_kernels_equal_fused_oracle_bitwise") {
        return;
    }
    let (f64_kernels, f32_kernels) = (
        tier_kernels_f64(SimdTier::Avx2Fma),
        tier_kernels_f32(SimdTier::Avx2Fma),
    );
    assert!(kernel_name(f64_kernels.axpy).starts_with("axpy_f64_avx2:"));
    assert!(kernel_name(f32_kernels.dot).starts_with("dot_f32_avx2:"));
    check_kernels_against_oracle(f64_kernels, Madd::Fused, "avx2 f64");
    check_kernels_against_oracle(f32_kernels, Madd::Fused, "avx2 f32");
}

/// Row counts of [`check_kernels_exact_alloc`]: every remainder of the vector
/// steps of both forms and both types (f64: axpy 16 / 4 / 1, dot 8 / 4 / 2 /
/// 1; f32: axpy 32 / 8 / 1, dot 16 / 8 / 4 / 1..=3), a full strip plus one
/// row, and a full strip followed by a smaller vector step (f64 axpy 16 + 4:
/// 20 and 21; f32 axpy 32 + 8: 40 and 41).
#[cfg(target_arch = "x86_64")]
const EXACT_ALLOC_ROWS: [usize; 25] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 31, 32, 33, 40, 41,
];
/// Column counts of [`check_kernels_exact_alloc`]: every column group (axpy
/// 8 / 4 / 1, dot 4 / 1), alone and combined (axpy 9 = 8 + 1 and 13 =
/// 8 + 4 + 1; dot 5 = 4 + 1 and 13 = 3 · 4 + 1), and several single columns
/// in a row (3). Kept short, so the test runs under Miri in about two
/// minutes.
#[cfg(target_arch = "x86_64")]
const EXACT_ALLOC_COLS: [usize; 7] = [0, 1, 3, 4, 5, 9, 13];

/// Both kernels of `kernels` on operands that end exactly at the end of
/// their allocations: `S` (`m × n`, leading dimension `m` or `m + 3`) is a
/// slice of exactly `(n − 1)·lda + m` elements, `x` and `y` slices of exactly
/// their lengths, so no element after the last one the kernel may read
/// exists. Every result is the fused contract bit for bit (the gap rows of a
/// padded `S` hold NaN, so a read of them would show).
///
/// Under Miri any load past the end of `S`, `x` or `y` (a wider step taken
/// for fewer rows than it covers) is reported as undefined behaviour:
///
/// ```text
/// RUSTFLAGS="-C target-feature=+avx2,+fma" OXIBLAS_REQUIRE_AVX2=1 \
///   MIRIFLAGS="-Zmiri-env-forward=OXIBLAS_REQUIRE_AVX2" \
///   cargo +nightly miri test -p oxiblas-blas --target x86_64-unknown-linux-gnu \
///   --lib test_gemm_thin_avx2_kernels_exact_alloc
/// ```
#[cfg(target_arch = "x86_64")]
fn check_kernels_exact_alloc<T: Elem>(kernels: ThinKernels<T>, what: &str) {
    for m in EXACT_ALLOC_ROWS {
        for n in EXACT_ALLOC_COLS {
            for pad in [0usize, 3] {
                let lda = m + pad;
                let len = if m == 0 || n == 0 {
                    0
                } else {
                    (n - 1) * lda + m
                };
                let data: Vec<T> = (0..len)
                    .map(|idx| {
                        if idx % lda.max(1) < m {
                            T::of(unit(idx * 31 + m * 7 + n))
                        } else {
                            T::nan()
                        }
                    })
                    .collect();
                let Some(s) = MatRef::from_strided(&data, m, n, lda, 1) else {
                    panic!("{what}: view {m}x{n} lda {lda}");
                };
                for form in [Form::Axpy, Form::Dot] {
                    let (xl, yl) = match form {
                        Form::Axpy => (n, m),
                        Form::Dot => (m, n),
                    };
                    let alpha = T::of(-0.75);
                    let x: Vec<T> = (0..xl).map(|p| T::of(unit(p * 13 + 5))).collect();
                    let mut y: Vec<T> = (0..yl).map(|i| T::of(unit(i * 17 + 3))).collect();
                    let mut expected = y.clone();
                    kernel_oracle(Madd::Fused, form, alpha, &s, &x, &mut expected);
                    run(kernels, form, alpha, s, &x, &mut y);
                    for (i, (got, want)) in y.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            got.bits(),
                            want.bits(),
                            "{what} {form:?} m={m} n={n} lda={lda}: y[{i}] = {got}, contract {want}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_thin_avx2_kernels_exact_alloc() {
    // Miri command: see `check_kernels_exact_alloc`.
    if !avx2_fma_tier_or_skip("test_gemm_thin_avx2_kernels_exact_alloc") {
        return;
    }
    check_kernels_exact_alloc(tier_kernels_f64(SimdTier::Avx2Fma), "avx2 f64");
    check_kernels_exact_alloc(tier_kernels_f32(SimdTier::Avx2Fma), "avx2 f32");
}

/// Runs the kernels of `kernels` on `y` whole and on `y` cut into pieces of
/// `cut` elements (every piece on its own rows or columns of `S`, starting at
/// unaligned addresses), and requires the same bits.
fn check_position_independent<T: Elem>(kernels: ThinKernels<T>, what: &str) {
    for (rows, cols) in [(203usize, 7usize), (7, 203), (100, 33), (45, 45)] {
        let store = fill::<T>(rows + 5, cols + 3, rows + cols);
        let s = store.as_ref().submatrix(3, 2, rows, cols);
        for form in [Form::Axpy, Form::Dot] {
            let (xl, yl) = match form {
                Form::Axpy => (cols, rows),
                Form::Dot => (rows, cols),
            };
            let x: Vec<T> = (0..xl).map(|p| T::of(unit(p * 3 + 1))).collect();
            let y0: Vec<T> = (0..yl).map(|i| T::of(unit(i * 5 + 2))).collect();
            let alpha = T::of(-0.75);
            let mut whole = y0.clone();
            run(kernels, form, alpha, s, &x, &mut whole);
            for cut in [1usize, 2, 3, 5, 7, 13, 17, 33, 63, 64, 65] {
                let mut parts = y0.clone();
                for (p, part) in parts.chunks_mut(cut).enumerate() {
                    let start = p * cut;
                    let sub = match form {
                        Form::Axpy => s.submatrix(start, 0, part.len(), cols),
                        Form::Dot => s.submatrix(0, start, rows, part.len()),
                    };
                    run(kernels, form, alpha, sub, &x, part);
                }
                for (i, (got, want)) in parts.iter().zip(&whole).enumerate() {
                    assert_eq!(
                        got.bits(),
                        want.bits(),
                        "{what} {form:?} {rows}x{cols} cut={cut} ({i})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_gemm_thin_kernels_position_independent() {
    // An element's result never depends on where its part of y starts, on
    // alignment, or on the row / column count of the part.
    for tier in [SimdTier::Portable, SimdTier::current()] {
        check_position_independent(tier_kernels_f64(tier), &format!("{tier:?} f64"));
        check_position_independent(tier_kernels_f32(tier), &format!("{tier:?} f32"));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_gemm_thin_run_split_bitwise_equals_sequential() {
    // The split itself, without the work gate: lengths around multiples of
    // the 64-element split unit, odd thread counts, both forms, both types,
    // the portable and the native kernels.
    fn check<T: Elem>(kernels: ThinKernels<T>, what: &str) {
        for len in [63usize, 64, 65, 127, 128, 129, 191, 192, 193, 321, 1_000] {
            for k in [5usize, 37] {
                for form in [Form::Axpy, Form::Dot] {
                    let store = match form {
                        Form::Axpy => fill::<T>(len, k, len + k),
                        Form::Dot => fill::<T>(k, len, len + k),
                    };
                    let s = store.as_ref();
                    let x: Vec<T> = (0..k).map(|p| T::of(unit(p * 11 + 4))).collect();
                    let y0: Vec<T> = (0..len).map(|i| T::of(unit(i * 7 + 9))).collect();
                    let alpha = T::of(-0.5);
                    let mut seq = y0.clone();
                    run(kernels, form, alpha, s, &x, &mut seq);
                    for threads in [2usize, 3, 5, 8] {
                        if len > SPLIT_ALIGN {
                            // y really is cut into more than one part.
                            assert!(part_len(len, threads) < len, "{len} {threads}");
                        }
                        let mut par = y0.clone();
                        run_split(kernels, form, alpha, s, &x, &mut par, threads);
                        for (i, (got, want)) in par.iter().zip(&seq).enumerate() {
                            assert_eq!(
                                got.bits(),
                                want.bits(),
                                "{what} {form:?} len={len} k={k} t={threads} ({i})"
                            );
                        }
                    }
                }
            }
        }
    }
    for tier in [SimdTier::Portable, SimdTier::current()] {
        check(tier_kernels_f64(tier), &format!("{tier:?} f64"));
        check(tier_kernels_f32(tier), &format!("{tier:?} f32"));
    }
}

/// Runs the same thin problems through `left` and `right` and hands every
/// pair of results to `same` with the componentwise error bound
/// `Σ_p |alpha·op(A)[i,p]·op(B)[p,j]| + |beta·C0[i,j]|` and the depth.
fn compare_plans<T: Elem>(
    left: &GemmPlan<T>,
    right: &GemmPlan<T>,
    same: impl Fn(&str, T, T, f64, usize),
) {
    let shapes = [
        (1usize, 1_000usize, 37usize),
        (37, 1_000, 1),
        (1, 17, 45),
        (45, 17, 1),
        (1, 64, 1),
        (1, 3, 9),
        (130, 257, 1),
        (1, 257, 130),
    ];
    for (m, k, n) in shapes {
        for ta in [Trans::NoTrans, Trans::Trans] {
            for tb in [Trans::NoTrans, Trans::Trans] {
                let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
                let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
                let a = fill::<T>(ar, ac, 41 + m);
                let b = fill::<T>(br, bc, 42 + n);
                let c0 = fill::<T>(m, n, 43);
                let (alpha, beta) = (T::of(-0.5), T::of(0.25));
                let results: Vec<Mat<T>> = [left, right]
                    .iter()
                    .map(|plan| {
                        let mut c = c0.clone();
                        gemm_planned(
                            ta,
                            tb,
                            alpha,
                            a.as_ref(),
                            b.as_ref(),
                            beta,
                            c.as_mut(),
                            Par::Seq,
                            BlockingSource::Default,
                            plan,
                        );
                        c
                    })
                    .collect();
                for j in 0..n {
                    for i in 0..m {
                        let mut bound = (beta.to_f64() * c0[(i, j)].to_f64()).abs();
                        for p in 0..k {
                            bound += (alpha.to_f64()
                                * read_op(&a.as_ref(), ta, i, p).to_f64()
                                * read_op(&b.as_ref(), tb, p, j).to_f64())
                            .abs();
                        }
                        let ctx = format!("{m}x{k}x{n} {ta:?}/{tb:?} ({i},{j})");
                        same(&ctx, results[0][(i, j)], results[1][(i, j)], bound, k);
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_thin_avx2_matches_portable() {
    if !avx2_fma_tier_or_skip("test_gemm_thin_avx2_matches_portable") {
        return;
    }
    fn check<T: Elem>() {
        let avx2 = native_plan::<T>();
        assert_eq!(avx2.ctx.tier(), SimdTier::Avx2Fma);
        let portable = thin_plan::<T>();
        // Bitwise when this build's portable kernels fuse (x86-64 built with
        // the `fma` target feature): both then run the contract with fused
        // multiply-adds. Otherwise the portable kernels round every product
        // and sum separately while AVX2 fuses, so the two agree only within
        // the contract's error bound (each is within (k + 2)·eps·bound of
        // the exact value), not bit for bit.
        let bitwise = madd_of(SimdTier::Portable) == Madd::Fused;
        compare_plans(&avx2, &portable, |ctx, got, want, bound, k| {
            if bitwise {
                assert_eq!(
                    got.bits(),
                    want.bits(),
                    "{} {ctx}: {got} vs {want}",
                    T::NAME
                );
            } else {
                let tol = 4.0 * (k as f64 + 2.0) * T::EPS * bound + f64::MIN_POSITIVE;
                assert!(
                    (got.to_f64() - want.to_f64()).abs() <= tol,
                    "{} {ctx}: {got} vs {want} (tol {tol})",
                    T::NAME
                );
            }
        });
    }
    check::<f64>();
    check::<f32>();
}

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[test]
fn test_gemm_thin_neon_matches_portable_bitwise() {
    // Bitwise: on aarch64 the portable kernels fuse every multiply-add of the
    // contract, exactly like the NEON kernels.
    fn check<T: Elem>() {
        let neon = native_plan::<T>();
        assert_eq!(neon.ctx.tier(), SimdTier::Neon);
        compare_plans(&neon, &thin_plan::<T>(), |ctx, got, want, _, _| {
            assert_eq!(
                got.bits(),
                want.bits(),
                "{} {ctx}: {got} vs {want}",
                T::NAME
            );
        });
    }
    check::<f64>();
    check::<f32>();
}

/// Checks that `plan`'s thin path gives the bits of the GEMV kernels `axpy`
/// and `dot` called directly on the same column-major operands (with `y`
/// scaled by `beta` first, as the contract says).
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
fn check_plan_equals_kernels<T: Elem>(plan: &GemmPlan<T>, axpy: ThinKernel<T>, dot: ThinKernel<T>) {
    for len in [1usize, 2, 3, 7, 16, 17, 33, 100] {
        for k in [1usize, 3, 8, 15, 16, 17, 64, 257] {
            let (alpha, beta) = (T::of(-0.75), T::of(0.5));
            for ta in [Trans::NoTrans, Trans::Trans] {
                // n == 1: op(A) = A runs `axpy` on A, Aᵀ runs `dot` on A.
                let a = if ta == Trans::NoTrans {
                    fill::<T>(len, k, 51)
                } else {
                    fill::<T>(k, len, 51)
                };
                let b = fill::<T>(k, 1, 52);
                let c0 = fill::<T>(len, 1, 53);
                let mut c = c0.clone();
                gemm_planned(
                    ta,
                    Trans::NoTrans,
                    alpha,
                    a.as_ref(),
                    b.as_ref(),
                    beta,
                    c.as_mut(),
                    Par::Seq,
                    BlockingSource::Default,
                    plan,
                );
                let x: Vec<T> = (0..k).map(|p| b[(p, 0)]).collect();
                let mut y: Vec<T> = (0..len).map(|i| beta * c0[(i, 0)]).collect();
                let kernel = if ta == Trans::NoTrans { axpy } else { dot };
                kernel(alpha, a.as_ref(), &x, &mut y);
                for (i, yi) in y.iter().enumerate() {
                    assert_eq!(c[(i, 0)].bits(), yi.bits(), "n=1 {ta:?} {len} {k} ({i})");
                }
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[test]
fn test_gemm_thin_neon_equals_gemv_kernels_bitwise() {
    // The NEON tier runs the level-2 NEON GEMV kernels, so the thin path is
    // bit for bit `level2::gemv` and the kernels called directly.
    use crate::level2::gemv::neon;
    let (f64_plan, f32_plan) = (native_plan::<f64>(), native_plan::<f32>());
    assert_eq!(f64_plan.ctx.tier(), SimdTier::Neon);
    check_plan_equals_gemv(&f64_plan);
    check_plan_equals_gemv(&f32_plan);
    check_plan_equals_kernels(&f64_plan, neon::notrans_f64, neon::trans_f64);
    check_plan_equals_kernels(&f32_plan, neon::notrans_f32, neon::trans_f32);
}

#[test]
fn test_gemm_thin_nan_inf_in_large_operand_propagate() {
    // NaN and infinity in the large operand reach exactly the elements of C
    // that use them, even where the vector x is zero (NaN·0 and ∞·0 are NaN):
    // no column or row is skipped. Positions cover the vector bodies, the
    // 2L / L steps and the scalar tails of both forms. Checked with
    // `is_nan`, not bits: NaN payloads are not part of the contract.
    fn check<T: Elem>(plan: &GemmPlan<T>, poison: T) {
        let (len, k) = (37usize, 41usize);
        let spots = [
            (0usize, 0usize),
            (5, 17),
            (36, 40),
            (20, 33),
            (33, 36),
            (16, 8),
        ];
        for t in [Trans::NoTrans, Trans::Trans] {
            for (i0, p0) in spots {
                // n == 1: A is large (op(A) = A: axpy form; Aᵀ: dot form).
                let mut a = if t == Trans::NoTrans {
                    fill::<T>(len, k, 61)
                } else {
                    fill::<T>(k, len, 61)
                };
                if t == Trans::NoTrans {
                    a[(i0, p0)] = poison;
                } else {
                    a[(p0, i0)] = poison;
                }
                let b: Mat<T> = Mat::zeros(k, 1);
                let mut c: Mat<T> = Mat::filled(len, 1, T::nan());
                gemm_planned(
                    t,
                    Trans::NoTrans,
                    T::one(),
                    a.as_ref(),
                    b.as_ref(),
                    T::zero(),
                    c.as_mut(),
                    Par::Seq,
                    BlockingSource::Default,
                    plan,
                );
                for i in 0..len {
                    assert_eq!(
                        c[(i, 0)].to_f64().is_nan(),
                        i == i0,
                        "n=1 {t:?} poison at ({i0},{p0}) -> ({i}) = {}",
                        c[(i, 0)]
                    );
                }
                // m == 1: B is large (op(B) = B: dot form; Bᵀ: axpy form).
                let mut bm = if t == Trans::NoTrans {
                    fill::<T>(k, len, 62)
                } else {
                    fill::<T>(len, k, 62)
                };
                if t == Trans::NoTrans {
                    bm[(p0, i0)] = poison;
                } else {
                    bm[(i0, p0)] = poison;
                }
                let am: Mat<T> = Mat::zeros(1, k);
                let mut r: Mat<T> = Mat::filled(1, len, T::nan());
                gemm_planned(
                    Trans::NoTrans,
                    t,
                    T::one(),
                    am.as_ref(),
                    bm.as_ref(),
                    T::zero(),
                    r.as_mut(),
                    Par::Seq,
                    BlockingSource::Default,
                    plan,
                );
                for j in 0..len {
                    assert_eq!(
                        r[(0, j)].to_f64().is_nan(),
                        j == i0,
                        "m=1 {t:?} poison at ({p0},{i0}) -> ({j}) = {}",
                        r[(0, j)]
                    );
                }
            }
        }
    }
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        check::<f64>(&thin_plan::<f64>(), poison);
        check::<f64>(&native_plan::<f64>(), poison);
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        check::<f32>(&thin_plan::<f32>(), poison);
        check::<f32>(&native_plan::<f32>(), poison);
    }
}
