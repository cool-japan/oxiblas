//! Tests for the small-matrix GEMM path: the alpha/beta contract of
//! `gemm_small` against a naive reference, the NEON kernels, the AVX2 + FMA
//! kernels and their seam, and the transposed small products copied to
//! column-major.

use super::*;
use num_traits::ToPrimitive;
use oxiblas_core::scalar::Scalar;
use oxiblas_matrix::Mat;

#[test]
fn test_gemm_2x2() {
    let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
    let mut c: Mat<f64> = Mat::zeros(2, 2);

    gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

    // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
    assert!((c[(0, 1)] - 22.0).abs() < 1e-10);
    assert!((c[(1, 0)] - 43.0).abs() < 1e-10);
    assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
}

#[test]
fn test_gemm_3x3() {
    let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
    let b: Mat<f64> = Mat::eye(3);
    let mut c: Mat<f64> = Mat::zeros(3, 3);

    gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

    // A*I = A
    for i in 0..3 {
        for j in 0..3 {
            assert!((c[(i, j)] - a[(i, j)]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_gemm_4x4() {
    let a: Mat<f64> = Mat::filled(4, 4, 2.0);
    let b: Mat<f64> = Mat::filled(4, 4, 3.0);
    let mut c: Mat<f64> = Mat::zeros(4, 4);

    gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

    // Each element = 4 * (2*3) = 24
    for i in 0..4 {
        for j in 0..4 {
            assert!((c[(i, j)] - 24.0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_gemm_small_8x8() {
    let n = 8;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let b: Mat<f64> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f64> = Mat::zeros(n, n);

    gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

    // Each element = n * (1*1) = n
    for i in 0..n {
        for j in 0..n {
            assert!((c[(i, j)] - n as f64).abs() < 1e-10);
        }
    }
}

#[test]
fn test_gemm_small_16x16() {
    let n = 16;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let b: Mat<f64> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f64> = Mat::zeros(n, n);

    gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

    for i in 0..n {
        for j in 0..n {
            assert!((c[(i, j)] - n as f64).abs() < 1e-10);
        }
    }
}

#[test]
fn test_gemm_small_general() {
    let m = 20;
    let k = 15;
    let n = 25;
    let a: Mat<f64> = Mat::filled(m, k, 1.0);
    let b: Mat<f64> = Mat::filled(k, n, 1.0);
    let mut c: Mat<f64> = Mat::zeros(m, n);

    gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

    for i in 0..m {
        for j in 0..n {
            assert!((c[(i, j)] - k as f64).abs() < 1e-10);
        }
    }
}

#[test]
fn test_gemm_with_alpha_beta() {
    let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b: Mat<f64> = Mat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let mut c: Mat<f64> = Mat::from_rows(&[&[10.0, 20.0], &[30.0, 40.0]]);

    // C = 2*A*B + 0.5*C = 2*A + 0.5*C
    gemm_small(2.0, &a.as_ref(), &b.as_ref(), 0.5, &mut c.as_mut());

    // Expected: 2*[[1,2],[3,4]] + 0.5*[[10,20],[30,40]]
    //         = [[2,4],[6,8]] + [[5,10],[15,20]]
    //         = [[7,14],[21,28]]
    assert!((c[(0, 0)] - 7.0).abs() < 1e-10);
    assert!((c[(0, 1)] - 14.0).abs() < 1e-10);
    assert!((c[(1, 0)] - 21.0).abs() < 1e-10);
    assert!((c[(1, 1)] - 28.0).abs() < 1e-10);
}

#[test]
fn test_gemm_f32() {
    let a: Mat<f32> = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
    let b: Mat<f32> = Mat::from_rows(&[&[5.0f32, 6.0], &[7.0, 8.0]]);
    let mut c: Mat<f32> = Mat::zeros(2, 2);

    gemm_small(1.0f32, &a.as_ref(), &b.as_ref(), 0.0f32, &mut c.as_mut());

    assert!((c[(0, 0)] - 19.0).abs() < 1e-5);
    assert!((c[(0, 1)] - 22.0).abs() < 1e-5);
    assert!((c[(1, 0)] - 43.0).abs() < 1e-5);
    assert!((c[(1, 1)] - 50.0).abs() < 1e-5);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemm_small_f64_simd() {
    let n = 8;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let b: Mat<f64> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f64> = Mat::zeros(n, n);

    gemm_small_f64_simd(1.0, &a.as_ref(), &b.as_ref(), &mut c.as_mut());

    for i in 0..n {
        for j in 0..n {
            assert!((c[(i, j)] - n as f64).abs() < 1e-10);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemm_small_f32_simd() {
    let n = 8;
    let a: Mat<f32> = Mat::filled(n, n, 1.0);
    let b: Mat<f32> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f32> = Mat::zeros(n, n);

    gemm_small_f32_simd(1.0, &a.as_ref(), &b.as_ref(), &mut c.as_mut());

    for i in 0..n {
        for j in 0..n {
            assert!((c[(i, j)] - n as f32).abs() < 1e-5);
        }
    }
}

// ------------------------------------------------------------------
// Issue #16 (cause 3): `gemm_small` dispatches to the SIMD kernels.
// These tests pin down the full alpha/beta contract of `gemm_small`
// (including the reference-BLAS rule that A/B are not read when
// alpha == 0 and C is not read when beta == 0) against an independent
// naive reference, for sizes on both sides of the 4x4 SIMD tile and
// for strided (transposed-view) operands that must take the generic path.
// ------------------------------------------------------------------

/// Deterministic, non-symmetric fill in roughly [-1, 1].
fn pattern<T: Field + bytemuck::Zeroable>(rows: usize, cols: usize, seed: usize) -> Mat<T> {
    let mut m: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            let v = ((i * 7 + j * 13 + seed * 5) % 23) as f64 / 11.0 - 1.0;
            m[(i, j)] = T::from_f64(v).unwrap_or_else(T::zero);
        }
    }
    m
}

/// Naive `alpha * A * B + beta * C` honouring the BLAS zero rules.
fn reference<T: Field + bytemuck::Zeroable>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &Mat<T>,
) -> Mat<T> {
    let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
    let mut out: Mat<T> = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let mut acc = T::zero();
            if alpha != T::zero() {
                for p in 0..k {
                    acc += a[(i, p)] * b[(p, j)];
                }
            }
            let scaled = if alpha == T::zero() {
                T::zero()
            } else {
                alpha * acc
            };
            out[(i, j)] = if beta == T::zero() {
                scaled
            } else {
                scaled + beta * c[(i, j)]
            };
        }
    }
    out
}

fn check_gemm_small_contract<T: Field + bytemuck::Zeroable>(tol: f64) {
    let nan = T::from_f64(f64::NAN).unwrap_or_else(T::zero);
    let sizes = [
        (2usize, 2usize, 3usize),
        (3, 5, 4),
        (4, 4, 1),
        (4, 4, 2),
        (4, 4, 4),
        (5, 7, 3),
        (8, 8, 8),
        (13, 6, 11),
        (16, 16, 16),
        (17, 9, 20),
        (32, 32, 32),
    ];
    let minus_two = T::from_f64(-2.0).unwrap_or_else(T::zero);
    let half = T::from_f64(0.5).unwrap_or_else(T::zero);
    for &(m, k, n) in &sizes {
        for &alpha in &[T::zero(), T::one(), minus_two] {
            for &beta in &[T::zero(), T::one(), half] {
                // alpha == 0 poisons A/B, beta == 0 poisons C: neither may
                // reach the result.
                let (a, b) = if alpha == T::zero() {
                    (Mat::filled(m, k, nan), Mat::filled(k, n, nan))
                } else {
                    (pattern::<T>(m, k, 1), pattern::<T>(k, n, 2))
                };
                let c_init = if beta == T::zero() {
                    Mat::filled(m, n, nan)
                } else {
                    pattern::<T>(m, n, 3)
                };
                let expected = reference(alpha, &a.as_ref(), &b.as_ref(), beta, &c_init);

                // Contiguous operands (SIMD-eligible when m, n >= 4).
                let mut c = c_init.clone();
                gemm_small(alpha, &a.as_ref(), &b.as_ref(), beta, &mut c.as_mut());
                assert_close(&c, &expected, tol, m, k, n);

                // Strided transposed views (col_stride != 1): generic path.
                let mut a_store: Mat<T> = Mat::zeros(k, m);
                let mut b_store: Mat<T> = Mat::zeros(n, k);
                for j in 0..k {
                    for i in 0..m {
                        a_store[(j, i)] = a[(i, j)];
                    }
                }
                for j in 0..n {
                    for i in 0..k {
                        b_store[(j, i)] = b[(i, j)];
                    }
                }
                let a_view = a_store.as_ref().transpose();
                let b_view = b_store.as_ref().transpose();
                let mut c = c_init.clone();
                gemm_small(alpha, &a_view, &b_view, beta, &mut c.as_mut());
                assert_close(&c, &expected, tol, m, k, n);
            }
        }
    }
}

fn assert_close<T: Field>(got: &Mat<T>, expected: &Mat<T>, tol: f64, m: usize, k: usize, n: usize) {
    for j in 0..n {
        for i in 0..m {
            let (g, e) = (got[(i, j)], expected[(i, j)]);
            let err = Scalar::abs(g - e).to_f64().unwrap_or(f64::NAN);
            let scale = Scalar::abs(e).to_f64().unwrap_or(f64::NAN).max(1.0);
            assert!(
                err <= tol * scale,
                "{m}x{k}x{n} ({i},{j}): got {g} expected {e}"
            );
        }
    }
}

#[test]
fn test_issue_16_gemm_small_matches_reference_f64() {
    check_gemm_small_contract::<f64>(1e-12);
}

#[test]
fn test_issue_16_gemm_small_matches_reference_f32() {
    check_gemm_small_contract::<f32>(1e-5);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_issue_16_simd_small_kernels_accumulate_into_c() {
    // The SIMD kernels compute C += alpha * A * B ("beta handled by the
    // caller"). Their tiny path (m*n*k < 64) used to call
    // gemm_small(.., beta = 0, ..), zero-filling C instead.
    for &(m, k, n) in &[(4usize, 2usize, 4usize), (4, 3, 5), (8, 8, 8), (6, 5, 7)] {
        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 1.0);
        let mut c: Mat<f64> = Mat::filled(m, n, 1.0);
        gemm_small_f64_simd(2.0, &a.as_ref(), &b.as_ref(), &mut c.as_mut());
        let expected = 1.0 + 2.0 * k as f64;
        for j in 0..n {
            for i in 0..m {
                assert_eq!(c[(i, j)], expected, "f64 {m}x{k}x{n} ({i},{j})");
            }
        }

        let a: Mat<f32> = Mat::filled(m, k, 1.0);
        let b: Mat<f32> = Mat::filled(k, n, 1.0);
        let mut c: Mat<f32> = Mat::filled(m, n, 1.0);
        gemm_small_f32_simd(2.0, &a.as_ref(), &b.as_ref(), &mut c.as_mut());
        let expected = 1.0 + 2.0 * k as f32;
        for j in 0..n {
            for i in 0..m {
                assert_eq!(c[(i, j)], expected, "f32 {m}x{k}x{n} ({i},{j})");
            }
        }
    }
}

// ------------------------------------------------------------------
// Transposed and strided small products copied to column-major
// (`gemm_small_trans_copy`): bit for bit the NoTrans/NoTrans small path on
// the same values, C never read when beta == 0, and no stale scratch.
// ------------------------------------------------------------------

use crate::level3::gemm::{BlockingSource, gemm_planned};
use crate::level3::gemm_plan::{CANDIDATE_POLICY, GemmPlan, GemmRoute, RouteInputs};
use oxiblas_core::parallel::Par;

/// The real element types of the copy route and of the SIMD small kernels.
trait Real: Field + GemmKernel + bytemuck::Zeroable {
    /// Rows of the AVX2 + FMA register tile (two 256-bit vectors).
    const AVX2_TILE_ROWS: usize;
    fn bits(self) -> u64;
    fn finite(self) -> bool;
    fn not_a_number(self) -> bool;
    /// `self * b + c` with one rounding.
    fn fused(self, b: Self, c: Self) -> Self;
    /// The unit roundoff `u` (half the machine epsilon).
    fn unit_roundoff() -> f64;
    /// A NaN with a recognisable payload for guard cells.
    fn guard() -> Self;
    fn of(v: f64) -> Self;
    fn wide(self) -> f64;
    /// `small_simd_f64` / `small_simd_f32` for this type.
    fn simd(
        ctx: GemmContext,
        alpha: Self,
        a: &MatRef<'_, Self>,
        b: &MatRef<'_, Self>,
        c: &mut MatMut<'_, Self>,
    ) -> bool;
}

impl Real for f64 {
    const AVX2_TILE_ROWS: usize = 8;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn not_a_number(self) -> bool {
        self.is_nan()
    }
    fn fused(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    fn unit_roundoff() -> f64 {
        f64::EPSILON / 2.0
    }
    fn guard() -> Self {
        f64::from_bits(0x7ff8_dead_beef_0001)
    }
    fn of(v: f64) -> Self {
        v
    }
    fn wide(self) -> f64 {
        self
    }
    fn simd(
        ctx: GemmContext,
        alpha: Self,
        a: &MatRef<'_, Self>,
        b: &MatRef<'_, Self>,
        c: &mut MatMut<'_, Self>,
    ) -> bool {
        small_simd_f64(ctx, alpha, a, b, c)
    }
}

impl Real for f32 {
    const AVX2_TILE_ROWS: usize = 16;
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn not_a_number(self) -> bool {
        self.is_nan()
    }
    fn fused(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    fn unit_roundoff() -> f64 {
        f64::from(f32::EPSILON) / 2.0
    }
    fn guard() -> Self {
        f32::from_bits(0x7fc0_beef)
    }
    fn of(v: f64) -> Self {
        v as f32
    }
    fn wide(self) -> f64 {
        f64::from(self)
    }
    fn simd(
        ctx: GemmContext,
        alpha: Self,
        a: &MatRef<'_, Self>,
        b: &MatRef<'_, Self>,
        c: &mut MatMut<'_, Self>,
    ) -> bool {
        small_simd_f32(ctx, alpha, a, b, c)
    }
}

/// Bits of every element of `m`, column by column.
fn bits_of<T: Real>(m: &Mat<T>) -> Vec<u64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out.push(m[(i, j)].bits());
        }
    }
    out
}

/// The transpose of `m` as a new column-major matrix.
fn transpose_copy<T: Field + bytemuck::Zeroable>(m: &Mat<T>) -> Mat<T> {
    let mut out: Mat<T> = Mat::zeros(m.ncols(), m.nrows());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out[(j, i)] = m[(i, j)];
        }
    }
    out
}

/// Storage for an operand whose `op(X)` is `op_x`: as stored (`view ==
/// false`) or as a `transpose()` view of the other orientation.
fn storage<T: Field + bytemuck::Zeroable>(op_x: &Mat<T>, trans: Trans, view: bool) -> Mat<T> {
    // With `trans == NoTrans` the view needs op(X)^T stored, and with a
    // transpose the plain storage needs it.
    if (trans == Trans::NoTrans) == view {
        transpose_copy(op_x)
    } else {
        op_x.clone()
    }
}

fn as_operand<'a, T: Field>(stored: &'a Mat<T>, view: bool) -> MatRef<'a, T> {
    if view {
        stored.as_ref().transpose()
    } else {
        stored.as_ref()
    }
}

/// Runs `C = alpha * op(A) * op(B) + beta * C0` through the forced copy
/// route and checks it bit for bit against the NoTrans/NoTrans small path
/// on column-major copies of op(A) and op(B) (the route NN takes).
#[allow(clippy::too_many_arguments)]
fn check_copy_vs_nn<T: Real>(
    plan: &GemmPlan<T>,
    (m, k, n): (usize, usize, usize),
    (ta, tb): (Trans, Trans),
    (a_view, b_view): (bool, bool),
    alpha: T,
    beta: T,
    seed: usize,
) {
    let nan = T::from_f64(f64::NAN).unwrap_or_else(T::zero);
    let op_a = pattern::<T>(m, k, seed);
    let op_b = pattern::<T>(k, n, seed + 1);
    let c0 = if beta == T::zero() {
        Mat::filled(m, n, nan)
    } else {
        pattern::<T>(m, n, seed + 2)
    };
    let ctx = format!(
        "{} {m}x{k}x{n} {ta:?}/{tb:?} views={a_view}/{b_view} alpha={alpha} beta={beta}",
        core::any::type_name::<T>()
    );
    let nn_plan = plan.with_forced_route(GemmRoute::Small);
    let nn = (Trans::NoTrans, Trans::NoTrans);
    let nn_inputs = RouteInputs::new(
        nn.0,
        nn.1,
        &op_a.as_ref(),
        &op_b.as_ref(),
        m,
        n,
        k,
        plan.ctx.tier(),
    );
    assert_eq!(nn_plan.route(&nn_inputs), GemmRoute::Small, "{ctx}");
    let mut c_nn = c0.clone();
    gemm_planned(
        nn.0,
        nn.1,
        alpha,
        op_a.as_ref(),
        op_b.as_ref(),
        beta,
        c_nn.as_mut(),
        Par::Seq,
        BlockingSource::Default,
        &nn_plan,
    );

    let a_store = storage(&op_a, ta, a_view);
    let b_store = storage(&op_b, tb, b_view);
    let (a, b) = (as_operand(&a_store, a_view), as_operand(&b_store, b_view));
    let copy_plan = plan.with_forced_route(GemmRoute::SmallTransCopy);
    let inputs = RouteInputs::new(ta, tb, &a, &b, m, n, k, plan.ctx.tier());
    assert_eq!(copy_plan.route(&inputs), GemmRoute::SmallTransCopy, "{ctx}");
    let mut c_copy = c0.clone();
    gemm_planned(
        ta,
        tb,
        alpha,
        a,
        b,
        beta,
        c_copy.as_mut(),
        Par::Seq,
        BlockingSource::Default,
        &copy_plan,
    );
    assert!(
        bits_of(&c_copy) == bits_of(&c_nn),
        "{ctx}: copy route differs from NoTrans/NoTrans"
    );
    if beta == T::zero() {
        for j in 0..n {
            for i in 0..m {
                assert!(
                    c_copy[(i, j)].finite(),
                    "{ctx}: beta = 0 read C at ({i},{j})"
                );
            }
        }
    }

    // The router itself takes the copy route on a tier with SIMD small
    // kernels, inside the policy's range.
    if plan.route(&inputs) == GemmRoute::SmallTransCopy {
        let mut c_auto = c0.clone();
        gemm_planned(
            ta,
            tb,
            alpha,
            a,
            b,
            beta,
            c_auto.as_mut(),
            Par::Seq,
            BlockingSource::Default,
            plan,
        );
        assert!(
            bits_of(&c_auto) == bits_of(&c_nn),
            "{ctx}: routed copy differs"
        );
    }
}

const COPY_SIZES: [(usize, usize, usize); 9] = [
    (8, 8, 8),
    (4, 32, 4),
    (13, 9, 11),
    (16, 16, 16),
    (31, 7, 29),
    (32, 32, 32),
    (20, 40, 24),
    (5, 100, 6),
    (6, 5, 17),
];

const COPY_TRANS: [(Trans, Trans); 6] = [
    (Trans::Trans, Trans::NoTrans),
    (Trans::NoTrans, Trans::Trans),
    (Trans::Trans, Trans::Trans),
    (Trans::ConjTrans, Trans::NoTrans),
    (Trans::NoTrans, Trans::ConjTrans),
    (Trans::NoTrans, Trans::NoTrans),
];

fn check_copy_bitwise<T: Real>() {
    let detected = GemmPlan::<T>::new(CANDIDATE_POLICY);
    let mut portable = detected;
    portable.ctx = portable.ctx.with_portable_tier();
    let half = T::from_f64(-0.5).unwrap_or_else(T::zero);
    let quarter = T::from_f64(0.25).unwrap_or_else(T::zero);
    for plan in [&detected, &portable] {
        for (s, &size) in COPY_SIZES.iter().enumerate() {
            for &trans in &COPY_TRANS {
                for views in [(false, false), (true, false), (false, true), (true, true)] {
                    // NoTrans/NoTrans goes through the copy only for views.
                    if trans == (Trans::NoTrans, Trans::NoTrans) && views == (false, false) {
                        continue;
                    }
                    for (alpha, beta) in
                        [(T::one(), T::zero()), (half, T::one()), (T::one(), quarter)]
                    {
                        check_copy_vs_nn(plan, size, trans, views, alpha, beta, 3 * s + 1);
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_trans_copy_bitwise_matches_nn_f64() {
    check_copy_bitwise::<f64>();
}

#[test]
fn test_gemm_small_trans_copy_bitwise_matches_nn_f32() {
    check_copy_bitwise::<f32>();
}

#[test]
fn test_gemm_small_trans_copy_beta_zero_nan_c() {
    // beta == 0: C (all NaN) is overwritten without being read, on every
    // transposed combination and both element types.
    let plan64 = GemmPlan::<f64>::new(CANDIDATE_POLICY);
    let plan32 = GemmPlan::<f32>::new(CANDIDATE_POLICY);
    for &trans in &COPY_TRANS[..5] {
        for &size in &[(8usize, 8usize, 8usize), (17, 23, 19), (32, 32, 32)] {
            check_copy_vs_nn(&plan64, size, trans, (false, false), -2.0, 0.0, 7);
            check_copy_vs_nn(&plan32, size, trans, (true, true), 1.5, 0.0, 8);
        }
    }
}

#[test]
fn test_gemm_small_trans_copy_no_stale_scratch() {
    // Fill this thread's cached A and B packing buffers with NaN, then run
    // copies of decreasing size (and copies of only one operand): every
    // element the kernels read must have been written by the copy.
    fn poison<T: Field + GemmKernel + bytemuck::Zeroable>() {
        let nan = T::from_f64(f64::NAN).unwrap_or_else(T::zero);
        for slot in [PackSlot::A, PackSlot::B] {
            let mut buf = PackBuffer::<T>::take(slot, 64 * 64);
            buf.as_mut_slice().fill(nan);
        }
    }
    fn check<T: Real>() {
        let plan = GemmPlan::<T>::new(CANDIDATE_POLICY);
        let one = T::one();
        let beta = T::from_f64(0.5).unwrap_or_else(T::zero);
        for (s, &size) in [
            (32usize, 32usize, 32usize),
            (19, 11, 23),
            (8, 9, 8),
            (5, 3, 4),
        ]
        .iter()
        .enumerate()
        {
            poison::<T>();
            check_copy_vs_nn(
                &plan,
                size,
                (Trans::Trans, Trans::Trans),
                (false, false),
                one,
                beta,
                s,
            );
            poison::<T>();
            // Only B is copied (A is NoTrans and column-major).
            check_copy_vs_nn(
                &plan,
                size,
                (Trans::NoTrans, Trans::Trans),
                (false, false),
                one,
                T::zero(),
                s,
            );
            poison::<T>();
            // Only A is copied.
            check_copy_vs_nn(
                &plan,
                size,
                (Trans::ConjTrans, Trans::NoTrans),
                (false, false),
                one,
                beta,
                s,
            );
        }
    }
    check::<f64>();
    check::<f32>();
}

// ------------------------------------------------------------------
// AVX2 + FMA small kernels (`gemm_small_avx2.rs`) and their seam.
//
// Tests that need the kernels to execute start with the shared gate
// `avx2_fma_tier_or_skip`; the rule tests run on every target. The size
// grid m in 5..=40, n in 4..=13 hits every row tail (m mod 8 for f64,
// m mod 16 for f32) and every column tail (n mod 4).
// ------------------------------------------------------------------

use crate::level3::gemm_plan::{
    CANDIDATE_AARCH64, CANDIDATE_X86, LEGACY_POLICY, avx2_fma_tier_or_skip, choose_route,
};

/// Rows of the AVX2 size grid (every row-tail class of both tile heights).
const AVX2_M: core::ops::RangeInclusive<usize> = 5..=40;
/// Columns of the AVX2 size grid (every column-tail class).
const AVX2_N: core::ops::RangeInclusive<usize> = 4..=13;
/// Depths of the AVX2 size grid.
const AVX2_K: [usize; 6] = [1, 2, 3, 7, 16, 33];

/// Deterministic xorshift64 generator (no dependency).
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [-1, 1) with 53 random bits.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// An integer in `-r..=r`.
    fn int(&mut self, r: u64) -> f64 {
        (self.next() % (2 * r + 1)) as f64 - r as f64
    }
}

/// Column-major `rows × cols` operand with leading dimension `ld >= rows`,
/// stored in a slice of exactly `(cols - 1)·ld + rows` elements (no slack
/// after the last element; empty for an empty matrix). The `ld - rows` gap
/// cells between columns hold the guard NaN, so a kernel that used a value
/// outside the operand would produce a NaN.
struct Exact<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    ld: usize,
}

impl<T: Real> Exact<T> {
    fn new(rows: usize, cols: usize, ld: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let len = if rows == 0 || cols == 0 {
            0
        } else {
            (cols - 1) * ld + rows
        };
        let mut data = vec![T::guard(); len];
        for j in 0..cols {
            for i in 0..rows {
                data[i + j * ld] = f(i, j);
            }
        }
        Self {
            data,
            rows,
            cols,
            ld,
        }
    }

    fn view(&self) -> MatRef<'_, T> {
        MatRef::from_strided(&self.data, self.rows, self.cols, self.ld, 1)
            .expect("exact operand view")
    }

    fn at(&self, i: usize, j: usize) -> T {
        self.data[i + j * self.ld]
    }
}

/// `rows × cols` C inside a guard border of `top`/`bottom` rows and
/// `left`/`right` columns (all guard NaN); the leading dimension is
/// `top + rows + bottom`, so it equals `rows` without row borders.
struct Bordered<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    ld: usize,
    top: usize,
    left: usize,
}

impl<T: Real> Bordered<T> {
    fn new(
        (rows, cols): (usize, usize),
        (top, bottom, left, right): (usize, usize, usize, usize),
        mut f: impl FnMut(usize, usize) -> T,
    ) -> Self {
        let ld = top + rows + bottom;
        let mut data = vec![T::guard(); ld * (left + cols + right)];
        for j in 0..cols {
            for i in 0..rows {
                data[top + i + (left + j) * ld] = f(i, j);
            }
        }
        Self {
            data,
            rows,
            cols,
            ld,
            top,
            left,
        }
    }

    fn view_mut(&mut self) -> MatMut<'_, T> {
        let offset = self.top + self.left * self.ld;
        MatMut::from_strided(&mut self.data[offset..], self.rows, self.cols, self.ld)
            .expect("bordered C view")
    }

    fn at(&self, i: usize, j: usize) -> T {
        self.data[self.top + i + (self.left + j) * self.ld]
    }

    /// Every cell outside the C block still holds the guard bits.
    fn border_untouched(&self) -> bool {
        let guard = T::guard().bits();
        self.data.iter().enumerate().all(|(idx, v)| {
            let (i, j) = (idx % self.ld, idx / self.ld);
            let inside = (self.top..self.top + self.rows).contains(&i)
                && (self.left..self.left + self.cols).contains(&j);
            inside || v.bits() == guard
        })
    }
}

/// C after `gemm_small_with`'s beta step: zero (not read) for `beta == 0`,
/// unchanged for `beta == 1`, `c·beta` otherwise.
fn beta_applied<T: Real>(beta: T, c: T) -> T {
    if beta == T::zero() {
        T::zero()
    } else if beta == T::one() {
        c
    } else {
        c * beta
    }
}

/// The documented per-element sequence of the AVX2 + FMA small kernels:
/// `acc = fma(A[i, p], B[p, j], acc)` for `p` ascending from `+0`, then
/// `fma(acc, alpha, c)`; `k == 0` leaves `c` as it is.
fn fma_model<T: Real>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: T,
    i: usize,
    j: usize,
) -> T {
    let k = a.ncols();
    if k == 0 {
        return c;
    }
    let mut acc = T::zero();
    for p in 0..k {
        acc = a[(i, p)].fused(b[(p, j)], acc);
    }
    acc.fused(alpha, c)
}

/// The same bits, or both NaN (NaN payloads are not part of the contract).
fn same_value<T: Real>(x: T, y: T) -> bool {
    x.bits() == y.bits() || (x.not_a_number() && y.not_a_number())
}

/// Scalars of the AVX2 grid: (alpha, beta), beta == 0 with a NaN C.
fn avx2_scalars<T: Real>() -> [(T, T); 6] {
    let (one, zero) = (T::one(), T::zero());
    let (minus_two, half) = (T::of(-2.0), T::of(0.5));
    [
        (one, zero),
        (minus_two, zero),
        (one, one),
        (minus_two, one),
        (one, half),
        (minus_two, half),
    ]
}

/// The C border of case `idx`: with and without row borders (the latter
/// makes `ldc == m`), always with guard columns on both sides.
fn border_of(idx: usize) -> (usize, usize, usize, usize) {
    match idx % 3 {
        0 => (0, 0, 1, 1),
        1 => (1, 2, 1, 1),
        _ => (3, 1, 2, 1),
    }
}

/// `gemm_small_with` on the AVX2 + FMA tier with exact integer data:
/// every element exact, the border untouched, NaN C ignored for beta == 0.
fn check_avx2_exact_on_integers<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let ctx = GemmContext::detect(CANDIDATE_X86);
    let mut rng = XorShift(0x1234_5678_9abc_def1);
    let mut idx = 0usize;
    for m in AVX2_M {
        for n in AVX2_N {
            for &k in &AVX2_K {
                idx += 1;
                let (alpha, beta) = avx2_scalars::<T>()[idx % 6];
                let a = Exact::new(m, k, m + idx % 3, |_, _| T::of(rng.int(8)));
                let b = Exact::new(k, n, k + (idx / 3) % 2, |_, _| T::of(rng.int(8)));
                let c0: Vec<f64> = (0..m * n).map(|_| rng.int(8)).collect();
                let mut c = Bordered::new((m, n), border_of(idx), |i, j| {
                    if beta == T::zero() {
                        T::guard()
                    } else {
                        T::of(c0[i + j * m])
                    }
                });
                assert!(avx2_small_applies(&ctx.policy, m, n));
                gemm_small_with(ctx, alpha, &a.view(), &b.view(), beta, &mut c.view_mut());
                assert!(c.border_untouched(), "{test} {m}x{k}x{n}: border written");
                for j in 0..n {
                    for i in 0..m {
                        let mut dot = 0.0f64;
                        for p in 0..k {
                            dot += a.at(i, p).wide() * b.at(p, j).wide();
                        }
                        let c_part = if beta == T::zero() {
                            0.0
                        } else {
                            beta.wide() * c0[i + j * m]
                        };
                        let expected = alpha.wide() * dot + c_part;
                        assert_eq!(
                            c.at(i, j).wide(),
                            expected,
                            "{test} {m}x{k}x{n} alpha={alpha} beta={beta} ({i},{j})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_avx2_exact_on_integers_f64() {
    check_avx2_exact_on_integers::<f64>("test_gemm_small_avx2_exact_on_integers_f64");
}

#[test]
fn test_gemm_small_avx2_exact_on_integers_f32() {
    check_avx2_exact_on_integers::<f32>("test_gemm_small_avx2_exact_on_integers_f32");
}

/// The seam takes every shape of the grid on the AVX2 + FMA tier, and each
/// element is bit for bit the documented fused sequence (random data, exact
/// operand slices, C inside a guard border). An Inf planted in A must reach
/// exactly the elements of its row.
fn check_avx2_matches_fma_model<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let ctx = GemmContext::detect(CANDIDATE_X86);
    let mut rng = XorShift(0x0f0f_1e1e_2d2d_3c3c);
    let mut idx = 0usize;
    for m in AVX2_M {
        for n in AVX2_N {
            for &k in &AVX2_K {
                idx += 1;
                let alpha = T::of(rng.unit() * 3.0);
                let inf_row = idx.is_multiple_of(7).then_some(idx % m);
                let a = Exact::new(m, k, m + idx % 2, |i, p| {
                    if Some(i) == inf_row && p == k / 2 {
                        T::of(f64::INFINITY)
                    } else {
                        T::of(rng.unit())
                    }
                });
                let b = Exact::new(k, n, k + (idx / 2) % 3, |_, _| T::of(rng.unit()));
                let c0: Vec<T> = (0..m * n).map(|_| T::of(rng.unit())).collect();
                let mut c = Bordered::new((m, n), border_of(idx), |i, j| c0[i + j * m]);
                let taken = T::simd(ctx, alpha, &a.view(), &b.view(), &mut c.view_mut());
                assert!(taken, "{test} {m}x{k}x{n}: the AVX2 kernel did not run");
                assert!(c.border_untouched(), "{test} {m}x{k}x{n}: border written");
                for j in 0..n {
                    for i in 0..m {
                        let want = fma_model(alpha, &a.view(), &b.view(), c0[i + j * m], i, j);
                        let got = c.at(i, j);
                        assert!(
                            same_value(got, want),
                            "{test} {m}x{k}x{n} ({i},{j}): got {got} model {want}"
                        );
                        assert_eq!(got.finite(), Some(i) != inf_row, "{test} ({i},{j})");
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_avx2_matches_fma_model_f64() {
    check_avx2_matches_fma_model::<f64>("test_gemm_small_avx2_matches_fma_model_f64");
}

#[test]
fn test_gemm_small_avx2_matches_fma_model_f32() {
    check_avx2_matches_fma_model::<f32>("test_gemm_small_avx2_matches_fma_model_f32");
}

/// `a·b` as an unevaluated sum `hi + lo` (exact).
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// `a + b` as an unevaluated sum `hi + lo` (exact).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `|C − C_ref| ≤ γ_{k+2}·(|α|·|A|·|B| + |β|·|C₀|)` on random data, with
/// the reference accumulated in double-double arithmetic (about 106 bits,
/// so its own error is far below the bound for both element types).
fn check_avx2_error_bound<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let ctx = GemmContext::detect(CANDIDATE_X86);
    let u = T::unit_roundoff();
    let mut rng = XorShift(0x5151_a2a2_f3f3_0404);
    for m in AVX2_M {
        for n in [4usize, 5, 7, 13] {
            for k in [1usize, 7, 33, 100] {
                let alpha = T::of(-1.7);
                let beta = T::of(0.3);
                let a = Exact::new(m, k, m, |_, _| T::of(rng.unit()));
                let b = Exact::new(k, n, k, |_, _| T::of(rng.unit()));
                let c0: Vec<T> = (0..m * n).map(|_| T::of(rng.unit())).collect();
                let mut c = Bordered::new((m, n), (1, 1, 1, 1), |i, j| c0[i + j * m]);
                gemm_small_with(ctx, alpha, &a.view(), &b.view(), beta, &mut c.view_mut());
                assert!(c.border_untouched(), "{test}: border written");
                let kf = (k + 2) as f64;
                let gamma = kf * u / (1.0 - kf * u);
                for j in 0..n {
                    for i in 0..m {
                        let (mut hi, mut lo) = (0.0f64, 0.0f64);
                        let mut mag = 0.0f64;
                        for p in 0..k {
                            let (x, y) = (a.at(i, p).wide(), b.at(p, j).wide());
                            let (ph, pl) = two_prod(x, y);
                            let (s, e) = two_sum(hi, ph);
                            hi = s;
                            lo += e + pl;
                            mag += (x * y).abs();
                        }
                        let (ah, al) = two_prod(alpha.wide(), hi);
                        let (bh, bl) = two_prod(beta.wide(), c0[i + j * m].wide());
                        let (rh, re) = two_sum(ah, bh);
                        let rl = re + al + bl + alpha.wide() * lo;
                        let err = ((c.at(i, j).wide() - rh) - rl).abs();
                        let bound = gamma
                            * (alpha.wide().abs() * mag
                                + beta.wide().abs() * c0[i + j * m].wide().abs());
                        assert!(
                            err <= bound * (1.0 + 1e-6) + f64::MIN_POSITIVE,
                            "{test} {m}x{k}x{n} ({i},{j}): error {err:e} > bound {bound:e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_avx2_error_bound_f64() {
    check_avx2_error_bound::<f64>("test_gemm_small_avx2_error_bound_f64");
}

#[test]
fn test_gemm_small_avx2_error_bound_f32() {
    check_avx2_error_bound::<f32>("test_gemm_small_avx2_error_bound_f32");
}

/// The bits of `rows × cols` C at rows `r0..`, columns `c0..` of `c`.
fn block_bits<T: Real>(c: &Mat<T>, (r0, c0): (usize, usize), rows: usize, cols: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            out.push(c[(r0 + i, c0 + j)].bits());
        }
    }
    out
}

/// Random `rows × cols` matrix in [-1, 1) from `seed`.
fn pattern_random<T: Real>(rows: usize, cols: usize, seed: u64) -> Mat<T> {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x2545_f491_4f6c_dd1d));
    let mut out: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            out[(i, j)] = T::of(rng.unit());
        }
    }
    out
}

/// Position independence on the AVX2 + FMA tier: the same logical product,
/// computed as a block of a larger product whose rows start at offset `r`
/// (shifting every element across lanes, tiles and row-tail classes) and
/// whose columns start at offset `s` (shifting across column panels), and
/// through `gemm_small_trans_copy` from transposed storage, gives the bits
/// of the stand-alone product.
fn check_avx2_position_independent<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let ctx = GemmContext::detect(CANDIDATE_X86);
    let (alpha, beta) = (T::of(-0.75), T::of(1.25));
    for (case, &(m, k, n)) in [
        (13usize, 9usize, 11usize),
        (29, 7, 6),
        (6, 16, 5),
        (40, 3, 13),
    ]
    .iter()
    .enumerate()
    {
        let op_a = pattern_random::<T>(m, k, 10 * case as u64 + 1);
        let op_b = pattern_random::<T>(k, n, 10 * case as u64 + 2);
        let c0 = pattern_random::<T>(m, n, 10 * case as u64 + 3);
        let mut alone = c0.clone();
        gemm_small_with(
            ctx,
            alpha,
            &op_a.as_ref(),
            &op_b.as_ref(),
            beta,
            &mut alone.as_mut(),
        );
        let want = block_bits(&alone, (0, 0), m, n);
        let ctx_msg = format!("{test} {m}x{k}x{n}");

        for r in 0..T::AVX2_TILE_ROWS {
            for s in 0..4 {
                let big_m = r + m + 3;
                let big_n = s + n + 2;
                let mut big_a = pattern_random::<T>(big_m, k, 77 + r as u64);
                let mut big_b = pattern_random::<T>(k, big_n, 99 + s as u64);
                let mut big_c = pattern_random::<T>(big_m, big_n, 55);
                for p in 0..k {
                    for i in 0..m {
                        big_a[(r + i, p)] = op_a[(i, p)];
                    }
                    for j in 0..n {
                        big_b[(p, s + j)] = op_b[(p, j)];
                    }
                }
                for j in 0..n {
                    for i in 0..m {
                        big_c[(r + i, s + j)] = c0[(i, j)];
                    }
                }
                gemm_small_with(
                    ctx,
                    alpha,
                    &big_a.as_ref(),
                    &big_b.as_ref(),
                    beta,
                    &mut big_c.as_mut(),
                );
                assert!(
                    block_bits(&big_c, (r, s), m, n) == want,
                    "{ctx_msg}: block at ({r},{s}) of {big_m}x{k}x{big_n} differs"
                );
            }
        }

        let a_t = transpose_copy(&op_a);
        let b_t = transpose_copy(&op_b);
        for (ta, tb) in [
            (Trans::Trans, Trans::NoTrans),
            (Trans::NoTrans, Trans::Trans),
            (Trans::Trans, Trans::Trans),
        ] {
            let a_store = if ta == Trans::NoTrans { &op_a } else { &a_t };
            let b_store = if tb == Trans::NoTrans { &op_b } else { &b_t };
            let mut c = c0.clone();
            let done = gemm_small_trans_copy(
                ctx,
                ta,
                tb,
                alpha,
                &a_store.as_ref(),
                &b_store.as_ref(),
                beta,
                &mut c.as_mut(),
            );
            assert!(done, "{ctx_msg} {ta:?}/{tb:?}: copy declined");
            assert!(
                block_bits(&c, (0, 0), m, n) == want,
                "{ctx_msg} {ta:?}/{tb:?}: copy route differs"
            );
        }
    }
}

#[test]
fn test_gemm_small_avx2_position_independent_f64() {
    check_avx2_position_independent::<f64>("test_gemm_small_avx2_position_independent_f64");
}

#[test]
fn test_gemm_small_avx2_position_independent_f32() {
    check_avx2_position_independent::<f32>("test_gemm_small_avx2_position_independent_f32");
}

/// On every tier and policy: the same logical product on operands and C
/// that are sub-views at different base offsets of larger storage gives
/// the same bits (the kernels' tiling is relative to the view).
fn check_views_at_offsets<T: Real>() {
    let (alpha, beta) = (T::of(1.5), T::of(-0.5));
    for policy in [LEGACY_POLICY, CANDIDATE_POLICY, CANDIDATE_X86] {
        let ctx = GemmContext::detect(policy);
        for &(m, k, n) in &[(13usize, 9usize, 11usize), (5, 4, 4), (17, 3, 9), (8, 8, 8)] {
            let op_a = pattern_random::<T>(m, k, 1);
            let op_b = pattern_random::<T>(k, n, 2);
            let c0 = pattern_random::<T>(m, n, 3);
            let mut alone = c0.clone();
            gemm_small_with(
                ctx,
                alpha,
                &op_a.as_ref(),
                &op_b.as_ref(),
                beta,
                &mut alone.as_mut(),
            );
            let want = block_bits(&alone, (0, 0), m, n);
            for (r, s) in [(1usize, 0usize), (3, 2), (7, 5), (9, 1)] {
                let mut sa = pattern_random::<T>(m + r + 2, k + s, 4);
                let mut sb = pattern_random::<T>(k + r, n + s + 1, 5);
                let mut sc = pattern_random::<T>(m + s + 1, n + r, 6);
                for p in 0..k {
                    for i in 0..m {
                        sa[(r + i, s + p)] = op_a[(i, p)];
                    }
                    for j in 0..n {
                        sb[(r + p, s + j)] = op_b[(p, j)];
                    }
                }
                for j in 0..n {
                    for i in 0..m {
                        sc[(s + i, r + j)] = c0[(i, j)];
                    }
                }
                let a_view = sa.as_ref().submatrix(r, s, m, k);
                let b_view = sb.as_ref().submatrix(r, s, k, n);
                let mut c_view = sc.as_mut().submatrix(s, r, m, n);
                gemm_small_with(ctx, alpha, &a_view, &b_view, beta, &mut c_view);
                assert!(
                    block_bits(&sc, (s, r), m, n) == want,
                    "{} {m}x{k}x{n} at ({r},{s}) {policy:?}",
                    core::any::type_name::<T>()
                );
            }
        }
    }
}

#[test]
fn test_gemm_small_position_independent_views_f64() {
    check_views_at_offsets::<f64>();
}

#[test]
fn test_gemm_small_position_independent_views_f32() {
    check_views_at_offsets::<f32>();
}

/// Signature of the AVX2 + FMA small kernels.
#[cfg(target_arch = "x86_64")]
type Avx2Kernel<T> =
    unsafe fn(usize, usize, usize, T, *const T, usize, *const T, usize, *mut T, usize);

/// `k == 0` leaves C untouched (also `-0.0`), and rows below the seam's
/// `m >= 5` and narrow panels run on the kernels directly, with operand
/// slices of exactly the operand's size (zero-length for `k == 0`) and
/// `ldc == m`.
#[cfg(target_arch = "x86_64")]
fn check_avx2_kernel_edges<T: Real>(test: &str, kernel: Avx2Kernel<T>) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let mut rng = XorShift(0x7777_1111_3333_9999);
    for m in 1..=2 * T::AVX2_TILE_ROWS + 1 {
        for n in 1..=5 {
            for k in [0usize, 1, 2, 5] {
                let alpha = T::of(rng.unit() * 2.0);
                let a = Exact::new(m, k, m, |_, _| T::of(rng.unit()));
                let b = Exact::new(k, n, k, |_, _| T::of(rng.unit()));
                let c0: Vec<T> = (0..m * n)
                    .map(|idx| {
                        if idx.is_multiple_of(3) {
                            T::of(-0.0)
                        } else {
                            T::of(rng.unit())
                        }
                    })
                    .collect();
                let mut c = Bordered::new((m, n), (0, 0, 1, 1), |i, j| c0[i + j * m]);
                let mut c_view = c.view_mut();
                let c_ptr = c_view.as_mut_ptr();
                // SAFETY: the gate above put this run on the AVX2 + FMA tier
                // (both features present); `a` is m × k (ld m), `b` is k × n
                // (ld k) and `c_view` is m × n (ld m) exactly, so every element
                // the kernel may access lies in these allocations, and C is
                // borrowed exclusively.
                unsafe {
                    kernel(
                        m,
                        n,
                        k,
                        alpha,
                        a.data.as_ptr(),
                        m,
                        b.data.as_ptr(),
                        k,
                        c_ptr,
                        m,
                    );
                }
                assert!(c.border_untouched(), "{test} {m}x{k}x{n}: border written");
                for j in 0..n {
                    for i in 0..m {
                        let want = fma_model(alpha, &a.view(), &b.view(), c0[i + j * m], i, j);
                        let got = c.at(i, j);
                        assert!(
                            got.bits() == want.bits(),
                            "{test} {m}x{k}x{n} ({i},{j}): got {got} model {want}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_small_avx2_kernel_edges_f64() {
    check_avx2_kernel_edges::<f64>(
        "test_gemm_small_avx2_kernel_edges_f64",
        avx2::small_f64_avx2,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_small_avx2_kernel_edges_f32() {
    check_avx2_kernel_edges::<f32>(
        "test_gemm_small_avx2_kernel_edges_f32",
        avx2::small_f32_avx2,
    );
}

/// The kernels on operands that end exactly at the end of their
/// allocations: A (`m × k`, ld `lda`), B (`k × n`, ld `ldb`) and C (`m × n`,
/// ld `ldc`) are slices of exactly `(cols − 1)·ld + rows` elements, with no
/// guard column after C, for `ld == rows` and `ld > rows` of each operand
/// (both kinds together and mixed). Every row tail of both tile modes (`m`
/// in `1..=tile + 1`) and every panel width (`n` in `1..=5`) is covered, so
/// the last column of C is reached in every tile mode and panel. Every
/// element is the documented fused sequence bit for bit, and the gap cells
/// between C's columns (guard NaN) stay untouched.
///
/// Under Miri a full-width load or store past the last live row of a tile
/// in the last column (a masked access that became a plain one) leaves the
/// allocation and is reported as undefined behaviour:
///
/// ```text
/// RUSTFLAGS="-C target-feature=+avx2,+fma" OXIBLAS_REQUIRE_AVX2=1 \
///   MIRIFLAGS="-Zmiri-env-forward=OXIBLAS_REQUIRE_AVX2" \
///   cargo +nightly miri test -p oxiblas-blas --target x86_64-unknown-linux-gnu \
///   --lib test_gemm_small_avx2_exact_alloc
/// ```
#[cfg(target_arch = "x86_64")]
fn check_avx2_exact_alloc<T: Real>(test: &str, kernel: Avx2Kernel<T>) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let mut rng = XorShift(0x5151_2323_7777_0101);
    let alpha = T::of(1.5);
    // (lda − m, ldb − k, ldc − m): all tight, all padded, and mixed.
    let pads = [(0usize, 0usize, 0usize), (3, 2, 1), (0, 3, 2), (2, 0, 0)];
    for m in 1..=T::AVX2_TILE_ROWS + 1 {
        for n in 1..=5usize {
            for k in [1usize, 2] {
                for (pa, pb, pc) in pads {
                    let (lda, ldb, ldc) = (m + pa, k + pb, m + pc);
                    let ctx = format!("{test} {m}x{k}x{n} lda={lda} ldb={ldb} ldc={ldc}");
                    let a = Exact::new(m, k, lda, |_, _| T::of(rng.unit()));
                    let b = Exact::new(k, n, ldb, |_, _| T::of(rng.unit()));
                    let c0 = Exact::new(m, n, ldc, |_, _| T::of(rng.unit()));
                    assert_eq!(a.data.len(), (k - 1) * lda + m);
                    assert_eq!(b.data.len(), (n - 1) * ldb + k);
                    assert_eq!(c0.data.len(), (n - 1) * ldc + m);
                    let mut c = c0.data.clone();
                    // SAFETY: the gate above put this run on the AVX2 + FMA
                    // tier (both features present); `a`, `b` and `c` hold
                    // every element of the m × k, k × n and m × n operands at
                    // their leading dimensions (the slices end at the last
                    // element of each), and `c` is a distinct, exclusively
                    // borrowed allocation.
                    unsafe {
                        kernel(
                            m,
                            n,
                            k,
                            alpha,
                            a.data.as_ptr(),
                            lda,
                            b.data.as_ptr(),
                            ldb,
                            c.as_mut_ptr(),
                            ldc,
                        );
                    }
                    for (idx, (&got, &before)) in c.iter().zip(&c0.data).enumerate() {
                        let (i, j) = (idx % ldc, idx / ldc);
                        if i < m {
                            let want = fma_model(alpha, &a.view(), &b.view(), before, i, j);
                            assert!(
                                got.bits() == want.bits(),
                                "{ctx} ({i},{j}): got {got} model {want}"
                            );
                        } else {
                            assert!(
                                got.bits() == before.bits(),
                                "{ctx}: gap cell ({i},{j}) written"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_small_avx2_exact_alloc_f64() {
    // Miri command: see `check_avx2_exact_alloc`.
    check_avx2_exact_alloc::<f64>("test_gemm_small_avx2_exact_alloc_f64", avx2::small_f64_avx2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_small_avx2_exact_alloc_f32() {
    // Miri command: see `check_avx2_exact_alloc`.
    check_avx2_exact_alloc::<f32>("test_gemm_small_avx2_exact_alloc_f32", avx2::small_f32_avx2);
}

/// Signed zeros and subnormals through the AVX2 seam, bit for bit against
/// the documented sequence (`acc` starts at `+0`): every product `−0` (A of
/// `+0` or `−0` against a negative B) with C of `−0.0` gives `+0` there
/// (`+0 + −0`), where an accumulator started at `−0` would give `−0`;
/// subnormal A, and C alternating `−0.0` and the smallest subnormal.
fn check_avx2_special_values<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let ctx = GemmContext::detect(CANDIDATE_X86);
    // The smallest positive subnormal of T.
    let tiny = if T::AVX2_TILE_ROWS == 8 {
        f64::from_bits(1)
    } else {
        f64::from(f32::from_bits(1))
    };
    assert!(
        T::of(tiny) != T::zero(),
        "{test}: tiny is not a subnormal of T"
    );
    for (m, k, n) in [(13usize, 3usize, 6usize), (5, 1, 4), (21, 2, 9)] {
        for case in 0..4 {
            let a = Exact::new(m, k, m, |i, p| match case {
                0 => T::of(0.0),
                1 => T::of(-0.0),
                2 => T::of(tiny * (1 + i + p) as f64),
                _ => {
                    if (i + p) % 2 == 0 {
                        T::of(0.0)
                    } else {
                        T::of(-0.0)
                    }
                }
            });
            let b = Exact::new(k, n, k, |p, j| match case {
                0 | 1 => T::of(-1.0 - (p + j) as f64),
                2 => T::of(0.5),
                _ => T::of(if j % 2 == 0 { 1.0 } else { -1.0 }),
            });
            for alpha in [T::of(1.0), T::of(-2.0), T::of(0.75)] {
                let c0: Vec<T> = (0..m * n)
                    .map(|idx| {
                        if idx % 2 == 0 {
                            T::of(-0.0)
                        } else {
                            T::of(tiny)
                        }
                    })
                    .collect();
                let mut c = Bordered::new((m, n), (0, 0, 1, 1), |i, j| c0[i + j * m]);
                let taken = T::simd(ctx, alpha, &a.view(), &b.view(), &mut c.view_mut());
                assert!(taken, "{test} {m}x{k}x{n}: the AVX2 kernel did not run");
                assert!(c.border_untouched(), "{test} {m}x{k}x{n}: border written");
                for j in 0..n {
                    for i in 0..m {
                        let want = fma_model(alpha, &a.view(), &b.view(), c0[i + j * m], i, j);
                        let got = c.at(i, j);
                        assert!(
                            got.bits() == want.bits(),
                            "{test} {m}x{k}x{n} case {case} alpha={alpha} ({i},{j}): \
                             got {got} model {want}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_avx2_special_values_f64() {
    check_avx2_special_values::<f64>("test_gemm_small_avx2_special_values_f64");
}

#[test]
fn test_gemm_small_avx2_special_values_f32() {
    check_avx2_special_values::<f32>("test_gemm_small_avx2_special_values_f32");
}

/// The legacy policy keeps the 0.2.3 portable loops on the AVX2 + FMA tier
/// (`small_simd_x86 == false`): the seam declines, and `gemm_small_with`
/// gives the bits of the portable tier. The public `gemm_small` follows the
/// build's default policy.
fn check_avx2_legacy_portable<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let legacy = GemmContext::detect(LEGACY_POLICY);
    let portable = legacy.with_portable_tier();
    let candidate = GemmContext::detect(CANDIDATE_X86);
    let (alpha, beta) = (T::of(0.7), T::of(-1.3));
    for &(m, k, n) in &[
        (5usize, 7usize, 4usize),
        (13, 9, 11),
        (32, 32, 32),
        (17, 33, 6),
    ] {
        let a = pattern_random::<T>(m, k, 11);
        let b = pattern_random::<T>(k, n, 12);
        let c0 = pattern_random::<T>(m, n, 13);
        let mut c = c0.clone();
        assert!(!T::simd(
            legacy,
            alpha,
            &a.as_ref(),
            &b.as_ref(),
            &mut c.as_mut()
        ));
        assert!(bits_of(&c) == bits_of(&c0), "{test}: declined seam wrote C");

        let mut c_legacy = c0.clone();
        gemm_small_with(
            legacy,
            alpha,
            &a.as_ref(),
            &b.as_ref(),
            beta,
            &mut c_legacy.as_mut(),
        );
        let mut c_portable = c0.clone();
        gemm_small_with(
            portable,
            alpha,
            &a.as_ref(),
            &b.as_ref(),
            beta,
            &mut c_portable.as_mut(),
        );
        assert!(
            bits_of(&c_legacy) == bits_of(&c_portable),
            "{test} {m}x{k}x{n}: legacy left the portable loops"
        );
        let mut c_candidate = c0.clone();
        gemm_small_with(
            candidate,
            alpha,
            &a.as_ref(),
            &b.as_ref(),
            beta,
            &mut c_candidate.as_mut(),
        );
        assert!(
            bits_of(&c_candidate) != bits_of(&c_portable),
            "{test} {m}x{k}x{n}: the candidate gave the portable bits"
        );
        let mut c_public = c0.clone();
        gemm_small(
            alpha,
            &a.as_ref(),
            &b.as_ref(),
            beta,
            &mut c_public.as_mut(),
        );
        let expected = if DEFAULT_POLICY.small_simd_x86 {
            &c_candidate
        } else {
            &c_portable
        };
        assert!(
            bits_of(&c_public) == bits_of(expected),
            "{test} {m}x{k}x{n}: gemm_small does not follow the default policy"
        );
    }
    assert_eq!(DEFAULT_POLICY.small_simd_x86, cfg!(oxiblas_gemm_candidate));
}

#[test]
fn test_gemm_small_avx2_legacy_keeps_portable_loops_f64() {
    check_avx2_legacy_portable::<f64>("test_gemm_small_avx2_legacy_keeps_portable_loops_f64");
}

#[test]
fn test_gemm_small_avx2_legacy_keeps_portable_loops_f32() {
    check_avx2_legacy_portable::<f32>("test_gemm_small_avx2_legacy_keeps_portable_loops_f32");
}

/// A column-major `NoTrans`/`NoTrans` f64/f32 problem for the router.
fn nn_inputs((m, k, n): (usize, usize, usize), tier: SimdTier) -> RouteInputs {
    RouteInputs {
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
        tier,
    }
}

#[test]
fn test_gemm_small_simd_route_threshold() {
    // The seam's AVX2 rule is the router's: for every policy and shape,
    // `avx2_small_applies` agrees with `RouteInputs::simd_small` on the
    // AVX2 + FMA tier (f64/f32), so a problem routed to the small path for
    // its SIMD kernels gets them.
    for policy in [
        LEGACY_POLICY,
        CANDIDATE_X86,
        CANDIDATE_AARCH64,
        CANDIDATE_POLICY,
    ] {
        for m in 0..=12 {
            for n in 0..=12 {
                let inp = nn_inputs((m, 8, n), SimdTier::Avx2Fma);
                assert_eq!(
                    avx2_small_applies(&policy, m, n),
                    inp.simd_small(&policy),
                    "{m}x{n} {policy:?}"
                );
            }
        }
    }
    assert!(!avx2_small_applies(&LEGACY_POLICY, 40, 40));
    assert!(avx2_small_applies(&CANDIDATE_X86, 5, 4));
    assert!(!avx2_small_applies(&CANDIDATE_X86, 4, 4));
    assert!(!avx2_small_applies(&CANDIDATE_X86, 5, 3));

    // Thresholds: 32³ stays the public small threshold; the x86 candidate
    // raises the AVX2 + FMA ceiling to 80³ for column-major NoTrans/NoTrans.
    assert_eq!(SMALL_THRESHOLD, 32 * 32 * 32);
    assert_eq!(LEGACY_POLICY.small_simd_max_work, SMALL_THRESHOLD);
    assert_eq!(CANDIDATE_X86.small_simd_max_work, 80 * 80 * 80);
    let route = |shape, policy: &GemmPolicy, tier| choose_route(&nn_inputs(shape, tier), policy);
    let avx2 = SimdTier::Avx2Fma;
    assert_eq!(route((80, 80, 80), &CANDIDATE_X86, avx2), GemmRoute::Small);
    assert_eq!(
        route((81, 80, 80), &CANDIDATE_X86, avx2),
        GemmRoute::Blocked
    );
    assert_eq!(
        route((5, 25_600, 4), &CANDIDATE_X86, avx2),
        GemmRoute::Small
    );
    assert_eq!(
        route((5, 25_601, 4), &CANDIDATE_X86, avx2),
        GemmRoute::Blocked
    );
    // m == 4 has no AVX2 small kernel: the 32³ ceiling applies.
    assert_eq!(route((4, 2048, 4), &CANDIDATE_X86, avx2), GemmRoute::Small);
    assert_eq!(
        route((4, 2049, 4), &CANDIDATE_X86, avx2),
        GemmRoute::Blocked
    );
    assert_eq!(route((32, 32, 32), &LEGACY_POLICY, avx2), GemmRoute::Small);
    assert_eq!(
        route((33, 32, 32), &LEGACY_POLICY, avx2),
        GemmRoute::Blocked
    );
    for tier in [SimdTier::Sse42, SimdTier::Portable] {
        assert_eq!(
            route((33, 32, 32), &CANDIDATE_X86, tier),
            GemmRoute::Blocked
        );
        assert_eq!(route((32, 32, 32), &CANDIDATE_X86, tier), GemmRoute::Small);
    }
}

#[cfg(feature = "max-simd-128")]
#[test]
fn test_gemm_small_max_simd_128_portable_and_32_cubed() {
    // A 128-bit SIMD cap keeps every AVX2 path out: the tier is never
    // AVX2 + FMA (even with AVX2 and FMA advertised), so on x86-64 the seam
    // declines and the small path keeps the 32³ ceiling.
    let ctx = GemmContext::detect(CANDIDATE_POLICY);
    assert_ne!(ctx.tier(), SimdTier::Avx2Fma);
    let a = pattern::<f64>(8, 8, 1);
    let b = pattern::<f64>(8, 8, 2);
    let mut c: Mat<f64> = Mat::zeros(8, 8);
    let taken = small_simd_f64(
        GemmContext::detect(CANDIDATE_X86),
        1.0,
        &a.as_ref(),
        &b.as_ref(),
        &mut c.as_mut(),
    );
    if cfg!(target_arch = "x86_64") {
        assert!(!taken, "an AVX2 small kernel ran under max-simd-128");
        assert!(bits_of(&c) == bits_of(&Mat::<f64>::zeros(8, 8)));
    }
    let tier = ctx.tier();
    let route = |shape| choose_route(&nn_inputs(shape, tier), &CANDIDATE_POLICY);
    assert_eq!(route((32, 32, 32)), GemmRoute::Small);
    assert_eq!(route((33, 32, 32)), GemmRoute::Blocked);
}

/// `gemm_small_trans_copy` on the AVX2 + FMA tier over the tail grid: bit
/// for bit the NoTrans/NoTrans small path (through `gemm_planned` with the
/// routes forced) and the documented fused sequence on the values of op(A)
/// and op(B), with C never read when beta == 0.
fn check_trans_copy_avx2<T: Real>(test: &str) {
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let plan = GemmPlan::<T>::new(CANDIDATE_X86);
    assert_eq!(plan.ctx.tier(), SimdTier::Avx2Fma);
    // Alphas with inexact products, so an unfused epilogue would show.
    let scalars = [
        (-0.7, 0.0),
        (1.1, 0.0),
        (1.3, 1.0),
        (-2.0, 1.0),
        (0.9, 0.5),
        (-1.7, 0.3),
    ]
    .map(|(alpha, beta)| (T::of(alpha), T::of(beta)));
    let mut idx = 0usize;
    for m in AVX2_M {
        for n in AVX2_N {
            idx += 1;
            let k = AVX2_K[idx % AVX2_K.len()];
            let trans = COPY_TRANS[idx % 5];
            let views = [(false, false), (true, false), (false, true), (true, true)][idx % 4];
            let (alpha, beta) = scalars[idx % scalars.len()];
            check_copy_vs_nn(&plan, (m, k, n), trans, views, alpha, beta, idx);

            // The copy route itself against the fused model.
            let op_a = pattern_random::<T>(m, k, idx as u64);
            let op_b = pattern_random::<T>(k, n, idx as u64 + 1);
            let c0 = pattern_random::<T>(m, n, idx as u64 + 2);
            let a_store = storage(&op_a, trans.0, views.0);
            let b_store = storage(&op_b, trans.1, views.1);
            let (a, b) = (as_operand(&a_store, views.0), as_operand(&b_store, views.1));
            let mut c = if beta == T::zero() {
                Mat::filled(m, n, T::guard())
            } else {
                c0.clone()
            };
            let done = gemm_small_trans_copy(
                plan.ctx,
                trans.0,
                trans.1,
                alpha,
                &a,
                &b,
                beta,
                &mut c.as_mut(),
            );
            assert!(done, "{test} {m}x{k}x{n}: copy declined");
            for j in 0..n {
                for i in 0..m {
                    let c_in = beta_applied(beta, c0[(i, j)]);
                    let want = fma_model(alpha, &op_a.as_ref(), &op_b.as_ref(), c_in, i, j);
                    assert!(
                        c[(i, j)].bits() == want.bits(),
                        "{test} {m}x{k}x{n} {trans:?} views={views:?} ({i},{j})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_gemm_small_trans_copy_avx2_bitwise_f64() {
    check_trans_copy_avx2::<f64>("test_gemm_small_trans_copy_avx2_bitwise_f64");
}

#[test]
fn test_gemm_small_trans_copy_avx2_bitwise_f32() {
    check_trans_copy_avx2::<f32>("test_gemm_small_trans_copy_avx2_bitwise_f32");
}
