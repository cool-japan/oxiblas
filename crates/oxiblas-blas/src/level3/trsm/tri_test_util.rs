//! Shared fixtures and oracles for the blocked TRSM/TRMM tests.

use super::types::{Diag, Side, Trans, Uplo};
use crate::level3::gemm_kernel::GemmKernel;
use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;

pub(super) const SIDES: [Side; 2] = [Side::Left, Side::Right];
pub(super) const UPLOS: [Uplo; 2] = [Uplo::Lower, Uplo::Upper];
pub(super) const TRANSES: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
pub(super) const DIAGS: [Diag; 2] = [Diag::NonUnit, Diag::Unit];

/// Every `(side, uplo, trans, diag)` combination (24).
pub(super) fn all_flags() -> Vec<(Side, Uplo, Trans, Diag)> {
    let mut v = Vec::new();
    for side in SIDES {
        for uplo in UPLOS {
            for trans in TRANSES {
                for diag in DIAGS {
                    v.push((side, uplo, trans, diag));
                }
            }
        }
    }
    v
}

/// Element types exercised by the tests, with conversions to `Complex64`
/// for oracles evaluated in double precision.
pub(super) trait TestScalar: Field + GemmKernel + bytemuck::Zeroable {
    /// Relative tolerance per unit of problem order.
    const TOL: f64;
    const COMPLEX: bool;
    fn from_parts(re: f64, im: f64) -> Self;
    fn to_c64(self) -> Complex64;
    fn bits(self) -> (u64, u64);
}

impl TestScalar for f64 {
    const TOL: f64 = 1e-13;
    const COMPLEX: bool = false;
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl TestScalar for f32 {
    const TOL: f64 = 5e-6;
    const COMPLEX: bool = false;
    fn from_parts(re: f64, _im: f64) -> Self {
        re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
}

impl TestScalar for Complex64 {
    const TOL: f64 = 1e-13;
    const COMPLEX: bool = true;
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn to_c64(self) -> Complex64 {
        self
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

impl TestScalar for Complex32 {
    const TOL: f64 = 5e-6;
    const COMPLEX: bool = true;
    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
}

/// Small deterministic LCG so the fixtures are reproducible without a RNG crate.
pub(super) struct Lcg(pub(super) u64);

impl Lcg {
    /// Next value in `[-0.5, 0.5)`.
    pub(super) fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    pub(super) fn scalar<T: TestScalar>(&mut self) -> T {
        let re = self.unit();
        let im = if T::COMPLEX { self.unit() } else { 0.0 };
        T::from_parts(re, im)
    }
}

/// Triangular fixture of order `s`: a dominant diagonal (`|d| ∈ [1, 1.5]`,
/// genuinely complex for complex types), small entries in the `uplo`
/// triangle and `junk` everywhere else (it must never be read).
pub(super) fn tri_fixture<T: TestScalar>(s: usize, uplo: Uplo, rng: &mut Lcg, junk: T) -> Mat<T> {
    let mut a: Mat<T> = Mat::zeros(s, s);
    let scale = 2.0 / (s.max(1) as f64);
    for j in 0..s {
        for i in 0..s {
            a[(i, j)] = if i == j {
                let re = 1.0 + rng.unit().abs();
                let im = if T::COMPLEX { 0.5 * rng.unit() } else { 0.0 };
                let sign = if (i * 7 + 3) % 5 == 0 { -1.0 } else { 1.0 };
                T::from_parts(sign * re, im)
            } else if (uplo == Uplo::Lower) == (i > j) {
                let re = rng.unit() * scale;
                let im = if T::COMPLEX { rng.unit() * scale } else { 0.0 };
                T::from_parts(re, im)
            } else {
                junk
            };
        }
    }
    a
}

/// Dense `m x n` fixture with entries in `[-0.5, 0.5)`.
pub(super) fn dense_fixture<T: TestScalar>(m: usize, n: usize, rng: &mut Lcg) -> Mat<T> {
    let mut b: Mat<T> = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            b[(i, j)] = rng.scalar();
        }
    }
    b
}

/// `op(T)[i][j]` in `Complex64` for the triangular matrix `T` held in `a`
/// (only its `uplo` triangle is read; the diagonal is 1 for `Diag::Unit`).
pub(super) fn op_t<T: TestScalar>(
    a: &Mat<T>,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    i: usize,
    j: usize,
) -> Complex64 {
    let (r, c) = if trans == Trans::NoTrans {
        (i, j)
    } else {
        (j, i)
    };
    let v = if r == c {
        if diag == Diag::Unit {
            Complex64::new(1.0, 0.0)
        } else {
            a[(r, c)].to_c64()
        }
    } else if (uplo == Uplo::Lower) == (r > c) {
        a[(r, c)].to_c64()
    } else {
        Complex64::new(0.0, 0.0)
    };
    if trans == Trans::ConjTrans {
        v.conj()
    } else {
        v
    }
}

/// `op(T)·Y` (Left) or `Y·op(T)` (Right) in `Complex64`, together with the
/// magnitude bound `|op(T)|·|Y|` used to scale the tolerance.
pub(super) fn tri_product<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    a: &Mat<T>,
    y: &Mat<T>,
) -> (Vec<Complex64>, Vec<f64>) {
    let (side, uplo, trans, diag) = flags;
    let (m, n) = (y.nrows(), y.ncols());
    let s = a.nrows();
    let mut prod = vec![Complex64::new(0.0, 0.0); m * n];
    let mut mag = vec![0.0f64; m * n];
    // Precompute op(T) densely once.
    let mut t = vec![Complex64::new(0.0, 0.0); s * s];
    for j in 0..s {
        for i in 0..s {
            t[i + j * s] = op_t(a, uplo, trans, diag, i, j);
        }
    }
    for j in 0..n {
        for i in 0..m {
            let mut acc = Complex64::new(0.0, 0.0);
            let mut bound = 0.0;
            match side {
                Side::Left => {
                    for l in 0..s {
                        let p = t[i + l * s] * y[(l, j)].to_c64();
                        acc += p;
                        bound += p.norm();
                    }
                }
                Side::Right => {
                    for l in 0..s {
                        let p = y[(i, l)].to_c64() * t[l + j * s];
                        acc += p;
                        bound += p.norm();
                    }
                }
            }
            prod[i + j * m] = acc;
            mag[i + j * m] = bound;
        }
    }
    (prod, mag)
}

/// Asserts that `x` solves `op(T)·X = alpha·B` (Left) or `X·op(T) = alpha·B`
/// (Right) to working accuracy.
pub(super) fn assert_trsm_residual<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    alpha: T,
    a: &Mat<T>,
    b: &Mat<T>,
    x: &Mat<T>,
    label: &str,
) {
    let (m, n) = (b.nrows(), b.ncols());
    let s = a.nrows();
    let (prod, mag) = tri_product(flags, a, x);
    let alpha = alpha.to_c64();
    for j in 0..n {
        for i in 0..m {
            let rhs = alpha * b[(i, j)].to_c64();
            let lhs = prod[i + j * m];
            let tol = T::TOL * (s as f64 + 4.0) * (mag[i + j * m] + rhs.norm() + f64::MIN_POSITIVE);
            assert!(
                (lhs - rhs).norm() <= tol,
                "{label} {flags:?} s={s} m={m} n={n}: residual at ({i},{j}): {lhs} vs {rhs} (tol {tol:e})"
            );
        }
    }
}

/// Asserts that `out` equals `alpha·op(T)·B` (Left) or `alpha·B·op(T)`
/// (Right) to working accuracy.
pub(super) fn assert_trmm_result<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    alpha: T,
    a: &Mat<T>,
    b: &Mat<T>,
    out: &Mat<T>,
    label: &str,
) {
    let (m, n) = (b.nrows(), b.ncols());
    let s = a.nrows();
    let (prod, mag) = tri_product(flags, a, b);
    let alpha = alpha.to_c64();
    for j in 0..n {
        for i in 0..m {
            let want = alpha * prod[i + j * m];
            let got = out[(i, j)].to_c64();
            let tol =
                T::TOL * (s as f64 + 4.0) * (alpha.norm() * mag[i + j * m] + f64::MIN_POSITIVE);
            assert!(
                (got - want).norm() <= tol,
                "{label} {flags:?} s={s} m={m} n={n}: at ({i},{j}): {got} vs {want} (tol {tol:e})"
            );
        }
    }
}

/// Asserts bitwise equality of two matrices.
pub(super) fn assert_bitwise_eq<T: TestScalar>(x: &Mat<T>, y: &Mat<T>, label: &str) {
    assert_eq!(x.nrows(), y.nrows(), "{label}: rows");
    assert_eq!(x.ncols(), y.ncols(), "{label}: cols");
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            assert_eq!(
                x[(i, j)].bits(),
                y[(i, j)].bits(),
                "{label}: ({i},{j}): {} vs {}",
                x[(i, j)],
                y[(i, j)]
            );
        }
    }
}

/// `alpha · b` elementwise (the prescaling of the in-place entry points).
pub(super) fn scaled<T: TestScalar>(alpha: T, b: &Mat<T>) -> Mat<T> {
    let mut out = b.clone();
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            out[(i, j)] = alpha * b[(i, j)];
        }
    }
    out
}

/// Shape `(m, n)` of B for a triangular order `s` and `nrhs` right-hand sides.
pub(super) fn rhs_shape(side: Side, s: usize, nrhs: usize) -> (usize, usize) {
    match side {
        Side::Left => (s, nrhs),
        Side::Right => (nrhs, s),
    }
}
