//! Hermitian Eigenvalue Decomposition using Divide-and-Conquer.
//!
//! For Hermitian matrices (A = A^H), all eigenvalues are real and eigenvectors form
//! a unitary matrix. The pipeline mirrors LAPACK `ZHEEVD`:
//!
//! 1. a matrix whose norm is tiny or huge is scaled by a power of two into the
//!    safe range (exact; the eigenvalues are scaled back at the end);
//! 2. Householder reduction to a *real* symmetric tridiagonal `T` (`ZHETRD`):
//!    complex reflectors `Q` followed by the unitary diagonal phase scaling `D`
//!    that makes the off-diagonal real, `A = (Q·D)·T·(Q·D)ᴴ` (the reduction
//!    shared with [`HermitianEvd`](super::HermitianEvd));
//! 3. Cuppen's divide and conquer on `T` in the real type `T::Real` (`DSTEDC`,
//!    the solver of [`SymmetricEvdDc`](super::SymmetricEvdDc)): Gu-Eisenstat
//!    deflation, a `DLAED4`-style secular-equation solver that reports
//!    non-convergence, Löwner-formula eigenvectors and GEMM merges, giving
//!    `T = Z·Λ·Zᵀ` with a real orthogonal `Z`;
//! 4. back-transformation `V = (Q·D)·Z` (`ZUNMTR`; complex times real as in
//!    `ZLACRM`): one real GEMM on the stacked real and imaginary parts of
//!    `Q·D`.
//!
//! For `Complex64` and `Complex32` the merges and the back-transformation run
//! through the `oxiblas_blas` GEMM. Other complex scalar types (none in this
//! workspace) use the same reduction followed by the implicit QR iteration.

use core::any::Any;

use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::dc_merge::{DcError, stedc};
use super::hermitian_tridiag::tridiagonalize_hermitian;
use super::sym_kernels::{BlasMatMul, MatMul};

/// Error type for divide-and-conquer Hermitian eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HermitianEvdDcError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
}

impl core::fmt::Display for HermitianEvdDcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for HermitianEvdDcError {}

impl From<DcError> for HermitianEvdDcError {
    fn from(e: DcError) -> Self {
        match e {
            DcError::NotConverged => Self::NotConverged,
            DcError::SecularEquationFailed => Self::SecularEquationFailed,
        }
    }
}

/// Hermitian eigenvalue decomposition using divide-and-conquer.
///
/// Computes A = U·D·U^H where U contains unitary eigenvectors and D is real diagonal.
/// For Hermitian matrices (A = A^H), all eigenvalues are real.
///
/// As in LAPACK, divide-and-conquer overtakes the QR algorithm
/// ([`HermitianEvd`](super::HermitianEvd)) on large matrices because almost
/// all of its eigenvector work is matrix multiplication.
#[derive(Debug, Clone)]
pub struct HermitianEvdDc<T: Scalar> {
    /// Eigenvalues (sorted in ascending order) - always real.
    eigenvalues: Vec<T::Real>,
    /// Eigenvectors (columns of U) - complex unitary matrix.
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

/// Leaf size of the real divide and conquer: blocks of at most this many rows
/// go to the QR iteration (LAPACK `SMLSIZ`, as in `SymmetricEvdDc`). The merge
/// is exact at every size, so this only trades speed.
const SMLSIZ: usize = 25;

/// Iteration limit, per row, of the QR iteration of the generic fallback.
const QR_MAX_ITER: usize = 100;

impl<T: Field + ComplexScalar + bytemuck::Zeroable> HermitianEvdDc<T>
where
    T::Real: Real,
{
    /// Computes the eigendecomposition of a Hermitian matrix using divide-and-conquer.
    ///
    /// # Arguments
    ///
    /// * `a` - Hermitian matrix. Only the upper triangle is read, and the
    ///   imaginary parts of its diagonal are ignored (taken as zero, as in
    ///   LAPACK `ZHEEVD`).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::HermitianEvdDc;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
    ///     &[Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
    /// ]);
    ///
    /// let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
    /// let eigs = evd.eigenvalues();
    ///
    /// // Eigenvalues are real: 1 and 4
    /// assert!((eigs[0] - 1.0).abs() < 1e-10);
    /// assert!((eigs[1] - 4.0).abs() < 1e-10);
    /// ```
    ///
    /// # Errors
    ///
    /// [`HermitianEvdDcError::EmptyMatrix`] or
    /// [`HermitianEvdDcError::NotSquare`] for an empty or non-square `a`;
    /// [`HermitianEvdDcError::NotConverged`] if the upper triangle of `a` has
    /// a non-finite (NaN or infinite) entry, or the QR iteration on a leaf
    /// block does not converge;
    /// [`HermitianEvdDcError::SecularEquationFailed`] if a secular equation of
    /// a merge does not converge within its iteration limit.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, HermitianEvdDcError> {
        let n = a.nrows();

        if n == 0 {
            return Err(HermitianEvdDcError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(HermitianEvdDcError::NotSquare);
        }
        if !upper_triangle_is_finite(a) {
            return Err(HermitianEvdDcError::NotConverged);
        }

        // Handle trivial case
        if n == 1 {
            let eigenvalues = vec![a[(0, 0)].real()];
            let mut eigenvectors: Mat<T> = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues,
                eigenvectors,
                n,
            });
        }

        let mut work = hermitian_from_upper(a);
        let mut eigenvalues = Vec::new();
        let mut eigenvectors: Mat<T> = Mat::zeros(n, n);
        match with_complex_backend(&mut work, &mut eigenvalues, &mut eigenvectors) {
            Some(result) => result?,
            None => qr_pipeline(&mut work, &mut eigenvalues, &mut eigenvectors)?,
        }

        Ok(Self {
            eigenvalues,
            eigenvectors,
            n,
        })
    }

    /// Returns the eigenvalues (sorted in ascending order).
    /// Eigenvalues of Hermitian matrices are always real.
    pub fn eigenvalues(&self) -> &[T::Real] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix U.
    ///
    /// Column i contains the eigenvector corresponding to eigenvalue i.
    /// U is unitary: U^H * U = I
    pub fn eigenvectors(&self) -> MatRef<'_, T> {
        self.eigenvectors.as_ref()
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = U * D * U^H
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a: Mat<T> = Mat::zeros(n, n);

        // A = U * D * U^H = sum_i lambda_i * u_i * u_i^H
        for k in 0..n {
            let lambda = T::from_real(self.eigenvalues[k]);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = a[(i, j)]
                        + lambda * self.eigenvectors[(i, k)] * self.eigenvectors[(j, k)].conj();
                }
            }
        }

        a
    }
}

/// `true` when every entry of the upper triangle of `a` (the part that is
/// read) has finite real and imaginary parts.
fn upper_triangle_is_finite<T: ComplexScalar>(a: MatRef<'_, T>) -> bool
where
    T::Real: Real,
{
    let n = a.nrows();
    (0..n).all(|j| {
        (0..=j).all(|i| {
            let z = a[(i, j)];
            num_traits::Float::is_finite(z.real()) && num_traits::Float::is_finite(z.imag())
        })
    })
}

/// The full Hermitian matrix defined by the upper triangle of `a`, with a real
/// diagonal (`ZHETRD` takes the imaginary parts of the diagonal as zero).
fn hermitian_from_upper<T: ComplexScalar + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Mat<T>
where
    T::Real: Real,
{
    let n = a.nrows();
    let mut work: Mat<T> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..j {
            let val = a[(i, j)];
            work[(i, j)] = val;
            work[(j, i)] = val.conj();
        }
        work[(j, j)] = T::from_real(a[(j, j)].real());
    }
    work
}

/// The largest power of two not above `x` (`1` unless `x` is positive and
/// finite).
fn pow2_below<R: Real>(x: R) -> R {
    let two = R::one() + R::one();
    let mut s = R::one();
    if !(x > R::zero()) || !num_traits::Float::is_finite(x) {
        return s;
    }
    let mut guard = 0;
    while s * two <= x && guard < 1 << 14 {
        s = s * two;
        guard += 1;
    }
    while s > x && guard < 1 << 14 {
        s = s / two;
        guard += 1;
    }
    s
}

/// Scales the Hermitian matrix `a` (full storage) by a power of two into the
/// safe range of LAPACK `ZHEEVD` (max-norm in `[sqrt(smlnum), sqrt(1/smlnum)]`,
/// `smlnum = safmin / eps`) so the reduction neither overflows nor loses
/// accuracy to underflow; returns the factor (`1` when the norm is already in
/// range or zero).
///
/// The eigenvalues of the scaled matrix are divided by the factor afterwards;
/// the eigenvectors are unchanged. A power of two makes both steps exact.
fn scale_into_safe_range<T: ComplexScalar>(a: &mut Mat<T>) -> T::Real
where
    T::Real: Real,
{
    let n = a.nrows();
    let mut anrm = T::Real::zero();
    for j in 0..n {
        for i in 0..=j {
            let v = Scalar::abs(a[(i, j)]);
            if v > anrm {
                anrm = v;
            }
        }
    }
    let one = T::Real::one();
    let two = one + one;
    let smlnum = <T::Real as Scalar>::min_positive() / <T::Real as Scalar>::epsilon();
    let rmin = <T::Real as Real>::sqrt(smlnum);
    let rmax = <T::Real as Real>::sqrt(one / smlnum);
    let sigma = if anrm > T::Real::zero() && anrm < rmin {
        two * pow2_below(rmin / anrm)
    } else if anrm > rmax && num_traits::Float::is_finite(anrm) {
        one / (two * pow2_below(anrm / rmax))
    } else {
        return one;
    };
    for j in 0..n {
        for i in 0..n {
            let z = a[(i, j)];
            a[(i, j)] = T::from_real_imag(z.real() * sigma, z.imag() * sigma);
        }
    }
    sigma
}

/// Divides the eigenvalues of the scaled matrix by the scaling factor.
fn unscale<R: Real>(values: &mut [R], sigma: R) {
    if sigma != R::one() {
        for v in values.iter_mut() {
            *v = *v / sigma;
        }
    }
}

/// Runs [`dc_pipeline`] with the `oxiblas_blas` GEMM when `T` is `Complex64`
/// or `Complex32`, by downcasting the owned buffers (as
/// `sym_kernels::with_gemm_backend` does for the real drivers; no `unsafe`).
///
/// Returns `None` for any other complex scalar type: the real divide and
/// conquer needs a real type with `Field` and `bytemuck::Zeroable`, which the
/// public bounds do not promise for a general `T::Real`.
fn with_complex_backend<T>(
    work: &mut Mat<T>,
    values: &mut Vec<T::Real>,
    vectors: &mut Mat<T>,
) -> Option<Result<(), HermitianEvdDcError>>
where
    T: Field + ComplexScalar + bytemuck::Zeroable,
    T::Real: Real,
{
    if let (Some(w), Some(v), Some(z)) = (
        (work as &mut dyn Any).downcast_mut::<Mat<Complex64>>(),
        (values as &mut dyn Any).downcast_mut::<Vec<f64>>(),
        (vectors as &mut dyn Any).downcast_mut::<Mat<Complex64>>(),
    ) {
        return Some(dc_pipeline::<Complex64, f64, BlasMatMul>(w, v, z));
    }
    if let (Some(w), Some(v), Some(z)) = (
        (work as &mut dyn Any).downcast_mut::<Mat<Complex32>>(),
        (values as &mut dyn Any).downcast_mut::<Vec<f32>>(),
        (vectors as &mut dyn Any).downcast_mut::<Mat<Complex32>>(),
    ) {
        return Some(dc_pipeline::<Complex32, f32, BlasMatMul>(w, v, z));
    }
    None
}

/// The divide-and-conquer pipeline (`ZHEEVD`) for the complex type `C` with
/// real type `R` and matrix-product back end `G`.
///
/// `work` holds the full Hermitian matrix (real diagonal) and is overwritten;
/// on success `values` holds the eigenvalues in ascending order and the
/// columns of `vectors` (`n × n`) the corresponding orthonormal eigenvectors.
fn dc_pipeline<C, R, G>(
    work: &mut Mat<C>,
    values: &mut Vec<R>,
    vectors: &mut Mat<C>,
) -> Result<(), HermitianEvdDcError>
where
    C: Field + ComplexScalar<Real = R> + bytemuck::Zeroable,
    R: Field + Real + bytemuck::Zeroable,
    G: MatMul<R>,
{
    let n = work.nrows();
    let sigma = scale_into_safe_range(work);
    // A = (Q·D)·T·(Q·D)ᴴ with T real symmetric tridiagonal; `qd` = Q·D.
    let mut qd: Mat<C> = Mat::eye(n);
    let (mut d, mut e) = tridiagonalize_hermitian(work, &mut qd, n);
    // T = Z·Λ·Zᵀ: eigenvalues ascending, Z real orthogonal.
    let mut z: Mat<R> = Mat::zeros(n, n);
    stedc::<R, G>(&mut d, &mut e, &mut z, SMLSIZ, Par::Seq)?;
    back_transform::<C, R, G>(&qd, &z, vectors);
    unscale(&mut d, sigma);
    *values = d;
    Ok(())
}

/// `v = qd·z` for complex `qd` and real `z` (`n × n`), as one real GEMM on the
/// stacked parts: `[Re v; Im v] = [Re qd; Im qd]·z` (LAPACK `ZLACRM`).
///
/// Every output entry is a real dot product of `n` terms, so the product is
/// as accurate as a complex one at half its cost.
fn back_transform<C, R, G>(qd: &Mat<C>, z: &Mat<R>, v: &mut Mat<C>)
where
    C: Field + ComplexScalar<Real = R> + bytemuck::Zeroable,
    R: Field + Real + bytemuck::Zeroable,
    G: MatMul<R>,
{
    let n = qd.nrows();
    let mut stacked: Mat<R> = Mat::zeros(2 * n, n);
    for j in 0..n {
        for i in 0..n {
            let x = qd[(i, j)];
            stacked[(i, j)] = x.real();
            stacked[(n + i, j)] = x.imag();
        }
    }
    let mut product: Mat<R> = Mat::zeros(2 * n, n);
    G::gemm(
        false,
        false,
        R::one(),
        stacked.as_ref(),
        z.as_ref(),
        R::zero(),
        product.as_mut(),
        Par::Seq,
    );
    for j in 0..n {
        for i in 0..n {
            v[(i, j)] = C::from_real_imag(product[(i, j)], product[(n + i, j)]);
        }
    }
}

/// Pipeline for complex scalar types without a GEMM back end (none in this
/// workspace): the same scaling and reduction, then the implicit QR iteration
/// on the real tridiagonal, accumulating its rotations into `Q·D`.
fn qr_pipeline<T>(
    work: &mut Mat<T>,
    values: &mut Vec<T::Real>,
    vectors: &mut Mat<T>,
) -> Result<(), HermitianEvdDcError>
where
    T: Field + ComplexScalar + bytemuck::Zeroable,
    T::Real: Real,
{
    let n = work.nrows();
    let sigma = scale_into_safe_range(work);
    let mut qd: Mat<T> = Mat::eye(n);
    let (d, e) = tridiagonalize_hermitian(work, &mut qd, n);
    let mut eigs = qr_algorithm(d, e, &mut qd, n)?;
    unscale(&mut eigs, sigma);
    *values = eigs;
    *vectors = qd;
    Ok(())
}

/// Implicit QR iteration (Wilkinson shift) on the real symmetric tridiagonal
/// `tridiag(off_diag, diag, off_diag)`, applying its rotations to the columns
/// of the complex `u`; returns the eigenvalues in ascending order (columns of
/// `u` sorted alike). Used by [`qr_pipeline`] only.
fn qr_algorithm<T: Field + ComplexScalar + bytemuck::Zeroable>(
    mut diag: Vec<T::Real>,
    mut off_diag: Vec<T::Real>,
    u: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T::Real>, HermitianEvdDcError>
where
    T::Real: Real,
{
    if n <= 1 {
        return Ok(diag);
    }

    let eps = <T::Real as Scalar>::epsilon() * T::Real::from_f64(100.0).unwrap_or(T::Real::one());
    let max_iter = QR_MAX_ITER * n;

    let mut m = n - 1;
    let mut iter = 0;

    while m > 0 && iter < max_iter {
        iter += 1;

        // Find largest m such that off_diag[m-1] is not negligible
        let mut l = m;
        while l > 0 {
            let test = diag[l - 1].abs() + diag[l].abs();
            if off_diag[l - 1].abs() <= eps * test {
                off_diag[l - 1] = T::Real::zero();
                break;
            }
            l -= 1;
        }

        if l == m {
            m -= 1;
            continue;
        }

        // Wilkinson shift
        let two = T::Real::one() + T::Real::one();
        let d = (diag[m - 1] - diag[m]) / two;
        let e = off_diag[m - 1];
        let sign_d = if d >= T::Real::zero() {
            T::Real::one()
        } else {
            -T::Real::one()
        };
        let mu = diag[m] - e * e / (d + sign_d * <T::Real as Real>::hypot(d, e));

        // Implicit QR step
        let mut x = diag[l] - mu;
        let mut z = off_diag[l];

        for k in l..m {
            let (c, s) = givens_rotation(x, z);

            if k > l {
                // Signed rotated off-diagonal r = c·x − s·z (NOT hypot(x, z)): dropping the
                // sign leaves eigenVALUES correct but the accumulated Givens rotations
                // inconsistent with the tridiagonal, so the eigenVECTORS come out wrong
                // (U stays orthonormal, hiding the defect). Matches evd/symmetric.rs.
                off_diag[k - 1] = c * x - s * z;
            }

            let d1 = diag[k];
            let d2 = diag[k + 1];
            let e = off_diag[k];

            diag[k] = c * c * d1 + s * s * d2 - (c + c) * s * e;
            diag[k + 1] = s * s * d1 + c * c * d2 + (c + c) * s * e;
            off_diag[k] = c * s * (d1 - d2) + (c * c - s * s) * e;

            if k < m - 1 {
                x = off_diag[k];
                z = -s * off_diag[k + 1];
                off_diag[k + 1] = c * off_diag[k + 1];
            }

            // Update eigenvectors (complex)
            let c_t = T::from_real(c);
            let s_t = T::from_real(s);
            for i in 0..n {
                let t1 = u[(i, k)];
                let t2 = u[(i, k + 1)];
                u[(i, k)] = c_t * t1 - s_t * t2;
                u[(i, k + 1)] = s_t * t1 + c_t * t2;
            }
        }
    }

    if iter >= max_iter {
        return Err(HermitianEvdDcError::NotConverged);
    }

    // Sort eigenvalues and eigenvectors
    sort_eigenvalues(&mut diag, u, n);

    Ok(diag)
}

/// Computes Givens rotation coefficients.
fn givens_rotation<R: Real>(a: R, b: R) -> (R, R) {
    if b == R::zero() {
        (R::one(), R::zero())
    } else if Scalar::abs(b) > Scalar::abs(a) {
        let t = -a / b;
        let s = R::one() / <R as Real>::sqrt(R::one() + t * t);
        (s * t, s)
    } else {
        let t = -b / a;
        let c = R::one() / <R as Real>::sqrt(R::one() + t * t);
        (c, c * t)
    }
}

/// Sorts eigenvalues in ascending order and rearranges eigenvectors accordingly.
fn sort_eigenvalues<T: Field + ComplexScalar>(eigenvalues: &mut [T::Real], u: &mut Mat<T>, n: usize)
where
    T::Real: Real,
{
    for i in 1..n {
        let key = eigenvalues[i];
        let mut j = i;
        while j > 0 && eigenvalues[j - 1] > key {
            eigenvalues[j] = eigenvalues[j - 1];
            for row in 0..n {
                let tmp = u[(row, j)];
                u[(row, j)] = u[(row, j - 1)];
                u[(row, j - 1)] = tmp;
            }
            j -= 1;
        }
        eigenvalues[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_hermitian_evd_dc_real_symmetric() {
        // A Hermitian matrix that's actually real symmetric
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_dc_complex() {
        // Hermitian matrix with complex off-diagonal entries
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            &[Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues should be 1 and 4
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 4.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_dc_unitary() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, 2.0)],
            &[Complex64::new(1.0, -2.0), Complex64::new(3.0, 0.0)],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let u = evd.eigenvectors();

        // Check U^H * U = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex64::zero();
                for k in 0..2 {
                    sum = sum + u[(k, i)].conj() * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum.re - expected).abs() < 1e-9 && sum.im.abs() < 1e-9,
                    "U^H*U[{},{}] = ({}, {}), expected {}",
                    i,
                    j,
                    sum.re,
                    sum.im,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_hermitian_evd_dc_diagonal() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues sorted: 1, 2, 3
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_dc_single() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(5.0, 0.0)]]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 1);
        assert!(approx_eq(eigs[0], 5.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_dc_trace() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 2.0),
            ],
            &[
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            &[
                Complex64::new(0.0, -2.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Trace = sum of eigenvalues
        let trace_eigs: f64 = eigs.iter().sum();
        let trace_a = a[(0, 0)].re + a[(1, 1)].re + a[(2, 2)].re;
        assert!(
            (trace_eigs - trace_a).abs() < 1e-8,
            "Trace mismatch: {} vs {}",
            trace_eigs,
            trace_a
        );
    }

    #[test]
    fn test_hermitian_evd_dc_f32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(2.0, 0.0), Complex32::new(1.0, 1.0)],
            &[Complex32::new(1.0, -1.0), Complex32::new(3.0, 0.0)],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues should be 1 and 4
        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 4.0).abs() < 1e-5);
    }

    /// Regression test for the missing diagonal phase-correction bug (D&C variant).
    ///
    /// A *genuinely* complex Hermitian matrix is used and EVERY eigenpair is verified via
    /// `A·v = λ·v`. Before the fix the eigenvalues were right but each eigenvector was off
    /// by a per-row unit-modulus phase, which only the residual check exposes.
    #[test]
    fn test_hermitian_evd_dc_complex_eigenpairs_5x5() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(0.5, -1.0),
                Complex64::new(2.0, 0.5),
                Complex64::new(-1.0, 1.0),
            ],
            &[
                Complex64::new(1.0, -2.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(0.5, -0.5),
                Complex64::new(1.0, 3.0),
            ],
            &[
                Complex64::new(0.5, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(1.0, -2.0),
                Complex64::new(0.5, 0.5),
            ],
            &[
                Complex64::new(2.0, -0.5),
                Complex64::new(0.5, 0.5),
                Complex64::new(1.0, 2.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, -1.0),
            ],
            &[
                Complex64::new(-1.0, -1.0),
                Complex64::new(1.0, -3.0),
                Complex64::new(0.5, -0.5),
                Complex64::new(3.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
        ]);

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        let u = evd.eigenvectors();
        let n = 5;

        for k in 1..n {
            assert!(eigs[k - 1] <= eigs[k] + 1e-12);
        }

        // A·v = λ·v for every eigenpair.
        for k in 0..n {
            for i in 0..n {
                let mut av = Complex64::zero();
                for j in 0..n {
                    av = av + a[(i, j)] * u[(j, k)];
                }
                let lv = Complex64::new(eigs[k], 0.0) * u[(i, k)];
                assert!(
                    (av.re - lv.re).abs() < 1e-8 && (av.im - lv.im).abs() < 1e-8,
                    "A*v != lambda*v at eigenpair {}, row {}: Av=({},{}) lv=({},{})",
                    k,
                    i,
                    av.re,
                    av.im,
                    lv.re,
                    lv.im
                );
            }
        }

        // Orthonormality: U^H·U = I.
        for i in 0..n {
            for j in 0..n {
                let mut s = Complex64::zero();
                for kk in 0..n {
                    s = s + u[(kk, i)].conj() * u[(kk, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s.re - expected).abs() < 1e-8 && s.im.abs() < 1e-8,
                    "U^H*U[{},{}] = ({},{}), expected {}",
                    i,
                    j,
                    s.re,
                    s.im,
                    expected
                );
            }
        }

        // Full reconstruction A = U·D·U^H.
        let recon = evd.reconstruct();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (recon[(i, j)].re - a[(i, j)].re).abs() < 1e-8
                        && (recon[(i, j)].im - a[(i, j)].im).abs() < 1e-8,
                    "reconstruct mismatch at [{},{}]: got ({},{}) want ({},{})",
                    i,
                    j,
                    recon[(i, j)].re,
                    recon[(i, j)].im,
                    a[(i, j)].re,
                    a[(i, j)].im
                );
            }
        }
    }

    /// Regression test with a genuine cluster of repeated eigenvalues (D&C variant).
    ///
    /// `A = 2·I + w·wᴴ` (complex `w`) has eigenvalue 2 with multiplicity 3 and
    /// `2 + ‖w‖²` once. Every eigenpair is checked against `A·v = λ·v`.
    #[test]
    fn test_hermitian_evd_dc_complex_clustered_eigenvalues() {
        let w = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
        ];
        let n = 4;
        let mut a: Mat<Complex64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut val = w[i] * w[j].conj();
                if i == j {
                    val = val + Complex64::new(2.0, 0.0);
                }
                a[(i, j)] = val;
            }
        }

        let evd = HermitianEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        let u = evd.eigenvectors();

        // ‖w‖² = 9  =>  spectrum {2, 2, 2, 11}.
        assert!(approx_eq(eigs[0], 2.0, 1e-8));
        assert!(approx_eq(eigs[1], 2.0, 1e-8));
        assert!(approx_eq(eigs[2], 2.0, 1e-8));
        assert!(approx_eq(eigs[3], 11.0, 1e-8));

        for k in 0..n {
            for i in 0..n {
                let mut av = Complex64::zero();
                for j in 0..n {
                    av = av + a[(i, j)] * u[(j, k)];
                }
                let lv = Complex64::new(eigs[k], 0.0) * u[(i, k)];
                assert!(
                    (av.re - lv.re).abs() < 1e-8 && (av.im - lv.im).abs() < 1e-8,
                    "A*v != lambda*v at eigenpair {}, row {}",
                    k,
                    i
                );
            }
        }

        for i in 0..n {
            for j in 0..n {
                let mut s = Complex64::zero();
                for kk in 0..n {
                    s = s + u[(kk, i)].conj() * u[(kk, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s.re - expected).abs() < 1e-8 && s.im.abs() < 1e-8,
                    "U^H*U[{},{}] not identity",
                    i,
                    j
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "hermitian_dc_tests.rs"]
mod accuracy_tests;
