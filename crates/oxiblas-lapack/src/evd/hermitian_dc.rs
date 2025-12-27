//! Hermitian Eigenvalue Decomposition using Divide-and-Conquer.
//!
//! Uses Householder tridiagonalization followed by the divide-and-conquer algorithm.
//! This is typically faster than the QR algorithm for large matrices.
//!
//! For Hermitian matrices (A = A^H), all eigenvalues are real and eigenvectors form
//! a unitary matrix.

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

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

/// Hermitian eigenvalue decomposition using divide-and-conquer.
///
/// Computes A = U·D·U^H where U contains unitary eigenvectors and D is real diagonal.
/// For Hermitian matrices (A = A^H), all eigenvalues are real.
///
/// The divide-and-conquer algorithm is typically faster than QR for large matrices.
#[derive(Debug, Clone)]
pub struct HermitianEvdDc<T: Scalar> {
    /// Eigenvalues (sorted in ascending order) - always real.
    eigenvalues: Vec<T::Real>,
    /// Eigenvectors (columns of U) - complex unitary matrix.
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

/// Threshold for switching to direct QR algorithm.
const DC_THRESHOLD: usize = 100;

/// Maximum iterations for secular equation solver.
const MAX_SECULAR_ITER: usize = 100;

impl<T: Field + ComplexScalar + bytemuck::Zeroable> HermitianEvdDc<T>
where
    T::Real: Real,
{
    /// Computes the eigendecomposition of a Hermitian matrix using divide-and-conquer.
    ///
    /// # Arguments
    ///
    /// * `a` - Hermitian matrix (only upper triangle is used)
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
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, HermitianEvdDcError> {
        let n = a.nrows();

        if n == 0 {
            return Err(HermitianEvdDcError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(HermitianEvdDcError::NotSquare);
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

        // Copy Hermitian matrix (use upper triangle, conjugate for lower)
        let mut work: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val = a[(i, j)];
                work[(i, j)] = val;
                work[(j, i)] = val.conj();
            }
        }

        // Initialize eigenvector matrix to identity
        let mut u: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            u[(i, i)] = T::one();
        }

        // Tridiagonalize: A = Q * T * Q^H
        // For Hermitian matrices, the tridiagonal form has real diagonal and off-diagonal
        let (diag, off_diag) = tridiagonalize(&mut work, &mut u, n);

        // Apply divide-and-conquer algorithm to real tridiagonal matrix
        let eigenvalues = divide_and_conquer(diag, off_diag, &mut u, n)?;

        Ok(Self {
            eigenvalues,
            eigenvectors: u,
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

/// Tridiagonalizes a Hermitian matrix using Householder reflections.
/// Returns (real diagonal, real off-diagonal) vectors.
/// For Hermitian matrices, the tridiagonal form is real.
fn tridiagonalize<T: Field + ComplexScalar>(
    a: &mut Mat<T>,
    u: &mut Mat<T>,
    n: usize,
) -> (Vec<T::Real>, Vec<T::Real>)
where
    T::Real: Real,
{
    let mut diag = vec![T::Real::zero(); n];
    let mut off_diag = vec![T::Real::zero(); n.saturating_sub(1)];

    // Working vectors
    let mut v: Vec<T> = vec![T::zero(); n];

    for k in 0..(n.saturating_sub(2)) {
        // Compute the norm of the subdiagonal part of column k
        let mut norm_sq = T::Real::zero();
        for i in (k + 1)..n {
            norm_sq = norm_sq + a[(i, k)].abs_sq();
        }
        let alpha = <T::Real as Real>::sqrt(norm_sq);

        if alpha > T::Real::zero() {
            // Get the first element of the subdiagonal
            let x0 = a[(k + 1, k)];
            let x0_norm = x0.abs();

            // Compute the signed norm (similar to LAPACK's choice)
            let beta = if x0_norm > T::Real::zero() {
                let sign = if x0.real() >= T::Real::zero() {
                    T::Real::one()
                } else {
                    -T::Real::one()
                };
                sign * alpha
            } else {
                alpha
            };

            // Construct Householder vector v
            let v0 = if x0_norm > T::Real::zero() {
                x0 + T::from_real(beta) * (x0 / T::from_real(x0_norm))
            } else {
                T::from_real(beta)
            };

            // Store the Householder vector (normalized so v[0] = 1)
            let v0_norm = v0.abs();
            if v0_norm > T::Real::zero() {
                let scale = T::from_real(T::Real::one() / v0_norm);
                v[k + 1] = T::one();
                for i in (k + 2)..n {
                    v[i] = a[(i, k)] * scale;
                }

                // Compute ||v||^2
                let mut v_norm_sq = T::Real::one();
                for i in (k + 2)..n {
                    v_norm_sq = v_norm_sq + v[i].abs_sq();
                }

                // tau = 2 / ||v||^2
                let two = T::Real::one() + T::Real::one();
                let tau = T::from_real(two / v_norm_sq);

                // Compute p = tau * A * v (for the submatrix A[k+1:, k+1:])
                let mut p: Vec<T> = vec![T::zero(); n];
                for i in (k + 1)..n {
                    let mut sum = T::zero();
                    for j in (k + 1)..n {
                        sum = sum + a[(i, j)] * v[j];
                    }
                    p[i] = tau * sum;
                }

                // Compute w = p - (tau/2) * (v^H * p) * v
                let mut vh_p = T::zero();
                for i in (k + 1)..n {
                    vh_p = vh_p + v[i].conj() * p[i];
                }
                let half_tau_vhp = tau * vh_p / T::from_real(two);

                let mut w: Vec<T> = vec![T::zero(); n];
                for i in (k + 1)..n {
                    w[i] = p[i] - half_tau_vhp * v[i];
                }

                // Update A: A = A - v*w^H - w*v^H
                for i in (k + 1)..n {
                    for j in (k + 1)..n {
                        a[(i, j)] = a[(i, j)] - v[i] * w[j].conj() - w[i] * v[j].conj();
                    }
                }

                // Update U: U = U * (I - tau * v * v^H)
                for i in 0..n {
                    let mut uv = T::zero();
                    for j in (k + 1)..n {
                        uv = uv + u[(i, j)] * v[j];
                    }
                    let tau_uv = tau * uv;
                    for j in (k + 1)..n {
                        u[(i, j)] = u[(i, j)] - tau_uv * v[j].conj();
                    }
                }

                off_diag[k] = beta;
            }
        }
    }

    // Extract diagonal elements (should be real for Hermitian)
    for i in 0..n {
        diag[i] = a[(i, i)].real();
    }

    // The last off-diagonal element
    if n >= 2 {
        off_diag[n - 2] = a[(n - 1, n - 2)].abs();
    }

    // Make all off_diag positive (they represent |e_k|)
    for i in 0..off_diag.len() {
        off_diag[i] = off_diag[i].abs();
    }

    (diag, off_diag)
}

/// Divide-and-conquer algorithm for real symmetric tridiagonal matrices.
fn divide_and_conquer<T: Field + ComplexScalar + bytemuck::Zeroable>(
    mut diag: Vec<T::Real>,
    mut off_diag: Vec<T::Real>,
    u: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T::Real>, HermitianEvdDcError>
where
    T::Real: Real,
{
    if n <= DC_THRESHOLD {
        // Use QR algorithm for small matrices
        return qr_algorithm(diag, off_diag, u, n);
    }

    // Split point
    let mid = n / 2;

    // Extract the connecting element
    let rho = off_diag[mid - 1];
    off_diag[mid - 1] = T::Real::zero();

    // Modify diagonal at split point (secular equation formulation)
    diag[mid - 1] = diag[mid - 1] - rho;
    diag[mid] = diag[mid] - rho;

    // Extract subproblems
    let diag1: Vec<T::Real> = diag[..mid].to_vec();
    let off_diag1: Vec<T::Real> = off_diag[..(mid - 1)].to_vec();
    let diag2: Vec<T::Real> = diag[mid..].to_vec();
    let off_diag2: Vec<T::Real> = off_diag[mid..].to_vec();

    // Solve subproblems recursively
    let mut u1: Mat<T> = Mat::zeros(mid, mid);
    let mut u2: Mat<T> = Mat::zeros(n - mid, n - mid);
    for i in 0..mid {
        u1[(i, i)] = T::one();
    }
    for i in 0..(n - mid) {
        u2[(i, i)] = T::one();
    }

    let mut d1 = divide_and_conquer(diag1, off_diag1, &mut u1, mid)?;
    let mut d2 = divide_and_conquer(diag2, off_diag2, &mut u2, n - mid)?;

    // Merge results
    merge_eigenvalues(&mut d1, &mut d2, &mut u1, &mut u2, u, rho, n, mid)
}

/// Merges two sets of eigenvalues/eigenvectors using the secular equation.
fn merge_eigenvalues<T: Field + ComplexScalar + bytemuck::Zeroable>(
    d1: &mut [T::Real],
    d2: &mut [T::Real],
    u1: &mut Mat<T>,
    u2: &mut Mat<T>,
    u: &mut Mat<T>,
    rho: T::Real,
    n: usize,
    mid: usize,
) -> Result<Vec<T::Real>, HermitianEvdDcError>
where
    T::Real: Real,
{
    // Construct z vector (last row of U1, first row of U2)
    let mut z: Vec<T::Real> = vec![T::Real::zero(); n];
    for i in 0..mid {
        z[i] = u1[(mid - 1, i)].real();
    }
    for i in 0..(n - mid) {
        z[mid + i] = u2[(0, i)].real();
    }

    // Combine and sort eigenvalues with indices
    let mut d: Vec<T::Real> = vec![T::Real::zero(); n];
    let mut perm: Vec<usize> = vec![0; n];
    for i in 0..mid {
        d[i] = d1[i];
        perm[i] = i;
    }
    for i in 0..(n - mid) {
        d[mid + i] = d2[i];
        perm[mid + i] = mid + i;
    }

    // Sort eigenvalues and track permutations
    for i in 1..n {
        let key_d = d[i];
        let key_z = z[i];
        let key_p = perm[i];
        let mut j = i;
        while j > 0 && d[j - 1] > key_d {
            d[j] = d[j - 1];
            z[j] = z[j - 1];
            perm[j] = perm[j - 1];
            j -= 1;
        }
        d[j] = key_d;
        z[j] = key_z;
        perm[j] = key_p;
    }

    // Solve secular equation for each eigenvalue
    let mut lambda: Vec<T::Real> = vec![T::Real::zero(); n];
    let eps = <T::Real as Scalar>::epsilon() * T::Real::from_f64(100.0).unwrap_or(T::Real::one());

    for k in 0..n {
        // Find eigenvalue in interval (d[k], d[k+1])
        let lo = d[k];
        let hi = if k < n - 1 {
            d[k + 1]
        } else {
            d[k] + rho.abs() + T::Real::one()
        };

        let mut lam = (lo + hi) / (T::Real::one() + T::Real::one());

        // Newton iteration for secular equation
        for _ in 0..MAX_SECULAR_ITER {
            let mut f = rho;
            let mut df = T::Real::zero();

            for i in 0..n {
                let diff = d[i] - lam;
                if diff.abs() > eps {
                    f = f + z[i] * z[i] / diff;
                    df = df + z[i] * z[i] / (diff * diff);
                }
            }

            if f.abs() < eps {
                break;
            }

            let delta = f / df;
            lam = lam + delta;

            // Clamp to interval
            if lam <= lo {
                lam = lo + eps;
            }
            if lam >= hi && k < n - 1 {
                lam = hi - eps;
            }

            if delta.abs() < eps * lam.abs() {
                break;
            }
        }

        lambda[k] = lam;
    }

    // Compute eigenvectors using the secular equation solution
    let mut v: Mat<T> = Mat::zeros(n, n);
    for k in 0..n {
        let lam = lambda[k];
        let mut norm_sq = T::Real::zero();

        for i in 0..n {
            let diff = d[i] - lam;
            let vi = if diff.abs() > eps {
                z[i] / diff
            } else {
                T::Real::one()
            };
            v[(i, k)] = T::from_real(vi);
            norm_sq = norm_sq + vi * vi;
        }

        // Normalize
        if norm_sq > T::Real::zero() {
            let norm = Real::sqrt(norm_sq);
            for i in 0..n {
                v[(i, k)] = v[(i, k)] / T::from_real(norm);
            }
        }
    }

    // Transform eigenvectors back
    // Apply U1 and U2 blocks
    let mut u_temp: Mat<T> = Mat::zeros(n, n);

    // Build the block diagonal matrix of U1 and U2
    for i in 0..mid {
        for j in 0..mid {
            u_temp[(i, j)] = u1[(i, j)];
        }
    }
    for i in 0..(n - mid) {
        for j in 0..(n - mid) {
            u_temp[(mid + i, mid + j)] = u2[(i, j)];
        }
    }

    // Multiply: U_new = U_old * u_temp * V
    // First: temp2 = u_temp * V (permuted)
    let mut temp2: Mat<T> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + u_temp[(i, k)] * v[(k, j)];
            }
            temp2[(i, j)] = sum;
        }
    }

    // Then: U = U_old * temp2
    let mut u_new: Mat<T> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + u[(i, k)] * temp2[(k, j)];
            }
            u_new[(i, j)] = sum;
        }
    }

    // Copy back to u
    for i in 0..n {
        for j in 0..n {
            u[(i, j)] = u_new[(i, j)];
        }
    }

    Ok(lambda)
}

/// QR algorithm for small real symmetric tridiagonal matrices.
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
    let max_iter = 100 * n;

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
                off_diag[k - 1] = <T::Real as Real>::hypot(x, z);
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
}
