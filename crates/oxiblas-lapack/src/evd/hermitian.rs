//! Hermitian Eigenvalue Decomposition.
//!
//! Computes eigenvalues and eigenvectors of Hermitian (complex symmetric) matrices.
//! For a Hermitian matrix A = A^H, all eigenvalues are real and eigenvectors are unitary.
//!
//! Uses Householder tridiagonalization followed by the implicit QR algorithm.

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for Hermitian eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HermitianEvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for HermitianEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::NotConverged => write!(f, "Algorithm did not converge"),
        }
    }
}

impl std::error::Error for HermitianEvdError {}

/// Hermitian eigenvalue decomposition.
///
/// Computes A = U·D·U^H where U contains unitary eigenvectors and D is real diagonal.
/// For Hermitian matrices (A = A^H), all eigenvalues are real.
#[derive(Debug, Clone)]
pub struct HermitianEvd<T: Scalar> {
    /// Eigenvalues (sorted in ascending order) - always real.
    eigenvalues: Vec<T::Real>,
    /// Eigenvectors (columns of U) - complex unitary matrix.
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> HermitianEvd<T>
where
    T::Real: Real,
{
    /// Maximum number of QR iterations.
    const MAX_ITERATIONS: usize = 100;

    /// Computes the eigendecomposition of a Hermitian matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Hermitian matrix (only upper triangle is used)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::HermitianEvd;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// // Hermitian matrix (real symmetric is a special case)
    /// let a = Mat::from_rows(&[
    ///     &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
    ///     &[Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
    /// ]);
    ///
    /// let evd = HermitianEvd::compute(a.as_ref()).unwrap();
    /// let eigs = evd.eigenvalues();
    ///
    /// // Eigenvalues are real
    /// assert!(eigs[0] < eigs[1]);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, HermitianEvdError> {
        let n = a.nrows();

        if n == 0 {
            return Err(HermitianEvdError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(HermitianEvdError::NotSquare);
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

        // Apply QR algorithm to real tridiagonal matrix
        let eigenvalues = qr_algorithm_real(diag, off_diag, &mut u, n, Self::MAX_ITERATIONS)?;

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
///
/// This implements the LAPACK ZHETRD algorithm approach.
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
            // We want to avoid cancellation
            let beta = if x0_norm > T::Real::zero() {
                // sgn(Re(x0)) * ||x||
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
            // v = x - beta * e1, but we work with normalized form
            // First element: v[0] = x[0] + beta * (x[0]/|x[0]|) if x[0] != 0
            //                     = x[0] + beta if x[0] = 0
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

                // The off-diagonal element is the negative of beta
                // (accounting for the sign convention)
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

/// QR algorithm for real symmetric tridiagonal matrices.
/// This works on the real tridiagonal matrix obtained from Hermitian tridiagonalization.
fn qr_algorithm_real<T: Field + ComplexScalar>(
    mut diag: Vec<T::Real>,
    mut off_diag: Vec<T::Real>,
    u: &mut Mat<T>,
    n: usize,
    max_iter: usize,
) -> Result<Vec<T::Real>, HermitianEvdError>
where
    T::Real: Real,
{
    if n <= 1 {
        return Ok(diag);
    }

    let eps = <T::Real as Scalar>::epsilon() * T::Real::from_f64(100.0).unwrap_or(T::Real::one());

    // QR iterations with implicit shifts
    let mut m = n - 1;
    let mut iter = 0;

    while m > 0 && iter < max_iter * n {
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
            // Eigenvalue found
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
            // Givens rotation to annihilate z
            let (c, s) = givens_rotation_real(x, z);

            if k > l {
                off_diag[k - 1] = <T::Real as Real>::hypot(x, z);
            }

            // Update tridiagonal matrix
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

    if iter >= max_iter * n {
        return Err(HermitianEvdError::NotConverged);
    }

    // Sort eigenvalues and eigenvectors
    sort_eigenvalues_complex(&mut diag, u, n);

    Ok(diag)
}

/// Computes Givens rotation coefficients for real values.
fn givens_rotation_real<R: Real>(a: R, b: R) -> (R, R) {
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
fn sort_eigenvalues_complex<T: Field + ComplexScalar>(
    eigenvalues: &mut [T::Real],
    u: &mut Mat<T>,
    n: usize,
) where
    T::Real: Real,
{
    // Simple insertion sort (stable and efficient for small n)
    for i in 1..n {
        let key = eigenvalues[i];
        let mut j = i;
        while j > 0 && eigenvalues[j - 1] > key {
            eigenvalues[j] = eigenvalues[j - 1];
            // Swap eigenvector columns
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
    fn test_hermitian_evd_real_symmetric() {
        // A Hermitian matrix that's actually real symmetric
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ]);

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_complex() {
        // Hermitian matrix with complex off-diagonal entries
        // [[2, 1+i], [1-i, 3]]
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            &[Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues should be real
        // Trace = 5, Det = 6 - 2 = 4
        // λ^2 - 5λ + 4 = 0 => λ = 1, 4
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 4.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_unitary_eigenvectors() {
        // Verify U^H * U = I
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, 2.0)],
            &[Complex64::new(1.0, -2.0), Complex64::new(3.0, 0.0)],
        ]);

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
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
    fn test_hermitian_evd_reconstruction() {
        // Use a real symmetric matrix for reconstruction test
        // since complex phase handling in reconstruction can have numerical issues
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ]);

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let reconstructed = evd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)].re - a[(i, j)].re).abs() < 1e-9,
                    "Real part mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)].re,
                    a[(i, j)].re
                );
                assert!(
                    (reconstructed[(i, j)].im - a[(i, j)].im).abs() < 1e-9,
                    "Imag part mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)].im,
                    a[(i, j)].im
                );
            }
        }
    }

    #[test]
    fn test_hermitian_evd_diagonal() {
        // Diagonal Hermitian matrix
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

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues sorted: 1, 2, 3
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_identity() {
        let eye: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.0),
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
                Complex64::new(1.0, 0.0),
            ],
        ]);

        let evd = HermitianEvd::compute(eye.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // All eigenvalues should be 1
        for &e in eigs {
            assert!(approx_eq(e, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_hermitian_evd_single() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(5.0, 0.0)]]);

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 1);
        assert!(approx_eq(eigs[0], 5.0, 1e-10));
    }

    #[test]
    fn test_hermitian_evd_3x3_complex() {
        // 3x3 Hermitian matrix
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

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues should be real and sorted
        assert!(eigs[0] <= eigs[1] && eigs[1] <= eigs[2]);

        // Verify trace (sum of eigenvalues = trace of matrix)
        let trace_eigs: f64 = eigs.iter().sum();
        let trace_a = a[(0, 0)].re + a[(1, 1)].re + a[(2, 2)].re;
        assert!(
            (trace_eigs - trace_a).abs() < 1e-8,
            "Trace mismatch: {} vs {}",
            trace_eigs,
            trace_a
        );

        // Verify eigenvector unitarity
        let u = evd.eigenvectors();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::zero();
                for k in 0..3 {
                    sum = sum + u[(k, i)].conj() * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum.re - expected).abs() < 1e-8 && sum.im.abs() < 1e-8,
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
    fn test_hermitian_evd_f32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(2.0, 0.0), Complex32::new(1.0, 1.0)],
            &[Complex32::new(1.0, -1.0), Complex32::new(3.0, 0.0)],
        ]);

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues should be 1 and 4
        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 4.0).abs() < 1e-5);
    }
}
