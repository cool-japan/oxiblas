//! Hermitian Eigenvalue Decomposition.
//!
//! Computes eigenvalues and eigenvectors of Hermitian (complex symmetric) matrices.
//! For a Hermitian matrix A = A^H, all eigenvalues are real and eigenvectors are unitary.
//!
//! Uses Householder tridiagonalization followed by the implicit QR algorithm.

use super::hermitian_tridiag::tridiagonalize_hermitian;
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

        // Tridiagonalize: A = (Q·D) * T * (Q·D)^H with T real symmetric tridiagonal.
        // `u` receives Q·D (the accumulated reflectors folded with the diagonal phase
        // correction D) so the real solver below yields the correct eigenvectors Q·D·V.
        let (diag, off_diag) = tridiagonalize_hermitian(&mut work, &mut u, n);

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
                // The rotated off-diagonal is the SIGNED value r = c·x − s·z, not its
                // magnitude. Using hypot(x, z) here would drop the sign: eigenVALUES are
                // unaffected (they don't see this sign) but every accumulated Givens
                // rotation would then be inconsistent with the tridiagonal it is supposed
                // to be diagonalizing, leaving the eigenVECTORS wrong even though U stays
                // orthonormal. This is the sign-preserving form used by the real
                // symmetric solver (see evd/symmetric.rs).
                off_diag[k - 1] = c * x - s * z;
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

    /// Regression test for the missing diagonal phase-correction bug.
    ///
    /// A *genuinely* complex Hermitian matrix (nonzero imaginary off-diagonals) is used,
    /// and EVERY eigenpair is verified via `A·v = λ·v`. Before the fix the eigenVALUES
    /// were correct (they are real, hence phase-independent) but each eigenVECTOR was off
    /// by a per-row unit-modulus phase, so this residual check — not the eigenvalue or the
    /// orthonormality check — is what exposes the bug.
    #[test]
    fn test_hermitian_evd_complex_eigenpairs_5x5() {
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

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        let u = evd.eigenvectors();
        let n = 5;

        // Eigenvalues real and ascending.
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

        // Full reconstruction A = U·D·U^H (only correct once the phase is restored).
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

    /// Regression test with a genuine cluster of repeated eigenvalues.
    ///
    /// `A = 2·I + w·wᴴ` (complex `w`) has eigenvalue 2 with multiplicity 3 and
    /// `2 + ‖w‖²` once. Degenerate subspaces are where a dropped phase most easily
    /// corrupts the eigenvectors, so every eigenpair is checked against `A·v = λ·v`.
    #[test]
    fn test_hermitian_evd_complex_clustered_eigenvalues() {
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

        let evd = HermitianEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        let u = evd.eigenvectors();

        // ‖w‖² = 1 + 1 + 2 + 5 = 9  =>  spectrum {2, 2, 2, 11}.
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
