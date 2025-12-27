//! Hermitian Cholesky decomposition (LL^H) for complex matrices.
//!
//! For Hermitian positive definite complex matrices, computes L such that A = LL^H,
//! where L is lower triangular and L^H is the conjugate transpose of L.

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error returned when Hermitian Cholesky decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HermitianCholeskyError {
    /// The matrix is not positive definite.
    NotPositiveDefinite {
        /// The row/column index where failure was detected.
        index: usize,
    },
    /// The matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Dimension mismatch in solve operation.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for HermitianCholeskyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HermitianCholeskyError::NotPositiveDefinite { index } => {
                write!(
                    f,
                    "Matrix is not Hermitian positive definite (detected at index {index})"
                )
            }
            HermitianCholeskyError::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}×{ncols}")
            }
            HermitianCholeskyError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for HermitianCholeskyError {}

/// Hermitian Cholesky decomposition (LL^H factorization).
///
/// For a Hermitian positive definite matrix A, computes L such that A = LL^H,
/// where L is lower triangular with positive real diagonal entries.
#[derive(Clone, Debug)]
pub struct HermitianCholesky<T: Scalar> {
    /// The L factor (lower triangular).
    l: Mat<T>,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> HermitianCholesky<T>
where
    T::Real: Real,
{
    /// Computes the Hermitian Cholesky decomposition of a Hermitian positive definite matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - A Hermitian positive definite matrix
    ///
    /// # Errors
    ///
    /// Returns `HermitianCholeskyError::NotSquare` if the matrix is not square.
    /// Returns `HermitianCholeskyError::NotPositiveDefinite` if the matrix is not positive definite.
    ///
    /// # Note
    ///
    /// Only the lower triangular part of `a` is used. The matrix is assumed
    /// to be Hermitian (A = A^H).
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, HermitianCholeskyError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(HermitianCholeskyError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(HermitianCholesky {
                l: Mat::zeros(0, 0),
            });
        }

        let mut l: Mat<T> = Mat::zeros(n, n);

        // Cholesky-Banachiewicz algorithm for Hermitian matrices
        // For A = LL^H:
        //   L[j,j] = sqrt(A[j,j] - sum_k |L[j,k]|^2)  (diagonal is real and positive)
        //   L[i,j] = (A[i,j] - sum_k L[i,k] * conj(L[j,k])) / L[j,j]  for i > j
        for j in 0..n {
            // Compute diagonal element
            // sum = sum_k |L[j,k]|^2 = sum_k L[j,k] * conj(L[j,k])
            let mut sum_diag = T::Real::zero();
            for k in 0..j {
                sum_diag = sum_diag + l[(j, k)].abs_sq();
            }

            // For Hermitian positive definite, A[j,j] is real
            let a_jj_real = a[(j, j)].real();
            let diag_val = a_jj_real - sum_diag;

            // Check for positive definiteness
            let tol = <T::Real as Scalar>::epsilon()
                * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
            if diag_val <= tol {
                return Err(HermitianCholeskyError::NotPositiveDefinite { index: j });
            }

            let l_jj = <T::Real as Real>::sqrt(diag_val);
            l[(j, j)] = T::from_real(l_jj);

            // Compute off-diagonal elements L[i,j] for i > j
            for i in (j + 1)..n {
                // sum = sum_k L[i,k] * conj(L[j,k])
                let mut sum_off = T::zero();
                for k in 0..j {
                    sum_off = sum_off + l[(i, k)] * l[(j, k)].conj();
                }

                l[(i, j)] = (a[(i, j)] - sum_off) / l[(j, j)];
            }
        }

        Ok(HermitianCholesky { l })
    }

    /// Returns the size of the matrix (n for an n×n matrix).
    #[inline]
    pub fn size(&self) -> usize {
        self.l.nrows()
    }

    /// Returns the L factor (lower triangular).
    pub fn l_factor(&self) -> Mat<T> {
        self.l.clone()
    }

    /// Computes the determinant of the original matrix.
    ///
    /// For Hermitian A = LL^H, det(A) = |det(L)|^2 = (∏ L\[i,i\])^2
    /// Since diagonal elements are real and positive, det is real and positive.
    pub fn determinant(&self) -> T::Real {
        let n = self.size();
        if n == 0 {
            return T::Real::one();
        }

        let mut det_l = T::Real::one();
        for i in 0..n {
            // Diagonal elements are real and stored as such
            det_l = det_l * self.l[(i, i)].real();
        }

        det_l * det_l
    }

    /// Solves the system Ax = b for Hermitian positive definite A.
    ///
    /// Given A = LL^H, solves:
    /// 1. Forward substitution: Ly = b
    /// 2. Back substitution: L^H x = y
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × m for multiple RHS)
    ///
    /// # Errors
    ///
    /// Returns `HermitianCholeskyError::DimensionMismatch` if b has wrong number of rows.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, HermitianCholeskyError> {
        let n = self.size();

        if b.nrows() != n {
            return Err(HermitianCholeskyError::DimensionMismatch {
                expected: n,
                actual: b.nrows(),
            });
        }

        let m = b.ncols();
        let mut x = Mat::zeros(n, m);

        // Copy b to x
        for j in 0..m {
            for i in 0..n {
                x[(i, j)] = b[(i, j)];
            }
        }

        // Forward substitution: Ly = b
        for k in 0..n {
            for j in 0..m {
                x[(k, j)] = x[(k, j)] / self.l[(k, k)];
            }

            for i in (k + 1)..n {
                let mult = self.l[(i, k)];
                for j in 0..m {
                    let val = x[(i, j)] - mult * x[(k, j)];
                    x[(i, j)] = val;
                }
            }
        }

        // Back substitution: L^H x = y
        // L^H is conjugate transpose, so L^H[k,i] = conj(L[i,k])
        for k in (0..n).rev() {
            // Divide by conj(L[k,k]) which equals L[k,k] since diagonal is real
            for j in 0..m {
                x[(k, j)] = x[(k, j)] / self.l[(k, k)].conj();
            }

            for i in 0..k {
                // L^H[i,k] = conj(L[k,i])
                let mult = self.l[(k, i)].conj();
                for j in 0..m {
                    let val = x[(i, j)] - mult * x[(k, j)];
                    x[(i, j)] = val;
                }
            }
        }

        Ok(x)
    }

    /// Computes the inverse of the original matrix.
    ///
    /// Solves AX = I to find A^(-1).
    pub fn inverse(&self) -> Result<Mat<T>, HermitianCholeskyError> {
        let n = self.size();
        let identity = Mat::<T>::eye(n);
        self.solve(identity.as_ref())
    }

    /// Returns the log-determinant of the original matrix.
    ///
    /// This is useful for numerical stability when det(A) is very large or small.
    /// log(det(A)) = 2 * sum(log(L\[i,i\]))
    pub fn log_determinant(&self) -> T::Real {
        let n = self.size();
        if n == 0 {
            return T::Real::zero();
        }

        let mut log_det = T::Real::zero();
        let two = T::Real::one() + T::Real::one();
        for i in 0..n {
            log_det = log_det + <T::Real as Real>::ln(self.l[(i, i)].real());
        }

        two * log_det
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn test_hermitian_cholesky_simple() {
        // Hermitian positive definite matrix
        // A = [[4, 2-i], [2+i, 5]]
        // A is Hermitian since A[0,1] = conj(A[1,0])
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(2.0, -1.0)],
            &[Complex64::new(2.0, 1.0), Complex64::new(5.0, 0.0)],
        ]);

        let chol =
            HermitianCholesky::compute(a.as_ref()).expect("Should be Hermitian positive definite");
        let l = chol.l_factor();

        // Verify A = L L^H
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + l[(i, k)] * l[(j, k)].conj();
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "A[{},{}] mismatch: {:?} vs {:?}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_hermitian_cholesky_solve() {
        // Hermitian positive definite matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, 1.0)],
            &[Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);

        let b: Mat<Complex64> =
            Mat::from_rows(&[&[Complex64::new(5.0, 1.0)], &[Complex64::new(4.0, -1.0)]]);

        let chol =
            HermitianCholesky::compute(a.as_ref()).expect("Should be Hermitian positive definite");
        let x = chol.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let n = a.nrows();
        for i in 0..n {
            let mut ax_i = Complex64::new(0.0, 0.0);
            for k in 0..n {
                ax_i = ax_i + a[(i, k)] * x[(k, 0)];
            }
            let diff = (ax_i - b[(i, 0)]).norm();
            assert!(
                diff < 1e-10,
                "Ax[{}] = {:?}, b[{}] = {:?}",
                i,
                ax_i,
                i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_hermitian_cholesky_determinant() {
        // A = [[2, i], [-i, 2]] is Hermitian positive definite
        // det(A) = 2*2 - i*(-i) = 4 - 1 = 3
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            &[Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);

        let chol =
            HermitianCholesky::compute(a.as_ref()).expect("Should be Hermitian positive definite");
        let det = chol.determinant();

        assert!((det - 3.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_hermitian_cholesky_inverse() {
        // A = [[2, i], [-i, 2]]
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            &[Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);

        let chol =
            HermitianCholesky::compute(a.as_ref()).expect("Should be Hermitian positive definite");
        let a_inv = chol.inverse().expect("Should invert");

        // Verify A * A^-1 = I
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + a[(i, k)] * a_inv[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(diff < 1e-10, "A*A^-1[{},{}] = {:?}", i, j, sum);
            }
        }
    }

    #[test]
    fn test_hermitian_cholesky_not_positive_definite() {
        // A = [[1, 2], [2, 1]] - symmetric but not positive definite (eigenvalues -1, 3)
        // In complex form:
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        let result = HermitianCholesky::compute(a.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_hermitian_cholesky_3x3() {
        // 3x3 Hermitian positive definite matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(1.0, -1.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(2.0, -1.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
        ]);

        let chol =
            HermitianCholesky::compute(a.as_ref()).expect("Should be Hermitian positive definite");
        let l = chol.l_factor();

        // Verify A = L L^H
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + l[(i, k)] * l[(j, k)].conj();
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "A[{},{}] mismatch: {:?} vs {:?}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_hermitian_cholesky_complex32() {
        // Test with Complex32
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(4.0, 0.0), Complex32::new(1.0, 1.0)],
            &[Complex32::new(1.0, -1.0), Complex32::new(3.0, 0.0)],
        ]);

        let b: Mat<Complex32> =
            Mat::from_rows(&[&[Complex32::new(5.0, 1.0)], &[Complex32::new(4.0, -1.0)]]);

        let chol =
            HermitianCholesky::compute(a.as_ref()).expect("Should be Hermitian positive definite");
        let x = chol.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let n = a.nrows();
        for i in 0..n {
            let mut ax_i = Complex32::new(0.0, 0.0);
            for k in 0..n {
                ax_i = ax_i + a[(i, k)] * x[(k, 0)];
            }
            let diff = (ax_i - b[(i, 0)]).norm();
            assert!(
                diff < 1e-5,
                "Ax[{}] = {:?}, b[{}] = {:?}",
                i,
                ax_i,
                i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_hermitian_cholesky_identity() {
        // Complex identity matrix
        let a: Mat<Complex64> = Mat::eye(3);

        let chol = HermitianCholesky::compute(a.as_ref())
            .expect("Identity is Hermitian positive definite");
        let l = chol.l_factor();

        // L should also be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (l[(i, j)] - expected).norm();
                assert!(diff < 1e-10, "L[{},{}] = {:?}", i, j, l[(i, j)]);
            }
        }
    }
}
