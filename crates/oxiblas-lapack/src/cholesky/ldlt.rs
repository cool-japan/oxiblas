//! LDL^T Decomposition.
//!
//! Factorization for symmetric matrices (not necessarily positive definite):
//! A = LDL^T where L is lower triangular with unit diagonal and D is diagonal.
//!
//! This is more general than Cholesky (LLT) as it works for indefinite matrices.

use num_traits::{FromPrimitive, One};
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error returned when LDLT decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdltError {
    /// The matrix is singular or has zero pivot.
    Singular {
        /// The row/column index where the singularity was detected.
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

impl core::fmt::Display for LdltError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LdltError::Singular { index } => {
                write!(f, "Matrix is singular at index {index}")
            }
            LdltError::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}×{ncols}")
            }
            LdltError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for LdltError {}

/// LDL^T Decomposition for symmetric matrices.
///
/// For a symmetric matrix A, computes L and D such that A = LDL^T,
/// where L is unit lower triangular and D is diagonal.
///
/// This factorization works for indefinite symmetric matrices, unlike
/// the standard Cholesky (LLT) which requires positive definiteness.
///
/// # Advantages over LLT
///
/// - Works for symmetric indefinite matrices
/// - No square roots required (better numerical stability for some matrices)
/// - Can detect inertia (number of positive/negative/zero eigenvalues)
///
/// # Limitations
///
/// - Fails for singular matrices (when a diagonal element becomes zero)
/// - For highly indefinite matrices, consider Bunch-Kaufman pivoting
#[derive(Clone, Debug)]
pub struct Ldlt<T: Scalar> {
    /// Combined storage: L (strictly lower) and D (diagonal).
    /// The diagonal of `ld` stores D.
    /// The strictly lower triangular part stores L (unit diagonal is implicit).
    ld: Mat<T>,
}

impl<T: Field + Real + bytemuck::Zeroable> Ldlt<T> {
    /// Computes the LDL^T decomposition of a symmetric matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - A symmetric matrix (only lower triangle is used)
    ///
    /// # Errors
    ///
    /// Returns `LdltError::NotSquare` if the matrix is not square.
    /// Returns `LdltError::Singular` if a zero pivot is encountered.
    ///
    /// # Note
    ///
    /// Only the lower triangular part of `a` is used. The matrix is assumed
    /// to be symmetric.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, LdltError> {
        Self::compute_with_tol(a, None)
    }

    /// Computes the LDL^T decomposition with a custom tolerance.
    ///
    /// # Arguments
    ///
    /// * `a` - A symmetric matrix
    /// * `tol` - Optional tolerance for detecting singularity
    pub fn compute_with_tol(a: MatRef<'_, T>, tol: Option<T>) -> Result<Self, LdltError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(LdltError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Ldlt {
                ld: Mat::zeros(0, 0),
            });
        }

        // Copy lower triangle of A
        let mut ld = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                ld[(i, j)] = a[(i, j)];
            }
        }

        // Default tolerance
        let default_tol = <T as Scalar>::epsilon()
            * <T as FromPrimitive>::from_usize(n).unwrap_or(<T as One>::one());
        let tolerance = tol.unwrap_or(default_tol);

        // LDL^T factorization (no pivoting)
        for j in 0..n {
            // Compute D[j] = A[j,j] - sum_{k<j} L[j,k]^2 * D[k]
            let mut d_j = ld[(j, j)];
            for k in 0..j {
                let l_jk = ld[(j, k)];
                let d_k = ld[(k, k)];
                d_j = d_j - l_jk * l_jk * d_k;
            }

            // Check for singularity
            if Scalar::abs(d_j) <= tolerance {
                return Err(LdltError::Singular { index: j });
            }

            ld[(j, j)] = d_j;

            // Compute L[i,j] for i > j
            // L[i,j] = (A[i,j] - sum_{k<j} L[i,k] * D[k] * L[j,k]) / D[j]
            for i in (j + 1)..n {
                let mut l_ij = ld[(i, j)];
                for k in 0..j {
                    l_ij = l_ij - ld[(i, k)] * ld[(k, k)] * ld[(j, k)];
                }
                ld[(i, j)] = l_ij / d_j;
            }
        }

        Ok(Ldlt { ld })
    }

    /// Returns the size of the matrix (n for an n×n matrix).
    #[inline]
    pub fn size(&self) -> usize {
        self.ld.nrows()
    }

    /// Extracts the L factor (unit lower triangular).
    pub fn l_factor(&self) -> Mat<T> {
        let n = self.size();
        let mut l = Mat::zeros(n, n);

        for i in 0..n {
            l[(i, i)] = T::one(); // Unit diagonal
            for j in 0..i {
                l[(i, j)] = self.ld[(i, j)];
            }
        }

        l
    }

    /// Extracts the D factor (diagonal matrix as a vector).
    pub fn d_diagonal(&self) -> Vec<T> {
        let n = self.size();
        (0..n).map(|i| self.ld[(i, i)]).collect()
    }

    /// Extracts the D factor as a diagonal matrix.
    pub fn d_factor(&self) -> Mat<T> {
        let n = self.size();
        let mut d = Mat::zeros(n, n);
        for i in 0..n {
            d[(i, i)] = self.ld[(i, i)];
        }
        d
    }

    /// Computes the determinant of the original matrix.
    ///
    /// For A = LDL^T, det(A) = det(D) = ∏ D\[i\]
    pub fn determinant(&self) -> T {
        let n = self.size();
        if n == 0 {
            return T::one();
        }

        let mut det = T::one();
        for i in 0..n {
            det = det * self.ld[(i, i)];
        }

        det
    }

    /// Returns the inertia (number of positive, negative, and zero eigenvalues).
    ///
    /// For a symmetric matrix, the inertia is (n+, n-, n0) where the diagonal
    /// elements of D have the same sign distribution as the eigenvalues.
    ///
    /// Returns (positive_count, negative_count, zero_count).
    pub fn inertia(&self) -> (usize, usize, usize) {
        let n = self.size();
        let mut pos = 0;
        let mut neg = 0;
        let zero_tol = <T as Scalar>::epsilon();

        for i in 0..n {
            let d = self.ld[(i, i)];
            if d > zero_tol {
                pos += 1;
            } else if d < -zero_tol {
                neg += 1;
            }
        }

        (pos, neg, n - pos - neg)
    }

    /// Returns true if the original matrix is positive definite.
    ///
    /// A symmetric matrix is positive definite if all diagonal elements of D are positive.
    pub fn is_positive_definite(&self) -> bool {
        let (pos, _, zero) = self.inertia();
        pos == self.size() && zero == 0
    }

    /// Returns true if the original matrix is negative definite.
    pub fn is_negative_definite(&self) -> bool {
        let (_, neg, zero) = self.inertia();
        neg == self.size() && zero == 0
    }

    /// Solves the system Ax = b for symmetric A.
    ///
    /// Given A = LDL^T, solves:
    /// 1. Forward substitution: Ly = b (L has unit diagonal)
    /// 2. Diagonal scaling: Dz = y
    /// 3. Back substitution: L^T x = z
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × m for multiple RHS)
    ///
    /// # Errors
    ///
    /// Returns `LdltError::DimensionMismatch` if b has wrong number of rows.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, LdltError> {
        let n = self.size();

        if b.nrows() != n {
            return Err(LdltError::DimensionMismatch {
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

        // Forward substitution: Ly = b (L has unit diagonal)
        for k in 0..n {
            for i in (k + 1)..n {
                let mult = self.ld[(i, k)];
                for j in 0..m {
                    let val = x[(i, j)] - mult * x[(k, j)];
                    x[(i, j)] = val;
                }
            }
        }

        // Diagonal scaling: z = D^(-1) y
        for k in 0..n {
            let d_inv = T::one() / self.ld[(k, k)];
            for j in 0..m {
                x[(k, j)] = x[(k, j)] * d_inv;
            }
        }

        // Back substitution: L^T x = z
        for k in (0..n).rev() {
            for i in 0..k {
                let mult = self.ld[(k, i)];
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
    pub fn inverse(&self) -> Result<Mat<T>, LdltError> {
        let n = self.size();
        let identity = Mat::<T>::eye(n);
        self.solve(identity.as_ref())
    }

    /// Returns the log-absolute-determinant and sign of the determinant.
    ///
    /// Returns (log|det(A)|, sign) where sign is +1 or -1.
    /// This is useful for numerical stability when det(A) is very large or small.
    pub fn log_abs_determinant(&self) -> (T, i32) {
        let n = self.size();
        if n == 0 {
            return (T::zero(), 1);
        }

        let mut log_det = T::zero();
        let mut sign = 1i32;

        for i in 0..n {
            let d = self.ld[(i, i)];
            log_det = log_det + Real::ln(Scalar::abs(d));
            if d < T::zero() {
                sign *= -1;
            }
        }

        (log_det, sign)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldlt_positive_definite() {
        // A = [4 2; 2 5] is symmetric positive definite
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");

        // Verify L and D
        let l = ldlt.l_factor();
        let d = ldlt.d_diagonal();

        // L should be unit lower triangular
        assert!((l[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((l[(1, 1)] - 1.0).abs() < 1e-10);
        assert!(l[(0, 1)].abs() < 1e-10);

        // D should be positive
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);

        // Verify LDL^T = A
        let d_mat = ldlt.d_factor();
        let n = a.nrows();
        let mut ldlt_prod: Mat<f64> = Mat::zeros(n, n);

        // Compute L * D
        let mut ld: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    ld[(i, j)] = ld[(i, j)] + l[(i, k)] * d_mat[(k, j)];
                }
            }
        }

        // Compute (L * D) * L^T
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    ldlt_prod[(i, j)] = ldlt_prod[(i, j)] + ld[(i, k)] * l[(j, k)];
                }
            }
        }

        // Check LDL^T ≈ A
        for i in 0..n {
            for j in 0..n {
                let diff = ldlt_prod[(i, j)] - a[(i, j)];
                assert!(
                    diff.abs() < 1e-10,
                    "LDL^T[{},{}] = {}, A[{},{}] = {}",
                    i,
                    j,
                    ldlt_prod[(i, j)],
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ldlt_indefinite() {
        // A = [1 2; 2 1] has eigenvalues 3 and -1 (indefinite)
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");

        // Check inertia: 1 positive, 1 negative
        let (pos, neg, zero) = ldlt.inertia();
        assert_eq!(pos, 1);
        assert_eq!(neg, 1);
        assert_eq!(zero, 0);

        assert!(!ldlt.is_positive_definite());
        assert!(!ldlt.is_negative_definite());
    }

    #[test]
    fn test_ldlt_solve() {
        // A = [4 2; 2 5]
        // b = [8; 11]
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");
        let x = ldlt.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_ldlt_determinant() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");
        let det = ldlt.determinant();

        // det = 4*5 - 2*2 = 16
        assert!((det - 16.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_ldlt_inverse() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");
        let a_inv = ldlt.inverse().expect("Should invert");

        // det = 16
        // A^-1 = [5/16 -2/16; -2/16 4/16]
        assert!((a_inv[(0, 0)] - 0.3125).abs() < 1e-10);
        assert!((a_inv[(0, 1)] + 0.125).abs() < 1e-10);
        assert!((a_inv[(1, 0)] + 0.125).abs() < 1e-10);
        assert!((a_inv[(1, 1)] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_ldlt_3x3() {
        let a: Mat<f64> = Mat::from_rows(&[
            &[4.0, 12.0, -16.0],
            &[12.0, 37.0, -43.0],
            &[-16.0, -43.0, 98.0],
        ]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");

        // Verify by solving Ax = b and checking Ax = b
        let b: Mat<f64> = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let x = ldlt.solve(b.as_ref()).expect("Should solve");

        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!((ax_i - b[(i, 0)]).abs() < 1e-10, "Ax[{}] = {}", i, ax_i);
        }
    }

    #[test]
    fn test_ldlt_identity() {
        let eye: Mat<f64> = Mat::eye(3);

        let ldlt = Ldlt::compute(eye.as_ref()).expect("Should decompose");

        // D should be all ones
        let d = ldlt.d_diagonal();
        for di in d {
            assert!((di - 1.0).abs() < 1e-10);
        }

        // Determinant should be 1
        let det = ldlt.determinant();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ldlt_singular() {
        // Singular matrix
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);

        let result = Ldlt::compute(a.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_ldlt_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Empty should succeed");
        assert_eq!(ldlt.size(), 0);
        assert_eq!(ldlt.determinant(), 1.0);
    }

    #[test]
    fn test_ldlt_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = Ldlt::compute(a.as_ref());
        assert!(matches!(result, Err(LdltError::NotSquare { .. })));
    }

    #[test]
    fn test_ldlt_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");
        let x = ldlt.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-5, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-5, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_ldlt_log_abs_determinant() {
        // Indefinite matrix with det = -3
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);

        let ldlt = Ldlt::compute(a.as_ref()).expect("Should decompose");
        let (log_det, sign) = ldlt.log_abs_determinant();

        assert!((log_det - 3.0f64.ln()).abs() < 1e-10);
        assert_eq!(sign, -1);
    }
}
