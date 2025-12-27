//! Generalized Eigenvalue Decomposition.
//!
//! Solves the generalized eigenvalue problem: A * x = λ * B * x
//!
//! Provides:
//! - **SymmetricGeneralizedEvd**: For symmetric A and symmetric positive definite B.
//!   Reduces to standard EVD via Cholesky factorization.
//! - **GeneralizedEvd**: For general matrices A and B (requires QZ algorithm).

use crate::cholesky::Cholesky;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for generalized eigenvalue decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralizedEvdError {
    /// Matrix A is empty.
    EmptyMatrix,
    /// Matrices have incompatible dimensions.
    DimensionMismatch {
        /// Number of rows in A.
        nrows_a: usize,
        /// Number of columns in A.
        ncols_a: usize,
        /// Number of rows in B.
        nrows_b: usize,
        /// Number of columns in B.
        ncols_b: usize,
    },
    /// Matrix A is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix B is not positive definite.
    BNotPositiveDefinite,
    /// Algorithm did not converge.
    NotConverged,
    /// Matrix is singular.
    Singular,
}

impl core::fmt::Display for GeneralizedEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::DimensionMismatch {
                nrows_a,
                ncols_a,
                nrows_b,
                ncols_b,
            } => {
                write!(
                    f,
                    "Dimension mismatch: A is {nrows_a}x{ncols_a}, B is {nrows_b}x{ncols_b}"
                )
            }
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::BNotPositiveDefinite => {
                write!(f, "Matrix B is not positive definite")
            }
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::Singular => write!(f, "Matrix is singular"),
        }
    }
}

impl std::error::Error for GeneralizedEvdError {}

/// Symmetric generalized eigenvalue decomposition.
///
/// Solves A * x = λ * B * x where:
/// - A is symmetric
/// - B is symmetric positive definite
///
/// The problem is reduced to a standard eigenvalue problem:
/// 1. B = L * L^T (Cholesky factorization)
/// 2. C = L^(-1) * A * L^(-T)
/// 3. C * y = λ * y (standard symmetric EVD)
/// 4. x = L^(-T) * y
#[derive(Debug, Clone)]
pub struct SymmetricGeneralizedEvd<T: Scalar> {
    /// Generalized eigenvalues.
    eigenvalues: Vec<T>,
    /// Generalized eigenvectors (columns).
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> SymmetricGeneralizedEvd<T> {
    /// Maximum number of QR iterations.
    #[allow(dead_code)]
    const MAX_ITERATIONS: usize = 100;

    /// Computes the generalized eigenvalue decomposition.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix A
    /// * `b` - Symmetric positive definite matrix B
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::SymmetricGeneralizedEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[5.0f64, 2.0],
    ///     &[2.0, 3.0],
    /// ]);
    /// let b = Mat::from_rows(&[
    ///     &[2.0f64, 0.0],
    ///     &[0.0, 2.0],
    /// ]);
    ///
    /// let gevd = SymmetricGeneralizedEvd::compute(a.as_ref(), b.as_ref()).unwrap();
    /// let eigs = gevd.eigenvalues();
    /// ```
    pub fn compute(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Result<Self, GeneralizedEvdError> {
        let n = a.nrows();

        // Validate dimensions
        if n == 0 {
            return Err(GeneralizedEvdError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(GeneralizedEvdError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        if n != b.nrows() || n != b.ncols() {
            return Err(GeneralizedEvdError::DimensionMismatch {
                nrows_a: a.nrows(),
                ncols_a: a.ncols(),
                nrows_b: b.nrows(),
                ncols_b: b.ncols(),
            });
        }

        // Handle trivial case
        if n == 1 {
            let eigenvalue = a[(0, 0)] / b[(0, 0)];
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues: vec![eigenvalue],
                eigenvectors,
                n,
            });
        }

        // Step 1: Cholesky factorization of B
        let chol = Cholesky::compute(b).map_err(|_| GeneralizedEvdError::BNotPositiveDefinite)?;
        let l = chol.l_factor();
        let l_ref = l.as_ref();

        // Step 2: Compute C = L^(-1) * A * L^(-T)
        // First compute L^(-1) * A using forward substitution
        let mut temp = Mat::zeros(n, n);
        for j in 0..n {
            // Solve L * x = A[:, j]
            let col_a = column_to_vec(&a, j);
            let col_temp = solve_lower_triangular(l_ref, &col_a);
            for i in 0..n {
                temp[(i, j)] = col_temp[i];
            }
        }

        // Then compute (L^(-1) * A) * L^(-T) = temp * L^(-T)
        // This is equivalent to solving L^T * X^T = temp^T
        let mut c = Mat::zeros(n, n);
        for i in 0..n {
            // Row i of C = (L^(-1) * (row i of temp^T))^T
            // = Solve L^T * x = temp[i, :]
            let row_temp: Vec<T> = (0..n).map(|j| temp[(i, j)]).collect();
            let row_c = solve_lower_transpose_triangular(l_ref, &row_temp);
            for j in 0..n {
                c[(i, j)] = row_c[j];
            }
        }

        // Make C symmetric (numerical cleanup)
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (c[(i, j)] + c[(j, i)]) / (T::one() + T::one());
                c[(i, j)] = avg;
                c[(j, i)] = avg;
            }
        }

        // Step 3: Compute eigenvalues and eigenvectors of C
        let evd = super::SymmetricEvd::compute(c.as_ref())
            .map_err(|_| GeneralizedEvdError::NotConverged)?;

        // Step 4: Transform eigenvectors back: x = L^(-T) * y
        let y_vecs = evd.eigenvectors();
        let mut eigenvectors = Mat::zeros(n, n);

        for j in 0..n {
            let y_col: Vec<T> = (0..n).map(|i| y_vecs[(i, j)]).collect();
            let x_col = solve_lower_transpose_triangular(l_ref, &y_col);
            for i in 0..n {
                eigenvectors[(i, j)] = x_col[i];
            }
        }

        // Note: B-normalization is optional
        // The eigenvectors already satisfy A*x = λ*B*x after the transformation

        Ok(Self {
            eigenvalues: evd.eigenvalues().to_vec(),
            eigenvectors,
            n,
        })
    }

    /// Returns the generalized eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the generalized eigenvectors (as columns).
    pub fn eigenvectors(&self) -> MatRef<'_, T> {
        self.eigenvectors.as_ref()
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Verifies the decomposition: A * V ≈ B * V * D
    ///
    /// Returns the maximum absolute error.
    pub fn verify(&self, a: MatRef<'_, T>, b: MatRef<'_, T>) -> T {
        let n = self.n;
        let mut max_error = T::zero();

        for j in 0..n {
            let lambda = self.eigenvalues[j];

            for i in 0..n {
                // Compute (A * v)[i]
                let mut av_i = T::zero();
                for k in 0..n {
                    av_i = av_i + a[(i, k)] * self.eigenvectors[(k, j)];
                }

                // Compute (λ * B * v)[i]
                let mut bv_i = T::zero();
                for k in 0..n {
                    bv_i = bv_i + b[(i, k)] * self.eigenvectors[(k, j)];
                }
                let lambda_bv_i = lambda * bv_i;

                let error = Scalar::abs(av_i - lambda_bv_i);
                if error > max_error {
                    max_error = error;
                }
            }
        }

        max_error
    }
}

/// Generalized eigenvalue decomposition for general matrices.
///
/// Solves A * x = λ * B * x where A and B are general (possibly non-symmetric) matrices.
///
/// Uses the QZ algorithm (generalized Schur decomposition) when available,
/// otherwise falls back to explicit matrix inverse (less stable).
#[derive(Debug, Clone)]
pub struct GeneralizedEvd<T: Scalar> {
    /// Real parts of eigenvalues.
    eigenvalues_real: Vec<T>,
    /// Imaginary parts of eigenvalues.
    eigenvalues_imag: Vec<T>,
    /// Right eigenvectors (columns).
    right_eigenvectors: Option<Mat<T>>,
    /// Left eigenvectors (columns).
    #[allow(dead_code)]
    left_eigenvectors: Option<Mat<T>>,
    /// Matrix dimension.
    n: usize,
    /// Finite eigenvalue flags.
    is_finite: Vec<bool>,
}

impl<T: Field + Real + bytemuck::Zeroable> GeneralizedEvd<T> {
    /// Computes the generalized eigenvalues of A * x = λ * B * x.
    ///
    /// This is a fallback implementation that inverts B and solves B^(-1) * A.
    /// For robust computation, use the QZ algorithm.
    ///
    /// # Warning
    ///
    /// This implementation may be numerically unstable when B is nearly singular.
    pub fn compute(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Result<Self, GeneralizedEvdError> {
        let n = a.nrows();

        // Validate dimensions
        if n == 0 {
            return Err(GeneralizedEvdError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(GeneralizedEvdError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        if n != b.nrows() || n != b.ncols() {
            return Err(GeneralizedEvdError::DimensionMismatch {
                nrows_a: a.nrows(),
                ncols_a: a.ncols(),
                nrows_b: b.nrows(),
                ncols_b: b.ncols(),
            });
        }

        // Handle trivial case
        if n == 1 {
            let b00 = b[(0, 0)];
            if Scalar::abs(b00) <= <T as Scalar>::epsilon() {
                // Infinite eigenvalue
                return Ok(Self {
                    eigenvalues_real: vec![T::zero()],
                    eigenvalues_imag: vec![T::zero()],
                    right_eigenvectors: None,
                    left_eigenvectors: None,
                    n,
                    is_finite: vec![false],
                });
            }

            let eigenvalue = a[(0, 0)] / b00;
            return Ok(Self {
                eigenvalues_real: vec![eigenvalue],
                eigenvalues_imag: vec![T::zero()],
                right_eigenvectors: None,
                left_eigenvectors: None,
                n,
                is_finite: vec![true],
            });
        }

        // Compute B^(-1) * A using LU factorization
        let lu = crate::lu::Lu::compute(b).map_err(|_| GeneralizedEvdError::Singular)?;

        // Solve B * X = A for X = B^(-1) * A
        let mut c = Mat::zeros(n, n);
        for j in 0..n {
            // Create column matrix from column j of A
            let mut col_a = Mat::zeros(n, 1);
            for i in 0..n {
                col_a[(i, 0)] = a[(i, j)];
            }
            let col_c = lu
                .solve(col_a.as_ref())
                .map_err(|_| GeneralizedEvdError::Singular)?;
            for i in 0..n {
                c[(i, j)] = col_c[(i, 0)];
            }
        }

        // Compute eigenvalues of C = B^(-1) * A
        let evd = super::GeneralEvd::eigenvalues_only(c.as_ref())
            .map_err(|_| GeneralizedEvdError::NotConverged)?;

        let eigenvalues = evd.eigenvalues();
        let real_parts: Vec<T> = eigenvalues.iter().map(|e| e.real).collect();
        let imag_parts: Vec<T> = eigenvalues.iter().map(|e| e.imag).collect();

        Ok(Self {
            eigenvalues_real: real_parts,
            eigenvalues_imag: imag_parts,
            right_eigenvectors: None,
            left_eigenvectors: None,
            n,
            is_finite: vec![true; n],
        })
    }

    /// Computes generalized eigenvalues and eigenvectors.
    pub fn compute_with_eigenvectors(
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
    ) -> Result<Self, GeneralizedEvdError> {
        let n = a.nrows();

        // Validate dimensions
        if n == 0 {
            return Err(GeneralizedEvdError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(GeneralizedEvdError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        if n != b.nrows() || n != b.ncols() {
            return Err(GeneralizedEvdError::DimensionMismatch {
                nrows_a: a.nrows(),
                ncols_a: a.ncols(),
                nrows_b: b.nrows(),
                ncols_b: b.ncols(),
            });
        }

        // Compute B^(-1) * A
        let lu = crate::lu::Lu::compute(b).map_err(|_| GeneralizedEvdError::Singular)?;

        let mut c = Mat::zeros(n, n);
        for j in 0..n {
            let mut col_a = Mat::zeros(n, 1);
            for i in 0..n {
                col_a[(i, 0)] = a[(i, j)];
            }
            let col_c = lu
                .solve(col_a.as_ref())
                .map_err(|_| GeneralizedEvdError::Singular)?;
            for i in 0..n {
                c[(i, j)] = col_c[(i, 0)];
            }
        }

        // Compute eigenvalues and eigenvectors of C
        let evd = super::GeneralEvd::compute(c.as_ref())
            .map_err(|_| GeneralizedEvdError::NotConverged)?;

        let eigenvalues = evd.eigenvalues();
        let real_parts: Vec<T> = eigenvalues.iter().map(|e| e.real).collect();
        let imag_parts: Vec<T> = eigenvalues.iter().map(|e| e.imag).collect();

        // The eigenvectors of C are the right eigenvectors of the pencil (A, B)
        let right_eigenvectors = if let Some(vecs) = evd.eigenvectors_real() {
            let mut v = Mat::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    v[(i, j)] = vecs[(i, j)];
                }
            }
            Some(v)
        } else {
            None
        };

        Ok(Self {
            eigenvalues_real: real_parts,
            eigenvalues_imag: imag_parts,
            right_eigenvectors,
            left_eigenvectors: None,
            n,
            is_finite: vec![true; n],
        })
    }

    /// Returns the real parts of the eigenvalues.
    pub fn eigenvalues_real(&self) -> &[T] {
        &self.eigenvalues_real
    }

    /// Returns the imaginary parts of the eigenvalues.
    pub fn eigenvalues_imag(&self) -> &[T] {
        &self.eigenvalues_imag
    }

    /// Returns (real, imag) pairs of eigenvalues.
    pub fn eigenvalues_as_pairs(&self) -> (&[T], &[T]) {
        (&self.eigenvalues_real, &self.eigenvalues_imag)
    }

    /// Returns the right eigenvectors if computed.
    pub fn right_eigenvectors(&self) -> Option<MatRef<'_, T>> {
        self.right_eigenvectors.as_ref().map(|m| m.as_ref())
    }

    /// Returns whether each eigenvalue is finite.
    pub fn is_finite(&self) -> &[bool] {
        &self.is_finite
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Extracts a column as a vector.
fn column_to_vec<T: Scalar>(m: &MatRef<'_, T>, j: usize) -> Vec<T> {
    (0..m.nrows()).map(|i| m[(i, j)]).collect()
}

/// Solves L * x = b for lower triangular L.
fn solve_lower_triangular<T: Field + Real>(l: MatRef<'_, T>, b: &[T]) -> Vec<T> {
    let n = l.nrows();
    let mut x = b.to_vec();

    for i in 0..n {
        for j in 0..i {
            x[i] = x[i] - l[(i, j)] * x[j];
        }
        x[i] = x[i] / l[(i, i)];
    }

    x
}

/// Solves L^T * x = b for lower triangular L.
fn solve_lower_transpose_triangular<T: Field + Real>(l: MatRef<'_, T>, b: &[T]) -> Vec<T> {
    let n = l.nrows();
    let mut x = b.to_vec();

    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] = x[i] - l[(j, i)] * x[j];
        }
        x[i] = x[i] / l[(i, i)];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_generalized_evd() {
        // Simple test: A = I, B = I => eigenvalues are 1
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let gevd = SymmetricGeneralizedEvd::compute(a.as_ref(), b.as_ref()).unwrap();
        let eigs = gevd.eigenvalues();

        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_generalized_evd_scaled() {
        // A = 2*I, B = I => eigenvalues are 2
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 2.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let gevd = SymmetricGeneralizedEvd::compute(a.as_ref(), b.as_ref()).unwrap();
        let eigs = gevd.eigenvalues();

        assert!((eigs[0] - 2.0).abs() < 1e-10);
        assert!((eigs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_generalized_evd_general() {
        // A = [[5, 2], [2, 3]], B = [[2, 0], [0, 2]]
        // Eigenvalues of (A, B) = eigenvalues of A/2 = eigenvalues of [[2.5, 1], [1, 1.5]]
        let a = Mat::from_rows(&[&[5.0f64, 2.0], &[2.0, 3.0]]);
        let b = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 2.0]]);

        let gevd = SymmetricGeneralizedEvd::compute(a.as_ref(), b.as_ref()).unwrap();
        let eigs = gevd.eigenvalues();

        // Eigenvalues of [[2.5, 1], [1, 1.5]]:
        // tr = 4, det = 2.5*1.5 - 1 = 2.75
        // λ = (4 ± sqrt(16 - 11)) / 2 = (4 ± sqrt(5)) / 2
        // λ1 ≈ 0.882, λ2 ≈ 3.118
        let expected_1 = (4.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_2 = (4.0 + 5.0_f64.sqrt()) / 2.0;

        assert!((eigs[0] - expected_1).abs() < 1e-10);
        assert!((eigs[1] - expected_2).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_generalized_evd_verify_identity() {
        // With B = I, this should reduce to standard eigenvalue problem
        // Use 2x2 matrix for simplicity
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 2.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let gevd = SymmetricGeneralizedEvd::compute(a.as_ref(), b.as_ref()).unwrap();

        // The eigenvalues should be 1 and 3
        let eigs = gevd.eigenvalues();
        assert!(
            (eigs[0] - 1.0).abs() < 1e-10,
            "Expected eigenvalue 1, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 3.0).abs() < 1e-10,
            "Expected eigenvalue 3, got {}",
            eigs[1]
        );

        // Verify A * v = λ * B * v (with B = I, just A * v = λ * v)
        let error = gevd.verify(a.as_ref(), b.as_ref());
        assert!(error < 1e-8, "Verification error: {error}");
    }

    #[test]
    fn test_generalized_evd_simple() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let gevd = GeneralizedEvd::compute(a.as_ref(), b.as_ref()).unwrap();
        let real = gevd.eigenvalues_real();

        // Eigenvalues should be 2 and 3
        let mut eigs: Vec<f64> = real.to_vec();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((eigs[0] - 2.0).abs() < 1e-10);
        assert!((eigs[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_generalized_evd_with_eigenvectors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let gevd = GeneralizedEvd::compute_with_eigenvectors(a.as_ref(), b.as_ref()).unwrap();

        // With B = I, this is just standard EVD
        let real = gevd.eigenvalues_real();
        let mut eigs: Vec<f64> = real.to_vec();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Upper triangular, eigenvalues are diagonal: 1 and 3
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_generalized_evd_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let result = GeneralizedEvd::compute(a.as_ref(), b.as_ref());
        assert!(matches!(
            result,
            Err(GeneralizedEvdError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_symmetric_generalized_evd_non_spd() {
        // B is not positive definite
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[-1.0f64, 0.0], &[0.0, 1.0]]);

        let result = SymmetricGeneralizedEvd::compute(a.as_ref(), b.as_ref());
        assert!(matches!(
            result,
            Err(GeneralizedEvdError::BNotPositiveDefinite)
        ));
    }
}
