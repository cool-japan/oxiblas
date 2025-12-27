//! Gauss-Seidel, SOR, and SSOR preconditioners.

use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Gauss-Seidel preconditioner.
pub struct GaussSeidel<T: Scalar> {
    /// The matrix (stored for forward substitution)
    matrix: CsrMatrix<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> GaussSeidel<T> {
    /// Create a new Gauss-Seidel preconditioner.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or has zero diagonal elements.
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        // Check for zero diagonals
        let n = a.nrows();
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut found = false;
            for k in start..end {
                if a.col_indices()[k] == i {
                    let diag_val = a.values()[k].clone();
                    if Scalar::abs(diag_val) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                        return Err(PreconditionerError::ZeroDiagonal(i));
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(PreconditionerError::ZeroDiagonal(i));
            }
        }

        Ok(Self { matrix: a.clone() })
    }

    /// Apply the preconditioner: solve (D + L) z = r via forward substitution.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.matrix.nrows();
        assert_eq!(r.len(), n, "r length must match matrix size");
        assert_eq!(z.len(), n, "z length must match matrix size");

        // Forward substitution for (D + L) z = r
        for i in 0..n {
            let start = self.matrix.row_ptrs()[i];
            let end = self.matrix.row_ptrs()[i + 1];

            let mut sum = r[i].clone();
            let mut diag = T::zero();

            // Compute sum -= L * z (for j < i) and find diagonal
            for k in start..end {
                let j = self.matrix.col_indices()[k];
                let a_ij = self.matrix.values()[k].clone();

                if j < i {
                    // Lower triangular part
                    sum = sum - a_ij * z[j].clone();
                } else if j == i {
                    // Diagonal
                    diag = a_ij;
                }
            }

            // z[i] = sum / diag
            z[i] = sum / diag;
        }
    }

    /// Get the size of the preconditioner.
    pub fn size(&self) -> usize {
        self.matrix.nrows()
    }
}

/// SOR (Successive Over-Relaxation) preconditioner.
///
/// SOR is a weighted variant of Gauss-Seidel with relaxation parameter ω:
/// ```text
/// x^{k+1} = (1-ω) x^k + ω D^{-1} (b - L x^{k+1} - U x^k)
/// ```
///
/// For ω = 1, this reduces to Gauss-Seidel.
/// Optimal ω is usually in range (1, 2) for convergence acceleration.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::SOR;
///
/// let sor = SOR::new(&matrix, 1.5)?; // ω = 1.5
/// let mut z = vec![0.0; n];
/// sor.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct SOR<T: Scalar> {
    /// The matrix
    matrix: CsrMatrix<T>,
    /// Relaxation parameter ω
    omega: T,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> SOR<T> {
    /// Create a new SOR preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix
    /// * `omega` - Relaxation parameter (typically 1.0 to 2.0)
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or has zero diagonal elements.
    pub fn new(a: &CsrMatrix<T>, omega: T) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        // Check for zero diagonals
        let n = a.nrows();
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut found = false;
            for k in start..end {
                if a.col_indices()[k] == i {
                    let diag_val = a.values()[k].clone();
                    if Scalar::abs(diag_val) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                        return Err(PreconditionerError::ZeroDiagonal(i));
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(PreconditionerError::ZeroDiagonal(i));
            }
        }

        Ok(Self {
            matrix: a.clone(),
            omega,
        })
    }

    /// Apply the preconditioner: solve via SOR iteration.
    ///
    /// Performs one SOR sweep: z = (D + ωL)^{-1} r
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.matrix.nrows();
        assert_eq!(r.len(), n, "r length must match matrix size");
        assert_eq!(z.len(), n, "z length must match matrix size");

        // Forward SOR sweep
        for i in 0..n {
            let start = self.matrix.row_ptrs()[i];
            let end = self.matrix.row_ptrs()[i + 1];

            let mut sum = r[i].clone();
            let mut diag = T::zero();

            for k in start..end {
                let j = self.matrix.col_indices()[k];
                let a_ij = self.matrix.values()[k].clone();

                if j < i {
                    // Lower triangular: use updated z[j]
                    sum = sum - a_ij * z[j].clone();
                } else if j == i {
                    diag = a_ij;
                } else {
                    // Upper triangular: use old z[j] (which is 0 initially)
                    sum = sum - a_ij * z[j].clone();
                }
            }

            // z[i] = ω * sum / diag + (1 - ω) * z[i]
            // Since z starts at 0, this simplifies to z[i] = ω * sum / diag
            z[i] = self.omega.clone() * sum / diag;
        }
    }

    /// Get the size of the preconditioner.
    pub fn size(&self) -> usize {
        self.matrix.nrows()
    }
}

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner.
///
/// SSOR applies SOR forward then backward, creating a symmetric preconditioner:
/// ```text
/// M^{-1} = (D + ωL) D^{-1} (D + ωU)
/// ```
///
/// This symmetry is beneficial for conjugate gradient methods.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::SSOR;
///
/// let ssor = SSOR::new(&matrix, 1.5)?; // ω = 1.5
/// let mut z = vec![0.0; n];
/// ssor.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct SSOR<T: Scalar> {
    /// The matrix
    matrix: CsrMatrix<T>,
    /// Relaxation parameter ω
    omega: T,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> SSOR<T> {
    /// Create a new SSOR preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix (should be symmetric for best results)
    /// * `omega` - Relaxation parameter (typically 1.0 to 2.0)
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or has zero diagonal elements.
    pub fn new(a: &CsrMatrix<T>, omega: T) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        // Check for zero diagonals
        let n = a.nrows();
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut found = false;
            for k in start..end {
                if a.col_indices()[k] == i {
                    let diag_val = a.values()[k].clone();
                    if Scalar::abs(diag_val) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                        return Err(PreconditionerError::ZeroDiagonal(i));
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(PreconditionerError::ZeroDiagonal(i));
            }
        }

        Ok(Self {
            matrix: a.clone(),
            omega,
        })
    }

    /// Apply the preconditioner: SSOR sweep (forward then backward).
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.matrix.nrows();
        assert_eq!(r.len(), n, "r length must match matrix size");
        assert_eq!(z.len(), n, "z length must match matrix size");

        // Temporary storage for intermediate result
        let mut temp = vec![T::zero(); n];

        // Forward sweep: (D + ωL) temp = r
        for i in 0..n {
            let start = self.matrix.row_ptrs()[i];
            let end = self.matrix.row_ptrs()[i + 1];

            let mut sum = r[i].clone();
            let mut diag = T::zero();

            for k in start..end {
                let j = self.matrix.col_indices()[k];
                let a_ij = self.matrix.values()[k].clone();

                if j < i {
                    sum = sum - self.omega.clone() * a_ij * temp[j].clone();
                } else if j == i {
                    diag = a_ij;
                }
            }

            temp[i] = sum / diag;
        }

        // Backward sweep: (D + ωU) z = D * temp
        // Starting from the last row going backward
        for i in (0..n).rev() {
            let start = self.matrix.row_ptrs()[i];
            let end = self.matrix.row_ptrs()[i + 1];

            let mut diag = T::zero();
            let mut sum = T::zero();

            // Find diagonal and compute D * temp[i]
            for k in start..end {
                if self.matrix.col_indices()[k] == i {
                    diag = self.matrix.values()[k].clone();
                    sum = diag.clone() * temp[i].clone();
                    break;
                }
            }

            // Subtract ω * U * z (for j > i)
            for k in start..end {
                let j = self.matrix.col_indices()[k];
                let a_ij = self.matrix.values()[k].clone();

                if j > i {
                    sum = sum - self.omega.clone() * a_ij * z[j].clone();
                }
            }

            z[i] = sum / diag;
        }
    }

    /// Get the size of the preconditioner.
    pub fn size(&self) -> usize {
        self.matrix.nrows()
    }
}
