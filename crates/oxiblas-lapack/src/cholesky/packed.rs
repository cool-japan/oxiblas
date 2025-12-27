//! Packed storage format for symmetric matrices.
//!
//! Packed format stores only the upper or lower triangle of a symmetric matrix
//! in a 1D array, reducing memory usage by about half.
//!
//! For an n×n symmetric matrix:
//! - Storage required: n*(n+1)/2 elements
//! - Lower packed: A[i,j] with i >= j stored at index j*n - j*(j+1)/2 + i
//! - Upper packed: A[i,j] with i <= j stored at index i + j*(j+1)/2
//!
//! # Factorizations
//!
//! - **PackedCholesky** (pptrf/ppsv): For symmetric positive definite matrices
//! - **PackedLdlt** (sptrf/spsv): For general symmetric matrices (may be indefinite)

use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Triangle storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uplo {
    /// Lower triangle stored (L)
    Lower,
    /// Upper triangle stored (U)
    Upper,
}

/// Error for packed Cholesky operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedCholeskyError {
    /// Matrix is not positive definite.
    NotPositiveDefinite {
        /// Index where failure was detected.
        index: usize,
    },
    /// Invalid packed array size.
    InvalidSize {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Dimension mismatch in solve.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for PackedCholeskyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PackedCholeskyError::NotPositiveDefinite { index } => {
                write!(
                    f,
                    "Matrix is not positive definite (detected at index {index})"
                )
            }
            PackedCholeskyError::InvalidSize { expected, actual } => {
                write!(f, "Invalid packed size: expected {expected}, got {actual}")
            }
            PackedCholeskyError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PackedCholeskyError {}

/// Error for packed LDL^T operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedLdltError {
    /// Matrix is singular.
    Singular {
        /// Index where zero pivot was found.
        index: usize,
    },
    /// Invalid packed array size.
    InvalidSize {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Dimension mismatch in solve.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for PackedLdltError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PackedLdltError::Singular { index } => {
                write!(f, "Matrix is singular (zero pivot at index {index})")
            }
            PackedLdltError::InvalidSize { expected, actual } => {
                write!(f, "Invalid packed size: expected {expected}, got {actual}")
            }
            PackedLdltError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PackedLdltError {}

// =============================================================================
// Packed storage indexing utilities
// =============================================================================

/// Returns the index in lower-packed storage for element (i, j) where i >= j.
#[inline]
pub fn packed_lower_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    debug_assert!(i < n);
    // Column j starts at: j*n - j*(j+1)/2
    // Then offset by (i - j) to reach row i
    j * n - (j * (j + 1)) / 2 + i
}

/// Returns the index in upper-packed storage for element (i, j) where i <= j.
#[inline]
pub fn packed_upper_index(_n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    // Element (i, j) with i <= j: sum of (1 + 2 + ... + j) + i = j*(j+1)/2 + i
    (j * (j + 1)) / 2 + i
}

/// Converts a dense symmetric matrix to lower-packed format.
///
/// Only the lower triangle is used.
pub fn dense_to_packed_lower<T: Scalar>(a: MatRef<'_, T>) -> Vec<T> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());

    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    for j in 0..n {
        for i in j..n {
            ap[packed_lower_index(n, i, j)] = a[(i, j)];
        }
    }

    ap
}

/// Converts a dense symmetric matrix to upper-packed format.
///
/// Only the upper triangle is used.
pub fn dense_to_packed_upper<T: Scalar>(a: MatRef<'_, T>) -> Vec<T> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());

    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    for j in 0..n {
        for i in 0..=j {
            ap[packed_upper_index(n, i, j)] = a[(i, j)];
        }
    }

    ap
}

/// Converts lower-packed format to a dense symmetric matrix.
pub fn packed_lower_to_dense<T: Scalar + bytemuck::Zeroable>(ap: &[T], n: usize) -> Mat<T> {
    let mut a = Mat::zeros(n, n);

    for j in 0..n {
        for i in j..n {
            let val = ap[packed_lower_index(n, i, j)];
            a[(i, j)] = val;
            a[(j, i)] = val; // Symmetric
        }
    }

    a
}

/// Converts upper-packed format to a dense symmetric matrix.
pub fn packed_upper_to_dense<T: Scalar + bytemuck::Zeroable>(ap: &[T], n: usize) -> Mat<T> {
    let mut a = Mat::zeros(n, n);

    for j in 0..n {
        for i in 0..=j {
            let val = ap[packed_upper_index(n, i, j)];
            a[(i, j)] = val;
            a[(j, i)] = val; // Symmetric
        }
    }

    a
}

// =============================================================================
// Packed Cholesky (pptrf/ppsv)
// =============================================================================

/// Packed Cholesky factorization (pptrf).
///
/// For a symmetric positive definite matrix A stored in packed format,
/// computes the Cholesky factorization A = L*L^T or A = U^T*U.
#[derive(Clone, Debug)]
pub struct PackedCholesky<T: Scalar> {
    /// Packed factor (L or U depending on uplo).
    factor: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Which triangle was stored/computed.
    uplo: Uplo,
}

impl<T: Field + Real + bytemuck::Zeroable> PackedCholesky<T> {
    /// Computes packed Cholesky factorization (pptrf).
    ///
    /// # Arguments
    ///
    /// * `ap` - The packed symmetric positive definite matrix
    /// * `n` - The matrix dimension
    /// * `uplo` - Whether the upper or lower triangle is stored
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::cholesky::{PackedCholesky, dense_to_packed_lower, Uplo};
    /// use oxiblas_matrix::Mat;
    ///
    /// // Symmetric positive definite matrix
    /// let a: Mat<f64> = Mat::from_rows(&[
    ///     &[4.0, 2.0],
    ///     &[2.0, 5.0],
    /// ]);
    ///
    /// // Convert to packed lower format
    /// let ap = dense_to_packed_lower(a.as_ref());
    /// assert_eq!(ap.len(), 3); // 2*(2+1)/2 = 3
    ///
    /// // Compute packed Cholesky
    /// let chol = PackedCholesky::compute(&ap, 2, Uplo::Lower).unwrap();
    /// ```
    pub fn compute(ap: &[T], n: usize, uplo: Uplo) -> Result<Self, PackedCholeskyError> {
        let expected_size = n * (n + 1) / 2;
        if ap.len() != expected_size {
            return Err(PackedCholeskyError::InvalidSize {
                expected: expected_size,
                actual: ap.len(),
            });
        }

        if n == 0 {
            return Ok(PackedCholesky {
                factor: Vec::new(),
                n: 0,
                uplo,
            });
        }

        let mut factor = ap.to_vec();

        match uplo {
            Uplo::Lower => {
                // Lower triangle: A = L * L^T
                for j in 0..n {
                    // Compute L[j,j]
                    let diag_idx = packed_lower_index(n, j, j);
                    let mut ajj = factor[diag_idx];

                    // ajj = a[j,j] - sum_{k<j} L[j,k]^2
                    for k in 0..j {
                        let ljk = factor[packed_lower_index(n, j, k)];
                        ajj = ajj - ljk * ljk;
                    }

                    let tol = <T as Scalar>::epsilon()
                        * <T as FromPrimitive>::from_usize(n).unwrap_or(T::one());
                    if ajj <= tol {
                        return Err(PackedCholeskyError::NotPositiveDefinite { index: j });
                    }

                    factor[diag_idx] = Real::sqrt(ajj);
                    let ljj = factor[diag_idx];

                    // Update L[i,j] for i > j
                    for i in (j + 1)..n {
                        let mut aij = factor[packed_lower_index(n, i, j)];

                        // aij = (a[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
                        for k in 0..j {
                            let lik = factor[packed_lower_index(n, i, k)];
                            let ljk = factor[packed_lower_index(n, j, k)];
                            aij = aij - lik * ljk;
                        }

                        factor[packed_lower_index(n, i, j)] = aij / ljj;
                    }
                }
            }
            Uplo::Upper => {
                // Upper triangle: A = U^T * U
                for j in 0..n {
                    // Compute U[j,j]
                    let diag_idx = packed_upper_index(n, j, j);
                    let mut ajj = factor[diag_idx];

                    // ajj = a[j,j] - sum_{k<j} U[k,j]^2
                    for k in 0..j {
                        let ukj = factor[packed_upper_index(n, k, j)];
                        ajj = ajj - ukj * ukj;
                    }

                    let tol = <T as Scalar>::epsilon()
                        * <T as FromPrimitive>::from_usize(n).unwrap_or(T::one());
                    if ajj <= tol {
                        return Err(PackedCholeskyError::NotPositiveDefinite { index: j });
                    }

                    factor[diag_idx] = Real::sqrt(ajj);
                    let ujj = factor[diag_idx];

                    // Update U[j,i] for i > j
                    for i in (j + 1)..n {
                        let mut aji = factor[packed_upper_index(n, j, i)];

                        // aji = (a[j,i] - sum_{k<j} U[k,j]*U[k,i]) / U[j,j]
                        for k in 0..j {
                            let ukj = factor[packed_upper_index(n, k, j)];
                            let uki = factor[packed_upper_index(n, k, i)];
                            aji = aji - ukj * uki;
                        }

                        factor[packed_upper_index(n, j, i)] = aji / ujj;
                    }
                }
            }
        }

        Ok(PackedCholesky { factor, n, uplo })
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns the packed factor.
    pub fn factor(&self) -> &[T] {
        &self.factor
    }

    /// Returns which triangle was computed.
    pub fn uplo(&self) -> Uplo {
        self.uplo
    }

    /// Solves A*x = b using the packed Cholesky factorization (ppsv).
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector or matrix
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, PackedCholeskyError> {
        if b.nrows() != self.n {
            return Err(PackedCholeskyError::DimensionMismatch {
                expected: self.n,
                actual: b.nrows(),
            });
        }

        let m = b.ncols();
        let mut x = Mat::zeros(self.n, m);

        // Copy b to x
        for j in 0..m {
            for i in 0..self.n {
                x[(i, j)] = b[(i, j)];
            }
        }

        match self.uplo {
            Uplo::Lower => {
                // A = L * L^T
                // Solve L * y = b (forward substitution)
                for k in 0..self.n {
                    let lkk = self.factor[packed_lower_index(self.n, k, k)];
                    for j in 0..m {
                        x[(k, j)] = x[(k, j)] / lkk;
                    }
                    for i in (k + 1)..self.n {
                        let lik = self.factor[packed_lower_index(self.n, i, k)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - lik * x[(k, j)];
                        }
                    }
                }

                // Solve L^T * x = y (backward substitution)
                for k in (0..self.n).rev() {
                    let lkk = self.factor[packed_lower_index(self.n, k, k)];
                    for j in 0..m {
                        x[(k, j)] = x[(k, j)] / lkk;
                    }
                    for i in 0..k {
                        let lki = self.factor[packed_lower_index(self.n, k, i)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - lki * x[(k, j)];
                        }
                    }
                }
            }
            Uplo::Upper => {
                // A = U^T * U
                // Solve U^T * y = b (forward substitution)
                for k in 0..self.n {
                    let ukk = self.factor[packed_upper_index(self.n, k, k)];
                    for j in 0..m {
                        x[(k, j)] = x[(k, j)] / ukk;
                    }
                    for i in (k + 1)..self.n {
                        let uki = self.factor[packed_upper_index(self.n, k, i)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - uki * x[(k, j)];
                        }
                    }
                }

                // Solve U * x = y (backward substitution)
                for k in (0..self.n).rev() {
                    let ukk = self.factor[packed_upper_index(self.n, k, k)];
                    for j in 0..m {
                        x[(k, j)] = x[(k, j)] / ukk;
                    }
                    for i in 0..k {
                        let uik = self.factor[packed_upper_index(self.n, i, k)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - uik * x[(k, j)];
                        }
                    }
                }
            }
        }

        Ok(x)
    }

    /// Computes the determinant of the original matrix.
    pub fn determinant(&self) -> T {
        if self.n == 0 {
            return T::one();
        }

        let mut det = T::one();
        for i in 0..self.n {
            let diag = match self.uplo {
                Uplo::Lower => self.factor[packed_lower_index(self.n, i, i)],
                Uplo::Upper => self.factor[packed_upper_index(self.n, i, i)],
            };
            det = det * diag * diag;
        }

        det
    }
}

// =============================================================================
// Packed LDL^T (sptrf/spsv)
// =============================================================================

/// Packed LDL^T factorization (sptrf).
///
/// For a symmetric matrix A stored in packed format,
/// computes the LDL^T factorization A = L*D*L^T or A = U*D*U^T
/// where L (or U) is unit triangular and D is diagonal.
#[derive(Clone, Debug)]
pub struct PackedLdlt<T: Scalar> {
    /// Packed factor containing L (or U) and D.
    factor: Vec<T>,
    /// Diagonal elements D.
    diagonal: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Which triangle was stored/computed.
    uplo: Uplo,
}

impl<T: Field + Real + bytemuck::Zeroable> PackedLdlt<T> {
    /// Computes packed LDL^T factorization (sptrf).
    ///
    /// # Arguments
    ///
    /// * `ap` - The packed symmetric matrix
    /// * `n` - The matrix dimension
    /// * `uplo` - Whether the upper or lower triangle is stored
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::cholesky::{PackedLdlt, dense_to_packed_lower, Uplo};
    /// use oxiblas_matrix::Mat;
    ///
    /// // Symmetric matrix (may be indefinite)
    /// let a: Mat<f64> = Mat::from_rows(&[
    ///     &[4.0, 2.0],
    ///     &[2.0, 1.0],  // det = 4 - 4 = 0, but submatrices are non-singular
    /// ]);
    ///
    /// let ap = dense_to_packed_lower(a.as_ref());
    /// // Note: This specific matrix will fail due to zero pivot at index 1
    /// ```
    pub fn compute(ap: &[T], n: usize, uplo: Uplo) -> Result<Self, PackedLdltError> {
        let expected_size = n * (n + 1) / 2;
        if ap.len() != expected_size {
            return Err(PackedLdltError::InvalidSize {
                expected: expected_size,
                actual: ap.len(),
            });
        }

        if n == 0 {
            return Ok(PackedLdlt {
                factor: Vec::new(),
                diagonal: Vec::new(),
                n: 0,
                uplo,
            });
        }

        let mut factor = ap.to_vec();
        let mut diagonal = vec![T::zero(); n];

        match uplo {
            Uplo::Lower => {
                // Lower triangle: A = L * D * L^T
                for j in 0..n {
                    // Compute D[j] = a[j,j] - sum_{k<j} L[j,k]^2 * D[k]
                    let diag_idx = packed_lower_index(n, j, j);
                    let mut dj = factor[diag_idx];

                    for k in 0..j {
                        let ljk = factor[packed_lower_index(n, j, k)];
                        dj = dj - ljk * ljk * diagonal[k];
                    }

                    let tol = <T as Scalar>::epsilon()
                        * <T as FromPrimitive>::from_usize(n).unwrap_or(T::one());
                    if Scalar::abs(dj) <= tol {
                        return Err(PackedLdltError::Singular { index: j });
                    }

                    diagonal[j] = dj;
                    factor[diag_idx] = T::one(); // Store 1 on diagonal of L

                    // Compute L[i,j] for i > j
                    // L[i,j] = (a[i,j] - sum_{k<j} L[i,k]*L[j,k]*D[k]) / D[j]
                    for i in (j + 1)..n {
                        let idx = packed_lower_index(n, i, j);
                        let mut lij = factor[idx];

                        for k in 0..j {
                            let lik = factor[packed_lower_index(n, i, k)];
                            let ljk = factor[packed_lower_index(n, j, k)];
                            lij = lij - lik * ljk * diagonal[k];
                        }

                        factor[idx] = lij / dj;
                    }
                }
            }
            Uplo::Upper => {
                // Upper triangle: A = U^T * D * U
                for j in 0..n {
                    // Compute D[j] = a[j,j] - sum_{k<j} U[k,j]^2 * D[k]
                    let diag_idx = packed_upper_index(n, j, j);
                    let mut dj = factor[diag_idx];

                    for k in 0..j {
                        let ukj = factor[packed_upper_index(n, k, j)];
                        dj = dj - ukj * ukj * diagonal[k];
                    }

                    let tol = <T as Scalar>::epsilon()
                        * <T as FromPrimitive>::from_usize(n).unwrap_or(T::one());
                    if Scalar::abs(dj) <= tol {
                        return Err(PackedLdltError::Singular { index: j });
                    }

                    diagonal[j] = dj;
                    factor[diag_idx] = T::one(); // Store 1 on diagonal of U

                    // Compute U[j,i] for i > j
                    // U[j,i] = (a[j,i] - sum_{k<j} U[k,j]*U[k,i]*D[k]) / D[j]
                    for i in (j + 1)..n {
                        let idx = packed_upper_index(n, j, i);
                        let mut uji = factor[idx];

                        for k in 0..j {
                            let ukj = factor[packed_upper_index(n, k, j)];
                            let uki = factor[packed_upper_index(n, k, i)];
                            uji = uji - ukj * uki * diagonal[k];
                        }

                        factor[idx] = uji / dj;
                    }
                }
            }
        }

        Ok(PackedLdlt {
            factor,
            diagonal,
            n,
            uplo,
        })
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns the packed factor (L or U with unit diagonal).
    pub fn factor(&self) -> &[T] {
        &self.factor
    }

    /// Returns the diagonal D.
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }

    /// Returns which triangle was computed.
    pub fn uplo(&self) -> Uplo {
        self.uplo
    }

    /// Returns whether the matrix is positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.diagonal.iter().all(|&d| d > T::zero())
    }

    /// Solves A*x = b using the packed LDL^T factorization (spsv).
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector or matrix
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, PackedLdltError> {
        if b.nrows() != self.n {
            return Err(PackedLdltError::DimensionMismatch {
                expected: self.n,
                actual: b.nrows(),
            });
        }

        let m = b.ncols();
        let mut x = Mat::zeros(self.n, m);

        // Copy b to x
        for j in 0..m {
            for i in 0..self.n {
                x[(i, j)] = b[(i, j)];
            }
        }

        match self.uplo {
            Uplo::Lower => {
                // A = L * D * L^T
                // Step 1: Solve L * y = b (forward substitution)
                // L has unit diagonal
                for k in 0..self.n {
                    for i in (k + 1)..self.n {
                        let lik = self.factor[packed_lower_index(self.n, i, k)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - lik * x[(k, j)];
                        }
                    }
                }

                // Step 2: Solve D * z = y
                for k in 0..self.n {
                    let dk = self.diagonal[k];
                    for j in 0..m {
                        x[(k, j)] = x[(k, j)] / dk;
                    }
                }

                // Step 3: Solve L^T * x = z (backward substitution)
                for k in (0..self.n).rev() {
                    for i in 0..k {
                        let lki = self.factor[packed_lower_index(self.n, k, i)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - lki * x[(k, j)];
                        }
                    }
                }
            }
            Uplo::Upper => {
                // A = U^T * D * U
                // Step 1: Solve U^T * y = b (forward substitution)
                // U has unit diagonal
                for k in 0..self.n {
                    for i in (k + 1)..self.n {
                        let uki = self.factor[packed_upper_index(self.n, k, i)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - uki * x[(k, j)];
                        }
                    }
                }

                // Step 2: Solve D * z = y
                for k in 0..self.n {
                    let dk = self.diagonal[k];
                    for j in 0..m {
                        x[(k, j)] = x[(k, j)] / dk;
                    }
                }

                // Step 3: Solve U * x = z (backward substitution)
                for k in (0..self.n).rev() {
                    for i in 0..k {
                        let uik = self.factor[packed_upper_index(self.n, i, k)];
                        for j in 0..m {
                            x[(i, j)] = x[(i, j)] - uik * x[(k, j)];
                        }
                    }
                }
            }
        }

        Ok(x)
    }

    /// Computes the determinant of the original matrix.
    pub fn determinant(&self) -> T {
        if self.n == 0 {
            return T::one();
        }

        let mut det = T::one();
        for &d in &self.diagonal {
            det = det * d;
        }

        det
    }
}

// =============================================================================
// Convenience functions (ppsv, spsv)
// =============================================================================

/// Solves A*x = b for symmetric positive definite A in packed format (ppsv).
///
/// This combines pptrf (factorization) and pptrs (solve) in one call.
pub fn ppsv<T: Field + Real + bytemuck::Zeroable>(
    ap: &[T],
    n: usize,
    uplo: Uplo,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, PackedCholeskyError> {
    let chol = PackedCholesky::compute(ap, n, uplo)?;
    chol.solve(b)
}

/// Solves A*x = b for symmetric A in packed format (spsv).
///
/// This combines sptrf (factorization) and sptrs (solve) in one call.
pub fn spsv<T: Field + Real + bytemuck::Zeroable>(
    ap: &[T],
    n: usize,
    uplo: Uplo,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, PackedLdltError> {
    let ldlt = PackedLdlt::compute(ap, n, uplo)?;
    ldlt.solve(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_packed_indexing_lower() {
        // For n=3, lower packed storage: (0,0), (1,0), (2,0), (1,1), (2,1), (2,2)
        // Indices:                          0,     1,     2,     3,     4,     5
        assert_eq!(packed_lower_index(3, 0, 0), 0);
        assert_eq!(packed_lower_index(3, 1, 0), 1);
        assert_eq!(packed_lower_index(3, 2, 0), 2);
        assert_eq!(packed_lower_index(3, 1, 1), 3);
        assert_eq!(packed_lower_index(3, 2, 1), 4);
        assert_eq!(packed_lower_index(3, 2, 2), 5);
    }

    #[test]
    fn test_packed_indexing_upper() {
        // For n=3, upper packed storage: (0,0), (0,1), (1,1), (0,2), (1,2), (2,2)
        // Indices:                          0,     1,     2,     3,     4,     5
        assert_eq!(packed_upper_index(3, 0, 0), 0);
        assert_eq!(packed_upper_index(3, 0, 1), 1);
        assert_eq!(packed_upper_index(3, 1, 1), 2);
        assert_eq!(packed_upper_index(3, 0, 2), 3);
        assert_eq!(packed_upper_index(3, 1, 2), 4);
        assert_eq!(packed_upper_index(3, 2, 2), 5);
    }

    #[test]
    fn test_dense_to_packed_lower() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        assert_eq!(ap.len(), 6);

        // ap = [4, 2, 1, 5, 3, 6] for lower packed
        assert_eq!(ap[0], 4.0); // (0,0)
        assert_eq!(ap[1], 2.0); // (1,0)
        assert_eq!(ap[2], 1.0); // (2,0)
        assert_eq!(ap[3], 5.0); // (1,1)
        assert_eq!(ap[4], 3.0); // (2,1)
        assert_eq!(ap[5], 6.0); // (2,2)
    }

    #[test]
    fn test_dense_to_packed_upper() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let ap = dense_to_packed_upper(a.as_ref());
        assert_eq!(ap.len(), 6);

        // ap = [4, 2, 5, 1, 3, 6] for upper packed
        assert_eq!(ap[0], 4.0); // (0,0)
        assert_eq!(ap[1], 2.0); // (0,1)
        assert_eq!(ap[2], 5.0); // (1,1)
        assert_eq!(ap[3], 1.0); // (0,2)
        assert_eq!(ap[4], 3.0); // (1,2)
        assert_eq!(ap[5], 6.0); // (2,2)
    }

    #[test]
    fn test_packed_lower_roundtrip() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let a_recovered = packed_lower_to_dense(&ap, 3);

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a[(i, j)] - a_recovered[(i, j)]).abs() < 1e-14,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_packed_upper_roundtrip() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let ap = dense_to_packed_upper(a.as_ref());
        let a_recovered = packed_upper_to_dense(&ap, 3);

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a[(i, j)] - a_recovered[(i, j)]).abs() < 1e-14,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_packed_cholesky_lower() {
        // SPD matrix
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let ap = dense_to_packed_lower(a.as_ref());

        let chol = PackedCholesky::compute(&ap, 2, Uplo::Lower).unwrap();

        // Check determinant: det(A) = 4*5 - 2*2 = 16
        let det = chol.determinant();
        assert!((det - 16.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_packed_cholesky_upper() {
        // SPD matrix
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let ap = dense_to_packed_upper(a.as_ref());

        let chol = PackedCholesky::compute(&ap, 2, Uplo::Upper).unwrap();

        // Check determinant
        let det = chol.determinant();
        assert!((det - 16.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_packed_cholesky_solve_lower() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let chol = PackedCholesky::compute(&ap, 2, Uplo::Lower).unwrap();
        let x = chol.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_packed_cholesky_solve_upper() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ap = dense_to_packed_upper(a.as_ref());
        let chol = PackedCholesky::compute(&ap, 2, Uplo::Upper).unwrap();
        let x = chol.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_packed_cholesky_3x3() {
        // Classic Cholesky test matrix
        let a: Mat<f64> = Mat::from_rows(&[
            &[4.0, 12.0, -16.0],
            &[12.0, 37.0, -43.0],
            &[-16.0, -43.0, 98.0],
        ]);
        let ap = dense_to_packed_lower(a.as_ref());

        let chol = PackedCholesky::compute(&ap, 3, Uplo::Lower).unwrap();

        // Solve Ax = b
        let b: Mat<f64> = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let x = chol.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 12.0 * x[(1, 0)] - 16.0 * x[(2, 0)];
        let ax1 = 12.0 * x[(0, 0)] + 37.0 * x[(1, 0)] - 43.0 * x[(2, 0)];
        let ax2 = -16.0 * x[(0, 0)] - 43.0 * x[(1, 0)] + 98.0 * x[(2, 0)];
        assert!((ax0 - 1.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 2.0).abs() < 1e-10, "Ax[1] = {}", ax1);
        assert!((ax2 - 3.0).abs() < 1e-10, "Ax[2] = {}", ax2);
    }

    #[test]
    fn test_packed_cholesky_not_spd() {
        // Indefinite matrix
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);
        let ap = dense_to_packed_lower(a.as_ref());

        let result = PackedCholesky::compute(&ap, 2, Uplo::Lower);
        assert!(result.is_err());
    }

    #[test]
    fn test_packed_ldlt_lower() {
        // SPD matrix (LDL^T works for any symmetric non-singular)
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let ap = dense_to_packed_lower(a.as_ref());

        let ldlt = PackedLdlt::compute(&ap, 2, Uplo::Lower).unwrap();

        // Should be positive definite
        assert!(ldlt.is_positive_definite());

        // Check determinant
        let det = ldlt.determinant();
        assert!((det - 16.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_packed_ldlt_solve_lower() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let ldlt = PackedLdlt::compute(&ap, 2, Uplo::Lower).unwrap();
        let x = ldlt.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_packed_ldlt_solve_upper() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ap = dense_to_packed_upper(a.as_ref());
        let ldlt = PackedLdlt::compute(&ap, 2, Uplo::Upper).unwrap();
        let x = ldlt.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_packed_ldlt_3x3() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let ldlt = PackedLdlt::compute(&ap, 3, Uplo::Lower).unwrap();
        let x = ldlt.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)] + 1.0 * x[(2, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)] + 3.0 * x[(2, 0)];
        let ax2 = 1.0 * x[(0, 0)] + 3.0 * x[(1, 0)] + 6.0 * x[(2, 0)];
        assert!((ax0 - 1.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 2.0).abs() < 1e-10, "Ax[1] = {}", ax1);
        assert!((ax2 - 3.0).abs() < 1e-10, "Ax[2] = {}", ax2);
    }

    #[test]
    fn test_ppsv() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let x = ppsv(&ap, 2, Uplo::Lower, b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10);
        assert!((ax1 - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_spsv() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let x = spsv(&ap, 2, Uplo::Lower, b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10);
        assert!((ax1 - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_packed_multiple_rhs() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0, 1.0], &[11.0, 2.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let chol = PackedCholesky::compute(&ap, 2, Uplo::Lower).unwrap();
        let x = chol.solve(b.as_ref()).unwrap();

        // Verify Ax = b for both RHS
        for col in 0..2 {
            let ax0 = 4.0 * x[(0, col)] + 2.0 * x[(1, col)];
            let ax1 = 2.0 * x[(0, col)] + 5.0 * x[(1, col)];
            assert!((ax0 - b[(0, col)]).abs() < 1e-10);
            assert!((ax1 - b[(1, col)]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_packed_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let ap = dense_to_packed_lower(a.as_ref());
        let chol = PackedCholesky::compute(&ap, 2, Uplo::Lower).unwrap();
        let x = chol.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-5);
        assert!((ax1 - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_packed_empty() {
        let ap: Vec<f64> = vec![];
        let chol = PackedCholesky::compute(&ap, 0, Uplo::Lower).unwrap();
        assert_eq!(chol.size(), 0);
        assert_eq!(chol.determinant(), 1.0);

        let ldlt = PackedLdlt::compute(&ap, 0, Uplo::Lower).unwrap();
        assert_eq!(ldlt.size(), 0);
        assert_eq!(ldlt.determinant(), 1.0);
    }
}
