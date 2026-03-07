//! LL^T Cholesky decomposition.
//!
//! Standard Cholesky factorization for symmetric positive definite matrices.
//! For large matrices, uses blocked algorithm with GEMM/TRSM for cache efficiency.

use num_traits::{FromPrimitive, One};
use oxiblas_blas::level3::gemm::gemm;
#[cfg(feature = "parallel")]
use oxiblas_blas::level3::gemm::gemm_with_par;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_blas::level3::trsm::{Diag, Side, Trans, Uplo, trsm_in_place};
#[cfg(feature = "parallel")]
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error returned when Cholesky decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CholeskyError {
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

impl core::fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CholeskyError::NotPositiveDefinite { index } => {
                write!(
                    f,
                    "Matrix is not positive definite (detected at index {index})"
                )
            }
            CholeskyError::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}×{ncols}")
            }
            CholeskyError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for CholeskyError {}

/// Cholesky decomposition (LL^T factorization).
///
/// For a symmetric positive definite matrix A, computes L such that A = LL^T,
/// where L is lower triangular with positive diagonal entries.
#[derive(Clone, Debug)]
pub struct Cholesky<T: Scalar> {
    /// The L factor (lower triangular).
    l: Mat<T>,
}

impl<T: Field + Real + bytemuck::Zeroable> Cholesky<T> {
    /// Computes the Cholesky decomposition of a symmetric positive definite matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - A symmetric positive definite matrix
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::cholesky::Cholesky;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a: Mat<f64> = Mat::from_rows(&[
    ///     &[4.0, 2.0],
    ///     &[2.0, 5.0],
    /// ]);
    ///
    /// let chol = Cholesky::compute(a.as_ref()).expect("Matrix is SPD");
    ///
    /// // Get L factor
    /// let l = chol.l_factor();
    ///
    /// // Compute determinant: det(A) = (det(L))^2
    /// let det = chol.determinant();
    /// assert!((det - 16.0).abs() < 1e-10); // 4*5 - 2*2 = 16
    ///
    /// // Solve Ax = b
    /// let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);
    /// let x = chol.solve(b.as_ref()).expect("Should solve");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `CholeskyError::NotSquare` if the matrix is not square.
    /// Returns `CholeskyError::NotPositiveDefinite` if the matrix is not positive definite.
    ///
    /// # Note
    ///
    /// Only the lower triangular part of `a` is used. The matrix is assumed
    /// to be symmetric.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, CholeskyError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(CholeskyError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Cholesky {
                l: Mat::zeros(0, 0),
            });
        }

        let mut l = Mat::zeros(n, n);

        // Cholesky-Banachiewicz algorithm
        for i in 0..n {
            for j in 0..=i {
                let mut sum = T::zero();

                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum = sum + l[(j, k)] * l[(j, k)];
                    }

                    let diag = a[(i, i)] - sum;

                    // Check for positive definiteness
                    let tol = <T as Scalar>::epsilon()
                        * <T as FromPrimitive>::from_usize(n).unwrap_or(<T as One>::one());
                    if diag <= tol {
                        return Err(CholeskyError::NotPositiveDefinite { index: i });
                    }

                    l[(i, j)] = Real::sqrt(diag);
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum = sum + l[(i, k)] * l[(j, k)];
                    }

                    l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
                }
            }
        }

        Ok(Cholesky { l })
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
    /// For A = LL^T, det(A) = det(L)^2 = (∏ L\[i,i\])^2
    pub fn determinant(&self) -> T {
        let n = self.size();
        if n == 0 {
            return T::one();
        }

        let mut det_l = T::one();
        for i in 0..n {
            det_l = det_l * self.l[(i, i)];
        }

        det_l * det_l
    }

    /// Solves the system Ax = b for symmetric positive definite A.
    ///
    /// Given A = LL^T, solves:
    /// 1. Forward substitution: Ly = b
    /// 2. Back substitution: L^T x = y
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × m for multiple RHS)
    ///
    /// # Errors
    ///
    /// Returns `CholeskyError::DimensionMismatch` if b has wrong number of rows.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, CholeskyError> {
        let n = self.size();

        if b.nrows() != n {
            return Err(CholeskyError::DimensionMismatch {
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

        // Back substitution: L^T x = y
        for k in (0..n).rev() {
            for j in 0..m {
                x[(k, j)] = x[(k, j)] / self.l[(k, k)];
            }

            for i in 0..k {
                let mult = self.l[(k, i)];
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
    pub fn inverse(&self) -> Result<Mat<T>, CholeskyError> {
        let n = self.size();
        let identity = Mat::<T>::eye(n);
        self.solve(identity.as_ref())
    }

    /// Returns the log-determinant of the original matrix.
    ///
    /// This is useful for numerical stability when det(A) is very large or small.
    /// log(det(A)) = 2 * sum(log(L\[i,i\]))
    pub fn log_determinant(&self) -> T {
        let n = self.size();
        if n == 0 {
            return T::zero();
        }

        let mut log_det = T::zero();
        let two = T::one() + T::one();
        for i in 0..n {
            log_det = log_det + Real::ln(self.l[(i, i)]);
        }

        two * log_det
    }
}

// Optimized blocked Cholesky factorization for types that support GEMM
impl<T: Field + Real + GemmKernel + bytemuck::Zeroable> Cholesky<T> {
    /// Computes the Cholesky decomposition using blocked algorithm for large matrices.
    ///
    /// Uses GEMM and TRSM for cache-efficient computation on large matrices.
    /// For matrices smaller than the block size, falls back to unblocked algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - A symmetric positive definite matrix
    ///
    /// # Returns
    ///
    /// The Cholesky decomposition on success.
    ///
    /// # Errors
    ///
    /// Returns `CholeskyError::NotSquare` if the matrix is not square.
    /// Returns `CholeskyError::NotPositiveDefinite` if the matrix is not positive definite.
    #[inline]
    pub fn compute_blocked(a: MatRef<'_, T>) -> Result<Self, CholeskyError> {
        const BLOCK_SIZE: usize = 64;
        Self::compute_with_block_size(a, BLOCK_SIZE)
    }

    /// Computes Cholesky decomposition with a specified block size.
    pub fn compute_with_block_size(a: MatRef<'_, T>, nb: usize) -> Result<Self, CholeskyError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(CholeskyError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Cholesky {
                l: Mat::zeros(0, 0),
            });
        }

        // Copy lower triangular part of A
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        // Use blocked algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor(&mut l, n, nb)?;
        } else {
            Self::unblocked_factor(&mut l, n, 0)?;
        }

        Ok(Cholesky { l })
    }

    /// Blocked Cholesky factorization using GEMM for symmetric updates.
    fn blocked_factor(l: &mut Mat<T>, n: usize, nb: usize) -> Result<(), CholeskyError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor diagonal block A11 using unblocked Cholesky
            Self::unblocked_factor_block(l, n, jb, jb_size)?;

            // If there are more rows below this block
            if jb + jb_size < n {
                let rows_remaining = n - jb - jb_size;

                // Extract L11 (the diagonal block we just factored)
                let mut l11: Mat<T> = Mat::zeros(jb_size, jb_size);
                for i in 0..jb_size {
                    for j in 0..=i {
                        l11[(i, j)] = l[(jb + i, jb + j)];
                    }
                }

                // Extract A21 block (will become L21)
                let mut l21: Mat<T> = Mat::zeros(rows_remaining, jb_size);
                for j in 0..jb_size {
                    for i in 0..rows_remaining {
                        l21[(i, j)] = l[(jb + jb_size + i, jb + j)];
                    }
                }

                // Solve L21 = A21 * L11^(-T) using TRSM: L11 * L21^T = A21^T
                // Equivalently: L21 = A21 * inv(L11^T)
                // TRSM: Right, Lower, Trans => B = B * inv(L^T)
                let _ = trsm_in_place(
                    Side::Right,
                    Uplo::Lower,
                    Trans::Trans,
                    Diag::NonUnit,
                    l11.as_ref(),
                    l21.as_mut(),
                );

                // Copy L21 back
                for j in 0..jb_size {
                    for i in 0..rows_remaining {
                        l[(jb + jb_size + i, jb + j)] = l21[(i, j)];
                    }
                }

                // Update A22 -= L21 * L21^T using GEMM (symmetric rank-k update)
                // A22 is the trailing (rows_remaining x rows_remaining) block
                let l21_t = l21.transpose();
                let mut update: Mat<T> = Mat::zeros(rows_remaining, rows_remaining);
                gemm(
                    T::one(),
                    l21.as_ref(),
                    l21_t.as_ref(),
                    T::zero(),
                    update.as_mut(),
                );

                // Subtract update from lower triangular part of A22
                for j in 0..rows_remaining {
                    for i in j..rows_remaining {
                        l[(jb + jb_size + i, jb + jb_size + j)] =
                            l[(jb + jb_size + i, jb + jb_size + j)] - update[(i, j)];
                    }
                }
            }

            jb += jb_size;
        }

        Ok(())
    }

    /// Unblocked Cholesky for a diagonal block only (rows/cols jb to jb+jb_size).
    fn unblocked_factor_block(
        l: &mut Mat<T>,
        n: usize,
        jb: usize,
        jb_size: usize,
    ) -> Result<(), CholeskyError> {
        for i in 0..jb_size {
            let gi = jb + i; // Global index

            // Compute diagonal element
            let mut sum = T::zero();
            for k in jb..gi {
                sum = sum + l[(gi, k)] * l[(gi, k)];
            }

            let diag = l[(gi, gi)] - sum;

            // Check for positive definiteness
            let tol = <T as Scalar>::epsilon()
                * <T as FromPrimitive>::from_usize(n).unwrap_or(<T as One>::one());
            if diag <= tol {
                return Err(CholeskyError::NotPositiveDefinite { index: gi });
            }

            l[(gi, gi)] = Real::sqrt(diag);

            // Compute off-diagonal elements in this column (within the block only)
            for j in (i + 1)..jb_size {
                let gj = jb + j;
                let mut sum = T::zero();
                for k in jb..gi {
                    sum = sum + l[(gj, k)] * l[(gi, k)];
                }
                l[(gj, gi)] = (l[(gj, gi)] - sum) / l[(gi, gi)];
            }
            // Note: Elements below the block (L21) are computed via TRSM, not here
        }

        Ok(())
    }

    /// Unblocked Cholesky factorization (for small matrices).
    fn unblocked_factor(l: &mut Mat<T>, n: usize, start: usize) -> Result<(), CholeskyError> {
        for i in start..n {
            for j in start..=i {
                let mut sum = T::zero();

                if j == i {
                    // Diagonal element
                    for k in start..j {
                        sum = sum + l[(j, k)] * l[(j, k)];
                    }

                    let diag = l[(i, i)] - sum;

                    let tol = <T as Scalar>::epsilon()
                        * <T as FromPrimitive>::from_usize(n).unwrap_or(<T as One>::one());
                    if diag <= tol {
                        return Err(CholeskyError::NotPositiveDefinite { index: i });
                    }

                    l[(i, j)] = Real::sqrt(diag);
                } else {
                    // Off-diagonal element
                    for k in start..j {
                        sum = sum + l[(i, k)] * l[(j, k)];
                    }

                    l[(i, j)] = (l[(i, j)] - sum) / l[(j, j)];
                }
            }
        }

        Ok(())
    }
}

// Recursive cache-oblivious Cholesky factorization
impl<T: Field + Real + GemmKernel + bytemuck::Zeroable> Cholesky<T> {
    /// Recursion threshold: matrices at or below this size use the unblocked algorithm.
    const RECURSIVE_THRESHOLD: usize = 64;

    /// Computes the Cholesky decomposition using a recursive cache-oblivious algorithm.
    ///
    /// This divide-and-conquer approach automatically adapts to the cache hierarchy
    /// by recursively splitting the matrix into quadrants. At each level:
    ///
    /// 1. Factor the top-left quadrant A11 recursively to get L11
    /// 2. Solve L21 = A21 * L11^{-T} via TRSM
    /// 3. Update A22 -= L21 * L21^T via SYRK (symmetric rank-k update)
    /// 4. Factor the updated A22 recursively to get L22
    ///
    /// For matrices smaller than the recursion threshold (64), falls back to the
    /// unblocked algorithm which is more efficient at that scale.
    ///
    /// # Arguments
    ///
    /// * `a` - A symmetric positive definite matrix (only lower triangle is read)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::cholesky::Cholesky;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 200;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     a[(i, i)] = 2.0;
    ///     if i > 0 {
    ///         a[(i, i - 1)] = -1.0;
    ///         a[(i - 1, i)] = -1.0;
    ///     }
    /// }
    ///
    /// let chol = Cholesky::compute_recursive(a.as_ref()).expect("Matrix is SPD");
    /// let det = chol.determinant();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `CholeskyError::NotSquare` if the matrix is not square.
    /// Returns `CholeskyError::NotPositiveDefinite` if the matrix is not positive definite.
    pub fn compute_recursive(a: MatRef<'_, T>) -> Result<Self, CholeskyError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(CholeskyError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Cholesky {
                l: Mat::zeros(0, 0),
            });
        }

        // Copy lower triangular part of A into L
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        Self::recursive_factor(&mut l, n, 0)?;

        Ok(Cholesky { l })
    }

    /// Recursive Cholesky factorization on a submatrix starting at (offset, offset).
    ///
    /// Operates in-place on the lower triangular part of `l`.
    /// The submatrix from (offset, offset) to (offset+size-1, offset+size-1) is factored.
    fn recursive_factor(l: &mut Mat<T>, size: usize, offset: usize) -> Result<(), CholeskyError> {
        // Base case: use unblocked algorithm for small matrices
        if size <= Self::RECURSIVE_THRESHOLD {
            let full_n = l.nrows();
            Self::unblocked_factor_block(l, full_n, offset, size)?;
            return Ok(());
        }

        // Split: n1 = size/2 (upper half), n2 = size - n1 (lower half)
        let n1 = size / 2;
        let n2 = size - n1;

        // Step 1: Recursively factor A11 (top-left n1 x n1 block)
        Self::recursive_factor(l, n1, offset)?;

        // Step 2: Solve L21 = A21 * L11^{-T} via TRSM
        // Extract L11 (lower triangular, already factored)
        let mut l11 = Mat::zeros(n1, n1);
        for i in 0..n1 {
            for j in 0..=i {
                l11[(i, j)] = l[(offset + i, offset + j)];
            }
        }

        // Extract A21 (the block that will become L21)
        let mut l21 = Mat::zeros(n2, n1);
        for j in 0..n1 {
            for i in 0..n2 {
                l21[(i, j)] = l[(offset + n1 + i, offset + j)];
            }
        }

        // TRSM: B = B * inv(L11^T), where B = L21
        // Side::Right, Uplo::Lower, Trans::Trans, Diag::NonUnit
        let _ = trsm_in_place(
            Side::Right,
            Uplo::Lower,
            Trans::Trans,
            Diag::NonUnit,
            l11.as_ref(),
            l21.as_mut(),
        );

        // Copy L21 back into l
        for j in 0..n1 {
            for i in 0..n2 {
                l[(offset + n1 + i, offset + j)] = l21[(i, j)];
            }
        }

        // Step 3: Symmetric rank-k update on A22
        // A22 -= L21 * L21^T (only lower triangle)
        let mut a22 = Mat::zeros(n2, n2);
        for j in 0..n2 {
            for i in j..n2 {
                a22[(i, j)] = l[(offset + n1 + i, offset + n1 + j)];
            }
        }

        use oxiblas_blas::level3::syrk::syrk;

        // SYRK: C = alpha*A*A^T + beta*C  =>  A22 = -1*L21*L21^T + 1*A22
        let _ = syrk(
            Uplo::Lower,
            Trans::NoTrans,
            -T::one(),
            l21.as_ref(),
            T::one(),
            a22.as_mut(),
        );

        // Copy updated A22 back into l (lower triangle only)
        for j in 0..n2 {
            for i in j..n2 {
                l[(offset + n1 + i, offset + n1 + j)] = a22[(i, j)];
            }
        }

        // Step 4: Recursively factor updated A22
        Self::recursive_factor(l, n2, offset + n1)?;

        Ok(())
    }
}

// Optimized automatic algorithm selection for f64 and f32
impl<T: Field + Real + GemmKernel + bytemuck::Zeroable> Cholesky<T> {
    /// Computes the Cholesky decomposition with automatic algorithm selection.
    ///
    /// For matrices with size >= 128, automatically uses the blocked algorithm
    /// for better cache efficiency and performance. Otherwise uses the unblocked
    /// algorithm which has less overhead for small matrices.
    ///
    /// This method is available for f32 and f64 types which have optimized GEMM kernels.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::cholesky::Cholesky;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 256;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     a[(i, i)] = 2.0;
    ///     if i > 0 {
    ///         a[(i, i - 1)] = -1.0;
    ///         a[(i - 1, i)] = -1.0;
    ///     }
    /// }
    ///
    /// // Automatically uses blocked algorithm for n >= 128
    /// let chol = Cholesky::compute_auto(a.as_ref()).unwrap();
    /// ```
    pub fn compute_auto(a: MatRef<'_, T>) -> Result<Self, CholeskyError> {
        const AUTO_BLOCK_THRESHOLD: usize = 128;
        let n = a.nrows();

        // For large matrices, use blocked algorithm automatically
        if n >= AUTO_BLOCK_THRESHOLD {
            Self::compute_blocked(a)
        } else {
            // Use unblocked for small matrices
            Self::compute(a)
        }
    }
}

// Parallel blocked Cholesky factorization
#[cfg(feature = "parallel")]
impl<T: Field + Real + GemmKernel + bytemuck::Zeroable + Send + Sync> Cholesky<T> {
    /// Computes the Cholesky decomposition using a parallel blocked algorithm.
    ///
    /// Parallelizes the GEMM (symmetric rank-k) updates within the blocked
    /// factorization using Rayon. For matrices smaller than the block size,
    /// falls back to the sequential unblocked algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - A symmetric positive definite matrix
    ///
    /// # Returns
    ///
    /// The Cholesky decomposition on success.
    ///
    /// # Errors
    ///
    /// Returns `CholeskyError::NotSquare` if the matrix is not square.
    /// Returns `CholeskyError::NotPositiveDefinite` if the matrix is not positive definite.
    #[inline]
    pub fn compute_blocked_par(a: MatRef<'_, T>) -> Result<Self, CholeskyError> {
        const BLOCK_SIZE: usize = 64;
        Self::compute_blocked_par_with_block_size(a, BLOCK_SIZE)
    }

    /// Computes parallel blocked Cholesky decomposition with a specified block size.
    pub fn compute_blocked_par_with_block_size(
        a: MatRef<'_, T>,
        nb: usize,
    ) -> Result<Self, CholeskyError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(CholeskyError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Cholesky {
                l: Mat::zeros(0, 0),
            });
        }

        // Copy lower triangular part of A
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        // Use blocked parallel algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor_par(&mut l, n, nb)?;
        } else {
            Self::unblocked_factor(&mut l, n, 0)?;
        }

        Ok(Cholesky { l })
    }

    /// Blocked Cholesky factorization with parallel GEMM for symmetric updates.
    fn blocked_factor_par(l: &mut Mat<T>, n: usize, nb: usize) -> Result<(), CholeskyError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor diagonal block A11 using unblocked Cholesky (sequential -- panel is small)
            Self::unblocked_factor_block(l, n, jb, jb_size)?;

            // If there are more rows below this block
            if jb + jb_size < n {
                let rows_remaining = n - jb - jb_size;

                // Extract L11 (the diagonal block we just factored)
                let mut l11: Mat<T> = Mat::zeros(jb_size, jb_size);
                for i in 0..jb_size {
                    for j in 0..=i {
                        l11[(i, j)] = l[(jb + i, jb + j)];
                    }
                }

                // Extract A21 block (will become L21)
                let mut l21: Mat<T> = Mat::zeros(rows_remaining, jb_size);
                for j in 0..jb_size {
                    for i in 0..rows_remaining {
                        l21[(i, j)] = l[(jb + jb_size + i, jb + j)];
                    }
                }

                // Solve L21 = A21 * L11^(-T) using TRSM
                // TRSM internally uses parallel GEMM when the "parallel" feature is active
                let _ = trsm_in_place(
                    Side::Right,
                    Uplo::Lower,
                    Trans::Trans,
                    Diag::NonUnit,
                    l11.as_ref(),
                    l21.as_mut(),
                );

                // Copy L21 back
                for j in 0..jb_size {
                    for i in 0..rows_remaining {
                        l[(jb + jb_size + i, jb + j)] = l21[(i, j)];
                    }
                }

                // Update A22 -= L21 * L21^T using parallel GEMM (symmetric rank-k update)
                let l21_t = l21.transpose();
                let mut update: Mat<T> = Mat::zeros(rows_remaining, rows_remaining);
                gemm_with_par(
                    T::one(),
                    l21.as_ref(),
                    l21_t.as_ref(),
                    T::zero(),
                    update.as_mut(),
                    Par::Rayon,
                );

                // Subtract update from lower triangular part of A22
                for j in 0..rows_remaining {
                    for i in j..rows_remaining {
                        l[(jb + jb_size + i, jb + jb_size + j)] =
                            l[(jb + jb_size + i, jb + jb_size + j)] - update[(i, j)];
                    }
                }
            }

            jb += jb_size;
        }

        Ok(())
    }
}
