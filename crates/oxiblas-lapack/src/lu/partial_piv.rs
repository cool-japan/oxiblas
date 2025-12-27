//! LU decomposition with partial pivoting.
//!
//! This is the standard LU decomposition algorithm used by LAPACK's DGETRF.
//! For large matrices, uses blocked algorithm with GEMM/TRSM for cache efficiency.

use num_traits::{FromPrimitive, One};
use oxiblas_blas::level3::gemm::gemm;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_blas::level3::trsm::{Diag, Side, Trans, Uplo, trsm_in_place};
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error returned when LU decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuError {
    /// The matrix is singular (has a zero or near-zero pivot).
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

impl core::fmt::Display for LuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LuError::Singular { index } => {
                write!(f, "Matrix is singular at index {index}")
            }
            LuError::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}×{ncols}")
            }
            LuError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for LuError {}

/// LU decomposition with partial (row) pivoting.
///
/// Stores the factorization PA = LU where:
/// - P is a permutation matrix (stored as pivot indices)
/// - L is lower triangular with unit diagonal
/// - U is upper triangular
///
/// The L and U factors are stored compactly in a single matrix,
/// with L below the diagonal and U on and above the diagonal.
#[derive(Clone, Debug)]
pub struct Lu<T: Scalar> {
    /// Combined L and U factors.
    /// L is stored below the diagonal (with implicit unit diagonal).
    /// U is stored on and above the diagonal.
    lu: Mat<T>,
    /// Pivot indices: row i was swapped with row pivot[i].
    pivot: Vec<usize>,
    /// Number of row swaps (for determinant sign).
    num_swaps: usize,
}

impl<T: Field + bytemuck::Zeroable> Lu<T> {
    /// Computes the LU decomposition of a square matrix.
    ///
    /// Uses partial pivoting (row permutations) for numerical stability.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::lu::Lu;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a: Mat<f64> = Mat::from_rows(&[
    ///     &[2.0, 1.0],
    ///     &[4.0, 3.0],
    /// ]);
    ///
    /// let lu = Lu::compute(a.as_ref()).expect("Matrix should be non-singular");
    ///
    /// // Compute determinant: det(A) = 2*3 - 1*4 = 2
    /// let det = lu.determinant();
    /// assert!((det - 2.0).abs() < 1e-10);
    ///
    /// // Solve Ax = b
    /// let b: Mat<f64> = Mat::from_rows(&[&[3.0], &[7.0]]);
    /// let x = lu.solve(b.as_ref()).expect("Should solve");
    /// assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
    /// assert!((x[(1, 0)] - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `LuError::NotSquare` if the matrix is not square.
    /// Returns `LuError::Singular` if the matrix is singular.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, LuError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LuError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Lu {
                lu: Mat::zeros(0, 0),
                pivot: Vec::new(),
                num_swaps: 0,
            });
        }

        // Copy A into LU matrix
        let mut lu = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                lu[(i, j)] = a[(i, j)];
            }
        }

        let mut pivot = vec![0usize; n];
        let mut num_swaps = 0;

        // Doolittle algorithm with partial pivoting
        for k in 0..n {
            // Find pivot: largest absolute value in column k, rows k..n
            let mut pivot_row = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);

            for i in (k + 1)..n {
                let val = Scalar::abs(lu[(i, k)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }

            // Check for singularity
            // Use a relative tolerance based on machine epsilon and matrix size
            let tol = T::epsilon()
                * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
            if pivot_val <= tol {
                return Err(LuError::Singular { index: k });
            }

            // Store pivot
            pivot[k] = pivot_row;

            // Swap rows if needed
            if pivot_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                num_swaps += 1;
            }

            // Compute multipliers (L's subdiagonal entries) and update
            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                // Multiplier (stored in L)
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;

                // Update remaining submatrix
                for j in (k + 1)..n {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }

    /// Returns the size of the matrix (n for an n×n matrix).
    #[inline]
    pub fn size(&self) -> usize {
        self.lu.nrows()
    }

    /// Returns a reference to the combined LU matrix.
    ///
    /// L is stored below the diagonal, U is on and above the diagonal.
    pub fn lu_matrix(&self) -> MatRef<'_, T> {
        self.lu.as_ref()
    }

    /// Returns the pivot indices.
    pub fn pivot(&self) -> &[usize] {
        &self.pivot
    }

    /// Computes the determinant of the original matrix.
    ///
    /// The determinant is the product of U's diagonal elements,
    /// negated if there was an odd number of row swaps.
    pub fn determinant(&self) -> T {
        let n = self.size();
        if n == 0 {
            return T::one();
        }

        let mut det = if self.num_swaps % 2 == 0 {
            T::one()
        } else {
            -T::one()
        };

        // Product of U's diagonal
        for i in 0..n {
            det = det * self.lu[(i, i)];
        }

        det
    }

    /// Solves the system Ax = b.
    ///
    /// Given the LU factorization PA = LU, solves:
    /// 1. Apply permutation: Pb
    /// 2. Forward substitution: Ly = Pb
    /// 3. Back substitution: Ux = y
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × m for multiple RHS)
    ///
    /// # Errors
    ///
    /// Returns `LuError::DimensionMismatch` if b has wrong number of rows.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, LuError> {
        let n = self.size();

        if b.nrows() != n {
            return Err(LuError::DimensionMismatch {
                expected: n,
                actual: b.nrows(),
            });
        }

        let m = b.ncols();
        let mut x = Mat::zeros(n, m);

        // Copy b to x, applying row permutation
        for k in 0..n {
            let pk = self.pivot[k];
            for j in 0..m {
                let tmp = if k != pk { b[(pk, j)] } else { b[(k, j)] };
                x[(k, j)] = tmp;
            }
        }

        // Apply permutation in-place
        for k in 0..n {
            let pk = self.pivot[k];
            if k != pk {
                for j in 0..m {
                    let tmp = x[(k, j)];
                    x[(k, j)] = x[(pk, j)];
                    x[(pk, j)] = tmp;
                }
            }
        }

        // Re-copy b with permutation (the swap above doesn't work correctly for all cases)
        // Let's fix by properly applying the permutation sequence
        for j in 0..m {
            for i in 0..n {
                x[(i, j)] = b[(i, j)];
            }
        }

        // Apply permutations in order
        for k in 0..n {
            let pk = self.pivot[k];
            if k != pk {
                for j in 0..m {
                    let tmp = x[(k, j)];
                    x[(k, j)] = x[(pk, j)];
                    x[(pk, j)] = tmp;
                }
            }
        }

        // Forward substitution: Ly = Pb (L has unit diagonal)
        for k in 0..n {
            for i in (k + 1)..n {
                let mult = self.lu[(i, k)];
                for j in 0..m {
                    let val = x[(i, j)] - mult * x[(k, j)];
                    x[(i, j)] = val;
                }
            }
        }

        // Back substitution: Ux = y
        for k in (0..n).rev() {
            let diag = self.lu[(k, k)];
            for j in 0..m {
                x[(k, j)] = x[(k, j)] / diag;
            }

            for i in 0..k {
                let mult = self.lu[(i, k)];
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
    pub fn inverse(&self) -> Result<Mat<T>, LuError> {
        let n = self.size();
        let identity = Mat::<T>::eye(n);
        self.solve(identity.as_ref())
    }

    /// Extracts the L factor (lower triangular with unit diagonal).
    pub fn l_factor(&self) -> Mat<T> {
        let n = self.size();
        let mut l = Mat::zeros(n, n);

        for i in 0..n {
            // Unit diagonal
            l[(i, i)] = T::one();
            // Below diagonal
            for j in 0..i {
                l[(i, j)] = self.lu[(i, j)];
            }
        }

        l
    }

    /// Extracts the U factor (upper triangular).
    pub fn u_factor(&self) -> Mat<T> {
        let n = self.size();
        let mut u = Mat::zeros(n, n);

        for i in 0..n {
            // On and above diagonal
            for j in i..n {
                u[(i, j)] = self.lu[(i, j)];
            }
        }

        u
    }

    /// Solves the system A^T x = b (transpose solve).
    ///
    /// Given the LU factorization PA = LU (so A = P^(-1) LU):
    /// A^T = U^T L^T P
    ///
    /// Solves:
    /// 1. Forward substitution: U^T z = b
    /// 2. Back substitution: L^T w = z
    /// 3. Apply inverse permutation: x = P^T w
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × m for multiple RHS)
    ///
    /// # Errors
    ///
    /// Returns `LuError::DimensionMismatch` if b has wrong number of rows.
    pub fn solve_transpose(&self, b: MatRef<'_, T>) -> Result<Mat<T>, LuError> {
        let n = self.size();

        if b.nrows() != n {
            return Err(LuError::DimensionMismatch {
                expected: n,
                actual: b.nrows(),
            });
        }

        let m = b.ncols();
        let mut x = Mat::zeros(n, m);

        // Copy b to x
        for i in 0..n {
            for j in 0..m {
                x[(i, j)] = b[(i, j)];
            }
        }

        // Forward substitution: U^T z = b
        // U^T is lower triangular (U's row i, col j becomes U^T row j, col i)
        // For each row k: sum_{j<=k} U[j,k] * z[j] = b[k]
        // z[k] = (b[k] - sum_{j<k} U[j,k] * z[j]) / U[k,k]
        for k in 0..n {
            let diag = self.lu[(k, k)];
            for j in 0..m {
                let mut sum = T::zero();
                for i in 0..k {
                    sum = sum + self.lu[(i, k)] * x[(i, j)];
                }
                x[(k, j)] = (x[(k, j)] - sum) / diag;
            }
        }

        // Back substitution: L^T w = z
        // L^T is upper triangular (L's row i, col j becomes L^T row j, col i)
        // L has unit diagonal, so L^T also has unit diagonal
        // For each row k (from n-1 down): w[k] + sum_{j>k} L[j,k] * w[j] = z[k]
        // w[k] = z[k] - sum_{j>k} L[j,k] * w[j]
        for k in (0..n).rev() {
            for j in 0..m {
                let mut sum = T::zero();
                for i in (k + 1)..n {
                    sum = sum + self.lu[(i, k)] * x[(i, j)];
                }
                x[(k, j)] = x[(k, j)] - sum;
            }
        }

        // Apply inverse permutation: reverse the swaps
        for k in (0..n).rev() {
            let pk = self.pivot[k];
            if k != pk {
                for j in 0..m {
                    let tmp = x[(k, j)];
                    x[(k, j)] = x[(pk, j)];
                    x[(pk, j)] = tmp;
                }
            }
        }

        Ok(x)
    }

    /// Constructs the permutation matrix P.
    ///
    /// P is such that PA = LU.
    pub fn permutation_matrix(&self) -> Mat<T> {
        let n = self.size();
        let mut p = Mat::eye(n);

        for k in 0..n {
            let pk = self.pivot[k];
            if k != pk {
                // Swap rows k and pk
                for j in 0..n {
                    let tmp = p[(k, j)];
                    p[(k, j)] = p[(pk, j)];
                    p[(pk, j)] = tmp;
                }
            }
        }

        p
    }
}

// Optimized blocked LU factorization for types that support GEMM
impl<T: Field + GemmKernel + bytemuck::Zeroable> Lu<T> {
    /// Computes the LU decomposition using blocked algorithm for large matrices.
    ///
    /// Uses GEMM and TRSM for cache-efficient computation on large matrices.
    /// For matrices smaller than the block size, falls back to unblocked algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix A (n×n)
    ///
    /// # Returns
    ///
    /// The LU decomposition on success.
    ///
    /// # Errors
    ///
    /// Returns `LuError::NotSquare` if the matrix is not square.
    /// Returns `LuError::Singular` if the matrix is singular.
    #[inline]
    pub fn compute_blocked(a: MatRef<'_, T>) -> Result<Self, LuError> {
        const BLOCK_SIZE: usize = 64;
        Self::compute_with_block_size(a, BLOCK_SIZE)
    }

    /// Computes LU decomposition with a specified block size.
    pub fn compute_with_block_size(a: MatRef<'_, T>, nb: usize) -> Result<Self, LuError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(LuError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(Lu {
                lu: Mat::zeros(0, 0),
                pivot: Vec::new(),
                num_swaps: 0,
            });
        }

        // Copy A into LU matrix
        let mut lu = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                lu[(i, j)] = a[(i, j)];
            }
        }

        let mut pivot = vec![0usize; n];
        let mut num_swaps = 0;

        // Use blocked algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, nb)?;
        } else {
            Self::unblocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0)?;
        }

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }

    /// Blocked LU factorization using GEMM for Schur complement updates.
    fn blocked_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        nb: usize,
    ) -> Result<(), LuError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor the current panel (columns jb:jb+jb_size)
            Self::factor_panel(lu, pivot, num_swaps, n, jb, jb_size)?;

            // If there are more columns after this panel
            if jb + jb_size < n {
                // Apply row interchanges to columns jb+jb_size:n
                for k in jb..jb + jb_size {
                    let pk = pivot[k];
                    if pk != k {
                        for j in (jb + jb_size)..n {
                            let tmp = lu[(k, j)];
                            lu[(k, j)] = lu[(pk, j)];
                            lu[(pk, j)] = tmp;
                        }
                    }
                }

                // Solve L11 * U12 = A12 using TRSM
                // Extract L11 (lower triangular with unit diagonal)
                let mut l11: Mat<T> = Mat::zeros(jb_size, jb_size);
                for i in 0..jb_size {
                    l11[(i, i)] = T::one();
                    for j in 0..i {
                        l11[(i, j)] = lu[(jb + i, jb + j)];
                    }
                }

                // Extract and update U12 block
                let mut u12: Mat<T> = Mat::zeros(jb_size, n - jb - jb_size);
                for j in 0..(n - jb - jb_size) {
                    for i in 0..jb_size {
                        u12[(i, j)] = lu[(jb + i, jb + jb_size + j)];
                    }
                }

                // Solve L11 * U12 = A12 (in-place on u12)
                let _ = trsm_in_place(
                    Side::Left,
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::Unit,
                    l11.as_ref(),
                    u12.as_mut(),
                );

                // Copy U12 back
                for j in 0..(n - jb - jb_size) {
                    for i in 0..jb_size {
                        lu[(jb + i, jb + jb_size + j)] = u12[(i, j)];
                    }
                }

                // Update trailing submatrix: A22 -= L21 * U12 using GEMM
                let rows_remaining = n - jb - jb_size;

                // Extract L21 block
                let mut l21: Mat<T> = Mat::zeros(rows_remaining, jb_size);
                for j in 0..jb_size {
                    for i in 0..rows_remaining {
                        l21[(i, j)] = lu[(jb + jb_size + i, jb + j)];
                    }
                }

                // Compute update = L21 * U12 and subtract from A22
                let mut update: Mat<T> = Mat::zeros(rows_remaining, n - jb - jb_size);
                gemm(
                    T::one(),
                    l21.as_ref(),
                    u12.as_ref(),
                    T::zero(),
                    update.as_mut(),
                );

                // A22 -= update
                for j in 0..(n - jb - jb_size) {
                    for i in 0..rows_remaining {
                        lu[(jb + jb_size + i, jb + jb_size + j)] =
                            lu[(jb + jb_size + i, jb + jb_size + j)] - update[(i, j)];
                    }
                }
            }

            jb += jb_size;
        }

        Ok(())
    }

    /// Factor a panel of columns using unblocked algorithm.
    fn factor_panel(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        jb: usize,
        jb_size: usize,
    ) -> Result<(), LuError> {
        for k in jb..(jb + jb_size) {
            // Find pivot in column k, rows k..n
            let mut pivot_row = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);

            for i in (k + 1)..n {
                let val = Scalar::abs(lu[(i, k)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }

            // Check for singularity
            let tol = T::epsilon()
                * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
            if pivot_val <= tol {
                return Err(LuError::Singular { index: k });
            }

            pivot[k] = pivot_row;

            // Swap rows k and pivot_row in columns 0..jb+jb_size
            if pivot_row != k {
                for j in 0..(jb + jb_size) {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                *num_swaps += 1;
            }

            // Compute multipliers and update within the panel
            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;

                // Update remaining columns in this panel
                for j in (k + 1)..(jb + jb_size) {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }

        Ok(())
    }

    /// Unblocked LU factorization (for small matrices or panels).
    fn unblocked_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        start: usize,
    ) -> Result<(), LuError> {
        for k in start..n {
            // Find pivot: largest absolute value in column k, rows k..n
            let mut pivot_row = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);

            for i in (k + 1)..n {
                let val = Scalar::abs(lu[(i, k)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }

            // Check for singularity
            let tol = T::epsilon()
                * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
            if pivot_val <= tol {
                return Err(LuError::Singular { index: k });
            }

            pivot[k] = pivot_row;

            // Swap rows if needed
            if pivot_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                *num_swaps += 1;
            }

            // Compute multipliers and update
            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;

                for j in (k + 1)..n {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }

        Ok(())
    }
}

// Optimized automatic algorithm selection for f64 and f32
impl<T: Field + GemmKernel + bytemuck::Zeroable> Lu<T> {
    /// Computes the LU decomposition with automatic algorithm selection.
    ///
    /// For matrices with size ≥ 128, automatically uses the blocked algorithm
    /// for better cache efficiency and performance. Otherwise uses the unblocked
    /// algorithm which has less overhead for small matrices.
    ///
    /// This method is available for f32 and f64 types which have optimized GEMM kernels.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::lu::Lu;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 256;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         a[(i, j)] = ((i + j) % 10 + 1) as f64;
    ///     }
    ///     a[(i, i)] += 100.0; // Make it diagonally dominant
    /// }
    ///
    /// // Automatically uses blocked algorithm for n >= 128
    /// let lu = Lu::compute_auto(a.as_ref()).unwrap();
    /// ```
    pub fn compute_auto(a: MatRef<'_, T>) -> Result<Self, LuError> {
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
