//! LU decomposition with partial pivoting.
//!
//! This is the standard LU decomposition algorithm used by LAPACK's DGETRF.
//! For large matrices, uses blocked algorithm with GEMM/TRSM for cache efficiency.

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_blas::level3::gemm::gemm;
#[cfg(feature = "parallel")]
use oxiblas_blas::level3::gemm::gemm_with_par;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_blas::level3::trsm::{Diag, Side, Trans, Uplo, trsm_in_place};
#[cfg(feature = "parallel")]
use oxiblas_core::parallel::Par;
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

/// Computes the infinity-norm (maximum absolute row sum) of `a`.
///
/// This is the scale reference consumed by [`singular_tol`]. It is deliberately
/// computed once from the *original* input matrix rather than from the
/// in-progress factor: doing so keeps the singularity decision byte-for-byte
/// identical across the unblocked, blocked, recursive and parallel variants,
/// which all factor the same matrix but visit its columns in different orders
/// (a norm taken mid-factorization would reflect the partially-eliminated Schur
/// complement, not `A`, and would drift between variants).
///
/// Works for complex `T` as well as real `T` because it accumulates
/// `Scalar::abs` (the modulus), which is always a real, non-negative
/// `T::Real`.
fn matrix_inf_norm<T: Field>(a: MatRef<'_, T>) -> T::Real {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let mut max_row = <T::Real as Zero>::zero();
    for i in 0..nrows {
        let mut row_sum = <T::Real as Zero>::zero();
        for j in 0..ncols {
            row_sum = row_sum + Scalar::abs(a[(i, j)]);
        }
        if row_sum > max_row {
            max_row = row_sum;
        }
    }
    max_row
}

/// Relative singularity tolerance `eps * ||A||_inf * n`.
///
/// # Why relative, not absolute
///
/// A bare absolute threshold such as `eps * n` is *not* scale-invariant, so it
/// falsely rejects perfectly well-conditioned matrices whose entries merely
/// happen to be small in magnitude. Concretely, `1e-16 * I` is trivially
/// invertible (its inverse is `1e16 * I`), yet every one of its pivots equals
/// `1e-16`, which is below `f64::EPSILON * n ≈ 2.2e-16 * n`, so an absolute
/// check would report it singular.
///
/// Scaling the threshold by `||A||_inf` makes the test *relative* to the
/// matrix's own magnitude, mirroring LAPACK's scale-aware rank/condition
/// checks: a pivot is treated as negligible only when it is tiny *compared to
/// the entries of `A`*, i.e. when the corresponding column is numerically
/// linearly dependent. A genuinely rank-deficient matrix still leaves a pivot
/// of order `eps * ||A||` after elimination and is correctly flagged, while a
/// uniformly scaled matrix (any `c * A`) yields the same pass/fail outcome as
/// `A` itself.
///
/// The comparison against this tolerance uses `<=`, not `<`, so an exactly
/// zero pivot in the zero matrix (`||A||_inf == 0`, hence `tol == 0`) is still
/// reported singular instead of driving a division by zero.
fn singular_tol<T: Field>(anorm: T::Real, n: usize) -> T::Real {
    let n_real =
        <T::Real as FromPrimitive>::from_usize(n).unwrap_or_else(<T::Real as One>::one);
    T::epsilon() * anorm * n_real
}

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

        // Scale reference for the relative singularity tolerance (see `singular_tol`).
        let anorm = matrix_inf_norm(a);

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

            // Reject the pivot only if it is negligible *relative to* ||A||;
            // an absolute threshold would wrongly flag well-conditioned matrices
            // with uniformly small entries (see `singular_tol`).
            if pivot_val <= singular_tol::<T>(anorm, n) {
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

        // Apply the row permutation P to b, forming Pb in x.
        //
        // P is stored as the sequence of interchanges performed during
        // factorization: at step k, row k was exchanged with row `pivot[k]`.
        // Replaying that same sequence in ascending k reproduces Pb — this is
        // LAPACK's forward application of interchanges (cf. DLASWP). We copy b
        // verbatim first and then swap in place so that a chain of pivots
        // (e.g. 0->2 followed by 1->2) is *composed* correctly, rather than
        // applied as independent per-row assignments which would be wrong for
        // any permutation that is not a single transposition.
        for j in 0..m {
            for i in 0..n {
                x[(i, j)] = b[(i, j)];
            }
        }
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

        // Scale reference for the relative singularity tolerance (see `singular_tol`),
        // taken from the original matrix so every code path uses the same threshold.
        let anorm = matrix_inf_norm(a);

        // Use blocked algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, nb, anorm)?;
        } else {
            Self::unblocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, anorm)?;
        }

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }

    /// Blocked LU factorization using GEMM for Schur complement updates.
    ///
    /// `anorm` is the infinity-norm of the *original* matrix, threaded through
    /// to the panel factorization for the scale-aware singularity check.
    fn blocked_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        nb: usize,
        anorm: T::Real,
    ) -> Result<(), LuError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor the current panel (columns jb:jb+jb_size)
            Self::factor_panel(lu, pivot, num_swaps, n, jb, jb_size, anorm)?;

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
    ///
    /// `anorm` is the infinity-norm of the *original* matrix, used for the
    /// scale-aware singularity check (see [`singular_tol`]).
    fn factor_panel(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        jb: usize,
        jb_size: usize,
        anorm: T::Real,
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

            // Reject the pivot only if it is negligible *relative to* ||A||;
            // an absolute threshold would wrongly flag well-conditioned matrices
            // with uniformly small entries (see `singular_tol`).
            if pivot_val <= singular_tol::<T>(anorm, n) {
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
    ///
    /// `anorm` is the infinity-norm of the *original* matrix, used for the
    /// scale-aware singularity check (see [`singular_tol`]).
    fn unblocked_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        start: usize,
        anorm: T::Real,
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

            // Reject the pivot only if it is negligible *relative to* ||A||;
            // an absolute threshold would wrongly flag well-conditioned matrices
            // with uniformly small entries (see `singular_tol`).
            if pivot_val <= singular_tol::<T>(anorm, n) {
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

// Recursive cache-oblivious LU factorization with partial pivoting
impl<T: Field + GemmKernel + bytemuck::Zeroable> Lu<T> {
    /// Recursion threshold: matrices at or below this size use the unblocked algorithm.
    const RECURSIVE_THRESHOLD: usize = 64;

    /// Computes the LU decomposition using a recursive cache-oblivious algorithm.
    ///
    /// This divide-and-conquer approach automatically adapts to the cache hierarchy
    /// by recursively splitting the matrix. At each level:
    ///
    /// 1. Split A into left panel (n x n1) and right panel (n x n2)
    /// 2. Recursively factor the left panel to get [L11, U11, P1] and L21
    /// 3. Apply P1 to the right panel
    /// 4. Solve U12 = L11^{-1} * A12 via TRSM
    /// 5. Update Schur complement: A22 -= L21 * U12 via GEMM
    /// 6. Recursively factor the Schur complement to get [L22, U22, P2]
    /// 7. Apply P2 to L21
    ///
    /// For matrices smaller than the recursion threshold (64), falls back to the
    /// unblocked algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - A square matrix
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::lu::Lu;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 200;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 / 100.0;
    ///     }
    ///     a[(i, i)] += 10.0; // Make diagonally dominant
    /// }
    ///
    /// let lu = Lu::compute_recursive(a.as_ref()).expect("Matrix is non-singular");
    /// let det = lu.determinant();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `LuError::NotSquare` if the matrix is not square.
    /// Returns `LuError::Singular` if the matrix is singular.
    pub fn compute_recursive(a: MatRef<'_, T>) -> Result<Self, LuError> {
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

        // Scale reference for the relative singularity tolerance (see `singular_tol`).
        let anorm = matrix_inf_norm(a);

        Self::recursive_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, n, anorm)?;

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }

    /// Recursive LU factorization on a submatrix.
    ///
    /// Factors columns `col_start..col_start+width` of the full n x n matrix `lu`,
    /// considering all rows from `col_start..n` for pivoting.
    ///
    /// # Arguments
    ///
    /// * `lu` - The full n x n working matrix (modified in place)
    /// * `pivot` - Pivot indices array (global indices)
    /// * `num_swaps` - Counter for row swaps
    /// * `n` - Full matrix dimension
    /// * `col_start` - Starting column of the current panel
    /// * `width` - Number of columns to factor in this call
    /// * `anorm` - Infinity-norm of the original matrix (scale-aware singularity check)
    fn recursive_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        col_start: usize,
        width: usize,
        anorm: T::Real,
    ) -> Result<(), LuError> {
        if width == 0 {
            return Ok(());
        }

        // Base case: use unblocked panel factorization for small widths
        if width <= Self::RECURSIVE_THRESHOLD {
            // Factor columns col_start..col_start+width as a panel, considering all rows
            Self::factor_panel(lu, pivot, num_swaps, n, col_start, width, anorm)?;

            // If there are trailing columns, apply updates
            let trailing_cols = n - col_start - width;
            if trailing_cols > 0 {
                // Apply row interchanges to trailing columns
                for k in col_start..col_start + width {
                    let pk = pivot[k];
                    if pk != k {
                        for j in (col_start + width)..n {
                            let tmp = lu[(k, j)];
                            lu[(k, j)] = lu[(pk, j)];
                            lu[(pk, j)] = tmp;
                        }
                    }
                }

                // Extract L11 (lower triangular with unit diagonal, width x width)
                let mut l11 = Mat::zeros(width, width);
                for i in 0..width {
                    l11[(i, i)] = T::one();
                    for j in 0..i {
                        l11[(i, j)] = lu[(col_start + i, col_start + j)];
                    }
                }

                // Extract U12 block (width x trailing_cols)
                let mut u12 = Mat::zeros(width, trailing_cols);
                for j in 0..trailing_cols {
                    for i in 0..width {
                        u12[(i, j)] = lu[(col_start + i, col_start + width + j)];
                    }
                }

                // Solve L11 * U12 = A12 via TRSM
                let _ = trsm_in_place(
                    Side::Left,
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::Unit,
                    l11.as_ref(),
                    u12.as_mut(),
                );

                // Copy U12 back
                for j in 0..trailing_cols {
                    for i in 0..width {
                        lu[(col_start + i, col_start + width + j)] = u12[(i, j)];
                    }
                }

                // Update trailing submatrix: A22 -= L21 * U12
                let rows_below = n - col_start - width;
                if rows_below > 0 {
                    // Extract L21 (rows_below x width)
                    let mut l21 = Mat::zeros(rows_below, width);
                    for j in 0..width {
                        for i in 0..rows_below {
                            l21[(i, j)] = lu[(col_start + width + i, col_start + j)];
                        }
                    }

                    // Compute update = L21 * U12
                    let mut update = Mat::zeros(rows_below, trailing_cols);
                    gemm(
                        T::one(),
                        l21.as_ref(),
                        u12.as_ref(),
                        T::zero(),
                        update.as_mut(),
                    );

                    // A22 -= update
                    for j in 0..trailing_cols {
                        for i in 0..rows_below {
                            lu[(col_start + width + i, col_start + width + j)] =
                                lu[(col_start + width + i, col_start + width + j)] - update[(i, j)];
                        }
                    }
                }
            }

            return Ok(());
        }

        // Recursive case: split the width in half
        let n1 = width / 2;
        let n2 = width - n1;

        // Step 1: Recursively factor the left half (columns col_start..col_start+n1)
        // This will also update the right half via the trailing column logic in the base case
        Self::recursive_factor(lu, pivot, num_swaps, n, col_start, n1, anorm)?;

        // Step 2: Now recursively factor the right half (columns col_start+n1..col_start+width)
        // The Schur complement for these columns has already been computed in step 1
        Self::recursive_factor(lu, pivot, num_swaps, n, col_start + n1, n2, anorm)?;

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

// Parallel blocked LU factorization
#[cfg(feature = "parallel")]
impl<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync> Lu<T> {
    /// Computes the LU decomposition using a parallel blocked algorithm.
    ///
    /// Parallelizes the GEMM (Schur complement) updates within the blocked
    /// factorization using Rayon. For matrices smaller than the block size,
    /// falls back to the sequential unblocked algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix A (n x n)
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
    pub fn compute_blocked_par(a: MatRef<'_, T>) -> Result<Self, LuError> {
        const BLOCK_SIZE: usize = 64;
        Self::compute_blocked_par_with_block_size(a, BLOCK_SIZE)
    }

    /// Computes parallel blocked LU decomposition with a specified block size.
    pub fn compute_blocked_par_with_block_size(
        a: MatRef<'_, T>,
        nb: usize,
    ) -> Result<Self, LuError> {
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

        // Scale reference for the relative singularity tolerance (see `singular_tol`).
        let anorm = matrix_inf_norm(a);

        // Use blocked parallel algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor_par(&mut lu, &mut pivot, &mut num_swaps, n, nb, anorm)?;
        } else {
            Self::unblocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, anorm)?;
        }

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }

    /// Blocked LU factorization with parallel GEMM for Schur complement updates.
    ///
    /// `anorm` is the infinity-norm of the *original* matrix, threaded through
    /// to the panel factorization for the scale-aware singularity check.
    fn blocked_factor_par(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        nb: usize,
        anorm: T::Real,
    ) -> Result<(), LuError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor the current panel (columns jb:jb+jb_size) -- sequential
            Self::factor_panel(lu, pivot, num_swaps, n, jb, jb_size, anorm)?;

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

                // Solve L11 * U12 = A12 (TRSM internally uses parallel GEMM)
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

                // Update trailing submatrix: A22 -= L21 * U12 using parallel GEMM
                let rows_remaining = n - jb - jb_size;

                // Extract L21 block
                let mut l21: Mat<T> = Mat::zeros(rows_remaining, jb_size);
                for j in 0..jb_size {
                    for i in 0..rows_remaining {
                        l21[(i, j)] = lu[(jb + jb_size + i, jb + j)];
                    }
                }

                // Compute update = L21 * U12 using parallel GEMM and subtract from A22
                let mut update: Mat<T> = Mat::zeros(rows_remaining, n - jb - jb_size);
                gemm_with_par(
                    T::one(),
                    l21.as_ref(),
                    u12.as_ref(),
                    T::zero(),
                    update.as_mut(),
                    Par::Rayon,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense matrix-vector product `a * x`, returned as a `Vec`.
    fn matvec(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
        let n = a.nrows();
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                s += a[(i, j)] * xj;
            }
            y[i] = s;
        }
        y
    }

    /// Asserts that `P A == L U` (the defining identity of the factorization),
    /// using an absolute tolerance scaled by the matrix magnitude so it works
    /// for both `O(1)` and tiny-magnitude inputs.
    fn assert_pa_eq_lu(lu: &Lu<f64>, a: &Mat<f64>, scale: f64) {
        let n = a.nrows();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();

        for i in 0..n {
            for j in 0..n {
                let mut pa = 0.0;
                let mut prod = 0.0;
                for k in 0..n {
                    pa += p[(i, k)] * a[(k, j)];
                    prod += l[(i, k)] * u[(k, j)];
                }
                let diff = (pa - prod).abs();
                assert!(
                    diff <= 1e-10 * scale,
                    "PA[{i},{j}] = {pa} != LU[{i},{j}] = {prod} (diff {diff})",
                );
            }
        }
    }

    /// The exact scenario from the audit finding: a well-conditioned matrix whose
    /// entries are simply small in magnitude (`1e-16 * I`). The old *absolute*
    /// tolerance `eps * n` (≈ 1.1e-15 here) exceeds every pivot (`1e-16`) and so
    /// wrongly reported this trivially-invertible matrix as singular. The
    /// corrected *relative* tolerance must accept it.
    #[test]
    fn test_tiny_identity_not_singular() {
        const C: f64 = 1e-16;
        let n = 5;
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = C;
        }

        let lu = Lu::compute(a.as_ref())
            .expect("1e-16 * I is perfectly invertible and must not be flagged singular");

        // det = C^n, exactly representable for these small n.
        let det = lu.determinant();
        let expected_det = C.powi(n as i32);
        let rel = ((det - expected_det) / expected_det).abs();
        assert!(rel < 1e-10, "det = {det}, expected {expected_det}");

        // A^-1 = (1/C) * I = 1e16 * I.
        let inv = lu.inverse().expect("should invert");
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 / C } else { 0.0 };
                let diff = (inv[(i, j)] - expected).abs();
                assert!(diff <= 1e-10 / C, "inv[{i},{j}] = {}", inv[(i, j)]);
            }
        }
    }

    /// A tiny-magnitude, well-conditioned matrix that *also* requires a genuine
    /// row interchange (the `(0,0)` entry is zero). Exercises the relative
    /// tolerance and the (de-duplicated) permutation path together.
    #[test]
    fn test_tiny_wellcond_needs_pivot_not_singular() {
        const C: f64 = 1e-16;
        // base is invertible (det = -22) and forces a swap at step 0.
        let base: Mat<f64> =
            Mat::from_rows(&[&[0.0, 2.0, 1.0], &[4.0, 3.0, 1.0], &[2.0, 1.0, 3.0]]);
        let mut a: Mat<f64> = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                a[(i, j)] = C * base[(i, j)];
            }
        }

        let x_true = [1.0, -2.0, 0.5];
        let b_vec = matvec(&a, &x_true);
        let b: Mat<f64> = Mat::from_rows(&[&[b_vec[0]], &[b_vec[1]], &[b_vec[2]]]);

        let lu = Lu::compute(a.as_ref())
            .expect("tiny well-conditioned matrix must not be flagged singular");
        let x = lu.solve(b.as_ref()).expect("should solve");

        for i in 0..3 {
            assert!(
                (x[(i, 0)] - x_true[i]).abs() < 1e-8,
                "x[{i}] = {}, expected {}",
                x[(i, 0)],
                x_true[i],
            );
        }
        assert_pa_eq_lu(&lu, &a, C);
    }

    /// A *genuinely* singular matrix at tiny scale must still be detected: the
    /// relative tolerance must not be so loose that it accepts rank deficiency.
    /// `1e-16 * [[1,2],[2,4]]` has a second pivot of exactly zero after
    /// elimination.
    #[test]
    fn test_tiny_singular_still_detected() {
        const C: f64 = 1e-16;
        let a: Mat<f64> = Mat::from_rows(&[&[C, 2.0 * C], &[2.0 * C, 4.0 * C]]);
        let result = Lu::compute(a.as_ref());
        assert!(
            matches!(result, Err(LuError::Singular { .. })),
            "rank-deficient matrix must be reported singular, got {result:?}",
        );
    }

    /// Regression across *all* factorization variants (unblocked, blocked and
    /// recursive) at once: a large (`n = 80`), well-conditioned but tiny-scale
    /// (`1e-16`) diagonally-dominant matrix. With the old absolute tolerance
    /// every variant's panel factorization aborted with `Singular`; the relative
    /// tolerance accepts it, and all variants must agree on the solution.
    #[test]
    fn test_tiny_scale_all_variants() {
        const C: f64 = 1e-16;
        let n = 80;
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let off = ((i * 7 + j * 13) % 5) as f64 * 0.001;
                let base = if i == j { 4.0 } else { off };
                a[(i, j)] = C * base;
            }
        }

        let mut x_true = vec![0.0f64; n];
        for (i, xi) in x_true.iter_mut().enumerate() {
            *xi = (i % 7) as f64 - 3.0;
        }
        let b_vec = matvec(&a, &x_true);
        let mut b: Mat<f64> = Mat::zeros(n, 1);
        for i in 0..n {
            b[(i, 0)] = b_vec[i];
        }

        let check = |lu: &Lu<f64>, label: &str| {
            let x = lu.solve(b.as_ref()).expect("should solve");
            for i in 0..n {
                assert!(
                    (x[(i, 0)] - x_true[i]).abs() < 1e-8,
                    "{label}: x[{i}] = {}, expected {}",
                    x[(i, 0)],
                    x_true[i],
                );
            }
        };

        let lu_unblocked = Lu::compute(a.as_ref()).expect("unblocked must not flag singular");
        check(&lu_unblocked, "unblocked");

        let lu_blocked =
            Lu::compute_blocked(a.as_ref()).expect("blocked must not flag singular");
        check(&lu_blocked, "blocked");

        let lu_recursive =
            Lu::compute_recursive(a.as_ref()).expect("recursive must not flag singular");
        check(&lu_recursive, "recursive");

        #[cfg(feature = "parallel")]
        {
            let lu_par = Lu::compute_blocked_par(a.as_ref())
                .expect("parallel blocked must not flag singular");
            check(&lu_par, "parallel");
        }
    }

    /// Guards the removal of the two dead permutation blocks in `solve`. The
    /// matrix is a row-shuffle of a diagonally-dominant (hence non-singular)
    /// matrix, so partial pivoting must apply a *chain* of interchanges to
    /// recover the natural order — precisely the case the deleted
    /// per-row-assignment code got wrong. Both the solution and the `PA = LU`
    /// identity must hold.
    #[test]
    fn test_solve_pivot_permutation_chain() {
        // Rows of a diagonally-dominant matrix, shuffled so that pivoting is
        // forced at several steps.
        let a: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 1.0, 10.0],
            &[2.0, 1.0, 10.0, 1.0],
            &[1.0, 10.0, 1.0, 2.0],
            &[10.0, 1.0, 2.0, 1.0],
        ]);

        let x_true = [1.0, 2.0, 3.0, 4.0];
        let b_vec = matvec(&a, &x_true);
        let b: Mat<f64> =
            Mat::from_rows(&[&[b_vec[0]], &[b_vec[1]], &[b_vec[2]], &[b_vec[3]]]);

        let lu = Lu::compute(a.as_ref()).expect("should not be singular");

        // The permutation must be non-trivial (at least one swap), otherwise the
        // test would not exercise the permutation path at all.
        let swapped = lu.pivot().iter().enumerate().any(|(k, &pk)| k != pk);
        assert!(swapped, "test matrix should force at least one row interchange");

        let x = lu.solve(b.as_ref()).expect("should solve");
        for i in 0..4 {
            assert!(
                (x[(i, 0)] - x_true[i]).abs() < 1e-10,
                "x[{i}] = {}, expected {}",
                x[(i, 0)],
                x_true[i],
            );
        }

        // Multiple right-hand sides through the same (chained) permutation.
        let b_multi: Mat<f64> = Mat::from_rows(&[
            &[b_vec[0], 1.0],
            &[b_vec[1], 0.0],
            &[b_vec[2], 0.0],
            &[b_vec[3], 0.0],
        ]);
        let x_multi = lu.solve(b_multi.as_ref()).expect("should solve multi-RHS");
        for i in 0..4 {
            assert!(
                (x_multi[(i, 0)] - x_true[i]).abs() < 1e-10,
                "multi x[{i},0] = {}, expected {}",
                x_multi[(i, 0)],
                x_true[i],
            );
        }

        assert_pa_eq_lu(&lu, &a, 1.0);
    }
}
