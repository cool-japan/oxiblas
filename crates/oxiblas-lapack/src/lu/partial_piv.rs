//! LU decomposition with partial pivoting.
//!
//! This is the standard LU decomposition algorithm used by LAPACK's DGETRF.
//! [`Lu::compute`] always runs the unblocked (column-by-column) algorithm,
//! whatever the size. The blocked algorithm with GEMM/TRSM updates, for cache
//! efficiency on large matrices, is used by [`Lu::compute_blocked`] /
//! [`Lu::compute_with_block_size`] (once the order reaches the block size) and
//! by [`Lu::compute_auto`], which calls `compute_blocked` from order 128.

use super::tolerance::lu_pivot_tolerances;
use oxiblas_blas::level3::gemm::gemm_with_par;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_blas::level3::trsm::{Diag, Side, Trans, Uplo, trsm_in_place_with_par};
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

/// Elimination step `k`: scales column `k` below the diagonal by
/// `1 / lu[(k, k)]`, which turns it into the multipliers of L, then subtracts
/// the rank-1 update `L[k+1.., k] * U[k, j]` from every column `j` in
/// `k+1..col_end`, rows `k+1..`.
///
/// The storage is column-major, so each column is a contiguous slice and the
/// update streams down columns instead of striding across rows, with no
/// per-element bounds check. Every element still gets exactly the operations
/// of the row-oriented formulation, on the same operands
/// (`l_ik = a_ik * (1 / a_kk)`, then `a_ij - l_ik * a_kj`), so the factors are
/// bit-identical to it: only the order of independent updates changes.
///
/// Also used by the full and rook pivoting factorizations, whose elimination
/// step is the same once their pivot is in place.
pub(super) fn eliminate_column<T: Field>(lu: &mut Mat<T>, k: usize, col_end: usize) {
    let n = lu.nrows();
    let ld = lu.row_stride();
    let pivot_inv = T::one() / lu[(k, k)];
    let data = lu.raw_data_mut();

    let col_k = k * ld;
    for l_ik in &mut data[col_k + k + 1..col_k + n] {
        *l_ik = *l_ik * pivot_inv;
    }

    for j in (k + 1)..col_end {
        // Column k ends before column j starts (k < j and n <= ld), so
        // splitting at the start of column j borrows both disjointly.
        let (before, from_j) = data.split_at_mut(j * ld);
        let l_col = &before[col_k + k + 1..col_k + n];
        let a_col = &mut from_j[..n];
        let u_kj = a_col[k];
        for (a_ij, &l_ik) in a_col[k + 1..].iter_mut().zip(l_col) {
            *a_ij = *a_ij - l_ik * u_kj;
        }
    }
}

/// Applies the row interchanges `pivot[k0..k1]`, in order, to the columns
/// `c0..c1` of `lu`.
///
/// Each column is a contiguous slice, so the interchanges are applied one
/// column at a time (as LAPACK `?laswp` does) instead of walking strided rows;
/// the result is the same, they only move data.
fn apply_row_swaps<T: Field>(
    lu: &mut Mat<T>,
    pivot: &[usize],
    k0: usize,
    k1: usize,
    c0: usize,
    c1: usize,
) {
    let swaps = &pivot[k0..k1];
    if (k0..k1).zip(swaps).all(|(k, &pk)| pk == k) {
        return;
    }
    let n = lu.nrows();
    let ld = lu.row_stride();
    let data = lu.raw_data_mut();
    for j in c0..c1 {
        let col = &mut data[j * ld..j * ld + n];
        for (k, &pk) in (k0..k1).zip(swaps) {
            col.swap(k, pk);
        }
    }
}

/// Finishes the block row of panel `jb..jb + w` and updates the columns
/// `jb + w..stop` below it, in place, once the panel columns are factored:
///
/// 1. applies the panel's row interchanges to those columns;
/// 2. solves `L11 · U12 = A12` (TRSM with the unit lower triangle of the
///    panel's diagonal block);
/// 3. updates `A22 := A22 − L21 · U12` with one GEMM (`alpha = −1`,
///    `beta = 1`), for every row below the panel.
///
/// The blocked drivers pass `stop = n` (the whole trailing submatrix); the
/// recursive panel factorization passes the end of the panel it splits.
///
/// Every operand is a view of `lu`, so nothing is copied. TRSM reads only the
/// strictly lower triangle of `L11` (unit diagonal), never the `U11` entries
/// stored in the same block.
///
/// `par` is handed to TRSM and GEMM. Both give bitwise the same result for
/// every `Par` (TRSM cuts the right-hand sides into fixed panels and GEMM keeps
/// the sequential K partition), so the parallel drivers reproduce the
/// sequential ones exactly.
///
/// Compared with forming `L21 · U12` in a zeroed temporary and subtracting it
/// afterwards, the in-place GEMM accumulates each K block of its partition
/// into `A22` directly: when the panel is wider than one K block of the GEMM,
/// `(a − p₁) − p₂` replaces `a − (p₁ + p₂)`, which can change the last bits.
fn update_trailing<T: Field + GemmKernel + bytemuck::Zeroable>(
    lu: &mut Mat<T>,
    pivot: &[usize],
    jb: usize,
    w: usize,
    stop: usize,
    par: Par,
) -> Result<(), LuError> {
    let n = lu.nrows();
    let end = jb + w;
    if end >= stop {
        return Ok(());
    }
    let (r, cols) = (n - end, stop - end);

    apply_row_swaps(lu, pivot, jb, end, end, stop);

    let (left, right) = lu.as_mut().split_cols(end);
    let left = left.rb();
    let l11 = left.submatrix(jb, jb, w, w);
    let l21 = left.submatrix(end, jb, r, w);
    let (mut u12, a22) = right.submatrix(jb, 0, w + r, cols).split_rows(w);

    let solved = trsm_in_place_with_par(
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::Unit,
        l11,
        u12.rb_mut(),
        par,
    );
    // A unit-diagonal solve fails only on inconsistent shapes, and these
    // come from the panel; report a shape error rather than ignore it.
    debug_assert!(solved.is_ok(), "TRSM of a factored LU panel failed");
    solved.map_err(|_| LuError::DimensionMismatch {
        expected: w,
        actual: u12.nrows(),
    })?;

    gemm_with_par(-T::one(), l21, u12.rb(), T::one(), a22, par);
    Ok(())
}

/// Panel width up to which the blocked drivers factor a panel with the
/// unblocked kernel instead of splitting it ([`Lu::factor_panel_recursive`]).
const PANEL_BASE: usize = 16;

/// LU decomposition with partial (row) pivoting.
///
/// Stores the factorization PA = LU where:
/// - P is a permutation matrix (stored as pivot indices)
/// - L is lower triangular with unit diagonal
/// - U is upper triangular
///
/// The L and U factors are stored compactly in a single matrix,
/// with L below the diagonal and U on and above the diagonal.
///
/// The pivot of column `j` is treated as zero, and reported as
/// [`LuError::Singular`], when its magnitude is at most `eps · n · s_j`, where
/// `s_j = max_i |a_ij|` is taken over the finite entries of column `j` of the
/// input. The threshold is relative to each column: scaling the columns of a
/// real matrix by powers of two, `A · diag(c)`, changes neither the pivots
/// nor the outcome (only column `j` of U, by `c_j`), as long as no entry or
/// intermediate value overflows or underflows. A badly scaled but
/// well-conditioned matrix such as `diag(1e20, 1)` therefore factors, while a
/// column that elimination reduces to rounding noise is still rejected.
#[derive(Clone, Debug)]
pub struct Lu<T: Scalar> {
    /// Combined L and U factors.
    /// L is stored below the diagonal (with implicit unit diagonal).
    /// U is stored on and above the diagonal.
    lu: Mat<T>,
    /// Pivot indices: row i was swapped with row `pivot[i]`.
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

        // Per-column singularity thresholds `eps · n · max_i |a_ij|`, computed
        // once from the input.
        let col_tol = lu_pivot_tolerances(a);

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

            // Check for singularity against the threshold of column k. A NaN
            // pivot fails this comparison and propagates instead of being
            // reported as singular.
            if pivot_val <= col_tol[k] {
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

            // Compute multipliers (L's subdiagonal entries) and update the
            // remaining submatrix, column by column.
            eliminate_column(&mut lu, k, n);
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

        let mut det = if self.num_swaps.is_multiple_of(2) {
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

        // Pb: copy b, then apply the row interchanges in factorization order.
        // `pivot` records the interchanges (row k was swapped with row
        // pivot[k] at step k, as LAPACK's IPIV), not a permutation map, so it
        // must be replayed swap by swap.
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
        let nb = crate::workspace::optimal_block_size_lu(a.nrows(), a.ncols());
        Self::compute_with_block_size(a, nb)
    }

    /// Computes LU decomposition with a specified block size.
    ///
    /// A block size of 0 is treated as 1 (one column per panel).
    pub fn compute_with_block_size(a: MatRef<'_, T>, nb: usize) -> Result<Self, LuError> {
        // A zero-width panel would never advance the blocked loop.
        let nb = nb.max(1);
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
        // Per-column singularity thresholds, shared by every path below.
        let col_tol = lu_pivot_tolerances(a);

        // Use blocked algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor(
                &mut lu,
                &mut pivot,
                &mut num_swaps,
                n,
                nb,
                &col_tol,
                Par::Seq,
            )?;
        } else {
            Self::unblocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, &col_tol)?;
        }

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }

    /// Blocked right-looking LU factorization (LAPACK `?getrf`).
    ///
    /// Each panel of `nb` columns is factored by
    /// [`Self::factor_panel_recursive`]; its row interchanges are then applied
    /// to the columns left of it, and [`update_trailing`] applies them to the
    /// columns right of it, solves for the block row of U with TRSM and
    /// updates the trailing submatrix with one GEMM, in place on `lu`. `par`
    /// is passed to TRSM and GEMM (also inside the panel); the rest runs on
    /// the calling thread.
    ///
    /// `col_tol[j]` is the singularity threshold of column `j`, computed from
    /// the input matrix before factorizing.
    #[allow(clippy::too_many_arguments)]
    fn blocked_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        nb: usize,
        col_tol: &[T::Real],
        par: Par,
    ) -> Result<(), LuError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor the current panel (columns jb:jb+jb_size)
            Self::factor_panel_recursive(lu, pivot, num_swaps, n, jb, jb_size, col_tol, par)?;

            // The panel's row interchanges for the columns left of it
            apply_row_swaps(lu, pivot, jb, jb + jb_size, 0, jb);

            // Row interchanges, U12 and the Schur complement of the columns
            // right of the panel
            update_trailing(lu, pivot, jb, jb_size, n, par)?;

            jb += jb_size;
        }

        Ok(())
    }

    /// Factors the panel `jb..jb + w` (rows `jb..n`) recursively, as LAPACK
    /// `?getrf2` does: the left half is factored, its row interchanges and
    /// block row of U are applied to the right half and the right half's
    /// Schur complement is updated by [`update_trailing`] (TRSM and GEMM),
    /// then the right half is factored and its interchanges are applied to
    /// the left half. Panels of at most [`PANEL_BASE`] columns are factored
    /// by [`Self::factor_panel`].
    ///
    /// The row interchanges are applied to the panel's columns only; the
    /// caller applies them to the columns outside it. Most of the panel's
    /// multiply-adds run in GEMM instead of rank-1 updates that stream the
    /// whole panel through memory for every column.
    #[allow(clippy::too_many_arguments)]
    fn factor_panel_recursive(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        jb: usize,
        w: usize,
        col_tol: &[T::Real],
        par: Par,
    ) -> Result<(), LuError> {
        if w <= PANEL_BASE {
            return Self::factor_panel(lu, pivot, num_swaps, n, jb, w, jb, col_tol);
        }
        let n1 = w / 2;
        Self::factor_panel_recursive(lu, pivot, num_swaps, n, jb, n1, col_tol, par)?;
        update_trailing(lu, pivot, jb, n1, jb + w, par)?;
        Self::factor_panel_recursive(lu, pivot, num_swaps, n, jb + n1, w - n1, col_tol, par)?;
        apply_row_swaps(lu, pivot, jb + n1, jb + w, jb, jb + n1);
        Ok(())
    }

    /// Factor a panel of columns using unblocked algorithm.
    ///
    /// Each row interchange is applied to the columns `swap_from..jb +
    /// jb_size`, so `swap_from = 0` swaps whole rows up to the end of the
    /// panel and `swap_from = jb` leaves the columns left of the panel to
    /// the caller.
    ///
    /// `col_tol[j]` is the singularity threshold of column `j`, computed from
    /// the input matrix before factorizing.
    #[allow(clippy::too_many_arguments)]
    fn factor_panel(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        jb: usize,
        jb_size: usize,
        swap_from: usize,
        col_tol: &[T::Real],
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

            // Check for singularity against the threshold of column k (a NaN
            // pivot fails the comparison)
            if pivot_val <= col_tol[k] {
                return Err(LuError::Singular { index: k });
            }

            pivot[k] = pivot_row;

            // Swap rows k and pivot_row in columns swap_from..jb+jb_size
            if pivot_row != k {
                for j in swap_from..(jb + jb_size) {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                *num_swaps += 1;
            }

            // Compute multipliers and update the remaining columns of the panel
            eliminate_column(lu, k, jb + jb_size);
        }

        Ok(())
    }

    /// Unblocked LU factorization (for small matrices or panels).
    ///
    /// `col_tol[j]` is the singularity threshold of column `j`, computed from
    /// the input matrix before factorizing.
    fn unblocked_factor(
        lu: &mut Mat<T>,
        pivot: &mut [usize],
        num_swaps: &mut usize,
        n: usize,
        start: usize,
        col_tol: &[T::Real],
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

            // Check for singularity against the threshold of column k (a NaN
            // pivot fails the comparison)
            if pivot_val <= col_tol[k] {
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

            // Compute multipliers and update the trailing submatrix
            eliminate_column(lu, k, n);
        }

        Ok(())
    }
}

// Recursive cache-oblivious LU factorization with partial pivoting
impl<T: Field + GemmKernel + bytemuck::Zeroable> Lu<T> {
    /// Computes the LU decomposition using a recursive cache-oblivious algorithm.
    ///
    /// This divide-and-conquer approach (LAPACK `?getrf2`) automatically
    /// adapts to the cache hierarchy by recursively splitting the columns. At
    /// each level:
    ///
    /// 1. Split A into left panel (n x n1) and right panel (n x n2)
    /// 2. Recursively factor the left panel to get [L11, U11, P1] and L21
    /// 3. Apply P1 to the right panel
    /// 4. Solve U12 = L11^{-1} * A12 via TRSM
    /// 5. Update Schur complement: A22 -= L21 * U12 via GEMM
    /// 6. Recursively factor the Schur complement to get [L22, U22, P2]
    /// 7. Apply P2 to L21
    ///
    /// Every step works in place on the factor's storage. Panels of at most
    /// 16 columns are factored by the unblocked algorithm.
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
        // Per-column singularity thresholds, shared by every path below.
        let col_tol = lu_pivot_tolerances(a);

        Self::factor_panel_recursive(
            &mut lu,
            &mut pivot,
            &mut num_swaps,
            n,
            0,
            n,
            &col_tol,
            Par::Seq,
        )?;

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }
}

// Automatic algorithm selection, for the element types with GEMM micro-kernels
// (f32, f64, Complex32 and Complex64).
impl<T: Field + GemmKernel + bytemuck::Zeroable> Lu<T> {
    /// Computes the LU decomposition with automatic algorithm selection.
    ///
    /// For matrices with size ≥ 128, automatically uses the blocked algorithm
    /// for better cache efficiency and performance. Otherwise uses the unblocked
    /// algorithm which has less overhead for small matrices.
    ///
    /// This method is available for the element types with GEMM micro-kernels
    /// (`f32`, `f64`, `Complex32` and `Complex64`).
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
    /// Same algorithm as [`Self::compute_blocked`], with the TRSM (block row
    /// of U) and GEMM (Schur complement) updates of every panel run on the
    /// Rayon pool; the panels themselves are factored on the calling thread.
    /// Both kernels give bitwise the same result for any number of threads,
    /// so this returns exactly the factors of [`Self::compute_blocked`]. For
    /// matrices smaller than the block size, falls back to the sequential
    /// unblocked algorithm.
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
        let nb = crate::workspace::optimal_block_size_lu(a.nrows(), a.ncols());
        Self::compute_blocked_par_with_block_size(a, nb)
    }

    /// Computes parallel blocked LU decomposition with a specified block size.
    ///
    /// Returns exactly the factors of [`Self::compute_with_block_size`] with
    /// the same `nb`; a block size of 0 is treated as 1 there and here.
    pub fn compute_blocked_par_with_block_size(
        a: MatRef<'_, T>,
        nb: usize,
    ) -> Result<Self, LuError> {
        // A zero-width panel would never advance the blocked loop.
        let nb = nb.max(1);
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
        // Per-column singularity thresholds, shared by every path below.
        let col_tol = lu_pivot_tolerances(a);

        // Blocked algorithm with parallel TRSM/GEMM updates for larger matrices
        if n >= nb {
            Self::blocked_factor(
                &mut lu,
                &mut pivot,
                &mut num_swaps,
                n,
                nb,
                &col_tol,
                Par::Rayon,
            )?;
        } else {
            Self::unblocked_factor(&mut lu, &mut pivot, &mut num_swaps, n, 0, &col_tol)?;
        }

        Ok(Lu {
            lu,
            pivot,
            num_swaps,
        })
    }
}

#[cfg(test)]
#[path = "partial_piv_tests.rs"]
mod tests;
