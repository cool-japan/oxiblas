//! LL^T Cholesky decomposition.
//!
//! Standard Cholesky factorization for symmetric positive definite matrices.
//! [`Cholesky::compute`] always runs the unblocked (column-by-column)
//! algorithm, whatever the size. The blocked algorithm with TRSM, GEMM and the
//! triangle-only SYRK, for cache efficiency on large matrices, is used by
//! [`Cholesky::compute_blocked`] / [`Cholesky::compute_with_block_size`] (once
//! the order reaches the block size) and by [`Cholesky::compute_auto`], which
//! calls `compute_blocked` from order 128.

use crate::lu::tolerance::llt_pivot_tolerances;
use oxiblas_blas::level3::gemm::gemm_transposed_with_par;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_blas::level3::syrk::syrk;
use oxiblas_blas::level3::trsm::{Diag, Side, Trans, TrsmError, Uplo, trsm_in_place_with_par};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Order up to which [`lower_update`] updates a diagonal block with a single
/// triangle-only SYRK instead of splitting it further.
const LOWER_UPDATE_LEAF: usize = 64;

/// `C := C − A · Aᵀ` on the lower triangle of the square view `c`, where `a`
/// has as many rows as `c` (the symmetric rank-k update of a blocked
/// Cholesky, LAPACK `?syrk` with `uplo = 'L'`).
///
/// The triangle is split recursively: the off-diagonal rectangle `C₂₁ −=
/// A₂ · A₁ᵀ` is one GEMM, the two diagonal halves recurse, and diagonal blocks
/// of order at most [`LOWER_UPDATE_LEAF`] are updated by one call of the
/// triangle-only `oxiblas_blas` SYRK. That does about half the multiply-adds
/// of a full `C −= A · Aᵀ`, in a few large GEMMs. Only the lower triangle of
/// `c` is read or written, so the zero strict upper triangle the drivers
/// start from stays exactly zero.
///
/// `Aᵀ` is read in place by the transpose-aware GEMM and by SYRK (no copy).
/// `par` is passed to every rectangle GEMM; a SYRK leaf runs on the thread
/// that reaches it.
///
/// # Errors
///
/// Returns [`CholeskyError::DimensionMismatch`] if SYRK rejects the shapes,
/// which the callers' views rule out.
fn lower_update<T: Field + GemmKernel + bytemuck::Zeroable>(
    c: MatMut<'_, T>,
    a: MatRef<'_, T>,
    par: Par,
) -> Result<(), CholeskyError> {
    let m = c.nrows();
    if m == 0 {
        return Ok(());
    }
    if m <= LOWER_UPDATE_LEAF {
        return syrk(Uplo::Lower, Trans::NoTrans, -T::one(), a, T::one(), c).map_err(|_| {
            CholeskyError::DimensionMismatch {
                expected: m,
                actual: a.nrows(),
            }
        });
    }
    let m1 = m / 2;
    let (c_left, c_right) = c.split_cols(m1);
    let (c11, c21) = c_left.split_rows(m1);
    let (_, c22) = c_right.split_rows(m1);
    let (a1, a2) = a.split_rows(m1);
    let rectangle = move || {
        gemm_transposed_with_par(
            Trans::NoTrans,
            Trans::Trans,
            -T::one(),
            a2,
            a1,
            T::one(),
            c21,
            par,
        );
    };
    // The three blocks are disjoint and read only `a`, so with a parallel
    // `par` they are updated concurrently (the small diagonal leaves would
    // otherwise run one after another); every block gets the same result
    // either way.
    #[cfg(feature = "parallel")]
    if !par.is_sequential() {
        let ((), (upper_half, lower_half)) = rayon::join(rectangle, || {
            rayon::join(|| lower_update(c11, a1, par), || lower_update(c22, a2, par))
        });
        return upper_half.and(lower_half);
    }
    rectangle();
    lower_update(c11, a1, par)?;
    lower_update(c22, a2, par)
}

/// Finishes the block column of the factored diagonal block `jb..jb + w` and
/// updates the rest of the trailing block `jb + w..stop`, in place on `l`:
///
/// 1. `L21 := A21 · L11⁻ᵀ` (TRSM with the lower triangle of the diagonal
///    block), for the rows `jb + w..stop`;
/// 2. `A22 := A22 − L21 · L21ᵀ` on the lower triangle of the trailing
///    diagonal block `jb + w..stop` ([`lower_update`]).
///
/// Every operand is a view of `l`; TRSM reads only the lower triangle of
/// `L11` and nothing writes above the diagonal of `l`. `par` is passed to
/// TRSM and GEMM, which give bitwise the same result for every `Par` (the
/// SYRK leaves of [`lower_update`] are sequential).
fn update_trailing<T: Field + Real + GemmKernel + bytemuck::Zeroable>(
    l: &mut Mat<T>,
    jb: usize,
    w: usize,
    stop: usize,
    par: Par,
) -> Result<(), CholeskyError> {
    let end = jb + w;
    if end >= stop {
        return Ok(());
    }
    let r = stop - end;

    let (left, right) = l.as_mut().split_cols(end);
    let (l11, mut a21) = left.submatrix(jb, jb, w + r, w).split_rows(w);
    let l11 = l11.rb();

    let solved = trsm_in_place_with_par(
        Side::Right,
        Uplo::Lower,
        Trans::Trans,
        Diag::NonUnit,
        l11,
        a21.rb_mut(),
        par,
    );
    // The diagonal of L11 holds square roots of radicands that passed the
    // positive-definiteness test, so it has no zero and the shapes come from
    // the block: TRSM cannot fail. Report a failure rather than ignore it.
    debug_assert!(solved.is_ok(), "TRSM of a factored Cholesky block failed");
    solved.map_err(|err| match err {
        TrsmError::Singular => CholeskyError::NotPositiveDefinite {
            index: jb + (0..w).find(|&i| l11[(i, i)] == T::zero()).unwrap_or(0),
        },
        TrsmError::NotSquare | TrsmError::DimensionMismatch => CholeskyError::DimensionMismatch {
            expected: w,
            actual: a21.ncols(),
        },
    })?;

    lower_update(right.submatrix(end, 0, r, r), a21.rb(), par)
}

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
///
/// Every variant reports [`CholeskyError::NotPositiveDefinite`] when the
/// radicand `a_jj - Σ l_jk²` of column `j` is at or below `eps · n · a_jj`,
/// where `a_jj` is the original diagonal entry (a NaN or infinite `a_jj`
/// counts as 0, as does a negative one). The threshold is relative to each
/// column: a symmetric scaling `D · A · D` by powers of two scales the radicand
/// and `a_jj` by `d_j²` alike, so it changes neither the outcome nor the
/// reported index (L becomes `D · L`), as long as no entry or intermediate
/// value overflows or underflows. A badly scaled but positive definite matrix
/// such as `diag(1e20, 1)` therefore factors.
///
/// A NaN radicand fails the comparison and propagates into L instead of being
/// reported, like the NaN pivots of this crate's LU factorizations. LAPACK
/// `dpotrf` differs here: it reports a NaN radicand as `INFO = j`.
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

        // Per-column positive-definiteness thresholds `eps · n · a_jj`, from
        // the finite diagonal entries of A.
        let col_tol = llt_pivot_tolerances(a);

        // Copy lower triangular part of A
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        // Cholesky-Banachiewicz arithmetic, evaluated column by column
        Self::unblocked_factor_block(&mut l, 0, n, &col_tol)?;

        Ok(Cholesky { l })
    }

    /// Unblocked Cholesky of the diagonal block `jb..jb + jb_size`, in place
    /// (the whole factorization for `jb = 0`, `jb_size = n`).
    ///
    /// Cholesky-Banachiewicz (dot-product) arithmetic, evaluated column by
    /// column: entry `(g, gi)` becomes `(a_{g,gi} − Σ_k l_{g,k} · l_{gi,k}) /
    /// l_{gi,gi}`, the sum running over the block's earlier columns `k` in
    /// increasing order from zero. The sums of one column are accumulated
    /// together, streaming down the contiguous columns `k` of the
    /// column-major storage, which gives every entry bit for bit the result
    /// of evaluating its dot product along the (strided) rows. Only the lower
    /// triangle of the block is read or written; rows below it are left to
    /// TRSM.
    ///
    /// `col_tol[j]` is the positive-definiteness threshold of column `j`,
    /// computed from the input matrix before factorizing.
    fn unblocked_factor_block(
        l: &mut Mat<T>,
        jb: usize,
        jb_size: usize,
        col_tol: &[T],
    ) -> Result<(), CholeskyError> {
        let ld = l.row_stride();
        let end = jb + jb_size;
        let data = l.raw_data_mut();
        // acc[g - jb] accumulates the dot product of rows g and gi
        let mut acc = vec![T::zero(); jb_size];

        for gi in jb..end {
            let sums = &mut acc[gi - jb..];
            sums.fill(T::zero());
            for k in jb..gi {
                // Rows gi..end of column k; its first entry is l[gi, k].
                let col = &data[k * ld + gi..k * ld + end];
                let l_ik = col[0];
                for (sum, &l_gk) in sums.iter_mut().zip(col) {
                    *sum = *sum + l_gk * l_ik;
                }
            }

            let col = &mut data[gi * ld + gi..gi * ld + end];
            let diag = col[0] - sums[0];

            // Check for positive definiteness (a NaN radicand propagates)
            if diag <= col_tol[gi] {
                return Err(CholeskyError::NotPositiveDefinite { index: gi });
            }

            let l_ii = Real::sqrt(diag);
            col[0] = l_ii;
            for (l_gi, &sum) in col[1..].iter_mut().zip(&sums[1..]) {
                *l_gi = (*l_gi - sum) / l_ii;
            }
        }

        Ok(())
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
        let nb = crate::workspace::optimal_block_size_cholesky(a.nrows());
        Self::compute_with_block_size(a, nb)
    }

    /// Computes Cholesky decomposition with a specified block size.
    ///
    /// A block size of 0 is treated as 1 (one column per diagonal block).
    pub fn compute_with_block_size(a: MatRef<'_, T>, nb: usize) -> Result<Self, CholeskyError> {
        // A zero-width block would never advance the blocked loop.
        let nb = nb.max(1);
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

        // Per-column positive-definiteness thresholds `eps · n · a_jj`, from
        // the original diagonal, shared by every path below.
        let col_tol = llt_pivot_tolerances(a);

        // Copy lower triangular part of A
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        // Use blocked algorithm for larger matrices
        if n >= nb {
            Self::blocked_factor(&mut l, n, nb, &col_tol, Par::Seq)?;
        } else {
            Self::unblocked_factor_block(&mut l, 0, n, &col_tol)?;
        }

        Ok(Cholesky { l })
    }

    /// Blocked right-looking Cholesky factorization (LAPACK `?potrf`).
    ///
    /// Each diagonal block of `nb` columns is factored by
    /// [`Self::unblocked_factor_block`]; [`update_trailing`] then solves for
    /// the block column below it with TRSM and updates the lower triangle of
    /// the trailing submatrix with GEMM and SYRK, in place on `l`. `par` is
    /// passed to TRSM and GEMM; the diagonal blocks are factored on the
    /// calling thread. The strict upper triangle of `l` is never written, so
    /// it keeps the zeros it starts with.
    ///
    /// `col_tol[j]` is the positive-definiteness threshold of column `j`,
    /// computed from the input matrix before factorizing.
    fn blocked_factor(
        l: &mut Mat<T>,
        n: usize,
        nb: usize,
        col_tol: &[T],
        par: Par,
    ) -> Result<(), CholeskyError> {
        let mut jb = 0;

        while jb < n {
            // Current block size (may be smaller for last block)
            let jb_size = nb.min(n - jb);

            // Factor diagonal block A11 using unblocked Cholesky
            Self::unblocked_factor_block(l, jb, jb_size, col_tol)?;

            // L21 and the lower triangle of the trailing submatrix
            update_trailing(l, jb, jb_size, n, par)?;

            jb += jb_size;
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
    /// 3. Update the lower triangle of A22 -= L21 * L21^T (symmetric rank-k
    ///    update) with GEMM on its off-diagonal blocks and the triangle-only
    ///    SYRK on its diagonal blocks
    /// 4. Factor the updated A22 recursively to get L22
    ///
    /// Every step works in place on the factor's storage. Blocks of order at
    /// most the recursion threshold (64) are factored by the unblocked
    /// algorithm, which is more efficient at that scale.
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

        // Per-column positive-definiteness thresholds `eps · n · a_jj`, from
        // the original diagonal, shared by every path below.
        let col_tol = llt_pivot_tolerances(a);

        // Copy lower triangular part of A into L
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        Self::recursive_factor(&mut l, n, 0, &col_tol)?;

        Ok(Cholesky { l })
    }

    /// Recursive Cholesky factorization on a submatrix starting at (offset, offset).
    ///
    /// Operates in-place on the lower triangular part of `l`.
    /// The submatrix from (offset, offset) to (offset+size-1, offset+size-1) is factored.
    /// Nothing above the diagonal is written.
    ///
    /// `col_tol[j]` is the positive-definiteness threshold of column `j`,
    /// computed from the input matrix before factorizing.
    fn recursive_factor(
        l: &mut Mat<T>,
        size: usize,
        offset: usize,
        col_tol: &[T],
    ) -> Result<(), CholeskyError> {
        // Base case: use unblocked algorithm for small matrices
        if size <= Self::RECURSIVE_THRESHOLD {
            return Self::unblocked_factor_block(l, offset, size, col_tol);
        }

        // Split: n1 = size/2 (upper half), n2 = size - n1 (lower half)
        let n1 = size / 2;

        // Step 1: Recursively factor A11 (top-left n1 x n1 block)
        Self::recursive_factor(l, n1, offset, col_tol)?;

        // Steps 2 and 3: L21 = A21 * L11^{-T} (TRSM), then the lower triangle
        // of A22 -= L21 * L21^T (GEMM and SYRK), in place
        update_trailing(l, offset, n1, offset + size, Par::Seq)?;

        // Step 4: Recursively factor updated A22
        Self::recursive_factor(l, size - n1, offset + n1, col_tol)
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
    /// Same algorithm as [`Self::compute_blocked`], with the TRSM (block
    /// column of L) and GEMM/SYRK (symmetric rank-k) updates of every block
    /// run on the Rayon pool; the diagonal blocks themselves are factored on
    /// the calling thread. TRSM and GEMM give bitwise the same result for any
    /// number of threads and each SYRK leaf runs on one thread, so this
    /// returns exactly the factor of
    /// [`Self::compute_blocked`]. For matrices smaller than the block size,
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
        let nb = crate::workspace::optimal_block_size_cholesky(a.nrows());
        Self::compute_blocked_par_with_block_size(a, nb)
    }

    /// Computes parallel blocked Cholesky decomposition with a specified block size.
    ///
    /// Returns exactly the factor of [`Self::compute_with_block_size`] with
    /// the same `nb`; a block size of 0 is treated as 1 there and here.
    pub fn compute_blocked_par_with_block_size(
        a: MatRef<'_, T>,
        nb: usize,
    ) -> Result<Self, CholeskyError> {
        // A zero-width block would never advance the blocked loop.
        let nb = nb.max(1);
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

        // Per-column positive-definiteness thresholds `eps · n · a_jj`, from
        // the original diagonal, shared by every path below.
        let col_tol = llt_pivot_tolerances(a);

        // Copy lower triangular part of A
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }

        // Blocked algorithm with parallel TRSM/GEMM updates for larger matrices
        if n >= nb {
            Self::blocked_factor(&mut l, n, nb, &col_tol, Par::Rayon)?;
        } else {
            Self::unblocked_factor_block(&mut l, 0, n, &col_tol)?;
        }

        Ok(Cholesky { l })
    }
}

#[cfg(test)]
#[path = "llt_tests.rs"]
mod tests;
