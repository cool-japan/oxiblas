//! Advanced pivoting strategies for sparse LU factorization.
//!
//! Provides three complementary pivoting strategies:
//!
//! - **Threshold pivoting** (`SparseLuThreshold`): Only pivots when the diagonal element
//!   is below a threshold relative to the largest element in the column. This reduces
//!   fill-in compared to full partial pivoting while maintaining numerical stability.
//!
//! - **Static pivoting** (`compute_static_pivoting`): Uses the ordering-determined pivot
//!   sequence but replaces small pivots with a perturbation (SuperLU-style). This preserves
//!   sparsity at the cost of some numerical accuracy.
//!
//! - **Diagonal pivoting / Bunch-Kaufman** (`SparseLdlt`): For symmetric indefinite matrices,
//!   uses 1x1 and 2x2 pivot blocks following the Bunch-Kaufman strategy. Produces an
//!   LDL^T factorization where D is block-diagonal with 1x1 and 2x2 blocks.

use crate::csc::CscMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::lu::SparseLUError;

// ---------------------------------------------------------------------------
// Threshold pivoting for sparse LU
// ---------------------------------------------------------------------------

/// Sparse LU factorization with threshold (Markowitz-style) pivoting.
///
/// Instead of always choosing the largest element in the column as pivot,
/// this strategy only pivots when the diagonal element is smaller than
/// `threshold * max_column_element`. When the diagonal is "good enough",
/// it is kept, which preserves sparsity better than full partial pivoting.
///
/// # Threshold semantics
///
/// - `threshold = 1.0` is equivalent to full partial pivoting (always pick max).
/// - `threshold = 0.0` is equivalent to no pivoting (always use diagonal).
/// - Typical values: 0.01 to 0.1.
#[derive(Debug, Clone)]
pub struct SparseLuThreshold<T: Scalar> {
    /// Lower triangular factor (unit diagonal, stored in CSC).
    l: CscMatrix<T>,
    /// Upper triangular factor (stored in CSC).
    u: CscMatrix<T>,
    /// Row permutation.
    perm: Vec<usize>,
    /// Inverse row permutation.
    #[allow(dead_code)]
    perm_inv: Vec<usize>,
    /// Threshold value used.
    threshold: T,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseLuThreshold<T> {
    /// Computes LU factorization with threshold pivoting using default threshold (0.1).
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SparseLUError> {
        // 0.1 = default threshold
        let threshold = T::from_f64(0.1).unwrap_or(T::zero());
        Self::with_threshold(a, threshold)
    }

    /// Computes LU factorization with the specified threshold.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix in CSC format
    /// * `threshold` - Pivoting threshold in [0, 1]. A row swap occurs only when
    ///   `|diag| < threshold * |max_in_column|`.
    ///
    /// # Errors
    ///
    /// Returns `SparseLUError::NotSquare` if the matrix is not square, or
    /// `SparseLUError::Singular` if a zero pivot is encountered.
    pub fn with_threshold(a: &CscMatrix<T>, threshold: T) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        let mut perm: Vec<usize> = (0..n).collect();

        // Dense working storage
        let mut l_data = vec![vec![T::zero(); n]; n];
        let mut u_data = vec![vec![T::zero(); n]; n];

        let mut work = vec![vec![T::zero(); n]; n];
        for col in 0..n {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                work[a.row_indices()[idx]][col] = a.values()[idx].clone();
            }
        }

        for k in 0..n {
            // Find maximum absolute value in column k, rows k..n
            let mut max_val = T::zero();
            let mut max_row = k;

            for i in k..n {
                let perm_i = perm[i];
                let val = Scalar::abs(work[perm_i][k].clone());
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val <= <T as Scalar>::epsilon() {
                return Err(SparseLUError::Singular { row: k });
            }

            // Threshold pivoting: only swap if diagonal is too small relative to max
            let diag_val = Scalar::abs(work[perm[k]][k].clone());
            if diag_val < threshold.clone() * max_val {
                // Diagonal is insufficiently large; swap with the max row
                perm.swap(k, max_row);
            }
            // Otherwise keep the diagonal (less fill-in)

            let pivot_row = perm[k];
            let pivot = work[pivot_row][k].clone();

            if Scalar::abs(pivot.clone()) <= <T as Scalar>::epsilon() {
                return Err(SparseLUError::Singular { row: k });
            }

            u_data[k][k] = pivot.clone();

            for i in (k + 1)..n {
                let row_i = perm[i];
                let lik = work[row_i][k].clone() / pivot.clone();

                l_data[i][k] = lik.clone();

                for j in (k + 1)..n {
                    work[row_i][j] =
                        work[row_i][j].clone() - lik.clone() * work[pivot_row][j].clone();
                }
            }

            for j in (k + 1)..n {
                u_data[k][j] = work[pivot_row][j].clone();
            }
        }

        // Set L diagonal to 1
        for i in 0..n {
            l_data[i][i] = T::one();
        }

        let l = dense_to_csc_lower(&l_data);
        let u = dense_to_csc_upper(&u_data);

        let mut perm_inv = vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        Ok(Self {
            l,
            u,
            perm,
            perm_inv,
            threshold,
        })
    }

    /// Returns the lower triangular factor L (unit diagonal).
    pub fn l(&self) -> &CscMatrix<T> {
        &self.l
    }

    /// Returns the upper triangular factor U.
    pub fn u(&self) -> &CscMatrix<T> {
        &self.u
    }

    /// Returns the row permutation.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the threshold used.
    pub fn threshold(&self) -> &T {
        &self.threshold
    }

    /// Solves A * x = b.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        let mut b_perm = vec![T::zero(); n];
        for i in 0..n {
            b_perm[i] = b[self.perm[i]].clone();
        }

        let y = super::triangular::solve_lower_csc(&self.l, &b_perm);
        super::triangular::solve_upper_csc(&self.u, &y)
    }
}

// ---------------------------------------------------------------------------
// Static pivoting (SuperLU-style)
// ---------------------------------------------------------------------------

/// Sparse LU factorization with static pivoting (SuperLU-style).
///
/// Uses the natural ordering (no row permutation) but replaces any pivot
/// whose absolute value is below `epsilon` with `sign(pivot) * epsilon`
/// (or just `epsilon` if the pivot is exactly zero). This ensures the
/// factorization always succeeds and preserves the sparsity pattern, at
/// the cost of introducing a small perturbation.
///
/// This is particularly useful when the fill-reducing ordering is good
/// but a few small pivots cause issues.
#[derive(Debug, Clone)]
pub struct SparseLuStaticPivot<T: Scalar> {
    /// Lower triangular factor (unit diagonal, stored in CSC).
    l: CscMatrix<T>,
    /// Upper triangular factor (stored in CSC).
    u: CscMatrix<T>,
    /// Perturbation threshold used.
    epsilon: T,
    /// Number of pivots that were perturbed.
    num_perturbed: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseLuStaticPivot<T> {
    /// Computes LU factorization with static pivoting.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix in CSC format
    /// * `epsilon` - Small pivots with `|pivot| < epsilon` are replaced by
    ///   `sign(pivot) * epsilon`. Must be positive.
    ///
    /// # Errors
    ///
    /// Returns `SparseLUError::NotSquare` if the matrix is not square.
    /// Unlike standard LU, this never returns `Singular` because small
    /// pivots are perturbed.
    pub fn new(a: &CscMatrix<T>, epsilon: T) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        let mut num_perturbed = 0usize;

        let mut l_data = vec![vec![T::zero(); n]; n];
        let mut u_data = vec![vec![T::zero(); n]; n];

        let mut work = vec![vec![T::zero(); n]; n];
        for col in 0..n {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                work[a.row_indices()[idx]][col] = a.values()[idx].clone();
            }
        }

        for k in 0..n {
            let mut pivot = work[k][k].clone();

            // Static pivoting: perturb small pivots
            if Scalar::abs(pivot.clone()) < epsilon.clone() {
                num_perturbed += 1;
                if pivot >= T::zero() {
                    pivot = epsilon.clone();
                } else {
                    pivot = T::zero() - epsilon.clone();
                }
                work[k][k] = pivot.clone();
            }

            u_data[k][k] = pivot.clone();

            for i in (k + 1)..n {
                let lik = work[i][k].clone() / pivot.clone();
                l_data[i][k] = lik.clone();

                for j in (k + 1)..n {
                    work[i][j] = work[i][j].clone() - lik.clone() * work[k][j].clone();
                }
            }

            for j in (k + 1)..n {
                u_data[k][j] = work[k][j].clone();
            }
        }

        for i in 0..n {
            l_data[i][i] = T::one();
        }

        let l = dense_to_csc_lower(&l_data);
        let u = dense_to_csc_upper(&u_data);

        Ok(Self {
            l,
            u,
            epsilon,
            num_perturbed,
        })
    }

    /// Returns the lower triangular factor L (unit diagonal).
    pub fn l(&self) -> &CscMatrix<T> {
        &self.l
    }

    /// Returns the upper triangular factor U.
    pub fn u(&self) -> &CscMatrix<T> {
        &self.u
    }

    /// Returns the perturbation threshold used.
    pub fn epsilon(&self) -> &T {
        &self.epsilon
    }

    /// Returns how many pivots were perturbed.
    pub fn num_perturbed(&self) -> usize {
        self.num_perturbed
    }

    /// Solves A * x = b (approximately, due to pivot perturbation).
    ///
    /// When pivots have been perturbed, the solution is an approximation.
    /// Use iterative refinement for improved accuracy.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        let y = super::triangular::solve_lower_csc(&self.l, b);
        super::triangular::solve_upper_csc(&self.u, &y)
    }
}

// ---------------------------------------------------------------------------
// Sparse LDL^T with Bunch-Kaufman diagonal pivoting
// ---------------------------------------------------------------------------

/// Pivot block type in Bunch-Kaufman factorization.
#[derive(Debug, Clone, PartialEq)]
pub enum PivotBlock<T: Scalar> {
    /// A 1x1 pivot at index `idx` with value `d`.
    OneByOne {
        /// Global index.
        idx: usize,
        /// Pivot value.
        d: T,
    },
    /// A 2x2 pivot block at indices `(idx1, idx2)` with values `[[d11, d21], [d21, d22]]`.
    TwoByTwo {
        /// First global index.
        idx1: usize,
        /// Second global index.
        idx2: usize,
        /// (1,1) element.
        d11: T,
        /// (2,1) = (1,2) element.
        d21: T,
        /// (2,2) element.
        d22: T,
    },
}

/// Sparse LDL^T factorization with Bunch-Kaufman diagonal pivoting.
///
/// For symmetric indefinite matrices, computes P * A * P^T = L * D * L^T where:
/// - P is a symmetric permutation
/// - L is unit lower triangular
/// - D is block-diagonal with 1x1 and 2x2 blocks
///
/// The Bunch-Kaufman strategy chooses between 1x1 and 2x2 pivots to
/// maintain bounded element growth without needing full pivoting.
///
/// # Algorithm
///
/// At each step k, the algorithm examines the reduced matrix A_k and decides:
/// 1. If the diagonal element `|a_kk|` is large enough relative to the largest
///    off-diagonal in column k, use a 1x1 pivot.
/// 2. Otherwise, find the row r with the largest off-diagonal in column k and
///    examine the 2x2 submatrix at (k,r). If the 2x2 pivot is acceptable, use it.
/// 3. If neither works, swap rows/columns and retry with a 1x1 pivot.
///
/// The growth factor alpha = (1 + sqrt(17)) / 8 ~ 0.6404 is the Bunch-Kaufman constant.
#[derive(Debug, Clone)]
pub struct SparseLdlt<T: Scalar> {
    /// Lower triangular factor L (unit diagonal, stored dense).
    l_data: Vec<Vec<T>>,
    /// Block-diagonal D pivot blocks.
    pivots: Vec<PivotBlock<T>>,
    /// Symmetric permutation.
    perm: Vec<usize>,
    /// Inverse permutation.
    #[allow(dead_code)]
    perm_inv: Vec<usize>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SparseLdlt<T> {
    /// Bunch-Kaufman constant alpha = (1 + sqrt(17)) / 8.
    fn bk_alpha() -> T {
        // alpha = (1 + sqrt(17)) / 8 ~ 0.6404
        T::from_f64(0.6403882032022076).unwrap_or(T::one())
    }

    /// Computes the LDL^T factorization with Bunch-Kaufman pivoting.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric square matrix in CSC format. Only the lower triangle
    ///   (including diagonal) is accessed, but the full matrix may be provided.
    ///
    /// # Errors
    ///
    /// Returns `SparseLUError::NotSquare` if not square, or
    /// `SparseLUError::Singular` if the matrix is exactly singular.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SparseLUError> {
        if a.nrows() != a.ncols() {
            return Err(SparseLUError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            return Ok(Self {
                l_data: vec![],
                pivots: vec![],
                perm: vec![],
                perm_inv: vec![],
                n: 0,
            });
        }

        let alpha = Self::bk_alpha();

        // Symmetrize into dense working storage
        let mut work = vec![vec![T::zero(); n]; n];
        for col in 0..n {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                let row = a.row_indices()[idx];
                let val = a.values()[idx].clone();
                work[row][col] = val.clone();
                work[col][row] = val;
            }
        }

        let mut perm: Vec<usize> = (0..n).collect();
        let mut l_data = vec![vec![T::zero(); n]; n];
        for i in 0..n {
            l_data[i][i] = T::one();
        }

        let mut pivots: Vec<PivotBlock<T>> = Vec::new();

        let mut k = 0usize;
        while k < n {
            if k == n - 1 {
                // Last element: must be 1x1 pivot
                let akk = work[perm[k]][perm[k]].clone();
                if Scalar::abs(akk.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: k });
                }
                pivots.push(PivotBlock::OneByOne { idx: k, d: akk });
                k += 1;
                continue;
            }

            // Find lambda_1 = max |a(i,k)| for i != k (in permuted indices)
            let pk = perm[k];
            let akk = work[pk][pk].clone();
            let mut lambda1 = T::zero();
            let mut r = k + 1; // row index of max off-diagonal in column k

            for i in (k + 1)..n {
                let pi = perm[i];
                let val = Scalar::abs(work[pi][pk].clone());
                if val > lambda1 {
                    lambda1 = val;
                    r = i;
                }
            }

            if Scalar::abs(lambda1.clone()) <= <T as Scalar>::epsilon() {
                // Column k has no off-diagonal entries; use 1x1 pivot
                if Scalar::abs(akk.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: k });
                }
                pivots.push(PivotBlock::OneByOne {
                    idx: k,
                    d: akk.clone(),
                });
                // No L updates needed (column is zero below diagonal)
                k += 1;
                continue;
            }

            // Test 1: |a_kk| >= alpha * lambda_1 => use 1x1 pivot
            if Scalar::abs(akk.clone()) >= alpha.clone() * lambda1.clone() {
                // 1x1 pivot at k
                let pivot = akk.clone();
                self_apply_1x1_pivot(&mut work, &mut l_data, &perm, k, n, pivot.clone());
                pivots.push(PivotBlock::OneByOne { idx: k, d: pivot });
                k += 1;
                continue;
            }

            // Find sigma = max |a(i,r)| for i != r (in permuted row r)
            let pr = perm[r];
            let mut sigma = T::zero();
            for i in k..n {
                if i == r {
                    continue;
                }
                let pi = perm[i];
                let val = Scalar::abs(work[pi][pr].clone());
                if val > sigma {
                    sigma = val;
                }
            }

            // Test 2: |a_kk| * sigma >= alpha * lambda_1^2
            let lhs = Scalar::abs(akk.clone()) * sigma.clone();
            let rhs = alpha.clone() * lambda1.clone() * lambda1.clone();

            if lhs >= rhs {
                // 1x1 pivot at k
                let pivot = akk.clone();
                if Scalar::abs(pivot.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: k });
                }
                self_apply_1x1_pivot(&mut work, &mut l_data, &perm, k, n, pivot.clone());
                pivots.push(PivotBlock::OneByOne { idx: k, d: pivot });
                k += 1;
                continue;
            }

            // Test 3: |a_rr| >= alpha * sigma => swap r to position k, use 1x1
            let arr = work[pr][pr].clone();
            if Scalar::abs(arr.clone()) >= alpha.clone() * sigma.clone() {
                // Swap rows/cols k and r in permutation
                perm.swap(k, r);
                let pivot = arr.clone();
                if Scalar::abs(pivot.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: k });
                }
                self_apply_1x1_pivot(&mut work, &mut l_data, &perm, k, n, pivot.clone());
                pivots.push(PivotBlock::OneByOne { idx: k, d: pivot });
                k += 1;
                continue;
            }

            // Use 2x2 pivot at (k, k+1) after swapping r to position k+1
            if r != k + 1 {
                perm.swap(k + 1, r);
            }

            let pk = perm[k];
            let pk1 = perm[k + 1];

            let d11 = work[pk][pk].clone();
            let d21 = work[pk1][pk].clone();
            let d22 = work[pk1][pk1].clone();

            // Determinant of 2x2 block
            let det = d11.clone() * d22.clone() - d21.clone() * d21.clone();
            if Scalar::abs(det.clone()) <= <T as Scalar>::epsilon() {
                return Err(SparseLUError::Singular { row: k });
            }

            // Inverse of 2x2 block D^{-1} = (1/det) * [[d22, -d21], [-d21, d11]]
            let inv_det = T::one() / det;
            let inv_d11 = d22.clone() * inv_det.clone();
            let inv_d21 = (T::zero() - d21.clone()) * inv_det.clone();
            let inv_d22 = d11.clone() * inv_det;

            // Compute L entries for rows i > k+1
            // L(i, k:k+1) = A(i, k:k+1) * D^{-1}
            let num_update = n - k - 2;
            let mut lik_vals = Vec::with_capacity(num_update);
            let mut lik1_vals = Vec::with_capacity(num_update);

            for i in (k + 2)..n {
                let pi = perm[i];
                let aik = work[pi][pk].clone();
                let aik1 = work[pi][pk1].clone();

                let lik = aik.clone() * inv_d11.clone() + aik1.clone() * inv_d21.clone();
                let lik1 = aik.clone() * inv_d21.clone() + aik1.clone() * inv_d22.clone();

                l_data[i][k] = lik.clone();
                l_data[i][k + 1] = lik1.clone();
                lik_vals.push(lik);
                lik1_vals.push(lik1);
            }

            // Read pivot row values from unmodified work
            let mut row_k_vals = Vec::with_capacity(num_update);
            let mut row_k1_vals = Vec::with_capacity(num_update);
            for j in (k + 2)..n {
                let pj = perm[j];
                row_k_vals.push(work[pk][pj].clone());
                row_k1_vals.push(work[pk1][pj].clone());
            }

            // Symmetric rank-2 update: only lower triangle
            for ii in 0..num_update {
                let i = k + 2 + ii;
                let pi = perm[i];

                for jj in 0..=ii {
                    let j = k + 2 + jj;
                    let pj = perm[j];
                    let update = lik_vals[ii].clone() * row_k_vals[jj].clone()
                        + lik1_vals[ii].clone() * row_k1_vals[jj].clone();
                    work[pi][pj] = work[pi][pj].clone() - update;
                    if i != j {
                        work[pj][pi] = work[pi][pj].clone();
                    }
                }
            }

            pivots.push(PivotBlock::TwoByTwo {
                idx1: k,
                idx2: k + 1,
                d11,
                d21,
                d22,
            });

            k += 2;
        }

        let mut perm_inv = vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        Ok(Self {
            l_data,
            pivots,
            perm,
            perm_inv,
            n,
        })
    }

    /// Returns the pivot blocks (D factor).
    pub fn pivots(&self) -> &[PivotBlock<T>] {
        &self.pivots
    }

    /// Returns the symmetric permutation.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of 2x2 pivot blocks used.
    pub fn num_2x2_pivots(&self) -> usize {
        self.pivots
            .iter()
            .filter(|p| matches!(p, PivotBlock::TwoByTwo { .. }))
            .count()
    }

    /// Solves A * x = b using the LDL^T factorization.
    ///
    /// Computes x = P^T * L^{-T} * D^{-1} * L^{-1} * P * b.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        if n == 0 {
            return vec![];
        }

        // Step 1: Apply permutation: y = P * b
        let mut y = vec![T::zero(); n];
        for i in 0..n {
            y[i] = b[self.perm[i]].clone();
        }

        // Step 2: Forward solve L * z = y (L is unit lower triangular)
        for k in 0..n {
            let yk = y[k].clone();
            for i in (k + 1)..n {
                let lik = self.l_data[i][k].clone();
                if Scalar::abs(lik.clone()) > <T as Scalar>::epsilon() {
                    y[i] = y[i].clone() - lik * yk.clone();
                }
            }
        }

        // Step 3: Solve D * w = z (block-diagonal solve)
        let mut w = y;
        for pivot in &self.pivots {
            match pivot {
                PivotBlock::OneByOne { idx, d } => {
                    w[*idx] = w[*idx].clone() / d.clone();
                }
                PivotBlock::TwoByTwo {
                    idx1,
                    idx2,
                    d11,
                    d21,
                    d22,
                } => {
                    let det = d11.clone() * d22.clone() - d21.clone() * d21.clone();
                    let inv_det = T::one() / det;
                    let w1 = w[*idx1].clone();
                    let w2 = w[*idx2].clone();
                    w[*idx1] =
                        (d22.clone() * w1.clone() - d21.clone() * w2.clone()) * inv_det.clone();
                    w[*idx2] = (d11.clone() * w2 - d21.clone() * w1) * inv_det;
                }
            }
        }

        // Step 4: Backward solve L^T * v = w
        for k in (0..n).rev() {
            let wk = w[k].clone();
            for i in (k + 1)..n {
                let lik = self.l_data[i][k].clone();
                if Scalar::abs(lik.clone()) > <T as Scalar>::epsilon() {
                    w[k] = w[k].clone() - lik * w[i].clone();
                }
            }
            // Note: for L^T solve, we subtract L[i,k] * w[i] from w[k]
            // The above loop already does this. But we also need to handle the
            // case where w[k] was modified by earlier iterations. Let's redo properly.
            // Actually, the backward solve for L^T x = b is:
            // x[k] = b[k] - sum_{i>k} L[i,k] * x[i]
            // Since we process k from n-1 down to 0, x[i] for i > k are already final.
            // But we modified w[k] in the loop. That's correct.
            let _ = wk; // suppress unused warning
        }

        // Step 5: Apply inverse permutation: x = P^T * v
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            x[self.perm[i]] = w[i].clone();
        }

        x
    }
}

// ---------------------------------------------------------------------------
// Helper: apply 1x1 pivot and update working matrix + L
// ---------------------------------------------------------------------------

/// Applies a 1x1 Bunch-Kaufman pivot at position `k` and updates the
/// working matrix and L factor in place.
fn self_apply_1x1_pivot<T: Scalar<Real = T> + Clone + Field + Real>(
    work: &mut [Vec<T>],
    l_data: &mut [Vec<T>],
    perm: &[usize],
    k: usize,
    n: usize,
    pivot: T,
) {
    let pk = perm[k];

    // First compute all L entries from the unmodified work matrix
    let mut lik_values = Vec::with_capacity(n - k - 1);
    for i in (k + 1)..n {
        let pi = perm[i];
        let lik = work[pi][pk].clone() / pivot.clone();
        l_data[i][k] = lik.clone();
        lik_values.push(lik);
    }

    // Read the pivot row values we need for updates (from unmodified work)
    let mut pivot_row_vals = Vec::with_capacity(n - k - 1);
    for j in (k + 1)..n {
        let pj = perm[j];
        pivot_row_vals.push(work[pk][pj].clone());
    }

    // Now apply the symmetric rank-1 update: A(i,j) -= L(i,k)*A(k,j)
    // Only update the lower triangle (i >= j) in permuted indices
    for ii in 0..(n - k - 1) {
        let i = k + 1 + ii;
        let pi = perm[i];
        let lik = &lik_values[ii];

        for jj in 0..=ii {
            let j = k + 1 + jj;
            let pj = perm[j];
            let update = lik.clone() * pivot_row_vals[jj].clone();
            work[pi][pj] = work[pi][pj].clone() - update;
            // Mirror to upper triangle for symmetric access
            if i != j {
                work[pj][pi] = work[pi][pj].clone();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dense-to-CSC conversion helpers (same logic as in lu.rs, but local)
// ---------------------------------------------------------------------------

/// Converts a dense lower triangular matrix to CSC format.
fn dense_to_csc_lower<T: Scalar + Clone + Field>(data: &[Vec<T>]) -> CscMatrix<T> {
    let n = data.len();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in j..n {
            let val = data[i][j].clone();
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() || i == j {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    // Safety: we construct valid CSC invariants (sorted row indices per column)
    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Converts a dense upper triangular matrix to CSC format.
fn dense_to_csc_upper<T: Scalar + Clone + Field>(data: &[Vec<T>]) -> CscMatrix<T> {
    let n = data.len();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        for i in 0..=j {
            let val = data[i][j].clone();
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() || i == j {
                row_indices.push(i);
                values.push(val);
            }
        }
        col_ptrs[j + 1] = values.len();
    }

    // Safety: we construct valid CSC invariants (sorted row indices per column)
    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Computes sparse LU with threshold pivoting (convenience wrapper).
///
/// See [`SparseLuThreshold::with_threshold`] for details.
pub fn compute_with_threshold<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CscMatrix<T>,
    threshold: T,
) -> Result<SparseLuThreshold<T>, SparseLUError> {
    SparseLuThreshold::with_threshold(a, threshold)
}

/// Computes sparse LU with static pivoting (convenience wrapper).
///
/// See [`SparseLuStaticPivot::new`] for details.
pub fn compute_static_pivoting<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CscMatrix<T>,
    epsilon: T,
) -> Result<SparseLuStaticPivot<T>, SparseLUError> {
    SparseLuStaticPivot::new(a, epsilon)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a well-conditioned 3x3 SPD matrix.
    fn make_test_matrix_3x3() -> CscMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];
        CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("valid 3x3 test matrix construction")
    }

    /// Creates a 5x5 tridiagonal matrix.
    fn make_test_matrix_5x5() -> CscMatrix<f64> {
        let n = 5;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0usize];

        for j in 0..n {
            if j > 0 {
                row_indices.push(j - 1);
                values.push(-1.0);
            }
            row_indices.push(j);
            values.push(4.0);
            if j < n - 1 {
                row_indices.push(j + 1);
                values.push(-1.0);
            }
            col_ptrs.push(values.len());
        }

        CscMatrix::new(n, n, col_ptrs, row_indices, values)
            .expect("valid 5x5 tridiagonal matrix construction")
    }

    /// Creates an ill-conditioned matrix with a very small diagonal element.
    fn make_ill_conditioned() -> CscMatrix<f64> {
        // A = [1e-12  1    0  ]
        //     [1      4    1  ]
        //     [0      1    4  ]
        let values = vec![1e-12, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];
        CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("valid ill-conditioned matrix construction")
    }

    /// Creates a symmetric indefinite matrix (has both positive and negative eigenvalues).
    fn make_symmetric_indefinite() -> CscMatrix<f64> {
        // A = [ 1  2  0]
        //     [ 2 -3  1]
        //     [ 0  1  2]
        // Eigenvalues are mixed sign => indefinite
        let values = vec![1.0, 2.0, 2.0, -3.0, 1.0, 1.0, 2.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];
        CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("valid symmetric indefinite matrix construction")
    }

    /// Helper: compute A * x for a CSC matrix.
    fn csc_matvec(a: &CscMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let n = a.nrows();
        let mut result = vec![0.0; n];
        for col in 0..a.ncols() {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                let row = a.row_indices()[idx];
                result[row] += a.values()[idx] * x[col];
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Test 1: Threshold pivoting correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_threshold_pivoting_correctness() {
        let a = make_test_matrix_3x3();
        let lu = SparseLuThreshold::with_threshold(&a, 0.1).expect("threshold LU should succeed");

        let b = vec![5.0, 6.0, 5.0];
        let x = lu.solve(&b);
        let ax = csc_matvec(&a, &x);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Threshold LU solve failed at index {i}: got {}, expected {}",
                ax[i],
                b[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Comparison of threshold pivoting with standard LU
    // -----------------------------------------------------------------------
    #[test]
    fn test_threshold_vs_standard_pivoting() {
        let a = make_test_matrix_5x5();
        let b = vec![1.0, -1.0, 2.0, -2.0, 1.0];

        // Standard partial pivoting (threshold = 1.0)
        let lu_full =
            SparseLuThreshold::with_threshold(&a, 1.0).expect("full pivoting LU should succeed");
        let x_full = lu_full.solve(&b);

        // Threshold pivoting (threshold = 0.1)
        let lu_thresh = SparseLuThreshold::with_threshold(&a, 0.1)
            .expect("threshold pivoting LU should succeed");
        let x_thresh = lu_thresh.solve(&b);

        // Both should give the same answer for this well-conditioned matrix
        for i in 0..5 {
            assert!(
                (x_full[i] - x_thresh[i]).abs() < 1e-10,
                "Full vs threshold pivoting differ at {i}: {} vs {}",
                x_full[i],
                x_thresh[i],
            );
        }

        // Verify both solve correctly
        let ax = csc_matvec(&a, &x_thresh);
        let residual: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(
            residual < 1e-10,
            "Threshold LU residual too large: {residual}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Ill-conditioned matrix with threshold pivoting
    // -----------------------------------------------------------------------
    #[test]
    fn test_threshold_pivoting_ill_conditioned() {
        let a = make_ill_conditioned();
        let b = vec![1.0, 6.0, 5.0];

        let lu = SparseLuThreshold::with_threshold(&a, 0.1)
            .expect("threshold LU on ill-conditioned matrix should succeed");
        let x = lu.solve(&b);

        let ax = csc_matvec(&a, &x);
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let residual: f64 = (0..3).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-6,
            "Ill-conditioned threshold LU relative residual too large: {}",
            residual / b_norm,
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Static pivoting correctness
    // -----------------------------------------------------------------------
    #[test]
    fn test_static_pivoting_correctness() {
        let a = make_test_matrix_3x3();

        let lu = SparseLuStaticPivot::new(&a, 1e-10).expect("static pivoting should succeed");

        // No pivots should be perturbed for a well-conditioned matrix
        assert_eq!(lu.num_perturbed(), 0);

        let b = vec![5.0, 6.0, 5.0];
        let x = lu.solve(&b);
        let ax = csc_matvec(&a, &x);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Static pivoting solve failed at index {i}: got {}, expected {}",
                ax[i],
                b[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Static pivoting with perturbation
    // -----------------------------------------------------------------------
    #[test]
    fn test_static_pivoting_perturbation() {
        let a = make_ill_conditioned();

        // Use a relatively large epsilon to force perturbation of the tiny diagonal
        let lu = SparseLuStaticPivot::new(&a, 1e-6).expect("static pivoting should not fail");

        // The 1e-12 diagonal element should be perturbed
        assert!(
            lu.num_perturbed() > 0,
            "Expected at least one perturbed pivot"
        );

        // Solution is approximate but should be reasonable
        let b = vec![1.0, 6.0, 5.0];
        let x = lu.solve(&b);

        // Verify solution is finite
        assert!(
            x.iter().all(|v| v.is_finite()),
            "Static pivoting produced non-finite solution"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Static pivoting 5x5
    // -----------------------------------------------------------------------
    #[test]
    fn test_static_pivoting_5x5() {
        let a = make_test_matrix_5x5();

        let lu =
            SparseLuStaticPivot::new(&a, 1e-14).expect("static pivoting on 5x5 should succeed");

        assert_eq!(lu.num_perturbed(), 0);

        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let x = lu.solve(&b);
        let ax = csc_matvec(&a, &x);

        let residual: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-10,
            "Static pivoting 5x5 relative residual: {}",
            residual / b_norm,
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Symmetric indefinite LDL^T with Bunch-Kaufman
    // -----------------------------------------------------------------------
    #[test]
    fn test_ldlt_symmetric_indefinite() {
        let a = make_symmetric_indefinite();

        let ldlt =
            SparseLdlt::new(&a).expect("LDL^T should succeed for symmetric indefinite matrix");

        let b = vec![3.0, 0.0, 3.0];
        let x = ldlt.solve(&b);
        let ax = csc_matvec(&a, &x);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8,
                "LDL^T solve failed at index {i}: got {}, expected {}",
                ax[i],
                b[i],
            );
        }

        // Should have used at least one 2x2 pivot for indefinite matrix
        // (though this depends on the specific matrix; the test matrix may
        // or may not trigger a 2x2 pivot)
        assert!(
            !ldlt.pivots().is_empty(),
            "Expected at least one pivot block"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: LDL^T on positive definite matrix (all 1x1 pivots)
    // -----------------------------------------------------------------------
    #[test]
    fn test_ldlt_positive_definite() {
        let a = make_test_matrix_3x3();

        let ldlt = SparseLdlt::new(&a).expect("LDL^T should succeed for SPD matrix");

        // SPD matrix should use only 1x1 pivots
        assert_eq!(
            ldlt.num_2x2_pivots(),
            0,
            "SPD matrix should not need 2x2 pivots"
        );

        let b = vec![5.0, 6.0, 5.0];
        let x = ldlt.solve(&b);
        let ax = csc_matvec(&a, &x);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "LDL^T SPD solve failed at index {i}: got {}, expected {}",
                ax[i],
                b[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 9: LDL^T 5x5 symmetric indefinite
    // -----------------------------------------------------------------------
    #[test]
    fn test_ldlt_5x5_indefinite() {
        // Larger symmetric indefinite matrix
        // A = diag(3, -2, 4, -1, 5) + off-diagonal connections
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0usize];

        let diag = [3.0, -2.0, 4.0, -1.0, 5.0];
        let n = 5;

        for j in 0..n {
            if j > 0 {
                row_indices.push(j - 1);
                values.push(1.0);
            }
            row_indices.push(j);
            values.push(diag[j]);
            if j < n - 1 {
                row_indices.push(j + 1);
                values.push(1.0);
            }
            col_ptrs.push(values.len());
        }

        let a = CscMatrix::new(n, n, col_ptrs, row_indices, values)
            .expect("valid 5x5 indefinite matrix construction");

        let ldlt = SparseLdlt::new(&a).expect("LDL^T should succeed for 5x5 indefinite matrix");

        let b = vec![4.0, -1.0, 5.0, 0.0, 6.0];
        let x = ldlt.solve(&b);
        let ax = csc_matvec(&a, &x);

        let residual: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-8,
            "LDL^T 5x5 indefinite relative residual: {}",
            residual / b_norm,
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: Threshold pivoting with different thresholds
    // -----------------------------------------------------------------------
    #[test]
    fn test_threshold_pivoting_various_thresholds() {
        let a = make_test_matrix_5x5();
        let b = vec![2.0, -1.0, 3.0, -2.0, 1.0];

        for &threshold in &[0.0, 0.01, 0.1, 0.5, 1.0] {
            let lu = SparseLuThreshold::with_threshold(&a, threshold)
                .unwrap_or_else(|e| panic!("threshold LU failed with threshold {threshold}: {e}"));

            let x = lu.solve(&b);
            let ax = csc_matvec(&a, &x);

            let residual: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
            assert!(
                residual < 1e-8,
                "Threshold {threshold}: residual = {residual}"
            );

            assert!(
                (*lu.threshold() - threshold).abs() < 1e-15,
                "Stored threshold does not match"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 11: Error handling - not square
    // -----------------------------------------------------------------------
    #[test]
    fn test_pivoting_not_square_errors() {
        let a = CscMatrix::<f64>::new(3, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
            .expect("non-square matrix construction should succeed");

        assert!(matches!(
            SparseLuThreshold::new(&a),
            Err(SparseLUError::NotSquare { .. })
        ));

        assert!(matches!(
            SparseLuStaticPivot::new(&a, 1e-10),
            Err(SparseLUError::NotSquare { .. })
        ));

        assert!(matches!(
            SparseLdlt::new(&a),
            Err(SparseLUError::NotSquare { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 12: Default threshold constructor
    // -----------------------------------------------------------------------
    #[test]
    fn test_threshold_default_constructor() {
        let a = make_test_matrix_3x3();
        let lu = SparseLuThreshold::new(&a).expect("default threshold LU should succeed");

        // Default threshold should be 0.1
        assert!(
            (*lu.threshold() - 0.1).abs() < 1e-10,
            "Default threshold should be 0.1, got {}",
            lu.threshold(),
        );

        let b = vec![5.0, 6.0, 5.0];
        let x = lu.solve(&b);
        let ax = csc_matvec(&a, &x);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Default threshold LU solve failed at index {i}"
            );
        }
    }
}
