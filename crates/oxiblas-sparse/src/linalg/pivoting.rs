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

        // Left-looking (Gilbert-Peierls) sparse LU with threshold partial
        // pivoting. The working storage scales with nnz(L) + nnz(U) plus an
        // O(n) sparse accumulator, never n^2. Because the row permutation is
        // built incrementally via `pinv` and L is relabelled only once at the
        // end, the resulting factorization satisfies P*A = L*U exactly for
        // every step (already-computed L rows are never left inconsistent with
        // the permutation, unlike a naive swap-the-permutation scheme).
        let factors = sparse_lu_left_looking(a, LuPivotMode::Threshold(threshold.clone()))?;

        let n = a.nrows();
        let mut perm_inv = vec![0usize; n];
        for (i, &p) in factors.perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        Ok(Self {
            l: factors.l,
            u: factors.u,
            perm: factors.perm,
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

        // Left-looking (Gilbert-Peierls) sparse LU using the natural row order
        // (no row interchanges). Any pivot with |pivot| < epsilon is replaced by
        // sign(pivot) * epsilon (SuperLU-style perturbation). Working storage
        // scales with nnz(L) + nnz(U) plus an O(n) sparse accumulator, not n^2.
        let factors = sparse_lu_left_looking(a, LuPivotMode::StaticPerturb(epsilon.clone()))?;

        Ok(Self {
            l: factors.l,
            u: factors.u,
            epsilon,
            num_perturbed: factors.num_perturbed,
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
                // Swap rows/cols k and r in permutation. `l_data` is indexed by
                // position (not physical row), so the already-computed columns
                // `0..k` of L must be swapped in lock-step with the permutation;
                // otherwise the multipliers stored for positions k and r would
                // become attached to the wrong physical rows and P*A*P^T != L*D*L^T.
                perm.swap(k, r);
                swap_l_rows(&mut l_data, k, r, k);
                let pivot = arr.clone();
                if Scalar::abs(pivot.clone()) <= <T as Scalar>::epsilon() {
                    return Err(SparseLUError::Singular { row: k });
                }
                self_apply_1x1_pivot(&mut work, &mut l_data, &perm, k, n, pivot.clone());
                pivots.push(PivotBlock::OneByOne { idx: k, d: pivot });
                k += 1;
                continue;
            }

            // Use 2x2 pivot at (k, k+1) after swapping r to position k+1.
            // As above, the already-computed columns `0..k` of the
            // position-indexed L must be swapped together with the permutation
            // so the multipliers stay attached to the correct physical rows.
            if r != k + 1 {
                perm.swap(k + 1, r);
                swap_l_rows(&mut l_data, k + 1, r, k);
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
// Symmetric-pivot L row swap (used by the Bunch-Kaufman LDL^T factorization)
// ---------------------------------------------------------------------------

/// Swaps rows `a` and `b` of the position-indexed lower factor `l_data` over the
/// already-computed columns `0..cols`.
///
/// The LDL^T factorization stores `l_data` indexed by *position* (not physical
/// row) and drives elimination through the `perm` array. When a symmetric pivot
/// swaps two positions, the multipliers already stored for those positions must
/// move with them, otherwise the position-indexed L becomes inconsistent with
/// the permutation and `P*A*P^T = L*D*L^T` no longer holds.
fn swap_l_rows<T>(l_data: &mut [Vec<T>], a: usize, b: usize, cols: usize) {
    if a == b || cols == 0 {
        return;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (left, right) = l_data.split_at_mut(hi);
    let row_lo = &mut left[lo];
    let row_hi = &mut right[0];
    for c in 0..cols {
        std::mem::swap(&mut row_lo[c], &mut row_hi[c]);
    }
}

// ---------------------------------------------------------------------------
// Left-looking (Gilbert-Peierls) sparse LU with pivoting
// ---------------------------------------------------------------------------

/// Pivot strategy for the shared left-looking sparse LU driver.
enum LuPivotMode<T> {
    /// Threshold partial pivoting with tolerance `tol` in `[0, 1]`.
    ///
    /// `tol = 1` is full partial pivoting (always the largest magnitude row);
    /// `tol = 0` keeps the natural diagonal whenever it is non-zero.
    Threshold(T),
    /// Static (SuperLU-style) pivoting: keep the natural row order and replace
    /// any pivot with `|pivot| < epsilon` by `sign(pivot) * epsilon`.
    StaticPerturb(T),
}

/// Result of a left-looking sparse LU factorization: `P*A = L*U`.
struct SparseLuFactors<T: Scalar> {
    /// Unit lower triangular factor (CSC, diagonal stored first in each column).
    l: CscMatrix<T>,
    /// Upper triangular factor (CSC).
    u: CscMatrix<T>,
    /// Row permutation `perm` such that row `k` of `P*A` is original row `perm[k]`.
    perm: Vec<usize>,
    /// Number of pivots perturbed (static pivoting only; `0` otherwise).
    num_perturbed: usize,
}

/// O(n) sparse accumulator and depth-first-search workspace for Gilbert-Peierls.
///
/// This is the only workspace whose size grows with `n`; together with the CSC
/// storage for `L` and `U` (which grows with the number of non-zeros produced)
/// the whole factorization uses `O(n + nnz(L) + nnz(U))` memory rather than the
/// `O(n^2)` dense working arrays used previously.
struct LuSpa<T> {
    /// Dense values of the current column solution (mostly zero).
    x: Vec<T>,
    /// Non-zero pattern of the current column, `xi[top..n]` in topological order.
    xi: Vec<usize>,
    /// DFS recursion stack of graph nodes (original row indices).
    rstack: Vec<usize>,
    /// DFS resume position within each stacked node's adjacency list.
    pstack: Vec<usize>,
    /// Per-node visited flag, reset after every column.
    marked: Vec<bool>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> LuSpa<T> {
    fn new(n: usize) -> Self {
        Self {
            x: vec![T::zero(); n],
            xi: vec![0usize; n],
            rstack: vec![0usize; n],
            pstack: vec![0usize; n],
            marked: vec![false; n],
        }
    }

    /// Iterative depth-first search of the graph of `L` starting at `start`.
    ///
    /// Nodes reachable from `start` are pushed (in post-order) into `xi`, filling
    /// it from index `top` downwards; the new `top` is returned.
    fn dfs(
        &mut self,
        start: usize,
        pinv: &[isize],
        l_col_ptrs: &[usize],
        l_rows: &[usize],
        mut top: usize,
    ) -> usize {
        let mut head: isize = 0;
        self.rstack[0] = start;
        while head >= 0 {
            let h = head as usize;
            let node = self.rstack[h];
            let node_col = pinv[node];
            if !self.marked[node] {
                self.marked[node] = true;
                // Skip the (self-loop) diagonal entry stored first in the column.
                self.pstack[h] = if node_col < 0 {
                    0
                } else {
                    l_col_ptrs[node_col as usize] + 1
                };
            }
            let p_end = if node_col < 0 {
                0
            } else {
                l_col_ptrs[node_col as usize + 1]
            };
            let mut done = true;
            let mut p = self.pstack[h];
            while p < p_end {
                let neighbor = l_rows[p];
                if self.marked[neighbor] {
                    p += 1;
                    continue;
                }
                // Pause this node, descend into `neighbor`.
                self.pstack[h] = p;
                head += 1;
                self.rstack[head as usize] = neighbor;
                done = false;
                break;
            }
            if done {
                top -= 1;
                self.xi[top] = node;
                head -= 1;
            }
        }
        top
    }

    /// Computes the non-zero pattern of `x = L \ A(:,col)` via reachability.
    ///
    /// Returns `top`; the pattern is `xi[top..n]` in topological order.
    fn reach(
        &mut self,
        a: &CscMatrix<T>,
        col: usize,
        pinv: &[isize],
        l_col_ptrs: &[usize],
        l_rows: &[usize],
    ) -> usize {
        let n = self.x.len();
        let mut top = n;
        let start = a.col_ptrs()[col];
        let end = a.col_ptrs()[col + 1];
        for idx in start..end {
            let row = a.row_indices()[idx];
            if !self.marked[row] {
                top = self.dfs(row, pinv, l_col_ptrs, l_rows, top);
            }
        }
        // Restore the visited flags for the next column.
        for p in top..n {
            let node = self.xi[p];
            self.marked[node] = false;
        }
        top
    }

    /// Sparse triangular solve `x = L \ A(:,col)`.
    ///
    /// On entry `x` must be identically zero (the driver restores this after each
    /// column). On return `xi[top..n]` holds the pattern and `x[i]` the values.
    fn spsolve(
        &mut self,
        a: &CscMatrix<T>,
        col: usize,
        pinv: &[isize],
        l_col_ptrs: &[usize],
        l_rows: &[usize],
        l_vals: &[T],
    ) -> usize {
        let n = self.x.len();
        let top = self.reach(a, col, pinv, l_col_ptrs, l_rows);

        // Scatter A(:,col) into the (all-zero) accumulator.
        let start = a.col_ptrs()[col];
        let end = a.col_ptrs()[col + 1];
        for idx in start..end {
            let row = a.row_indices()[idx];
            self.x[row] = a.values()[idx].clone();
        }

        // Forward substitution in topological order.
        for p_idx in top..n {
            let node = self.xi[p_idx];
            let node_col = pinv[node];
            if node_col < 0 {
                // Column of L not yet formed: `node` is a pivot candidate, its
                // value stays as-is (L has unit diagonal, so no self-division).
                continue;
            }
            let jj = node_col as usize;
            let xj = self.x[node].clone();
            for p in (l_col_ptrs[jj] + 1)..l_col_ptrs[jj + 1] {
                let i = l_rows[p];
                self.x[i] = self.x[i].clone() - l_vals[p].clone() * xj.clone();
            }
        }

        top
    }
}

/// Sorts every column of a CSC triple in place by ascending row index.
///
/// This both satisfies the crate's CSC invariant (sorted row indices per column)
/// and places the unit diagonal of `L` first / the diagonal of `U` last, matching
/// what the sparse triangular solvers expect.
fn sort_csc_columns<T: Clone>(col_ptrs: &[usize], rows: &mut [usize], vals: &mut [T]) {
    let ncols = col_ptrs.len().saturating_sub(1);
    for j in 0..ncols {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        if end - start <= 1 {
            continue;
        }
        let mut order: Vec<usize> = (start..end).collect();
        order.sort_unstable_by_key(|&p| rows[p]);
        let sorted_rows: Vec<usize> = order.iter().map(|&p| rows[p]).collect();
        let sorted_vals: Vec<T> = order.iter().map(|&p| vals[p].clone()).collect();
        for (offset, r) in sorted_rows.into_iter().enumerate() {
            rows[start + offset] = r;
        }
        for (offset, v) in sorted_vals.into_iter().enumerate() {
            vals[start + offset] = v;
        }
    }
}

/// Left-looking (Gilbert-Peierls) sparse LU factorization `P*A = L*U`.
///
/// Each column `k` is computed by a sparse triangular solve against the columns
/// of `L` already produced (`x = L \ A(:,k)`), followed by pivot selection and
/// storage of `L(:,k)` and `U(:,k)`. The row permutation is built incrementally
/// in `pinv` and `L`'s row indices are relabelled to permuted space exactly once
/// at the end, so `P*A = L*U` holds for every intermediate step (there is no
/// stale, half-permuted `L` as in a naive swap-the-permutation implementation).
///
/// Memory and work scale with `nnz(L) + nnz(U)` plus an `O(n)` accumulator.
fn sparse_lu_left_looking<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CscMatrix<T>,
    mode: LuPivotMode<T>,
) -> Result<SparseLuFactors<T>, SparseLUError> {
    let n = a.nrows();
    if n == 0 {
        // Empty factorization: 0x0 L and U.
        let l = unsafe { CscMatrix::new_unchecked(0, 0, vec![0usize], Vec::new(), Vec::new()) };
        let u = unsafe { CscMatrix::new_unchecked(0, 0, vec![0usize], Vec::new(), Vec::new()) };
        return Ok(SparseLuFactors {
            l,
            u,
            perm: Vec::new(),
            num_perturbed: 0,
        });
    }

    let eps = <T as Scalar>::epsilon();

    // pinv[orig] = k means original row `orig` is the k-th pivot (position k);
    // `-1` marks a row that is not yet pivotal.
    let mut pinv = vec![-1isize; n];
    let mut perm = vec![0usize; n];
    let mut spa = LuSpa::<T>::new(n);
    let mut num_perturbed = 0usize;

    let mut l_col_ptrs = vec![0usize; n + 1];
    let mut l_rows: Vec<usize> = Vec::new();
    let mut l_vals: Vec<T> = Vec::new();
    let mut u_col_ptrs = vec![0usize; n + 1];
    let mut u_rows: Vec<usize> = Vec::new();
    let mut u_vals: Vec<T> = Vec::new();

    for k in 0..n {
        l_col_ptrs[k] = l_rows.len();
        u_col_ptrs[k] = u_rows.len();

        // No column permutation: the diagonal candidate for column k is row k.
        let col = k;
        let top = spa.spsolve(a, col, &pinv, &l_col_ptrs, &l_rows, &l_vals);

        // --- Pivot selection --------------------------------------------------
        let ipiv: usize;
        let pivot: T;
        match &mode {
            LuPivotMode::Threshold(tol) => {
                let mut amax = T::zero();
                let mut best: isize = -1;
                for p_idx in top..n {
                    let i = spa.xi[p_idx];
                    if pinv[i] < 0 {
                        let t = Scalar::abs(spa.x[i].clone());
                        if t > amax {
                            amax = t;
                            best = i as isize;
                        }
                    }
                }
                if best < 0 || amax <= eps {
                    return Err(SparseLUError::Singular { row: k });
                }
                let mut chosen = best as usize;
                // Threshold rule: keep the natural diagonal row if it is still
                // available and large enough relative to the column maximum.
                if pinv[col] < 0 {
                    let dval = Scalar::abs(spa.x[col].clone());
                    if dval > eps && dval >= amax * tol.clone() {
                        chosen = col;
                    }
                }
                ipiv = chosen;
                pivot = spa.x[ipiv].clone();
                if Scalar::abs(pivot.clone()) <= eps {
                    return Err(SparseLUError::Singular { row: k });
                }
            }
            LuPivotMode::StaticPerturb(epsilon) => {
                let mut pv = spa.x[col].clone();
                if Scalar::abs(pv.clone()) < epsilon.clone() {
                    num_perturbed += 1;
                    pv = if pv >= T::zero() {
                        epsilon.clone()
                    } else {
                        T::zero() - epsilon.clone()
                    };
                    spa.x[col] = pv.clone();
                }
                ipiv = col;
                pivot = pv;
            }
        }

        // --- Store U(:,k): already-pivotal entries, then the diagonal ---------
        for p_idx in top..n {
            let i = spa.xi[p_idx];
            let pos = pinv[i];
            if pos >= 0 {
                u_rows.push(pos as usize);
                u_vals.push(spa.x[i].clone());
            }
        }
        u_rows.push(k);
        u_vals.push(pivot.clone());

        // --- Store L(:,k): unit diagonal, then scaled sub-diagonal ------------
        l_rows.push(ipiv); // original row index; relabelled after the loop
        l_vals.push(T::one());
        for p_idx in top..n {
            let i = spa.xi[p_idx];
            if pinv[i] < 0 && i != ipiv {
                let scaled = spa.x[i].clone() / pivot.clone();
                l_rows.push(i);
                l_vals.push(scaled);
            }
        }

        // Register the pivot and clear the accumulator for the next column.
        pinv[ipiv] = k as isize;
        perm[k] = ipiv;
        for p_idx in top..n {
            let i = spa.xi[p_idx];
            spa.x[i] = T::zero();
        }
    }

    l_col_ptrs[n] = l_rows.len();
    u_col_ptrs[n] = u_rows.len();

    // Relabel L's row indices from original rows to permuted (position) space,
    // making L unit lower triangular with respect to P*A.
    for r in l_rows.iter_mut() {
        *r = pinv[*r] as usize;
    }

    sort_csc_columns(&l_col_ptrs, &mut l_rows, &mut l_vals);
    sort_csc_columns(&u_col_ptrs, &mut u_rows, &mut u_vals);

    // Safety: columns are sorted by row index with no duplicates, and both
    // factors are square n x n, satisfying the CSC invariants.
    let l = unsafe { CscMatrix::new_unchecked(n, n, l_col_ptrs, l_rows, l_vals) };
    let u = unsafe { CscMatrix::new_unchecked(n, n, u_col_ptrs, u_rows, u_vals) };

    Ok(SparseLuFactors {
        l,
        u,
        perm,
        num_perturbed,
    })
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

    // -----------------------------------------------------------------------
    // Reconstruction helpers and randomized factorization identity checks
    // -----------------------------------------------------------------------

    /// Expands a CSC matrix to a dense row-major matrix.
    fn csc_to_dense(a: &CscMatrix<f64>) -> Vec<Vec<f64>> {
        let (nr, nc) = a.shape();
        let mut d = vec![vec![0.0f64; nc]; nr];
        for col in 0..nc {
            for idx in a.col_ptrs()[col]..a.col_ptrs()[col + 1] {
                d[a.row_indices()[idx]][col] = a.values()[idx];
            }
        }
        d
    }

    /// Builds a CSC matrix from a dense (row-major) square matrix.
    fn dense_full_to_csc(m: &[Vec<f64>]) -> CscMatrix<f64> {
        let n = m.len();
        let mut col_ptrs = vec![0usize; n + 1];
        let mut rows = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            for (i, row) in m.iter().enumerate() {
                if row[j] != 0.0 {
                    rows.push(i);
                    vals.push(row[j]);
                }
            }
            col_ptrs[j + 1] = rows.len();
        }
        CscMatrix::new(n, n, col_ptrs, rows, vals).expect("valid csc from dense")
    }

    /// Dense square matrix product `a * b`.
    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut r = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i][k];
                if aik != 0.0 {
                    for j in 0..n {
                        r[i][j] += aik * b[k][j];
                    }
                }
            }
        }
        r
    }

    /// Small deterministic LCG returning values in `[-1, 1)`.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / ((1u64 << 31) as f64) * 2.0 - 1.0
    }

    fn max_abs(m: &[Vec<f64>]) -> f64 {
        m.iter()
            .flat_map(|row| row.iter())
            .fold(0.0f64, |acc, &v| acc.max(v.abs()))
    }

    // -----------------------------------------------------------------------
    // Test 13: threshold sparse LU actually satisfies P*A == L*U
    // -----------------------------------------------------------------------
    //
    // This is the definitive check for the "already-computed L rows are not
    // swapped when the permutation changes" bug: if L and the permutation ever
    // disagreed, the reconstructed L*U would not equal P*A. Randomized,
    // non-diagonally-dominant draws force interchanges at steps k > 0.
    #[test]
    fn test_threshold_reconstruction_pa_eq_lu() {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut factored = 0usize;
        for trial in 0..300usize {
            let n = 3 + trial % 5; // sizes 3..=7
            let mut m = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in 0..n {
                    if lcg(&mut state) > -0.2 {
                        m[i][j] = lcg(&mut state) * 5.0;
                    }
                }
                if m[i][i].abs() < 0.25 {
                    m[i][i] = 0.5; // keep a (non-dominant) diagonal present
                }
            }
            let a = dense_full_to_csc(&m);
            let ad = csc_to_dense(&a);
            let scale = 1.0 + max_abs(&ad);

            for &tol in &[1.0f64, 0.5, 0.1, 0.0] {
                let lu = match SparseLuThreshold::with_threshold(&a, tol) {
                    Ok(f) => f,
                    Err(_) => continue, // structurally singular draw
                };
                factored += 1;

                let l = csc_to_dense(lu.l());
                let u = csc_to_dense(lu.u());
                let prod = mat_mul(&l, &u);
                let perm = lu.perm();
                for i in 0..n {
                    for j in 0..n {
                        let pa = ad[perm[i]][j];
                        assert!(
                            (pa - prod[i][j]).abs() < 1e-9 * scale,
                            "trial {trial} tol {tol}: (P A)[{i}][{j}]={pa} != (L U)={}",
                            prod[i][j],
                        );
                    }
                }

                // And the solve is accurate for a known solution.
                let x_true: Vec<f64> = (0..n).map(|i| i as f64 - 1.5).collect();
                let b = csc_matvec(&a, &x_true);
                let x = lu.solve(&b);
                let ax = csc_matvec(&a, &x);
                let resid: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
                assert!(
                    resid < 1e-6 * scale,
                    "trial {trial} tol {tol}: solve residual {resid}",
                );
            }
        }
        assert!(factored > 50, "too few successful factorizations: {factored}");
    }

    // -----------------------------------------------------------------------
    // Test 14: Bunch-Kaufman LDL^T satisfies P*A*P^T == L*D*L^T
    // -----------------------------------------------------------------------
    //
    // Guards the symmetric-pivot L-row-swap fix: interchange (1x1) and 2x2
    // pivots at steps k > 0 permute already-computed L columns, and the
    // reconstruction would fail if those rows were not moved with the
    // permutation. Tiny/indefinite diagonals force both pivot kinds.
    #[test]
    fn test_ldlt_reconstruction_papt_eq_ldlt() {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut factored = 0usize;
        let mut with_2x2 = 0usize;
        for trial in 0..500usize {
            let n = 4 + trial % 4; // sizes 4..=7
            let mut m = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in i..n {
                    if lcg(&mut state) > -0.1 {
                        let v = lcg(&mut state) * 4.0;
                        m[i][j] = v;
                        m[j][i] = v;
                    }
                }
            }
            // Bias odd diagonals small/indefinite to force interchanges & 2x2 pivots.
            for i in 0..n {
                if i % 2 == 1 {
                    m[i][i] = lcg(&mut state) * 0.4;
                }
            }

            let a = dense_full_to_csc(&m);
            let ldlt = match SparseLdlt::new(&a) {
                Ok(f) => f,
                Err(_) => continue, // exactly singular draw
            };
            factored += 1;
            if ldlt.num_2x2_pivots() > 0 {
                with_2x2 += 1;
            }

            let l = &ldlt.l_data;
            let mut dmat = vec![vec![0.0f64; n]; n];
            for p in ldlt.pivots() {
                match p {
                    PivotBlock::OneByOne { idx, d } => dmat[*idx][*idx] = *d,
                    PivotBlock::TwoByTwo {
                        idx1,
                        idx2,
                        d11,
                        d21,
                        d22,
                    } => {
                        dmat[*idx1][*idx1] = *d11;
                        dmat[*idx2][*idx2] = *d22;
                        dmat[*idx1][*idx2] = *d21;
                        dmat[*idx2][*idx1] = *d21;
                    }
                }
            }

            // L * D * L^T
            let ld = mat_mul(l, &dmat);
            let mut lt = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in 0..n {
                    lt[j][i] = l[i][j];
                }
            }
            let prod = mat_mul(&ld, &lt);

            let ad = csc_to_dense(&a);
            let perm = ldlt.perm();
            let scale = 1.0 + max_abs(&ad);
            for i in 0..n {
                for j in 0..n {
                    let pap = ad[perm[i]][perm[j]];
                    assert!(
                        (pap - prod[i][j]).abs() < 1e-6 * scale,
                        "trial {trial}: (P A P^T)[{i}][{j}]={pap} != (L D L^T)={}",
                        prod[i][j],
                    );
                }
            }
        }
        assert!(factored > 50, "too few successful factorizations: {factored}");
        assert!(with_2x2 > 0, "expected some 2x2 pivot blocks to be exercised");
    }
}
