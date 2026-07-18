//! Supernodal sparse factorization methods.
//!
//! Supernodal methods exploit the fact that many sparse matrices have groups of
//! columns with similar or identical sparsity patterns. These groups form "supernodes"
//! that can be processed using dense BLAS-3 operations (GEMM, TRSM), providing
//! significant performance improvements over column-by-column methods.
//!
//! This module provides:
//! - `SupernodalCholesky`: Supernodal Cholesky factorization for SPD matrices
//! - `SupernodalLU`: Supernodal LU factorization for general matrices

use crate::csc::CscMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for supernodal factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupernodalError {
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix is not positive definite.
    NotPositiveDefinite {
        /// Index where failure occurred.
        index: usize,
    },
    /// Matrix is singular.
    Singular {
        /// Index where singularity was detected.
        index: usize,
    },
    /// Zero pivot encountered.
    ZeroPivot {
        /// Index of zero pivot.
        index: usize,
    },
}

impl core::fmt::Display for SupernodalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::NotPositiveDefinite { index } => {
                write!(f, "Matrix is not positive definite at index {index}")
            }
            Self::Singular { index } => {
                write!(f, "Matrix is singular at index {index}")
            }
            Self::ZeroPivot { index } => {
                write!(f, "Zero pivot at index {index}")
            }
        }
    }
}

impl std::error::Error for SupernodalError {}

/// A supernode is a group of consecutive columns with similar sparsity patterns.
#[derive(Debug, Clone)]
pub struct Supernode {
    /// First column in the supernode.
    pub first_col: usize,
    /// Number of columns in the supernode.
    pub size: usize,
    /// Row indices below the diagonal block (sorted).
    pub sub_rows: Vec<usize>,
}

impl Supernode {
    /// Returns the last column index (inclusive).
    pub fn last_col(&self) -> usize {
        self.first_col + self.size - 1
    }

    /// Returns the range of columns.
    pub fn cols(&self) -> core::ops::Range<usize> {
        self.first_col..(self.first_col + self.size)
    }
}

/// Supernodal Cholesky factorization for symmetric positive definite matrices.
///
/// Uses BLAS-3 operations on dense diagonal blocks for improved performance.
/// The algorithm:
/// 1. Symbolic analysis to detect supernodes
/// 2. Numeric factorization using panel-panel updates
///
/// # Example
///
/// ```ignore
/// let a = make_spd_matrix();
/// let chol = SupernodalCholesky::new(&a)?;
/// let x = chol.solve(&b);
/// ```
#[derive(Debug, Clone)]
pub struct SupernodalCholesky<T: Scalar> {
    /// Size of the matrix.
    n: usize,
    /// Supernodes (groups of columns).
    supernodes: Vec<Supernode>,
    /// Dense storage for diagonal blocks (concatenated).
    /// Each supernode k stores a (size_k x size_k) lower triangular block.
    diag_blocks: Vec<T>,
    /// Offsets into diag_blocks for each supernode.
    diag_offsets: Vec<usize>,
    /// Dense storage for off-diagonal blocks.
    /// Each supernode k stores a (|sub_rows_k| x size_k) dense block.
    offdiag_blocks: Vec<T>,
    /// Offsets into offdiag_blocks for each supernode.
    offdiag_offsets: Vec<usize>,
    /// Fill-reducing permutation (column ordering).
    perm: Vec<usize>,
    /// Inverse permutation.
    #[allow(dead_code)]
    perm_inv: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SupernodalCholesky<T> {
    /// Computes the supernodal Cholesky factorization.
    ///
    /// Uses AMD ordering by default for fill reduction.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SupernodalError> {
        if a.nrows() != a.ncols() {
            return Err(SupernodalError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            return Ok(Self {
                n: 0,
                supernodes: vec![],
                diag_blocks: vec![],
                diag_offsets: vec![0],
                offdiag_blocks: vec![],
                offdiag_offsets: vec![0],
                perm: vec![],
                perm_inv: vec![],
            });
        }

        // Compute fill-reducing ordering (AMD)
        let perm = super::ordering::approximate_minimum_degree(a);
        let mut perm_inv = vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        // Permute the matrix
        let ap = permute_symmetric(a, &perm, &perm_inv);

        // Symbolic factorization: compute elimination tree and detect supernodes
        let (supernodes, etree) = Self::symbolic_analysis(&ap);

        // Allocate storage
        let (diag_blocks, diag_offsets, offdiag_blocks, offdiag_offsets) =
            Self::allocate_storage(&supernodes);

        let mut result = Self {
            n,
            supernodes,
            diag_blocks,
            diag_offsets,
            offdiag_blocks,
            offdiag_offsets,
            perm,
            perm_inv,
        };

        // Numeric factorization
        result.numeric_factorization(&ap, &etree)?;

        Ok(result)
    }

    /// Symbolic analysis: build the elimination tree and detect supernodes.
    ///
    /// The input `a` is the fill-reducing-permuted matrix stored with its lower
    /// triangle only. To size each supernode correctly the analysis must account
    /// for fill-in: the sub-diagonal structure of `L` is generally a strict
    /// superset of the pattern of `A`. This routine therefore
    ///
    /// 1. recovers the full symmetric adjacency pattern of `a`,
    /// 2. builds a correct elimination tree from that pattern,
    /// 3. computes the fill-aware column structures of `L` via the standard
    ///    symbolic-Cholesky recursion over the elimination tree, and
    /// 4. amalgamates consecutive columns whose structures nest into supernodes,
    ///    taking each supernode's `sub_rows` from the fill-aware structures.
    ///
    /// This mirrors the (correct) approach in [`super::multifrontal_cholesky`],
    /// so that fill-generating matrices produce the right factor.
    fn symbolic_analysis(a: &CscMatrix<T>) -> (Vec<Supernode>, Vec<isize>) {
        let n = a.nrows();

        // Recover the full symmetric adjacency pattern (both triangles).
        let (sym_col_ptrs, sym_row_indices) = full_symmetric_pattern(a);

        // Elimination tree (parent pointers; -1 marks a root).
        let etree = elimination_tree(&sym_col_ptrs, &sym_row_indices, n);

        // Fill-aware column structures of L (sorted rows strictly below diagonal).
        let l_struct = column_structures(&sym_col_ptrs, &sym_row_indices, &etree, n);

        // Detect fundamental supernodes: maximal chains of consecutive columns
        // whose fill-aware structures nest, i.e. struct(L[:,j]) == struct(L[:,j-1])
        // with the entry `j` (the shared parent) removed.
        let mut supernodes = Vec::new();
        let mut j = 0;

        while j < n {
            let start = j;
            j += 1;

            while j < n
                && etree[j - 1] == j as isize
                && structures_nest(&l_struct[j - 1], &l_struct[j], j)
            {
                j += 1;
            }

            let end = j;

            // The off-diagonal rows of the supernode are the union of the
            // fill-aware structures of every column it spans, restricted to rows
            // below the (dense) diagonal block.
            let mut sub_set = std::collections::BTreeSet::new();
            for col in start..end {
                for &row in &l_struct[col] {
                    if row >= end {
                        sub_set.insert(row);
                    }
                }
            }
            let sub_rows: Vec<usize> = sub_set.into_iter().collect();

            supernodes.push(Supernode {
                first_col: start,
                size: end - start,
                sub_rows,
            });
        }

        (supernodes, etree)
    }

    /// Allocate storage for diagonal and off-diagonal blocks.
    fn allocate_storage(supernodes: &[Supernode]) -> (Vec<T>, Vec<usize>, Vec<T>, Vec<usize>) {
        let mut diag_size = 0usize;
        let mut offdiag_size = 0usize;
        let mut diag_offsets = vec![0usize];
        let mut offdiag_offsets = vec![0usize];

        for sn in supernodes {
            // Diagonal block: size x size (lower triangular stored as full for simplicity)
            diag_size += sn.size * sn.size;
            diag_offsets.push(diag_size);

            // Off-diagonal block: |sub_rows| x size
            offdiag_size += sn.sub_rows.len() * sn.size;
            offdiag_offsets.push(offdiag_size);
        }

        let diag_blocks = vec![T::zero(); diag_size];
        let offdiag_blocks = vec![T::zero(); offdiag_size];

        (diag_blocks, diag_offsets, offdiag_blocks, offdiag_offsets)
    }

    /// Numeric Cholesky factorization using supernodal updates.
    fn numeric_factorization(
        &mut self,
        a: &CscMatrix<T>,
        _etree: &[isize],
    ) -> Result<(), SupernodalError> {
        let num_sn = self.supernodes.len();

        // Process each supernode
        for sn_idx in 0..num_sn {
            // Initialize diagonal and off-diagonal blocks from A
            self.load_supernode_from_a(sn_idx, a);

            // Apply updates from ancestor supernodes (those that contribute to this one)
            for ancestor_idx in 0..sn_idx {
                self.apply_supernode_update(ancestor_idx, sn_idx);
            }

            // Factor the diagonal block: Cholesky on dense submatrix
            self.factor_diagonal_block(sn_idx)?;

            // Solve for off-diagonal block: L21 = A21 * L11^{-T}
            self.solve_offdiag_block(sn_idx);
        }

        Ok(())
    }

    /// Load values from A into supernode's blocks.
    fn load_supernode_from_a(&mut self, sn_idx: usize, a: &CscMatrix<T>) {
        let sn = &self.supernodes[sn_idx];
        let diag_offset = self.diag_offsets[sn_idx];
        let offdiag_offset = self.offdiag_offsets[sn_idx];
        let size = sn.size;
        let first = sn.first_col;

        // Build row-to-index map for sub_rows
        let mut row_to_idx: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (idx, &row) in sn.sub_rows.iter().enumerate() {
            row_to_idx.insert(row, idx);
        }

        for local_j in 0..size {
            let j = first + local_j;
            let col_start = a.col_ptrs()[j];
            let col_end = a.col_ptrs()[j + 1];

            for idx in col_start..col_end {
                let i = a.row_indices()[idx];
                let val = a.values()[idx].clone();

                if i >= first && i < first + size {
                    // Diagonal block: row i, col local_j
                    let local_i = i - first;
                    // Store in column-major order for the dense block
                    self.diag_blocks[diag_offset + local_j * size + local_i] = val;
                } else if i >= first + size {
                    // Off-diagonal block
                    if let Some(&sub_idx) = row_to_idx.get(&i) {
                        self.offdiag_blocks
                            [offdiag_offset + local_j * sn.sub_rows.len() + sub_idx] = val;
                    }
                }
            }
        }
    }

    /// Apply updates from ancestor supernode to current supernode.
    fn apply_supernode_update(&mut self, ancestor_idx: usize, target_idx: usize) {
        let ancestor = &self.supernodes[ancestor_idx];
        let target = &self.supernodes[target_idx];

        // Find rows in ancestor's off-diagonal that belong to target's columns
        let target_first = target.first_col;
        let target_last = target.first_col + target.size;

        // Rows that update the target's diagonal
        let mut update_rows_diag: Vec<(usize, usize)> = Vec::new(); // (ancestor_sub_idx, target_local_col)
        // Rows that update the target's off-diagonal
        let mut update_rows_offdiag: Vec<(usize, usize)> = Vec::new(); // (ancestor_sub_idx, target_sub_idx)

        for (sub_idx, &row) in ancestor.sub_rows.iter().enumerate() {
            if row >= target_first && row < target_last {
                update_rows_diag.push((sub_idx, row - target_first));
            }
            // Check if row is in target's sub_rows
            if let Some(target_sub_idx) = target.sub_rows.iter().position(|&r| r == row) {
                update_rows_offdiag.push((sub_idx, target_sub_idx));
            }
        }

        if update_rows_diag.is_empty() && update_rows_offdiag.is_empty() {
            return;
        }

        // Get ancestor's off-diagonal block
        let anc_offdiag_offset = self.offdiag_offsets[ancestor_idx];
        let anc_size = ancestor.size;
        let anc_sub_len = ancestor.sub_rows.len();

        // Apply rank-k update to target's diagonal block
        // This is a SYRK-like operation: L11 -= L21 * L21^T
        if !update_rows_diag.is_empty() {
            let target_diag_offset = self.diag_offsets[target_idx];
            let target_size = target.size;

            for &(anc_i, tgt_col_i) in &update_rows_diag {
                for &(anc_j, tgt_col_j) in &update_rows_diag {
                    if tgt_col_i >= tgt_col_j {
                        // Only lower triangle
                        let mut sum = T::zero();
                        for k in 0..anc_size {
                            let val_i = self.offdiag_blocks
                                [anc_offdiag_offset + k * anc_sub_len + anc_i]
                                .clone();
                            let val_j = self.offdiag_blocks
                                [anc_offdiag_offset + k * anc_sub_len + anc_j]
                                .clone();
                            sum = sum + val_i * val_j;
                        }
                        self.diag_blocks
                            [target_diag_offset + tgt_col_j * target_size + tgt_col_i] = self
                            .diag_blocks[target_diag_offset + tgt_col_j * target_size + tgt_col_i]
                            .clone()
                            - sum;
                    }
                }
            }
        }

        // Apply GEMM update to target's off-diagonal block
        // L31 -= L32 * L21^T (where 3 is target's sub, 2 is ancestor's cols in target, 1 is ancestor)
        if !update_rows_offdiag.is_empty() && !update_rows_diag.is_empty() {
            let target_offdiag_offset = self.offdiag_offsets[target_idx];
            let _target_size = target.size;
            let target_sub_len = target.sub_rows.len();

            for &(anc_off_i, tgt_sub_i) in &update_rows_offdiag {
                for &(anc_diag_j, tgt_col_j) in &update_rows_diag {
                    let mut sum = T::zero();
                    for k in 0..anc_size {
                        let val_i = self.offdiag_blocks
                            [anc_offdiag_offset + k * anc_sub_len + anc_off_i]
                            .clone();
                        let val_j = self.offdiag_blocks
                            [anc_offdiag_offset + k * anc_sub_len + anc_diag_j]
                            .clone();
                        sum = sum + val_i * val_j;
                    }
                    self.offdiag_blocks
                        [target_offdiag_offset + tgt_col_j * target_sub_len + tgt_sub_i] = self
                        .offdiag_blocks
                        [target_offdiag_offset + tgt_col_j * target_sub_len + tgt_sub_i]
                        .clone()
                        - sum;
                }
            }
        }
    }

    /// Factor the diagonal block using dense Cholesky.
    fn factor_diagonal_block(&mut self, sn_idx: usize) -> Result<(), SupernodalError> {
        let sn = &self.supernodes[sn_idx];
        let size = sn.size;
        let offset = self.diag_offsets[sn_idx];

        // Dense Cholesky on the diagonal block (column-major storage)
        for j in 0..size {
            // Compute L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
            let mut diag = self.diag_blocks[offset + j * size + j].clone();
            for k in 0..j {
                let ljk = self.diag_blocks[offset + k * size + j].clone();
                diag = diag - ljk.clone() * ljk;
            }

            if !(diag > T::zero()) {
                return Err(SupernodalError::NotPositiveDefinite {
                    index: sn.first_col + j,
                });
            }

            let ljj = Real::sqrt(diag);
            self.diag_blocks[offset + j * size + j] = ljj.clone();

            // Compute L[i,j] for i > j
            if Scalar::abs(ljj.clone()) <= <T as Scalar>::epsilon() {
                return Err(SupernodalError::ZeroPivot {
                    index: sn.first_col + j,
                });
            }

            let ljj_inv = T::one() / ljj;

            for i in (j + 1)..size {
                let mut val = self.diag_blocks[offset + j * size + i].clone();
                for k in 0..j {
                    let lik = self.diag_blocks[offset + k * size + i].clone();
                    let ljk = self.diag_blocks[offset + k * size + j].clone();
                    val = val - lik * ljk;
                }
                self.diag_blocks[offset + j * size + i] = val * ljj_inv.clone();
            }
        }

        Ok(())
    }

    /// Solve for off-diagonal block: L21 = A21 * L11^{-T}
    fn solve_offdiag_block(&mut self, sn_idx: usize) {
        let sn = &self.supernodes[sn_idx];
        let size = sn.size;
        let sub_len = sn.sub_rows.len();

        if sub_len == 0 {
            return;
        }

        let diag_offset = self.diag_offsets[sn_idx];
        let offdiag_offset = self.offdiag_offsets[sn_idx];

        // Forward substitution for each column of L21
        // L21[i, j] = (A21[i, j] - sum_{k<j} L21[i, k] * L11[j, k]) / L11[j, j]
        for j in 0..size {
            let ljj = self.diag_blocks[diag_offset + j * size + j].clone();
            let ljj_inv = T::one() / ljj;

            for i in 0..sub_len {
                let mut val = self.offdiag_blocks[offdiag_offset + j * sub_len + i].clone();
                for k in 0..j {
                    let l21_ik = self.offdiag_blocks[offdiag_offset + k * sub_len + i].clone();
                    let l11_jk = self.diag_blocks[diag_offset + k * size + j].clone();
                    val = val - l21_ik * l11_jk;
                }
                self.offdiag_blocks[offdiag_offset + j * sub_len + i] = val * ljj_inv.clone();
            }
        }
    }

    /// Solves A * x = b.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Apply permutation: b_perm = P * b
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            x[i] = b[self.perm[i]].clone();
        }

        // Forward solve: L * y = b_perm
        x = self.forward_solve(&x);

        // Backward solve: L^T * z = y
        x = self.backward_solve(&x);

        // Apply inverse permutation: result = P^T * z
        let mut result = vec![T::zero(); n];
        for i in 0..n {
            result[self.perm[i]] = x[i].clone();
        }

        result
    }

    /// Forward substitution with supernodal L.
    fn forward_solve(&self, b: &[T]) -> Vec<T> {
        let mut x = b.to_vec();

        for sn_idx in 0..self.supernodes.len() {
            let sn = &self.supernodes[sn_idx];
            let first = sn.first_col;
            let size = sn.size;
            let diag_offset = self.diag_offsets[sn_idx];
            let offdiag_offset = self.offdiag_offsets[sn_idx];
            let sub_len = sn.sub_rows.len();

            // Solve diagonal block: L11 * y1 = b1
            for j in 0..size {
                for k in 0..j {
                    let ljk = self.diag_blocks[diag_offset + k * size + j].clone();
                    x[first + j] = x[first + j].clone() - ljk * x[first + k].clone();
                }
                let ljj = self.diag_blocks[diag_offset + j * size + j].clone();
                x[first + j] = x[first + j].clone() / ljj;
            }

            // Update sub-diagonal: x2 -= L21 * y1
            for i in 0..sub_len {
                let row = sn.sub_rows[i];
                for j in 0..size {
                    let l21_ij = self.offdiag_blocks[offdiag_offset + j * sub_len + i].clone();
                    x[row] = x[row].clone() - l21_ij * x[first + j].clone();
                }
            }
        }

        x
    }

    /// Backward substitution with supernodal L^T.
    fn backward_solve(&self, b: &[T]) -> Vec<T> {
        let mut x = b.to_vec();

        for sn_idx in (0..self.supernodes.len()).rev() {
            let sn = &self.supernodes[sn_idx];
            let first = sn.first_col;
            let size = sn.size;
            let diag_offset = self.diag_offsets[sn_idx];
            let offdiag_offset = self.offdiag_offsets[sn_idx];
            let sub_len = sn.sub_rows.len();

            // Update from sub-diagonal: y1 -= L21^T * x2
            for j in 0..size {
                for i in 0..sub_len {
                    let row = sn.sub_rows[i];
                    let l21_ij = self.offdiag_blocks[offdiag_offset + j * sub_len + i].clone();
                    x[first + j] = x[first + j].clone() - l21_ij * x[row].clone();
                }
            }

            // Solve diagonal block: L11^T * x1 = y1
            for j in (0..size).rev() {
                let ljj = self.diag_blocks[diag_offset + j * size + j].clone();
                x[first + j] = x[first + j].clone() / ljj;
                for k in 0..j {
                    let ljk = self.diag_blocks[diag_offset + k * size + j].clone();
                    x[first + k] = x[first + k].clone() - ljk * x[first + j].clone();
                }
            }
        }

        x
    }

    /// Returns the number of supernodes.
    pub fn num_supernodes(&self) -> usize {
        self.supernodes.len()
    }

    /// Returns information about supernodes.
    pub fn supernodes(&self) -> &[Supernode] {
        &self.supernodes
    }

    /// Returns the permutation used.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Computes the log determinant.
    pub fn log_determinant(&self) -> T {
        let mut log_det = T::zero();

        for sn_idx in 0..self.supernodes.len() {
            let sn = &self.supernodes[sn_idx];
            let size = sn.size;
            let diag_offset = self.diag_offsets[sn_idx];

            for j in 0..size {
                let ljj = self.diag_blocks[diag_offset + j * size + j].clone();
                log_det = log_det + Real::ln(ljj);
            }
        }

        // det(A) = det(L)^2, so log(det(A)) = 2 * log(det(L))
        log_det + log_det
    }
}

/// Builds the full symmetric sparsity pattern (both triangles) from a matrix
/// whose stored entries form the lower triangle (rows `>=` column).
///
/// Returns `(col_ptrs, row_indices)` in CSC layout with each column's row
/// indices sorted ascending. Only the pattern is needed for the elimination
/// tree and fill analysis, so values are not carried.
fn full_symmetric_pattern<T: Scalar>(a: &CscMatrix<T>) -> (Vec<usize>, Vec<usize>) {
    let n = a.ncols();
    let col_ptrs_in = a.col_ptrs();
    let row_in = a.row_indices();

    // Count entries per column of the symmetrized pattern.
    let mut counts = vec![0usize; n];
    for j in 0..n {
        for idx in col_ptrs_in[j]..col_ptrs_in[j + 1] {
            let i = row_in[idx];
            counts[j] += 1;
            if i != j {
                counts[i] += 1;
            }
        }
    }

    let mut col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        col_ptrs[j + 1] = col_ptrs[j] + counts[j];
    }

    let mut row_indices = vec![0usize; col_ptrs[n]];
    let mut next = col_ptrs[..n].to_vec();
    for j in 0..n {
        for idx in col_ptrs_in[j]..col_ptrs_in[j + 1] {
            let i = row_in[idx];
            row_indices[next[j]] = i;
            next[j] += 1;
            if i != j {
                row_indices[next[i]] = j;
                next[i] += 1;
            }
        }
    }

    for j in 0..n {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        row_indices[start..end].sort_unstable();
    }

    (col_ptrs, row_indices)
}

/// Builds the elimination tree of a symmetric matrix from its full pattern.
///
/// Returns parent pointers, using `-1` for roots. This is the classic
/// path-compressing algorithm (equivalent to `cs_etree` for Cholesky).
fn elimination_tree(col_ptrs: &[usize], row_indices: &[usize], n: usize) -> Vec<isize> {
    let mut parent = vec![-1isize; n];
    let mut ancestor = vec![-1isize; n];

    for k in 0..n {
        for idx in col_ptrs[k]..col_ptrs[k + 1] {
            let i = row_indices[idx];
            if i < k {
                // Walk from node `i` toward the current root, path-compressing.
                let mut node = i as isize;
                while node != -1 && (node as usize) < k {
                    let next = ancestor[node as usize];
                    ancestor[node as usize] = k as isize;
                    if next == -1 {
                        parent[node as usize] = k as isize;
                    }
                    node = next;
                }
            }
        }
    }

    parent
}

/// Computes the fill-aware column structures of the Cholesky factor `L`.
///
/// For each column `j`, returns the sorted row indices strictly below the
/// diagonal that are nonzero in `L`. Uses the exact symbolic-Cholesky recursion
///
/// ```text
/// struct(L[:,j]) = { i > j : A[i,j] != 0 }
///                  ∪ ( ∪_{child c of j in etree} struct(L[:,c]) \ {c} )
/// ```
///
/// Because every child index is smaller than its parent, processing columns in
/// increasing order guarantees each child's structure is available when its
/// parent is reached. This captures all fill entries, unlike a bare scan of `A`.
fn column_structures(
    col_ptrs: &[usize],
    row_indices: &[usize],
    etree: &[isize],
    n: usize,
) -> Vec<Vec<usize>> {
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, &parent) in etree.iter().enumerate() {
        if parent >= 0 {
            children[parent as usize].push(c);
        }
    }

    let mut structs: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); n];

    for j in 0..n {
        let mut set = std::collections::BTreeSet::new();

        // Direct contributions from the symmetric pattern of A.
        for idx in col_ptrs[j]..col_ptrs[j + 1] {
            let row = row_indices[idx];
            if row > j {
                set.insert(row);
            }
        }

        // Fill contributions propagated from children in the elimination tree.
        for &c in &children[j] {
            for &row in &structs[c] {
                if row > j {
                    set.insert(row);
                }
            }
        }

        structs[j] = set;
    }

    structs
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

/// Tests whether column `j`'s structure nests inside column `j - 1`'s structure,
/// i.e. `struct(L[:,j-1]) == {j} ∪ struct(L[:,j])`.
///
/// This is the fundamental-supernode amalgamation criterion: when it holds (and
/// `j` is the elimination-tree parent of `j - 1`), the two columns share the
/// same off-diagonal structure and can be processed as one dense panel.
fn structures_nest(prev: &[usize], next: &[usize], j: usize) -> bool {
    // `prev` must begin with the shared parent `j`; the remainder must equal
    // `next` exactly (both are sorted, duplicate-free row lists).
    if prev.first() != Some(&j) {
        return false;
    }
    prev.len() == next.len() + 1 && prev[1..] == *next
}

/// Permutes a symmetric matrix: returns P * A * P^T
fn permute_symmetric<T: Scalar + Clone + Field>(
    a: &CscMatrix<T>,
    perm: &[usize],
    perm_inv: &[usize],
) -> CscMatrix<T> {
    let n = a.nrows();

    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for new_j in 0..n {
        let old_j = perm[new_j];

        let mut entries: Vec<(usize, T)> = Vec::new();

        let start = a.col_ptrs()[old_j];
        let end = a.col_ptrs()[old_j + 1];

        for idx in start..end {
            let old_i = a.row_indices()[idx];
            let new_i = perm_inv[old_i];

            if new_i >= new_j {
                entries.push((new_i, a.values()[idx].clone()));
            }
        }

        // Also pick up entries from upper triangle that map to lower
        for old_k in 0..n {
            if old_k == old_j {
                continue;
            }

            let new_k = perm_inv[old_k];
            if new_k < new_j {
                continue;
            }

            let k_start = a.col_ptrs()[old_k];
            let k_end = a.col_ptrs()[old_k + 1];

            for idx in k_start..k_end {
                if a.row_indices()[idx] == old_j {
                    entries.push((new_k, a.values()[idx].clone()));
                    break;
                }
            }
        }

        entries.sort_by_key(|(row, _)| *row);
        entries.dedup_by_key(|(row, _)| *row);

        for (row, val) in entries {
            row_indices.push(row);
            values.push(val);
        }

        col_ptrs[new_j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Supernodal LU factorization for general square matrices.
///
/// Factorizes `P * A = L * U` with partial (row) pivoting, where `L` is unit
/// lower triangular and `U` is upper triangular. The triangular factors are
/// stored sparsely (in CSC layout) so that [`SupernodalLU::solve`] performs
/// genuine forward/backward substitution.
#[derive(Debug, Clone)]
pub struct SupernodalLU<T: Scalar> {
    /// Size of the matrix.
    n: usize,
    /// Supernodes for L (one column each in this general-matrix implementation).
    l_supernodes: Vec<Supernode>,
    /// Column pointers for the strictly-lower entries of `L` (unit diagonal).
    l_col_ptrs: Vec<usize>,
    /// Row indices of the strictly-lower entries of `L`.
    l_row_indices: Vec<usize>,
    /// Values of the strictly-lower entries of `L`.
    l_values: Vec<T>,
    /// Column pointers for the strictly-upper entries of `U`.
    u_col_ptrs: Vec<usize>,
    /// Row indices of the strictly-upper entries of `U`.
    u_row_indices: Vec<usize>,
    /// Values of the strictly-upper entries of `U`.
    u_values: Vec<T>,
    /// Diagonal (pivot) entries of `U`.
    u_diag: Vec<T>,
    /// Row permutation from partial pivoting: `row_perm[i]` is the original row
    /// now residing at position `i`, so `(P * b)[i] == b[row_perm[i]]`.
    row_perm: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SupernodalLU<T> {
    /// Computes the LU factorization `P * A = L * U` with partial pivoting.
    ///
    /// # Errors
    ///
    /// Returns [`SupernodalError::NotSquare`] if `a` is not square, or
    /// [`SupernodalError::Singular`] if a column has no usable pivot.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, SupernodalError> {
        if a.nrows() != a.ncols() {
            return Err(SupernodalError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            return Ok(Self {
                n: 0,
                l_supernodes: vec![],
                l_col_ptrs: vec![0],
                l_row_indices: vec![],
                l_values: vec![],
                u_col_ptrs: vec![0],
                u_row_indices: vec![],
                u_values: vec![],
                u_diag: vec![],
                row_perm: vec![],
            });
        }

        // Supernode summary for reporting: each column is its own supernode in
        // this general-matrix implementation.
        let l_supernodes: Vec<Supernode> = (0..n)
            .map(|j| {
                let mut sub_rows = Vec::new();
                let col_start = a.col_ptrs()[j];
                let col_end = a.col_ptrs()[j + 1];
                for idx in col_start..col_end {
                    let row = a.row_indices()[idx];
                    if row > j {
                        sub_rows.push(row);
                    }
                }
                Supernode {
                    first_col: j,
                    size: 1,
                    sub_rows,
                }
            })
            .collect();

        // Dense working storage in column-major order: work[col * n + row].
        // Gaussian elimination with partial pivoting is performed in place; the
        // sparse triangular factors are extracted afterwards.
        let mut work = vec![T::zero(); n * n];
        for col in 0..n {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                let row = a.row_indices()[idx];
                work[col * n + row] = a.values()[idx].clone();
            }
        }

        // Partial-pivoting LU: at each step choose the largest-magnitude pivot in
        // the active column and record the row interchange.
        let mut row_perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut pivot_row = k;
            let mut pivot_mag = Scalar::abs(work[k * n + k].clone());
            for i in (k + 1)..n {
                let mag = Scalar::abs(work[k * n + i].clone());
                if mag > pivot_mag {
                    pivot_mag = mag;
                    pivot_row = i;
                }
            }

            if pivot_mag <= <T as Scalar>::epsilon() {
                return Err(SupernodalError::Singular { index: k });
            }

            if pivot_row != k {
                for col in 0..n {
                    work.swap(col * n + k, col * n + pivot_row);
                }
                row_perm.swap(k, pivot_row);
            }

            let pivot = work[k * n + k].clone();
            for i in (k + 1)..n {
                let factor = work[k * n + i].clone() / pivot.clone();
                work[k * n + i] = factor.clone();
                for col in (k + 1)..n {
                    let updated =
                        work[col * n + i].clone() - factor.clone() * work[col * n + k].clone();
                    work[col * n + i] = updated;
                }
            }
        }

        // Extract the sparse triangular factors. `L` holds the strict lower part
        // (unit diagonal implied); `U` holds the diagonal plus strict upper part.
        let mut l_col_ptrs = vec![0usize; n + 1];
        let mut l_row_indices = Vec::new();
        let mut l_values = Vec::new();
        let mut u_col_ptrs = vec![0usize; n + 1];
        let mut u_row_indices = Vec::new();
        let mut u_values = Vec::new();
        let mut u_diag = vec![T::zero(); n];

        for col in 0..n {
            for row in 0..col {
                let val = work[col * n + row].clone();
                if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                    u_row_indices.push(row);
                    u_values.push(val);
                }
            }
            u_col_ptrs[col + 1] = u_values.len();

            u_diag[col] = work[col * n + col].clone();

            for row in (col + 1)..n {
                let val = work[col * n + row].clone();
                if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                    l_row_indices.push(row);
                    l_values.push(val);
                }
            }
            l_col_ptrs[col + 1] = l_values.len();
        }

        Ok(Self {
            n,
            l_supernodes,
            l_col_ptrs,
            l_row_indices,
            l_values,
            u_col_ptrs,
            u_row_indices,
            u_values,
            u_diag,
            row_perm,
        })
    }

    /// Solves `A * x = b` using the stored `L` and `U` factors.
    ///
    /// Applies the pivot permutation (`c = P * b`), performs forward
    /// substitution against the unit-lower factor `L` (`L * z = c`), then
    /// backward substitution against the upper factor `U` (`U * x = z`).
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix size");

        // Apply the pivot permutation: c = P * b.
        let mut z = vec![T::zero(); n];
        for i in 0..n {
            z[i] = b[self.row_perm[i]].clone();
        }

        // Forward substitution: solve L * z = c (L is unit lower triangular).
        // Column-oriented: once z[j] is final, eliminate it from lower rows.
        for j in 0..n {
            let zj = z[j].clone();
            for idx in self.l_col_ptrs[j]..self.l_col_ptrs[j + 1] {
                let i = self.l_row_indices[idx];
                z[i] = z[i].clone() - self.l_values[idx].clone() * zj.clone();
            }
        }

        // Backward substitution: solve U * x = z, processing columns in reverse.
        let mut x = z;
        for j in (0..n).rev() {
            let xj = x[j].clone() / self.u_diag[j].clone();
            x[j] = xj.clone();
            for idx in self.u_col_ptrs[j]..self.u_col_ptrs[j + 1] {
                let i = self.u_row_indices[idx];
                x[i] = x[i].clone() - self.u_values[idx].clone() * xj.clone();
            }
        }

        x
    }

    /// Returns the number of supernodes.
    pub fn num_supernodes(&self) -> usize {
        self.l_supernodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spd_matrix() -> CscMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];

        CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap()
    }

    fn make_larger_spd_matrix() -> CscMatrix<f64> {
        // 5x5 tridiagonal SPD matrix
        let n = 5;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0usize];

        for j in 0..n {
            if j > 0 {
                row_indices.push(j - 1);
                values.push(1.0);
            }
            row_indices.push(j);
            values.push(4.0);
            if j < n - 1 {
                row_indices.push(j + 1);
                values.push(1.0);
            }
            col_ptrs.push(values.len());
        }

        CscMatrix::new(n, n, col_ptrs, row_indices, values).unwrap()
    }

    #[test]
    fn test_supernodal_cholesky_small() {
        let a = make_spd_matrix();
        let chol = SupernodalCholesky::new(&a).unwrap();

        assert!(chol.num_supernodes() > 0);
        assert!(chol.num_supernodes() <= 3);
    }

    #[test]
    fn test_supernodal_cholesky_solve() {
        let a = make_spd_matrix();
        let chol = SupernodalCholesky::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let x = chol.solve(&b);

        // Verify A * x ≈ b
        let mut ax = [0.0; 3];
        for col in 0..3 {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                ax[a.row_indices()[idx]] += a.values()[idx] * x[col];
            }
        }

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Solution verification failed at index {}: ax={}, b={}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_supernodal_cholesky_larger() {
        let a = make_larger_spd_matrix();
        let chol = SupernodalCholesky::new(&a).unwrap();

        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let x = chol.solve(&b);

        // Verify A * x ≈ b
        let mut ax = [0.0; 5];
        for col in 0..5 {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                ax[a.row_indices()[idx]] += a.values()[idx] * x[col];
            }
        }

        for i in 0..5 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-9,
                "Solution verification failed at index {}: ax={}, b={}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_supernodal_cholesky_log_det() {
        let a = make_spd_matrix();
        let chol = SupernodalCholesky::new(&a).unwrap();

        let log_det = chol.log_determinant();

        // For tridiagonal [4,1,0; 1,4,1; 0,1,4]:
        // det = 4*(4*4 - 1*1) - 1*(1*4 - 0) = 4*15 - 4 = 56
        let expected_log_det = 56.0f64.ln();

        assert!(
            (log_det - expected_log_det).abs() < 1e-9,
            "Log determinant error: got {}, expected {}",
            log_det,
            expected_log_det
        );
    }

    #[test]
    fn test_supernodal_cholesky_not_spd() {
        // Negative definite matrix
        let values = vec![-4.0, 1.0, 1.0, -4.0, 1.0, 1.0, -4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];

        let a = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        let result = SupernodalCholesky::new(&a);

        assert!(matches!(
            result,
            Err(SupernodalError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn test_supernode_detection() {
        // For tridiagonal matrix, we should detect supernodes
        let a = make_larger_spd_matrix();
        let chol = SupernodalCholesky::new(&a).unwrap();

        // Print supernode info for debugging
        for (i, sn) in chol.supernodes().iter().enumerate() {
            println!(
                "Supernode {}: cols {}-{}, size {}, sub_rows {:?}",
                i,
                sn.first_col,
                sn.last_col(),
                sn.size,
                sn.sub_rows
            );
        }

        // Should have fewer supernodes than columns (some grouping)
        // For tridiagonal, may or may not group depending on implementation
        assert!(chol.num_supernodes() <= 5);
    }

    /// Builds a `k x k` 2D grid 5-point Laplacian as a full symmetric SPD matrix.
    ///
    /// Diagonal `4`, nearest-neighbor coupling `-1`. Such matrices generate
    /// genuine fill-in under *any* elimination ordering, which exercises the
    /// fill-aware symbolic analysis.
    fn make_grid_laplacian(k: usize) -> CscMatrix<f64> {
        let n = k * k;
        let idx = |r: usize, c: usize| r * k + c;

        let mut col_ptrs = vec![0usize];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        for lin in 0..n {
            let r = lin / k;
            let c = lin % k;
            let mut entries: Vec<(usize, f64)> = Vec::new();

            if r > 0 {
                entries.push((idx(r - 1, c), -1.0));
            }
            if c > 0 {
                entries.push((idx(r, c - 1), -1.0));
            }
            entries.push((lin, 4.0));
            if c + 1 < k {
                entries.push((idx(r, c + 1), -1.0));
            }
            if r + 1 < k {
                entries.push((idx(r + 1, c), -1.0));
            }

            entries.sort_by_key(|(row, _)| *row);
            for (row, val) in entries {
                row_indices.push(row);
                values.push(val);
            }
            col_ptrs.push(values.len());
        }

        CscMatrix::new(n, n, col_ptrs, row_indices, values).unwrap()
    }

    /// Computes `A * x` for a CSC matrix (works for any square matrix).
    fn csc_matvec(a: &CscMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let n = a.nrows();
        let mut y = vec![0.0; n];
        for col in 0..a.ncols() {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                y[a.row_indices()[idx]] += a.values()[idx] * x[col];
            }
        }
        y
    }

    #[test]
    fn test_supernodal_cholesky_fill_grid_laplacian() {
        // A 4x4 grid Laplacian (n = 16) generates fill-in that the pattern of A
        // alone does not contain. If sub_rows ignored fill, the factor would be
        // wrong and the residual large.
        let a = make_grid_laplacian(4);
        let n = a.nrows();
        let chol = SupernodalCholesky::new(&a).expect("grid Laplacian is SPD");

        let b: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.5) - 3.0).collect();
        let x = chol.solve(&b);

        let ax = csc_matvec(&a, &x);
        let residual: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-9,
            "grid Laplacian relative residual too large: {}",
            residual / b_norm
        );
    }

    #[test]
    fn test_supernodal_cholesky_fill_grid_laplacian_3x3() {
        // A second, smaller fill-generating case (3x3 grid, n = 9) with a
        // different right-hand side. Ground truth is the residual A * x - b.
        let a = make_grid_laplacian(3);
        let n = a.nrows();
        let chol = SupernodalCholesky::new(&a).expect("grid Laplacian is SPD");

        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64)).collect();
        let x = chol.solve(&b);

        let ax = csc_matvec(&a, &x);
        let residual: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            residual / b_norm < 1e-9,
            "3x3 grid Laplacian relative residual too large: {}",
            residual / b_norm
        );
    }

    #[test]
    fn test_supernodal_lu_solve_general() {
        // A general (non-symmetric) invertible matrix.
        // A = [2 1 1]
        //     [4 3 3]
        //     [8 7 9]
        let values = vec![2.0, 4.0, 8.0, 1.0, 3.0, 7.0, 1.0, 3.0, 9.0];
        let row_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let col_ptrs = vec![0, 3, 6, 9];
        let a = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();

        let lu = SupernodalLU::new(&a).expect("matrix is nonsingular");

        let b = vec![4.0, 10.0, 24.0];
        let x = lu.solve(&b);

        let ax = csc_matvec(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "LU solve residual at {}: {} vs {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_supernodal_lu_requires_pivoting() {
        // Zero on the diagonal forces a row interchange.
        // A = [0 1]
        //     [1 0]
        let values = vec![1.0, 1.0];
        let row_indices = vec![1, 0];
        let col_ptrs = vec![0, 1, 2];
        let a = CscMatrix::new(2, 2, col_ptrs, row_indices, values).unwrap();

        let lu = SupernodalLU::new(&a).expect("permutation matrix is nonsingular");

        let b = vec![1.0, 2.0];
        let x = lu.solve(&b);

        // A x = [x1, x0] = [1, 2] => x = [2, 1].
        assert!((x[0] - 2.0).abs() < 1e-12, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x1 = {}", x[1]);
    }

    #[test]
    fn test_supernodal_lu_identity() {
        let a = CscMatrix::<f64>::eye(4);
        let lu = SupernodalLU::new(&a).expect("identity is nonsingular");

        let b = vec![1.5, -2.0, 3.25, 4.0];
        let x = lu.solve(&b);

        for i in 0..4 {
            assert!(
                (x[i] - b[i]).abs() < 1e-12,
                "identity solve at {}: {}",
                i,
                x[i]
            );
        }
    }

    #[test]
    fn test_supernodal_lu_larger() {
        // A denser 5x5 non-symmetric, diagonally dominant matrix.
        let n = 5;
        let mut col_ptrs = vec![0usize];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v = if i == j {
                    10.0 + (i as f64)
                } else {
                    ((i as f64) - (j as f64)) * 0.5 + 1.0
                };
                row_indices.push(i);
                values.push(v);
            }
            col_ptrs.push(values.len());
        }
        let a = CscMatrix::new(n, n, col_ptrs, row_indices, values).unwrap();

        let lu = SupernodalLU::new(&a).expect("diagonally dominant matrix is nonsingular");

        let b: Vec<f64> = (0..n).map(|i| (i as f64) - 2.0).collect();
        let x = lu.solve(&b);

        let ax = csc_matvec(&a, &x);
        let residual: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(residual < 1e-9, "LU residual norm too large: {}", residual);
    }

    #[test]
    fn test_supernodal_lu_singular() {
        // Rank-deficient matrix: column 1 == 2 * column 0.
        // A = [1 2]
        //     [2 4]
        let values = vec![1.0, 2.0, 2.0, 4.0];
        let row_indices = vec![0, 1, 0, 1];
        let col_ptrs = vec![0, 2, 4];
        let a = CscMatrix::new(2, 2, col_ptrs, row_indices, values).unwrap();

        let result = SupernodalLU::new(&a);
        assert!(matches!(result, Err(SupernodalError::Singular { .. })));
    }
}
