//! Multifrontal LU factorization for general sparse matrices.
//!
//! The multifrontal LU method extends the multifrontal approach to non-symmetric
//! matrices using partial pivoting. It processes the elimination tree bottom-up:
//!
//! 1. For each node, assemble a frontal matrix from original entries + child contributions
//! 2. Factor the fully-summed rows/columns using dense LU with partial pivoting
//! 3. Compute the Schur complement (update matrix) and pass it to the parent
//!
//! Unlike the Cholesky variant, the LU method must handle asymmetric fill-in and
//! row pivoting, which requires maintaining separate row and column index mappings.

use crate::csc::CscMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::multifrontal_cholesky::MultifrontalError;

/// An LU update matrix passed from child to parent in the elimination tree.
#[derive(Debug, Clone)]
struct LuUpdateMatrix<T: Scalar> {
    /// Global row indices.
    row_indices: Vec<usize>,
    /// Global column indices.
    col_indices: Vec<usize>,
    /// Dense data in column-major order (nrows x ncols).
    data: Vec<T>,
}

/// A frontal matrix for LU factorization.
///
/// Structure:
/// ```text
///   [ F11  F12 ]   fully-summed rows/cols (ns x ns)
///   [ F21  F22 ]   update part
/// ```
/// where F11 undergoes LU with pivoting, and the Schur complement
/// S = F22 - F21 * F11^{-1} * F12 is passed to parent.
#[derive(Debug)]
struct LuFrontalMatrix<T: Scalar> {
    /// Global row indices.
    row_indices: Vec<usize>,
    /// Global column indices (same as row_indices for symmetric structure initially).
    col_indices: Vec<usize>,
    /// Number of fully-summed variables.
    num_fully_summed: usize,
    /// Dense data in column-major order (nrows x ncols).
    data: Vec<T>,
    /// Local row permutation applied during pivoting.
    local_row_perm: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> LuFrontalMatrix<T> {
    /// Creates a new LU frontal matrix.
    fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, num_fully_summed: usize) -> Self {
        let nrows = row_indices.len();
        let ncols = col_indices.len();
        let local_row_perm: Vec<usize> = (0..nrows).collect();
        Self {
            row_indices,
            col_indices,
            num_fully_summed,
            data: vec![T::zero(); nrows * ncols],
            local_row_perm,
        }
    }

    fn nrows(&self) -> usize {
        self.row_indices.len()
    }

    fn ncols(&self) -> usize {
        self.col_indices.len()
    }

    /// Gets element at (i, j) in column-major storage.
    fn get(&self, i: usize, j: usize) -> &T {
        let nrows = self.nrows();
        &self.data[j * nrows + i]
    }

    /// Gets mutable element at (i, j).
    fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        let nrows = self.nrows();
        &mut self.data[j * nrows + i]
    }

    /// Assembles original matrix entries from the CSC matrix.
    ///
    /// For the fully-summed column `node`, loads:
    /// 1. All entries from column `node` whose rows are in the frontal row set
    /// 2. All entries in row `node` whose columns are in the frontal column set
    ///    (by scanning those columns of A)
    fn assemble_original(&mut self, a: &CscMatrix<T>, fully_summed_cols: &[usize]) {
        let mut row_to_local = std::collections::HashMap::new();
        for (local, &global) in self.row_indices.iter().enumerate() {
            row_to_local.insert(global, local);
        }
        let mut col_to_local = std::collections::HashMap::new();
        for (local, &global) in self.col_indices.iter().enumerate() {
            col_to_local.insert(global, local);
        }

        // 1. Load entries from the fully-summed column(s)
        for &col in fully_summed_cols {
            let local_j = match col_to_local.get(&col) {
                Some(&lj) => lj,
                None => continue,
            };

            let col_start = a.col_ptrs()[col];
            let col_end = a.col_ptrs()[col + 1];

            for idx in col_start..col_end {
                let row = a.row_indices()[idx];
                if let Some(&local_i) = row_to_local.get(&row) {
                    let val = a.values()[idx].clone();
                    let current = self.get(local_i, local_j).clone();
                    *self.get_mut(local_i, local_j) = current + val;
                }
            }
        }

        // 2. Load entries from the fully-summed row(s) by scanning other columns
        for &row in fully_summed_cols {
            let local_i = match row_to_local.get(&row) {
                Some(&li) => li,
                None => continue,
            };

            // Scan all columns in the frontal column set to find entries in row `row`
            for &col_global in &self.col_indices.clone() {
                // Skip the fully-summed column itself (already loaded above)
                if fully_summed_cols.contains(&col_global) {
                    continue;
                }

                let local_j = match col_to_local.get(&col_global) {
                    Some(&lj) => lj,
                    None => continue,
                };

                let col_start = a.col_ptrs()[col_global];
                let col_end = a.col_ptrs()[col_global + 1];

                for idx in col_start..col_end {
                    if a.row_indices()[idx] == row {
                        let val = a.values()[idx].clone();
                        let current = self.get(local_i, local_j).clone();
                        *self.get_mut(local_i, local_j) = current + val;
                        break;
                    }
                }
            }
        }
    }

    /// Assembles an LU update matrix from a child.
    fn assemble_lu_update(&mut self, update: &LuUpdateMatrix<T>) {
        let mut row_to_local = std::collections::HashMap::new();
        for (local, &global) in self.row_indices.iter().enumerate() {
            row_to_local.insert(global, local);
        }
        let mut col_to_local = std::collections::HashMap::new();
        for (local, &global) in self.col_indices.iter().enumerate() {
            col_to_local.insert(global, local);
        }

        let u_nrows = update.row_indices.len();
        for (uj, &gcol) in update.col_indices.iter().enumerate() {
            let local_j = match col_to_local.get(&gcol) {
                Some(&lj) => lj,
                None => continue,
            };
            for (ui, &grow) in update.row_indices.iter().enumerate() {
                let local_i = match row_to_local.get(&grow) {
                    Some(&li) => li,
                    None => continue,
                };
                let val = update.data[uj * u_nrows + ui].clone();
                if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                    let current = self.get(local_i, local_j).clone();
                    *self.get_mut(local_i, local_j) = current + val;
                }
            }
        }
    }

    /// Factors the fully-summed part using LU with partial pivoting and
    /// computes the Schur complement.
    fn factor_and_schur(
        &mut self,
        first_global_col: usize,
    ) -> Result<Option<LuUpdateMatrix<T>>, MultifrontalError> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let ns = self.num_fully_summed;
        let nu_rows = nrows - ns;
        let nu_cols = ncols - ns;

        // Step 1: LU with partial pivoting on the ns x ns leading block
        for k in 0..ns {
            // Find pivot: maximum |F[i,k]| for i in k..nrows
            let mut max_val = T::zero();
            let mut max_row = k;

            for i in k..nrows {
                let val = Scalar::abs(self.get(i, k).clone());
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val <= <T as Scalar>::epsilon() {
                return Err(MultifrontalError::ZeroPivot {
                    index: first_global_col + k,
                });
            }

            // Swap rows k and max_row
            if max_row != k {
                for j in 0..ncols {
                    let tmp = self.get(k, j).clone();
                    *self.get_mut(k, j) = self.get(max_row, j).clone();
                    *self.get_mut(max_row, j) = tmp;
                }
                self.row_indices.swap(k, max_row);
                self.local_row_perm.swap(k, max_row);
            }

            let pivot = self.get(k, k).clone();

            // Compute L entries: L[i,k] = F[i,k] / pivot for i > k
            for i in (k + 1)..nrows {
                let lik = self.get(i, k).clone() / pivot.clone();
                *self.get_mut(i, k) = lik.clone();

                // Update remaining matrix: F[i,j] -= L[i,k] * F[k,j] for j > k
                for j in (k + 1)..ncols {
                    let fkj = self.get(k, j).clone();
                    let fij = self.get(i, j).clone();
                    *self.get_mut(i, j) = fij - lik.clone() * fkj;
                }
            }
        }

        // Step 2: Extract Schur complement (already computed in-place in F22)
        if nu_rows == 0 || nu_cols == 0 {
            return Ok(None);
        }

        let update_row_indices: Vec<usize> = self.row_indices[ns..].to_vec();
        let update_col_indices: Vec<usize> = self.col_indices[ns..].to_vec();
        let mut update_data = vec![T::zero(); nu_rows * nu_cols];

        for uj in 0..nu_cols {
            for ui in 0..nu_rows {
                update_data[uj * nu_rows + ui] = self.get(ns + ui, ns + uj).clone();
            }
        }

        Ok(Some(LuUpdateMatrix {
            row_indices: update_row_indices,
            col_indices: update_col_indices,
            data: update_data,
        }))
    }
}

/// Multifrontal LU factorization for general sparse matrices.
///
/// Computes P * A * Q = L * U where:
/// - P is a row permutation (from partial pivoting)
/// - Q is a column permutation (from fill-reducing ordering)
/// - L is unit lower triangular
/// - U is upper triangular
///
/// # Algorithm
///
/// Uses the elimination tree of A^T * A (column elimination tree) to guide
/// the bottom-up assembly. At each node:
/// 1. Assemble frontal matrix from original entries + child contributions
/// 2. Factor fully-summed part using dense LU with partial pivoting
/// 3. Pass Schur complement to parent
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::MultifrontalLU;
/// let lu = MultifrontalLU::new(&a)?;
/// let x = lu.solve(&b);
/// ```
#[derive(Debug, Clone)]
pub struct MultifrontalLU<T: Scalar> {
    /// Matrix dimension.
    n: usize,
    /// Elimination tree parent pointers.
    #[allow(dead_code)]
    etree_parent: Vec<Option<usize>>,
    /// Children lists.
    #[allow(dead_code)]
    etree_children: Vec<Vec<usize>>,
    /// Postorder traversal.
    #[allow(dead_code)]
    postorder: Vec<usize>,
    /// For each column j: L values below diagonal.
    /// l_values[j][k] = L[l_row_indices[j][k], j]
    l_row_indices: Vec<Vec<usize>>,
    /// L values (unit diagonal implicit, stored entries are below diagonal).
    l_values: Vec<Vec<T>>,
    /// For each row j of U: U values to the right of diagonal.
    /// u_col_indices[j][k] = column index, u_values[j][0] = U[j,j] diagonal
    u_col_indices: Vec<Vec<usize>>,
    /// U values: [U[j,j], U[j, u_col_indices[j][0]], ...]
    u_values: Vec<Vec<T>>,
    /// Row permutation (from pivoting within frontal matrices).
    row_perm: Vec<usize>,
    /// Inverse row permutation.
    #[allow(dead_code)]
    row_perm_inv: Vec<usize>,
    /// Column permutation (from fill-reducing ordering).
    col_perm: Vec<usize>,
    /// Inverse column permutation.
    #[allow(dead_code)]
    col_perm_inv: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> MultifrontalLU<T> {
    /// Computes the multifrontal LU factorization.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix in CSC format
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or is singular.
    pub fn new(a: &CscMatrix<T>) -> Result<Self, MultifrontalError> {
        if a.nrows() != a.ncols() {
            return Err(MultifrontalError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }

        let n = a.nrows();
        if n == 0 {
            return Ok(Self {
                n: 0,
                etree_parent: vec![],
                etree_children: vec![],
                postorder: vec![],
                l_row_indices: vec![],
                l_values: vec![],
                u_col_indices: vec![],
                u_values: vec![],
                row_perm: vec![],
                row_perm_inv: vec![],
                col_perm: vec![],
                col_perm_inv: vec![],
            });
        }

        // For LU, we use AMD on A^T*A (column etree) for fill-reducing column ordering
        // Simplified: use AMD on the symmetrized pattern |A| + |A|^T
        let a_sym = symmetrize_pattern(a);

        let col_perm = super::ordering::approximate_minimum_degree(&a_sym);
        let mut col_perm_inv = vec![0; n];
        for (i, &p) in col_perm.iter().enumerate() {
            col_perm_inv[p] = i;
        }

        // Apply column permutation to A: A_q = A * Q
        let a_q = permute_columns(a, &col_perm);

        // Build column elimination tree from symmetrized |A_q|^T * |A_q|
        let a_q_sym = symmetrize_pattern(&a_q);
        let etree_parent = build_col_etree(&a_q_sym);
        let mut etree_children = vec![Vec::new(); n];
        for (j, parent) in etree_parent.iter().enumerate() {
            if let Some(p) = parent {
                etree_children[*p].push(j);
            }
        }

        let postorder = compute_postorder_lu(&etree_parent, &etree_children, n);

        // Symbolic analysis for row structure
        let col_struct = symbolic_lu(&a_q, &etree_parent);

        // Numeric factorization
        let (l_row_indices, l_values, u_col_indices, u_values, row_perm) =
            Self::multifrontal_lu_factorize(
                &a_q,
                &etree_parent,
                &etree_children,
                &postorder,
                &col_struct,
            )?;

        let mut row_perm_inv = vec![0; n];
        for (i, &p) in row_perm.iter().enumerate() {
            row_perm_inv[p] = i;
        }

        Ok(Self {
            n,
            etree_parent,
            etree_children,
            postorder,
            l_row_indices,
            l_values,
            u_col_indices,
            u_values,
            row_perm,
            row_perm_inv,
            col_perm,
            col_perm_inv,
        })
    }

    /// Performs the multifrontal LU numeric factorization.
    fn multifrontal_lu_factorize(
        a: &CscMatrix<T>,
        _etree_parent: &[Option<usize>],
        etree_children: &[Vec<usize>],
        postorder: &[usize],
        col_struct: &[Vec<usize>],
    ) -> Result<
        (
            Vec<Vec<usize>>,
            Vec<Vec<T>>,
            Vec<Vec<usize>>,
            Vec<Vec<T>>,
            Vec<usize>,
        ),
        MultifrontalError,
    > {
        let n = a.nrows();
        let mut l_row_indices: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut l_values: Vec<Vec<T>> = vec![Vec::new(); n];
        let mut u_col_indices: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut u_values: Vec<Vec<T>> = vec![Vec::new(); n];

        // Global row permutation tracking
        let mut global_row_perm: Vec<usize> = (0..n).collect();

        // Pending update matrices
        let mut pending_updates: Vec<Option<LuUpdateMatrix<T>>> = vec![None; n];

        for &node in postorder {
            // Build frontal matrix indices
            // Rows: {node} union col_struct[node] (rows that have entries in this column)
            // Cols: same as rows initially (for the column elimination tree approach)
            let mut frontal_rows = vec![node];
            frontal_rows.extend_from_slice(&col_struct[node]);

            let mut frontal_cols = vec![node];
            frontal_cols.extend_from_slice(&col_struct[node]);

            // Include indices from children's updates
            for &child in &etree_children[node] {
                if let Some(ref update) = pending_updates[child] {
                    for &idx in &update.row_indices {
                        if !frontal_rows.contains(&idx) {
                            frontal_rows.push(idx);
                        }
                    }
                    for &idx in &update.col_indices {
                        if !frontal_cols.contains(&idx) {
                            frontal_cols.push(idx);
                        }
                    }
                }
            }

            // Sort: keep 'node' first, then sort the rest
            let mut other_rows: Vec<usize> = frontal_rows[1..].to_vec();
            other_rows.sort_unstable();
            other_rows.dedup();
            frontal_rows = vec![node];
            frontal_rows.extend_from_slice(&other_rows);

            let mut other_cols: Vec<usize> = frontal_cols[1..].to_vec();
            other_cols.sort_unstable();
            other_cols.dedup();
            frontal_cols = vec![node];
            frontal_cols.extend_from_slice(&other_cols);

            let num_fully_summed = 1;

            let mut frontal =
                LuFrontalMatrix::new(frontal_rows.clone(), frontal_cols.clone(), num_fully_summed);

            // Assemble original entries
            frontal.assemble_original(a, &[node]);

            // Assemble children's updates
            for &child in &etree_children[node] {
                if let Some(update) = pending_updates[child].take() {
                    frontal.assemble_lu_update(&update);
                }
            }

            // Factor and compute Schur complement
            let update = frontal.factor_and_schur(node)?;

            // Track pivoting: the frontal's row_indices[0] after pivoting
            // tells us which global row ended up at position 0 (the pivot row)
            let pivot_global_row = frontal.row_indices[0];
            if pivot_global_row != node {
                // Swap in global permutation
                let pos_node = global_row_perm
                    .iter()
                    .position(|&r| r == node)
                    .unwrap_or(node);
                let pos_pivot = global_row_perm
                    .iter()
                    .position(|&r| r == pivot_global_row)
                    .unwrap_or(pivot_global_row);
                global_row_perm.swap(pos_node, pos_pivot);
            }

            // Extract L column (below diagonal, unit diagonal implicit)
            let total_rows = frontal.nrows();
            for i in 1..total_rows {
                let val = frontal.get(i, 0).clone();
                if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                    l_row_indices[node].push(frontal.row_indices[i]);
                    l_values[node].push(val);
                }
            }

            // Extract U row (at and right of diagonal)
            let total_cols = frontal.ncols();
            // U[node, node] = pivot
            u_values[node].push(frontal.get(0, 0).clone());
            for j in 1..total_cols {
                let val = frontal.get(0, j).clone();
                if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                    u_col_indices[node].push(frontal.col_indices[j]);
                    u_values[node].push(val);
                }
            }

            // Store update matrix at this node (to be consumed by parent)
            if let Some(u) = update {
                pending_updates[node] = Some(u);
            }
        }

        Ok((
            l_row_indices,
            l_values,
            u_col_indices,
            u_values,
            global_row_perm,
        ))
    }

    /// Solves A * x = b.
    ///
    /// Computes x = Q * U^{-1} * L^{-1} * P * b
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix dimension");

        // Apply row permutation: y = P * b
        let mut y = vec![T::zero(); n];
        for i in 0..n {
            y[i] = b[self.row_perm[i]].clone();
        }

        // Forward solve: L * z = y (L has unit diagonal)
        for j in 0..n {
            for (k, &row) in self.l_row_indices[j].iter().enumerate() {
                let lij = self.l_values[j][k].clone();
                y[row] = y[row].clone() - lij * y[j].clone();
            }
        }

        // Backward solve: U * w = z
        for j in (0..n).rev() {
            if self.u_values[j].is_empty() {
                continue;
            }

            // Subtract contributions from U entries right of diagonal
            for (k, &col) in self.u_col_indices[j].iter().enumerate() {
                let uij = self.u_values[j][k + 1].clone();
                y[j] = y[j].clone() - uij * y[col].clone();
            }

            // Divide by diagonal
            let ujj = self.u_values[j][0].clone();
            y[j] = y[j].clone() / ujj;
        }

        // Apply inverse column permutation: x = Q * w
        // col_perm maps new_col -> old_col, so x[col_perm[i]] = w[i]
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            x[self.col_perm[i]] = y[i].clone();
        }

        x
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the row permutation.
    pub fn row_perm(&self) -> &[usize] {
        &self.row_perm
    }

    /// Returns the column permutation.
    pub fn col_perm(&self) -> &[usize] {
        &self.col_perm
    }

    /// Returns the approximate number of nonzeros in L + U.
    pub fn nnz(&self) -> usize {
        let mut nnz = 0usize;
        for j in 0..self.n {
            nnz += self.l_values[j].len();
            nnz += self.u_values[j].len();
        }
        nnz
    }

    /// Computes the determinant.
    pub fn determinant(&self) -> T {
        let mut det = T::one();
        for j in 0..self.n {
            if !self.u_values[j].is_empty() {
                det = det * self.u_values[j][0].clone();
            }
        }

        // Account for permutation signs
        let row_sign = permutation_sign(&self.row_perm);
        let col_sign = permutation_sign(&self.col_perm);
        if (row_sign * col_sign) < 0 {
            det = T::zero() - det;
        }

        det
    }
}

/// Symmetrizes the sparsity pattern: |A| + |A|^T.
fn symmetrize_pattern<T: Scalar + Clone + Field>(a: &CscMatrix<T>) -> CscMatrix<T> {
    let n = a.nrows();
    let mut entries: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];

    // Add entries from A
    for j in 0..n {
        let start = a.col_ptrs()[j];
        let end = a.col_ptrs()[j + 1];
        for idx in start..end {
            let i = a.row_indices()[idx];
            let val = a.values()[idx].clone();
            entries[j].push((i, val.clone()));
            if i != j {
                entries[i].push((j, val));
            }
        }
    }

    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for j in 0..n {
        entries[j].sort_by_key(|(r, _)| *r);
        entries[j].dedup_by_key(|(r, _)| *r);
        for (r, v) in &entries[j] {
            row_indices.push(*r);
            values.push(v.clone());
        }
        col_ptrs[j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

/// Permutes columns of a CSC matrix: A_q = A * Q where Q[new_j] = old_j.
fn permute_columns<T: Scalar + Clone>(a: &CscMatrix<T>, perm: &[usize]) -> CscMatrix<T> {
    let n = a.ncols();
    let nrows = a.nrows();
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    for new_j in 0..n {
        let old_j = perm[new_j];
        let start = a.col_ptrs()[old_j];
        let end = a.col_ptrs()[old_j + 1];

        let mut entries: Vec<(usize, T)> = Vec::new();
        for idx in start..end {
            entries.push((a.row_indices()[idx], a.values()[idx].clone()));
        }
        entries.sort_by_key(|(r, _)| *r);

        for (r, v) in entries {
            row_indices.push(r);
            values.push(v);
        }
        col_ptrs[new_j + 1] = values.len();
    }

    unsafe { CscMatrix::new_unchecked(nrows, n, col_ptrs, row_indices, values) }
}

/// Builds column elimination tree from symmetrized pattern.
fn build_col_etree<T: Scalar>(a: &CscMatrix<T>) -> Vec<Option<usize>> {
    let n = a.nrows();
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut ancestor = vec![0usize; n];

    for k in 0..n {
        ancestor[k] = k;
        let col_start = a.col_ptrs()[k];
        let col_end = a.col_ptrs()[k + 1];

        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i < k {
                let mut r = i;
                while ancestor[r] != r && ancestor[r] != k {
                    let next = ancestor[r];
                    ancestor[r] = k;
                    r = next;
                }
                if ancestor[r] == r {
                    parent[r] = Some(k);
                }
                let mut j = i;
                while ancestor[j] != k {
                    let next = ancestor[j];
                    ancestor[j] = k;
                    j = next;
                }
            }
        }
    }

    parent
}

/// Computes postorder for LU elimination tree.
fn compute_postorder_lu(parent: &[Option<usize>], children: &[Vec<usize>], n: usize) -> Vec<usize> {
    let roots: Vec<usize> = (0..n).filter(|&i| parent[i].is_none()).collect();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, bool)> = Vec::new();
    let mut visited = vec![false; n];

    for &root in &roots {
        stack.push((root, false));
    }

    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
        } else if !visited[node] {
            visited[node] = true;
            stack.push((node, true));
            for &child in children[node].iter().rev() {
                if !visited[child] {
                    stack.push((child, false));
                }
            }
        }
    }

    order
}

/// Symbolic analysis for LU: determines row structure for each column.
fn symbolic_lu<T: Scalar>(a: &CscMatrix<T>, parent: &[Option<usize>]) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut col_struct: Vec<Vec<usize>> = vec![Vec::new(); n];

    for j in 0..n {
        let mut row_set = std::collections::BTreeSet::new();

        // Add row indices from A[:,j] that are > j
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let row = a.row_indices()[idx];
            if row > j {
                row_set.insert(row);
            }
        }

        // Propagate through elimination tree
        let rows: Vec<usize> = row_set.iter().copied().collect();
        for &r in &rows {
            let mut current = r;
            while let Some(p) = parent[current] {
                if p <= j {
                    break;
                }
                row_set.insert(p);
                current = p;
            }
        }

        col_struct[j] = row_set.into_iter().collect();
    }

    col_struct
}

/// Computes the sign of a permutation (+1 for even, -1 for odd).
fn permutation_sign(perm: &[usize]) -> i32 {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut sign = 1i32;

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut cycle_len = 0;
        let mut j = i;
        while !visited[j] {
            visited[j] = true;
            j = perm[j];
            cycle_len += 1;
        }
        if cycle_len % 2 == 0 {
            sign = -sign;
        }
    }

    sign
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a 3x3 general matrix.
    fn make_general_3x3() -> CscMatrix<f64> {
        // A = [2 1 0]
        //     [1 3 1]
        //     [0 1 2]
        let values = vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];
        CscMatrix::new(3, 3, col_ptrs, row_indices, values).expect("valid 3x3 matrix construction")
    }

    /// Creates a 3x3 asymmetric matrix.
    fn make_asymmetric_3x3() -> CscMatrix<f64> {
        // A = [4 1 2]
        //     [3 5 1]
        //     [1 2 6]
        let values = vec![4.0, 3.0, 1.0, 1.0, 5.0, 2.0, 2.0, 1.0, 6.0];
        let row_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let col_ptrs = vec![0, 3, 6, 9];
        CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("valid asymmetric 3x3 matrix construction")
    }

    /// Creates a 5x5 general tridiagonal matrix.
    fn make_general_5x5() -> CscMatrix<f64> {
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

    #[test]
    fn test_multifrontal_lu_3x3_symmetric() {
        let a = make_general_3x3();
        let lu = MultifrontalLU::new(&a).expect("LU factorization should succeed");

        let b = vec![3.0, 5.0, 3.0];
        let x = lu.solve(&b);

        let ax = csc_matvec(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-9,
                "LU 3x3 residual at {}: {} vs {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_lu_3x3_asymmetric() {
        let a = make_asymmetric_3x3();
        let lu = MultifrontalLU::new(&a).expect("LU factorization should succeed");

        let b = vec![7.0, 9.0, 9.0];
        let x = lu.solve(&b);

        let ax = csc_matvec(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-9,
                "LU asymmetric residual at {}: {} vs {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_lu_5x5() {
        let a = make_general_5x5();
        let lu = MultifrontalLU::new(&a).expect("LU factorization should succeed");

        let b = vec![1.0, -1.0, 2.0, -2.0, 1.0];
        let x = lu.solve(&b);

        let ax = csc_matvec(&a, &x);
        let residual: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-9,
            "LU 5x5 relative residual too large: {}",
            residual / b_norm
        );
    }

    #[test]
    fn test_multifrontal_lu_identity() {
        let a = CscMatrix::<f64>::eye(4);
        let lu = MultifrontalLU::new(&a).expect("identity LU should succeed");

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = lu.solve(&b);

        for i in 0..4 {
            assert!(
                (x[i] - b[i]).abs() < 1e-10,
                "Identity solve failed at {}: {} vs {}",
                i,
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_lu_not_square() {
        let a = CscMatrix::new(3, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
            .expect("matrix construction should succeed");
        let result = MultifrontalLU::new(&a);
        assert!(matches!(result, Err(MultifrontalError::NotSquare { .. })));
    }

    #[test]
    fn test_multifrontal_lu_empty() {
        let a = CscMatrix::<f64>::new(0, 0, vec![0], vec![], vec![])
            .expect("empty matrix construction should succeed");
        let lu = MultifrontalLU::new(&a).expect("empty LU should succeed");
        assert_eq!(lu.n(), 0);
    }

    #[test]
    fn test_multifrontal_lu_determinant() {
        let a = make_general_3x3();
        let lu = MultifrontalLU::new(&a).expect("LU should succeed");
        let det = lu.determinant();

        // det([2,1,0; 1,3,1; 0,1,2]) = 2*(6-1) - 1*(2-0) + 0 = 10 - 2 = 8
        assert!(
            (det.abs() - 8.0).abs() < 1e-9,
            "Determinant: got {}, expected +/-8.0",
            det
        );
    }

    #[test]
    fn test_multifrontal_lu_vs_direct() {
        let a = make_general_5x5();
        let mf_lu = MultifrontalLU::new(&a).expect("multifrontal LU should succeed");
        let direct_lu = super::super::SparseLU::new(&a).expect("direct LU should succeed");

        let b = vec![2.0, -1.0, 3.0, -2.0, 1.0];

        let x_mf = mf_lu.solve(&b);
        let x_direct = direct_lu.solve(&b);

        for i in 0..5 {
            assert!(
                (x_mf[i] - x_direct[i]).abs() < 1e-8,
                "Multifrontal vs direct LU differ at {}: {} vs {}",
                i,
                x_mf[i],
                x_direct[i]
            );
        }
    }
}
