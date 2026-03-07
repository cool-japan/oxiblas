//! Multifrontal Cholesky factorization for symmetric positive definite sparse matrices.
//!
//! The multifrontal method works bottom-up through the elimination tree:
//! 1. For each leaf node, assemble a frontal matrix from original matrix entries
//! 2. For each internal node, assemble frontal matrix from original entries
//!    plus contribution blocks (update matrices) from children
//! 3. Factor the fully-summed rows/columns of the frontal matrix using dense operations
//! 4. Compute the Schur complement (update matrix) and pass it to the parent
//!
//! This approach naturally uses dense BLAS-3 operations on the frontal matrices,
//! achieving high computational intensity compared to column-by-column methods.

use crate::csc::CscMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for multifrontal factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultifrontalError {
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix is not positive definite.
    NotPositiveDefinite {
        /// Column index where failure occurred.
        index: usize,
    },
    /// Zero pivot encountered.
    ZeroPivot {
        /// Column index of zero pivot.
        index: usize,
    },
    /// Internal error in assembly.
    AssemblyError {
        /// Description of the error.
        message: String,
    },
}

impl core::fmt::Display for MultifrontalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::NotPositiveDefinite { index } => {
                write!(f, "Matrix is not positive definite at column {index}")
            }
            Self::ZeroPivot { index } => {
                write!(f, "Zero pivot at column {index}")
            }
            Self::AssemblyError { message } => {
                write!(f, "Assembly error: {message}")
            }
        }
    }
}

impl std::error::Error for MultifrontalError {}

/// An update matrix (contribution block / Schur complement) passed from child to parent.
///
/// Stored as a dense symmetric matrix with associated global row/column indices.
#[derive(Debug, Clone)]
struct UpdateMatrix<T: Scalar> {
    /// Global row/column indices for the update matrix.
    indices: Vec<usize>,
    /// Dense symmetric data stored in column-major lower triangle.
    /// Size: indices.len() * indices.len()
    data: Vec<T>,
}

/// A frontal matrix assembled at a node of the elimination tree.
///
/// The frontal matrix has the structure:
/// ```text
///   [ F11  F12^T ]   fully-summed (size x size)
///   [ F21  F22   ]   update part  (update_size x update_size)
/// ```
/// where F11 is factored, F21 becomes part of L, and F22 - F21*F11^{-1}*F21^T
/// is the Schur complement passed to the parent.
#[derive(Debug)]
struct FrontalMatrix<T: Scalar> {
    /// All global indices: first `num_fully_summed` are pivots, rest are update indices.
    indices: Vec<usize>,
    /// Number of fully-summed (pivot) variables.
    num_fully_summed: usize,
    /// Dense frontal matrix data in column-major order.
    /// Total size: total_size * total_size where total_size = indices.len()
    data: Vec<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> FrontalMatrix<T> {
    /// Creates a new frontal matrix with the given indices and fully-summed count.
    fn new(indices: Vec<usize>, num_fully_summed: usize) -> Self {
        let total = indices.len();
        Self {
            indices,
            num_fully_summed,
            data: vec![T::zero(); total * total],
        }
    }

    /// Returns the total size of the frontal matrix.
    fn total_size(&self) -> usize {
        self.indices.len()
    }

    /// Gets a mutable reference to element (i, j) in column-major storage.
    fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        let total = self.total_size();
        &mut self.data[j * total + i]
    }

    /// Gets element (i, j) in column-major storage.
    fn get(&self, i: usize, j: usize) -> &T {
        let total = self.total_size();
        &self.data[j * total + i]
    }

    /// Assembles original matrix entries into the frontal matrix.
    ///
    /// For Cholesky (symmetric), loads entries from column `node` of A.
    /// Only entries where both row and column are in the frontal index set
    /// are assembled. Since A is symmetric, we symmetrize: an entry A[i,j]
    /// from column j (with i in frontal) fills both (local_i, local_j) and
    /// (local_j, local_i).
    fn assemble_original(&mut self, a: &CscMatrix<T>, global_cols: &[usize]) {
        // Build local index lookup: global -> local position
        let mut global_to_local = std::collections::HashMap::new();
        for (local, &global) in self.indices.iter().enumerate() {
            global_to_local.insert(global, local);
        }

        for &col in global_cols {
            let local_j = match global_to_local.get(&col) {
                Some(&lj) => lj,
                None => continue,
            };

            let col_start = a.col_ptrs()[col];
            let col_end = a.col_ptrs()[col + 1];

            for idx in col_start..col_end {
                let row = a.row_indices()[idx];
                if let Some(&local_i) = global_to_local.get(&row) {
                    let val = a.values()[idx].clone();
                    // Place the entry at (local_i, local_j)
                    let current = self.get(local_i, local_j).clone();
                    *self.get_mut(local_i, local_j) = current + val.clone();
                    // For symmetric assembly, also place at (local_j, local_i)
                    if local_i != local_j {
                        let current_sym = self.get(local_j, local_i).clone();
                        *self.get_mut(local_j, local_i) = current_sym + val;
                    }
                }
            }
        }
    }

    /// Assembles an update matrix (contribution block) from a child into this frontal.
    fn assemble_update(&mut self, update: &UpdateMatrix<T>) {
        let mut global_to_local = std::collections::HashMap::new();
        for (local, &global) in self.indices.iter().enumerate() {
            global_to_local.insert(global, local);
        }

        let update_size = update.indices.len();
        for uj in 0..update_size {
            let global_j = update.indices[uj];
            let local_j = match global_to_local.get(&global_j) {
                Some(&lj) => lj,
                None => continue,
            };

            for ui in uj..update_size {
                let global_i = update.indices[ui];
                let local_i = match global_to_local.get(&global_i) {
                    Some(&li) => li,
                    None => continue,
                };

                let val = update.data[uj * update_size + ui].clone();
                if Scalar::abs(val.clone()) <= <T as Scalar>::epsilon() {
                    continue;
                }

                // Add to lower triangle
                if local_i >= local_j {
                    let current = self.get(local_i, local_j).clone();
                    *self.get_mut(local_i, local_j) = current + val.clone();
                }
                // Add to upper triangle (symmetric)
                if local_j >= local_i && local_i != local_j {
                    let current = self.get(local_j, local_i).clone();
                    *self.get_mut(local_j, local_i) = current + val;
                }
            }
        }
    }

    /// Factors the fully-summed part and computes the Schur complement.
    ///
    /// After this call:
    /// - The diagonal block F11 contains the Cholesky factor L11
    /// - F21 contains L21 = A21 * L11^{-T}
    /// - Returns the update matrix (Schur complement) F22 - L21 * L21^T
    fn factor_and_schur(
        &mut self,
        first_global_col: usize,
    ) -> Result<Option<UpdateMatrix<T>>, MultifrontalError> {
        let total = self.total_size();
        let ns = self.num_fully_summed;
        let nu = total - ns; // update size

        // Step 1: Dense Cholesky on F11 (the ns x ns diagonal block)
        // Using a left-looking dense Cholesky
        for j in 0..ns {
            // Compute diagonal: L[j,j] = sqrt(F[j,j] - sum_{k<j} L[j,k]^2)
            let mut diag = self.get(j, j).clone();
            for k in 0..j {
                let ljk = self.get(j, k).clone();
                diag = diag - ljk.clone() * ljk;
            }

            if !(diag > T::zero()) {
                return Err(MultifrontalError::NotPositiveDefinite {
                    index: first_global_col + j,
                });
            }

            let ljj = Real::sqrt(diag);
            if Scalar::abs(ljj.clone()) <= <T as Scalar>::epsilon() {
                return Err(MultifrontalError::ZeroPivot {
                    index: first_global_col + j,
                });
            }
            *self.get_mut(j, j) = ljj.clone();

            let ljj_inv = T::one() / ljj;

            // Compute L[i,j] for i in (j+1)..ns (within diagonal block)
            for i in (j + 1)..ns {
                let mut val = self.get(i, j).clone();
                for k in 0..j {
                    let lik = self.get(i, k).clone();
                    let ljk = self.get(j, k).clone();
                    val = val - lik * ljk;
                }
                *self.get_mut(i, j) = val * ljj_inv.clone();
            }

            // Compute L21[i,j] for i in ns..total (off-diagonal block)
            for i in ns..total {
                let mut val = self.get(i, j).clone();
                for k in 0..j {
                    let lik = self.get(i, k).clone();
                    let ljk = self.get(j, k).clone();
                    val = val - lik * ljk;
                }
                *self.get_mut(i, j) = val * ljj_inv.clone();
            }
        }

        // Step 2: Compute Schur complement: S = F22 - L21 * L21^T
        if nu == 0 {
            return Ok(None);
        }

        let update_indices: Vec<usize> = self.indices[ns..].to_vec();
        let mut update_data = vec![T::zero(); nu * nu];

        // S[i,j] = F22[i,j] - sum_{k=0..ns} L21[i+ns,k] * L21[j+ns,k]
        for uj in 0..nu {
            for ui in uj..nu {
                let fi = ui + ns;
                let fj = uj + ns;
                let mut val = self.get(fi, fj).clone();
                for k in 0..ns {
                    let l21_ik = self.get(fi, k).clone();
                    let l21_jk = self.get(fj, k).clone();
                    val = val - l21_ik * l21_jk;
                }
                // Store in column-major lower triangle
                update_data[uj * nu + ui] = val;
            }
        }

        Ok(Some(UpdateMatrix {
            indices: update_indices,
            data: update_data,
        }))
    }
}

/// Multifrontal Cholesky factorization for symmetric positive definite matrices.
///
/// The multifrontal method processes the elimination tree bottom-up, assembling
/// frontal matrices at each node from original entries and child contributions,
/// then factoring using dense BLAS-3 operations.
///
/// # Algorithm
///
/// 1. **Symbolic analysis**: Build elimination tree, compute postorder, determine
///    structure of frontal matrices
/// 2. **Numeric factorization** (bottom-up through elimination tree):
///    - Assemble frontal matrix from original entries + child update matrices
///    - Factor fully-summed variables using dense Cholesky
///    - Compute Schur complement (update matrix)
///    - Pass update matrix to parent node
/// 3. **Solve**: Forward/backward substitution through the elimination tree
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::MultifrontalCholesky;
/// let chol = MultifrontalCholesky::new(&a)?;
/// let x = chol.solve(&b);
/// ```
#[derive(Debug, Clone)]
pub struct MultifrontalCholesky<T: Scalar> {
    /// Matrix dimension.
    n: usize,
    /// Elimination tree parent pointers (None for roots).
    etree_parent: Vec<Option<usize>>,
    /// Children lists for elimination tree.
    #[allow(dead_code)]
    etree_children: Vec<Vec<usize>>,
    /// Postorder traversal of the elimination tree.
    #[allow(dead_code)]
    postorder: Vec<usize>,
    /// For each column j, the row indices of L below the diagonal (sorted).
    l_row_indices: Vec<Vec<usize>>,
    /// Dense L factor storage: for column j, stores [L[j,j], L[sub_rows[0],j], ...]
    l_values: Vec<Vec<T>>,
    /// Fill-reducing permutation.
    perm: Vec<usize>,
    /// Inverse permutation.
    perm_inv: Vec<usize>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> MultifrontalCholesky<T> {
    /// Computes the multifrontal Cholesky factorization.
    ///
    /// Uses AMD ordering for fill reduction, then performs bottom-up
    /// multifrontal factorization through the elimination tree.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive definite matrix in CSC format
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or not positive definite.
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

        // Permute the matrix: P * A * P^T
        let ap = permute_symmetric_csc(a, &perm, &perm_inv);

        // Build elimination tree
        let etree_parent = build_elimination_tree(&ap);
        let mut etree_children = vec![Vec::new(); n];
        for (j, parent) in etree_parent.iter().enumerate() {
            if let Some(p) = parent {
                etree_children[*p].push(j);
            }
        }

        // Compute postorder (leaves first, roots last)
        let postorder = compute_postorder(&etree_parent, &etree_children, n);

        // Symbolic analysis: determine structure of each column of L
        let l_struct = symbolic_factorization(&ap, &etree_parent);

        // Numeric factorization using multifrontal method
        let (l_row_indices, l_values) = Self::multifrontal_factorize(
            &ap,
            &etree_parent,
            &etree_children,
            &postorder,
            &l_struct,
        )?;

        Ok(Self {
            n,
            etree_parent,
            etree_children,
            postorder,
            l_row_indices,
            l_values,
            perm,
            perm_inv,
        })
    }

    /// Performs the multifrontal numeric factorization.
    ///
    /// Processes nodes in postorder (leaves first), assembling frontal matrices
    /// and passing update matrices up the elimination tree.
    fn multifrontal_factorize(
        a: &CscMatrix<T>,
        _etree_parent: &[Option<usize>],
        etree_children: &[Vec<usize>],
        postorder: &[usize],
        l_struct: &[Vec<usize>],
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<T>>), MultifrontalError> {
        let n = a.nrows();
        let mut l_row_indices: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut l_values: Vec<Vec<T>> = vec![Vec::new(); n];

        // Storage for pending update matrices from children
        // Each node may have an update matrix waiting to be assembled into parent
        let mut pending_updates: Vec<Option<UpdateMatrix<T>>> = vec![None; n];

        // Process nodes in postorder (leaves first)
        for &node in postorder {
            // Determine the frontal matrix structure for this node.
            // The frontal indices are: {node} union l_struct[node]
            let mut frontal_indices = vec![node];
            frontal_indices.extend_from_slice(&l_struct[node]);

            // Also include indices from children's update matrices that are not yet present
            for &child in &etree_children[node] {
                if let Some(ref update) = pending_updates[child] {
                    for &idx in &update.indices {
                        if idx != node && !frontal_indices.contains(&idx) {
                            frontal_indices.push(idx);
                        }
                    }
                }
            }

            // Sort indices, keeping 'node' first (it's the fully-summed variable)
            let mut update_indices: Vec<usize> = frontal_indices[1..].to_vec();
            update_indices.sort_unstable();
            update_indices.dedup();

            frontal_indices = vec![node];
            frontal_indices.extend_from_slice(&update_indices);

            // Determine how many variables are fully summed at this node.
            // In the basic (non-supernode) multifrontal method, each node
            // has exactly one fully-summed variable.
            let num_fully_summed = 1;

            // Create frontal matrix
            let mut frontal = FrontalMatrix::new(frontal_indices.clone(), num_fully_summed);

            // Assemble original matrix entries for this column
            frontal.assemble_original(a, &[node]);

            // Assemble update matrices from children
            for &child in &etree_children[node] {
                if let Some(update) = pending_updates[child].take() {
                    frontal.assemble_update(&update);
                }
            }

            // Factor the fully-summed part and compute Schur complement
            let update = frontal.factor_and_schur(node)?;

            // Extract L column from the factored frontal matrix
            // L[node, node] = frontal[0, 0] (diagonal)
            // L[frontal_indices[i], node] = frontal[i, 0] for i > 0
            let total = frontal.total_size();
            let mut col_rows = Vec::with_capacity(total);
            let mut col_vals = Vec::with_capacity(total);

            // Diagonal
            col_vals.push(frontal.get(0, 0).clone());

            // Off-diagonal entries
            for i in 1..total {
                let val = frontal.get(i, 0).clone();
                if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                    col_rows.push(frontal_indices[i]);
                    col_vals.push(val);
                }
            }

            l_row_indices[node] = col_rows;
            l_values[node] = col_vals;

            // Store update matrix at this node (to be consumed by parent)
            if let Some(u) = update {
                pending_updates[node] = Some(u);
            }
        }

        Ok((l_row_indices, l_values))
    }

    /// Solves A * x = b using the multifrontal Cholesky factorization.
    ///
    /// Performs forward substitution (L * y = P * b) then
    /// backward substitution (L^T * z = y), then applies P^T.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let n = self.n;
        assert_eq!(b.len(), n, "RHS length must match matrix dimension");

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

    /// Forward substitution: L * y = b.
    fn forward_solve(&self, b: &[T]) -> Vec<T> {
        let mut x = b.to_vec();

        for j in 0..self.n {
            if self.l_values[j].is_empty() {
                continue;
            }

            // L[j,j] is the first element
            let ljj = self.l_values[j][0].clone();
            x[j] = x[j].clone() / ljj;

            // Update: x[row] -= L[row,j] * x[j]
            for (k, &row) in self.l_row_indices[j].iter().enumerate() {
                let lij = self.l_values[j][k + 1].clone();
                x[row] = x[row].clone() - lij * x[j].clone();
            }
        }

        x
    }

    /// Backward substitution: L^T * z = y.
    fn backward_solve(&self, b: &[T]) -> Vec<T> {
        let mut x = b.to_vec();

        for j in (0..self.n).rev() {
            if self.l_values[j].is_empty() {
                continue;
            }

            // Update: x[j] -= L[row,j]^T * x[row] for rows below diagonal
            for (k, &row) in self.l_row_indices[j].iter().enumerate() {
                let lij = self.l_values[j][k + 1].clone();
                x[j] = x[j].clone() - lij * x[row].clone();
            }

            // L[j,j] is the first element
            let ljj = self.l_values[j][0].clone();
            x[j] = x[j].clone() / ljj;
        }

        x
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the fill-reducing permutation.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the inverse permutation.
    pub fn perm_inv(&self) -> &[usize] {
        &self.perm_inv
    }

    /// Returns the elimination tree parent pointers.
    pub fn etree_parent(&self) -> &[Option<usize>] {
        &self.etree_parent
    }

    /// Returns the number of nonzeros in L (including diagonal).
    pub fn nnz_l(&self) -> usize {
        let mut nnz = 0usize;
        for j in 0..self.n {
            nnz += self.l_values[j].len();
        }
        nnz
    }

    /// Computes the log determinant of A.
    ///
    /// Since A = L * L^T, log(det(A)) = 2 * sum(log(L\[j,j\])).
    pub fn log_determinant(&self) -> T {
        let mut log_det = T::zero();
        for j in 0..self.n {
            if !self.l_values[j].is_empty() {
                let ljj = self.l_values[j][0].clone();
                log_det = log_det + Real::ln(ljj);
            }
        }
        log_det + log_det
    }

    /// Extracts the lower triangular factor L as a CSC matrix.
    pub fn l_csc(&self) -> CscMatrix<T> {
        let n = self.n;
        let mut col_ptrs = vec![0usize; n + 1];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        for j in 0..n {
            if self.l_values[j].is_empty() {
                col_ptrs[j + 1] = col_ptrs[j];
                continue;
            }

            // Diagonal entry
            row_indices.push(j);
            values.push(self.l_values[j][0].clone());

            // Off-diagonal entries (already sorted)
            for (k, &row) in self.l_row_indices[j].iter().enumerate() {
                row_indices.push(row);
                values.push(self.l_values[j][k + 1].clone());
            }

            col_ptrs[j + 1] = values.len();
        }

        // Safety: we build valid CSC structure with sorted row indices per column
        unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
    }
}

/// Builds the elimination tree from a symmetric matrix (lower triangle).
fn build_elimination_tree<T: Scalar>(a: &CscMatrix<T>) -> Vec<Option<usize>> {
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
                // Walk up the tree, path-compressing
                let mut r = i;
                while ancestor[r] != r && ancestor[r] != k {
                    let next = ancestor[r];
                    ancestor[r] = k;
                    r = next;
                }
                if ancestor[r] == r {
                    parent[r] = Some(k);
                }
                // Path compression for i
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

/// Computes a postorder traversal of the elimination tree (leaves first).
fn compute_postorder(_parent: &[Option<usize>], children: &[Vec<usize>], n: usize) -> Vec<usize> {
    // Find roots (nodes with no parent)
    let roots: Vec<usize> = (0..n).filter(|&i| _parent[i].is_none()).collect();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, bool)> = Vec::new();

    for &root in &roots {
        stack.push((root, false));
    }

    let mut visited = vec![false; n];
    while let Some((node, processed)) = stack.pop() {
        if processed {
            order.push(node);
        } else if !visited[node] {
            visited[node] = true;
            stack.push((node, true));
            // Push children in reverse order so they're processed left-to-right
            for &child in children[node].iter().rev() {
                if !visited[child] {
                    stack.push((child, false));
                }
            }
        }
    }

    order
}

/// Symbolic factorization: determine the sparsity structure of each column of L.
///
/// For each column j, l_struct[j] contains the sorted row indices of L below the diagonal.
fn symbolic_factorization<T: Scalar>(
    a: &CscMatrix<T>,
    parent: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut l_struct: Vec<Vec<usize>> = vec![Vec::new(); n];

    // For each column j, the structure of L[:,j] below diagonal is the
    // union of:
    // 1. Row indices from A[:,j] that are > j
    // 2. For each child c of j in etree, the row indices of L[:,c] that are > j

    // Process in natural order (0..n)
    for j in 0..n {
        let mut row_set = std::collections::BTreeSet::new();

        // Add entries from A
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let row = a.row_indices()[idx];
            if row > j {
                row_set.insert(row);
            }
        }

        // Propagate structure from children via elimination tree
        // For each row r in l_struct[j], if parent[r] exists and > j, add it
        // This is a simplified column count / reachability
        let rows: Vec<usize> = row_set.iter().copied().collect();
        for &r in &rows {
            // Walk up the elimination tree from r
            let mut current = r;
            while let Some(p) = parent[current] {
                if p <= j {
                    break;
                }
                if !row_set.contains(&p) {
                    row_set.insert(p);
                }
                current = p;
            }
        }

        l_struct[j] = row_set.into_iter().collect();
    }

    l_struct
}

/// Permutes a symmetric matrix: returns P * A * P^T.
///
/// Stores the full symmetric matrix (both triangles) in the result.
fn permute_symmetric_csc<T: Scalar + Clone + Field>(
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

        // Get entries from column old_j of A
        let start = a.col_ptrs()[old_j];
        let end = a.col_ptrs()[old_j + 1];

        for idx in start..end {
            let old_i = a.row_indices()[idx];
            let new_i = perm_inv[old_i];
            entries.push((new_i, a.values()[idx].clone()));
        }

        // Also pick up entries from other columns that have row = old_j
        // (to handle the upper triangle mapping to lower)
        for old_k in 0..n {
            if old_k == old_j {
                continue;
            }
            let new_k = perm_inv[old_k];
            let k_start = a.col_ptrs()[old_k];
            let k_end = a.col_ptrs()[old_k + 1];

            for idx in k_start..k_end {
                if a.row_indices()[idx] == old_j {
                    entries.push((new_k, a.values()[idx].clone()));
                    break;
                }
            }
        }

        // Sort by row index and deduplicate
        entries.sort_by_key(|(row, _)| *row);
        entries.dedup_by_key(|(row, _)| *row);

        for (row, val) in entries {
            row_indices.push(row);
            values.push(val);
        }

        col_ptrs[new_j + 1] = values.len();
    }

    // Safety: we sorted row indices per column
    unsafe { CscMatrix::new_unchecked(n, n, col_ptrs, row_indices, values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a 3x3 SPD tridiagonal matrix.
    fn make_spd_3x3() -> CscMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];
        CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("valid 3x3 SPD matrix construction")
    }

    /// Creates a 5x5 SPD tridiagonal Laplacian-like matrix.
    fn make_spd_5x5() -> CscMatrix<f64> {
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

        CscMatrix::new(n, n, col_ptrs, row_indices, values)
            .expect("valid 5x5 SPD matrix construction")
    }

    /// Creates a 1x1 SPD matrix.
    fn make_spd_1x1() -> CscMatrix<f64> {
        CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![5.0]).expect("valid 1x1 matrix construction")
    }

    /// Creates a 2x2 SPD matrix.
    fn make_spd_2x2() -> CscMatrix<f64> {
        // A = [4 1]
        //     [1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0];
        let row_indices = vec![0, 1, 0, 1];
        let col_ptrs = vec![0, 2, 4];
        CscMatrix::new(2, 2, col_ptrs, row_indices, values)
            .expect("valid 2x2 SPD matrix construction")
    }

    /// Creates an 8x8 2D Laplacian SPD matrix (banded).
    fn make_spd_laplacian_8x8() -> CscMatrix<f64> {
        let n = 8;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0usize];

        for j in 0..n {
            let mut entries: Vec<(usize, f64)> = Vec::new();

            // Bandwidth-2 Laplacian: connections at j-2, j-1, j, j+1, j+2
            if j >= 2 {
                entries.push((j - 2, -0.5));
            }
            if j >= 1 {
                entries.push((j - 1, -1.0));
            }
            entries.push((j, 6.0)); // Strongly diagonally dominant
            if j + 1 < n {
                entries.push((j + 1, -1.0));
            }
            if j + 2 < n {
                entries.push((j + 2, -0.5));
            }

            entries.sort_by_key(|(r, _)| *r);
            for (r, v) in entries {
                row_indices.push(r);
                values.push(v);
            }
            col_ptrs.push(values.len());
        }

        CscMatrix::new(n, n, col_ptrs, row_indices, values)
            .expect("valid 8x8 Laplacian matrix construction")
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
    fn test_multifrontal_cholesky_1x1() {
        let a = make_spd_1x1();
        let chol =
            MultifrontalCholesky::new(&a).expect("factorization of 1x1 SPD matrix should succeed");

        let b = vec![10.0];
        let x = chol.solve(&b);

        assert!(
            (x[0] - 2.0).abs() < 1e-10,
            "1x1 solve: expected 2.0, got {}",
            x[0]
        );
    }

    #[test]
    fn test_multifrontal_cholesky_2x2() {
        let a = make_spd_2x2();
        let chol =
            MultifrontalCholesky::new(&a).expect("factorization of 2x2 SPD matrix should succeed");

        let b = vec![5.0, 5.0];
        let x = chol.solve(&b);

        let ax = csc_matvec(&a, &x);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "2x2 residual at {}: {} vs {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_cholesky_3x3() {
        let a = make_spd_3x3();
        let chol =
            MultifrontalCholesky::new(&a).expect("factorization of 3x3 SPD matrix should succeed");

        let b = vec![1.0, 2.0, 3.0];
        let x = chol.solve(&b);

        let ax = csc_matvec(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "3x3 residual at {}: {} vs {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_cholesky_5x5() {
        let a = make_spd_5x5();
        let chol =
            MultifrontalCholesky::new(&a).expect("factorization of 5x5 SPD matrix should succeed");

        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let x = chol.solve(&b);

        let ax = csc_matvec(&a, &x);
        for i in 0..5 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-9,
                "5x5 residual at {}: {} vs {}",
                i,
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_cholesky_8x8_laplacian() {
        let a = make_spd_laplacian_8x8();
        let chol =
            MultifrontalCholesky::new(&a).expect("factorization of 8x8 Laplacian should succeed");

        let b = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let x = chol.solve(&b);

        let ax = csc_matvec(&a, &x);
        let residual: f64 = (0..8).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            residual / b_norm < 1e-10,
            "8x8 relative residual too large: {}",
            residual / b_norm
        );
    }

    #[test]
    fn test_multifrontal_cholesky_identity() {
        let a = CscMatrix::<f64>::eye(5);
        let chol = MultifrontalCholesky::new(&a).expect("factorization of identity should succeed");

        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = chol.solve(&b);

        for i in 0..5 {
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
    fn test_multifrontal_cholesky_not_spd() {
        // Negative definite matrix
        let values = vec![-4.0, 1.0, 1.0, -4.0, 1.0, 1.0, -4.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let col_ptrs = vec![0, 2, 5, 7];
        let a = CscMatrix::new(3, 3, col_ptrs, row_indices, values)
            .expect("matrix construction should succeed");

        let result = MultifrontalCholesky::new(&a);
        assert!(
            result.is_err(),
            "Negative definite matrix should fail factorization"
        );
    }

    #[test]
    fn test_multifrontal_cholesky_not_square() {
        let a = CscMatrix::new(3, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0])
            .expect("matrix construction should succeed");
        let result = MultifrontalCholesky::new(&a);
        assert!(matches!(result, Err(MultifrontalError::NotSquare { .. })));
    }

    #[test]
    fn test_multifrontal_cholesky_log_determinant() {
        let a = make_spd_3x3();
        let chol = MultifrontalCholesky::new(&a).expect("factorization should succeed");

        let log_det = chol.log_determinant();

        // det([4,1,0; 1,4,1; 0,1,4]) = 4*(16-1) - 1*(4-0) = 60 - 4 = 56
        let expected_log_det = 56.0f64.ln();

        assert!(
            (log_det - expected_log_det).abs() < 1e-9,
            "Log determinant: got {}, expected {}",
            log_det,
            expected_log_det
        );
    }

    #[test]
    fn test_multifrontal_cholesky_nnz() {
        let a = make_spd_3x3();
        let chol = MultifrontalCholesky::new(&a).expect("factorization should succeed");

        let nnz = chol.nnz_l();
        // L for tridiagonal 3x3 has at most 5 nonzeros (3 diag + 2 off-diag)
        assert!(nnz >= 3, "L should have at least 3 diagonal entries");
        assert!(nnz <= 6, "L should have at most 6 entries for 3x3 tridiag");
    }

    #[test]
    fn test_multifrontal_cholesky_l_reconstruction() {
        let a = make_spd_3x3();
        let chol = MultifrontalCholesky::new(&a).expect("factorization should succeed");

        let l = chol.l_csc();

        // Compute L * L^T and verify it approximates P * A * P^T
        let n = 3;
        let mut llt = vec![vec![0.0; n]; n];

        for j in 0..n {
            let col_start = l.col_ptrs()[j];
            let col_end = l.col_ptrs()[j + 1];

            for idx_i in col_start..col_end {
                let i = l.row_indices()[idx_i];
                let l_ij = l.values()[idx_i];

                for idx_k in col_start..col_end {
                    let k = l.row_indices()[idx_k];
                    let l_kj = l.values()[idx_k];
                    llt[i][k] += l_ij * l_kj;
                }
            }
        }

        // Reconstruct original A from L*L^T with permutation
        // P^T * L * L^T * P should equal A
        let perm = chol.perm();
        let mut reconstructed = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                reconstructed[perm[i]][perm[j]] = llt[i][j];
            }
        }

        // Compare with original A (densified)
        let mut a_dense = vec![vec![0.0; n]; n];
        for col in 0..n {
            let start = a.col_ptrs()[col];
            let end = a.col_ptrs()[col + 1];
            for idx in start..end {
                let row = a.row_indices()[idx];
                a_dense[row][col] = a.values()[idx];
            }
        }

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (reconstructed[i][j] - a_dense[i][j]).abs() < 1e-10,
                    "L*L^T reconstruction failed at ({},{}): {} vs {}",
                    i,
                    j,
                    reconstructed[i][j],
                    a_dense[i][j]
                );
            }
        }
    }

    #[test]
    fn test_multifrontal_vs_direct_cholesky() {
        let a = make_spd_5x5();
        let mf_chol =
            MultifrontalCholesky::new(&a).expect("multifrontal factorization should succeed");
        let direct_chol =
            super::super::SparseCholesky::new(&a).expect("direct cholesky should succeed");

        let b = vec![2.0, -1.0, 3.0, -2.0, 1.0];

        let x_mf = mf_chol.solve(&b);
        let x_direct = direct_chol.solve(&b);

        for i in 0..5 {
            assert!(
                (x_mf[i] - x_direct[i]).abs() < 1e-9,
                "Multifrontal vs direct differ at {}: {} vs {}",
                i,
                x_mf[i],
                x_direct[i]
            );
        }
    }

    #[test]
    fn test_multifrontal_cholesky_empty() {
        let a = CscMatrix::<f64>::new(0, 0, vec![0], vec![], vec![])
            .expect("empty matrix construction should succeed");
        let chol =
            MultifrontalCholesky::new(&a).expect("empty matrix factorization should succeed");
        assert_eq!(chol.n(), 0);
        assert_eq!(chol.nnz_l(), 0);
    }
}
