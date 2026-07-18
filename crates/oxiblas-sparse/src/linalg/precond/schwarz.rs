//! Additive Schwarz domain decomposition preconditioner.

use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Local solver type for subdomain problems in Additive Schwarz.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum LocalSolverType {
    /// Genuinely exact solve of the local subdomain block: a dense LU
    /// factorization with partial pivoting and full fill-in (no dropping).
    /// This solves `A_local x = b` to floating-point precision, unlike
    /// [`LocalSolverType::ILU0`], which only approximates it.
    ExactLU,
    /// Approximate solve using an incomplete LU(0) factorization: fill-in
    /// is restricted to the sparsity pattern of the local block, which is
    /// cheaper than [`LocalSolverType::ExactLU`] but not an exact solve.
    #[default]
    ILU0,
    /// Jacobi iteration (diagonal scaling only).
    Jacobi,
}

/// Configuration for Additive Schwarz preconditioner.
#[derive(Debug, Clone)]
pub struct AdditiveSchwarzConfig {
    /// Number of subdomains (default: 4).
    pub num_subdomains: usize,
    /// Overlap level between subdomains (default: 1).
    pub overlap: usize,
    /// Type of local solver (default: ILU0).
    pub local_solver: LocalSolverType,
}

impl Default for AdditiveSchwarzConfig {
    fn default() -> Self {
        Self {
            num_subdomains: 4,
            overlap: 1,
            local_solver: LocalSolverType::ILU0,
        }
    }
}

/// Local subdomain data for Additive Schwarz.
#[derive(Debug, Clone)]
struct Subdomain<T: Scalar> {
    /// Indices of unknowns in this subdomain (with overlap).
    indices: Vec<usize>,
    /// Mapping from global index to local index.
    #[allow(dead_code)]
    global_to_local: Vec<usize>,
    /// Local matrix diagonal for Jacobi solver.
    diag_inv: Vec<T>,
    /// Local ILU factors (L values).
    ilu_l_values: Vec<T>,
    ilu_l_col_indices: Vec<usize>,
    ilu_l_row_ptrs: Vec<usize>,
    /// Local ILU factors (U values).
    ilu_u_values: Vec<T>,
    ilu_u_col_indices: Vec<usize>,
    ilu_u_row_ptrs: Vec<usize>,
    /// Dense exact LU factors (row-major `local_n x local_n`), used only by
    /// [`LocalSolverType::ExactLU`]. Strictly-lower entries store the unit
    /// lower-triangular factor `L` (implicit unit diagonal); on-and-above
    /// diagonal entries store the upper-triangular factor `U`.
    exact_lu_values: Vec<T>,
    /// Row permutation from partial pivoting for the exact LU factorization:
    /// `exact_lu_piv[i]` is the original local row moved into position `i`.
    exact_lu_piv: Vec<usize>,
}

/// Additive Schwarz domain decomposition preconditioner.
///
/// The Additive Schwarz method partitions the domain into overlapping subdomains
/// and solves local problems independently, then combines the results.
///
/// For a partition into k subdomains, the preconditioner is:
///
/// M^{-1} = Σ_i R_i^T A_i^{-1} R_i
///
/// where R_i is the restriction operator and A_i is the local matrix.
///
/// # Advantages
///
/// - Highly parallelizable (local solves are independent)
/// - Scalable to large problems
/// - Can be used as a coarse preconditioner
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::{AdditiveSchwarz, AdditiveSchwarzConfig};
///
/// let config = AdditiveSchwarzConfig::default();
/// let schwarz = AdditiveSchwarz::new(&matrix, config)?;
/// let mut z = vec![0.0; n];
/// schwarz.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct AdditiveSchwarz<T: Scalar> {
    /// Subdomains with their local data.
    subdomains: Vec<Subdomain<T>>,
    /// Matrix dimension.
    n: usize,
    /// Local solver type.
    local_solver: LocalSolverType,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd + Real> AdditiveSchwarz<T> {
    /// Create a new Additive Schwarz preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition (CSR format).
    /// * `config` - Configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or if local factorization fails.
    pub fn new(
        a: &CsrMatrix<T>,
        config: AdditiveSchwarzConfig,
    ) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let n = a.nrows();

        if n == 0 {
            return Ok(Self {
                subdomains: Vec::new(),
                n: 0,
                local_solver: config.local_solver,
            });
        }

        // Partition unknowns into subdomains
        let num_subdomains = config.num_subdomains.max(1).min(n);
        let base_size = n / num_subdomains;
        let remainder = n % num_subdomains;

        // Create initial non-overlapping partition
        let mut partition: Vec<Vec<usize>> = Vec::with_capacity(num_subdomains);
        let mut start = 0;
        for i in 0..num_subdomains {
            let size = base_size + if i < remainder { 1 } else { 0 };
            let indices: Vec<usize> = (start..start + size).collect();
            partition.push(indices);
            start += size;
        }

        // Add overlap to each subdomain
        let overlap = config.overlap;
        let mut extended_partition: Vec<Vec<usize>> = Vec::with_capacity(num_subdomains);

        for base_indices in partition.iter() {
            let mut extended: Vec<usize> = base_indices.clone();

            // Extend overlap by finding neighbors in the graph
            for _ in 0..overlap {
                let mut new_neighbors: Vec<usize> = Vec::new();
                for &idx in &extended {
                    // Add all neighbors of idx
                    let start = a.row_ptrs()[idx];
                    let end = a.row_ptrs()[idx + 1];
                    for ptr in start..end {
                        let neighbor = a.col_indices()[ptr];
                        if !extended.contains(&neighbor) && !new_neighbors.contains(&neighbor) {
                            new_neighbors.push(neighbor);
                        }
                    }
                }
                extended.extend(new_neighbors);
            }

            extended.sort_unstable();
            extended.dedup();
            extended_partition.push(extended);
        }

        // Build subdomains with local solvers
        let mut subdomains = Vec::with_capacity(num_subdomains);

        for indices in extended_partition {
            let local_n = indices.len();
            if local_n == 0 {
                continue;
            }

            // Build global to local mapping
            let mut global_to_local = vec![usize::MAX; n];
            for (local, &global) in indices.iter().enumerate() {
                global_to_local[global] = local;
            }

            // Extract local matrix
            let mut local_values = Vec::new();
            let mut local_col_indices = Vec::new();
            let mut local_row_ptrs = vec![0];

            for &global_row in &indices {
                let start = a.row_ptrs()[global_row];
                let end = a.row_ptrs()[global_row + 1];

                for ptr in start..end {
                    let global_col = a.col_indices()[ptr];
                    let local_col = global_to_local[global_col];
                    if local_col != usize::MAX {
                        local_col_indices.push(local_col);
                        local_values.push(a.values()[ptr].clone());
                    }
                }
                local_row_ptrs.push(local_values.len());
            }

            // Build local solver data
            let subdomain = match config.local_solver {
                LocalSolverType::Jacobi => {
                    // Extract diagonal for Jacobi
                    let mut diag_inv = vec![T::one(); local_n];
                    for local_row in 0..local_n {
                        let start = local_row_ptrs[local_row];
                        let end = local_row_ptrs[local_row + 1];
                        for ptr in start..end {
                            if local_col_indices[ptr] == local_row {
                                let diag = local_values[ptr].clone();
                                if Scalar::abs(diag.clone())
                                    > T::from_f64(1e-14).unwrap_or(T::zero())
                                {
                                    diag_inv[local_row] = T::one() / diag;
                                }
                                break;
                            }
                        }
                    }
                    Subdomain {
                        indices,
                        global_to_local,
                        diag_inv,
                        ilu_l_values: Vec::new(),
                        ilu_l_col_indices: Vec::new(),
                        ilu_l_row_ptrs: Vec::new(),
                        ilu_u_values: Vec::new(),
                        ilu_u_col_indices: Vec::new(),
                        ilu_u_row_ptrs: Vec::new(),
                        exact_lu_values: Vec::new(),
                        exact_lu_piv: Vec::new(),
                    }
                }
                LocalSolverType::ILU0 => {
                    // Compute ILU(0) factorization of local matrix (approximate:
                    // fill-in is restricted to the original sparsity pattern).
                    let (l_values, l_col, l_row, u_values, u_col, u_row) = Self::compute_ilu0(
                        &local_values,
                        &local_col_indices,
                        &local_row_ptrs,
                        local_n,
                    );

                    Subdomain {
                        indices,
                        global_to_local,
                        diag_inv: Vec::new(),
                        ilu_l_values: l_values,
                        ilu_l_col_indices: l_col,
                        ilu_l_row_ptrs: l_row,
                        ilu_u_values: u_values,
                        ilu_u_col_indices: u_col,
                        ilu_u_row_ptrs: u_row,
                        exact_lu_values: Vec::new(),
                        exact_lu_piv: Vec::new(),
                    }
                }
                LocalSolverType::ExactLU => {
                    // Compute a genuinely exact dense LU factorization (with
                    // partial pivoting and full fill-in, no dropping) of the
                    // local subdomain block.
                    let (exact_lu_values, exact_lu_piv) = Self::compute_exact_lu(
                        &local_values,
                        &local_col_indices,
                        &local_row_ptrs,
                        local_n,
                    )?;

                    Subdomain {
                        indices,
                        global_to_local,
                        diag_inv: Vec::new(),
                        ilu_l_values: Vec::new(),
                        ilu_l_col_indices: Vec::new(),
                        ilu_l_row_ptrs: Vec::new(),
                        ilu_u_values: Vec::new(),
                        ilu_u_col_indices: Vec::new(),
                        ilu_u_row_ptrs: Vec::new(),
                        exact_lu_values,
                        exact_lu_piv,
                    }
                }
            };

            subdomains.push(subdomain);
        }

        Ok(Self {
            subdomains,
            n,
            local_solver: config.local_solver,
        })
    }

    /// Compute ILU(0) factorization of a local matrix.
    fn compute_ilu0(
        values: &[T],
        col_indices: &[usize],
        row_ptrs: &[usize],
        n: usize,
    ) -> (
        Vec<T>,
        Vec<usize>,
        Vec<usize>,
        Vec<T>,
        Vec<usize>,
        Vec<usize>,
    ) {
        // Build dense working storage for rows
        let mut l_values: Vec<T> = Vec::new();
        let mut l_col_indices: Vec<usize> = Vec::new();
        let mut l_row_ptrs: Vec<usize> = vec![0];
        let mut u_values: Vec<T> = Vec::new();
        let mut u_col_indices: Vec<usize> = Vec::new();
        let mut u_row_ptrs: Vec<usize> = vec![0];

        // Dense working row
        let mut work = vec![T::zero(); n];
        let mut work_idx = vec![false; n];

        for i in 0..n {
            // Load row i into work array
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];

            for ptr in start..end {
                let j = col_indices[ptr];
                work[j] = values[ptr].clone();
                work_idx[j] = true;
            }

            // Eliminate with previous rows
            for k in 0..i {
                if !work_idx[k] {
                    continue;
                }

                // Find U[k,k]
                let mut u_kk = T::zero();
                let u_start = u_row_ptrs[k];
                let u_end = u_row_ptrs[k + 1];
                for ptr in u_start..u_end {
                    if u_col_indices[ptr] == k {
                        u_kk = u_values[ptr].clone();
                        break;
                    }
                }

                if Scalar::abs(u_kk.clone()) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                    continue;
                }

                let l_ik = work[k].clone() / u_kk;
                work[k] = l_ik.clone();

                // Update row i using row k of U
                for ptr in u_start..u_end {
                    let j = u_col_indices[ptr];
                    if j > k && work_idx[j] {
                        work[j] = work[j].clone() - l_ik.clone() * u_values[ptr].clone();
                    }
                }
            }

            // Store L (lower triangular, j < i)
            for j in 0..i {
                if work_idx[j] {
                    let val = work[j].clone();
                    if Scalar::abs(val.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                        l_col_indices.push(j);
                        l_values.push(val);
                    }
                }
            }
            // Add diagonal 1 to L
            l_col_indices.push(i);
            l_values.push(T::one());
            l_row_ptrs.push(l_values.len());

            // Store U (upper triangular, j >= i)
            for j in i..n {
                if work_idx[j] {
                    let val = work[j].clone();
                    if Scalar::abs(val.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                        u_col_indices.push(j);
                        u_values.push(val);
                    }
                }
            }
            u_row_ptrs.push(u_values.len());

            // Clear work array
            for ptr in start..end {
                let j = col_indices[ptr];
                work[j] = T::zero();
                work_idx[j] = false;
            }
            // Also clear L entries
            for j in 0..i {
                if work_idx[j]
                    || Scalar::abs(work[j].clone()) > T::from_f64(1e-20).unwrap_or(T::zero())
                {
                    work[j] = T::zero();
                    work_idx[j] = false;
                }
            }
        }

        (
            l_values,
            l_col_indices,
            l_row_ptrs,
            u_values,
            u_col_indices,
            u_row_ptrs,
        )
    }

    /// Compute a genuinely exact dense LU factorization (Doolittle form,
    /// with partial pivoting) of a local subdomain block.
    ///
    /// Unlike [`Self::compute_ilu0`], which restricts fill-in to the
    /// original sparsity pattern and therefore only *approximates*
    /// `A_local^{-1}`, this routine densifies the local block and performs
    /// complete Gaussian elimination with partial pivoting, keeping every
    /// fill-in entry produced during elimination. The resulting
    /// factorization satisfies `P * A_local = L * U` to floating-point
    /// precision, so solving with it gives the exact local subdomain
    /// solution (as documented for [`LocalSolverType::ExactLU`]), not an
    /// incomplete approximation.
    ///
    /// Returns `(lu, piv)` where `lu` is `n x n`, stored row-major: entries
    /// strictly below the diagonal are the multipliers of the unit
    /// lower-triangular factor `L` (diagonal implicitly 1), and entries on
    /// and above the diagonal are the upper-triangular factor `U`. `piv[i]`
    /// is the original local row that partial pivoting moved into pivot
    /// position `i`.
    ///
    /// # Errors
    ///
    /// Returns [`PreconditionerError::SingularBlock`] if the local
    /// subdomain matrix is (numerically) singular, since an exact LU
    /// solve is not well-defined in that case.
    fn compute_exact_lu(
        values: &[T],
        col_indices: &[usize],
        row_ptrs: &[usize],
        n: usize,
    ) -> Result<(Vec<T>, Vec<usize>), PreconditionerError> {
        // Densify the local block; ExactLU keeps full fill-in rather than
        // restricting elimination to the sparsity pattern of A_local.
        let mut a = vec![T::zero(); n * n];
        for row in 0..n {
            let start = row_ptrs[row];
            let end = row_ptrs[row + 1];
            for ptr in start..end {
                let col = col_indices[ptr];
                a[row * n + col] = values[ptr].clone();
            }
        }

        let mut piv: Vec<usize> = (0..n).collect();
        let zero_tol = T::from_f64(1e-14).unwrap_or(T::zero());

        for k in 0..n {
            // Partial pivoting: pick the largest-magnitude entry in column k
            // among rows k..n for numerical stability.
            let mut max_row = k;
            let mut max_val = Scalar::abs(a[k * n + k].clone());
            for i in (k + 1)..n {
                let candidate = Scalar::abs(a[i * n + k].clone());
                if candidate > max_val {
                    max_val = candidate;
                    max_row = i;
                }
            }

            if max_val <= zero_tol {
                return Err(PreconditionerError::SingularBlock(k));
            }

            if max_row != k {
                for col in 0..n {
                    a.swap(k * n + col, max_row * n + col);
                }
                piv.swap(k, max_row);
            }

            let pivot = a[k * n + k].clone();
            for i in (k + 1)..n {
                let factor = a[i * n + k].clone() / pivot.clone();
                a[i * n + k] = factor.clone();
                for col in (k + 1)..n {
                    let sub = factor.clone() * a[k * n + col].clone();
                    a[i * n + col] = a[i * n + col].clone() - sub;
                }
            }
        }

        Ok((a, piv))
    }

    /// Solve `A_local x = b` exactly using the dense pivoted LU
    /// factorization produced by [`Self::compute_exact_lu`].
    fn solve_exact_lu(lu: &[T], piv: &[usize], n: usize, b: &[T], x: &mut [T]) {
        // Apply the row permutation: y starts as P * b.
        let mut y: Vec<T> = piv.iter().map(|&p| b[p].clone()).collect();

        // Forward solve L * y = P * b (unit lower triangular), in place.
        for i in 0..n {
            let mut sum = y[i].clone();
            for j in 0..i {
                sum = sum - lu[i * n + j].clone() * y[j].clone();
            }
            y[i] = sum;
        }

        // Backward solve U * x = y.
        for i in (0..n).rev() {
            let mut sum = y[i].clone();
            for j in (i + 1)..n {
                sum = sum - lu[i * n + j].clone() * x[j].clone();
            }
            x[i] = sum / lu[i * n + i].clone();
        }
    }

    /// Solve L*y = b (lower triangular).
    fn solve_lower(
        l_values: &[T],
        l_col_indices: &[usize],
        l_row_ptrs: &[usize],
        b: &[T],
        y: &mut [T],
    ) {
        let n = y.len();
        for i in 0..n {
            let mut sum = b[i].clone();
            let start = l_row_ptrs[i];
            let end = l_row_ptrs[i + 1];

            for ptr in start..end {
                let j = l_col_indices[ptr];
                if j < i {
                    sum = sum - l_values[ptr].clone() * y[j].clone();
                }
            }
            // Diagonal of L is 1
            y[i] = sum;
        }
    }

    /// Solve U*x = y (upper triangular).
    fn solve_upper(
        u_values: &[T],
        u_col_indices: &[usize],
        u_row_ptrs: &[usize],
        y: &[T],
        x: &mut [T],
    ) {
        let n = x.len();
        for i in (0..n).rev() {
            let mut sum = y[i].clone();
            let start = u_row_ptrs[i];
            let end = u_row_ptrs[i + 1];

            let mut diag = T::one();
            for ptr in start..end {
                let j = u_col_indices[ptr];
                if j == i {
                    diag = u_values[ptr].clone();
                } else if j > i {
                    sum = sum - u_values[ptr].clone() * x[j].clone();
                }
            }

            if Scalar::abs(diag.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                x[i] = sum / diag;
            } else {
                x[i] = sum;
            }
        }
    }

    /// Apply the preconditioner: z = M^{-1} * r.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        assert_eq!(r.len(), self.n, "r length must match matrix size");
        assert_eq!(z.len(), self.n, "z length must match matrix size");

        if self.n == 0 {
            return;
        }

        // Initialize z to zero
        for val in z.iter_mut() {
            *val = T::zero();
        }

        // Apply each subdomain and accumulate
        for subdomain in &self.subdomains {
            let local_n = subdomain.indices.len();
            if local_n == 0 {
                continue;
            }

            // Restrict r to local subdomain
            let local_r: Vec<T> = subdomain.indices.iter().map(|&i| r[i].clone()).collect();

            // Solve local system
            let mut local_z = vec![T::zero(); local_n];

            match self.local_solver {
                LocalSolverType::Jacobi => {
                    for i in 0..local_n {
                        local_z[i] = local_r[i].clone() * subdomain.diag_inv[i].clone();
                    }
                }
                LocalSolverType::ILU0 => {
                    // Approximate solve via incomplete LU(0) factors: solve
                    // L*y = r, then U*z = y.
                    let mut y = vec![T::zero(); local_n];
                    Self::solve_lower(
                        &subdomain.ilu_l_values,
                        &subdomain.ilu_l_col_indices,
                        &subdomain.ilu_l_row_ptrs,
                        &local_r,
                        &mut y,
                    );

                    Self::solve_upper(
                        &subdomain.ilu_u_values,
                        &subdomain.ilu_u_col_indices,
                        &subdomain.ilu_u_row_ptrs,
                        &y,
                        &mut local_z,
                    );
                }
                LocalSolverType::ExactLU => {
                    // Genuinely exact solve via the dense pivoted LU
                    // factorization (full fill-in, no dropping).
                    Self::solve_exact_lu(
                        &subdomain.exact_lu_values,
                        &subdomain.exact_lu_piv,
                        local_n,
                        &local_r,
                        &mut local_z,
                    );
                }
            }

            // Prolong and accumulate to global z
            for (local_i, &global_i) in subdomain.indices.iter().enumerate() {
                z[global_i] = z[global_i].clone() + local_z[local_i].clone();
            }
        }
    }

    /// Returns the number of subdomains.
    pub fn num_subdomains(&self) -> usize {
        self.subdomains.len()
    }

    /// Returns the dimension of the preconditioner.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the local solver type.
    pub fn local_solver(&self) -> LocalSolverType {
        self.local_solver
    }

    /// Returns the total number of unknowns across all subdomains (with overlap counted).
    pub fn total_subdomain_size(&self) -> usize {
        self.subdomains.iter().map(|s| s.indices.len()).sum()
    }
}
