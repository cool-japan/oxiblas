//! Algebraic Multigrid (AMG) preconditioner - classical Ruge-Stüben AMG.

use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Configuration for Algebraic Multigrid (AMG) preconditioner.
#[derive(Debug, Clone)]
pub struct AMGConfig<T: Scalar> {
    /// Maximum number of levels in the hierarchy.
    pub max_levels: usize,
    /// Coarse grid size threshold - stop coarsening below this size.
    pub coarse_size_threshold: usize,
    /// Strength of connection threshold (typically 0.25 for symmetric, 0.5 for non-symmetric).
    pub strength_threshold: T,
    /// Number of pre-smoothing iterations.
    pub pre_smooths: usize,
    /// Number of post-smoothing iterations.
    pub post_smooths: usize,
    /// Relaxation parameter for smoother.
    pub smoother_omega: T,
    /// Cycle type: 'V' or 'W'.
    pub cycle_type: AMGCycleType,
}

impl<T: Scalar + Clone> Default for AMGConfig<T> {
    fn default() -> Self {
        Self {
            max_levels: 25,
            coarse_size_threshold: 50,
            strength_threshold: T::from_f64(0.25).unwrap_or(T::zero()),
            pre_smooths: 1,
            post_smooths: 1,
            smoother_omega: T::from_f64(0.67).unwrap_or(T::one()),
            cycle_type: AMGCycleType::V,
        }
    }
}

/// AMG cycle type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AMGCycleType {
    /// V-cycle: one recursion per level.
    V,
    /// W-cycle: two recursions per level.
    W,
}

/// Classification of point in AMG coarsening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CFSplitting {
    /// Coarse point.
    Coarse,
    /// Fine point.
    Fine,
    /// Undecided (temporary state).
    Undecided,
}

/// One level of the AMG hierarchy.
#[derive(Debug, Clone)]
struct AMGLevel<T: Scalar> {
    /// System matrix at this level.
    matrix: CsrMatrix<T>,
    /// Interpolation operator from coarse to fine (P).
    interpolation: Option<CsrMatrix<T>>,
    /// Restriction operator from fine to coarse (R = P^T for symmetric problems).
    restriction: Option<CsrMatrix<T>>,
    /// Inverse diagonal for Jacobi smoothing.
    diag_inv: Vec<T>,
}

/// Algebraic Multigrid (AMG) preconditioner.
///
/// Implements classical Ruge-Stüben AMG with:
/// - Strength-based coarsening.
/// - Classical interpolation.
/// - Galerkin coarse grid operator.
/// - Jacobi smoothing.
/// - V-cycle or W-cycle.
///
/// AMG is one of the most powerful preconditioners for sparse systems
/// arising from elliptic PDEs and similar problems.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::{AMG, AMGConfig};
///
/// let config = AMGConfig::default();
/// let amg = AMG::new(&matrix, config)?;
/// let mut z = vec![0.0; n];
/// amg.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct AMG<T: Scalar> {
    /// AMG hierarchy levels.
    levels: Vec<AMGLevel<T>>,
    /// Configuration.
    config: AMGConfig<T>,
    /// Size of finest grid.
    size: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> AMG<T> {
    /// Create a new AMG preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition (should be SPD for best results).
    /// * `config` - AMG configuration.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or setup fails.
    pub fn new(a: &CsrMatrix<T>, config: AMGConfig<T>) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let size = a.nrows();
        if size == 0 {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix cannot be empty".to_string(),
            ));
        }

        // Build AMG hierarchy
        let levels = Self::build_hierarchy(a, &config)?;

        Ok(Self {
            levels,
            config,
            size,
        })
    }

    /// Build the AMG hierarchy.
    fn build_hierarchy(
        a: &CsrMatrix<T>,
        config: &AMGConfig<T>,
    ) -> Result<Vec<AMGLevel<T>>, PreconditionerError> {
        let mut levels = Vec::new();
        let mut current_matrix = a.clone();

        for _level in 0..config.max_levels {
            let n = current_matrix.nrows();

            // Extract diagonal for smoother
            let diag_inv = Self::compute_diag_inv(&current_matrix)?;

            // Check if we should stop coarsening
            if n <= config.coarse_size_threshold {
                levels.push(AMGLevel {
                    matrix: current_matrix,
                    interpolation: None,
                    restriction: None,
                    diag_inv,
                });
                break;
            }

            // Compute strength of connection
            let strength =
                Self::compute_strength(&current_matrix, config.strength_threshold.clone());

            // Perform C/F splitting
            let splitting = Self::cf_splitting(&strength, n);

            // Count coarse points
            let num_coarse: usize = splitting
                .iter()
                .filter(|&&s| s == CFSplitting::Coarse)
                .count();

            if num_coarse == 0 || num_coarse >= n {
                // Coarsening failed - stop here
                levels.push(AMGLevel {
                    matrix: current_matrix,
                    interpolation: None,
                    restriction: None,
                    diag_inv,
                });
                break;
            }

            // Build interpolation operator
            let interpolation =
                Self::build_interpolation(&current_matrix, &strength, &splitting, num_coarse);

            // Build restriction (transpose of interpolation for symmetric)
            let restriction = Self::transpose_csr(&interpolation);

            // Build coarse grid operator: A_c = R * A * P
            let ap = Self::spmm_csr(&current_matrix, &interpolation);
            let coarse_matrix = Self::spmm_csr(&restriction, &ap);

            levels.push(AMGLevel {
                matrix: current_matrix,
                interpolation: Some(interpolation),
                restriction: Some(restriction),
                diag_inv,
            });

            current_matrix = coarse_matrix;
        }

        if levels.is_empty() {
            return Err(PreconditionerError::InvalidMatrix(
                "AMG hierarchy construction failed".to_string(),
            ));
        }

        Ok(levels)
    }

    /// Compute inverse of diagonal elements.
    fn compute_diag_inv(a: &CsrMatrix<T>) -> Result<Vec<T>, PreconditionerError> {
        let n = a.nrows();
        let mut diag_inv = vec![T::zero(); n];

        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut found = false;
            for k in start..end {
                if a.col_indices()[k] == i {
                    let val = a.values()[k].clone();
                    if Scalar::abs(val.clone()) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                        return Err(PreconditionerError::ZeroDiagonal(i));
                    }
                    diag_inv[i] = T::one() / val;
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(PreconditionerError::ZeroDiagonal(i));
            }
        }

        Ok(diag_inv)
    }

    /// Compute strength of connection matrix.
    /// Returns a binary strength matrix where S(i,j) = 1 if j strongly influences i.
    fn compute_strength(a: &CsrMatrix<T>, threshold: T) -> Vec<Vec<usize>> {
        let n = a.nrows();
        let mut strength = vec![Vec::new(); n];

        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            // Find max off-diagonal magnitude in row i
            let mut max_offdiag = T::zero();
            for k in start..end {
                let j = a.col_indices()[k];
                if j != i {
                    let val = Scalar::abs(a.values()[k].clone());
                    if val > max_offdiag {
                        max_offdiag = val;
                    }
                }
            }

            // Mark strongly connected points
            let threshold_val = threshold.clone() * max_offdiag;
            for k in start..end {
                let j = a.col_indices()[k];
                if j != i {
                    let val = Scalar::abs(a.values()[k].clone());
                    if val >= threshold_val {
                        strength[i].push(j);
                    }
                }
            }
        }

        strength
    }

    /// Perform C/F splitting using classical RS coarsening.
    fn cf_splitting(strength: &[Vec<usize>], n: usize) -> Vec<CFSplitting> {
        let mut splitting = vec![CFSplitting::Undecided; n];

        // Compute initial weights (number of points strongly influenced by i)
        let mut weights = vec![0usize; n];
        for i in 0..n {
            for &j in &strength[i] {
                weights[j] += 1;
            }
        }

        // Build transpose of strength (who influences each point)
        let mut strength_t = vec![Vec::new(); n];
        for i in 0..n {
            for &j in &strength[i] {
                strength_t[j].push(i);
            }
        }

        let mut remaining = n;

        while remaining > 0 {
            // Find undecided point with maximum weight
            let mut max_weight = 0;
            let mut max_idx = None;

            for i in 0..n {
                if splitting[i] == CFSplitting::Undecided && weights[i] >= max_weight {
                    max_weight = weights[i];
                    max_idx = Some(i);
                }
            }

            match max_idx {
                Some(c) => {
                    // Make this point coarse
                    splitting[c] = CFSplitting::Coarse;
                    remaining -= 1;

                    // Points strongly connected to c become fine
                    for &j in &strength_t[c] {
                        if splitting[j] == CFSplitting::Undecided {
                            splitting[j] = CFSplitting::Fine;
                            remaining -= 1;

                            // Update weights: points strongly connected to j get +1
                            for &k in &strength_t[j] {
                                if splitting[k] == CFSplitting::Undecided {
                                    weights[k] += 1;
                                }
                            }
                        }
                    }

                    // Points that c strongly influences lose c from consideration
                    for &j in &strength[c] {
                        if weights[j] > 0 {
                            weights[j] -= 1;
                        }
                    }
                }
                None => {
                    // No undecided points with positive weight - make remaining fine
                    for i in 0..n {
                        if splitting[i] == CFSplitting::Undecided {
                            splitting[i] = CFSplitting::Fine;
                            remaining -= 1;
                        }
                    }
                }
            }
        }

        splitting
    }

    /// Build interpolation operator P.
    fn build_interpolation(
        a: &CsrMatrix<T>,
        strength: &[Vec<usize>],
        splitting: &[CFSplitting],
        num_coarse: usize,
    ) -> CsrMatrix<T> {
        let n = a.nrows();

        // Map coarse points to coarse indices
        let mut coarse_idx = vec![0usize; n];
        let mut idx = 0;
        for i in 0..n {
            if splitting[i] == CFSplitting::Coarse {
                coarse_idx[i] = idx;
                idx += 1;
            }
        }

        // Build interpolation matrix
        let mut row_ptrs = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            if splitting[i] == CFSplitting::Coarse {
                // Coarse point: P(i, coarse_idx[i]) = 1
                col_indices.push(coarse_idx[i]);
                values.push(T::one());
            } else {
                // Fine point: interpolate from strongly connected coarse points
                let start = a.row_ptrs()[i];
                let end = a.row_ptrs()[i + 1];

                // Get diagonal element
                let mut diag = T::zero();
                for k in start..end {
                    if a.col_indices()[k] == i {
                        diag = a.values()[k].clone();
                        break;
                    }
                }

                // Find strongly connected coarse points and sum of strong connections
                let mut strong_coarse = Vec::new();
                let mut strong_fine = Vec::new();
                let strong_set: std::collections::HashSet<usize> =
                    strength[i].iter().cloned().collect();

                for k in start..end {
                    let j = a.col_indices()[k];
                    if j != i && strong_set.contains(&j) {
                        if splitting[j] == CFSplitting::Coarse {
                            strong_coarse.push((j, a.values()[k].clone()));
                        } else {
                            strong_fine.push((j, a.values()[k].clone()));
                        }
                    }
                }

                if strong_coarse.is_empty() {
                    // No strong coarse connections - use direct injection from any coarse neighbor
                    let mut found = false;
                    for k in start..end {
                        let j = a.col_indices()[k];
                        if j != i && splitting[j] == CFSplitting::Coarse {
                            col_indices.push(coarse_idx[j]);
                            values.push(T::one());
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        // Fallback: zero row (point will rely on smoothing)
                    }
                } else {
                    // Classical interpolation formula
                    // Sum of weak connections to add to diagonal
                    let mut weak_sum = T::zero();
                    for k in start..end {
                        let j = a.col_indices()[k];
                        if j != i && !strong_set.contains(&j) {
                            weak_sum = weak_sum + a.values()[k].clone();
                        }
                    }

                    // Modified diagonal
                    let mod_diag = diag + weak_sum;

                    // For each strongly connected fine point, distribute to coarse
                    // This is simplified - full RS would distribute through fine points
                    let mut coarse_weights = std::collections::HashMap::new();

                    for (j, a_ij) in &strong_coarse {
                        let entry = coarse_weights.entry(coarse_idx[*j]).or_insert(T::zero());
                        *entry = entry.clone() + a_ij.clone();
                    }

                    // Add contributions from strongly connected fine points
                    // Simplified: distribute fine contributions equally to their coarse neighbors
                    for (j, a_ij) in &strong_fine {
                        // Find coarse points that j is strongly connected to
                        let j_strong_coarse: Vec<_> = strength[*j]
                            .iter()
                            .filter(|&&k| splitting[k] == CFSplitting::Coarse)
                            .cloned()
                            .collect();

                        if !j_strong_coarse.is_empty() {
                            let contrib = a_ij.clone()
                                / T::from_usize(j_strong_coarse.len()).unwrap_or(T::one());
                            for k in j_strong_coarse {
                                let entry =
                                    coarse_weights.entry(coarse_idx[k]).or_insert(T::zero());
                                *entry = entry.clone() + contrib.clone();
                            }
                        }
                    }

                    // Compute interpolation weights
                    if Scalar::abs(mod_diag.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                        for (c_idx, weight) in coarse_weights {
                            let p_val = T::zero() - weight / mod_diag.clone();
                            col_indices.push(c_idx);
                            values.push(p_val);
                        }
                    }
                }
            }

            row_ptrs[i + 1] = col_indices.len();
        }

        // Sort each row by column index
        for i in 0..n {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            if end > start + 1 {
                let mut pairs: Vec<_> = (start..end)
                    .map(|k| (col_indices[k], values[k].clone()))
                    .collect();
                pairs.sort_by_key(|(c, _)| *c);
                for (k, (c, v)) in (start..end).zip(pairs) {
                    col_indices[k] = c;
                    values[k] = v;
                }
            }
        }

        CsrMatrix::new(n, num_coarse, row_ptrs, col_indices, values).unwrap_or_else(|_| {
            CsrMatrix::new(n, num_coarse, vec![0; n + 1], vec![], vec![])
                .expect("CSR matrix construction with valid parameters")
        })
    }

    /// Transpose a CSR matrix.
    fn transpose_csr(a: &CsrMatrix<T>) -> CsrMatrix<T> {
        let nrows = a.nrows();
        let ncols = a.ncols();

        // Count entries per column (will become rows)
        let mut col_counts = vec![0usize; ncols];
        for &c in a.col_indices() {
            col_counts[c] += 1;
        }

        // Build row pointers for transpose
        let mut row_ptrs = vec![0usize; ncols + 1];
        for i in 0..ncols {
            row_ptrs[i + 1] = row_ptrs[i] + col_counts[i];
        }

        // Fill in values
        let nnz = a.col_indices().len();
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut current = vec![0usize; ncols];

        for i in 0..nrows {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            for k in start..end {
                let j = a.col_indices()[k];
                let pos = row_ptrs[j] + current[j];
                col_indices[pos] = i;
                values[pos] = a.values()[k].clone();
                current[j] += 1;
            }
        }

        CsrMatrix::new(ncols, nrows, row_ptrs, col_indices, values).unwrap_or_else(|_| {
            CsrMatrix::new(ncols, nrows, vec![0; ncols + 1], vec![], vec![])
                .expect("CSR matrix construction with valid parameters")
        })
    }

    /// Sparse matrix-matrix multiplication: C = A * B
    fn spmm_csr(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T> {
        let m = a.nrows();
        let n = b.ncols();

        if n == 0 || m == 0 {
            return CsrMatrix::new(m, n, vec![0; m + 1], vec![], vec![])
                .expect("CSR matrix construction with valid parameters");
        }

        let mut row_ptrs = vec![0usize; m + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Use row-specific marker for tracking columns in current row
        // marker[c] = i means column c was added in row i
        // Using m+1 as "not seen in any row" since valid rows are 0..m
        let mut marker = vec![m + 1; n];
        let mut row_col_idx = vec![0usize; n]; // Position in output for column c in current row

        for i in 0..m {
            let start_i = a.row_ptrs()[i];
            let end_i = a.row_ptrs()[i + 1];

            let row_start = col_indices.len();

            for k in start_i..end_i {
                let j = a.col_indices()[k];
                let a_ij = a.values()[k].clone();

                let start_j = b.row_ptrs()[j];
                let end_j = b.row_ptrs()[j + 1];

                for l in start_j..end_j {
                    let c = b.col_indices()[l];
                    let b_jc = b.values()[l].clone();

                    if marker[c] != i {
                        // First time seeing this column in row i
                        marker[c] = i;
                        row_col_idx[c] = col_indices.len() - row_start;
                        col_indices.push(c);
                        values.push(a_ij.clone() * b_jc);
                    } else {
                        // Already have this column in current row
                        let idx = row_start + row_col_idx[c];
                        values[idx] = values[idx].clone() + a_ij.clone() * b_jc;
                    }
                }
            }

            // Sort row by column index
            let row_end = col_indices.len();
            if row_end > row_start + 1 {
                let mut pairs: Vec<_> = (row_start..row_end)
                    .map(|k| (col_indices[k], values[k].clone()))
                    .collect();
                pairs.sort_by_key(|(c, _)| *c);
                for (k, (c, v)) in (row_start..row_end).zip(pairs) {
                    col_indices[k] = c;
                    values[k] = v;
                }
            }

            row_ptrs[i + 1] = col_indices.len();
        }

        CsrMatrix::new(m, n, row_ptrs, col_indices, values).unwrap_or_else(|_| {
            CsrMatrix::new(m, n, vec![0; m + 1], vec![], vec![])
                .expect("CSR matrix construction with valid parameters")
        })
    }

    /// Apply the preconditioner using V-cycle or W-cycle.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        assert_eq!(r.len(), self.size, "r length must match matrix size");
        assert_eq!(z.len(), self.size, "z length must match matrix size");

        // Initialize z to zero
        for z_i in z.iter_mut() {
            *z_i = T::zero();
        }

        // Allocate workspace for residuals at each level
        let mut residuals: Vec<Vec<T>> = self
            .levels
            .iter()
            .map(|level| vec![T::zero(); level.matrix.nrows()])
            .collect();

        let mut solutions: Vec<Vec<T>> = self
            .levels
            .iter()
            .map(|level| vec![T::zero(); level.matrix.nrows()])
            .collect();

        // Copy initial residual
        residuals[0].copy_from_slice(r);

        // Perform cycle
        match self.config.cycle_type {
            AMGCycleType::V => self.v_cycle(&mut residuals, &mut solutions, 0),
            AMGCycleType::W => self.w_cycle(&mut residuals, &mut solutions, 0),
        }

        // Copy result
        z.copy_from_slice(&solutions[0]);
    }

    /// V-cycle recursion.
    fn v_cycle(&self, residuals: &mut [Vec<T>], solutions: &mut [Vec<T>], level: usize) {
        let n = self.levels[level].matrix.nrows();

        if level == self.levels.len() - 1 {
            // Coarsest level: solve directly using many relaxation iterations
            for _ in 0..50 {
                self.smooth(level, &residuals[level].clone(), &mut solutions[level]);
            }
            return;
        }

        // Pre-smoothing
        for _ in 0..self.config.pre_smooths {
            self.smooth(level, &residuals[level].clone(), &mut solutions[level]);
        }

        // Compute residual: r = b - A*x
        let mut r_fine = vec![T::zero(); n];
        self.compute_residual(level, &residuals[level], &solutions[level], &mut r_fine);

        // Restrict residual to coarse grid
        let n_coarse = self.levels[level + 1].matrix.nrows();
        let mut r_coarse = vec![T::zero(); n_coarse];
        self.restrict(level, &r_fine, &mut r_coarse);

        // Copy to coarse residual
        residuals[level + 1].copy_from_slice(&r_coarse);

        // Clear coarse solution
        for x in solutions[level + 1].iter_mut() {
            *x = T::zero();
        }

        // Recurse
        self.v_cycle(residuals, solutions, level + 1);

        // Interpolate coarse correction to fine grid
        let mut e_fine = vec![T::zero(); n];
        self.interpolate(level, &solutions[level + 1], &mut e_fine);

        // Apply correction
        for i in 0..n {
            solutions[level][i] = solutions[level][i].clone() + e_fine[i].clone();
        }

        // Post-smoothing
        for _ in 0..self.config.post_smooths {
            self.smooth(level, &residuals[level].clone(), &mut solutions[level]);
        }
    }

    /// W-cycle recursion.
    fn w_cycle(&self, residuals: &mut [Vec<T>], solutions: &mut [Vec<T>], level: usize) {
        let n = self.levels[level].matrix.nrows();

        if level == self.levels.len() - 1 {
            // Coarsest level: solve directly
            for _ in 0..50 {
                self.smooth(level, &residuals[level].clone(), &mut solutions[level]);
            }
            return;
        }

        // Pre-smoothing
        for _ in 0..self.config.pre_smooths {
            self.smooth(level, &residuals[level].clone(), &mut solutions[level]);
        }

        // Compute residual
        let mut r_fine = vec![T::zero(); n];
        self.compute_residual(level, &residuals[level], &solutions[level], &mut r_fine);

        // Restrict to coarse grid
        let n_coarse = self.levels[level + 1].matrix.nrows();
        let mut r_coarse = vec![T::zero(); n_coarse];
        self.restrict(level, &r_fine, &mut r_coarse);

        residuals[level + 1].copy_from_slice(&r_coarse);

        // Clear coarse solution
        for x in solutions[level + 1].iter_mut() {
            *x = T::zero();
        }

        // Two recursive calls for W-cycle
        self.w_cycle(residuals, solutions, level + 1);

        // Interpolate and update
        let mut e_fine = vec![T::zero(); n];
        self.interpolate(level, &solutions[level + 1], &mut e_fine);
        for i in 0..n {
            solutions[level][i] = solutions[level][i].clone() + e_fine[i].clone();
        }

        // Compute residual again
        self.compute_residual(level, &residuals[level], &solutions[level], &mut r_fine);
        self.restrict(level, &r_fine, &mut r_coarse);
        residuals[level + 1].copy_from_slice(&r_coarse);

        for x in solutions[level + 1].iter_mut() {
            *x = T::zero();
        }

        // Second recursive call
        self.w_cycle(residuals, solutions, level + 1);

        // Second interpolation and correction
        self.interpolate(level, &solutions[level + 1], &mut e_fine);
        for i in 0..n {
            solutions[level][i] = solutions[level][i].clone() + e_fine[i].clone();
        }

        // Post-smoothing
        for _ in 0..self.config.post_smooths {
            self.smooth(level, &residuals[level].clone(), &mut solutions[level]);
        }
    }

    /// Weighted Jacobi smoothing.
    fn smooth(&self, level: usize, b: &[T], x: &mut [T]) {
        let a = &self.levels[level].matrix;
        let diag_inv = &self.levels[level].diag_inv;
        let omega = &self.config.smoother_omega;
        let n = a.nrows();

        // Jacobi: x_new = x + omega * D^{-1} * (b - A*x)
        let mut ax = vec![T::zero(); n];
        self.matvec(level, x, &mut ax);

        for i in 0..n {
            let r_i = b[i].clone() - ax[i].clone();
            x[i] = x[i].clone() + omega.clone() * diag_inv[i].clone() * r_i;
        }
    }

    /// Compute residual: r = b - A*x
    fn compute_residual(&self, level: usize, b: &[T], x: &[T], r: &mut [T]) {
        let a = &self.levels[level].matrix;
        let n = a.nrows();

        // Compute A*x
        self.matvec(level, x, r);

        // r = b - A*x
        for i in 0..n {
            r[i] = b[i].clone() - r[i].clone();
        }
    }

    /// Matrix-vector product at a level.
    fn matvec(&self, level: usize, x: &[T], y: &mut [T]) {
        let a = &self.levels[level].matrix;
        let n = a.nrows();

        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut sum = T::zero();
            for k in start..end {
                let j = a.col_indices()[k];
                sum = sum + a.values()[k].clone() * x[j].clone();
            }
            y[i] = sum;
        }
    }

    /// Restrict from fine to coarse grid: r_c = R * r_f
    fn restrict(&self, level: usize, r_fine: &[T], r_coarse: &mut [T]) {
        if let Some(ref restriction) = self.levels[level].restriction {
            let n_coarse = restriction.nrows();

            for i in 0..n_coarse {
                let start = restriction.row_ptrs()[i];
                let end = restriction.row_ptrs()[i + 1];

                let mut sum = T::zero();
                for k in start..end {
                    let j = restriction.col_indices()[k];
                    sum = sum + restriction.values()[k].clone() * r_fine[j].clone();
                }
                r_coarse[i] = sum;
            }
        }
    }

    /// Interpolate from coarse to fine grid: e_f = P * e_c
    fn interpolate(&self, level: usize, e_coarse: &[T], e_fine: &mut [T]) {
        if let Some(ref interpolation) = self.levels[level].interpolation {
            let n_fine = interpolation.nrows();

            for i in 0..n_fine {
                let start = interpolation.row_ptrs()[i];
                let end = interpolation.row_ptrs()[i + 1];

                let mut sum = T::zero();
                for k in start..end {
                    let j = interpolation.col_indices()[k];
                    sum = sum + interpolation.values()[k].clone() * e_coarse[j].clone();
                }
                e_fine[i] = sum;
            }
        }
    }

    /// Get the size of the preconditioner.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the number of levels in the hierarchy.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get the grid complexity (sum of all level sizes / finest level size).
    pub fn grid_complexity(&self) -> f64 {
        let total: usize = self.levels.iter().map(|l| l.matrix.nrows()).sum();
        total as f64 / self.size as f64
    }

    /// Get the operator complexity (sum of all level nnz / finest level nnz).
    pub fn operator_complexity(&self) -> f64 {
        let total: usize = self
            .levels
            .iter()
            .map(|l| l.matrix.col_indices().len())
            .sum();
        let finest_nnz = self.levels[0].matrix.col_indices().len();
        if finest_nnz > 0 {
            total as f64 / finest_nnz as f64
        } else {
            1.0
        }
    }
}
