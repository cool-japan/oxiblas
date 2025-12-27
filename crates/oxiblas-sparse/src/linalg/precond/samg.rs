//! Smoothed Aggregation AMG (SA-AMG) preconditioner.

use super::amg::AMGCycleType;
use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Configuration for Smoothed Aggregation AMG (SA-AMG) preconditioner.
#[derive(Debug, Clone)]
pub struct SAMGConfig<T: Scalar> {
    /// Maximum number of levels in the hierarchy.
    pub max_levels: usize,
    /// Coarse grid size threshold - stop coarsening below this size.
    pub coarse_size_threshold: usize,
    /// Strength of connection threshold for aggregation (typically 0.0 to 0.25).
    pub strength_threshold: T,
    /// Number of pre-smoothing iterations.
    pub pre_smooths: usize,
    /// Number of post-smoothing iterations.
    pub post_smooths: usize,
    /// Relaxation parameter for smoother (Jacobi).
    pub smoother_omega: T,
    /// Relaxation parameter for prolongation smoothing (typically 4/3 * 1/rho(D^{-1}A)).
    pub prolongation_omega: T,
    /// Number of smoothing steps for prolongator.
    pub prolongation_smoothing_steps: usize,
    /// Cycle type: 'V' or 'W'.
    pub cycle_type: AMGCycleType,
    /// Maximum aggregate size (0 = no limit).
    pub max_aggregate_size: usize,
}

impl<T: Scalar + Clone> Default for SAMGConfig<T> {
    fn default() -> Self {
        Self {
            max_levels: 25,
            coarse_size_threshold: 50,
            strength_threshold: T::from_f64(0.08).unwrap_or(T::zero()),
            pre_smooths: 1,
            post_smooths: 1,
            smoother_omega: T::from_f64(0.67).unwrap_or(T::one()),
            prolongation_omega: T::from_f64(0.67).unwrap_or(T::one()),
            prolongation_smoothing_steps: 1,
            cycle_type: AMGCycleType::V,
            max_aggregate_size: 0,
        }
    }
}

/// Aggregate assignment for SA-AMG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateState {
    /// Not yet assigned to any aggregate.
    Unassigned,
    /// Assigned to aggregate with given index.
    Assigned(usize),
}

/// One level of the SA-AMG hierarchy.
#[derive(Debug, Clone)]
struct SAMGLevel<T: Scalar> {
    /// System matrix at this level.
    matrix: CsrMatrix<T>,
    /// Prolongation operator from coarse to fine (P).
    prolongation: Option<CsrMatrix<T>>,
    /// Restriction operator from fine to coarse (R = P^T).
    restriction: Option<CsrMatrix<T>>,
    /// Inverse diagonal for Jacobi smoothing.
    diag_inv: Vec<T>,
}

/// Smoothed Aggregation Algebraic Multigrid (SA-AMG) preconditioner.
///
/// SA-AMG is a variant of AMG that uses aggregation-based coarsening and
/// smoothed prolongation operators. It is particularly effective for:
/// - Systems from finite element discretizations
/// - Problems with near-null space components
/// - Elasticity and structural mechanics
///
/// The algorithm:
/// 1. **Aggregation**: Group nodes into aggregates based on strength of connection
/// 2. **Tentative prolongator**: Build P_tent with one coarse DOF per aggregate
/// 3. **Smoothed prolongator**: P = (I - omega * D^{-1} * A) * P_tent
/// 4. **Galerkin coarse grid**: A_c = P^T * A * P
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::{SAMG, SAMGConfig};
///
/// let config = SAMGConfig::default();
/// let samg = SAMG::new(&matrix, config)?;
/// let mut z = vec![0.0; n];
/// samg.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct SAMG<T: Scalar> {
    /// SA-AMG hierarchy levels.
    levels: Vec<SAMGLevel<T>>,
    /// Configuration.
    config: SAMGConfig<T>,
    /// Size of finest grid.
    size: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> SAMG<T> {
    /// Create a new SA-AMG preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition (should be SPD for best results).
    /// * `config` - SA-AMG configuration.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or setup fails.
    pub fn new(a: &CsrMatrix<T>, config: SAMGConfig<T>) -> Result<Self, PreconditionerError> {
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

        // Build SA-AMG hierarchy
        let levels = Self::build_hierarchy(a, &config)?;

        Ok(Self {
            levels,
            config,
            size,
        })
    }

    /// Build the SA-AMG hierarchy.
    fn build_hierarchy(
        a: &CsrMatrix<T>,
        config: &SAMGConfig<T>,
    ) -> Result<Vec<SAMGLevel<T>>, PreconditionerError> {
        let mut levels = Vec::new();
        let mut current_matrix = a.clone();

        for _level in 0..config.max_levels {
            let n = current_matrix.nrows();

            // Extract diagonal for smoother
            let diag_inv = Self::compute_diag_inv(&current_matrix)?;

            // Check if we should stop coarsening
            if n <= config.coarse_size_threshold {
                levels.push(SAMGLevel {
                    matrix: current_matrix,
                    prolongation: None,
                    restriction: None,
                    diag_inv,
                });
                break;
            }

            // Compute strength of connection
            let strength =
                Self::compute_strength(&current_matrix, config.strength_threshold.clone());

            // Perform aggregation
            let (aggregates, num_aggregates) =
                Self::aggregate(&strength, n, config.max_aggregate_size);

            if num_aggregates == 0 || num_aggregates >= n {
                // Aggregation failed - stop here
                levels.push(SAMGLevel {
                    matrix: current_matrix,
                    prolongation: None,
                    restriction: None,
                    diag_inv,
                });
                break;
            }

            // Build tentative prolongator
            let p_tent = Self::build_tentative_prolongator(&aggregates, n, num_aggregates);

            // Smooth the prolongator: P = (I - omega * D^{-1} * A) * P_tent
            let mut prolongation = p_tent;
            for _ in 0..config.prolongation_smoothing_steps {
                prolongation = Self::smooth_prolongator(
                    &current_matrix,
                    &diag_inv,
                    &prolongation,
                    config.prolongation_omega.clone(),
                );
            }

            // Build restriction (transpose of prolongation)
            let restriction = Self::transpose_csr(&prolongation);

            // Build coarse grid operator: A_c = R * A * P = P^T * A * P
            let ap = Self::spmm_csr(&current_matrix, &prolongation);
            let coarse_matrix = Self::spmm_csr(&restriction, &ap);

            levels.push(SAMGLevel {
                matrix: current_matrix,
                prolongation: Some(prolongation),
                restriction: Some(restriction),
                diag_inv,
            });

            current_matrix = coarse_matrix;
        }

        if levels.is_empty() {
            return Err(PreconditionerError::InvalidMatrix(
                "SA-AMG hierarchy construction failed".to_string(),
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

    /// Compute strength of connection for aggregation.
    /// Returns adjacency list of strongly connected neighbors.
    fn compute_strength(a: &CsrMatrix<T>, threshold: T) -> Vec<Vec<usize>> {
        let n = a.nrows();
        let mut strength = vec![Vec::new(); n];

        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            // Get diagonal value
            let mut diag_i = T::zero();
            for k in start..end {
                if a.col_indices()[k] == i {
                    diag_i = Scalar::abs(a.values()[k].clone());
                    break;
                }
            }

            // For each off-diagonal entry
            for k in start..end {
                let j = a.col_indices()[k];
                if j != i {
                    let a_ij = Scalar::abs(a.values()[k].clone());

                    // Get diagonal of j
                    let mut diag_j = T::zero();
                    let start_j = a.row_ptrs()[j];
                    let end_j = a.row_ptrs()[j + 1];
                    for kk in start_j..end_j {
                        if a.col_indices()[kk] == j {
                            diag_j = Scalar::abs(a.values()[kk].clone());
                            break;
                        }
                    }

                    // Symmetric strength: |a_ij|^2 >= threshold^2 * |a_ii| * |a_jj|
                    // This is equivalent to: |a_ij| >= threshold * sqrt(|a_ii| * |a_jj|)
                    let product = diag_i.clone() * diag_j;
                    let thresh_sq = threshold.clone() * threshold.clone();

                    if a_ij.clone() * a_ij >= thresh_sq * product {
                        strength[i].push(j);
                    }
                }
            }
        }

        strength
    }

    /// Perform aggregation using greedy algorithm.
    /// Returns (aggregate assignment, number of aggregates).
    fn aggregate(
        strength: &[Vec<usize>],
        n: usize,
        max_size: usize,
    ) -> (Vec<AggregateState>, usize) {
        let mut aggregates = vec![AggregateState::Unassigned; n];
        let mut num_aggregates = 0;
        let mut aggregate_sizes = Vec::new();

        // Phase 1: Create seed aggregates from unassigned nodes
        // Nodes with more unassigned neighbors are preferred as seeds
        let mut node_order: Vec<usize> = (0..n).collect();
        node_order.sort_by(|&a, &b| {
            let count_a = strength[a]
                .iter()
                .filter(|&&j| aggregates[j] == AggregateState::Unassigned)
                .count();
            let count_b = strength[b]
                .iter()
                .filter(|&&j| aggregates[j] == AggregateState::Unassigned)
                .count();
            count_b.cmp(&count_a)
        });

        for &i in &node_order {
            if aggregates[i] != AggregateState::Unassigned {
                continue;
            }

            // Check if this node has any strongly connected unassigned neighbors
            let unassigned_neighbors: Vec<usize> = strength[i]
                .iter()
                .filter(|&&j| aggregates[j] == AggregateState::Unassigned)
                .cloned()
                .collect();

            // Create new aggregate with this node as root
            let agg_idx = num_aggregates;
            aggregates[i] = AggregateState::Assigned(agg_idx);
            let mut size = 1;

            // Add strongly connected unassigned neighbors to this aggregate
            for &j in &unassigned_neighbors {
                if max_size > 0 && size >= max_size {
                    break;
                }
                if aggregates[j] == AggregateState::Unassigned {
                    aggregates[j] = AggregateState::Assigned(agg_idx);
                    size += 1;
                }
            }

            aggregate_sizes.push(size);
            num_aggregates += 1;
        }

        // Phase 2: Assign any remaining isolated nodes
        // (nodes with no strong connections that weren't assigned)
        for i in 0..n {
            if aggregates[i] == AggregateState::Unassigned {
                // Try to join an existing aggregate through weak connection
                let mut best_agg = None;
                let mut best_size = usize::MAX;

                // Look at all neighbors in the matrix
                for &j in &strength[i] {
                    if let AggregateState::Assigned(agg) = aggregates[j] {
                        if aggregate_sizes[agg] < best_size {
                            best_agg = Some(agg);
                            best_size = aggregate_sizes[agg];
                        }
                    }
                }

                if let Some(agg) = best_agg {
                    aggregates[i] = AggregateState::Assigned(agg);
                    aggregate_sizes[agg] += 1;
                } else {
                    // Create singleton aggregate
                    aggregates[i] = AggregateState::Assigned(num_aggregates);
                    aggregate_sizes.push(1);
                    num_aggregates += 1;
                }
            }
        }

        (aggregates, num_aggregates)
    }

    /// Build tentative prolongator P_tent.
    /// Each fine point maps to its aggregate with weight 1.
    fn build_tentative_prolongator(
        aggregates: &[AggregateState],
        n_fine: usize,
        n_coarse: usize,
    ) -> CsrMatrix<T> {
        let mut row_ptrs = vec![0usize; n_fine + 1];
        let mut col_indices = Vec::with_capacity(n_fine);
        let mut values = Vec::with_capacity(n_fine);

        for i in 0..n_fine {
            if let AggregateState::Assigned(agg) = aggregates[i] {
                col_indices.push(agg);
                values.push(T::one());
            }
            row_ptrs[i + 1] = col_indices.len();
        }

        CsrMatrix::new(n_fine, n_coarse, row_ptrs, col_indices, values).unwrap_or_else(|_| {
            CsrMatrix::new(n_fine, n_coarse, vec![0; n_fine + 1], vec![], vec![]).unwrap()
        })
    }

    /// Smooth the prolongator: P_new = (I - omega * D^{-1} * A) * P_old
    /// This is equivalent to one Jacobi smoothing step.
    fn smooth_prolongator(
        a: &CsrMatrix<T>,
        diag_inv: &[T],
        p: &CsrMatrix<T>,
        omega: T,
    ) -> CsrMatrix<T> {
        let n_fine = a.nrows();
        let n_coarse = p.ncols();

        // Compute A * P
        let ap = Self::spmm_csr(a, p);

        // Build P_new = P - omega * D^{-1} * (A * P)
        // This is done row by row
        let mut row_ptrs = vec![0usize; n_fine + 1];
        let mut col_indices_map: Vec<std::collections::HashMap<usize, T>> =
            vec![std::collections::HashMap::new(); n_fine];

        // Start with P
        for i in 0..n_fine {
            let start = p.row_ptrs()[i];
            let end = p.row_ptrs()[i + 1];
            for k in start..end {
                let j = p.col_indices()[k];
                let v = p.values()[k].clone();
                col_indices_map[i].insert(j, v);
            }
        }

        // Subtract omega * D^{-1} * (A * P)
        for i in 0..n_fine {
            let start = ap.row_ptrs()[i];
            let end = ap.row_ptrs()[i + 1];
            let scale = omega.clone() * diag_inv[i].clone();

            for k in start..end {
                let j = ap.col_indices()[k];
                let v = scale.clone() * ap.values()[k].clone();

                let entry = col_indices_map[i].entry(j).or_insert(T::zero());
                *entry = entry.clone() - v;
            }
        }

        // Convert to CSR format
        let mut all_col_indices = Vec::new();
        let mut all_values = Vec::new();

        for i in 0..n_fine {
            let mut sorted: Vec<_> = col_indices_map[i].drain().collect();
            sorted.sort_by_key(|(c, _)| *c);

            for (c, v) in sorted {
                // Only keep non-tiny entries
                if Scalar::abs(v.clone()) > T::from_f64(1e-14).unwrap_or(T::zero()) {
                    all_col_indices.push(c);
                    all_values.push(v);
                }
            }
            row_ptrs[i + 1] = all_col_indices.len();
        }

        CsrMatrix::new(n_fine, n_coarse, row_ptrs, all_col_indices, all_values).unwrap_or_else(
            |_| CsrMatrix::new(n_fine, n_coarse, vec![0; n_fine + 1], vec![], vec![]).unwrap(),
        )
    }

    /// Transpose a CSR matrix.
    fn transpose_csr(a: &CsrMatrix<T>) -> CsrMatrix<T> {
        let nrows = a.nrows();
        let ncols = a.ncols();

        // Count entries per column
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
            CsrMatrix::new(ncols, nrows, vec![0; ncols + 1], vec![], vec![]).unwrap()
        })
    }

    /// Sparse matrix-matrix multiplication: C = A * B
    fn spmm_csr(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T> {
        let m = a.nrows();
        let n = b.ncols();

        if n == 0 || m == 0 {
            return CsrMatrix::new(m, n, vec![0; m + 1], vec![], vec![]).unwrap();
        }

        let mut row_ptrs = vec![0usize; m + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let mut marker = vec![m + 1; n];
        let mut row_col_idx = vec![0usize; n];

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
                        marker[c] = i;
                        row_col_idx[c] = col_indices.len() - row_start;
                        col_indices.push(c);
                        values.push(a_ij.clone() * b_jc);
                    } else {
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

        CsrMatrix::new(m, n, row_ptrs, col_indices, values)
            .unwrap_or_else(|_| CsrMatrix::new(m, n, vec![0; m + 1], vec![], vec![]).unwrap())
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

        // Allocate workspace
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

        // Restrict
        let n_coarse = self.levels[level + 1].matrix.nrows();
        let mut r_coarse = vec![T::zero(); n_coarse];
        self.restrict(level, &r_fine, &mut r_coarse);

        residuals[level + 1].copy_from_slice(&r_coarse);

        // Clear coarse solution
        for x in solutions[level + 1].iter_mut() {
            *x = T::zero();
        }

        // Recurse
        self.v_cycle(residuals, solutions, level + 1);

        // Interpolate correction
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
            // Coarsest level
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

        // Restrict
        let n_coarse = self.levels[level + 1].matrix.nrows();
        let mut r_coarse = vec![T::zero(); n_coarse];
        self.restrict(level, &r_fine, &mut r_coarse);

        residuals[level + 1].copy_from_slice(&r_coarse);

        for x in solutions[level + 1].iter_mut() {
            *x = T::zero();
        }

        // First recursion
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

        // Second recursion
        self.w_cycle(residuals, solutions, level + 1);

        // Second interpolation
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

        self.matvec(level, x, r);

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

    /// Restrict from fine to coarse grid.
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

    /// Interpolate from coarse to fine grid.
    fn interpolate(&self, level: usize, e_coarse: &[T], e_fine: &mut [T]) {
        if let Some(ref prolongation) = self.levels[level].prolongation {
            let n_fine = prolongation.nrows();

            for i in 0..n_fine {
                let start = prolongation.row_ptrs()[i];
                let end = prolongation.row_ptrs()[i + 1];

                let mut sum = T::zero();
                for k in start..end {
                    let j = prolongation.col_indices()[k];
                    sum = sum + prolongation.values()[k].clone() * e_coarse[j].clone();
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
