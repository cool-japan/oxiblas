//! Block eigenvalue methods for sparse matrices.
//!
//! This module provides block versions of Lanczos and Arnoldi iteration algorithms
//! for computing eigenvalues and eigenvectors of sparse matrices. Block methods
//! are particularly effective when:
//! - Multiple eigenvalues are clustered or degenerate
//! - Computing many eigenvalues simultaneously
//! - The eigenvalue problem has multiple right-hand sides
//!
//! # Block Lanczos
//!
//! The Block Lanczos algorithm extends the standard Lanczos algorithm by using blocks
//! of p vectors instead of single vectors. It is designed for symmetric matrices and
//! builds a block-orthonormal basis that reduces the matrix to block-tridiagonal form.
//!
//! # Block Arnoldi
//!
//! The Block Arnoldi algorithm extends the standard Arnoldi algorithm for general
//! (non-symmetric) matrices. It builds a block-orthonormal basis that reduces the
//! matrix to block upper Hessenberg form.

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::{EigenvalueError, WhichEigenvalues};
use super::utils::{dot, givens_rotation, norm};

// ============================================================================
// Block Lanczos Algorithm
// ============================================================================

/// Configuration for Block Lanczos iteration.
#[derive(Debug, Clone)]
pub struct BlockLanczosConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Block size (number of vectors per block).
    pub block_size: usize,
    /// Which eigenvalues to compute.
    pub which: WhichEigenvalues,
    /// Maximum number of block iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Number of blocks in Krylov subspace (total dim = block_size * num_blocks).
    pub num_blocks: usize,
    /// Whether to use full reorthogonalization (more stable, more expensive).
    pub full_reorthogonalization: bool,
}

impl Default for BlockLanczosConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            block_size: 2,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            num_blocks: 10,
            full_reorthogonalization: true,
        }
    }
}

impl Default for BlockLanczosConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            block_size: 2,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            num_blocks: 10,
            full_reorthogonalization: true,
        }
    }
}

/// Result of Block Lanczos eigenvalue computation.
#[derive(Debug, Clone)]
pub struct BlockLanczosResult<T> {
    /// Computed eigenvalues (Ritz values).
    pub eigenvalues: Vec<T>,
    /// Eigenvectors (Ritz vectors), stored as column vectors.
    /// Only populated if eigenvectors were requested.
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of block iterations performed.
    pub iterations: usize,
    /// Residual norms for each converged eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
}

/// Block Lanczos iteration for sparse symmetric matrices.
///
/// Computes the k largest or smallest eigenvalues (and optionally eigenvectors)
/// of a symmetric matrix A using the Block Lanczos algorithm.
///
/// # Algorithm
///
/// Block Lanczos extends the standard Lanczos algorithm by using blocks of p
/// vectors instead of single vectors. This is particularly effective when:
/// - Multiple eigenvalues are clustered or degenerate
/// - Computing many eigenvalues simultaneously
/// - The eigenvalue problem has multiple right-hand sides
///
/// The algorithm builds a block-orthonormal basis V = [V_0, V_1, ..., V_m]
/// where each V_j is an n x p block, and reduces A to block-tridiagonal form:
///
/// ```text
/// T = V^T A V = [ A_0   B_1^T              ]
///               [ B_1   A_1   B_2^T        ]
///               [       B_2   A_2   ...    ]
///               [            ...          ]
/// ```
///
/// where A_j are p x p symmetric blocks and B_j are p x p blocks.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{BlockLanczos, BlockLanczosConfig, WhichEigenvalues};
///
/// // Create a sparse symmetric matrix
/// let a = CsrMatrix::<f64>::eye(100);
///
/// let config = BlockLanczosConfig {
///     num_eigenvalues: 10,
///     block_size: 3,
///     which: WhichEigenvalues::LargestMagnitude,
///     ..Default::default()
/// };
///
/// let block_lanczos = BlockLanczos::new(config);
/// let result = block_lanczos.compute(&a, None).unwrap();
/// println!("Eigenvalues: {:?}", result.eigenvalues);
/// ```
pub struct BlockLanczos<T> {
    config: BlockLanczosConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> BlockLanczos<T> {
    /// Create a new Block Lanczos solver with the given configuration.
    pub fn new(config: BlockLanczosConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues (and optionally eigenvectors) of a symmetric matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric sparse matrix in CSR format
    /// * `initial_block` - Optional starting block (p columns, random if None)
    ///
    /// # Returns
    ///
    /// Computed eigenvalues and eigenvectors.
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_block: Option<&[Vec<T>]>,
    ) -> Result<BlockLanczosResult<T>, EigenvalueError> {
        let n = a.nrows();
        let p = self.config.block_size;

        if a.ncols() != n {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        let k = self.config.num_eigenvalues;
        let num_blocks = self.config.num_blocks.max(k.div_ceil(p) + 1).min(n / p);

        if k > n {
            return Err(EigenvalueError::TooManyEigenvalues {
                requested: k,
                max_allowed: n,
            });
        }

        if p > n {
            return Err(EigenvalueError::ComputationError(format!(
                "Block size {} exceeds matrix dimension {}",
                p, n
            )));
        }

        // Initialize starting block V_0 (n x p)
        let mut v_block: Vec<Vec<T>> = if let Some(v0) = initial_block {
            if v0.len() != p {
                return Err(EigenvalueError::ComputationError(format!(
                    "Initial block has {} columns, expected {}",
                    v0.len(),
                    p
                )));
            }
            for v in v0 {
                if v.len() != n {
                    return Err(EigenvalueError::DimensionMismatch {
                        expected: n,
                        actual: v.len(),
                    });
                }
            }
            v0.to_vec()
        } else {
            // Generate initial block using deterministic pattern
            let mut block = Vec::with_capacity(p);
            for j in 0..p {
                let mut v = vec![T::zero(); n];
                // Use different starting patterns for each column
                for i in 0..n {
                    let val = T::from_f64(((i + j * 7 + 1) % 17) as f64 / 17.0 - 0.5)
                        .unwrap_or_else(T::one);
                    v[i] = val;
                }
                block.push(v);
            }
            block
        };

        // Orthonormalize initial block using modified Gram-Schmidt with QR
        self.orthonormalize_block(&mut v_block)?;

        // Storage for all Lanczos blocks (list of blocks, each block is p vectors of length n)
        let mut lanczos_blocks: Vec<Vec<Vec<T>>> = Vec::with_capacity(num_blocks);
        lanczos_blocks.push(v_block.clone());

        // Block tridiagonal matrix elements
        // A_blocks: diagonal blocks (p x p symmetric)
        // B_blocks: off-diagonal blocks (p x p)
        let mut a_blocks: Vec<Vec<Vec<T>>> = Vec::with_capacity(num_blocks);
        let mut b_blocks: Vec<Vec<Vec<T>>> = Vec::with_capacity(num_blocks);

        // Previous block for recurrence
        let mut v_prev: Vec<Vec<T>> = vec![vec![T::zero(); n]; p];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(1000.0).unwrap_or_else(T::one);

        // Block Lanczos iteration
        let mut actual_blocks = 0;
        for j in 0..num_blocks {
            // W = A * V_j (apply A to each column of V_j)
            let mut w_block: Vec<Vec<T>> = Vec::with_capacity(p);
            for v in &v_block {
                let mut w = vec![T::zero(); n];
                spmv(T::one(), a, v, T::zero(), &mut w);
                w_block.push(w);
            }

            // Compute A_j = V_j^T * W (p x p block)
            let a_j = self.compute_block_inner_product(&v_block, &w_block);
            a_blocks.push(a_j.clone());

            // W = W - V_j * A_j
            for (l, w) in w_block.iter_mut().enumerate() {
                for (m, v) in v_block.iter().enumerate() {
                    let a_lm = a_j[l][m].clone();
                    for i in 0..n {
                        w[i] = w[i].clone() - a_lm.clone() * v[i].clone();
                    }
                }
            }

            // W = W - V_{j-1} * B_j^T (if j > 0)
            if j > 0 && !b_blocks.is_empty() {
                let b_j = &b_blocks[j - 1];
                for (l, w) in w_block.iter_mut().enumerate() {
                    for (m, v) in v_prev.iter().enumerate() {
                        // B_j^T[l][m] = B_j[m][l]
                        let b_ml = b_j[m][l].clone();
                        for i in 0..n {
                            w[i] = w[i].clone() - b_ml.clone() * v[i].clone();
                        }
                    }
                }
            }

            // Full reorthogonalization against all previous blocks
            if self.config.full_reorthogonalization {
                for prev_block in &lanczos_blocks {
                    self.orthogonalize_against_block(&mut w_block, prev_block);
                }
            }

            // QR factorization of W to get V_{j+1} and B_{j+1}
            let (q_block, r_block) = self.qr_factorization_block(&w_block)?;

            // Check for breakdown (all columns have small norm)
            let mut max_r_diag = T::zero();
            for l in 0..p.min(r_block.len()) {
                if l < r_block[l].len() {
                    let abs_r = Scalar::abs(r_block[l][l].clone());
                    if abs_r > max_r_diag {
                        max_r_diag = abs_r;
                    }
                }
            }

            if max_r_diag <= tol_breakdown {
                // Breakdown - invariant subspace found
                actual_blocks = j + 1;
                break;
            }

            // Store B_{j+1} = R
            b_blocks.push(r_block);

            // Update for next iteration
            v_prev = v_block;
            v_block = q_block;

            if j + 1 < num_blocks {
                lanczos_blocks.push(v_block.clone());
            }

            actual_blocks = j + 1;
        }

        // Build the full block tridiagonal matrix and solve for eigenvalues
        let (ritz_values, ritz_vectors) =
            self.solve_block_tridiagonal(&a_blocks, &b_blocks, p, actual_blocks)?;

        // Select which eigenvalues to return based on configuration
        let (selected_indices, selected_eigenvalues) = self.select_eigenvalues(&ritz_values, k);

        // Compute residual norms
        let mut residual_norms = Vec::with_capacity(k);
        let mut converged_count = 0;

        for idx in &selected_indices {
            // Approximate residual norm using the last block
            let residual = if *idx < ritz_vectors.len() && !b_blocks.is_empty() {
                // Residual ~ ||B_m * s_last|| where s_last is the last p components
                let last_b = b_blocks.last().expect("collection should be non-empty");
                let y = &ritz_vectors[*idx];
                let total_dim = actual_blocks * p;

                if y.len() >= p && total_dim >= p {
                    let start_idx = total_dim.saturating_sub(p);
                    let end_idx = total_dim.min(y.len());

                    let mut res_sq = T::zero();
                    for (l, row) in last_b.iter().enumerate().take(p) {
                        let mut sum = T::zero();
                        for (m, b_lm) in row.iter().enumerate().take(p) {
                            if start_idx + m < end_idx {
                                sum = sum + b_lm.clone() * y[start_idx + m].clone();
                            }
                        }
                        res_sq = res_sq + sum.clone() * sum;
                        let _ = l;
                    }
                    Real::sqrt(res_sq)
                } else {
                    T::zero()
                }
            } else {
                T::zero()
            };

            residual_norms.push(residual.clone());
            if residual <= self.config.tolerance {
                converged_count += 1;
            }
        }

        // Compute eigenvectors if requested
        let eigenvectors = if self.config.compute_eigenvectors {
            let mut evecs = Vec::with_capacity(selected_indices.len());

            for &idx in &selected_indices {
                if idx < ritz_vectors.len() {
                    // Transform Ritz vector from block Lanczos basis to original basis
                    // x = V * y where V = [V_0, V_1, ..., V_m] and y is the Ritz vector
                    let y = &ritz_vectors[idx];
                    let mut x = vec![T::zero(); n];

                    for (block_idx, block) in lanczos_blocks.iter().enumerate() {
                        for (col_idx, v) in block.iter().enumerate() {
                            let y_idx = block_idx * p + col_idx;
                            if y_idx < y.len() {
                                for i in 0..n {
                                    x[i] = x[i].clone() + y[y_idx].clone() * v[i].clone();
                                }
                            }
                        }
                    }

                    // Normalize eigenvector
                    let x_norm = norm(&x);
                    if x_norm > <T as Scalar>::epsilon() {
                        for xi in &mut x {
                            *xi = xi.clone() / x_norm.clone();
                        }
                    }

                    evecs.push(x);
                }
            }

            Some(evecs)
        } else {
            None
        };

        let converged = converged_count >= k;

        Ok(BlockLanczosResult {
            eigenvalues: selected_eigenvalues,
            eigenvectors,
            iterations: actual_blocks,
            residual_norms,
            converged,
        })
    }

    /// Orthonormalize a block of vectors using modified Gram-Schmidt.
    fn orthonormalize_block(&self, block: &mut [Vec<T>]) -> Result<(), EigenvalueError> {
        let p = block.len();
        if p == 0 {
            return Ok(());
        }

        for j in 0..p {
            // Orthogonalize against previous vectors
            for i in 0..j {
                let h = dot(&block[i], &block[j]);
                let n_len = block[j].len();
                for k in 0..n_len {
                    block[j][k] = block[j][k].clone() - h.clone() * block[i][k].clone();
                }
            }

            // Normalize
            let v_norm = norm(&block[j]);
            if v_norm <= <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one) {
                // Replace with a new random-like vector
                let n_len = block[j].len();
                for k in 0..n_len {
                    block[j][k] = T::from_f64(((k + j * 11 + 3) % 13) as f64 / 13.0 - 0.5)
                        .unwrap_or_else(T::one);
                }
                // Re-orthogonalize
                for i in 0..j {
                    let h = dot(&block[i], &block[j]);
                    for k in 0..n_len {
                        block[j][k] = block[j][k].clone() - h.clone() * block[i][k].clone();
                    }
                }
                let v_norm2 = norm(&block[j]);
                if v_norm2 > <T as Scalar>::epsilon() {
                    for k in 0..n_len {
                        block[j][k] = block[j][k].clone() / v_norm2.clone();
                    }
                }
            } else {
                let n_len = block[j].len();
                for k in 0..n_len {
                    block[j][k] = block[j][k].clone() / v_norm.clone();
                }
            }
        }

        Ok(())
    }

    /// Compute inner product of two blocks: result[i][j] = block1[i] . block2[j]
    fn compute_block_inner_product(&self, block1: &[Vec<T>], block2: &[Vec<T>]) -> Vec<Vec<T>> {
        let p1 = block1.len();
        let p2 = block2.len();

        let mut result = vec![vec![T::zero(); p2]; p1];

        for i in 0..p1 {
            for j in 0..p2 {
                result[i][j] = dot(&block1[i], &block2[j]);
            }
        }

        result
    }

    /// Orthogonalize w_block against a previous block.
    fn orthogonalize_against_block(&self, w_block: &mut [Vec<T>], prev_block: &[Vec<T>]) {
        let n = if w_block.is_empty() {
            0
        } else {
            w_block[0].len()
        };

        for w in w_block.iter_mut() {
            for v in prev_block {
                let h = dot(v, w);
                for i in 0..n {
                    w[i] = w[i].clone() - h.clone() * v[i].clone();
                }
            }
        }
    }

    /// QR factorization of a block of vectors.
    /// Returns (Q, R) where Q is orthonormal block and R is upper triangular.
    fn qr_factorization_block(
        &self,
        block: &[Vec<T>],
    ) -> Result<(Vec<Vec<T>>, Vec<Vec<T>>), EigenvalueError> {
        let p = block.len();
        if p == 0 {
            return Ok((vec![], vec![]));
        }

        let n = block[0].len();

        // Copy block to Q
        let mut q: Vec<Vec<T>> = block.iter().cloned().collect();

        // R matrix (p x p upper triangular)
        let mut r = vec![vec![T::zero(); p]; p];

        // Modified Gram-Schmidt QR
        for j in 0..p {
            // Orthogonalize against previous Q columns
            for i in 0..j {
                r[i][j] = dot(&q[i], &q[j]);
                for k in 0..n {
                    q[j][k] = q[j][k].clone() - r[i][j].clone() * q[i][k].clone();
                }
            }

            // Normalize
            r[j][j] = norm(&q[j]);
            if Scalar::abs(r[j][j].clone())
                > <T as Scalar>::epsilon() * T::from_f64(10.0).unwrap_or_else(T::one)
            {
                for k in 0..n {
                    q[j][k] = q[j][k].clone() / r[j][j].clone();
                }
            } else {
                // Zero column - fill with something orthogonal
                for k in 0..n {
                    q[j][k] = T::zero();
                }
                if j < n {
                    q[j][j] = T::one();
                }
            }
        }

        Ok((q, r))
    }

    /// Solve eigenvalue problem for block tridiagonal matrix.
    fn solve_block_tridiagonal(
        &self,
        a_blocks: &[Vec<Vec<T>>],
        b_blocks: &[Vec<Vec<T>>],
        p: usize,
        num_blocks: usize,
    ) -> Result<(Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        if num_blocks == 0 || a_blocks.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Build full tridiagonal matrix as dense matrix
        let total_dim = num_blocks * p;
        let mut t_matrix = vec![vec![T::zero(); total_dim]; total_dim];

        // Fill diagonal blocks A_j
        for (block_idx, a_j) in a_blocks.iter().enumerate().take(num_blocks) {
            let row_offset = block_idx * p;
            for i in 0..p.min(a_j.len()) {
                for j in 0..p.min(a_j[i].len()) {
                    t_matrix[row_offset + i][row_offset + j] = a_j[i][j].clone();
                }
            }
        }

        // Fill off-diagonal blocks B_j and B_j^T
        for (block_idx, b_j) in b_blocks.iter().enumerate() {
            if block_idx + 1 >= num_blocks {
                break;
            }
            let row_offset = block_idx * p;
            let col_offset = (block_idx + 1) * p;

            for i in 0..p.min(b_j.len()) {
                for j in 0..p.min(b_j[i].len()) {
                    // B_j at position (block_idx, block_idx+1)
                    if row_offset + i < total_dim && col_offset + j < total_dim {
                        t_matrix[row_offset + i][col_offset + j] = b_j[i][j].clone();
                    }
                    // B_j^T at position (block_idx+1, block_idx)
                    if col_offset + i < total_dim && row_offset + j < total_dim {
                        t_matrix[col_offset + i][row_offset + j] = b_j[j][i].clone();
                    }
                }
            }
        }

        // Solve eigenvalue problem for dense symmetric matrix using QR iteration
        self.solve_dense_symmetric(&t_matrix, total_dim)
    }

    /// Solve eigenvalue problem for dense symmetric matrix using QR iteration.
    fn solve_dense_symmetric(
        &self,
        matrix: &[Vec<T>],
        n: usize,
    ) -> Result<(Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        // First reduce to tridiagonal form using Householder reflections
        let (tridiag_d, tridiag_e, q_transform) = self.tridiagonalize(matrix, n);

        // Then solve tridiagonal eigenvalue problem
        let (eigenvalues, eigenvectors_tridiag) =
            self.solve_tridiagonal_qr(&tridiag_d, &tridiag_e)?;

        // Transform eigenvectors back: V = Q * V_tridiag
        let mut eigenvectors = Vec::with_capacity(n);
        for k in 0..n {
            let mut v = vec![T::zero(); n];
            for i in 0..n {
                for j in 0..n {
                    if k < eigenvectors_tridiag.len() && j < eigenvectors_tridiag[k].len() {
                        v[i] = v[i].clone()
                            + q_transform[i][j].clone() * eigenvectors_tridiag[k][j].clone();
                    }
                }
            }
            eigenvectors.push(v);
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Reduce symmetric matrix to tridiagonal form using Householder reflections.
    fn tridiagonalize(&self, a: &[Vec<T>], n: usize) -> (Vec<T>, Vec<T>, Vec<Vec<T>>) {
        // Copy matrix
        let mut h: Vec<Vec<T>> = a.iter().take(n).map(|row| row[..n].to_vec()).collect();

        // Initialize Q as identity
        let mut q: Vec<Vec<T>> = (0..n)
            .map(|i| {
                let mut row = vec![T::zero(); n];
                row[i] = T::one();
                row
            })
            .collect();

        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for k in 0..n.saturating_sub(2) {
            // Compute Householder vector for column k below diagonal
            let mut x = vec![T::zero(); n - k - 1];
            for i in 0..n - k - 1 {
                x[i] = h[k + 1 + i][k].clone();
            }

            let x_norm = norm(&x);
            if x_norm <= <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one) {
                continue;
            }

            // v = x + sign(x[0]) * ||x|| * e_1
            let mut v = x.clone();
            let sign = if v[0] >= T::zero() {
                T::one()
            } else {
                T::zero() - T::one()
            };
            v[0] = v[0].clone() + sign * x_norm.clone();

            let v_norm = norm(&v);
            if v_norm <= <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one) {
                continue;
            }

            // Normalize v
            for vi in &mut v {
                *vi = vi.clone() / v_norm.clone();
            }

            // Apply Householder reflection: H = I - 2*v*v^T
            // A' = H * A * H

            // First compute A * H for columns k+1:n
            for j in k..n {
                // Compute v^T * a[k+1:n, j]
                let mut dot_product = T::zero();
                for i in 0..n - k - 1 {
                    dot_product = dot_product + v[i].clone() * h[k + 1 + i][j].clone();
                }

                // a[k+1:n, j] -= 2 * (v^T * a) * v
                for i in 0..n - k - 1 {
                    h[k + 1 + i][j] =
                        h[k + 1 + i][j].clone() - two.clone() * dot_product.clone() * v[i].clone();
                }
            }

            // Then compute H * A for rows k+1:n
            for i in k..n {
                let mut dot_product = T::zero();
                for j in 0..n - k - 1 {
                    dot_product = dot_product + v[j].clone() * h[i][k + 1 + j].clone();
                }

                for j in 0..n - k - 1 {
                    h[i][k + 1 + j] =
                        h[i][k + 1 + j].clone() - two.clone() * dot_product.clone() * v[j].clone();
                }
            }

            // Accumulate Q = Q * H
            for i in 0..n {
                let mut dot_product = T::zero();
                for j in 0..n - k - 1 {
                    dot_product = dot_product + v[j].clone() * q[i][k + 1 + j].clone();
                }

                for j in 0..n - k - 1 {
                    q[i][k + 1 + j] =
                        q[i][k + 1 + j].clone() - two.clone() * dot_product.clone() * v[j].clone();
                }
            }
        }

        // Extract diagonal and off-diagonal
        let mut d = vec![T::zero(); n];
        let mut e = vec![T::zero(); n.saturating_sub(1)];

        for i in 0..n {
            d[i] = h[i][i].clone();
        }
        for i in 0..n.saturating_sub(1) {
            e[i] = h[i + 1][i].clone();
        }

        (d, e, q)
    }

    /// Solve tridiagonal eigenvalue problem using QR iteration with implicit shifts.
    fn solve_tridiagonal_qr(
        &self,
        d: &[T],
        e: &[T],
    ) -> Result<(Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        let n = d.len();
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        // Copy diagonal and off-diagonal
        let mut diag = d.to_vec();
        let mut off_diag = if e.is_empty() {
            vec![T::zero(); n.saturating_sub(1)]
        } else {
            let mut od = e.to_vec();
            od.truncate(n.saturating_sub(1));
            while od.len() < n.saturating_sub(1) {
                od.push(T::zero());
            }
            od
        };

        // Initialize eigenvector matrix as identity
        let mut z: Vec<Vec<T>> = (0..n)
            .map(|i| {
                let mut row = vec![T::zero(); n];
                row[i] = T::one();
                row
            })
            .collect();

        let max_iter = 30 * n;
        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for _iter in 0..max_iter {
            // Find largest unreduced submatrix
            let mut l = 0;
            for i in (0..n.saturating_sub(1)).rev() {
                if Scalar::abs(off_diag[i].clone())
                    <= tol.clone()
                        * (Scalar::abs(diag[i].clone()) + Scalar::abs(diag[i + 1].clone()))
                {
                    off_diag[i] = T::zero();
                } else {
                    l = i + 1;
                    break;
                }
            }

            if l == 0 {
                break;
            }

            // Find start of unreduced block
            let mut m = l;
            for i in (0..l).rev() {
                if Scalar::abs(off_diag[i].clone())
                    <= tol.clone()
                        * (Scalar::abs(diag[i].clone()) + Scalar::abs(diag[i + 1].clone()))
                {
                    off_diag[i] = T::zero();
                    m = i + 1;
                    break;
                }
                if i == 0 {
                    m = 0;
                }
            }

            // Wilkinson shift
            let p = (diag[l - 1].clone() - diag[l].clone()) / two.clone();
            let r = Real::sqrt(
                p.clone() * p.clone() + off_diag[l - 1].clone() * off_diag[l - 1].clone(),
            );
            let shift = if p >= T::zero() {
                diag[l].clone()
                    - off_diag[l - 1].clone() * off_diag[l - 1].clone() / (p.clone() + r)
            } else {
                diag[l].clone()
                    - off_diag[l - 1].clone() * off_diag[l - 1].clone() / (p.clone() - r)
            };

            // Apply implicit QR step
            let mut g = diag[m].clone() - shift.clone();
            let mut s = T::one();
            let mut c = T::one();
            let mut p_val = T::zero();

            for i in m..l {
                let f = s.clone() * off_diag[i].clone();
                let b = c.clone() * off_diag[i].clone();

                // Givens rotation
                if Scalar::abs(f.clone()) >= Scalar::abs(g.clone()) {
                    c = g.clone() / f.clone();
                    let r = Real::sqrt(c.clone() * c.clone() + T::one());
                    if i > m {
                        off_diag[i - 1] = f.clone() * r.clone();
                    }
                    s = T::one() / r.clone();
                    c = c * s.clone();
                } else {
                    s = f.clone() / g.clone();
                    let r = Real::sqrt(s.clone() * s.clone() + T::one());
                    if i > m {
                        off_diag[i - 1] = g.clone() * r.clone();
                    }
                    c = T::one() / r.clone();
                    s = s * c.clone();
                }

                g = diag[i].clone() - p_val.clone();
                let r = (diag[i + 1].clone() - g.clone()) * s.clone()
                    + two.clone() * c.clone() * b.clone();
                p_val = s.clone() * r.clone();
                diag[i] = g.clone() + p_val.clone();
                g = c.clone() * r.clone() - b;

                // Update eigenvectors
                for k in 0..n {
                    let temp = z[k][i + 1].clone();
                    z[k][i + 1] = s.clone() * z[k][i].clone() + c.clone() * temp.clone();
                    z[k][i] = c.clone() * z[k][i].clone() - s.clone() * temp;
                }
            }

            diag[l] = diag[l].clone() - p_val;
            off_diag[l - 1] = g;
        }

        // Extract eigenvalues and eigenvectors
        let eigenvalues = diag;
        let eigenvectors: Vec<Vec<T>> = (0..n)
            .map(|i| z.iter().map(|row| row[i].clone()).collect())
            .collect();

        Ok((eigenvalues, eigenvectors))
    }

    /// Select eigenvalues based on configuration.
    fn select_eigenvalues(&self, eigenvalues: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        if eigenvalues.is_empty() {
            return (vec![], vec![]);
        }

        let n = eigenvalues.len();
        let k = k.min(n);

        let mut indexed: Vec<(usize, T)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect();

        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|a, b| {
                    Scalar::abs(a.1.clone())
                        .partial_cmp(&Scalar::abs(b.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WhichEigenvalues::LargestAlgebraic => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestAlgebraic => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::NearTarget => {
                indexed.sort_by(|a, b| {
                    Scalar::abs(a.1.clone())
                        .partial_cmp(&Scalar::abs(b.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        let values: Vec<T> = indexed.iter().take(k).map(|(_, v)| v.clone()).collect();

        (indices, values)
    }
}

// ============================================================================
// Block Arnoldi Algorithm
// ============================================================================

/// Configuration for Block Arnoldi iteration.
#[derive(Debug, Clone)]
pub struct BlockArnoldiConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Block size (number of vectors per block).
    pub block_size: usize,
    /// Which eigenvalues to compute.
    pub which: WhichEigenvalues,
    /// Maximum number of block iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Number of blocks in Krylov subspace (total dim = block_size * num_blocks).
    pub num_blocks: usize,
}

impl Default for BlockArnoldiConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            block_size: 2,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            num_blocks: 10,
        }
    }
}

impl Default for BlockArnoldiConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            block_size: 2,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            num_blocks: 10,
        }
    }
}

/// Result of Block Arnoldi eigenvalue computation.
#[derive(Debug, Clone)]
pub struct BlockArnoldiResult<T> {
    /// Real parts of computed eigenvalues.
    pub eigenvalues_real: Vec<T>,
    /// Imaginary parts of computed eigenvalues.
    pub eigenvalues_imag: Vec<T>,
    /// Eigenvectors (if requested), stored as column vectors.
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of block iterations performed.
    pub iterations: usize,
    /// Residual norms for each eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
}

/// Block Arnoldi iteration for sparse general (non-symmetric) matrices.
///
/// Computes the k largest or smallest eigenvalues (and optionally eigenvectors)
/// of a general square matrix A using the Block Arnoldi algorithm.
///
/// # Algorithm
///
/// Block Arnoldi extends the standard Arnoldi algorithm by using blocks of p
/// vectors instead of single vectors. This is particularly effective when:
/// - Multiple eigenvalues are clustered
/// - Computing many eigenvalues simultaneously
/// - The matrix has multiple repeated eigenvalues
///
/// The algorithm builds a block-orthonormal basis V = [V_0, V_1, ..., V_m]
/// where each V_j is an n x p block, and reduces A to block upper Hessenberg form.
pub struct BlockArnoldi<T> {
    config: BlockArnoldiConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> BlockArnoldi<T> {
    /// Create a new Block Arnoldi solver with the given configuration.
    pub fn new(config: BlockArnoldiConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues (and optionally eigenvectors) of a general matrix.
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_block: Option<&[Vec<T>]>,
    ) -> Result<BlockArnoldiResult<T>, EigenvalueError> {
        let n = a.nrows();
        let p = self.config.block_size;

        if a.ncols() != n {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        let k = self.config.num_eigenvalues;
        let num_blocks = self.config.num_blocks.max(k.div_ceil(p) + 1).min(n / p);

        if k > n {
            return Err(EigenvalueError::TooManyEigenvalues {
                requested: k,
                max_allowed: n,
            });
        }

        if p > n {
            return Err(EigenvalueError::ComputationError(format!(
                "Block size {} exceeds matrix dimension {}",
                p, n
            )));
        }

        // Initialize starting block V_0 (n x p)
        let mut v_block: Vec<Vec<T>> = if let Some(v0) = initial_block {
            if v0.len() != p {
                return Err(EigenvalueError::ComputationError(format!(
                    "Initial block has {} columns, expected {}",
                    v0.len(),
                    p
                )));
            }
            for v in v0 {
                if v.len() != n {
                    return Err(EigenvalueError::DimensionMismatch {
                        expected: n,
                        actual: v.len(),
                    });
                }
            }
            v0.to_vec()
        } else {
            let mut block = Vec::with_capacity(p);
            for j in 0..p {
                let mut v = vec![T::zero(); n];
                for i in 0..n {
                    let val = T::from_f64(((i + j * 7 + 1) % 17) as f64 / 17.0 - 0.5)
                        .unwrap_or_else(T::one);
                    v[i] = val;
                }
                block.push(v);
            }
            block
        };

        // Orthonormalize initial block
        self.orthonormalize_block(&mut v_block)?;

        // Storage for all Arnoldi blocks
        let mut arnoldi_blocks: Vec<Vec<Vec<T>>> = Vec::with_capacity(num_blocks);
        arnoldi_blocks.push(v_block.clone());

        // Block upper Hessenberg matrix H
        let mut h_blocks: Vec<Vec<Vec<Vec<T>>>> =
            vec![vec![vec![vec![T::zero(); p]; p]; num_blocks]; num_blocks + 1];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(1000.0).unwrap_or_else(T::one);

        // Block Arnoldi iteration
        let mut actual_blocks = 0;
        for j in 0..num_blocks {
            // W = A * V_j
            let mut w_block: Vec<Vec<T>> = Vec::with_capacity(p);
            for v in &v_block {
                let mut w = vec![T::zero(); n];
                spmv(T::one(), a, v, T::zero(), &mut w);
                w_block.push(w);
            }

            // Modified Gram-Schmidt against all previous blocks
            for i in 0..=j {
                let h_ij = self.compute_block_inner_product(&arnoldi_blocks[i], &w_block);
                h_blocks[i][j] = h_ij.clone();

                for (l, w) in w_block.iter_mut().enumerate() {
                    for (m, v) in arnoldi_blocks[i].iter().enumerate() {
                        let h_lm = h_ij[l][m].clone();
                        for ii in 0..n {
                            w[ii] = w[ii].clone() - h_lm.clone() * v[ii].clone();
                        }
                    }
                }
            }

            // QR factorization of W
            let (q_block, r_block) = self.qr_factorization_block(&w_block)?;
            h_blocks[j + 1][j] = r_block.clone();

            // Check for breakdown
            let mut max_r_diag = T::zero();
            for l in 0..p.min(r_block.len()) {
                if l < r_block[l].len() {
                    let abs_r = Scalar::abs(r_block[l][l].clone());
                    if abs_r > max_r_diag {
                        max_r_diag = abs_r;
                    }
                }
            }

            if max_r_diag <= tol_breakdown {
                actual_blocks = j + 1;
                break;
            }

            v_block = q_block;
            if j + 1 < num_blocks {
                arnoldi_blocks.push(v_block.clone());
            }
            actual_blocks = j + 1;
        }

        // Solve eigenvalue problem for block Hessenberg matrix
        let (eigenvalues_real, eigenvalues_imag, eigenvectors_h) =
            self.solve_block_hessenberg(&h_blocks, p, actual_blocks)?;

        // Select eigenvalues
        let (selected_indices, selected_real, selected_imag) =
            self.select_eigenvalues_complex(&eigenvalues_real, &eigenvalues_imag, k);

        // Compute residual norms
        let mut residual_norms = Vec::with_capacity(k);
        let mut converged_count = 0;

        for idx in &selected_indices {
            let residual = if *idx < eigenvectors_h.len() && actual_blocks > 0 {
                let last_h = &h_blocks[actual_blocks][actual_blocks - 1];
                let y = &eigenvectors_h[*idx];
                let total_dim = actual_blocks * p;

                if y.len() >= p && total_dim >= p {
                    let start_idx = total_dim.saturating_sub(p);
                    let end_idx = total_dim.min(y.len());

                    let mut res_sq = T::zero();
                    for row in last_h.iter().take(p) {
                        let mut sum = T::zero();
                        for (m, h_lm) in row.iter().enumerate().take(p) {
                            if start_idx + m < end_idx {
                                sum = sum + h_lm.clone() * y[start_idx + m].clone();
                            }
                        }
                        res_sq = res_sq + sum.clone() * sum;
                    }
                    Real::sqrt(res_sq)
                } else {
                    T::zero()
                }
            } else {
                T::zero()
            };

            residual_norms.push(residual.clone());
            if residual <= self.config.tolerance {
                converged_count += 1;
            }
        }

        // Compute eigenvectors if requested
        let eigenvectors = if self.config.compute_eigenvectors {
            let mut evecs = Vec::with_capacity(selected_indices.len());

            for &idx in &selected_indices {
                if idx < eigenvectors_h.len() {
                    let y = &eigenvectors_h[idx];
                    let mut x = vec![T::zero(); n];

                    for (block_idx, block) in arnoldi_blocks.iter().enumerate() {
                        for (col_idx, v) in block.iter().enumerate() {
                            let y_idx = block_idx * p + col_idx;
                            if y_idx < y.len() {
                                for i in 0..n {
                                    x[i] = x[i].clone() + y[y_idx].clone() * v[i].clone();
                                }
                            }
                        }
                    }

                    let x_norm = norm(&x);
                    if x_norm > <T as Scalar>::epsilon() {
                        for xi in &mut x {
                            *xi = xi.clone() / x_norm.clone();
                        }
                    }
                    evecs.push(x);
                }
            }
            Some(evecs)
        } else {
            None
        };

        let converged = converged_count >= k;

        Ok(BlockArnoldiResult {
            eigenvalues_real: selected_real,
            eigenvalues_imag: selected_imag,
            eigenvectors,
            iterations: actual_blocks,
            residual_norms,
            converged,
        })
    }

    /// Orthonormalize a block of vectors using modified Gram-Schmidt.
    fn orthonormalize_block(&self, block: &mut [Vec<T>]) -> Result<(), EigenvalueError> {
        let p = block.len();
        if p == 0 {
            return Ok(());
        }

        for j in 0..p {
            for i in 0..j {
                let h = dot(&block[i], &block[j]);
                let n_len = block[j].len();
                for k in 0..n_len {
                    block[j][k] = block[j][k].clone() - h.clone() * block[i][k].clone();
                }
            }

            let v_norm = norm(&block[j]);
            if v_norm <= <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one) {
                let n_len = block[j].len();
                for k in 0..n_len {
                    block[j][k] = T::from_f64(((k + j * 11 + 3) % 13) as f64 / 13.0 - 0.5)
                        .unwrap_or_else(T::one);
                }
                for i in 0..j {
                    let h = dot(&block[i], &block[j]);
                    for k in 0..n_len {
                        block[j][k] = block[j][k].clone() - h.clone() * block[i][k].clone();
                    }
                }
                let v_norm2 = norm(&block[j]);
                if v_norm2 > <T as Scalar>::epsilon() {
                    for k in 0..n_len {
                        block[j][k] = block[j][k].clone() / v_norm2.clone();
                    }
                }
            } else {
                let n_len = block[j].len();
                for k in 0..n_len {
                    block[j][k] = block[j][k].clone() / v_norm.clone();
                }
            }
        }
        Ok(())
    }

    /// Compute inner product of two blocks: result[i][j] = block1[i] . block2[j]
    fn compute_block_inner_product(&self, block1: &[Vec<T>], block2: &[Vec<T>]) -> Vec<Vec<T>> {
        let p1 = block1.len();
        let p2 = block2.len();
        let mut result = vec![vec![T::zero(); p2]; p1];
        for i in 0..p1 {
            for j in 0..p2 {
                result[i][j] = dot(&block1[i], &block2[j]);
            }
        }
        result
    }

    /// QR factorization of a block of vectors.
    fn qr_factorization_block(
        &self,
        block: &[Vec<T>],
    ) -> Result<(Vec<Vec<T>>, Vec<Vec<T>>), EigenvalueError> {
        let p = block.len();
        if p == 0 {
            return Ok((vec![], vec![]));
        }
        let n = block[0].len();
        let mut q: Vec<Vec<T>> = block.iter().cloned().collect();
        let mut r = vec![vec![T::zero(); p]; p];

        for j in 0..p {
            for i in 0..j {
                r[i][j] = dot(&q[i], &q[j]);
                for k in 0..n {
                    q[j][k] = q[j][k].clone() - r[i][j].clone() * q[i][k].clone();
                }
            }
            r[j][j] = norm(&q[j]);
            if Scalar::abs(r[j][j].clone())
                > <T as Scalar>::epsilon() * T::from_f64(10.0).unwrap_or_else(T::one)
            {
                for k in 0..n {
                    q[j][k] = q[j][k].clone() / r[j][j].clone();
                }
            } else {
                for k in 0..n {
                    q[j][k] = T::zero();
                }
                if j < n {
                    q[j][j] = T::one();
                }
            }
        }
        Ok((q, r))
    }

    /// Solve eigenvalue problem for block Hessenberg matrix.
    fn solve_block_hessenberg(
        &self,
        h_blocks: &[Vec<Vec<Vec<T>>>],
        p: usize,
        num_blocks: usize,
    ) -> Result<(Vec<T>, Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        if num_blocks == 0 {
            return Ok((vec![], vec![], vec![]));
        }

        let total_dim = num_blocks * p;
        let mut h_matrix = vec![vec![T::zero(); total_dim]; total_dim];

        for block_i in 0..num_blocks.min(h_blocks.len()) {
            for block_j in 0..num_blocks.min(h_blocks[block_i].len()) {
                let h_ij = &h_blocks[block_i][block_j];
                let row_offset = block_i * p;
                let col_offset = block_j * p;

                for i in 0..p.min(h_ij.len()) {
                    for j in 0..p.min(h_ij[i].len()) {
                        if row_offset + i < total_dim && col_offset + j < total_dim {
                            h_matrix[row_offset + i][col_offset + j] = h_ij[i][j].clone();
                        }
                    }
                }
            }
        }

        self.solve_hessenberg_qr(&h_matrix, total_dim)
    }

    /// Solve Hessenberg eigenvalue problem using QR iteration.
    fn solve_hessenberg_qr(
        &self,
        h: &[Vec<T>],
        n: usize,
    ) -> Result<(Vec<T>, Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        if n == 0 {
            return Ok((vec![], vec![], vec![]));
        }

        let mut a: Vec<Vec<T>> = h.iter().take(n).map(|row| row[..n].to_vec()).collect();
        let mut eigenvalues_real = vec![T::zero(); n];
        let mut eigenvalues_imag = vec![T::zero(); n];

        let mut z: Vec<Vec<T>> = (0..n)
            .map(|i| {
                let mut row = vec![T::zero(); n];
                row[i] = T::one();
                row
            })
            .collect();

        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let four = T::from_f64(4.0).unwrap_or_else(T::one);
        let max_iter = 30 * n;

        let mut p_idx = n;

        for _iter in 0..max_iter {
            if p_idx <= 1 {
                if p_idx == 1 {
                    eigenvalues_real[0] = a[0][0].clone();
                }
                break;
            }

            let l = p_idx - 1;
            if Scalar::abs(a[l][l - 1].clone())
                <= tol.clone()
                    * (Scalar::abs(a[l - 1][l - 1].clone()) + Scalar::abs(a[l][l].clone()))
            {
                eigenvalues_real[l] = a[l][l].clone();
                p_idx = l;
                continue;
            }

            if p_idx >= 2
                && l >= 2
                && Scalar::abs(a[l - 1][l - 2].clone())
                    <= tol.clone()
                        * (Scalar::abs(a[l - 2][l - 2].clone())
                            + Scalar::abs(a[l - 1][l - 1].clone()))
            {
                let a11 = a[l - 1][l - 1].clone();
                let a12 = a[l - 1][l].clone();
                let a21 = a[l][l - 1].clone();
                let a22 = a[l][l].clone();

                let trace = a11.clone() + a22.clone();
                let det = a11 * a22 - a12 * a21;
                let disc = trace.clone() * trace.clone() / four.clone() - det;

                if disc >= T::zero() {
                    let sqrt_disc = Real::sqrt(disc);
                    eigenvalues_real[l - 1] = trace.clone() / two.clone() + sqrt_disc.clone();
                    eigenvalues_real[l] = trace / two.clone() - sqrt_disc;
                } else {
                    let sqrt_disc = Real::sqrt(T::zero() - disc);
                    eigenvalues_real[l - 1] = trace.clone() / two.clone();
                    eigenvalues_real[l] = trace / two.clone();
                    eigenvalues_imag[l - 1] = sqrt_disc.clone();
                    eigenvalues_imag[l] = T::zero() - sqrt_disc;
                }

                p_idx = l - 1;
                continue;
            }

            let shift = self.compute_wilkinson_shift_local(&a, p_idx);
            self.qr_step_local(&mut a, &mut z, p_idx, shift);
        }

        let eigenvectors: Vec<Vec<T>> = (0..n)
            .map(|i| z.iter().map(|row| row[i].clone()).collect())
            .collect();

        Ok((eigenvalues_real, eigenvalues_imag, eigenvectors))
    }

    /// Compute Wilkinson shift for QR iteration.
    fn compute_wilkinson_shift_local(&self, a: &[Vec<T>], p: usize) -> T {
        if p < 2 {
            return a[p - 1][p - 1].clone();
        }

        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let four = T::from_f64(4.0).unwrap_or_else(T::one);

        let n_idx = p;
        let a11 = a[n_idx - 2][n_idx - 2].clone();
        let a12 = a[n_idx - 2][n_idx - 1].clone();
        let a21 = a[n_idx - 1][n_idx - 2].clone();
        let a22 = a[n_idx - 1][n_idx - 1].clone();

        let trace = a11.clone() + a22.clone();
        let det = a11 * a22 - a12 * a21;
        let disc = trace.clone() * trace.clone() / four.clone() - det;

        if disc >= T::zero() {
            let sqrt_disc = Real::sqrt(disc);
            let lambda1 = trace.clone() / two.clone() + sqrt_disc.clone();
            let lambda2 = trace / two.clone() - sqrt_disc;

            let corner = a[n_idx - 1][n_idx - 1].clone();
            if Scalar::abs(lambda1.clone() - corner.clone()) < Scalar::abs(lambda2.clone() - corner)
            {
                lambda1
            } else {
                lambda2
            }
        } else {
            trace / two
        }
    }

    /// Perform one QR step with shift.
    fn qr_step_local(&self, a: &mut [Vec<T>], z: &mut [Vec<T>], p: usize, shift: T) {
        for i in 0..p {
            a[i][i] = a[i][i].clone() - shift.clone();
        }

        for i in 0..p - 1 {
            if Scalar::abs(a[i + 1][i].clone()) <= <T as Scalar>::epsilon() {
                continue;
            }

            let (c, s, r) = givens_rotation(a[i][i].clone(), a[i + 1][i].clone());
            a[i][i] = r;
            a[i + 1][i] = T::zero();

            for j in i + 1..p {
                let temp = c.clone() * a[i][j].clone() + s.clone() * a[i + 1][j].clone();
                a[i + 1][j] =
                    T::zero() - s.clone() * a[i][j].clone() + c.clone() * a[i + 1][j].clone();
                a[i][j] = temp;
            }

            let col_end = (i + 3).min(p);
            for j in 0..col_end {
                let temp = c.clone() * a[j][i].clone() + s.clone() * a[j][i + 1].clone();
                a[j][i + 1] =
                    T::zero() - s.clone() * a[j][i].clone() + c.clone() * a[j][i + 1].clone();
                a[j][i] = temp;
            }

            for j in 0..z.len() {
                let temp = c.clone() * z[j][i].clone() + s.clone() * z[j][i + 1].clone();
                z[j][i + 1] =
                    T::zero() - s.clone() * z[j][i].clone() + c.clone() * z[j][i + 1].clone();
                z[j][i] = temp;
            }
        }

        for i in 0..p {
            a[i][i] = a[i][i].clone() + shift.clone();
        }
    }

    /// Select eigenvalues based on configuration for complex eigenvalues.
    fn select_eigenvalues_complex(
        &self,
        real: &[T],
        imag: &[T],
        k: usize,
    ) -> (Vec<usize>, Vec<T>, Vec<T>) {
        let n = real.len();
        if n == 0 {
            return (vec![], vec![], vec![]);
        }

        let k = k.min(n);

        let mut indexed: Vec<(usize, T)> = real
            .iter()
            .zip(imag.iter())
            .enumerate()
            .map(|(i, (r, im))| {
                let mag = Real::sqrt(r.clone() * r.clone() + im.clone() * im.clone());
                (i, mag)
            })
            .collect();

        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                indexed = real
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (i, r.clone()))
                    .collect();
                match self.config.which {
                    WhichEigenvalues::LargestAlgebraic => {
                        indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    _ => {
                        indexed.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }
        }

        let indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        let selected_real: Vec<T> = indices.iter().map(|&i| real[i].clone()).collect();
        let selected_imag: Vec<T> = indices.iter().map(|&i| imag[i].clone()).collect();

        (indices, selected_real, selected_imag)
    }
}
