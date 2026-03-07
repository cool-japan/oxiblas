//! Randomized Eigenvalue Decomposition (EVD) for symmetric matrices.
//!
//! Implements the Halko-Martinsson-Tropp (HMT) randomized algorithm for
//! computing k dominant eigenpairs of a symmetric matrix efficiently.
//!
//! # Algorithm
//!
//! Given symmetric A (n×n), find k dominant eigenpairs:
//! 1. Draw random Gaussian matrix Ω (n×(k+p)) where p is oversampling
//! 2. Power iteration: Y = A^q * Ω (improves accuracy for slow spectral decay)
//! 3. Thin QR of Y → Q (n×(k+p))
//! 4. Project: B = Q^T * A * Q (small (k+p)×(k+p) symmetric matrix)
//! 5. Dense EVD of B → eigenvalues Λ, eigenvectors V
//! 6. Extract k eigenpairs: eigenvalues Λ\[:k\], eigenvectors U = Q * V\[:, :k\]
//!
//! # References
//!
//! - Halko, Martinsson, Tropp: "Finding Structure with Randomness:
//!   Probabilistic Algorithms for Constructing Approximate Matrix
//!   Decompositions" (2011)
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::{RandomizedEvd, RandomizedEvdConfig};
//! use oxiblas_matrix::Mat;
//!
//! // 4×4 symmetric matrix
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 1.0, 0.0, 0.0],
//!     &[1.0, 3.0, 0.0, 0.0],
//!     &[0.0, 0.0, 2.0, 1.0],
//!     &[0.0, 0.0, 1.0, 2.0],
//! ]);
//!
//! let revd = RandomizedEvd::new(RandomizedEvdConfig::new(2)).unwrap();
//! let result = revd.compute(a.as_ref()).unwrap();
//!
//! assert_eq!(result.eigenvalues.len(), 2);
//! assert_eq!(result.eigenvectors.ncols(), 2);
//! ```

use oxiblas_core::scalar::Scalar;
use oxiblas_matrix::{Mat, MatRef};

use crate::evd::symmetric::SymmetricEvd;
use crate::qr::Qr;

/// Which eigenvalues to target in a randomized EVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomizedEvdTarget {
    /// k eigenvalues with largest absolute value (default).
    Largest,
    /// k eigenvalues with smallest absolute value (shift-flip trick).
    Smallest,
}

/// Configuration for randomized EVD.
#[derive(Debug, Clone)]
pub struct RandomizedEvdConfig {
    /// Number of eigenvalues/eigenvectors to compute.
    pub rank: usize,
    /// Oversampling parameter (default 10). More oversampling = better accuracy.
    pub oversampling: usize,
    /// Number of power iterations (default 2). More = slower but more accurate.
    pub power_iters: usize,
    /// Which eigenvalues to compute (default: Largest magnitude).
    pub which: RandomizedEvdTarget,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomizedEvdConfig {
    fn default() -> Self {
        Self {
            rank: 6,
            oversampling: 10,
            power_iters: 2,
            which: RandomizedEvdTarget::Largest,
            seed: 0x1234_5678_9ABC_DEF0,
        }
    }
}

impl RandomizedEvdConfig {
    /// Creates configuration for computing the k largest eigenpairs.
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    /// Sets the oversampling parameter.
    #[must_use]
    pub fn with_oversampling(mut self, oversampling: usize) -> Self {
        self.oversampling = oversampling;
        self
    }

    /// Sets the number of power iterations.
    #[must_use]
    pub fn with_power_iters(mut self, power_iters: usize) -> Self {
        self.power_iters = power_iters;
        self
    }

    /// Sets which eigenvalues to target.
    #[must_use]
    pub fn with_target(mut self, which: RandomizedEvdTarget) -> Self {
        self.which = which;
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Error type for randomized EVD operations.
#[derive(Debug, Clone)]
pub enum RandomizedEvdError {
    /// Invalid configuration (e.g., rank is zero or too large).
    InvalidConfig(String),
    /// Numerical failure during computation.
    NumericalFailure(String),
    /// Dimension mismatch between expected and actual size.
    DimensionMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size received.
        got: usize,
    },
    /// Matrix failed symmetry check.
    NotSymmetric,
}

impl core::fmt::Display for RandomizedEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::NumericalFailure(msg) => write!(f, "Numerical failure: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::NotSymmetric => write!(f, "Matrix is not symmetric"),
        }
    }
}

impl std::error::Error for RandomizedEvdError {}

/// Result of a randomized EVD computation.
#[derive(Debug, Clone)]
pub struct RandomizedEvdResult<T: Scalar> {
    /// Eigenvalues in descending order by magnitude (length k).
    pub eigenvalues: Vec<T>,
    /// Eigenvectors stored as columns (n × k matrix).
    pub eigenvectors: Mat<T>,
    /// Estimated relative error ||A - U·Λ·U^T||_F / ||A||_F.
    pub relative_error: T,
}

/// Randomized Eigenvalue Decomposition for symmetric matrices.
///
/// Uses the Halko-Martinsson-Tropp (HMT) algorithm to efficiently compute
/// a few dominant eigenpairs without performing a full decomposition.
#[derive(Debug, Clone)]
pub struct RandomizedEvd {
    config: RandomizedEvdConfig,
}

/// Simple linear congruential generator for reproducible Gaussian samples.
///
/// Uses Knuth's constants for the LCG and Box-Muller transform for Gaussian output.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a standard normal sample via Box-Muller transform.
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = (self.state >> 11) as f64 / (1u64 << 53) as f64;
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = (self.state >> 11) as f64 / (1u64 << 53) as f64;
        // Clamp u1 away from zero to avoid ln(0)
        let u1 = u1.max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

impl RandomizedEvd {
    /// Creates a new `RandomizedEvd` with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if rank is zero.
    pub fn new(config: RandomizedEvdConfig) -> Result<Self, RandomizedEvdError> {
        if config.rank == 0 {
            return Err(RandomizedEvdError::InvalidConfig(
                "rank must be at least 1".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Computes k dominant eigenpairs of a symmetric f64 matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (n×n). Upper triangle is used for the projected system.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is non-square, if rank exceeds the matrix
    /// dimension, or if an internal numerical failure occurs.
    pub fn compute(
        &self,
        a: MatRef<'_, f64>,
    ) -> Result<RandomizedEvdResult<f64>, RandomizedEvdError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(RandomizedEvdError::DimensionMismatch {
                expected: n,
                got: a.ncols(),
            });
        }
        if n == 0 {
            return Err(RandomizedEvdError::InvalidConfig(
                "Matrix is empty".to_string(),
            ));
        }

        let k = self.config.rank;
        let p = self.config.oversampling;
        let l = (k + p).min(n);

        if k > n {
            return Err(RandomizedEvdError::InvalidConfig(format!(
                "rank {} exceeds matrix dimension {}",
                k, n
            )));
        }

        match self.config.which {
            RandomizedEvdTarget::Largest => self.compute_largest_f64(a, n, k, l),
            RandomizedEvdTarget::Smallest => self.compute_smallest_f64(a, n, k, l),
        }
    }

    fn compute_largest_f64(
        &self,
        a: MatRef<'_, f64>,
        n: usize,
        k: usize,
        l: usize,
    ) -> Result<RandomizedEvdResult<f64>, RandomizedEvdError> {
        // Step 1: Random Gaussian matrix Ω (n × l)
        let omega = Self::random_gaussian_f64(n, l, self.config.seed);

        // Step 2: Power iteration Y = A^q * Ω
        let y = Self::power_iteration_f64(a, &omega, self.config.power_iters)?;

        // Step 3: Thin QR of Y → Q (n × effective_l)
        let qr = Qr::compute(y.as_ref()).map_err(|e| {
            RandomizedEvdError::NumericalFailure(format!("QR factorization failed: {:?}", e))
        })?;
        let q = qr.q();

        // Step 4: Project B = Q^T * A * Q  (effective_l × effective_l)
        let aq = mat_mul_f64(a, q.as_ref());
        let b = mat_mul_transpose_left_f64(&q, &aq);

        // Step 5: Dense EVD of small symmetric matrix B
        let (mut eigenvalues, mut eigenvectors) = Self::dense_evd_f64(b.as_ref())?;

        // Reorder descending by magnitude and truncate to k
        let actual_k = k.min(eigenvalues.len());
        sort_eigenpairs_by_magnitude_desc(&mut eigenvalues, &mut eigenvectors);

        let eigenvalues_k = eigenvalues[..actual_k].to_vec();

        // Step 6: Recover U = Q * V[:, :k]  (n × k)
        let v_k = extract_cols(&eigenvectors, actual_k);
        let u = mat_mul_f64(q.as_ref(), v_k.as_ref());

        // Estimate relative error
        let rel_err = compute_relative_error_f64(a, &u, &eigenvalues_k, n, actual_k);

        Ok(RandomizedEvdResult {
            eigenvalues: eigenvalues_k,
            eigenvectors: u,
            relative_error: rel_err,
        })
    }

    fn compute_smallest_f64(
        &self,
        a: MatRef<'_, f64>,
        n: usize,
        k: usize,
        l: usize,
    ) -> Result<RandomizedEvdResult<f64>, RandomizedEvdError> {
        // Shift-flip trick:
        //   mu ≈ spectral radius upper bound (Frobenius norm)
        //   B = mu*I - A  =>  smallest eigenvalues of A become largest of B
        let mu = frobenius_norm_f64(a);

        let mut b_shifted = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b_shifted[(i, j)] = if i == j { mu - a[(i, j)] } else { -a[(i, j)] };
            }
        }

        // Compute largest eigenpairs of shifted matrix
        let omega = Self::random_gaussian_f64(n, l, self.config.seed);
        let y = Self::power_iteration_f64(b_shifted.as_ref(), &omega, self.config.power_iters)?;

        let qr = Qr::compute(y.as_ref()).map_err(|e| {
            RandomizedEvdError::NumericalFailure(format!("QR factorization failed: {:?}", e))
        })?;
        let q = qr.q();

        let aq = mat_mul_f64(b_shifted.as_ref(), q.as_ref());
        let b = mat_mul_transpose_left_f64(&q, &aq);

        let (mut shifted_eigs, mut eigenvectors) = Self::dense_evd_f64(b.as_ref())?;
        let actual_k = k.min(shifted_eigs.len());

        // Sort largest-of-shifted first (= smallest of original)
        sort_eigenpairs_by_magnitude_desc(&mut shifted_eigs, &mut eigenvectors);

        // Recover original eigenvalues: lambda_orig = mu - lambda_shifted
        let eigenvalues_k: Vec<f64> = shifted_eigs[..actual_k].iter().map(|&s| mu - s).collect();

        let v_k = extract_cols(&eigenvectors, actual_k);
        let u = mat_mul_f64(q.as_ref(), v_k.as_ref());

        let rel_err = compute_relative_error_f64(a, &u, &eigenvalues_k, n, actual_k);

        Ok(RandomizedEvdResult {
            eigenvalues: eigenvalues_k,
            eigenvectors: u,
            relative_error: rel_err,
        })
    }

    /// Computes k dominant eigenpairs of a symmetric f32 matrix.
    ///
    /// Internally promotes to f64 for numerical stability, then demotes results.
    pub fn compute_f32(
        &self,
        a: MatRef<'_, f32>,
    ) -> Result<RandomizedEvdResult<f32>, RandomizedEvdError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(RandomizedEvdError::DimensionMismatch {
                expected: n,
                got: a.ncols(),
            });
        }
        if n == 0 {
            return Err(RandomizedEvdError::InvalidConfig(
                "Matrix is empty".to_string(),
            ));
        }

        let k = self.config.rank;
        if k > n {
            return Err(RandomizedEvdError::InvalidConfig(format!(
                "rank {} exceeds matrix dimension {}",
                k, n
            )));
        }

        // Promote to f64, compute, then demote
        let a_f64 = promote_f32_to_f64(a);
        let result_f64 = self.compute(a_f64.as_ref())?;

        let eigenvalues_f32: Vec<f32> = result_f64.eigenvalues.iter().map(|&v| v as f32).collect();
        let (nrows, ncols) = (
            result_f64.eigenvectors.nrows(),
            result_f64.eigenvectors.ncols(),
        );
        let mut eigvecs_f32 = Mat::<f32>::zeros(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                eigvecs_f32[(i, j)] = result_f64.eigenvectors[(i, j)] as f32;
            }
        }

        Ok(RandomizedEvdResult {
            eigenvalues: eigenvalues_f32,
            eigenvectors: eigvecs_f32,
            relative_error: result_f64.relative_error as f32,
        })
    }

    /// Generates a random Gaussian matrix (n × k) using the LCG PRNG.
    fn random_gaussian_f64(n: usize, k: usize, seed: u64) -> Mat<f64> {
        let mut mat = Mat::<f64>::zeros(n, k);
        let mut lcg = Lcg::new(seed);
        for i in 0..n {
            for j in 0..k {
                mat[(i, j)] = lcg.next_f64();
            }
        }
        mat
    }

    /// Power iteration: computes Y = A^q * Ω with intermediate QR orthogonalization.
    ///
    /// - q = 0: returns Ω unchanged (no multiplication)
    /// - q = 1: returns A * Ω
    /// - q = 2: returns A * QR(A * Ω).Q  (one orthogonalization between steps)
    fn power_iteration_f64(
        a: MatRef<'_, f64>,
        omega: &Mat<f64>,
        q: usize,
    ) -> Result<Mat<f64>, RandomizedEvdError> {
        if q == 0 {
            // Return a copy of Ω
            let mut y = Mat::<f64>::zeros(omega.nrows(), omega.ncols());
            for i in 0..omega.nrows() {
                for j in 0..omega.ncols() {
                    y[(i, j)] = omega[(i, j)];
                }
            }
            return Ok(y);
        }

        // Y_1 = A * Ω
        let mut y = mat_mul_f64(a, omega.as_ref());

        // Iterations 2..=q with intermediate orthogonalization for numerical stability
        for _iter in 2..=q {
            let qr = Qr::compute(y.as_ref()).map_err(|e| {
                RandomizedEvdError::NumericalFailure(format!(
                    "QR in power iteration failed: {:?}",
                    e
                ))
            })?;
            let q_mat = qr.q();
            y = mat_mul_f64(a, q_mat.as_ref());
        }

        Ok(y)
    }

    /// Computes the dense symmetric EVD of a small matrix using `SymmetricEvd`.
    fn dense_evd_f64(b: MatRef<'_, f64>) -> Result<(Vec<f64>, Mat<f64>), RandomizedEvdError> {
        let evd = SymmetricEvd::compute(b).map_err(|e| {
            RandomizedEvdError::NumericalFailure(format!("Dense symmetric EVD failed: {:?}", e))
        })?;
        let eigenvalues = evd.eigenvalues().to_vec();
        let evecs_ref = evd.eigenvectors();
        let mut eigenvectors = Mat::<f64>::zeros(evecs_ref.nrows(), evecs_ref.ncols());
        for i in 0..evecs_ref.nrows() {
            for j in 0..evecs_ref.ncols() {
                eigenvectors[(i, j)] = evecs_ref[(i, j)];
            }
        }
        Ok((eigenvalues, eigenvectors))
    }
}

// ============================================================================
// Private helper functions
// ============================================================================

/// Matrix multiplication C = A * B for f64 matrices.
fn mat_mul_f64(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let m = a.nrows();
    let kk = a.ncols();
    let n = b.ncols();
    let mut c = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..kk {
                sum += a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}

/// Matrix multiplication C = A^T * B for f64 matrices.
fn mat_mul_transpose_left_f64(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.ncols(); // A^T rows = A cols
    let kk = a.nrows(); // A^T cols = A rows
    let n = b.ncols();
    let mut c = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..kk {
                sum += a[(l, i)] * b[(l, j)]; // A^T[i,l] = A[l,i]
            }
            c[(i, j)] = sum;
        }
    }
    c
}

/// Sorts eigenpairs in-place in descending order by eigenvalue magnitude.
fn sort_eigenpairs_by_magnitude_desc(eigenvalues: &mut Vec<f64>, eigenvectors: &mut Mat<f64>) {
    let n = eigenvalues.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .abs()
            .partial_cmp(&eigenvalues[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let old_eigs = eigenvalues.clone();
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        eigenvalues[new_idx] = old_eigs[old_idx];
    }

    let old_vecs = eigenvectors.clone();
    let nrows = eigenvectors.nrows();
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..nrows {
            eigenvectors[(row, new_col)] = old_vecs[(row, old_col)];
        }
    }
}

/// Extracts the first k columns of a matrix into a new matrix.
fn extract_cols(mat: &Mat<f64>, k: usize) -> Mat<f64> {
    let nrows = mat.nrows();
    let actual_k = k.min(mat.ncols());
    let mut result = Mat::<f64>::zeros(nrows, actual_k);
    for i in 0..nrows {
        for j in 0..actual_k {
            result[(i, j)] = mat[(i, j)];
        }
    }
    result
}

/// Computes the Frobenius norm of a matrix.
fn frobenius_norm_f64(a: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += a[(i, j)] * a[(i, j)];
        }
    }
    sum.sqrt()
}

/// Computes the relative Frobenius error ||A - U·diag(λ)·U^T||_F / ||A||_F.
fn compute_relative_error_f64(
    a: MatRef<'_, f64>,
    u: &Mat<f64>,
    eigenvalues: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    // Reconstruct low-rank approximation: Â = Σ_i λ_i * u_i * u_i^T
    let mut approx = Mat::<f64>::zeros(n, n);
    for idx in 0..k {
        let lambda = eigenvalues[idx];
        for i in 0..n {
            for j in 0..n {
                approx[(i, j)] += lambda * u[(i, idx)] * u[(j, idx)];
            }
        }
    }

    let mut error_sq = 0.0f64;
    let mut a_norm_sq = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let diff = a[(i, j)] - approx[(i, j)];
            error_sq += diff * diff;
            a_norm_sq += a[(i, j)] * a[(i, j)];
        }
    }

    let a_norm = a_norm_sq.sqrt();
    if a_norm > 0.0 {
        error_sq.sqrt() / a_norm
    } else {
        0.0
    }
}

/// Promotes an f32 matrix to f64.
fn promote_f32_to_f64(a: MatRef<'_, f32>) -> Mat<f64> {
    let mut result = Mat::<f64>::zeros(a.nrows(), a.ncols());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            result[(i, j)] = a[(i, j)] as f64;
        }
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_randomized_evd_config_default() {
        let config = RandomizedEvdConfig::default();
        assert_eq!(config.rank, 6);
        assert_eq!(config.oversampling, 10);
        assert_eq!(config.power_iters, 2);
        assert_eq!(config.which, RandomizedEvdTarget::Largest);
    }

    #[test]
    fn test_randomized_evd_2x2_known() {
        // A = [[3, 1], [1, 3]], eigenvalues are 4 and 2
        let a = Mat::from_rows(&[&[3.0f64, 1.0], &[1.0, 3.0]]);
        let config = RandomizedEvdConfig::new(1).with_seed(42);
        let revd = RandomizedEvd::new(config).unwrap();
        let result = revd.compute(a.as_ref()).unwrap();

        assert_eq!(result.eigenvalues.len(), 1);
        assert_eq!(result.eigenvectors.ncols(), 1);
        // Largest eigenvalue should be close to 4
        assert!(
            approx_eq(result.eigenvalues[0], 4.0, 0.5),
            "Expected eigenvalue ≈ 4, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_randomized_evd_diagonal() {
        // 10×10 diagonal matrix with eigenvalues 10, 9, ..., 1
        let n = 10usize;
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (n - i) as f64;
        }

        let config = RandomizedEvdConfig::new(3)
            .with_oversampling(5)
            .with_seed(123);
        let revd = RandomizedEvd::new(config).unwrap();
        let result = revd.compute(a.as_ref()).unwrap();

        assert_eq!(result.eigenvalues.len(), 3);
        // Largest 3 eigenvalues should be close to 10, 9, 8
        assert!(
            approx_eq(result.eigenvalues[0].abs(), 10.0, 1.0),
            "Expected |λ₀| ≈ 10, got {}",
            result.eigenvalues[0]
        );
        assert!(
            approx_eq(result.eigenvalues[1].abs(), 9.0, 1.0),
            "Expected |λ₁| ≈ 9, got {}",
            result.eigenvalues[1]
        );
        assert!(
            approx_eq(result.eigenvalues[2].abs(), 8.0, 1.0),
            "Expected |λ₂| ≈ 8, got {}",
            result.eigenvalues[2]
        );
    }

    #[test]
    fn test_randomized_evd_accuracy() {
        // Build a 20×20 SPD matrix with strongly decaying spectrum so that
        // k=5 dominant eigenpairs capture nearly all the spectral energy.
        //
        // Construction: A = Q * D * Q^T where D = diag(1000, 100, 10, 5, 2, 0.1, ..., 0.01)
        // Q is the QR factor of a simple Vandermonde-like matrix.
        let n = 20usize;
        let k = 5usize;

        // Build a simple orthogonal Q via QR of a full-rank matrix
        // Use the identity plus a small perturbation for Q
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = 1.0;
            // Small off-diagonal: v[i, (i+1)%n] = 0.1
            v[(i, (i + 1) % n)] += 0.1;
        }
        // QR of v gives an orthogonal Q
        let qr = crate::qr::Qr::compute(v.as_ref()).unwrap();
        let q = qr.q();

        // Strongly decaying eigenvalues: first 5 are large, rest tiny
        let mut eigenvalues_true = vec![0.0f64; n];
        eigenvalues_true[0] = 1000.0;
        eigenvalues_true[1] = 200.0;
        eigenvalues_true[2] = 50.0;
        eigenvalues_true[3] = 10.0;
        eigenvalues_true[4] = 2.0;
        for i in 5..n {
            eigenvalues_true[i] = 0.001 * (i as f64);
        }

        // Build A = Q * D * Q^T
        let mut a = Mat::<f64>::zeros(n, n);
        for idx in 0..n {
            let lam = eigenvalues_true[idx];
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] += lam * q[(i, idx)] * q[(j, idx)];
                }
            }
        }
        // Enforce exact symmetry numerically
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (a[(i, j)] + a[(j, i)]) * 0.5;
                a[(i, j)] = avg;
                a[(j, i)] = avg;
            }
        }

        // Reference: full symmetric EVD
        let full_evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let full_eigs = full_evd.eigenvalues();

        // Randomized EVD - k=5 should capture almost all spectral energy
        let config = RandomizedEvdConfig::new(k)
            .with_oversampling(10)
            .with_power_iters(3)
            .with_seed(999);
        let revd = RandomizedEvd::new(config).unwrap();
        let result = revd.compute(a.as_ref()).unwrap();

        assert_eq!(result.eigenvalues.len(), k);

        // With strongly decaying spectrum, relative error should be very small
        assert!(
            result.relative_error < 1e-4,
            "Relative error {} should be < 1e-4",
            result.relative_error
        );

        // Top k eigenvalues from full EVD (ascending, take last k)
        let len = full_eigs.len();
        let mut full_top_k: Vec<f64> = full_eigs[(len - k)..].to_vec();
        full_top_k.sort_by(|a, b| {
            b.abs()
                .partial_cmp(&a.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..k {
            assert!(
                approx_eq(result.eigenvalues[i].abs(), full_top_k[i].abs(), 1.0),
                "Eigenvalue {} mismatch: randomized={}, full={}",
                i,
                result.eigenvalues[i],
                full_top_k[i]
            );
        }
    }

    #[test]
    fn test_randomized_evd_f32() {
        // 10×10 diagonal f32 matrix with eigenvalues 1..=10
        let n = 10usize;
        let mut a = Mat::<f32>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (i + 1) as f32;
        }

        let config = RandomizedEvdConfig::new(3).with_seed(77);
        let revd = RandomizedEvd::new(config).unwrap();
        let result = revd.compute_f32(a.as_ref()).unwrap();

        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.eigenvectors.ncols(), 3);
        // Largest eigenvalue should be close to 10
        assert!(
            (result.eigenvalues[0].abs() - 10.0f32).abs() < 2.0,
            "Expected |λ₀| ≈ 10, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_randomized_evd_smallest() {
        // 10×10 diagonal matrix with eigenvalues 1..=10
        // k=2 smallest should yield eigenvalues ≈ 1, 2
        let n = 10usize;
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (i + 1) as f64;
        }

        let config = RandomizedEvdConfig::new(2)
            .with_oversampling(5)
            .with_power_iters(2)
            .with_target(RandomizedEvdTarget::Smallest)
            .with_seed(314);
        let revd = RandomizedEvd::new(config).unwrap();
        let result = revd.compute(a.as_ref()).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);

        // Sort by magnitude ascending to compare
        let mut eigs_sorted = result.eigenvalues.clone();
        eigs_sorted.sort_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert!(
            approx_eq(eigs_sorted[0], 1.0, 1.5),
            "Expected smallest eigenvalue ≈ 1, got {}",
            eigs_sorted[0]
        );
        assert!(
            approx_eq(eigs_sorted[1], 2.0, 1.5),
            "Expected second smallest eigenvalue ≈ 2, got {}",
            eigs_sorted[1]
        );
    }

    #[test]
    fn test_randomized_evd_error_cases() {
        // Zero rank must fail
        let result = RandomizedEvd::new(RandomizedEvdConfig {
            rank: 0,
            ..Default::default()
        });
        assert!(result.is_err(), "rank=0 should return an error");

        // Rank larger than matrix dimension must fail
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
        let config = RandomizedEvdConfig::new(5);
        let revd = RandomizedEvd::new(config).unwrap();
        let result = revd.compute(a.as_ref());
        assert!(result.is_err(), "rank > n should return an error");
    }
}
