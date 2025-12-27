//! Randomized Singular Value Decomposition (rSVD).
//!
//! Provides fast approximate SVD using random projections, particularly
//! useful for computing truncated SVD of large matrices.
//!
//! # Algorithm
//!
//! The basic randomized SVD algorithm:
//! 1. Generate random test matrix Ω (n × k) where k = rank + oversampling
//! 2. Compute Y = A × Ω to sample the column space
//! 3. Compute QR: Y = Q × R
//! 4. Project: B = Q^T × A
//! 5. Compute full SVD of B: B = Ũ × Σ × V^T
//! 6. Recover: U = Q × Ũ
//!
//! # Power Iteration Variant
//!
//! For matrices with slowly decaying singular values, use power iteration:
//! Y = (A × A^T)^q × A × Ω
//!
//! This emphasizes the dominant singular values for better approximation.
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
//! use oxiblas_lapack::svd::RandomizedSvd;
//! use oxiblas_matrix::Mat;
//!
//! // Create a low-rank matrix
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 3.0, 4.0],
//!     &[5.0, 6.0, 7.0, 8.0],
//!     &[9.0, 10.0, 11.0, 12.0],
//! ]);
//!
//! // Compute rank-2 approximation
//! let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();
//!
//! let sigma = rsvd.singular_values();
//! assert_eq!(sigma.len(), 2);
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::qr::Qr;
use crate::svd::{Svd, SvdDc};

/// Error type for randomized SVD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomizedSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Requested rank exceeds matrix dimensions.
    RankTooLarge {
        /// Requested rank.
        requested: usize,
        /// Maximum possible rank.
        max_rank: usize,
    },
    /// Invalid parameter.
    InvalidParameter,
    /// Internal SVD computation failed.
    SvdFailed,
    /// QR factorization failed.
    QrFailed,
}

impl core::fmt::Display for RandomizedSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::RankTooLarge {
                requested,
                max_rank,
            } => {
                write!(
                    f,
                    "Requested rank {} exceeds maximum {}",
                    requested, max_rank
                )
            }
            Self::InvalidParameter => write!(f, "Invalid parameter"),
            Self::SvdFailed => write!(f, "Internal SVD computation failed"),
            Self::QrFailed => write!(f, "QR factorization failed"),
        }
    }
}

impl std::error::Error for RandomizedSvdError {}

/// Configuration for randomized SVD.
#[derive(Debug, Clone)]
pub struct RandomizedSvdConfig {
    /// Target rank for the approximation.
    pub target_rank: usize,
    /// Oversampling parameter (typically 5-10 for good accuracy).
    /// The algorithm uses rank + oversampling random vectors.
    pub oversampling: usize,
    /// Number of power iterations (0 for basic algorithm).
    /// Higher values improve accuracy for slowly decaying spectra.
    pub power_iterations: usize,
    /// Random seed for reproducibility (None for random).
    pub seed: Option<u64>,
    /// Use divide-and-conquer SVD for the reduced problem.
    pub use_divide_conquer: bool,
}

impl Default for RandomizedSvdConfig {
    fn default() -> Self {
        Self {
            target_rank: 10,
            oversampling: 5,
            power_iterations: 0,
            seed: None,
            use_divide_conquer: true,
        }
    }
}

impl RandomizedSvdConfig {
    /// Creates a new configuration with the specified target rank.
    pub fn new(target_rank: usize) -> Self {
        Self {
            target_rank,
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
    pub fn with_power_iterations(mut self, iterations: usize) -> Self {
        self.power_iterations = iterations;
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Uses Jacobi SVD instead of divide-and-conquer for the reduced problem.
    #[must_use]
    pub fn use_jacobi(mut self) -> Self {
        self.use_divide_conquer = false;
        self
    }
}

/// Result of randomized SVD computation.
///
/// Stores the truncated SVD: A ≈ U × Σ × V^T
#[derive(Debug, Clone)]
pub struct RandomizedSvd<T: Scalar> {
    /// Left singular vectors (m × k).
    u: Mat<T>,
    /// Singular values (k elements, descending order).
    sigma: Vec<T>,
    /// Right singular vectors (n × k, stored as V not V^T).
    v: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
    /// Target rank used.
    rank: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> RandomizedSvd<T> {
    /// Computes a rank-k approximation of the SVD using randomized methods.
    ///
    /// Uses default parameters (oversampling=5, no power iteration).
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m × n)
    /// * `rank` - Target rank for the approximation
    ///
    /// # Returns
    ///
    /// Truncated SVD with k singular values and vectors.
    pub fn compute(a: MatRef<'_, T>, rank: usize) -> Result<Self, RandomizedSvdError> {
        let config = RandomizedSvdConfig::new(rank);
        Self::compute_with_config(a, config)
    }

    /// Computes a rank-k approximation with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m × n)
    /// * `config` - Configuration parameters
    ///
    /// # Returns
    ///
    /// Truncated SVD with k singular values and vectors.
    pub fn compute_with_config(
        a: MatRef<'_, T>,
        config: RandomizedSvdConfig,
    ) -> Result<Self, RandomizedSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(RandomizedSvdError::EmptyMatrix);
        }

        let max_rank = m.min(n);
        if config.target_rank > max_rank {
            return Err(RandomizedSvdError::RankTooLarge {
                requested: config.target_rank,
                max_rank,
            });
        }

        let target_rank = config.target_rank;

        // Effective dimension for random projection
        let k = (target_rank + config.oversampling).min(max_rank);

        // Step 1: Generate random test matrix Ω (n × k)
        let omega = generate_random_matrix(n, k, config.seed);

        // Step 2: Compute Y = A × Ω (with optional power iteration)
        let y = if config.power_iterations > 0 {
            compute_with_power_iteration(&a, &omega, config.power_iterations)
        } else {
            mat_mul(&a, omega.as_ref())
        };

        // Step 3: QR factorization of Y
        let qr = Qr::compute(y.as_ref()).map_err(|_| RandomizedSvdError::QrFailed)?;
        let q = qr.q();

        // Step 4: Form B = Q^T × A
        let b = mat_mul_transpose_left(&q, &a);

        // Step 5: Compute SVD of B
        let (u_b, sigma_full, vt_full) = if config.use_divide_conquer {
            let svd_b = SvdDc::compute(b.as_ref()).map_err(|_| RandomizedSvdError::SvdFailed)?;
            let u_b = mat_from_ref(svd_b.u());
            let vt = mat_from_ref(svd_b.vt());
            (u_b, svd_b.singular_values().to_vec(), vt)
        } else {
            let svd_b = Svd::compute(b.as_ref()).map_err(|_| RandomizedSvdError::SvdFailed)?;
            let u_b = mat_from_ref(svd_b.u());
            let vt = mat_from_ref(svd_b.vt());
            (u_b, svd_b.singular_values().to_vec(), vt)
        };

        // Transpose V^T to get V (n × k matrix where each column is a right singular vector)
        let v_full = transpose(&vt_full);

        // Step 6: Recover U = Q × Ũ
        let u_full = mat_mul(&q.as_ref(), u_b.as_ref());

        // Truncate to target rank
        let (u, sigma, v) = truncate_svd(&u_full, &sigma_full, &v_full, target_rank);

        Ok(Self {
            u,
            sigma,
            v,
            m,
            n,
            rank: target_rank,
        })
    }

    /// Returns the left singular vectors U (m × k matrix).
    pub fn u_matrix(&self) -> Mat<T> {
        self.u.clone()
    }

    /// Returns a reference to the left singular vectors.
    pub fn u(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }

    /// Returns the singular values in descending order.
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the right singular vectors V (n × k matrix).
    pub fn v_matrix(&self) -> Mat<T> {
        self.v.clone()
    }

    /// Returns a reference to the right singular vectors.
    pub fn v(&self) -> MatRef<'_, T> {
        self.v.as_ref()
    }

    /// Returns the target rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the original matrix dimensions (m, n).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Reconstructs the low-rank approximation A ≈ U × Σ × V^T.
    pub fn reconstruct(&self) -> Mat<T> {
        let mut result = Mat::zeros(self.m, self.n);

        // A = U × diag(σ) × V^T
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k in 0..self.rank {
                    sum = sum + self.u[(i, k)] * self.sigma[k] * self.v[(j, k)];
                }
                result[(i, j)] = sum;
            }
        }

        result
    }

    /// Computes the Frobenius norm of the approximation error.
    ///
    /// This requires recomputing A × V to avoid storing the original matrix.
    pub fn reconstruction_error(&self, a: MatRef<'_, T>) -> T {
        let approx = self.reconstruct();
        let mut error_sq = T::zero();

        for i in 0..self.m {
            for j in 0..self.n {
                let diff = a[(i, j)] - approx[(i, j)];
                error_sq = error_sq + diff * diff;
            }
        }

        Real::sqrt(error_sq)
    }

    /// Computes the relative Frobenius error ||A - Â||_F / ||A||_F.
    pub fn relative_error(&self, a: MatRef<'_, T>) -> T {
        let error = self.reconstruction_error(a);
        let mut a_norm_sq = T::zero();

        for i in 0..self.m {
            for j in 0..self.n {
                a_norm_sq = a_norm_sq + a[(i, j)] * a[(i, j)];
            }
        }

        let a_norm = Real::sqrt(a_norm_sq);
        if a_norm > T::zero() {
            error / a_norm
        } else {
            T::zero()
        }
    }

    /// Returns the sum of singular values (nuclear norm of approximation).
    pub fn nuclear_norm(&self) -> T {
        self.sigma.iter().copied().fold(T::zero(), |acc, s| acc + s)
    }

    /// Returns the condition number of the approximation (σ_1 / σ_k).
    pub fn condition_number(&self) -> T {
        if self.sigma.is_empty() {
            return T::one();
        }

        let sigma_max = self.sigma[0];
        let sigma_min = self.sigma[self.rank - 1];

        if sigma_min > T::zero() {
            sigma_max / sigma_min
        } else {
            T::infinity()
        }
    }
}

/// Generates a random Gaussian matrix using a simple PRNG.
fn generate_random_matrix<T: Field + Real + bytemuck::Zeroable>(
    rows: usize,
    cols: usize,
    seed: Option<u64>,
) -> Mat<T> {
    let mut mat = Mat::zeros(rows, cols);

    // Simple xorshift64 PRNG
    let mut state = seed.unwrap_or(0x1234_5678_9ABC_DEF0);

    for i in 0..rows {
        for j in 0..cols {
            // Generate uniform random in [0, 1)
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u1 = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D)) as f64) / (u64::MAX as f64);

            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u2 = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D)) as f64) / (u64::MAX as f64);

            // Box-Muller transform for Gaussian
            let r = (-2.0 * u1.max(1e-300).ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let gaussian = r * theta.cos();

            mat[(i, j)] = T::from_f64(gaussian).unwrap_or(T::zero());
        }
    }

    mat
}

/// Computes Y = (A × A^T)^q × A × Ω using power iteration.
fn compute_with_power_iteration<T: Field + Real + bytemuck::Zeroable>(
    a: &MatRef<'_, T>,
    omega: &Mat<T>,
    power_iterations: usize,
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let k = omega.ncols();

    // Y = A × Ω
    let mut y = mat_mul(a, omega.as_ref());

    // Power iteration: Y = (A × A^T)^q × Y
    // Computed as alternating A^T and A multiplications with orthogonalization
    for _iter in 0..power_iterations {
        // QR for numerical stability
        let qr = match Qr::compute(y.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return y,
        };
        let q = qr.q();

        // Y = A^T × Q (n × k)
        let mut y_temp = Mat::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                let mut sum = T::zero();
                for l in 0..m {
                    sum = sum + a[(l, i)] * q[(l, j)];
                }
                y_temp[(i, j)] = sum;
            }
        }

        // QR for numerical stability
        let qr2 = match Qr::compute(y_temp.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return mat_mul(a, omega.as_ref()),
        };
        let q2 = qr2.q();

        // Y = A × Q2 (m × k)
        y = Mat::zeros(m, k);
        for i in 0..m {
            for j in 0..k {
                let mut sum = T::zero();
                for l in 0..n {
                    sum = sum + a[(i, l)] * q2[(l, j)];
                }
                y[(i, j)] = sum;
            }
        }
    }

    y
}

/// Matrix multiplication C = A × B.
fn mat_mul<T: Field + Real + bytemuck::Zeroable>(a: &MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    debug_assert_eq!(k, b.nrows());

    let mut c = Mat::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] = sum;
        }
    }

    c
}

/// Matrix multiplication C = A^T × B.
fn mat_mul_transpose_left<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    b: &MatRef<'_, T>,
) -> Mat<T> {
    let m = a.ncols(); // A^T has ncols(A) rows
    let k = a.nrows(); // A^T has nrows(A) columns
    let n = b.ncols();

    debug_assert_eq!(k, b.nrows());

    let mut c = Mat::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + a[(l, i)] * b[(l, j)]; // A^T[i,l] = A[l,i]
            }
            c[(i, j)] = sum;
        }
    }

    c
}

/// Converts a MatRef to an owned Mat.
fn mat_from_ref<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Mat<T> {
    let mut mat = Mat::zeros(a.nrows(), a.ncols());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            mat[(i, j)] = a[(i, j)];
        }
    }
    mat
}

/// Transpose a matrix.
fn transpose<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>) -> Mat<T> {
    let mut mat = Mat::zeros(a.ncols(), a.nrows());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            mat[(j, i)] = a[(i, j)];
        }
    }
    mat
}

/// Truncates SVD to the specified rank.
fn truncate_svd<T: Field + Real + bytemuck::Zeroable>(
    u: &Mat<T>,
    sigma: &[T],
    v: &Mat<T>,
    rank: usize,
) -> (Mat<T>, Vec<T>, Mat<T>) {
    let m = u.nrows();
    let n = v.nrows();
    let actual_rank = rank.min(sigma.len());

    // Truncate U to m × rank
    let mut u_trunc = Mat::zeros(m, actual_rank);
    for i in 0..m {
        for j in 0..actual_rank {
            u_trunc[(i, j)] = u[(i, j)];
        }
    }

    // Truncate sigma
    let sigma_trunc = sigma[..actual_rank].to_vec();

    // Truncate V to n × rank
    let mut v_trunc = Mat::zeros(n, actual_rank);
    for i in 0..n {
        for j in 0..actual_rank {
            v_trunc[(i, j)] = v[(i, j)];
        }
    }

    (u_trunc, sigma_trunc, v_trunc)
}

/// Convenience function to compute randomized SVD.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `rank` - Target rank
///
/// # Returns
///
/// Tuple of (U, singular_values, V) for the truncated SVD.
pub fn rsvd<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    rank: usize,
) -> Result<(Mat<T>, Vec<T>, Mat<T>), RandomizedSvdError> {
    let result = RandomizedSvd::compute(a, rank)?;
    Ok((
        result.u_matrix(),
        result.singular_values().to_vec(),
        result.v_matrix(),
    ))
}

/// Computes randomized SVD with power iteration for improved accuracy.
///
/// Use this for matrices with slowly decaying singular values.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `rank` - Target rank
/// * `power_iterations` - Number of power iterations (typically 1-2)
///
/// # Returns
///
/// Tuple of (U, singular_values, V) for the truncated SVD.
pub fn rsvd_power<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    rank: usize,
    power_iterations: usize,
) -> Result<(Mat<T>, Vec<T>, Mat<T>), RandomizedSvdError> {
    let config = RandomizedSvdConfig::new(rank).with_power_iterations(power_iterations);
    let result = RandomizedSvd::compute_with_config(a, config)?;
    Ok((
        result.u_matrix(),
        result.singular_values().to_vec(),
        result.v_matrix(),
    ))
}

/// Randomized low-rank matrix approximation.
///
/// Computes the best rank-k approximation in Frobenius norm.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `rank` - Target rank
///
/// # Returns
///
/// Low-rank approximation matrix.
pub fn low_rank_approximation<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    rank: usize,
) -> Result<Mat<T>, RandomizedSvdError> {
    let rsvd = RandomizedSvd::compute(a, rank)?;
    Ok(rsvd.reconstruct())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_random_matrix_generation() {
        let mat: Mat<f64> = generate_random_matrix(10, 5, Some(42));
        assert_eq!(mat.nrows(), 10);
        assert_eq!(mat.ncols(), 5);

        // Check that values are roughly Gaussian (should have some non-zero values)
        let mut sum = 0.0;
        for i in 0..10 {
            for j in 0..5 {
                sum += mat[(i, j)].abs();
            }
        }
        assert!(sum > 0.0);
    }

    #[test]
    fn test_rsvd_rank1_matrix() {
        // Create a rank-1 matrix: A = u * v^T
        let mut a = Mat::zeros(5, 4);
        for i in 0..5 {
            for j in 0..4 {
                a[(i, j)] = (i + 1) as f64 * (j + 1) as f64;
            }
        }

        let rsvd = RandomizedSvd::compute(a.as_ref(), 1).unwrap();
        assert_eq!(rsvd.rank(), 1);
        assert_eq!(rsvd.singular_values().len(), 1);

        // Reconstruction should be close to original
        let _approx = rsvd.reconstruct();
        let error = rsvd.relative_error(a.as_ref());
        assert!(
            error < 0.01,
            "Relative error {} should be small for rank-1 matrix",
            error
        );
    }

    #[test]
    fn test_rsvd_low_rank_matrix() {
        // Create a rank-2 matrix
        let mut a = Mat::zeros(10, 8);
        for i in 0..10 {
            for j in 0..8 {
                // Two components
                a[(i, j)] =
                    (i + 1) as f64 * (j + 1) as f64 + 0.5 * ((i + 2) as f64) * ((8 - j) as f64);
            }
        }

        let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();
        assert_eq!(rsvd.rank(), 2);

        let error = rsvd.relative_error(a.as_ref());
        assert!(
            error < 0.1,
            "Relative error {} should be small for low-rank matrix",
            error
        );
    }

    #[test]
    fn test_rsvd_orthogonality() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);

        let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();
        let u = rsvd.u_matrix();
        let v = rsvd.v_matrix();

        // Check U^T * U ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += u[(k, i)] * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 0.1),
                    "U^T*U[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Check V^T * V ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 0.1),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_rsvd_with_power_iteration() {
        // Matrix with slowly decaying singular values
        let mut a = Mat::zeros(20, 15);
        for i in 0..20 {
            for j in 0..15 {
                a[(i, j)] = 1.0 / (1.0 + (i as f64 - j as f64).abs());
            }
        }

        // Without power iteration
        let config0 = RandomizedSvdConfig::new(5)
            .with_seed(123)
            .with_power_iterations(0);
        let rsvd0 = RandomizedSvd::compute_with_config(a.as_ref(), config0).unwrap();
        let error0 = rsvd0.relative_error(a.as_ref());

        // With power iteration
        let config2 = RandomizedSvdConfig::new(5)
            .with_seed(123)
            .with_power_iterations(2);
        let rsvd2 = RandomizedSvd::compute_with_config(a.as_ref(), config2).unwrap();
        let error2 = rsvd2.relative_error(a.as_ref());

        // Power iteration should generally improve accuracy
        // (Though this isn't guaranteed for all matrices)
        println!("Error without power iter: {}", error0);
        println!("Error with 2 power iters: {}", error2);
    }

    #[test]
    fn test_rsvd_singular_values_descending() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0, 5.0],
            &[6.0, 7.0, 8.0, 9.0, 10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0],
            &[16.0, 17.0, 18.0, 19.0, 20.0],
        ]);

        let rsvd = RandomizedSvd::compute(a.as_ref(), 3).unwrap();
        let sigma = rsvd.singular_values();

        for i in 0..sigma.len() - 1 {
            assert!(
                sigma[i] >= sigma[i + 1],
                "Singular values should be descending: σ[{}]={} < σ[{}]={}",
                i,
                sigma[i],
                i + 1,
                sigma[i + 1]
            );
        }
    }

    #[test]
    fn test_rsvd_config_builder() {
        let config = RandomizedSvdConfig::new(10)
            .with_oversampling(8)
            .with_power_iterations(3)
            .with_seed(42)
            .use_jacobi();

        assert_eq!(config.target_rank, 10);
        assert_eq!(config.oversampling, 8);
        assert_eq!(config.power_iterations, 3);
        assert_eq!(config.seed, Some(42));
        assert!(!config.use_divide_conquer);
    }

    #[test]
    fn test_rsvd_dimensions() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0],
        ]);

        let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();

        assert_eq!(rsvd.dimensions(), (5, 3));
        assert_eq!(rsvd.u().nrows(), 5);
        assert_eq!(rsvd.u().ncols(), 2);
        assert_eq!(rsvd.v().nrows(), 3);
        assert_eq!(rsvd.v().ncols(), 2);
        assert_eq!(rsvd.singular_values().len(), 2);
    }

    #[test]
    fn test_rsvd_error_empty_matrix() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let result = RandomizedSvd::compute(a.as_ref(), 1);
        assert!(matches!(result, Err(RandomizedSvdError::EmptyMatrix)));
    }

    #[test]
    fn test_rsvd_error_rank_too_large() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let result = RandomizedSvd::compute(a.as_ref(), 5);
        assert!(matches!(
            result,
            Err(RandomizedSvdError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn test_rsvd_convenience_functions() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let (u, sigma, v) = rsvd(a.as_ref(), 2).unwrap();
        assert_eq!(u.nrows(), 3);
        assert_eq!(u.ncols(), 2);
        assert_eq!(sigma.len(), 2);
        assert_eq!(v.nrows(), 3);
        assert_eq!(v.ncols(), 2);
    }

    #[test]
    fn test_rsvd_power_convenience() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);

        let (u, sigma, v) = rsvd_power(a.as_ref(), 2, 1).unwrap();
        assert_eq!(u.nrows(), 3);
        assert_eq!(sigma.len(), 2);
        assert_eq!(v.nrows(), 4);
    }

    #[test]
    fn test_low_rank_approximation() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let approx = low_rank_approximation(a.as_ref(), 2).unwrap();
        assert_eq!(approx.nrows(), 3);
        assert_eq!(approx.ncols(), 3);
    }

    #[test]
    fn test_rsvd_nuclear_norm() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);

        let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();
        let nuclear = rsvd.nuclear_norm();
        // Nuclear norm should be sum of singular values ≈ 1 + 2 = 3
        assert!(
            approx_eq(nuclear, 3.0, 0.5),
            "Nuclear norm {} should be close to 3",
            nuclear
        );
    }

    #[test]
    fn test_rsvd_condition_number() {
        // Diagonal matrix with condition number 10
        let a = Mat::from_rows(&[&[10.0f64, 0.0], &[0.0, 1.0]]);

        let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();
        let cond = rsvd.condition_number();
        assert!(
            approx_eq(cond, 10.0, 1.0),
            "Condition number {} should be close to 10",
            cond
        );
    }

    #[test]
    fn test_rsvd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let rsvd = RandomizedSvd::compute(a.as_ref(), 2).unwrap();
        assert_eq!(rsvd.singular_values().len(), 2);
    }

    #[test]
    fn test_rsvd_deterministic_with_seed() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);

        let config1 = RandomizedSvdConfig::new(2).with_seed(12345);
        let rsvd1 = RandomizedSvd::compute_with_config(a.as_ref(), config1).unwrap();

        let config2 = RandomizedSvdConfig::new(2).with_seed(12345);
        let rsvd2 = RandomizedSvd::compute_with_config(a.as_ref(), config2).unwrap();

        // Same seed should give same singular values
        for (s1, s2) in rsvd1
            .singular_values()
            .iter()
            .zip(rsvd2.singular_values().iter())
        {
            assert!(approx_eq(*s1, *s2, 1e-10));
        }
    }

    #[test]
    fn test_rsvd_comparison_with_full_svd() {
        // For small matrices, compare with full SVD
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        // Full SVD
        let full_svd = Svd::compute(a.as_ref()).unwrap();
        let full_sigma = full_svd.singular_values();

        // Randomized SVD with high oversampling for accuracy
        let config = RandomizedSvdConfig::new(2)
            .with_oversampling(10)
            .with_power_iterations(2)
            .with_seed(42);
        let rsvd = RandomizedSvd::compute_with_config(a.as_ref(), config).unwrap();

        // Top 2 singular values should be close
        for i in 0..2 {
            assert!(
                approx_eq(rsvd.singular_values()[i], full_sigma[i], 0.5),
                "σ[{}]: rsvd={}, full={}",
                i,
                rsvd.singular_values()[i],
                full_sigma[i]
            );
        }
    }
}
