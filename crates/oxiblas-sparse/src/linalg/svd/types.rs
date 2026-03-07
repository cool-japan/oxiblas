//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use crate::csr::CsrMatrix;
use crate::linalg::eigenvalue::*;
use crate::ops::*;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Configuration for randomized sparse SVD computation.
#[derive(Debug, Clone)]
pub struct RandomizedSparseSvdConfig {
    /// Number of singular values to compute (k).
    pub num_singular_values: usize,
    /// Oversampling parameter (p). The algorithm uses k+p random vectors.
    /// Higher values improve accuracy at the cost of more computation.
    /// Default: 10.
    pub oversampling: usize,
    /// Number of power iterations (q). Improves accuracy for matrices
    /// with slowly decaying singular values. Default: 2.
    pub power_iterations: usize,
    /// Random seed for reproducibility. If None, uses system randomness.
    pub seed: Option<u64>,
    /// Whether to compute singular vectors (U and V).
    pub compute_vectors: bool,
}
/// Configuration for truncated SVD computation.
#[derive(Debug, Clone)]
pub struct TruncatedSVDConfig<T> {
    /// Number of singular values to compute (k)
    pub num_singular_values: usize,
    /// Maximum number of Lanczos iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: T,
    /// Whether to compute singular vectors (U and V)
    pub compute_vectors: bool,
    /// Size of Krylov subspace (should be > num_singular_values)
    pub krylov_dimension: usize,
    /// Use full reorthogonalization (more accurate but slower)
    pub full_reorthogonalization: bool,
}
/// Errors that can occur during SVD computation.
#[derive(Debug, Clone, PartialEq)]
pub enum SVDError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Computation failed
    ComputationError(String),
    /// Failed to converge
    ConvergenceError(String),
}
/// Configuration for incremental SVD.
#[derive(Debug, Clone)]
pub struct IncrementalSVDConfig<T> {
    /// Maximum rank to maintain
    pub max_rank: usize,
    /// Tolerance for rank reduction
    pub tolerance: T,
    /// Whether to maintain orthogonality via reorthogonalization
    pub reorthogonalize: bool,
}
/// Result of truncated SVD computation.
#[derive(Debug, Clone)]
pub struct TruncatedSVDResult<T> {
    /// Singular values (in descending order)
    pub singular_values: Vec<T>,
    /// Left singular vectors (U), columns are vectors (m × k)
    pub u: Option<Vec<Vec<T>>>,
    /// Right singular vectors (V), columns are vectors (n × k)
    pub v: Option<Vec<Vec<T>>>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}
/// Randomized Sparse SVD solver.
///
/// Uses random projections to efficiently compute truncated SVD of large sparse matrices.
/// This method is particularly efficient when:
/// - The matrix is very large (millions of rows/columns)
/// - Only a small number of singular values are needed
/// - Approximate results are acceptable
///
/// # Algorithm
///
/// 1. Generate random test matrix Ω (n × (k+p))
/// 2. Form Y = A * Ω
/// 3. Orthonormalize Y via QR to get Q
/// 4. Optionally apply power iteration: Y = (A * A^T)^q * Y
/// 5. Form B = Q^T * A (small k × n matrix)
/// 6. Compute dense SVD of B: B = U_B * Σ * V^T
/// 7. U = Q * U_B
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::svd::{RandomizedSparseSvd, RandomizedSparseSvdConfig};
///
/// let config = RandomizedSparseSvdConfig {
///     num_singular_values: 10,
///     oversampling: 5,
///     power_iterations: 2,
///     ..Default::default()
/// };
///
/// let rsvd = RandomizedSparseSvd::new(config);
/// let result = rsvd.compute(&matrix)?;
///
/// println!("Top singular values: {:?}", result.singular_values);
/// ```
pub struct RandomizedSparseSvd {
    config: RandomizedSparseSvdConfig,
}
impl RandomizedSparseSvd {
    /// Create a new randomized sparse SVD solver with the given configuration.
    pub fn new(config: RandomizedSparseSvdConfig) -> Self {
        Self { config }
    }
    /// Compute randomized SVD of the sparse matrix A.
    ///
    /// Returns the k largest singular values and (optionally) the corresponding
    /// left and right singular vectors.
    pub fn compute<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
        &self,
        a: &CsrMatrix<T>,
    ) -> Result<RandomizedSparseSvdResult<T>, SVDError> {
        let m = a.nrows();
        let n = a.ncols();
        let k = self.config.num_singular_values;
        let p = self.config.oversampling;
        let q = self.config.power_iterations;
        if k == 0 {
            return Err(SVDError::InvalidConfig(
                "num_singular_values must be positive".to_string(),
            ));
        }
        if k + p > n || k + p > m {
            return Err(SVDError::InvalidConfig(
                "k + oversampling must be <= min(m, n)".to_string(),
            ));
        }
        let sample_size = k + p;
        let omega: Vec<Vec<T>> = self.generate_random_matrix(n, sample_size);
        let mut y = vec![vec![T::zero(); sample_size]; m];
        for j in 0..sample_size {
            let omega_col: Vec<T> = (0..n).map(|i| omega[i][j].clone()).collect();
            let mut result = vec![T::zero(); m];
            spmv(T::one(), a, &omega_col, T::zero(), &mut result);
            for i in 0..m {
                y[i][j] = result[i].clone();
            }
        }
        for _ in 0..q {
            let mut y_new = vec![vec![T::zero(); sample_size]; m];
            for j in 0..sample_size {
                let y_col: Vec<T> = (0..m).map(|i| y[i][j].clone()).collect();
                let mut aty = vec![T::zero(); n];
                spmv_transpose(T::one(), a, &y_col, T::zero(), &mut aty);
                let mut result = vec![T::zero(); m];
                spmv(T::one(), a, &aty, T::zero(), &mut result);
                for i in 0..m {
                    y_new[i][j] = result[i].clone();
                }
            }
            y = y_new;
            self.orthonormalize_columns(&mut y)?;
        }
        self.orthonormalize_columns(&mut y)?;
        let q_matrix = y;
        let b = self.compute_qt_times_a(&q_matrix, a)?;
        let (u_b, sigma, v) = self.dense_svd_of_small_matrix(&b, k)?;
        let u = if self.config.compute_vectors {
            let mut u_vecs = Vec::with_capacity(k);
            for j in 0..k {
                let mut u_col = vec![T::zero(); m];
                for i in 0..m {
                    let mut sum = T::zero();
                    for l in 0..sample_size {
                        sum = sum + q_matrix[i][l].clone() * u_b[l][j].clone();
                    }
                    u_col[i] = sum;
                }
                u_vecs.push(u_col);
            }
            Some(u_vecs)
        } else {
            None
        };
        Ok(RandomizedSparseSvdResult {
            singular_values: sigma,
            u,
            v: if self.config.compute_vectors {
                Some(v)
            } else {
                None
            },
        })
    }
    /// Generate a random matrix using a simple LCG (Linear Congruential Generator).
    fn generate_random_matrix<T: Field + FromPrimitive>(
        &self,
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<T>> {
        let mut state = self.config.seed.unwrap_or(12345);
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 1 << 31;
        let mut matrix = vec![vec![T::zero(); cols]; rows];
        for col in matrix.iter_mut().take(rows) {
            for val in col.iter_mut().take(cols) {
                state = (a.wrapping_mul(state).wrapping_add(c)) % m;
                let u = (state as f64) / (m as f64);
                *val = T::from_f64(u - 0.5).unwrap_or(T::zero());
            }
        }
        matrix
    }
    /// Orthonormalize columns using modified Gram-Schmidt.
    fn orthonormalize_columns<T: Field + Real + FromPrimitive>(
        &self,
        y: &mut Vec<Vec<T>>,
    ) -> Result<(), SVDError> {
        if y.is_empty() || y[0].is_empty() {
            return Ok(());
        }
        let m = y.len();
        let n = y[0].len();
        let eps = T::from_f64(1e-14).unwrap_or(T::zero());
        for j in 0..n {
            for k in 0..j {
                let mut dot = T::zero();
                for i in 0..m {
                    dot = dot + y[i][j].clone() * y[i][k].clone();
                }
                for i in 0..m {
                    y[i][j] = y[i][j].clone() - dot.clone() * y[i][k].clone();
                }
            }
            let mut norm_sq = T::zero();
            for row in y.iter().take(m) {
                norm_sq = norm_sq + row[j].clone() * row[j].clone();
            }
            let norm = Real::sqrt(norm_sq);
            if norm > eps {
                for row in y.iter_mut().take(m) {
                    row[j] = row[j].clone() / norm.clone();
                }
            }
        }
        Ok(())
    }
    /// Compute B = Q^T * A where Q is stored as row-major vectors.
    fn compute_qt_times_a<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
        &self,
        q: &[Vec<T>],
        a: &CsrMatrix<T>,
    ) -> Result<Vec<Vec<T>>, SVDError> {
        let m = q.len();
        let sample_size = if m > 0 { q[0].len() } else { 0 };
        let n = a.ncols();
        let mut b = vec![vec![T::zero(); n]; sample_size];
        for j in 0..sample_size {
            for l in 0..m {
                let q_lj = q[l][j].clone();
                if q_lj == T::zero() {
                    continue;
                }
                let row_start = a.row_ptrs()[l];
                let row_end = a.row_ptrs()[l + 1];
                for idx in row_start..row_end {
                    let col = a.col_indices()[idx];
                    let val = a.values()[idx].clone();
                    b[j][col] = b[j][col].clone() + q_lj.clone() * val;
                }
            }
        }
        Ok(b)
    }
    /// Compute dense SVD of small matrix B (sample_size × n) and return top k components.
    fn dense_svd_of_small_matrix<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
        &self,
        b: &[Vec<T>],
        k: usize,
    ) -> Result<(Vec<Vec<T>>, Vec<T>, Vec<Vec<T>>), SVDError> {
        let sample_size = b.len();
        let n = if sample_size > 0 { b[0].len() } else { 0 };
        if sample_size == 0 || n == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let mut bbt = vec![vec![T::zero(); sample_size]; sample_size];
        for i in 0..sample_size {
            for j in 0..sample_size {
                let mut sum = T::zero();
                for l in 0..n {
                    sum = sum + b[i][l].clone() * b[j][l].clone();
                }
                bbt[i][j] = sum;
            }
        }
        let (eigenvalues, eigenvectors) = self.symmetric_eigen(&bbt, k)?;
        let sigma: Vec<T> = eigenvalues
            .iter()
            .map(|&e| {
                if e > T::zero() {
                    Real::sqrt(e)
                } else {
                    T::zero()
                }
            })
            .collect();
        let u_b = eigenvectors;
        let mut v = Vec::with_capacity(k);
        for j in 0..k {
            let mut v_col = vec![T::zero(); n];
            let sigma_inv = if sigma[j] > T::from_f64(1e-15).unwrap_or(T::zero()) {
                T::one() / sigma[j].clone()
            } else {
                T::zero()
            };
            for i in 0..n {
                let mut sum = T::zero();
                for l in 0..sample_size {
                    sum = sum + b[l][i].clone() * u_b[l][j].clone();
                }
                v_col[i] = sum * sigma_inv.clone();
            }
            v.push(v_col);
        }
        Ok((u_b, sigma, v))
    }
    /// Simple symmetric eigenvalue decomposition using power iteration with deflation.
    fn symmetric_eigen<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
        &self,
        a: &[Vec<T>],
        k: usize,
    ) -> Result<(Vec<T>, Vec<Vec<T>>), SVDError> {
        let n = a.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let eps = T::from_f64(1e-12).unwrap_or(T::zero());
        let max_iter = 1000;
        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors = vec![vec![T::zero(); k]; n];
        let mut work = a.to_vec();
        for idx in 0..k.min(n) {
            let mut v: Vec<T> = (0..n)
                .map(|i| T::from_f64((i + 1) as f64).unwrap_or(T::one()))
                .collect();
            let mut norm_sq = T::zero();
            for vi in &v {
                norm_sq = norm_sq + vi.clone() * vi.clone();
            }
            let norm = Real::sqrt(norm_sq);
            for vi in &mut v {
                *vi = vi.clone() / norm.clone();
            }
            let mut lambda = T::zero();
            for _ in 0..max_iter {
                let mut v_new = vec![T::zero(); n];
                for i in 0..n {
                    for j in 0..n {
                        v_new[i] = v_new[i].clone() + work[i][j].clone() * v[j].clone();
                    }
                }
                let mut numerator = T::zero();
                let mut denominator = T::zero();
                for i in 0..n {
                    numerator = numerator + v[i].clone() * v_new[i].clone();
                    denominator = denominator + v[i].clone() * v[i].clone();
                }
                let new_lambda = numerator / denominator;
                let mut norm_sq = T::zero();
                for vi in &v_new {
                    norm_sq = norm_sq + vi.clone() * vi.clone();
                }
                let norm = Real::sqrt(norm_sq);
                if norm < eps {
                    break;
                }
                for vi in &mut v_new {
                    *vi = vi.clone() / norm.clone();
                }
                if Scalar::abs(new_lambda.clone() - lambda.clone()) < eps {
                    break;
                }
                lambda = new_lambda;
                v = v_new;
            }
            eigenvalues.push(lambda.clone());
            for i in 0..n {
                eigenvectors[i][idx] = v[i].clone();
            }
            for i in 0..n {
                for j in 0..n {
                    work[i][j] = work[i][j].clone() - lambda.clone() * v[i].clone() * v[j].clone();
                }
            }
        }
        Ok((eigenvalues, eigenvectors))
    }
}
/// Truncated SVD solver using Lanczos bidiagonalization.
///
/// Computes the k largest singular values and (optionally) vectors of a sparse matrix.
///
/// # Algorithm
///
/// Uses Lanczos iteration on A^T A to find eigenvalues and eigenvectors:
/// - Eigenvalues of A^T A = σ^2 (squared singular values)
/// - Eigenvectors of A^T A = V (right singular vectors)
/// - Left singular vectors: U = A V Σ^{-1}
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::svd::{TruncatedSVD, TruncatedSVDConfig};
///
/// let config = TruncatedSVDConfig {
///     num_singular_values: 5,
///     ..Default::default()
/// };
///
/// let svd_solver = TruncatedSVD::new(config);
/// let result = svd_solver.compute(&matrix)?;
///
/// println!("Singular values: {:?}", result.singular_values);
/// ```
pub struct TruncatedSVD<T> {
    config: TruncatedSVDConfig<T>,
}
impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> TruncatedSVD<T> {
    /// Create a new truncated SVD solver with the given configuration.
    pub fn new(config: TruncatedSVDConfig<T>) -> Self {
        Self { config }
    }
    /// Compute truncated SVD of the matrix A.
    ///
    /// Returns the k largest singular values and (optionally) the corresponding
    /// left and right singular vectors.
    pub fn compute(&self, a: &CsrMatrix<T>) -> Result<TruncatedSVDResult<T>, SVDError> {
        if self.config.num_singular_values == 0 {
            return Err(SVDError::InvalidConfig(
                "num_singular_values must be positive".to_string(),
            ));
        }
        if self.config.num_singular_values >= a.nrows().min(a.ncols()) {
            return Err(SVDError::InvalidConfig(
                "num_singular_values must be less than min(m, n)".to_string(),
            ));
        }
        if self.config.krylov_dimension <= self.config.num_singular_values {
            return Err(SVDError::InvalidConfig(
                "krylov_dimension must be greater than num_singular_values".to_string(),
            ));
        }
        let m = a.nrows();
        let n = a.ncols();
        let use_ata = n < m;
        if use_ata {
            self.compute_via_ata(a)
        } else {
            self.compute_via_aat(a)
        }
    }
    /// Compute SVD via eigendecomposition of A^T A.
    ///
    /// This is efficient when n < m (more rows than columns).
    /// We get V directly from eigenvectors, then compute U = A V Σ^{-1}.
    fn compute_via_ata(&self, a: &CsrMatrix<T>) -> Result<TruncatedSVDResult<T>, SVDError> {
        let m = a.nrows();
        let _n = a.ncols();
        let ata_matrix = ata(a);
        let lanczos_config = LanczosConfig {
            num_eigenvalues: self.config.num_singular_values,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            compute_eigenvectors: self.config.compute_vectors,
            krylov_dimension: self.config.krylov_dimension,
            full_reorthogonalization: self.config.full_reorthogonalization,
        };
        let lanczos = Lanczos::new(lanczos_config);
        let lanczos_result = lanczos
            .compute(&ata_matrix, None)
            .map_err(|e| SVDError::ComputationError(format!("Lanczos failed: {:?}", e)))?;
        let singular_values: Vec<T> = lanczos_result
            .eigenvalues
            .iter()
            .map(|&lambda| {
                if lambda < T::zero() {
                    T::zero()
                } else {
                    Real::sqrt(lambda)
                }
            })
            .collect();
        let v = lanczos_result.eigenvectors;
        let u = if self.config.compute_vectors && v.is_some() {
            let v_vecs = v.as_ref().expect("value should be present");
            let mut u_vecs = Vec::with_capacity(singular_values.len());
            for (i, v_i) in v_vecs.iter().enumerate() {
                let mut av = vec![T::zero(); m];
                spmv(T::one(), a, v_i, T::zero(), &mut av);
                let sigma_inv = if singular_values[i] > T::from_f64(1e-15).unwrap_or(T::zero()) {
                    T::one() / singular_values[i].clone()
                } else {
                    T::zero()
                };
                let u_i: Vec<T> = av.iter().map(|&x| x * sigma_inv.clone()).collect();
                u_vecs.push(u_i);
            }
            Some(u_vecs)
        } else {
            None
        };
        Ok(TruncatedSVDResult {
            singular_values,
            u,
            v,
            iterations: lanczos_result.iterations,
            converged: lanczos_result.converged,
        })
    }
    /// Compute SVD via eigendecomposition of A A^T.
    ///
    /// This is efficient when m < n (more columns than rows).
    /// We get U directly from eigenvectors, then compute V = A^T U Σ^{-1}.
    fn compute_via_aat(&self, a: &CsrMatrix<T>) -> Result<TruncatedSVDResult<T>, SVDError> {
        let _m = a.nrows();
        let n = a.ncols();
        let aat_matrix = aat(a);
        let lanczos_config = LanczosConfig {
            num_eigenvalues: self.config.num_singular_values,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            compute_eigenvectors: self.config.compute_vectors,
            krylov_dimension: self.config.krylov_dimension,
            full_reorthogonalization: self.config.full_reorthogonalization,
        };
        let lanczos = Lanczos::new(lanczos_config);
        let lanczos_result = lanczos
            .compute(&aat_matrix, None)
            .map_err(|e| SVDError::ComputationError(format!("Lanczos failed: {:?}", e)))?;
        let singular_values: Vec<T> = lanczos_result
            .eigenvalues
            .iter()
            .map(|&lambda| {
                if lambda < T::zero() {
                    T::zero()
                } else {
                    Real::sqrt(lambda)
                }
            })
            .collect();
        let u = lanczos_result.eigenvectors;
        let v = if self.config.compute_vectors && u.is_some() {
            let u_vecs = u.as_ref().expect("value should be present");
            let mut v_vecs = Vec::with_capacity(singular_values.len());
            for (i, u_i) in u_vecs.iter().enumerate() {
                let mut atu = vec![T::zero(); n];
                spmv_transpose(T::one(), a, u_i, T::zero(), &mut atu);
                let sigma_inv = if singular_values[i] > T::from_f64(1e-15).unwrap_or(T::zero()) {
                    T::one() / singular_values[i].clone()
                } else {
                    T::zero()
                };
                let v_i: Vec<T> = atu.iter().map(|&x| x * sigma_inv.clone()).collect();
                v_vecs.push(v_i);
            }
            Some(v_vecs)
        } else {
            None
        };
        Ok(TruncatedSVDResult {
            singular_values,
            u,
            v,
            iterations: lanczos_result.iterations,
            converged: lanczos_result.converged,
        })
    }
}
/// Result of randomized sparse SVD computation.
#[derive(Debug, Clone)]
pub struct RandomizedSparseSvdResult<T> {
    /// Singular values (in descending order).
    pub singular_values: Vec<T>,
    /// Left singular vectors (U), columns are vectors (m × k).
    pub u: Option<Vec<Vec<T>>>,
    /// Right singular vectors (V), columns are vectors (n × k).
    pub v: Option<Vec<Vec<T>>>,
}
/// Incremental SVD solver for online/streaming applications.
///
/// Maintains an approximate low-rank SVD A ≈ UΣV^T and efficiently
/// updates it when new rows or columns are added. Based on Brand (2006) algorithm.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::svd::{IncrementalSVD, IncrementalSVDConfig};
///
/// let config = IncrementalSVDConfig::default();
/// let mut isvd = IncrementalSVD::new(config);
///
/// // Initialize with matrix
/// isvd.initialize(&initial_matrix, 5).unwrap();
///
/// // Add new rows
/// isvd.add_rows(&new_rows).unwrap();
///
/// // Get current SVD
/// let (u, s, vt) = isvd.get_svd();
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalSVD<T> {
    config: IncrementalSVDConfig<T>,
    /// Left singular vectors (m × k)
    u: Vec<Vec<T>>,
    /// Singular values (k)
    s: Vec<T>,
    /// Right singular vectors transposed (k × n)
    vt: Vec<Vec<T>>,
    /// Current number of rows
    m: usize,
    /// Current number of columns
    n: usize,
    /// Current rank
    k: usize,
}
impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> IncrementalSVD<T> {
    /// Create a new incremental SVD solver.
    pub fn new(config: IncrementalSVDConfig<T>) -> Self {
        Self {
            config,
            u: Vec::new(),
            s: Vec::new(),
            vt: Vec::new(),
            m: 0,
            n: 0,
            k: 0,
        }
    }
    /// Initialize the SVD with an initial matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Initial sparse matrix (m × n)
    /// * `rank` - Target rank for approximation
    ///
    /// # Returns
    ///
    /// * `Ok(())` on success
    /// * `Err(SVDError)` on failure
    pub fn initialize(&mut self, a: &CsrMatrix<T>, rank: usize) -> Result<(), SVDError> {
        self.m = a.nrows();
        self.n = a.ncols();
        if self.m == 0 || self.n == 0 {
            return Err(SVDError::InvalidConfig(
                "Matrix must be non-empty".to_string(),
            ));
        }
        let k = rank.min(self.m).min(self.n);
        if k == 0 {
            return Err(SVDError::InvalidConfig("Rank must be positive".to_string()));
        }
        let min_dim = self.m.min(self.n);
        let oversampling = if k + 5 <= min_dim {
            5
        } else {
            min_dim.saturating_sub(k)
        };
        let rsvd_config = RandomizedSparseSvdConfig {
            num_singular_values: k,
            oversampling,
            power_iterations: 2,
            compute_vectors: true,
            ..Default::default()
        };
        let rsvd = RandomizedSparseSvd::new(rsvd_config);
        let result = rsvd.compute(a)?;
        self.s = result.singular_values;
        self.k = self.s.len();
        if let (Some(u_vecs), Some(v_vecs)) = (result.u, result.v) {
            self.u = transpose_dense(&u_vecs);
            self.vt = v_vecs;
        } else {
            return Err(SVDError::ComputationError(
                "Failed to compute singular vectors".to_string(),
            ));
        }
        Ok(())
    }
    /// Add new rows to the matrix and update SVD.
    ///
    /// Updates A from (m × n) to ((m+p) × n) where p = new_rows.len().
    ///
    /// # Arguments
    ///
    /// * `new_rows` - New rows as dense vectors
    ///
    /// # Returns
    ///
    /// * `Ok(())` on success
    /// * `Err(SVDError)` on failure
    pub fn add_rows(&mut self, new_rows: &[Vec<T>]) -> Result<(), SVDError> {
        if new_rows.is_empty() {
            return Ok(());
        }
        let p = new_rows.len();
        for row in new_rows {
            if row.len() != self.n {
                return Err(SVDError::InvalidConfig(format!(
                    "New row has {} columns, expected {}",
                    row.len(),
                    self.n
                )));
            }
        }
        if self.k == 0 {
            return Err(SVDError::ComputationError(
                "SVD not initialized".to_string(),
            ));
        }
        let mut m_matrix = vec![vec![T::zero(); self.k]; p];
        for i in 0..p {
            for j in 0..self.k {
                let mut sum = T::zero();
                for l in 0..self.n {
                    sum = sum + new_rows[i][l].clone() * self.vt[j][l].clone();
                }
                m_matrix[i][j] = sum;
            }
        }
        let mut r = new_rows.to_vec();
        for i in 0..p {
            for l in 0..self.n {
                for j in 0..self.k {
                    r[i][l] = r[i][l].clone() - m_matrix[i][j].clone() * self.vt[j][l].clone();
                }
            }
        }
        let r_transpose = transpose_dense(&r);
        let (q_r, n_matrix) = qr_decompose_dense(&r_transpose);
        let p_prime = q_r[0].len();
        if p_prime == 0 {
            for i in 0..p {
                let mut u_row = vec![T::zero(); self.k];
                for j in 0..self.k {
                    if Scalar::abs(self.s[j].clone()) > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                        u_row[j] = m_matrix[i][j].clone() / self.s[j].clone();
                    }
                }
                self.u.push(u_row);
            }
            self.m += p;
            return Ok(());
        }
        let new_k = self.k + p_prime;
        let mut k_matrix = vec![vec![T::zero(); new_k]; new_k];
        for i in 0..self.k {
            k_matrix[i][i] = self.s[i].clone();
        }
        for i in 0..self.k {
            for j in 0..p {
                k_matrix[i][self.k + j] = m_matrix[j][i].clone();
            }
        }
        for i in 0..p_prime {
            for j in 0..p {
                if i < n_matrix.len() && j < n_matrix[0].len() {
                    k_matrix[self.k + j][self.k + i] = n_matrix[i][j].clone();
                }
            }
        }
        let k_svd_result = dense_svd_full(&k_matrix)?;
        let u_k = k_svd_result.u.ok_or_else(|| {
            SVDError::ComputationError("Failed to compute U in K SVD".to_string())
        })?;
        let s_new = k_svd_result.singular_values;
        let k_new = s_new.len().min(self.config.max_rank);
        let mut u_new = vec![vec![T::zero(); k_new]; self.m + p];
        for i in 0..self.m {
            for j in 0..k_new {
                let mut sum = T::zero();
                for l in 0..self.k {
                    sum = sum + self.u[i][l].clone() * u_k[l][j].clone();
                }
                u_new[i][j] = sum;
            }
        }
        for i in 0..p {
            for j in 0..k_new {
                u_new[self.m + i][j] = u_k[self.k + i][j].clone();
            }
        }
        self.u = u_new;
        self.s = s_new.into_iter().take(k_new).collect();
        while self.vt.len() < k_new {
            self.vt.push(vec![T::zero(); self.n]);
        }
        self.k = k_new;
        self.m += p;
        Ok(())
    }
    /// Add new columns to the matrix and update SVD.
    ///
    /// Updates A from (m × n) to (m × (n+q)) where q = new_cols.len().
    ///
    /// # Arguments
    ///
    /// * `new_cols` - New columns as dense vectors (each vector has m elements)
    ///
    /// # Returns
    ///
    /// * `Ok(())` on success
    /// * `Err(SVDError)` on failure
    pub fn add_columns(&mut self, new_cols: &[Vec<T>]) -> Result<(), SVDError> {
        if new_cols.is_empty() {
            return Ok(());
        }
        let q = new_cols.len();
        for col in new_cols {
            if col.len() != self.m {
                return Err(SVDError::InvalidConfig(format!(
                    "New column has {} rows, expected {}",
                    col.len(),
                    self.m
                )));
            }
        }
        if self.k == 0 {
            return Err(SVDError::ComputationError(
                "SVD not initialized".to_string(),
            ));
        }
        let mut p_matrix = vec![vec![T::zero(); q]; self.k];
        for i in 0..self.k {
            for j in 0..q {
                let mut sum = T::zero();
                for l in 0..self.m {
                    sum = sum + self.u[l][i].clone() * new_cols[j][l].clone();
                }
                p_matrix[i][j] = sum;
            }
        }
        let mut q_c = new_cols.to_vec();
        for j in 0..q {
            for l in 0..self.m {
                for i in 0..self.k {
                    q_c[j][l] = q_c[j][l].clone() - self.u[l][i].clone() * p_matrix[i][j].clone();
                }
            }
        }
        let mut q_c_matrix = vec![vec![T::zero(); q]; self.m];
        for i in 0..self.m {
            for j in 0..q {
                q_c_matrix[i][j] = q_c[j][i].clone();
            }
        }
        let (q_q, n_matrix) = qr_decompose_dense(&q_c_matrix);
        let q_prime = if q_q.is_empty() { 0 } else { q_q[0].len() };
        if q_prime == 0 {
            for i in 0..self.k {
                for j in 0..q {
                    if Scalar::abs(self.s[i].clone()) > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                        self.vt[i].push(p_matrix[i][j].clone() / self.s[i].clone());
                    } else {
                        self.vt[i].push(T::zero());
                    }
                }
            }
            self.n += q;
            return Ok(());
        }
        let new_k = self.k + q_prime;
        let mut l_matrix = vec![vec![T::zero(); new_k]; new_k];
        for i in 0..self.k {
            l_matrix[i][i] = self.s[i].clone();
        }
        for i in 0..self.k {
            for j in 0..q {
                l_matrix[i][self.k + j] = p_matrix[i][j].clone();
            }
        }
        for i in 0..q_prime {
            for j in 0..q {
                if i < n_matrix.len() && j < n_matrix[0].len() {
                    l_matrix[self.k + j][self.k + i] = n_matrix[i][j].clone();
                }
            }
        }
        let l_svd_result = dense_svd_full(&l_matrix)?;
        let v_l = l_svd_result.v.ok_or_else(|| {
            SVDError::ComputationError("Failed to compute V in L SVD".to_string())
        })?;
        let s_new = l_svd_result.singular_values;
        let k_new = s_new.len().min(self.config.max_rank);
        let mut vt_new = vec![vec![T::zero(); self.n + q]; k_new];
        for i in 0..k_new {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..self.k {
                    sum = sum + v_l[l][i].clone() * self.vt[l][j].clone();
                }
                vt_new[i][j] = sum;
            }
        }
        for i in 0..k_new {
            for j in 0..q {
                vt_new[i][self.n + j] = v_l[self.k + j][i].clone();
            }
        }
        self.vt = vt_new;
        self.s = s_new.into_iter().take(k_new).collect();
        self.k = k_new;
        self.n += q;
        Ok(())
    }
    /// Get current SVD factors.
    ///
    /// # Returns
    ///
    /// * `(U, Σ, V^T)` - Left vectors, singular values, right vectors transposed
    pub fn get_svd(&self) -> (&[Vec<T>], &[T], &[Vec<T>]) {
        (&self.u, &self.s, &self.vt)
    }
    /// Get current rank.
    pub fn rank(&self) -> usize {
        self.k
    }
    /// Get current dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}
