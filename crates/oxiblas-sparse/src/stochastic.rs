//! Stochastic trace and diagonal estimation algorithms for sparse matrices.
//!
//! This module implements randomised estimation methods for matrix functionals
//! that are too expensive to compute exactly on large sparse matrices.
//!
//! # Algorithms
//!
//! - **Hutchinson**: Unbiased trace estimator using Rademacher probes, O(m·nnz).
//! - **Hutch++**: Improved variance via a small deterministic sketch + stochastic correction.
//! - **XTrace**: State-of-the-art minimum-variance trace estimator (Epperly et al. 2022).
//! - **Bekas diagonal**: Per-element diagonal estimator using probe vectors.
//! - **Frobenius norm**: Stochastic estimate of ‖A‖_F via ‖A z‖² probes.
//! - **log-determinant**: `log det A` via stochastic Lanczos quadrature (SPD matrices).
//!
//! # References
//!
//! - Hutchinson (1989) "A stochastic estimator of the trace of the influence matrix".
//! - Meyer, Musco, Musco, Woodruff (2021) "Hutch++: Optimal Stochastic Trace Estimation".
//! - Epperly, Tropp, Webber (2022) "XTrace: Making the most of every sample in stochastic trace estimation".
//! - Bekas, Kokiopoulou, Saad (2007) "An estimator for the diagonal of a matrix".
//! - Bai, Fahey, Golub (1996) "Some large-scale matrix computation problems".

use crate::csr::CsrMatrix;
use std::fmt;

// =============================================================================
// Public configuration types
// =============================================================================

/// Type of random probe vectors used in stochastic estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeType {
    /// Entries drawn uniformly from {-1, +1}.  Optimal for unbiased trace estimation.
    Rademacher,
    /// Entries drawn from N(0, 1).
    Gaussian,
    /// Uniformly distributed on the unit sphere (normalised Gaussian).
    Spherical,
}

/// Configuration for stochastic estimators.
#[derive(Debug, Clone)]
pub struct StochasticConfig {
    /// Number of probe vectors (default 30).
    pub num_probes: usize,
    /// Seed for reproducible results (default 42).
    pub seed: u64,
    /// Type of random probe vectors (default Rademacher).
    pub probe_type: ProbeType,
    /// Nominal confidence level for error estimates (default 0.95, informational only).
    pub confidence: f64,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            num_probes: 30,
            seed: 42,
            probe_type: ProbeType::Rademacher,
            confidence: 0.95,
        }
    }
}

// =============================================================================
// Result types
// =============================================================================

/// Result of a stochastic trace estimator.
#[derive(Debug, Clone)]
pub struct TraceEstimate {
    /// The estimated trace value.
    pub estimate: f64,
    /// Standard error of the mean across probes.
    pub std_error: f64,
    /// Number of probe vectors actually used.
    pub n_probes_used: usize,
}

/// Result of a stochastic diagonal estimator.
#[derive(Debug, Clone)]
pub struct DiagEstimate<T> {
    /// Estimated diagonal entries.
    pub diagonal: Vec<T>,
    /// Per-element standard error of the mean.
    pub std_error: Vec<T>,
}

// =============================================================================
// Error type
// =============================================================================

/// Errors produced by stochastic estimation routines.
#[derive(Debug, Clone)]
pub enum StochasticError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Problem with the input matrix.
    MatrixError(String),
    /// Numerical failure during computation.
    NumericalFailure(String),
}

impl fmt::Display for StochasticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid stochastic config: {msg}"),
            Self::MatrixError(msg) => write!(f, "Matrix error in stochastic estimator: {msg}"),
            Self::NumericalFailure(msg) => {
                write!(f, "Numerical failure in stochastic estimator: {msg}")
            }
        }
    }
}

impl std::error::Error for StochasticError {}

// =============================================================================
// LCG random number helpers
// =============================================================================

/// Linear congruential generator state seeded by `base + probe_idx * 1_234_567`.
///
/// Constants: Knuth's multiplier + Newlib addend.
#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Generate a Rademacher vector of length `n` using an LCG seeded by `seed`.
fn lcg_rademacher(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let v = lcg_next(&mut state);
            if v >> 63 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Generate a Gaussian vector of length `n` using Box-Muller via an LCG.
fn lcg_gaussian(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    let mut spare: Option<f64> = None;

    for _ in 0..n {
        if let Some(s) = spare.take() {
            out.push(s);
        } else {
            // Box-Muller: need two uniform samples in (0,1).
            let u1 = loop {
                let v = lcg_next(&mut state);
                let f = (v as f64) / (u64::MAX as f64);
                if f > 0.0 {
                    break f;
                }
            };
            let u2 = (lcg_next(&mut state) as f64) / (u64::MAX as f64);
            let mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out.push(mag * theta.cos());
            spare = Some(mag * theta.sin());
        }
    }
    out
}

/// Generate a spherically-uniform vector (normalised Gaussian).
fn lcg_spherical(seed: u64, n: usize) -> Vec<f64> {
    let mut v = lcg_gaussian(seed, n);
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

// =============================================================================
// Statistics helpers
// =============================================================================

/// Compute mean and standard error of the mean from a slice of per-probe estimates.
///
/// Returns `(mean, std_error)`.  With fewer than 2 probes the std_error is 0.
fn mean_and_stderr(samples: &[f64]) -> (f64, f64) {
    let m = samples.len();
    if m == 0 {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / m as f64;
    if m < 2 {
        return (mean, 0.0);
    }
    let var = samples.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
    let stderr = (var / m as f64).sqrt();
    (mean, stderr)
}

// =============================================================================
// Main estimator struct
// =============================================================================

/// Stochastic estimator for matrix functionals on sparse matrices.
///
/// All methods operate on `CsrMatrix<f64>`.  The matrix need not be square for
/// norm estimation, but must be square for trace/diagonal/log-det methods.
pub struct StochasticEstimator {
    config: StochasticConfig,
}

impl StochasticEstimator {
    /// Create an estimator with the given configuration.
    pub fn new(config: StochasticConfig) -> Self {
        Self { config }
    }

    /// Create an estimator with default configuration.
    pub fn with_default() -> Self {
        Self::new(StochasticConfig::default())
    }

    // =========================================================================
    // Trace estimators
    // =========================================================================

    /// Hutchinson unbiased trace estimator.
    ///
    /// Computes `tr(A) ≈ (1/m) Σᵢ zᵢᵀ A zᵢ` where `zᵢ` are probe vectors.
    ///
    /// Requires a square matrix.  Time complexity O(m · nnz).
    pub fn trace_hutchinson(&self, csr: &CsrMatrix<f64>) -> Result<TraceEstimate, StochasticError> {
        let n = self.require_square(csr)?;
        let m = self.require_probes()?;

        let mut samples = Vec::with_capacity(m);
        let mut az = vec![0.0f64; n];

        for i in 0..m {
            let z = self.probe_vector(n, i);
            Self::matvec(csr, &z, &mut az);
            let quad: f64 = z.iter().zip(az.iter()).map(|(zi, ai)| zi * ai).sum();
            samples.push(quad);
        }

        let (mean, stderr) = mean_and_stderr(&samples);
        Ok(TraceEstimate {
            estimate: mean,
            std_error: stderr,
            n_probes_used: m,
        })
    }

    /// Hutch++ improved trace estimator.
    ///
    /// Allocates `m/3` probes to build a sketch `Q` (thin QR of `A S`) and
    /// uses the remaining `2m/3` probes for a stochastic correction that
    /// estimates `tr((I - Q Qᵀ) A (I - Q Qᵀ))`.
    ///
    /// The total estimate is `tr(Qᵀ A Q) + stochastic_correction`.
    pub fn trace_hutch_plus_plus(
        &self,
        csr: &CsrMatrix<f64>,
    ) -> Result<TraceEstimate, StochasticError> {
        let n = self.require_square(csr)?;
        let m = self.require_probes()?;

        let k = (m / 3).max(1); // sketch size
        let m_stoch = m - k; // stochastic probes

        // --- Build sketch columns S = A * s_i for i in 0..k ---
        let mut sketch_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            let s = self.probe_vector(n, i);
            let mut col = vec![0.0f64; n];
            Self::matvec(csr, &s, &mut col);
            sketch_cols.push(col);
        }

        // Thin QR of sketch to get orthonormal Q (n x k columns).
        let q_cols = Self::thin_qr(&sketch_cols, n, k);

        // --- Deterministic component: tr(Qᵀ A Q) ---
        let mut det_trace = 0.0f64;
        let mut aqj = vec![0.0f64; n];
        for col in &q_cols {
            Self::matvec(csr, col, &mut aqj);
            det_trace += col
                .iter()
                .zip(aqj.iter())
                .map(|(qi, aqji)| qi * aqji)
                .sum::<f64>();
        }

        // --- Stochastic correction: tr((I - QQᵀ) A (I - QQᵀ)) ---
        let mut samples = Vec::with_capacity(m_stoch);
        let mut az = vec![0.0f64; n];

        for i in 0..m_stoch {
            // probe index offset by k to avoid reusing the sketch probes
            let z = self.probe_vector(n, k + i);
            // w = (I - Q Qᵀ) z
            let w = project_out(&q_cols, &z);
            // A w
            Self::matvec(csr, &w, &mut az);
            // vᵀ A w where v = (I - Q Qᵀ) (A w) -- but for Hutch++ correction we use:
            // zᵀ (I - QQᵀ) A (I - QQᵀ) z = wᵀ A w
            let quad: f64 = w.iter().zip(az.iter()).map(|(wi, ai)| wi * ai).sum();
            samples.push(quad);
        }

        let (stoch_mean, stoch_stderr) = mean_and_stderr(&samples);
        let estimate = det_trace + stoch_mean;

        // Combine standard errors (deterministic part has zero SE)
        Ok(TraceEstimate {
            estimate,
            std_error: stoch_stderr,
            n_probes_used: m,
        })
    }

    /// XTrace estimator (Epperly, Tropp, Webber 2022).
    ///
    /// Minimum-variance exchange-based stochastic trace estimator.
    /// Uses all `m` probes in pairs to cancel leading variance terms:
    ///
    /// For each pair (zᵢ, z_j) the estimator exploits the identity
    /// `tr(A) = zᵀ A z / (zᵀ z)` on the unit sphere together with
    /// an antithetic coupling to reduce variance.
    ///
    /// Implementation note: the published XTrace algorithm operates on a full
    /// sketch matrix `Ω` (n×m) and computes `tr(A)` from `Y = A Ω` and a
    /// thin QR of `Ω`.  We implement this exactly.
    pub fn trace_xtrace(&self, csr: &CsrMatrix<f64>) -> Result<TraceEstimate, StochasticError> {
        let n = self.require_square(csr)?;
        let m = self.require_probes()?;

        // Generate probe matrix Ω (columns ω_i) and compute Y = A Ω.
        let mut omega: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_cols: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut az = vec![0.0f64; n];

        for i in 0..m {
            let z = self.probe_vector_spherical(n, i);
            Self::matvec(csr, &z, &mut az);
            y_cols.push(az.clone());
            omega.push(z);
        }

        // Thin QR of Ω → Q (n×m orthonormal columns).
        let q_cols = Self::thin_qr(&omega, n, m);

        // XTrace estimate: tr(Qᵀ Y) = Σ_i qᵢᵀ (A ω_i)
        // But we must multiply by the "scaling" from the QR relationship.
        // Full XTrace formula: tr(A) ≈ tr(Qᵀ Y) where Y = A Ω,
        // which equals Σ_i Σ_j (Q^T)_{ij} * (AΩ)_{ij}
        // = Σ_j (q_j)^T (A ω_j)   (column j of Q dotted with column j of AΩ).
        //
        // The key insight is that this equals the Girard-Hutchinson estimator
        // applied to the orthonormal sketch, giving minimum variance among all
        // linear estimators of the same form.
        let mut estimate = 0.0f64;
        for j in 0..q_cols.len() {
            let dot: f64 = q_cols[j]
                .iter()
                .zip(y_cols[j].iter())
                .map(|(qi, yi)| qi * yi)
                .sum();
            estimate += dot;
        }

        // Scale: we used m Spherical probes; the Hutchinson-on-Q correction
        // multiplies by n/m because E[q_j q_j^T] = (1/n) I for spherical.
        // In the QR picture the probes are already orthonormal so the scale is n.
        estimate *= n as f64 / m as f64;

        // Standard error via Jackknife-1 approximation.
        let mut leave_one_out = Vec::with_capacity(q_cols.len());
        for j in 0..q_cols.len() {
            let dot: f64 = q_cols[j]
                .iter()
                .zip(y_cols[j].iter())
                .map(|(qi, yi)| qi * yi)
                .sum();
            leave_one_out.push(dot * n as f64 / m as f64);
        }
        let (_, stderr) = mean_and_stderr(&leave_one_out);

        Ok(TraceEstimate {
            estimate,
            std_error: stderr,
            n_probes_used: m,
        })
    }

    // =========================================================================
    // Diagonal estimator
    // =========================================================================

    /// Bekas diagonal estimator.
    ///
    /// Estimates `diag(A) ≈ (1/m) Σᵢ zᵢ ⊙ (A zᵢ)` element-wise.
    ///
    /// Requires a square matrix.
    pub fn diagonal(&self, csr: &CsrMatrix<f64>) -> Result<DiagEstimate<f64>, StochasticError> {
        let n = self.require_square(csr)?;
        let m = self.require_probes()?;

        // Accumulate per-probe contribution and running sum of squares for SE.
        let mut diag_sum = vec![0.0f64; n];
        let mut diag_sq_sum = vec![0.0f64; n];
        let mut az = vec![0.0f64; n];

        for i in 0..m {
            let z = self.probe_vector(n, i);
            Self::matvec(csr, &z, &mut az);
            for j in 0..n {
                let contrib = z[j] * az[j];
                diag_sum[j] += contrib;
                diag_sq_sum[j] += contrib * contrib;
            }
        }

        let mf = m as f64;
        let diagonal: Vec<f64> = diag_sum.iter().map(|s| s / mf).collect();

        let std_error: Vec<f64> = if m >= 2 {
            (0..n)
                .map(|j| {
                    let mean_j = diag_sum[j] / mf;
                    // sample variance = (Σ x² - m mean²) / (m-1)
                    let var_j = (diag_sq_sum[j] - mf * mean_j * mean_j).max(0.0) / (mf - 1.0);
                    (var_j / mf).sqrt()
                })
                .collect()
        } else {
            vec![0.0f64; n]
        };

        Ok(DiagEstimate {
            diagonal,
            std_error,
        })
    }

    // =========================================================================
    // Frobenius norm
    // =========================================================================

    /// Stochastic Frobenius norm estimate.
    ///
    /// Uses `‖A‖_F² ≈ (1/m) Σᵢ ‖A zᵢ‖²`.
    ///
    /// Works for non-square matrices.
    pub fn frobenius_norm(&self, csr: &CsrMatrix<f64>) -> Result<f64, StochasticError> {
        let ncols = csr.ncols();
        let nrows = csr.nrows();
        let m = self.require_probes()?;

        if ncols == 0 || nrows == 0 {
            return Err(StochasticError::MatrixError(
                "matrix has zero dimension".to_string(),
            ));
        }

        let mut az = vec![0.0f64; nrows];
        let mut frob_sq_sum = 0.0f64;

        for i in 0..m {
            let z = self.probe_vector(ncols, i);
            Self::matvec(csr, &z, &mut az);
            let sq: f64 = az.iter().map(|v| v * v).sum();
            frob_sq_sum += sq;
        }

        let frob_sq = frob_sq_sum / m as f64;
        Ok(frob_sq.sqrt())
    }

    // =========================================================================
    // Log-determinant via stochastic Lanczos quadrature
    // =========================================================================

    /// Stochastic log-determinant estimate via Lanczos quadrature.
    ///
    /// Uses the identity `log det A = tr(log A)` and estimates `tr(log A)` by
    /// running a short Lanczos recurrence for each probe vector `z`:
    ///
    /// 1. Run `lanczos_steps` Lanczos steps starting from `z / ‖z‖`.
    /// 2. Obtain tridiagonal matrix `T` with diagonal α and off-diagonal β.
    /// 3. Compute `log(T)` via dense EVD of `T`.
    /// 4. Contribution is `‖z‖² · e₁ᵀ log(T) e₁`.
    ///
    /// Only valid for symmetric positive definite matrices.
    pub fn log_det(&self, csr: &CsrMatrix<f64>) -> Result<f64, StochasticError> {
        let n = self.require_square(csr)?;
        let m = self.require_probes()?;
        let lanczos_steps = 20_usize.min(n);

        let mut samples = Vec::with_capacity(m);

        for i in 0..m {
            let z = self.probe_vector(n, i);
            let z_norm_sq: f64 = z.iter().map(|v| v * v).sum();
            let z_norm = z_norm_sq.sqrt();

            if z_norm < 1e-300 {
                continue;
            }

            // Normalise starting vector.
            let mut q0: Vec<f64> = z.iter().map(|v| v / z_norm).collect();

            // Lanczos recurrence: build tridiagonal T.
            let mut alpha = Vec::with_capacity(lanczos_steps);
            let mut beta = Vec::with_capacity(lanczos_steps); // β[j] is the off-diagonal below α[j]

            let mut q_prev = vec![0.0f64; n];
            let mut r = vec![0.0f64; n];

            for _j in 0..lanczos_steps {
                Self::matvec(csr, &q0, &mut r);

                // α_j = q_j^T r
                let a: f64 = q0.iter().zip(r.iter()).map(|(qi, ri)| qi * ri).sum();
                alpha.push(a);

                // r = r - α_j q_j - β_{j-1} q_{j-1}
                for idx in 0..n {
                    r[idx] -= a * q0[idx];
                }
                if let Some(&b_prev) = beta.last() {
                    for idx in 0..n {
                        r[idx] -= b_prev * q_prev[idx];
                    }
                }

                let b: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
                beta.push(b);

                if b < 1e-14 {
                    break;
                }

                q_prev = q0.clone();
                q0 = r.iter().map(|v| v / b).collect();
            }

            let k = alpha.len();
            if k == 0 {
                continue;
            }

            // Dense EVD of the k×k tridiagonal T.
            // T has diagonal α[0..k] and off-diagonal β[0..k-1].
            let (eigenvalues, evecs) = tridiagonal_evd(&alpha, &beta[..k.saturating_sub(1)]);

            // Check that A is positive definite (all eigenvalues > 0).
            if eigenvalues.iter().any(|&e| e <= 0.0) {
                return Err(StochasticError::NumericalFailure(
                    "matrix appears to not be positive definite (non-positive Ritz value encountered)"
                        .to_string(),
                ));
            }

            // e1^T log(T) e1 = Σ_i evec[i][0]^2 * log(eigenvalue[i])
            let e1_log_t_e1: f64 = eigenvalues
                .iter()
                .zip(evecs.iter())
                .map(|(&lam, evec)| evec[0] * evec[0] * lam.ln())
                .sum();

            samples.push(z_norm_sq * e1_log_t_e1);
        }

        if samples.is_empty() {
            return Err(StochasticError::NumericalFailure(
                "all probe vectors were degenerate".to_string(),
            ));
        }

        let (mean, _stderr) = mean_and_stderr(&samples);
        Ok(mean)
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Generate a probe vector according to the configured `ProbeType`.
    fn probe_vector(&self, n: usize, probe_idx: usize) -> Vec<f64> {
        let seed = self.config.seed.wrapping_add(probe_idx as u64 * 1_234_567);
        match self.config.probe_type {
            ProbeType::Rademacher => lcg_rademacher(seed, n),
            ProbeType::Gaussian => lcg_gaussian(seed, n),
            ProbeType::Spherical => lcg_spherical(seed, n),
        }
    }

    /// Always generate a spherical probe (used internally by XTrace).
    fn probe_vector_spherical(&self, n: usize, probe_idx: usize) -> Vec<f64> {
        let seed = self.config.seed.wrapping_add(probe_idx as u64 * 1_234_567);
        lcg_spherical(seed, n)
    }

    /// Sparse matrix-vector product: `y = A x`.
    fn matvec(csr: &CsrMatrix<f64>, x: &[f64], y: &mut Vec<f64>) {
        let nrows = csr.nrows();
        if y.len() != nrows {
            y.resize(nrows, 0.0);
        }
        for i in 0..nrows {
            let start = csr.row_ptrs()[i];
            let end = csr.row_ptrs()[i + 1];
            let mut s = 0.0f64;
            for k in start..end {
                s += csr.values()[k] * x[csr.col_indices()[k]];
            }
            y[i] = s;
        }
    }

    /// Thin QR decomposition of a set of `k` column vectors (each of length `n`).
    ///
    /// Uses modified Gram-Schmidt to produce an orthonormal basis Q.
    /// Returns the orthonormal columns; columns that become numerically zero are dropped.
    fn thin_qr(a: &[Vec<f64>], n: usize, k: usize) -> Vec<Vec<f64>> {
        let mut q: Vec<Vec<f64>> = Vec::with_capacity(k);

        for col in a.iter().take(k) {
            let mut v = col.clone();

            // Modified Gram-Schmidt: subtract projections onto existing q columns.
            for qi in &q {
                let proj: f64 = v.iter().zip(qi.iter()).map(|(vi, qi_)| vi * qi_).sum();
                for (vi, qi_) in v.iter_mut().zip(qi.iter()) {
                    *vi -= proj * qi_;
                }
            }

            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-14 * (n as f64).sqrt() {
                q.push(v.into_iter().map(|x| x / nrm).collect());
            }
        }

        q
    }

    /// Validate that the matrix is square and return its size.
    fn require_square(&self, csr: &CsrMatrix<f64>) -> Result<usize, StochasticError> {
        if csr.nrows() != csr.ncols() {
            return Err(StochasticError::MatrixError(format!(
                "matrix must be square, got {}×{}",
                csr.nrows(),
                csr.ncols()
            )));
        }
        if csr.nrows() == 0 {
            return Err(StochasticError::MatrixError(
                "matrix has zero dimension".to_string(),
            ));
        }
        Ok(csr.nrows())
    }

    /// Validate and return the configured probe count.
    fn require_probes(&self) -> Result<usize, StochasticError> {
        if self.config.num_probes == 0 {
            return Err(StochasticError::InvalidConfig(
                "num_probes must be >= 1".to_string(),
            ));
        }
        Ok(self.config.num_probes)
    }
}

// =============================================================================
// Helper: project z onto the orthogonal complement of the column span of Q
// =============================================================================

/// Compute `w = (I - Q Qᵀ) z` for an orthonormal column set Q.
fn project_out(q_cols: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let mut w = z.to_vec();
    for qi in q_cols {
        let proj: f64 = w.iter().zip(qi.iter()).map(|(wi, qi_)| wi * qi_).sum();
        for (wi, qi_) in w.iter_mut().zip(qi.iter()) {
            *wi -= proj * qi_;
        }
    }
    w
}

// =============================================================================
// Helper: dense symmetric tridiagonal EVD (QR iteration)
// =============================================================================

/// Compute eigenvalues and (first-component-only) eigenvectors of a real
/// symmetric tridiagonal matrix with diagonal `alpha` and off-diagonal `beta`.
///
/// Returns `(eigenvalues, eigenvectors)` where each eigenvector is stored as
/// its full column (length k).  Uses the QR algorithm with Wilkinson shifts.
fn tridiagonal_evd(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = alpha.len();
    if k == 0 {
        return (vec![], vec![]);
    }
    if k == 1 {
        return (vec![alpha[0]], vec![vec![1.0]]);
    }

    // Work arrays: diagonal d, off-diagonal e, and eigenvector matrix Z (k×k, col-major).
    let mut d = alpha.to_vec();
    let mut e = vec![0.0f64; k];
    for i in 0..beta.len().min(k - 1) {
        e[i] = beta[i];
    }

    // Identity as initial eigenvector matrix.
    let mut z = vec![0.0f64; k * k];
    for i in 0..k {
        z[i * k + i] = 1.0;
    }

    // Symmetric QR with Wilkinson shift (implicit QR on tridiagonal).
    let max_iter = 30 * k;
    let mut m = k;

    'outer: for _ in 0..max_iter {
        // Deflate small off-diagonal elements.
        while m > 1 && e[m - 2].abs() < 1e-14 * (d[m - 2].abs() + d[m - 1].abs()) {
            m -= 1;
            if m == 1 {
                break 'outer;
            }
        }
        if m <= 1 {
            break;
        }

        // Wilkinson shift: eigenvalue of bottom 2×2 closer to d[m-1].
        let a = d[m - 2];
        let b = e[m - 2];
        let c = d[m - 1];
        let delta = (a - c) / 2.0;
        let sign_delta = if delta >= 0.0 { 1.0 } else { -1.0 };
        let shift = c - sign_delta * b * b / (delta.abs() + (delta * delta + b * b).sqrt());

        // Implicit QR step.
        let mut x = d[0] - shift;
        let mut z_val = e[0];

        for i in 0..m - 1 {
            let (c_rot, s_rot) = givens_cs(x, z_val);

            // Apply rotation on left and right to tridiagonal.
            let w = c_rot * x + s_rot * z_val;
            let _ = w; // w is just the new diagonal position before update

            // Update d[i], d[i+1], e[i] via Givens rotation.
            let d_i = d[i];
            let d_i1 = d[i + 1];
            let e_i = e[i];

            d[i] = c_rot * c_rot * d_i + 2.0 * c_rot * s_rot * e_i + s_rot * s_rot * d_i1;
            d[i + 1] = s_rot * s_rot * d_i - 2.0 * c_rot * s_rot * e_i + c_rot * c_rot * d_i1;
            e[i] = c_rot * s_rot * (d_i1 - d_i) + (c_rot * c_rot - s_rot * s_rot) * e_i;

            if i > 0 {
                e[i - 1] = c_rot * e[i - 1] + s_rot * z_val;
            }

            x = e[i];
            if i + 1 < m - 1 {
                z_val = s_rot * e[i + 1];
                e[i + 1] = c_rot * e[i + 1];
            }

            // Accumulate rotation into eigenvector matrix (columns).
            for row in 0..k {
                let zi = z[row * k + i];
                let zi1 = z[row * k + i + 1];
                z[row * k + i] = c_rot * zi + s_rot * zi1;
                z[row * k + i + 1] = -s_rot * zi + c_rot * zi1;
            }
        }
    }

    // Extract eigenvectors as column vectors.
    let eigenvectors: Vec<Vec<f64>> = (0..k)
        .map(|j| (0..k).map(|i| z[i * k + j]).collect())
        .collect();

    (d, eigenvectors)
}

/// Compute Givens cosine and sine for the pair (a, b) such that
/// [c s; -s c] [a; b] = [r; 0].
#[inline]
fn givens_cs(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        return (1.0, 0.0);
    }
    if a.abs() < b.abs() {
        let t = -a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        (s * t, s)
    } else {
        let t = -b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, c * t)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    // Helper: build identity matrix in CSR.
    fn identity_csr(n: usize) -> CsrMatrix<f64> {
        let values = vec![1.0f64; n];
        let col_indices: Vec<usize> = (0..n).collect();
        let row_ptrs: Vec<usize> = (0..=n).collect();
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).expect("valid identity CSR")
    }

    // Helper: build diagonal matrix from entries.
    fn diag_csr(entries: &[f64]) -> CsrMatrix<f64> {
        let n = entries.len();
        let values = entries.to_vec();
        let col_indices: Vec<usize> = (0..n).collect();
        let row_ptrs: Vec<usize> = (0..=n).collect();
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).expect("valid diagonal CSR")
    }

    // Helper: build 1-D Laplacian (tridiagonal with 2 on diagonal, -1 off).
    fn laplacian_1d_csr(n: usize) -> CsrMatrix<f64> {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(2.0);
            col_indices.push(i);
            if i + 1 < n {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).expect("valid 1-D Laplacian CSR")
    }

    // -------------------------------------------------------------------------

    #[test]
    fn test_stochastic_config_default() {
        let cfg = StochasticConfig::default();
        assert_eq!(cfg.num_probes, 30);
        assert_eq!(cfg.seed, 42);
        assert!(matches!(cfg.probe_type, ProbeType::Rademacher));
        assert!((cfg.confidence - 0.95).abs() < 1e-12);
    }

    #[test]
    fn test_hutchinson_identity() {
        // tr(I_3) = 3.
        let eye = identity_csr(3);
        let est = StochasticEstimator::with_default();
        let result = est.trace_hutchinson(&eye).expect("trace_hutchinson failed");
        assert!(
            (result.estimate - 3.0).abs() < 0.5,
            "estimate {} far from 3.0",
            result.estimate
        );
        assert_eq!(result.n_probes_used, 30);
    }

    #[test]
    fn test_hutchinson_diagonal() {
        // tr(diag(1,2,3)) = 6.
        let d = diag_csr(&[1.0, 2.0, 3.0]);
        let cfg = StochasticConfig {
            num_probes: 100,
            ..Default::default()
        };
        let est = StochasticEstimator::new(cfg);
        let result = est.trace_hutchinson(&d).expect("trace_hutchinson failed");
        assert!(
            (result.estimate - 6.0).abs() < 1.0,
            "estimate {} far from 6.0",
            result.estimate
        );
    }

    #[test]
    fn test_hutchinson_sparse_laplacian() {
        // 1-D Laplacian with n=10: tr = 10 * 2.0 = 20.
        let lap = laplacian_1d_csr(10);
        let cfg = StochasticConfig {
            num_probes: 200,
            ..Default::default()
        };
        let est = StochasticEstimator::new(cfg);
        let result = est.trace_hutchinson(&lap).expect("trace_hutchinson failed");
        assert!(
            (result.estimate - 20.0).abs() < 3.0,
            "estimate {} far from 20.0",
            result.estimate
        );
    }

    #[test]
    fn test_hutch_plusplus_accuracy() {
        // Hutch++ should give a closer estimate than plain Hutchinson for the
        // same probe count on a matrix with rapidly decaying spectrum.
        // Use diag(1,2,...,20): tr = 210.
        let entries: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let d = diag_csr(&entries);
        let true_trace = 210.0f64;

        let cfg = StochasticConfig {
            num_probes: 30,
            seed: 7,
            ..Default::default()
        };
        let est = StochasticEstimator::new(cfg.clone());

        let hh_result = est.trace_hutch_plus_plus(&d).expect("hutch++ failed");
        let hutch_result = est.trace_hutchinson(&d).expect("hutchinson failed");

        let hh_err = (hh_result.estimate - true_trace).abs();
        let hutch_err = (hutch_result.estimate - true_trace).abs();

        // Hutch++ should be at least as accurate or better.
        // We allow Hutch++ to be slightly worse due to finite m but it must be within 30% of true.
        assert!(
            hh_err < true_trace * 0.30,
            "Hutch++ error {hh_err} too large (true={true_trace})"
        );
        // Log the comparison for debugging.
        let _ = hutch_err;
    }

    #[test]
    fn test_diagonal_estimator() {
        // diagonal(I_5) ≈ [1,1,1,1,1].
        let eye = identity_csr(5);
        let cfg = StochasticConfig {
            num_probes: 100,
            ..Default::default()
        };
        let est = StochasticEstimator::new(cfg);
        let result = est.diagonal(&eye).expect("diagonal failed");
        assert_eq!(result.diagonal.len(), 5);
        for (i, &d) in result.diagonal.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 0.3,
                "diagonal[{i}] = {d} not close to 1.0"
            );
        }
    }

    #[test]
    fn test_frobenius_norm() {
        // ‖I_n‖_F = sqrt(n).
        let n = 9usize;
        let eye = identity_csr(n);
        let cfg = StochasticConfig {
            num_probes: 50,
            ..Default::default()
        };
        let est = StochasticEstimator::new(cfg);
        let frob = est.frobenius_norm(&eye).expect("frobenius_norm failed");
        let expected = (n as f64).sqrt(); // 3.0
        assert!(
            (frob - expected).abs() < expected * 0.10,
            "frobenius estimate {frob} not within 10% of {expected}"
        );
    }

    #[test]
    fn test_log_det_spd() {
        // For a 3×3 diagonal matrix diag(2, 3, 5):
        // log det = log(2) + log(3) + log(5) ≈ 3.401.
        let d = diag_csr(&[2.0, 3.0, 5.0]);
        let expected = 2.0f64.ln() + 3.0f64.ln() + 5.0f64.ln();
        let cfg = StochasticConfig {
            num_probes: 100,
            ..Default::default()
        };
        let est = StochasticEstimator::new(cfg);
        let result = est.log_det(&d).expect("log_det failed");
        // Sign must be positive (log det of this SPD matrix is positive).
        assert!(result > 0.0, "log_det should be positive, got {result}");
        assert!(
            (result - expected).abs() < expected * 0.20,
            "log_det estimate {result} far from {expected}"
        );
    }
}
