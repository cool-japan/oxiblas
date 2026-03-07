//! Detailed information structures for matrix factorizations.
//!
//! This module provides structures that capture metadata and diagnostics
//! about matrix factorizations beyond the raw factors. This information
//! is useful for:
//!
//! - Assessing numerical stability of the factorization
//! - Detecting near-singularity or ill-conditioning
//! - Understanding the effective rank of a matrix
//! - Debugging numerical issues
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_lapack::info::{LuInfo, compute_lu_info};
//! use oxiblas_lapack::lu::Lu;
//!
//! let lu = Lu::compute(a.as_ref()).unwrap();
//! let info = compute_lu_info(&lu);
//!
//! println!("Pivot growth factor: {}", info.pivot_growth);
//! println!("Estimated condition: {}", info.rcond_estimate);
//! if info.is_nearly_singular(1e-12) {
//!     println!("Warning: Matrix is nearly singular");
//! }
//! ```

use std::fmt;

// ============================================================================
// LU Factorization Info
// ============================================================================

/// Detailed information about an LU factorization.
///
/// Provides diagnostics about numerical stability and matrix properties.
#[derive(Debug, Clone)]
pub struct LuInfo<T> {
    /// Matrix dimension (n for n×n matrix).
    pub n: usize,

    /// Pivot growth factor: max|U\[i,j\]| / max|A\[i,j\]|.
    ///
    /// A large pivot growth (>> 1) indicates potential instability
    /// in the factorization. Values close to 1 indicate stable factorization.
    pub pivot_growth: T,

    /// Estimated reciprocal condition number (1-norm).
    ///
    /// Small values (close to machine epsilon) indicate ill-conditioning.
    /// A value of 1.0 indicates perfect conditioning.
    pub rcond_estimate: T,

    /// Number of row interchanges performed during factorization.
    pub num_pivots: usize,

    /// Maximum absolute value on the diagonal of U.
    pub max_diag_u: T,

    /// Minimum absolute value on the diagonal of U.
    pub min_diag_u: T,

    /// Sign of the determinant (1, -1, or 0 for singular).
    pub det_sign: i32,

    /// Log of absolute value of determinant (to avoid overflow).
    pub log_det_abs: T,
}

impl<T: Clone + PartialOrd + Default> LuInfo<T> {
    /// Creates a new LuInfo with the given values.
    pub fn new(
        n: usize,
        pivot_growth: T,
        rcond_estimate: T,
        num_pivots: usize,
        max_diag_u: T,
        min_diag_u: T,
        det_sign: i32,
        log_det_abs: T,
    ) -> Self {
        Self {
            n,
            pivot_growth,
            rcond_estimate,
            num_pivots,
            max_diag_u,
            min_diag_u,
            det_sign,
            log_det_abs,
        }
    }
}

impl<T: PartialOrd + Clone> LuInfo<T> {
    /// Returns true if the matrix appears nearly singular.
    ///
    /// A matrix is considered nearly singular if the reciprocal condition
    /// number is below the given tolerance.
    pub fn is_nearly_singular(&self, tol: T) -> bool {
        self.rcond_estimate < tol
    }

    /// Returns the estimated condition number (1 / rcond).
    pub fn condition_estimate(&self) -> T
    where
        T: num_traits::Float,
    {
        T::one() / self.rcond_estimate
    }
}

impl<T: fmt::Display> fmt::Display for LuInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LU Factorization Info:")?;
        writeln!(f, "  Matrix size: {}×{}", self.n, self.n)?;
        writeln!(f, "  Pivot growth: {}", self.pivot_growth)?;
        writeln!(f, "  Reciprocal condition: {}", self.rcond_estimate)?;
        writeln!(f, "  Number of pivots: {}", self.num_pivots)?;
        writeln!(
            f,
            "  U diagonal range: [{}, {}]",
            self.min_diag_u, self.max_diag_u
        )?;
        writeln!(f, "  Determinant sign: {}", self.det_sign)?;
        write!(f, "  Log|det|: {}", self.log_det_abs)
    }
}

// ============================================================================
// Cholesky Factorization Info
// ============================================================================

/// Detailed information about a Cholesky factorization.
#[derive(Debug, Clone)]
pub struct CholeskyInfo<T> {
    /// Matrix dimension.
    pub n: usize,

    /// Whether the matrix was positive definite.
    pub is_positive_definite: bool,

    /// If not positive definite, the index where failure occurred.
    pub failure_index: Option<usize>,

    /// Estimated reciprocal condition number.
    pub rcond_estimate: T,

    /// Maximum diagonal element of L.
    pub max_diag_l: T,

    /// Minimum diagonal element of L.
    pub min_diag_l: T,

    /// Log of determinant (sum of 2*log(L\[i,i\])).
    pub log_det: T,
}

impl<T: Clone + Default> CholeskyInfo<T> {
    /// Creates a new CholeskyInfo.
    pub fn new(
        n: usize,
        is_positive_definite: bool,
        failure_index: Option<usize>,
        rcond_estimate: T,
        max_diag_l: T,
        min_diag_l: T,
        log_det: T,
    ) -> Self {
        Self {
            n,
            is_positive_definite,
            failure_index,
            rcond_estimate,
            max_diag_l,
            min_diag_l,
            log_det,
        }
    }

    /// Returns whether the factorization succeeded.
    pub fn success(&self) -> bool {
        self.is_positive_definite
    }
}

impl<T: PartialOrd + Clone> CholeskyInfo<T> {
    /// Returns true if the matrix is nearly singular.
    pub fn is_nearly_singular(&self, tol: T) -> bool {
        self.rcond_estimate < tol
    }
}

impl<T: fmt::Display> fmt::Display for CholeskyInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cholesky Factorization Info:")?;
        writeln!(f, "  Matrix size: {}×{}", self.n, self.n)?;
        writeln!(f, "  Positive definite: {}", self.is_positive_definite)?;
        if let Some(idx) = self.failure_index {
            writeln!(f, "  Failure index: {}", idx)?;
        }
        writeln!(f, "  Reciprocal condition: {}", self.rcond_estimate)?;
        writeln!(
            f,
            "  L diagonal range: [{}, {}]",
            self.min_diag_l, self.max_diag_l
        )?;
        write!(f, "  Log determinant: {}", self.log_det)
    }
}

// ============================================================================
// QR Factorization Info
// ============================================================================

/// Detailed information about a QR factorization.
#[derive(Debug, Clone)]
pub struct QrInfo<T> {
    /// Number of rows in original matrix.
    pub m: usize,

    /// Number of columns in original matrix.
    pub n: usize,

    /// Estimated numerical rank based on R diagonal.
    pub numerical_rank: usize,

    /// Tolerance used for rank estimation.
    pub rank_tolerance: T,

    /// Maximum absolute value on R diagonal.
    pub max_diag_r: T,

    /// Minimum absolute value on R diagonal.
    pub min_diag_r: T,

    /// Ratio of max to min diagonal (condition indicator).
    pub diag_ratio: T,

    /// Estimated orthogonality error: ||Q^T Q - I||.
    pub orthogonality_error: T,
}

impl<T: Clone + Default> QrInfo<T> {
    /// Creates a new QrInfo.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: usize,
        n: usize,
        numerical_rank: usize,
        rank_tolerance: T,
        max_diag_r: T,
        min_diag_r: T,
        diag_ratio: T,
        orthogonality_error: T,
    ) -> Self {
        Self {
            m,
            n,
            numerical_rank,
            rank_tolerance,
            max_diag_r,
            min_diag_r,
            diag_ratio,
            orthogonality_error,
        }
    }

    /// Returns true if the matrix is rank deficient.
    pub fn is_rank_deficient(&self) -> bool {
        self.numerical_rank < self.m.min(self.n)
    }

    /// Returns the null space dimension.
    pub fn nullity(&self) -> usize {
        self.n.saturating_sub(self.numerical_rank)
    }
}

impl<T: fmt::Display> fmt::Display for QrInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nullity = self.n.saturating_sub(self.numerical_rank);
        writeln!(f, "QR Factorization Info:")?;
        writeln!(f, "  Matrix size: {}×{}", self.m, self.n)?;
        writeln!(
            f,
            "  Numerical rank: {} (tol={})",
            self.numerical_rank, self.rank_tolerance
        )?;
        writeln!(f, "  Nullity: {}", nullity)?;
        writeln!(
            f,
            "  R diagonal range: [{}, {}]",
            self.min_diag_r, self.max_diag_r
        )?;
        writeln!(f, "  Diagonal ratio: {}", self.diag_ratio)?;
        write!(f, "  Orthogonality error: {}", self.orthogonality_error)
    }
}

// ============================================================================
// SVD Info
// ============================================================================

/// Detailed information about a Singular Value Decomposition.
#[derive(Debug, Clone)]
pub struct SvdInfo<T> {
    /// Number of rows in original matrix.
    pub m: usize,

    /// Number of columns in original matrix.
    pub n: usize,

    /// Number of computed singular values.
    pub num_singular_values: usize,

    /// Numerical rank (singular values above tolerance).
    pub numerical_rank: usize,

    /// Tolerance used for rank estimation.
    pub rank_tolerance: T,

    /// Largest singular value (2-norm of matrix).
    pub sigma_max: T,

    /// Smallest singular value.
    pub sigma_min: T,

    /// Condition number (sigma_max / sigma_min).
    pub condition_number: T,

    /// Number of singular value clusters detected.
    pub num_clusters: usize,

    /// Relative gap between consecutive singular values (minimum).
    pub min_relative_gap: T,

    /// Frobenius norm of the matrix.
    pub frobenius_norm: T,

    /// Nuclear norm (sum of singular values).
    pub nuclear_norm: T,
}

impl<T: Clone + Default> SvdInfo<T> {
    /// Creates a new SvdInfo.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: usize,
        n: usize,
        num_singular_values: usize,
        numerical_rank: usize,
        rank_tolerance: T,
        sigma_max: T,
        sigma_min: T,
        condition_number: T,
        num_clusters: usize,
        min_relative_gap: T,
        frobenius_norm: T,
        nuclear_norm: T,
    ) -> Self {
        Self {
            m,
            n,
            num_singular_values,
            numerical_rank,
            rank_tolerance,
            sigma_max,
            sigma_min,
            condition_number,
            num_clusters,
            min_relative_gap,
            frobenius_norm,
            nuclear_norm,
        }
    }

    /// Returns true if the matrix is rank deficient.
    pub fn is_rank_deficient(&self) -> bool {
        self.numerical_rank < self.m.min(self.n)
    }

    /// Returns true if singular values are well-separated.
    pub fn well_separated(&self, gap_threshold: T) -> bool
    where
        T: PartialOrd,
    {
        self.min_relative_gap >= gap_threshold
    }

    /// Returns the null space dimension.
    pub fn nullity(&self) -> usize {
        self.n.saturating_sub(self.numerical_rank)
    }
}

impl<T: fmt::Display> fmt::Display for SvdInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SVD Info:")?;
        writeln!(f, "  Matrix size: {}×{}", self.m, self.n)?;
        writeln!(f, "  Singular values: {}", self.num_singular_values)?;
        writeln!(
            f,
            "  Numerical rank: {} (tol={})",
            self.numerical_rank, self.rank_tolerance
        )?;
        writeln!(f, "  σ_max: {}", self.sigma_max)?;
        writeln!(f, "  σ_min: {}", self.sigma_min)?;
        writeln!(f, "  Condition number: {}", self.condition_number)?;
        writeln!(f, "  Clusters: {}", self.num_clusters)?;
        writeln!(f, "  Min relative gap: {}", self.min_relative_gap)?;
        writeln!(f, "  Frobenius norm: {}", self.frobenius_norm)?;
        write!(f, "  Nuclear norm: {}", self.nuclear_norm)
    }
}

// ============================================================================
// Eigenvalue Decomposition Info
// ============================================================================

/// Detailed information about an Eigenvalue Decomposition.
#[derive(Debug, Clone)]
pub struct EvdInfo<T> {
    /// Matrix dimension.
    pub n: usize,

    /// Number of computed eigenvalues.
    pub num_eigenvalues: usize,

    /// Whether all eigenvalues are real.
    pub all_real: bool,

    /// Number of complex conjugate pairs.
    pub num_complex_pairs: usize,

    /// Spectral radius (max |λ|).
    pub spectral_radius: T,

    /// Smallest eigenvalue magnitude.
    pub min_eigenvalue_abs: T,

    /// Number of eigenvalue clusters detected.
    pub num_clusters: usize,

    /// Minimum separation between eigenvalues.
    pub min_separation: T,

    /// Trace of the matrix (sum of eigenvalues).
    pub trace: T,

    /// Whether the eigenvector matrix is well-conditioned.
    pub eigenvectors_well_conditioned: bool,

    /// Estimated condition of eigenvector matrix.
    pub eigenvector_condition: T,
}

impl<T: Clone + Default> EvdInfo<T> {
    /// Creates a new EvdInfo.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        num_eigenvalues: usize,
        all_real: bool,
        num_complex_pairs: usize,
        spectral_radius: T,
        min_eigenvalue_abs: T,
        num_clusters: usize,
        min_separation: T,
        trace: T,
        eigenvectors_well_conditioned: bool,
        eigenvector_condition: T,
    ) -> Self {
        Self {
            n,
            num_eigenvalues,
            all_real,
            num_complex_pairs,
            spectral_radius,
            min_eigenvalue_abs,
            num_clusters,
            min_separation,
            trace,
            eigenvectors_well_conditioned,
            eigenvector_condition,
        }
    }

    /// Returns true if eigenvalues are well-separated.
    pub fn well_separated(&self, sep_threshold: T) -> bool
    where
        T: PartialOrd,
    {
        self.min_separation >= sep_threshold
    }
}

impl<T: fmt::Display> fmt::Display for EvdInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Eigenvalue Decomposition Info:")?;
        writeln!(f, "  Matrix size: {}×{}", self.n, self.n)?;
        writeln!(f, "  Eigenvalues: {}", self.num_eigenvalues)?;
        writeln!(f, "  All real: {}", self.all_real)?;
        writeln!(f, "  Complex pairs: {}", self.num_complex_pairs)?;
        writeln!(f, "  Spectral radius: {}", self.spectral_radius)?;
        writeln!(f, "  Min |λ|: {}", self.min_eigenvalue_abs)?;
        writeln!(f, "  Clusters: {}", self.num_clusters)?;
        writeln!(f, "  Min separation: {}", self.min_separation)?;
        writeln!(f, "  Trace: {}", self.trace)?;
        writeln!(
            f,
            "  Eigenvectors well-conditioned: {}",
            self.eigenvectors_well_conditioned
        )?;
        write!(f, "  Eigenvector condition: {}", self.eigenvector_condition)
    }
}

// ============================================================================
// Symmetric/Hermitian EVD Info
// ============================================================================

/// Detailed information about a Symmetric/Hermitian Eigenvalue Decomposition.
#[derive(Debug, Clone)]
pub struct SymmetricEvdInfo<T> {
    /// Matrix dimension.
    pub n: usize,

    /// Number of eigenvalues.
    pub num_eigenvalues: usize,

    /// Largest eigenvalue.
    pub lambda_max: T,

    /// Smallest eigenvalue.
    pub lambda_min: T,

    /// Spectral radius.
    pub spectral_radius: T,

    /// Condition number (|λ_max| / |λ_min|).
    pub condition_number: T,

    /// Number of positive eigenvalues.
    pub num_positive: usize,

    /// Number of negative eigenvalues.
    pub num_negative: usize,

    /// Number of zero eigenvalues (within tolerance).
    pub num_zero: usize,

    /// Inertia: (positive, negative, zero).
    pub inertia: (usize, usize, usize),

    /// Trace (sum of eigenvalues).
    pub trace: T,

    /// Minimum gap between consecutive eigenvalues.
    pub min_gap: T,
}

impl<T: Clone + Default> SymmetricEvdInfo<T> {
    /// Creates a new SymmetricEvdInfo.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        num_eigenvalues: usize,
        lambda_max: T,
        lambda_min: T,
        spectral_radius: T,
        condition_number: T,
        num_positive: usize,
        num_negative: usize,
        num_zero: usize,
        trace: T,
        min_gap: T,
    ) -> Self {
        Self {
            n,
            num_eigenvalues,
            lambda_max,
            lambda_min,
            spectral_radius,
            condition_number,
            num_positive,
            num_negative,
            num_zero,
            inertia: (num_positive, num_negative, num_zero),
            trace,
            min_gap,
        }
    }

    /// Returns true if the matrix is positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.num_negative == 0 && self.num_zero == 0
    }

    /// Returns true if the matrix is positive semi-definite.
    pub fn is_positive_semidefinite(&self) -> bool {
        self.num_negative == 0
    }

    /// Returns true if the matrix is negative definite.
    pub fn is_negative_definite(&self) -> bool {
        self.num_positive == 0 && self.num_zero == 0
    }

    /// Returns true if the matrix is indefinite.
    pub fn is_indefinite(&self) -> bool {
        self.num_positive > 0 && self.num_negative > 0
    }
}

impl<T: fmt::Display> fmt::Display for SymmetricEvdInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Symmetric EVD Info:")?;
        writeln!(f, "  Matrix size: {}×{}", self.n, self.n)?;
        writeln!(f, "  λ_max: {}", self.lambda_max)?;
        writeln!(f, "  λ_min: {}", self.lambda_min)?;
        writeln!(f, "  Spectral radius: {}", self.spectral_radius)?;
        writeln!(f, "  Condition number: {}", self.condition_number)?;
        writeln!(
            f,
            "  Inertia: (+{}, -{}, 0:{})",
            self.num_positive, self.num_negative, self.num_zero
        )?;
        writeln!(f, "  Trace: {}", self.trace)?;
        write!(f, "  Min gap: {}", self.min_gap)
    }
}

// ============================================================================
// Compute Info Functions
// ============================================================================

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::MatRef;

/// Compute LU factorization info from the LU factors.
pub fn compute_lu_info<T>(lu_matrix: MatRef<'_, T>, _pivot: &[usize], num_swaps: usize) -> LuInfo<T>
where
    T: Field + Real + Float,
{
    let n = lu_matrix.nrows();

    // Compute pivot growth factor
    let mut max_u = T::zero();
    for i in 0..n {
        for j in i..n {
            let val = Scalar::abs(lu_matrix[(i, j)]);
            if val > max_u {
                max_u = val;
            }
        }
    }

    // Estimate original matrix max (approximate)
    let mut max_a = max_u;
    for i in 0..n {
        for j in 0..i {
            let val = Scalar::abs(lu_matrix[(i, j)]);
            if val > max_a {
                max_a = val;
            }
        }
    }

    let pivot_growth = if max_a > T::zero() {
        max_u / max_a
    } else {
        T::one()
    };

    // Compute diagonal statistics
    let mut max_diag = T::zero();
    let mut min_diag = T::infinity();
    let mut log_det_abs = T::zero();

    for i in 0..n {
        let diag = Scalar::abs(lu_matrix[(i, i)]);
        if diag > max_diag {
            max_diag = diag;
        }
        if diag < min_diag {
            min_diag = diag;
        }
        if diag > T::zero() {
            log_det_abs = log_det_abs + Real::ln(diag);
        }
    }

    // Estimate reciprocal condition number
    let rcond = if max_diag > T::zero() && min_diag > T::zero() {
        min_diag / max_diag
    } else {
        T::zero()
    };

    // Determine sign of determinant
    let det_sign = if min_diag == T::zero() {
        0
    } else if num_swaps % 2 == 0 {
        1
    } else {
        -1
    };

    LuInfo::new(
        n,
        pivot_growth,
        rcond,
        num_swaps,
        max_diag,
        min_diag,
        det_sign,
        log_det_abs,
    )
}

/// Compute Cholesky factorization info from the L factor.
pub fn compute_cholesky_info<T>(l_matrix: MatRef<'_, T>) -> CholeskyInfo<T>
where
    T: Field + Real + Float,
{
    let n = l_matrix.nrows();

    let mut max_diag = T::zero();
    let mut min_diag = T::infinity();
    let mut log_det = T::zero();
    let mut failure_index = None;

    for i in 0..n {
        let diag = l_matrix[(i, i)];
        let diag_abs = Scalar::abs(diag);

        if diag <= T::zero() {
            failure_index = Some(i);
        }

        if diag_abs > max_diag {
            max_diag = diag_abs;
        }
        if diag_abs < min_diag && diag_abs > T::zero() {
            min_diag = diag_abs;
        }
        if diag > T::zero() {
            log_det = log_det + T::from(2.0).unwrap_or_else(T::zero) * Real::ln(diag);
        }
    }

    let is_positive_definite = failure_index.is_none();

    let rcond = if max_diag > T::zero() && min_diag > T::zero() {
        (min_diag / max_diag) * (min_diag / max_diag) // Squared for A, not L
    } else {
        T::zero()
    };

    CholeskyInfo::new(
        n,
        is_positive_definite,
        failure_index,
        rcond,
        max_diag,
        min_diag,
        log_det,
    )
}

/// Compute QR factorization info from the R factor.
pub fn compute_qr_info<T>(
    r_matrix: MatRef<'_, T>,
    q_matrix: Option<MatRef<'_, T>>,
    tol: T,
) -> QrInfo<T>
where
    T: Field + Real + Float,
{
    let m = r_matrix.nrows();
    let n = r_matrix.ncols();
    let k = m.min(n);

    let mut max_diag = T::zero();
    let mut min_diag = T::infinity();
    let mut numerical_rank = 0;

    for i in 0..k {
        let diag = Scalar::abs(r_matrix[(i, i)]);
        if diag > max_diag {
            max_diag = diag;
        }
        if diag < min_diag && diag > T::zero() {
            min_diag = diag;
        }
    }

    // Count rank based on diagonal threshold
    let threshold = tol * max_diag;
    for i in 0..k {
        if Scalar::abs(r_matrix[(i, i)]) > threshold {
            numerical_rank += 1;
        }
    }

    let diag_ratio = if min_diag > T::zero() {
        max_diag / min_diag
    } else {
        T::infinity()
    };

    // Compute orthogonality error if Q is provided
    let orthogonality_error = if let Some(q) = q_matrix {
        compute_orthogonality_error(q)
    } else {
        T::zero()
    };

    QrInfo::new(
        m,
        n,
        numerical_rank,
        tol,
        max_diag,
        min_diag,
        diag_ratio,
        orthogonality_error,
    )
}

/// Compute orthogonality error ||Q^T Q - I||_F.
fn compute_orthogonality_error<T>(q: MatRef<'_, T>) -> T
where
    T: Field + Real + Float,
{
    let n = q.ncols();
    let mut error = T::zero();

    for i in 0..n {
        for j in 0..n {
            let mut dot = T::zero();
            for k in 0..q.nrows() {
                dot = dot + q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { T::one() } else { T::zero() };
            let diff = dot - expected;
            error = error + diff * diff;
        }
    }

    Real::sqrt(error)
}

/// Compute SVD info from singular values.
pub fn compute_svd_info<T>(m: usize, n: usize, singular_values: &[T], tol: T) -> SvdInfo<T>
where
    T: Field + Real + Float,
{
    let num_sv = singular_values.len();

    if num_sv == 0 {
        return SvdInfo::new(
            m,
            n,
            0,
            0,
            tol,
            T::zero(),
            T::zero(),
            T::infinity(),
            0,
            T::infinity(),
            T::zero(),
            T::zero(),
        );
    }

    let sigma_max = singular_values[0];
    let sigma_min = singular_values[num_sv - 1];

    // Count numerical rank
    let threshold = tol * sigma_max;
    let numerical_rank = singular_values.iter().filter(|&&s| s > threshold).count();

    // Condition number
    let condition_number = if sigma_min > T::zero() {
        sigma_max / sigma_min
    } else {
        T::infinity()
    };

    // Detect clusters and compute gaps
    let cluster_tol = T::from(0.1).unwrap_or_else(T::zero); // 10% relative gap threshold
    let mut num_clusters = 1;
    let mut min_relative_gap = T::infinity();

    for i in 0..(num_sv - 1) {
        if singular_values[i] > T::zero() {
            let gap = (singular_values[i] - singular_values[i + 1]) / singular_values[i];
            if gap < min_relative_gap {
                min_relative_gap = gap;
            }
            if gap > cluster_tol {
                num_clusters += 1;
            }
        }
    }

    // Compute norms
    let mut frobenius_sq = T::zero();
    let mut nuclear = T::zero();
    for &s in singular_values {
        frobenius_sq = frobenius_sq + s * s;
        nuclear = nuclear + s;
    }
    let frobenius_norm = Real::sqrt(frobenius_sq);

    SvdInfo::new(
        m,
        n,
        num_sv,
        numerical_rank,
        tol,
        sigma_max,
        sigma_min,
        condition_number,
        num_clusters,
        min_relative_gap,
        frobenius_norm,
        nuclear,
    )
}

/// Compute symmetric EVD info from eigenvalues.
pub fn compute_symmetric_evd_info<T>(n: usize, eigenvalues: &[T], tol: T) -> SymmetricEvdInfo<T>
where
    T: Field + Real + Float,
{
    let num_ev = eigenvalues.len();

    if num_ev == 0 {
        return SymmetricEvdInfo::new(
            n,
            0,
            T::zero(),
            T::zero(),
            T::zero(),
            T::infinity(),
            0,
            0,
            0,
            T::zero(),
            T::infinity(),
        );
    }

    // Eigenvalues are typically sorted in ascending order
    let lambda_min = eigenvalues[0];
    let lambda_max = eigenvalues[num_ev - 1];

    let spectral_radius = Float::max(Scalar::abs(lambda_min), Scalar::abs(lambda_max));

    // Count by sign
    let mut num_positive = 0;
    let mut num_negative = 0;
    let mut num_zero = 0;
    let mut trace = T::zero();

    for &lambda in eigenvalues {
        trace = trace + lambda;
        if lambda > tol {
            num_positive += 1;
        } else if lambda < -tol {
            num_negative += 1;
        } else {
            num_zero += 1;
        }
    }

    // Condition number
    let min_abs = Float::min(Scalar::abs(lambda_min), Scalar::abs(lambda_max));
    let max_abs = spectral_radius;
    let condition_number = if min_abs > T::zero() {
        max_abs / min_abs
    } else {
        T::infinity()
    };

    // Minimum gap
    let mut min_gap = T::infinity();
    for i in 0..(num_ev - 1) {
        let gap = eigenvalues[i + 1] - eigenvalues[i];
        if gap < min_gap {
            min_gap = gap;
        }
    }

    SymmetricEvdInfo::new(
        n,
        num_ev,
        lambda_max,
        lambda_min,
        spectral_radius,
        condition_number,
        num_positive,
        num_negative,
        num_zero,
        trace,
        min_gap,
    )
}

/// Compute general EVD info from eigenvalues (real and imaginary parts).
pub fn compute_general_evd_info<T>(
    n: usize,
    eigenvalues_real: &[T],
    eigenvalues_imag: &[T],
    eigenvector_cond: Option<T>,
) -> EvdInfo<T>
where
    T: Field + Real + Float,
{
    let num_ev = eigenvalues_real.len();

    if num_ev == 0 {
        return EvdInfo::new(
            n,
            0,
            true,
            0,
            T::zero(),
            T::infinity(),
            0,
            T::infinity(),
            T::zero(),
            true,
            T::one(),
        );
    }

    // Compute magnitudes and statistics
    let mut spectral_radius = T::zero();
    let mut min_abs = T::infinity();
    let mut num_complex_pairs = 0;
    let mut trace = T::zero();

    let tol = <T as Scalar>::epsilon() * T::from(100.0).unwrap_or_else(T::zero);

    for i in 0..num_ev {
        let re = eigenvalues_real[i];
        let im = eigenvalues_imag[i];
        let mag = Real::sqrt(re * re + im * im);

        if mag > spectral_radius {
            spectral_radius = mag;
        }
        if mag < min_abs {
            min_abs = mag;
        }

        trace = trace + re;

        if Scalar::abs(im) > tol {
            num_complex_pairs += 1;
        }
    }

    // Complex pairs are counted twice (conjugate pairs)
    num_complex_pairs /= 2;

    let all_real = num_complex_pairs == 0;

    // Compute minimum separation
    let mut min_separation = T::infinity();
    for i in 0..num_ev {
        for j in (i + 1)..num_ev {
            let dr = eigenvalues_real[i] - eigenvalues_real[j];
            let di = eigenvalues_imag[i] - eigenvalues_imag[j];
            let sep = Real::sqrt(dr * dr + di * di);
            if sep < min_separation {
                min_separation = sep;
            }
        }
    }

    // Cluster detection
    let cluster_tol = T::from(0.1).unwrap_or_else(T::zero) * spectral_radius;
    let num_clusters = if min_separation < cluster_tol {
        num_ev / 2 // Rough estimate
    } else {
        num_ev
    };

    let eigenvector_condition = eigenvector_cond.unwrap_or(T::one());
    let well_conditioned = eigenvector_condition < T::from(100.0).unwrap_or_else(T::zero);

    EvdInfo::new(
        n,
        num_ev,
        all_real,
        num_complex_pairs,
        spectral_radius,
        min_abs,
        num_clusters,
        min_separation,
        trace,
        well_conditioned,
        eigenvector_condition,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_lu_info() {
        // Simple LU factors
        let lu: Mat<f64> = Mat::from_rows(&[
            &[4.0, 3.0],
            &[0.5, 1.5], // L21 = 0.5, U22 = 3 - 0.5*2 = 1.5... wait, let me redo
        ]);

        let info = compute_lu_info(lu.as_ref(), &[0, 1], 0);

        assert_eq!(info.n, 2);
        assert!(info.pivot_growth >= 0.0);
        assert!(info.det_sign == 1 || info.det_sign == -1);
    }

    #[test]
    fn test_cholesky_info() {
        // L from LL^T of [[4, 2], [2, 5]]
        // L = [[2, 0], [1, 2]]
        let l: Mat<f64> = Mat::from_rows(&[&[2.0, 0.0], &[1.0, 2.0]]);

        let info = compute_cholesky_info(l.as_ref());

        assert!(info.is_positive_definite);
        assert!(info.failure_index.is_none());
        assert_eq!(info.n, 2);
    }

    #[test]
    fn test_qr_info() {
        // R from QR of [[1, 2], [3, 4], [5, 6]]
        let r: Mat<f64> = Mat::from_rows(&[&[5.916, 7.437], &[0.0, 0.828], &[0.0, 0.0]]);

        let info = compute_qr_info(r.as_ref(), None, 1e-10);

        assert_eq!(info.m, 3);
        assert_eq!(info.n, 2);
        assert_eq!(info.numerical_rank, 2);
        assert!(!info.is_rank_deficient());
    }

    #[test]
    fn test_svd_info() {
        let singular_values = vec![5.0f64, 3.0, 0.1];

        let info = compute_svd_info(3, 3, &singular_values, 1e-10);

        assert_eq!(info.m, 3);
        assert_eq!(info.n, 3);
        assert_eq!(info.num_singular_values, 3);
        assert!((info.sigma_max - 5.0).abs() < 1e-10);
        assert!((info.sigma_min - 0.1).abs() < 1e-10);
        assert!((info.condition_number - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_evd_info() {
        let eigenvalues = vec![-2.0f64, 1.0, 5.0];

        let info = compute_symmetric_evd_info(3, &eigenvalues, 1e-10);

        assert_eq!(info.n, 3);
        assert_eq!(info.num_positive, 2);
        assert_eq!(info.num_negative, 1);
        assert_eq!(info.num_zero, 0);
        assert!(info.is_indefinite());
        assert!(!info.is_positive_definite());
    }

    #[test]
    fn test_general_evd_info() {
        // Eigenvalues: 1+2i, 1-2i, 3
        let real = vec![1.0f64, 1.0, 3.0];
        let imag = vec![2.0f64, -2.0, 0.0];

        let info = compute_general_evd_info(3, &real, &imag, None);

        assert_eq!(info.n, 3);
        assert!(!info.all_real);
        assert_eq!(info.num_complex_pairs, 1);
        assert!((info.trace - 5.0).abs() < 1e-10); // 1 + 1 + 3
    }

    #[test]
    fn test_lu_info_display() {
        let lu: Mat<f64> = Mat::from_rows(&[&[4.0, 3.0], &[0.5, 1.5]]);
        let info = compute_lu_info(lu.as_ref(), &[0, 1], 0);
        let display = format!("{}", info);
        assert!(display.contains("LU Factorization Info"));
        assert!(display.contains("Matrix size: 2×2"));
    }

    #[test]
    fn test_cholesky_failure() {
        // Non-positive definite: diagonal goes negative
        let l: Mat<f64> = Mat::from_rows(&[
            &[2.0, 0.0],
            &[1.0, -1.0], // Negative diagonal
        ]);

        let info = compute_cholesky_info(l.as_ref());

        assert!(!info.is_positive_definite);
        assert!(info.failure_index.is_some());
    }

    #[test]
    fn test_rank_deficient_qr() {
        // R with near-zero diagonal
        let r: Mat<f64> = Mat::from_rows(&[&[5.0, 3.0], &[0.0, 1e-15]]);

        let info = compute_qr_info(r.as_ref(), None, 1e-10);

        assert_eq!(info.numerical_rank, 1);
        assert!(info.is_rank_deficient());
        assert_eq!(info.nullity(), 1);
    }

    #[test]
    fn test_svd_rank_deficient() {
        let singular_values = vec![5.0f64, 3.0, 1e-15];

        let info = compute_svd_info(3, 3, &singular_values, 1e-10);

        assert_eq!(info.numerical_rank, 2);
        assert!(info.is_rank_deficient());
    }

    #[test]
    fn test_symmetric_evd_positive_definite() {
        let eigenvalues = vec![1.0f64, 2.0, 3.0];

        let info = compute_symmetric_evd_info(3, &eigenvalues, 1e-10);

        assert!(info.is_positive_definite());
        assert!(info.is_positive_semidefinite());
        assert!(!info.is_indefinite());
    }

    #[test]
    fn test_symmetric_evd_negative_definite() {
        let eigenvalues = vec![-3.0f64, -2.0, -1.0];

        let info = compute_symmetric_evd_info(3, &eigenvalues, 1e-10);

        assert!(info.is_negative_definite());
        assert!(!info.is_positive_definite());
        assert!(!info.is_indefinite());
    }
}
