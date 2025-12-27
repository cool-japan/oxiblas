//! Automatic algorithm selection for OxiBLAS.
//!
//! This module provides intelligent algorithm selection based on matrix
//! properties such as size, structure, and conditioning.
//!
//! # Examples
//!
//! ```
//! use oxiblas::prelude::*;
//! use oxiblas::auto::*;
//!
//! // Automatic matrix multiply - picks best algorithm based on size
//! let a = MatBuilder::<f64>::random(100, 100, 42);
//! let b = MatBuilder::<f64>::random(100, 100, 43);
//! let c = auto_matmul(a.as_ref(), b.as_ref());
//! ```

use oxiblas_blas::level3;
use oxiblas_blas::level3::GemmKernel;
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Helper to convert MatRef to owned Mat.
fn matref_to_mat<T: Scalar>(r: MatRef<'_, T>) -> Mat<T>
where
    T: bytemuck::Zeroable,
{
    let mut m = Mat::zeros(r.nrows(), r.ncols());
    m.copy_from(&r);
    m
}

/// Threshold for switching to parallel algorithms.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 256;

/// Matrix structure hint for algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixStructure {
    /// General matrix with no special structure.
    General,
    /// Symmetric matrix (A = A^T).
    Symmetric,
    /// Hermitian matrix (A = A^H).
    Hermitian,
    /// Positive definite matrix.
    PositiveDefinite,
    /// Upper triangular matrix.
    UpperTriangular,
    /// Lower triangular matrix.
    LowerTriangular,
    /// Diagonal matrix.
    Diagonal,
}

/// Automatic matrix multiplication.
///
/// Selects the best algorithm based on matrix dimensions:
/// - Medium matrices: Use standard blocked GEMM
/// - Large matrices: Use parallel GEMM (with `parallel` feature)
///
/// # Arguments
///
/// * `a` - Left matrix (m × k)
/// * `b` - Right matrix (k × n)
///
/// # Returns
///
/// The result matrix C = A × B (m × n).
///
/// # Examples
///
/// ```
/// use oxiblas::prelude::*;
/// use oxiblas::auto::auto_matmul;
///
/// let a = MatBuilder::<f64>::identity(50);
/// let b = MatBuilder::<f64>::hilbert(50);
/// let c = auto_matmul(a.as_ref(), b.as_ref());
/// ```
pub fn auto_matmul<T>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T>
where
    T: Scalar + Field + GemmKernel + bytemuck::Zeroable,
{
    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();

    assert_eq!(k, b.nrows(), "Inner dimensions must match");

    let mut c = Mat::zeros(m, n);
    auto_gemm(T::one(), a, b, T::zero(), c.as_mut());
    c
}

/// Automatic GEMM with alpha/beta scaling.
///
/// Computes C = α × A × B + β × C with automatic algorithm selection.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A × B
/// * `a` - Left matrix
/// * `b` - Right matrix
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (modified in place)
///
/// # Examples
///
/// ```
/// use oxiblas::prelude::*;
/// use oxiblas::auto::auto_gemm;
///
/// let a = MatBuilder::<f64>::random(100, 100, 42);
/// let b = MatBuilder::<f64>::random(100, 100, 43);
/// let mut c = MatBuilder::<f64>::zeros(100, 100);
///
/// auto_gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
/// ```
pub fn auto_gemm<T>(alpha: T, a: MatRef<'_, T>, b: MatRef<'_, T>, beta: T, c: MatMut<'_, T>)
where
    T: Scalar + Field + GemmKernel + bytemuck::Zeroable,
{
    // Select algorithm based on dimensions
    #[cfg(feature = "parallel")]
    {
        let m = a.nrows();
        let n = b.ncols();
        let k = a.ncols();
        let max_dim = m.max(n).max(k);
        if max_dim >= PARALLEL_THRESHOLD {
            // Use parallel GEMM for large matrices
            level3::gemm_with_par(alpha, a, b, beta, c, oxiblas_core::Par::Rayon);
            return;
        }
    }

    // Use standard GEMM for smaller matrices
    level3::gemm(alpha, a, b, beta, c);
}

/// Result type for solve operations.
pub type SolveResult<T> = Result<Mat<T>, SolveError>;

/// Error type for solve operations.
#[derive(Debug, Clone)]
pub enum SolveError {
    /// Matrix is singular or nearly singular.
    Singular,
    /// Matrix is not positive definite.
    NotPositiveDefinite,
    /// Dimension mismatch.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },
    /// Generic error.
    Other(String),
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::Singular => write!(f, "Matrix is singular"),
            SolveError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            SolveError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            SolveError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SolveError {}

/// Automatic linear system solve for f64.
///
/// Solves A × x = b using the most appropriate algorithm:
/// - First tries Cholesky (if matrix appears SPD)
/// - Falls back to LU with partial pivoting
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n × n)
/// * `b` - Right-hand side (n × m)
///
/// # Returns
///
/// The solution matrix x (n × m), or an error if the solve fails.
///
/// # Examples
///
/// ```
/// use oxiblas::prelude::*;
/// use oxiblas::auto::auto_solve_f64;
///
/// let a = MatBuilder::<f64>::random_spd(10, 42);
/// let b = MatBuilder::<f64>::random(10, 1, 43);
///
/// let x = auto_solve_f64(a.as_ref(), b.as_ref()).expect("Solve failed");
/// ```
pub fn auto_solve_f64(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> SolveResult<f64> {
    use oxiblas_lapack::cholesky::Cholesky;
    use oxiblas_lapack::lu::Lu;

    let n = a.nrows();

    if a.ncols() != n {
        return Err(SolveError::DimensionMismatch {
            expected: n,
            got: a.ncols(),
        });
    }

    if b.nrows() != n {
        return Err(SolveError::DimensionMismatch {
            expected: n,
            got: b.nrows(),
        });
    }

    // Try Cholesky first if matrix looks SPD
    if is_likely_spd_f64(&a) {
        if let Ok(chol) = Cholesky::compute_auto(a) {
            if let Ok(x) = chol.solve(b) {
                return Ok(x);
            }
        }
    }

    // Fall back to LU
    match Lu::compute_auto(a) {
        Ok(lu) => match lu.solve(b) {
            Ok(x) => Ok(x),
            Err(_) => Err(SolveError::Singular),
        },
        Err(_) => Err(SolveError::Singular),
    }
}

/// Automatic linear system solve for f32.
pub fn auto_solve_f32(a: MatRef<'_, f32>, b: MatRef<'_, f32>) -> SolveResult<f32> {
    use oxiblas_lapack::cholesky::Cholesky;
    use oxiblas_lapack::lu::Lu;

    let n = a.nrows();

    if a.ncols() != n {
        return Err(SolveError::DimensionMismatch {
            expected: n,
            got: a.ncols(),
        });
    }

    if b.nrows() != n {
        return Err(SolveError::DimensionMismatch {
            expected: n,
            got: b.nrows(),
        });
    }

    // Try Cholesky first if matrix looks SPD
    if is_likely_spd_f32(&a) {
        if let Ok(chol) = Cholesky::compute_auto(a) {
            if let Ok(x) = chol.solve(b) {
                return Ok(x);
            }
        }
    }

    // Fall back to LU
    match Lu::compute_auto(a) {
        Ok(lu) => match lu.solve(b) {
            Ok(x) => Ok(x),
            Err(_) => Err(SolveError::Singular),
        },
        Err(_) => Err(SolveError::Singular),
    }
}

/// Heuristic to check if a matrix is likely symmetric positive definite (f64).
fn is_likely_spd_f64(a: &MatRef<'_, f64>) -> bool {
    let n = a.nrows();
    if n != a.ncols() {
        return false;
    }

    // Check if diagonal elements are positive
    for i in 0..n {
        let diag = a[(i, i)];
        if diag <= f64::EPSILON {
            return false;
        }
    }

    // Check if approximately symmetric (sample a few off-diagonal elements)
    let samples = n.min(5);
    for i in 0..samples {
        for j in (i + 1)..n.min(i + samples) {
            let diff = (a[(i, j)] - a[(j, i)]).abs();
            let scale = a[(i, j)].abs() + a[(j, i)].abs() + f64::EPSILON;
            if diff / scale > 1e-6 {
                return false;
            }
        }
    }

    true
}

/// Heuristic to check if a matrix is likely symmetric positive definite (f32).
fn is_likely_spd_f32(a: &MatRef<'_, f32>) -> bool {
    let n = a.nrows();
    if n != a.ncols() {
        return false;
    }

    for i in 0..n {
        let diag = a[(i, i)];
        if diag <= f32::EPSILON {
            return false;
        }
    }

    let samples = n.min(5);
    for i in 0..samples {
        for j in (i + 1)..n.min(i + samples) {
            let diff = (a[(i, j)] - a[(j, i)]).abs();
            let scale = a[(i, j)].abs() + a[(j, i)].abs() + f32::EPSILON;
            if diff / scale > 1e-4 {
                return false;
            }
        }
    }

    true
}

/// Algorithm selection hint for SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvdAlgorithm {
    /// Automatically select the best algorithm.
    #[default]
    Auto,
    /// Standard bidiagonal SVD (more accurate).
    Standard,
    /// Divide and conquer (faster for large matrices).
    DivideConquer,
}

/// Compute SVD with automatic algorithm selection (f64).
///
/// Selects the best SVD algorithm based on matrix size:
/// - Small matrices (< 100): Use standard bidiagonal SVD
/// - Large matrices (>= 100): Use divide and conquer
///
/// # Arguments
///
/// * `a` - Input matrix
///
/// # Returns
///
/// Tuple of (U, S, Vt) where A ≈ U × diag(S) × Vt.
///
/// # Examples
///
/// ```
/// use oxiblas::prelude::*;
/// use oxiblas::auto::auto_svd_f64;
///
/// let a = MatBuilder::<f64>::random(50, 30, 42);
/// let (u, s, vt) = auto_svd_f64(a.as_ref());
/// ```
pub fn auto_svd_f64(a: MatRef<'_, f64>) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    auto_svd_f64_with_algorithm(a, SvdAlgorithm::Auto)
}

/// Compute SVD with specified algorithm (f64).
pub fn auto_svd_f64_with_algorithm(
    a: MatRef<'_, f64>,
    algorithm: SvdAlgorithm,
) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let min_dim = m.min(n);

    let algo = match algorithm {
        SvdAlgorithm::Auto => {
            if min_dim < 100 {
                SvdAlgorithm::Standard
            } else {
                SvdAlgorithm::DivideConquer
            }
        }
        other => other,
    };

    match algo {
        SvdAlgorithm::Standard | SvdAlgorithm::Auto => {
            use oxiblas_lapack::svd::Svd;
            let svd = Svd::compute(a.to_owned()).expect("SVD failed");
            (
                matref_to_mat(svd.u()),
                svd.singular_values().to_vec(),
                matref_to_mat(svd.vt()),
            )
        }
        SvdAlgorithm::DivideConquer => {
            use oxiblas_lapack::svd::SvdDc;
            let svd = SvdDc::compute(a.to_owned()).expect("SVD DC failed");
            (
                matref_to_mat(svd.u()),
                svd.singular_values().to_vec(),
                matref_to_mat(svd.vt()),
            )
        }
    }
}

/// Compute SVD with automatic algorithm selection (f32).
pub fn auto_svd_f32(a: MatRef<'_, f32>) -> (Mat<f32>, Vec<f32>, Mat<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let min_dim = m.min(n);

    if min_dim < 100 {
        use oxiblas_lapack::svd::Svd;
        let svd = Svd::compute(a.to_owned()).expect("SVD failed");
        (
            matref_to_mat(svd.u()),
            svd.singular_values().to_vec(),
            matref_to_mat(svd.vt()),
        )
    } else {
        use oxiblas_lapack::svd::SvdDc;
        let svd = SvdDc::compute(a.to_owned()).expect("SVD DC failed");
        (
            matref_to_mat(svd.u()),
            svd.singular_values().to_vec(),
            matref_to_mat(svd.vt()),
        )
    }
}

/// Compute eigenvalues with automatic algorithm selection (f64).
///
/// For symmetric matrices, uses efficient symmetric EVD (real eigenvalues only).
/// For general matrices, uses Schur decomposition and returns real parts.
///
/// # Arguments
///
/// * `a` - Square input matrix
/// * `symmetric` - Hint if the matrix is symmetric
///
/// # Returns
///
/// Vector of eigenvalues (real parts for general matrices).
///
/// # Examples
///
/// ```
/// use oxiblas::prelude::*;
/// use oxiblas::auto::auto_eigenvalues_f64;
///
/// let a = MatBuilder::<f64>::random_spd(20, 42);
/// let eigvals = auto_eigenvalues_f64(a.as_ref(), true);
/// ```
pub fn auto_eigenvalues_f64(a: MatRef<'_, f64>, symmetric: bool) -> Vec<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Matrix must be square");

    if symmetric {
        use oxiblas_lapack::evd::SymmetricEvd;
        let evd = SymmetricEvd::compute(a).expect("EVD failed");
        evd.eigenvalues().to_vec()
    } else {
        use oxiblas_lapack::evd::GeneralEvd;
        let evd = GeneralEvd::compute(a).expect("EVD failed");
        // For general matrices, eigenvalues may be complex. Return real parts.
        evd.eigenvalues().iter().map(|e| e.real).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::MatBuilder;

    #[test]
    fn test_auto_matmul() {
        let a = MatBuilder::<f64>::identity(10);
        let b = MatBuilder::<f64>::hilbert(10);
        let c = auto_matmul(a.as_ref(), b.as_ref());

        // Identity * Hilbert = Hilbert
        for i in 0..10 {
            for j in 0..10 {
                let expected = 1.0 / ((i + j + 1) as f64);
                assert!((c[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_auto_solve_spd() {
        let a = MatBuilder::<f64>::random_spd(10, 42);
        let b = MatBuilder::<f64>::random(10, 1, 43);

        let x = auto_solve_f64(a.as_ref(), b.as_ref()).expect("Solve failed");

        // Verify: A * x ≈ b
        let mut ax = MatBuilder::<f64>::zeros(10, 1);
        auto_gemm(1.0, a.as_ref(), x.as_ref(), 0.0, ax.as_mut());

        for i in 0..10 {
            assert!((ax[(i, 0)] - b[(i, 0)]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_auto_svd() {
        let a = MatBuilder::<f64>::random(20, 10, 42);
        let (u, s, vt) = auto_svd_f64(a.as_ref());

        // Full SVD: U is m×m, Vt is n×n
        assert_eq!(u.nrows(), 20);
        assert_eq!(u.ncols(), 20);
        assert_eq!(s.len(), 10);
        assert_eq!(vt.nrows(), 10);
        assert_eq!(vt.ncols(), 10);

        // Singular values should be non-negative and sorted descending
        for i in 1..s.len() {
            assert!(s[i - 1] >= s[i]);
            assert!(s[i] >= 0.0);
        }
    }

    #[test]
    fn test_auto_eigenvalues_symmetric() {
        let a = MatBuilder::<f64>::random_spd(10, 42);
        let eigvals = auto_eigenvalues_f64(a.as_ref(), true);

        assert_eq!(eigvals.len(), 10);

        // Eigenvalues of SPD should be positive
        for &e in &eigvals {
            assert!(e > 0.0);
        }
    }

    #[test]
    fn test_is_likely_spd() {
        let spd = MatBuilder::<f64>::random_spd(10, 42);
        assert!(is_likely_spd_f64(&spd.as_ref()));
    }
}
