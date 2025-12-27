//! Tridiagonal System Solvers.
//!
//! Provides efficient O(n) solvers for tridiagonal linear systems using
//! the Thomas algorithm (tridiagonal matrix algorithm).
//!
//! A tridiagonal system has the form:
//! ```text
//! | b₀  c₀   0   0  ... |   | x₀ |   | d₀ |
//! | a₁  b₁  c₁   0  ... |   | x₁ |   | d₁ |
//! |  0  a₂  b₂  c₂  ... | × | x₂ | = | d₂ |
//! | ...                  |   | .. |   | .. |
//! ```
//!
//! where:
//! - `a` is the sub-diagonal (length n-1)
//! - `b` is the main diagonal (length n)
//! - `c` is the super-diagonal (length n-1)
//! - `d` is the right-hand side (length n)

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for tridiagonal solve operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagError {
    /// System is singular (zero pivot encountered).
    Singular {
        /// The index where singularity was detected.
        index: usize,
    },
    /// Dimension mismatch in input arrays.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// System is empty.
    EmptySystem,
}

impl core::fmt::Display for TridiagError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Singular { index } => {
                write!(f, "Tridiagonal system is singular at index {index}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
            Self::EmptySystem => write!(f, "Empty tridiagonal system"),
        }
    }
}

impl std::error::Error for TridiagError {}

/// Solves a tridiagonal system Ax = d using the Thomas algorithm.
///
/// The matrix A is tridiagonal with:
/// - `dl`: sub-diagonal (below main diagonal), length n-1
/// - `d_diag`: main diagonal, length n
/// - `du`: super-diagonal (above main diagonal), length n-1
///
/// # Arguments
///
/// * `dl` - Sub-diagonal elements (length n-1)
/// * `d_diag` - Main diagonal elements (length n)
/// * `du` - Super-diagonal elements (length n-1)
/// * `b` - Right-hand side vector (length n)
///
/// # Returns
///
/// Solution vector x such that Ax = b.
///
/// # Algorithm
///
/// Uses the Thomas algorithm (O(n) complexity):
/// 1. Forward elimination: eliminate sub-diagonal
/// 2. Back substitution: solve for x
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::tridiag_solve;
///
/// // Solve: [2 -1  0] [x0]   [1]
/// //        [-1 2 -1] [x1] = [0]
/// //        [0 -1  2] [x2]   [1]
///
/// let dl = [-1.0f64, -1.0];  // sub-diagonal
/// let d = [2.0f64, 2.0, 2.0]; // main diagonal
/// let du = [-1.0f64, -1.0];   // super-diagonal
/// let b = [1.0f64, 0.0, 1.0]; // right-hand side
///
/// let x = tridiag_solve(&dl, &d, &du, &b).unwrap();
///
/// // Verify: x should be [1, 1, 1]
/// assert!((x[0] - 1.0).abs() < 1e-10);
/// assert!((x[1] - 1.0).abs() < 1e-10);
/// assert!((x[2] - 1.0).abs() < 1e-10);
/// ```
pub fn tridiag_solve<T: Field + Real>(
    dl: &[T],
    d_diag: &[T],
    du: &[T],
    b: &[T],
) -> Result<Vec<T>, TridiagError> {
    let n = d_diag.len();

    if n == 0 {
        return Err(TridiagError::EmptySystem);
    }

    if dl.len() != n - 1 {
        return Err(TridiagError::DimensionMismatch {
            expected: n - 1,
            actual: dl.len(),
        });
    }

    if du.len() != n - 1 {
        return Err(TridiagError::DimensionMismatch {
            expected: n - 1,
            actual: du.len(),
        });
    }

    if b.len() != n {
        return Err(TridiagError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }

    // Special case: n = 1
    if n == 1 {
        let eps = <T as Scalar>::epsilon();
        if Scalar::abs(d_diag[0]) <= eps {
            return Err(TridiagError::Singular { index: 0 });
        }
        return Ok(vec![b[0] / d_diag[0]]);
    }

    // Thomas algorithm
    // Modified coefficients
    let mut c_prime = vec![T::zero(); n - 1];
    let mut d_prime = vec![T::zero(); n];

    let eps = <T as Scalar>::epsilon();

    // Forward sweep
    // c'[0] = c[0] / b[0]
    // d'[0] = d[0] / b[0]
    if Scalar::abs(d_diag[0]) <= eps {
        return Err(TridiagError::Singular { index: 0 });
    }
    c_prime[0] = du[0] / d_diag[0];
    d_prime[0] = b[0] / d_diag[0];

    for i in 1..n {
        // denom = b[i] - a[i] * c'[i-1]
        let denom = d_diag[i] - dl[i - 1] * c_prime[i - 1];

        if Scalar::abs(denom) <= eps {
            return Err(TridiagError::Singular { index: i });
        }

        if i < n - 1 {
            // c'[i] = c[i] / denom
            c_prime[i] = du[i] / denom;
        }

        // d'[i] = (d[i] - a[i] * d'[i-1]) / denom
        d_prime[i] = (b[i] - dl[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solves a tridiagonal system with multiple right-hand sides.
///
/// Solves AX = B where A is tridiagonal and B has multiple columns.
///
/// # Arguments
///
/// * `dl` - Sub-diagonal elements (length n-1)
/// * `d_diag` - Main diagonal elements (length n)
/// * `du` - Super-diagonal elements (length n-1)
/// * `b` - Right-hand side matrix (n × nrhs)
///
/// # Returns
///
/// Solution matrix X such that AX = B.
pub fn tridiag_solve_multiple<T: Field + Real + bytemuck::Zeroable>(
    dl: &[T],
    d_diag: &[T],
    du: &[T],
    b: MatRef<'_, T>,
) -> Result<Mat<T>, TridiagError> {
    let n = d_diag.len();
    let nrhs = b.ncols();

    if n == 0 {
        return Err(TridiagError::EmptySystem);
    }

    if b.nrows() != n {
        return Err(TridiagError::DimensionMismatch {
            expected: n,
            actual: b.nrows(),
        });
    }

    let mut result = Mat::zeros(n, nrhs);

    // Solve each column separately
    for j in 0..nrhs {
        let col: Vec<T> = (0..n).map(|i| b[(i, j)]).collect();
        let x = tridiag_solve(dl, d_diag, du, &col)?;
        for i in 0..n {
            result[(i, j)] = x[i];
        }
    }

    Ok(result)
}

/// Result of tridiagonal factorization.
#[derive(Debug, Clone)]
pub struct TridiagFactors<T: Scalar> {
    /// Modified sub-diagonal (length n-1)
    pub dl_modified: Vec<T>,
    /// Modified main diagonal (length n)
    pub d_modified: Vec<T>,
    /// Original super-diagonal (length n-1)
    pub du: Vec<T>,
    /// System size
    pub n: usize,
}

/// Computes the LU factorization of a tridiagonal matrix (without pivoting).
///
/// The factorization is A = LU where:
/// - L is unit lower bidiagonal
/// - U is upper bidiagonal
///
/// # Arguments
///
/// * `dl` - Sub-diagonal elements (length n-1)
/// * `d_diag` - Main diagonal elements (length n)
/// * `du` - Super-diagonal elements (length n-1)
///
/// # Returns
///
/// Factorization data that can be used with `tridiag_solve_factored`.
pub fn tridiag_factor<T: Field + Real>(
    dl: &[T],
    d_diag: &[T],
    du: &[T],
) -> Result<TridiagFactors<T>, TridiagError> {
    let n = d_diag.len();

    if n == 0 {
        return Err(TridiagError::EmptySystem);
    }

    if dl.len() != n - 1 || du.len() != n - 1 {
        return Err(TridiagError::DimensionMismatch {
            expected: n - 1,
            actual: dl.len().min(du.len()),
        });
    }

    let eps = <T as Scalar>::epsilon();

    // LU factorization without pivoting
    // L is unit lower bidiagonal with multipliers in dl_modified
    // U is upper bidiagonal with diagonal in d_modified
    let mut dl_modified = vec![T::zero(); n - 1];
    let mut d_modified = vec![T::zero(); n];

    d_modified[0] = d_diag[0];

    if Scalar::abs(d_modified[0]) <= eps {
        return Err(TridiagError::Singular { index: 0 });
    }

    for i in 1..n {
        // Multiplier: l[i] = a[i] / u[i-1]
        dl_modified[i - 1] = dl[i - 1] / d_modified[i - 1];

        // U diagonal: u[i] = b[i] - l[i] * c[i-1]
        d_modified[i] = d_diag[i] - dl_modified[i - 1] * du[i - 1];

        if Scalar::abs(d_modified[i]) <= eps {
            return Err(TridiagError::Singular { index: i });
        }
    }

    Ok(TridiagFactors {
        dl_modified,
        d_modified,
        du: du.to_vec(),
        n,
    })
}

/// Solves a tridiagonal system using precomputed factors.
///
/// Given factors from `tridiag_factor`, solves Ax = b.
///
/// # Arguments
///
/// * `factors` - Precomputed LU factors
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// Solution vector x.
pub fn tridiag_solve_factored<T: Field + Real>(
    factors: &TridiagFactors<T>,
    b: &[T],
) -> Result<Vec<T>, TridiagError> {
    let n = factors.n;

    if b.len() != n {
        return Err(TridiagError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }

    // Forward substitution: Ly = b
    let mut y = vec![T::zero(); n];
    y[0] = b[0];

    for i in 1..n {
        y[i] = b[i] - factors.dl_modified[i - 1] * y[i - 1];
    }

    // Back substitution: Ux = y
    let mut x = vec![T::zero(); n];
    x[n - 1] = y[n - 1] / factors.d_modified[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = (y[i] - factors.du[i] * x[i + 1]) / factors.d_modified[i];
    }

    Ok(x)
}

/// Result of symmetric positive definite tridiagonal factorization.
#[derive(Debug, Clone)]
pub struct TridiagSPDFactors<T: Scalar> {
    /// Factored main diagonal (D in LDL^T)
    pub d_factor: Vec<T>,
    /// Sub-diagonal multipliers (L in LDL^T, unit lower bidiagonal)
    pub l_factor: Vec<T>,
    /// System size
    pub n: usize,
}

/// Computes the LDL^T factorization of a symmetric positive definite tridiagonal matrix.
///
/// For SPD tridiagonal matrix A, computes A = L * D * L^T where:
/// - L is unit lower bidiagonal
/// - D is diagonal
///
/// This is equivalent to LAPACK's DPTTRF.
///
/// # Arguments
///
/// * `d_diag` - Main diagonal (must be positive), length n
/// * `e` - Off-diagonal elements, length n-1
///
/// # Returns
///
/// `TridiagSPDFactors` containing the factorization, or error if not SPD.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::tridiag_factor_spd;
///
/// // SPD tridiagonal: [4 -1  0]
/// //                  [-1 4 -1]
/// //                  [0 -1  4]
/// let d = [4.0f64, 4.0, 4.0];
/// let e = [-1.0f64, -1.0];
///
/// let factors = tridiag_factor_spd(&d, &e).unwrap();
/// // Can now use tridiag_solve_factored_spd to solve multiple systems
/// ```
pub fn tridiag_factor_spd<T: Field + Real>(
    d_diag: &[T],
    e: &[T],
) -> Result<TridiagSPDFactors<T>, TridiagError> {
    let n = d_diag.len();

    if n == 0 {
        return Err(TridiagError::EmptySystem);
    }

    if e.len() != n.saturating_sub(1) && n > 1 {
        return Err(TridiagError::DimensionMismatch {
            expected: n - 1,
            actual: e.len(),
        });
    }

    let eps = <T as Scalar>::epsilon();

    // LDL^T factorization
    let mut d_factor = vec![T::zero(); n];
    let mut l_factor = vec![T::zero(); n.saturating_sub(1)];

    d_factor[0] = d_diag[0];
    if d_factor[0] <= eps {
        return Err(TridiagError::Singular { index: 0 });
    }

    for i in 1..n {
        // L_i = e[i-1] / d_factor[i-1]
        l_factor[i - 1] = e[i - 1] / d_factor[i - 1];

        // d_factor[i] = d_diag[i] - l_factor[i-1]^2 * d_factor[i-1]
        d_factor[i] = d_diag[i] - l_factor[i - 1] * l_factor[i - 1] * d_factor[i - 1];

        if d_factor[i] <= eps {
            return Err(TridiagError::Singular { index: i });
        }
    }

    Ok(TridiagSPDFactors {
        d_factor,
        l_factor,
        n,
    })
}

/// Solves a symmetric positive definite tridiagonal system using precomputed factors.
///
/// Given factors from `tridiag_factor_spd`, solves Ax = b.
/// This is equivalent to LAPACK's DPTTRS.
///
/// # Arguments
///
/// * `factors` - Precomputed LDL^T factors from `tridiag_factor_spd`
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// Solution vector x.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::{tridiag_factor_spd, tridiag_solve_factored_spd};
///
/// let d = [4.0f64, 4.0, 4.0];
/// let e = [-1.0f64, -1.0];
/// let b = [3.0f64, 2.0, 3.0];
///
/// let factors = tridiag_factor_spd(&d, &e).unwrap();
/// let x = tridiag_solve_factored_spd(&factors, &b).unwrap();
///
/// // Verify solution (Ax = b)
/// let ax0 = d[0] * x[0] + e[0] * x[1];
/// assert!((ax0 - b[0]).abs() < 1e-10);
/// ```
pub fn tridiag_solve_factored_spd<T: Field + Real>(
    factors: &TridiagSPDFactors<T>,
    b: &[T],
) -> Result<Vec<T>, TridiagError> {
    let n = factors.n;

    if b.len() != n {
        return Err(TridiagError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    // Forward substitution: Ly = b
    let mut y = vec![T::zero(); n];
    y[0] = b[0];

    for i in 1..n {
        y[i] = b[i] - factors.l_factor[i - 1] * y[i - 1];
    }

    // Diagonal scaling: Dz = y
    let mut z = vec![T::zero(); n];
    for i in 0..n {
        z[i] = y[i] / factors.d_factor[i];
    }

    // Back substitution: L^T x = z
    let mut x = vec![T::zero(); n];
    x[n - 1] = z[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = z[i] - factors.l_factor[i] * x[i + 1];
    }

    Ok(x)
}

/// Solves a symmetric positive definite tridiagonal system.
///
/// For SPD tridiagonal matrices, uses a more numerically stable algorithm
/// based on Cholesky-like factorization.
///
/// # Arguments
///
/// * `d_diag` - Main diagonal (must be positive)
/// * `e` - Off-diagonal (symmetric, so only one array needed)
/// * `b` - Right-hand side
///
/// # Returns
///
/// Solution vector x.
pub fn tridiag_solve_spd<T: Field + Real>(
    d_diag: &[T],
    e: &[T],
    b: &[T],
) -> Result<Vec<T>, TridiagError> {
    let n = d_diag.len();

    if n == 0 {
        return Err(TridiagError::EmptySystem);
    }

    if e.len() != n - 1 {
        return Err(TridiagError::DimensionMismatch {
            expected: n - 1,
            actual: e.len(),
        });
    }

    if b.len() != n {
        return Err(TridiagError::DimensionMismatch {
            expected: n,
            actual: b.len(),
        });
    }

    let eps = <T as Scalar>::epsilon();

    // LDL^T factorization for symmetric tridiagonal
    // A = LDL^T where L is unit lower bidiagonal
    let mut d_factor = vec![T::zero(); n];
    let mut l_factor = vec![T::zero(); n - 1];

    d_factor[0] = d_diag[0];
    if d_factor[0] <= eps {
        return Err(TridiagError::Singular { index: 0 });
    }

    for i in 1..n {
        l_factor[i - 1] = e[i - 1] / d_factor[i - 1];
        d_factor[i] = d_diag[i] - l_factor[i - 1] * l_factor[i - 1] * d_factor[i - 1];

        if d_factor[i] <= eps {
            return Err(TridiagError::Singular { index: i });
        }
    }

    // Forward substitution: Ly = b
    let mut y = vec![T::zero(); n];
    y[0] = b[0];

    for i in 1..n {
        y[i] = b[i] - l_factor[i - 1] * y[i - 1];
    }

    // Diagonal scaling: Dz = y
    let mut z = vec![T::zero(); n];
    for i in 0..n {
        z[i] = y[i] / d_factor[i];
    }

    // Back substitution: L^T x = z
    let mut x = vec![T::zero(); n];
    x[n - 1] = z[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = z[i] - l_factor[i] * x[i + 1];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_tridiag_solve_simple() {
        // Solve: [2 -1  0] [x0]   [1]
        //        [-1 2 -1] [x1] = [0]
        //        [0 -1  2] [x2]   [1]
        // Solution: x = [1, 1, 1]

        let dl = [-1.0f64, -1.0];
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];
        let b = [1.0f64, 0.0, 1.0];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], 1.0, 1e-10));
        assert!(approx_eq(x[2], 1.0, 1e-10));
    }

    #[test]
    fn test_tridiag_solve_2x2() {
        // [2 1] [x0]   [5]
        // [1 3] [x1] = [7]
        // Solution: x0 = 1.6, x1 = 1.8

        let dl = [1.0f64];
        let d = [2.0f64, 3.0];
        let du = [1.0f64];
        let b = [5.0f64, 7.0];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        assert!(approx_eq(x[0], 1.6, 1e-10));
        assert!(approx_eq(x[1], 1.8, 1e-10));
    }

    #[test]
    fn test_tridiag_solve_1x1() {
        let dl: [f64; 0] = [];
        let d = [2.0f64];
        let du: [f64; 0] = [];
        let b = [4.0f64];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        assert!(approx_eq(x[0], 2.0, 1e-10));
    }

    #[test]
    fn test_tridiag_solve_diagonal() {
        // Diagonal system (no off-diagonals)
        let dl = [0.0f64, 0.0];
        let d = [2.0f64, 3.0, 4.0];
        let du = [0.0f64, 0.0];
        let b = [4.0f64, 9.0, 16.0];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        assert!(approx_eq(x[0], 2.0, 1e-10));
        assert!(approx_eq(x[1], 3.0, 1e-10));
        assert!(approx_eq(x[2], 4.0, 1e-10));
    }

    #[test]
    fn test_tridiag_solve_singular() {
        let dl = [1.0f64];
        let d = [0.0f64, 1.0]; // Zero pivot
        let du = [1.0f64];
        let b = [1.0f64, 1.0];

        let result = tridiag_solve(&dl, &d, &du, &b);
        assert!(matches!(result, Err(TridiagError::Singular { index: 0 })));
    }

    #[test]
    fn test_tridiag_solve_verify() {
        // Random system, verify Ax = b
        let dl = [1.0f64, 2.0, 1.5];
        let d = [4.0f64, 5.0, 6.0, 7.0];
        let du = [1.0f64, 1.0, 2.0];
        let b = [10.0f64, 20.0, 30.0, 40.0];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        // Verify: Ax = b
        let ax0 = d[0] * x[0] + du[0] * x[1];
        let ax1 = dl[0] * x[0] + d[1] * x[1] + du[1] * x[2];
        let ax2 = dl[1] * x[1] + d[2] * x[2] + du[2] * x[3];
        let ax3 = dl[2] * x[2] + d[3] * x[3];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
        assert!(approx_eq(ax2, b[2], 1e-10));
        assert!(approx_eq(ax3, b[3], 1e-10));
    }

    #[test]
    fn test_tridiag_factor_solve() {
        let dl = [-1.0f64, -1.0];
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];
        let b = [1.0f64, 0.0, 1.0];

        let factors = tridiag_factor(&dl, &d, &du).unwrap();
        let x = tridiag_solve_factored(&factors, &b).unwrap();

        assert!(approx_eq(x[0], 1.0, 1e-10));
        assert!(approx_eq(x[1], 1.0, 1e-10));
        assert!(approx_eq(x[2], 1.0, 1e-10));
    }

    #[test]
    fn test_tridiag_factor_multiple_rhs() {
        let dl = [-1.0f64, -1.0];
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];

        let factors = tridiag_factor(&dl, &d, &du).unwrap();

        // First RHS
        let b1 = [1.0f64, 0.0, 1.0];
        let x1 = tridiag_solve_factored(&factors, &b1).unwrap();

        // Second RHS
        let b2 = [2.0f64, 0.0, 2.0];
        let x2 = tridiag_solve_factored(&factors, &b2).unwrap();

        // x2 should be 2 * x1
        for i in 0..3 {
            assert!(approx_eq(x2[i], 2.0 * x1[i], 1e-10));
        }
    }

    #[test]
    fn test_tridiag_solve_spd() {
        // SPD tridiagonal: [4 -1  0]
        //                  [-1 4 -1]
        //                  [0 -1  4]
        let d = [4.0f64, 4.0, 4.0];
        let e = [-1.0f64, -1.0];
        let b = [3.0f64, 2.0, 3.0];

        let x = tridiag_solve_spd(&d, &e, &b).unwrap();

        // Verify Ax = b
        let ax0 = d[0] * x[0] + e[0] * x[1];
        let ax1 = e[0] * x[0] + d[1] * x[1] + e[1] * x[2];
        let ax2 = e[1] * x[1] + d[2] * x[2];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
        assert!(approx_eq(ax2, b[2], 1e-10));
    }

    #[test]
    fn test_tridiag_solve_multiple() {
        let dl = [-1.0f64, -1.0];
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];

        // Two right-hand sides
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 0.0], &[1.0, 2.0]]);

        let x = tridiag_solve_multiple(&dl, &d, &du, b.as_ref()).unwrap();

        // First column should be [1, 1, 1]
        assert!(approx_eq(x[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 1.0, 1e-10));
        assert!(approx_eq(x[(2, 0)], 1.0, 1e-10));

        // Second column should be [2, 2, 2]
        assert!(approx_eq(x[(0, 1)], 2.0, 1e-10));
        assert!(approx_eq(x[(1, 1)], 2.0, 1e-10));
        assert!(approx_eq(x[(2, 1)], 2.0, 1e-10));
    }

    #[test]
    fn test_tridiag_solve_f32() {
        let dl = [-1.0f32, -1.0];
        let d = [2.0f32, 2.0, 2.0];
        let du = [-1.0f32, -1.0];
        let b = [1.0f32, 0.0, 1.0];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 1.0).abs() < 1e-5);
        assert!((x[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tridiag_dimension_mismatch() {
        let dl = [-1.0f64]; // Wrong size
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];
        let b = [1.0f64, 0.0, 1.0];

        let result = tridiag_solve(&dl, &d, &du, &b);
        assert!(matches!(
            result,
            Err(TridiagError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_tridiag_empty() {
        let dl: [f64; 0] = [];
        let d: [f64; 0] = [];
        let du: [f64; 0] = [];
        let b: [f64; 0] = [];

        let result = tridiag_solve(&dl, &d, &du, &b);
        assert!(matches!(result, Err(TridiagError::EmptySystem)));
    }

    #[test]
    fn test_tridiag_large_system() {
        // Large system: second-order finite difference matrix
        let n = 100;
        let dl: Vec<f64> = vec![-1.0; n - 1];
        let d: Vec<f64> = vec![2.0; n];
        let du: Vec<f64> = vec![-1.0; n - 1];

        // RHS: all ones
        let b: Vec<f64> = vec![1.0; n];

        let x = tridiag_solve(&dl, &d, &du, &b).unwrap();

        // Verify Ax = b at a few points
        let ax0 = d[0] * x[0] + du[0] * x[1];
        let ax_mid = dl[n / 2 - 1] * x[n / 2 - 1] + d[n / 2] * x[n / 2] + du[n / 2] * x[n / 2 + 1];
        let ax_last = dl[n - 2] * x[n - 2] + d[n - 1] * x[n - 1];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax_mid, b[n / 2], 1e-10));
        assert!(approx_eq(ax_last, b[n - 1], 1e-10));
    }

    // ===== SPD Tridiagonal Factorization Tests (PTTRF/PTTRS equivalent) =====

    #[test]
    fn test_tridiag_factor_spd_basic() {
        // SPD tridiagonal: [4 -1  0]
        //                  [-1 4 -1]
        //                  [0 -1  4]
        let d = [4.0f64, 4.0, 4.0];
        let e = [-1.0f64, -1.0];

        let factors = tridiag_factor_spd(&d, &e).unwrap();

        // Check dimensions
        assert_eq!(factors.n, 3);
        assert_eq!(factors.d_factor.len(), 3);
        assert_eq!(factors.l_factor.len(), 2);

        // Verify all d_factor values are positive (SPD property)
        for &df in &factors.d_factor {
            assert!(df > 0.0);
        }
    }

    #[test]
    fn test_tridiag_factor_solve_spd() {
        let d = [4.0f64, 4.0, 4.0];
        let e = [-1.0f64, -1.0];
        let b = [3.0f64, 2.0, 3.0];

        // Factor then solve
        let factors = tridiag_factor_spd(&d, &e).unwrap();
        let x = tridiag_solve_factored_spd(&factors, &b).unwrap();

        // Verify Ax = b
        let ax0 = d[0] * x[0] + e[0] * x[1];
        let ax1 = e[0] * x[0] + d[1] * x[1] + e[1] * x[2];
        let ax2 = e[1] * x[1] + d[2] * x[2];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
        assert!(approx_eq(ax2, b[2], 1e-10));
    }

    #[test]
    fn test_tridiag_factor_spd_multiple_rhs() {
        let d = [4.0f64, 4.0, 4.0];
        let e = [-1.0f64, -1.0];

        let factors = tridiag_factor_spd(&d, &e).unwrap();

        // First RHS
        let b1 = [3.0f64, 2.0, 3.0];
        let x1 = tridiag_solve_factored_spd(&factors, &b1).unwrap();

        // Second RHS
        let b2 = [6.0f64, 4.0, 6.0];
        let x2 = tridiag_solve_factored_spd(&factors, &b2).unwrap();

        // x2 should be 2 * x1
        for i in 0..3 {
            assert!(approx_eq(x2[i], 2.0 * x1[i], 1e-10));
        }
    }

    #[test]
    fn test_tridiag_factor_spd_1x1() {
        let d = [4.0f64];
        let e: [f64; 0] = [];

        let factors = tridiag_factor_spd(&d, &e).unwrap();

        assert_eq!(factors.n, 1);
        assert!(approx_eq(factors.d_factor[0], 4.0, 1e-10));

        let b = [8.0f64];
        let x = tridiag_solve_factored_spd(&factors, &b).unwrap();
        assert!(approx_eq(x[0], 2.0, 1e-10));
    }

    #[test]
    fn test_tridiag_factor_spd_not_positive() {
        // Not SPD: negative diagonal
        let d = [-4.0f64, 4.0, 4.0];
        let e = [-1.0f64, -1.0];

        let result = tridiag_factor_spd(&d, &e);
        assert!(matches!(result, Err(TridiagError::Singular { index: 0 })));
    }

    #[test]
    fn test_tridiag_factor_spd_not_definite() {
        // Not positive definite: off-diagonal too large
        let d = [1.0f64, 1.0, 1.0];
        let e = [2.0f64, 2.0]; // |e| > d will cause negative pivot

        let result = tridiag_factor_spd(&d, &e);
        assert!(matches!(result, Err(TridiagError::Singular { .. })));
    }

    #[test]
    fn test_tridiag_factor_spd_large() {
        // Large SPD tridiagonal: second-order finite difference (2 on diagonal)
        let n = 100;
        let d: Vec<f64> = vec![2.0; n];
        let e: Vec<f64> = vec![-1.0; n - 1];
        let b: Vec<f64> = vec![1.0; n];

        let factors = tridiag_factor_spd(&d, &e).unwrap();
        let x = tridiag_solve_factored_spd(&factors, &b).unwrap();

        // Verify Ax = b at a few points
        let ax0 = d[0] * x[0] + e[0] * x[1];
        let ax_mid = e[n / 2 - 1] * x[n / 2 - 1] + d[n / 2] * x[n / 2] + e[n / 2] * x[n / 2 + 1];
        let ax_last = e[n - 2] * x[n - 2] + d[n - 1] * x[n - 1];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax_mid, b[n / 2], 1e-10));
        assert!(approx_eq(ax_last, b[n - 1], 1e-10));
    }

    #[test]
    fn test_tridiag_factor_spd_f32() {
        let d = [4.0f32, 4.0, 4.0];
        let e = [-1.0f32, -1.0];
        let b = [3.0f32, 2.0, 3.0];

        let factors = tridiag_factor_spd(&d, &e).unwrap();
        let x = tridiag_solve_factored_spd(&factors, &b).unwrap();

        // Verify Ax = b (with lower precision)
        let ax0 = d[0] * x[0] + e[0] * x[1];
        let ax1 = e[0] * x[0] + d[1] * x[1] + e[1] * x[2];
        let ax2 = e[1] * x[1] + d[2] * x[2];

        assert!((ax0 - b[0]).abs() < 1e-5);
        assert!((ax1 - b[1]).abs() < 1e-5);
        assert!((ax2 - b[2]).abs() < 1e-5);
    }

    #[test]
    fn test_tridiag_factor_spd_consistency() {
        // Verify that factored solve matches direct solve
        let d = [4.0f64, 4.0, 4.0, 4.0];
        let e = [-1.0f64, -1.0, -1.0];
        let b = [3.0f64, 2.0, 2.0, 3.0];

        // Direct solve
        let x_direct = tridiag_solve_spd(&d, &e, &b).unwrap();

        // Factored solve
        let factors = tridiag_factor_spd(&d, &e).unwrap();
        let x_factored = tridiag_solve_factored_spd(&factors, &b).unwrap();

        // Should match
        for i in 0..4 {
            assert!(approx_eq(x_direct[i], x_factored[i], 1e-14));
        }
    }

    #[test]
    fn test_tridiag_factor_spd_empty() {
        let d: [f64; 0] = [];
        let e: [f64; 0] = [];

        let result = tridiag_factor_spd(&d, &e);
        assert!(matches!(result, Err(TridiagError::EmptySystem)));
    }

    #[test]
    fn test_tridiag_factor_spd_dimension_mismatch() {
        let d = [4.0f64, 4.0, 4.0];
        let e = [-1.0f64]; // Wrong size

        let result = tridiag_factor_spd(&d, &e);
        assert!(matches!(
            result,
            Err(TridiagError::DimensionMismatch { .. })
        ));
    }
}
