//! Standard test matrix generators for sparse operations.
//!
//! Provides well-known test matrices commonly used in numerical linear algebra
//! benchmarks and validation. All generators return `CsrMatrix<f64>`.
//!
//! # Available Generators
//!
//! - [`laplacian_2d`] - 2D Laplacian with 5-point stencil
//! - [`laplacian_3d`] - 3D Laplacian with 7-point stencil
//! - [`tridiagonal`] - General tridiagonal matrix
//! - [`diagonal`] - Diagonal matrix
//! - [`arrow_matrix`] - Arrow/arrowhead matrix
//! - [`random_spd`] - Random symmetric positive definite matrix
//! - [`poisson_1d`] - 1D Poisson matrix
//!
//! # Properties
//!
//! Most generators produce matrices with known mathematical properties
//! (symmetry, positive definiteness, specific eigenvalue distributions)
//! making them suitable for validating solvers and preconditioners.

use crate::CooMatrixBuilder;
use crate::csr::CsrMatrix;

/// Error type for test matrix generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestMatrixError {
    /// Invalid dimensions (zero or too large).
    InvalidDimension {
        /// Description of the invalid parameter.
        param: String,
        /// The invalid value.
        value: usize,
    },
    /// Density out of valid range [0, 1].
    InvalidDensity {
        /// The invalid density value.
        density: String,
    },
    /// Matrix construction failed internally.
    ConstructionError {
        /// Description of the failure.
        description: String,
    },
}

impl core::fmt::Display for TestMatrixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimension { param, value } => {
                write!(f, "Invalid dimension for {param}: {value}")
            }
            Self::InvalidDensity { density } => {
                write!(f, "Invalid density: {density} (must be in [0, 1])")
            }
            Self::ConstructionError { description } => {
                write!(f, "Matrix construction error: {description}")
            }
        }
    }
}

impl std::error::Error for TestMatrixError {}

/// Creates a 2D Laplacian matrix using the 5-point stencil.
///
/// For an `nx` x `ny` grid, produces an `(nx*ny)` x `(nx*ny)` sparse matrix
/// representing the discrete Laplacian operator with Dirichlet boundary conditions.
///
/// The stencil at interior point (i,j) is:
/// ```text
///       -1
///   -1   4  -1
///       -1
/// ```
///
/// # Properties
/// - Symmetric positive definite (SPD)
/// - Eigenvalues: 4 - 2*cos(pi*k/(nx+1)) - 2*cos(pi*l/(ny+1)) for k=1..nx, l=1..ny
/// - Condition number grows as O(max(nx,ny)^2)
/// - nnz = 5*nx*ny - 2*(nx+ny)
///
/// # Errors
/// Returns error if `nx` or `ny` is zero.
pub fn laplacian_2d(nx: usize, ny: usize) -> Result<CsrMatrix<f64>, TestMatrixError> {
    if nx == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "nx".to_string(),
            value: 0,
        });
    }
    if ny == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "ny".to_string(),
            value: 0,
        });
    }

    let n = nx * ny;
    let mut builder = CooMatrixBuilder::new(n, n);

    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;

            // Diagonal: 4
            builder.add(idx, idx, 4.0);

            // Left neighbor: -1
            if i > 0 {
                builder.add(idx, idx - 1, -1.0);
            }

            // Right neighbor: -1
            if i < nx - 1 {
                builder.add(idx, idx + 1, -1.0);
            }

            // Bottom neighbor: -1
            if j > 0 {
                builder.add(idx, idx - nx, -1.0);
            }

            // Top neighbor: -1
            if j < ny - 1 {
                builder.add(idx, idx + nx, -1.0);
            }
        }
    }

    Ok(builder.build().to_csr())
}

/// Creates a 3D Laplacian matrix using the 7-point stencil.
///
/// For an `nx` x `ny` x `nz` grid, produces an `(nx*ny*nz)` x `(nx*ny*nz)` sparse matrix
/// representing the discrete 3D Laplacian operator with Dirichlet boundary conditions.
///
/// The stencil at interior point (i,j,k) is:
/// ```text
/// z-1: -1    y-1: -1    center: -1  6  -1    y+1: -1    z+1: -1
///                                x-1     x+1
/// ```
///
/// # Properties
/// - Symmetric positive definite (SPD)
/// - Diagonal value: 6, off-diagonal values: -1
/// - nnz = 7*nx*ny*nz - 2*(nx*ny + ny*nz + nx*nz)
///
/// # Errors
/// Returns error if `nx`, `ny`, or `nz` is zero.
pub fn laplacian_3d(nx: usize, ny: usize, nz: usize) -> Result<CsrMatrix<f64>, TestMatrixError> {
    if nx == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "nx".to_string(),
            value: 0,
        });
    }
    if ny == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "ny".to_string(),
            value: 0,
        });
    }
    if nz == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "nz".to_string(),
            value: 0,
        });
    }

    let n = nx * ny * nz;
    let nxy = nx * ny;
    let mut builder = CooMatrixBuilder::new(n, n);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = k * nxy + j * nx + i;

                // Diagonal: 6
                builder.add(idx, idx, 6.0);

                // x-direction neighbors
                if i > 0 {
                    builder.add(idx, idx - 1, -1.0);
                }
                if i < nx - 1 {
                    builder.add(idx, idx + 1, -1.0);
                }

                // y-direction neighbors
                if j > 0 {
                    builder.add(idx, idx - nx, -1.0);
                }
                if j < ny - 1 {
                    builder.add(idx, idx + nx, -1.0);
                }

                // z-direction neighbors
                if k > 0 {
                    builder.add(idx, idx - nxy, -1.0);
                }
                if k < nz - 1 {
                    builder.add(idx, idx + nxy, -1.0);
                }
            }
        }
    }

    Ok(builder.build().to_csr())
}

/// Creates a tridiagonal matrix with specified sub-diagonal, diagonal, and super-diagonal values.
///
/// Produces an `n` x `n` matrix:
/// ```text
/// [ diag  sup   0    0  ... ]
/// [ sub   diag  sup  0  ... ]
/// [ 0     sub   diag sup ... ]
/// [ ...                      ]
/// ```
///
/// # Properties
/// - Bandwidth = 1
/// - SPD if `sub == sup` and `diag > 2*|sub|` (diagonal dominance)
/// - nnz = 3n - 2 for n >= 2
///
/// # Errors
/// Returns error if `n` is zero.
pub fn tridiagonal(
    n: usize,
    sub: f64,
    diag: f64,
    sup: f64,
) -> Result<CsrMatrix<f64>, TestMatrixError> {
    if n == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "n".to_string(),
            value: 0,
        });
    }

    let mut builder = CooMatrixBuilder::new(n, n);

    for i in 0..n {
        // Main diagonal
        builder.add(i, i, diag);

        // Sub-diagonal
        if i > 0 {
            builder.add(i, i - 1, sub);
        }

        // Super-diagonal
        if i < n - 1 {
            builder.add(i, i + 1, sup);
        }
    }

    Ok(builder.build().to_csr())
}

/// Creates a diagonal matrix with a constant value on the main diagonal.
///
/// Produces an `n` x `n` matrix with `value` on all diagonal entries and zeros elsewhere.
///
/// # Properties
/// - Symmetric
/// - Positive definite if `value > 0`
/// - All eigenvalues equal `value`
/// - nnz = n
/// - Condition number = 1
///
/// # Errors
/// Returns error if `n` is zero.
pub fn diagonal(n: usize, value: f64) -> Result<CsrMatrix<f64>, TestMatrixError> {
    if n == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "n".to_string(),
            value: 0,
        });
    }

    let mut builder = CooMatrixBuilder::new(n, n);

    for i in 0..n {
        builder.add(i, i, value);
    }

    Ok(builder.build().to_csr())
}

/// Creates an arrow (arrowhead) matrix.
///
/// The arrow matrix has a dense first row, dense first column, and a diagonal:
/// ```text
/// [ d0  a1  a2  a3 ... ]
/// [ a1  d1  0   0  ... ]
/// [ a2  0   d2  0  ... ]
/// [ a3  0   0   d3 ... ]
/// [ ...                 ]
/// ```
///
/// where `d_i = n + 1 - i` (diagonal values) and `a_i = 1` (arrow entries).
///
/// # Properties
/// - Symmetric
/// - Positive definite (with chosen diagonal values)
/// - Tests sparse solvers with dense row/column structure
/// - nnz = 3n - 2 for n >= 2
///
/// # Errors
/// Returns error if `n` is zero.
pub fn arrow_matrix(n: usize) -> Result<CsrMatrix<f64>, TestMatrixError> {
    if n == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "n".to_string(),
            value: 0,
        });
    }

    if n == 1 {
        let mut builder = CooMatrixBuilder::new(1, 1);
        builder.add(0, 0, 1.0);
        return Ok(builder.build().to_csr());
    }

    let mut builder = CooMatrixBuilder::new(n, n);

    // First row and column (arrow entries)
    // Diagonal d0 = n + 1 to ensure diagonal dominance
    builder.add(0, 0, (n + 1) as f64);

    for i in 1..n {
        // First row: a_i = 1
        builder.add(0, i, 1.0);
        // First column: a_i = 1 (symmetric)
        builder.add(i, 0, 1.0);
        // Diagonal: d_i = n + 1 - i to ensure positive definiteness
        // We need d_i > 1/d_0 * sum(...) for SPD, so use generous values
        builder.add(i, i, (n + 1 - i) as f64);
    }

    Ok(builder.build().to_csr())
}

/// Creates a random symmetric positive definite (SPD) matrix with a given density.
///
/// Constructs A = L * L^T + n*I where L is a sparse lower triangular matrix
/// with entries drawn from a simple deterministic pseudo-random pattern.
/// The `n*I` shift guarantees positive definiteness.
///
/// # Arguments
/// - `n` - Matrix dimension
/// - `density` - Target density of non-zeros in [0.0, 1.0]
///
/// # Properties
/// - Symmetric positive definite (guaranteed)
/// - Approximately `density * n * n` non-zeros
///
/// # Errors
/// Returns error if `n` is zero or `density` is not in [0, 1].
pub fn random_spd(n: usize, density: f64) -> Result<CsrMatrix<f64>, TestMatrixError> {
    if n == 0 {
        return Err(TestMatrixError::InvalidDimension {
            param: "n".to_string(),
            value: 0,
        });
    }
    if !(0.0..=1.0).contains(&density) {
        return Err(TestMatrixError::InvalidDensity {
            density: format!("{density}"),
        });
    }

    // Use a simple deterministic hash-based pseudo-random generator
    // to avoid needing the rand crate in production code.
    // We build a lower triangular L, then compute A = L*L^T + n*I.
    let mut builder = CooMatrixBuilder::new(n, n);

    // Add n*I for guaranteed positive definiteness
    for i in 0..n {
        builder.add(i, i, n as f64);
    }

    if density <= 0.0 || n == 1 {
        return Ok(builder.build().to_csr());
    }

    // Target number of lower-triangular non-zeros (excluding diagonal)
    let max_lt_entries = n * (n - 1) / 2;
    let target_lt_nnz = ((max_lt_entries as f64) * density).ceil() as usize;

    // Deterministic pseudo-random entry generation using a simple hash
    let mut seed: u64 = 0x517cc1b727220a95;
    let mut generated = 0usize;
    let max_attempts = max_lt_entries * 3; // prevent infinite loop on high density

    for attempt in 0..max_attempts {
        if generated >= target_lt_nnz {
            break;
        }

        // Simple xorshift-style hash
        seed ^= seed.wrapping_shl(13);
        seed ^= seed.wrapping_shr(7);
        seed ^= seed.wrapping_shl(17);
        seed = seed.wrapping_add(attempt as u64);

        let row = ((seed >> 16) as usize) % n;
        let col = ((seed >> 32) as usize) % n;

        if row > col {
            // Deterministic value in range [0.1, 1.0]
            let val = 0.1 + 0.9 * ((seed & 0xFF) as f64) / 255.0;
            // Add both (row, col) and (col, row) for symmetry
            builder.add(row, col, val);
            builder.add(col, row, val);
            // Also strengthen the diagonal to maintain SPD
            builder.add(row, row, val);
            builder.add(col, col, val);
            generated += 1;
        }
    }

    let mut coo = builder.build();
    coo.sum_duplicates();
    Ok(coo.to_csr())
}

/// Creates the 1D Poisson matrix (second-difference operator).
///
/// Produces an `n` x `n` tridiagonal matrix:
/// ```text
/// [ 2  -1   0   0  ... ]
/// [-1   2  -1   0  ... ]
/// [ 0  -1   2  -1  ... ]
/// [ ...                 ]
/// [ 0   0  ... -1   2  ]
/// ```
///
/// This is the standard discretization of -u''(x) = f(x) on \[0,1\]
/// with uniform mesh spacing h = 1/(n+1), scaled by h^2.
///
/// # Properties
/// - Symmetric positive definite (SPD)
/// - Eigenvalues: 2 - 2*cos(k*pi/(n+1)) for k=1..n
/// - Condition number: O(n^2)
/// - nnz = 3n - 2 for n >= 2
///
/// # Errors
/// Returns error if `n` is zero.
pub fn poisson_1d(n: usize) -> Result<CsrMatrix<f64>, TestMatrixError> {
    tridiagonal(n, -1.0, 2.0, -1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // laplacian_2d tests
    // ========================================================================

    #[test]
    fn test_laplacian_2d_basic() {
        let mat = laplacian_2d(3, 3).expect("Failed to create 2D Laplacian");
        assert_eq!(mat.nrows(), 9);
        assert_eq!(mat.ncols(), 9);
        // Interior points have 4 neighbors, boundary points have 2-3
        // 5-point stencil on 3x3 grid:
        // corners: 2 off-diag + 1 diag = 3 entries each (4 corners)
        // edges: 3 off-diag + 1 diag = 4 entries each (4 edges for 3x3 minus corners = 4)
        // center: 4 off-diag + 1 diag = 5 entries (1 center)
        // Total: 4*3 + 4*4 + 1*5 = 12 + 16 + 5 = 33? No...
        // Actually: nnz = 5*nx*ny - 2*(nx+ny) = 5*9 - 2*6 = 45 - 12 = 33
        // But the formula includes the diagonal, so nnz = 5*3*3 - 2*(3+3) = 33
        assert_eq!(mat.nnz(), 33);
    }

    #[test]
    fn test_laplacian_2d_symmetry() {
        let mat = laplacian_2d(4, 3).expect("Failed to create 2D Laplacian");
        let n = mat.nrows();

        // Check A[i,j] == A[j,i] for all entries
        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-14,
                "Laplacian 2D not symmetric at ({row}, {col}): {val} vs {transpose_val}"
            );
        }
        // Also check dimensions
        assert_eq!(n, 12);
    }

    #[test]
    fn test_laplacian_2d_positive_definiteness() {
        // Verify diagonal dominance (sufficient for SPD)
        let mat = laplacian_2d(5, 5).expect("Failed to create 2D Laplacian");
        let n = mat.nrows();

        for i in 0..n {
            let diag = mat.get_or_zero(i, i);
            let mut off_diag_sum = 0.0;
            for (col, val) in mat.row_iter(i) {
                if col != i {
                    off_diag_sum += val.abs();
                }
            }
            assert!(
                diag >= off_diag_sum,
                "Row {i}: diagonal {diag} < off-diagonal sum {off_diag_sum}"
            );
        }
    }

    #[test]
    fn test_laplacian_2d_1x1() {
        let mat = laplacian_2d(1, 1).expect("Failed to create 1x1 Laplacian");
        assert_eq!(mat.nrows(), 1);
        assert_eq!(mat.ncols(), 1);
        assert_eq!(mat.nnz(), 1);
        assert!((mat.get_or_zero(0, 0) - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_laplacian_2d_zero_dimension() {
        assert!(laplacian_2d(0, 5).is_err());
        assert!(laplacian_2d(5, 0).is_err());
    }

    // ========================================================================
    // laplacian_3d tests
    // ========================================================================

    #[test]
    fn test_laplacian_3d_basic() {
        let mat = laplacian_3d(3, 3, 3).expect("Failed to create 3D Laplacian");
        assert_eq!(mat.nrows(), 27);
        assert_eq!(mat.ncols(), 27);
        // nnz = 7*nx*ny*nz - 2*(nx*ny + ny*nz + nx*nz)
        // = 7*27 - 2*(9 + 9 + 9) = 189 - 54 = 135
        assert_eq!(mat.nnz(), 135);
    }

    #[test]
    fn test_laplacian_3d_symmetry() {
        let mat = laplacian_3d(3, 2, 2).expect("Failed to create 3D Laplacian");

        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-14,
                "Laplacian 3D not symmetric at ({row}, {col}): {val} vs {transpose_val}"
            );
        }
    }

    #[test]
    fn test_laplacian_3d_diagonal_dominance() {
        let mat = laplacian_3d(3, 3, 3).expect("Failed to create 3D Laplacian");
        let n = mat.nrows();

        for i in 0..n {
            let diag = mat.get_or_zero(i, i);
            let mut off_diag_sum = 0.0;
            for (col, val) in mat.row_iter(i) {
                if col != i {
                    off_diag_sum += val.abs();
                }
            }
            assert!(
                diag >= off_diag_sum,
                "Row {i}: diagonal {diag} < off-diagonal sum {off_diag_sum}"
            );
        }
    }

    #[test]
    fn test_laplacian_3d_zero_dimension() {
        assert!(laplacian_3d(0, 3, 3).is_err());
        assert!(laplacian_3d(3, 0, 3).is_err());
        assert!(laplacian_3d(3, 3, 0).is_err());
    }

    // ========================================================================
    // tridiagonal tests
    // ========================================================================

    #[test]
    fn test_tridiagonal_basic() {
        let mat = tridiagonal(5, -1.0, 2.0, -1.0).expect("Failed to create tridiagonal");
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), 5);
        // 3*5 - 2 = 13
        assert_eq!(mat.nnz(), 13);
    }

    #[test]
    fn test_tridiagonal_values() {
        let mat = tridiagonal(4, -1.0, 3.0, -2.0).expect("Failed to create tridiagonal");

        // Check diagonal
        for i in 0..4 {
            assert!((mat.get_or_zero(i, i) - 3.0).abs() < 1e-14);
        }

        // Check sub-diagonal
        for i in 1..4 {
            assert!((mat.get_or_zero(i, i - 1) - (-1.0)).abs() < 1e-14);
        }

        // Check super-diagonal
        for i in 0..3 {
            assert!((mat.get_or_zero(i, i + 1) - (-2.0)).abs() < 1e-14);
        }

        // Check zeros
        assert!((mat.get_or_zero(0, 2)).abs() < 1e-14);
        assert!((mat.get_or_zero(0, 3)).abs() < 1e-14);
        assert!((mat.get_or_zero(3, 0)).abs() < 1e-14);
    }

    #[test]
    fn test_tridiagonal_symmetric() {
        let mat = tridiagonal(5, -1.0, 4.0, -1.0).expect("Failed to create symmetric tridiagonal");

        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-14,
                "Symmetric tridiagonal not symmetric at ({row}, {col})"
            );
        }
    }

    #[test]
    fn test_tridiagonal_size_1() {
        let mat = tridiagonal(1, -1.0, 5.0, -1.0).expect("Failed to create 1x1 tridiagonal");
        assert_eq!(mat.nrows(), 1);
        assert_eq!(mat.nnz(), 1);
        assert!((mat.get_or_zero(0, 0) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_tridiagonal_zero() {
        assert!(tridiagonal(0, -1.0, 2.0, -1.0).is_err());
    }

    // ========================================================================
    // diagonal tests
    // ========================================================================

    #[test]
    fn test_diagonal_basic() {
        let mat = diagonal(5, 3.0).expect("Failed to create diagonal");
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), 5);
        assert_eq!(mat.nnz(), 5);

        for i in 0..5 {
            assert!((mat.get_or_zero(i, i) - 3.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_diagonal_off_diagonal_zeros() {
        let mat = diagonal(4, 2.0).expect("Failed to create diagonal");

        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        mat.get_or_zero(i, j).abs() < 1e-14,
                        "Off-diagonal entry ({i}, {j}) is not zero"
                    );
                }
            }
        }
    }

    #[test]
    fn test_diagonal_symmetry() {
        let mat = diagonal(10, 7.5).expect("Failed to create diagonal");

        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-14,
                "Diagonal matrix not symmetric at ({row}, {col})"
            );
        }
    }

    #[test]
    fn test_diagonal_zero() {
        assert!(diagonal(0, 1.0).is_err());
    }

    // ========================================================================
    // arrow_matrix tests
    // ========================================================================

    #[test]
    fn test_arrow_basic() {
        let mat = arrow_matrix(5).expect("Failed to create arrow matrix");
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), 5);
        // nnz = n (diagonal) + 2*(n-1) (first row/col excluding diagonal) = 3n - 2
        assert_eq!(mat.nnz(), 13);
    }

    #[test]
    fn test_arrow_symmetry() {
        let mat = arrow_matrix(8).expect("Failed to create arrow matrix");

        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-14,
                "Arrow matrix not symmetric at ({row}, {col}): {val} vs {transpose_val}"
            );
        }
    }

    #[test]
    fn test_arrow_structure() {
        let n = 5;
        let mat = arrow_matrix(n).expect("Failed to create arrow matrix");

        // First row should have entries in all columns
        let mut first_row_cols = Vec::new();
        for (col, _) in mat.row_iter(0) {
            first_row_cols.push(col);
        }
        assert_eq!(first_row_cols.len(), n, "First row should have {n} entries");

        // Other rows should have exactly 2 entries (diagonal + first column)
        for i in 1..n {
            let mut row_nnz = 0;
            for _ in mat.row_iter(i) {
                row_nnz += 1;
            }
            assert_eq!(
                row_nnz, 2,
                "Row {i} should have exactly 2 entries, got {row_nnz}"
            );
        }
    }

    #[test]
    fn test_arrow_diagonal_dominance() {
        let mat = arrow_matrix(10).expect("Failed to create arrow matrix");
        let n = mat.nrows();

        for i in 0..n {
            let diag = mat.get_or_zero(i, i);
            let mut off_diag_sum = 0.0;
            for (col, val) in mat.row_iter(i) {
                if col != i {
                    off_diag_sum += val.abs();
                }
            }
            assert!(
                diag >= off_diag_sum,
                "Arrow row {i}: diagonal {diag} < off-diagonal sum {off_diag_sum}"
            );
        }
    }

    #[test]
    fn test_arrow_size_1() {
        let mat = arrow_matrix(1).expect("Failed to create 1x1 arrow");
        assert_eq!(mat.nrows(), 1);
        assert_eq!(mat.nnz(), 1);
    }

    #[test]
    fn test_arrow_zero() {
        assert!(arrow_matrix(0).is_err());
    }

    // ========================================================================
    // random_spd tests
    // ========================================================================

    #[test]
    fn test_random_spd_basic() {
        let mat = random_spd(10, 0.3).expect("Failed to create random SPD");
        assert_eq!(mat.nrows(), 10);
        assert_eq!(mat.ncols(), 10);
        assert!(mat.nnz() >= 10, "Should have at least n entries (diagonal)");
    }

    #[test]
    fn test_random_spd_symmetry() {
        let mat = random_spd(20, 0.2).expect("Failed to create random SPD");

        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-10,
                "Random SPD not symmetric at ({row}, {col}): {val} vs {transpose_val}"
            );
        }
    }

    #[test]
    fn test_random_spd_positive_diagonal() {
        let mat = random_spd(15, 0.4).expect("Failed to create random SPD");

        for i in 0..mat.nrows() {
            let diag = mat.get_or_zero(i, i);
            assert!(
                diag > 0.0,
                "Random SPD diagonal at {i} should be positive, got {diag}"
            );
        }
    }

    #[test]
    fn test_random_spd_zero_density() {
        let mat = random_spd(5, 0.0).expect("Failed to create diagonal SPD");
        assert_eq!(mat.nnz(), 5, "Zero density should yield diagonal matrix");
    }

    #[test]
    fn test_random_spd_invalid() {
        assert!(random_spd(0, 0.5).is_err());
        assert!(random_spd(5, -0.1).is_err());
        assert!(random_spd(5, 1.1).is_err());
    }

    // ========================================================================
    // poisson_1d tests
    // ========================================================================

    #[test]
    fn test_poisson_1d_basic() {
        let mat = poisson_1d(5).expect("Failed to create 1D Poisson");
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), 5);
        assert_eq!(mat.nnz(), 13); // 3*5 - 2
    }

    #[test]
    fn test_poisson_1d_values() {
        let mat = poisson_1d(4).expect("Failed to create 1D Poisson");

        // Check diagonal = 2
        for i in 0..4 {
            assert!((mat.get_or_zero(i, i) - 2.0).abs() < 1e-14);
        }

        // Check off-diagonal = -1
        for i in 0..3 {
            assert!((mat.get_or_zero(i, i + 1) - (-1.0)).abs() < 1e-14);
            assert!((mat.get_or_zero(i + 1, i) - (-1.0)).abs() < 1e-14);
        }
    }

    #[test]
    fn test_poisson_1d_symmetry() {
        let mat = poisson_1d(10).expect("Failed to create 1D Poisson");

        for (row, col, &val) in mat.iter() {
            let transpose_val = mat.get_or_zero(col, row);
            assert!(
                (val - transpose_val).abs() < 1e-14,
                "Poisson 1D not symmetric at ({row}, {col})"
            );
        }
    }

    #[test]
    fn test_poisson_1d_spd() {
        // Check diagonal dominance (sufficient for SPD)
        let mat = poisson_1d(10).expect("Failed to create 1D Poisson");

        for i in 0..mat.nrows() {
            let diag = mat.get_or_zero(i, i);
            let mut off_diag_sum = 0.0;
            for (col, val) in mat.row_iter(i) {
                if col != i {
                    off_diag_sum += val.abs();
                }
            }
            assert!(
                diag >= off_diag_sum,
                "Poisson 1D row {i}: diagonal {diag} < off-diagonal sum {off_diag_sum}"
            );
        }
    }

    #[test]
    fn test_poisson_1d_zero() {
        assert!(poisson_1d(0).is_err());
    }

    // ========================================================================
    // Cross-generator property tests
    // ========================================================================

    #[test]
    fn test_poisson_1d_equals_tridiagonal() {
        let poisson = poisson_1d(10).expect("poisson_1d");
        let tri = tridiagonal(10, -1.0, 2.0, -1.0).expect("tridiagonal");

        assert_eq!(poisson.nnz(), tri.nnz());
        assert_eq!(poisson.nrows(), tri.nrows());

        for i in 0..10 {
            for j in 0..10 {
                let pval = poisson.get_or_zero(i, j);
                let tval = tri.get_or_zero(i, j);
                assert!(
                    (pval - tval).abs() < 1e-14,
                    "Poisson != tridiag at ({i}, {j}): {pval} vs {tval}"
                );
            }
        }
    }

    #[test]
    fn test_laplacian_2d_1d_consistency() {
        // laplacian_2d(n, 1) should produce a similar structure to a 1D Laplacian
        // (tridiagonal with diagonal=4, off-diagonal=-1) since there's only 1 row in y
        let mat = laplacian_2d(5, 1).expect("laplacian_2d(5,1)");
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), 5);
        // Should be tridiagonal-like with diagonal=4
        for i in 0..5 {
            assert!((mat.get_or_zero(i, i) - 4.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_generators_produce_valid_csr() {
        // All generators should produce valid CSR matrices
        let matrices: Vec<CsrMatrix<f64>> = vec![
            laplacian_2d(4, 4).expect("lap2d"),
            laplacian_3d(3, 3, 3).expect("lap3d"),
            tridiagonal(10, -1.0, 2.0, -1.0).expect("tridiag"),
            diagonal(10, 5.0).expect("diag"),
            arrow_matrix(10).expect("arrow"),
            random_spd(10, 0.3).expect("rspd"),
            poisson_1d(10).expect("poisson"),
        ];

        for mat in &matrices {
            // Row pointers should be monotonically increasing
            let row_ptrs = mat.row_ptrs();
            for i in 1..row_ptrs.len() {
                assert!(row_ptrs[i] >= row_ptrs[i - 1], "Row pointers not monotonic");
            }

            // Column indices should be in bounds
            for &col in mat.col_indices() {
                assert!(col < mat.ncols(), "Column index out of bounds");
            }

            // nnz should match row_ptrs
            assert_eq!(
                mat.nnz(),
                row_ptrs[mat.nrows()],
                "nnz mismatch with row_ptrs"
            );
        }
    }
}
