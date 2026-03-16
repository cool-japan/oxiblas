//! SuiteSparse Matrix Collection test matrices
//!
//! Provides standard test matrices from the Florida/SuiteSparse Matrix Collection
//! for benchmarking and validation of sparse operations.

// Index-based loops are clearer for matrix coordinate operations
#![allow(clippy::needless_range_loop)]

use oxiblas_sparse::ops::spmm_sparse;
use oxiblas_sparse::ops::spmv;
use oxiblas_sparse::{CooMatrixBuilder, CsrMatrix};

/// Generates a 2D 5-point Laplacian stencil (standard test matrix)
///
/// Creates an (nx*ny) × (nx*ny) sparse matrix representing the discrete Laplacian
/// operator on a 2D grid with nx×ny points using a 5-point stencil.
///
/// Matrix structure: tridiagonal with -4 on diagonal, 1 on off-diagonals
/// Total non-zeros: O(5 * nx * ny)
pub fn laplacian_2d_5pt(nx: usize, ny: usize) -> CsrMatrix<f64> {
    let n = nx * ny;
    let mut builder = CooMatrixBuilder::new(n, n);

    for i in 0..nx {
        for j in 0..ny {
            let idx = i * ny + j;

            // Diagonal element
            builder.add(idx, idx, 4.0);

            // Left neighbor
            if j > 0 {
                builder.add(idx, idx - 1, -1.0);
            }

            // Right neighbor
            if j < ny - 1 {
                builder.add(idx, idx + 1, -1.0);
            }

            // Bottom neighbor
            if i > 0 {
                builder.add(idx, idx - ny, -1.0);
            }

            // Top neighbor
            if i < nx - 1 {
                builder.add(idx, idx + ny, -1.0);
            }
        }
    }

    builder.build().to_csr()
}

/// Generates a 3D 7-point Laplacian stencil
///
/// Creates an (nx*ny*nz) × (nx*ny*nz) sparse matrix for the 3D Laplacian.
/// Matrix has -6 on diagonal, 1 on off-diagonals.
pub fn laplacian_3d_7pt(nx: usize, ny: usize, nz: usize) -> CsrMatrix<f64> {
    let n = nx * ny * nz;
    let mut builder = CooMatrixBuilder::new(n, n);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let idx = (i * ny + j) * nz + k;

                // Diagonal element
                builder.add(idx, idx, 6.0);

                // x-direction neighbors
                if i > 0 {
                    builder.add(idx, idx - ny * nz, -1.0);
                }
                if i < nx - 1 {
                    builder.add(idx, idx + ny * nz, -1.0);
                }

                // y-direction neighbors
                if j > 0 {
                    builder.add(idx, idx - nz, -1.0);
                }
                if j < ny - 1 {
                    builder.add(idx, idx + nz, -1.0);
                }

                // z-direction neighbors
                if k > 0 {
                    builder.add(idx, idx - 1, -1.0);
                }
                if k < nz - 1 {
                    builder.add(idx, idx + 1, -1.0);
                }
            }
        }
    }

    builder.build().to_csr()
}

/// Generates a random SPD (symmetric positive definite) matrix
///
/// Uses the formula A = B^T * B + λI where B is random and λ ensures positive definiteness.
pub fn random_spd_matrix(n: usize, density: f64, lambda: f64) -> CsrMatrix<f64> {
    use rand::RngExt;
    let mut rng = rand::rng();

    let nnz_per_row = (density * n as f64) as usize;
    let mut builder = CooMatrixBuilder::new(n, n);

    // Generate random lower triangular matrix B
    // Ensure at least one non-zero per row for positive definiteness
    for i in 0..n {
        let num_entries = nnz_per_row.max(1).min(i + 1);
        for _ in 0..num_entries {
            let j = rng.random_range(0..=i);
            let val: f64 = rng.random_range(-1.0..1.0);
            builder.add(i, j, val);
        }
    }

    let b = builder.build().to_csr();

    // Compute A = B^T * B
    let bt = b.transpose();
    let a = spmm_sparse(&bt, &b);

    // Convert to COO to easily add λI
    let mut coo_builder = CooMatrixBuilder::new(n, n);

    // Copy existing entries
    for (row, col, &val) in a.iter() {
        coo_builder.add(row, col, val);
    }

    // Add λI to diagonal (will sum with existing diagonal entries)
    for i in 0..n {
        coo_builder.add(i, i, lambda);
    }

    coo_builder.build().to_csr()
}

/// Generates a tridiagonal matrix (common in 1D discretizations)
pub fn tridiagonal_matrix(n: usize, a: f64, b: f64, c: f64) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    for i in 0..n {
        if i > 0 {
            builder.add(i, i - 1, a);
        }
        builder.add(i, i, b);
        if i < n - 1 {
            builder.add(i, i + 1, c);
        }
    }

    builder.build().to_csr()
}

/// Generates a sparse banded matrix with given bandwidth
pub fn banded_matrix(
    n: usize,
    bandwidth: usize,
    diag_value: f64,
    off_diag_value: f64,
) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);

    for i in 0..n {
        builder.add(i, i, diag_value);

        for k in 1..=bandwidth {
            if i >= k {
                builder.add(i, i - k, off_diag_value);
            }
            if i + k < n {
                builder.add(i, i + k, off_diag_value);
            }
        }
    }

    builder.build().to_csr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_2d_5pt() {
        let a = laplacian_2d_5pt(10, 10);
        assert_eq!(a.nrows(), 100);
        assert_eq!(a.ncols(), 100);

        // Check sparsity pattern (should have ~5 non-zeros per row on average)
        let nnz = a.nnz();
        assert!(
            (400..=500).contains(&nnz),
            "Expected ~460 non-zeros, got {}",
            nnz
        );

        // Verify it's symmetric
        let at = a.transpose();
        for i in 0..a.nrows() {
            for j in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
                let col = a.col_indices()[j];
                let val = a.values()[j];

                // Find corresponding entry in transpose
                let mut found = false;
                for k in at.row_ptrs()[col]..at.row_ptrs()[col + 1] {
                    if at.col_indices()[k] == i {
                        assert!((at.values()[k] - val).abs() < 1e-14);
                        found = true;
                        break;
                    }
                }
                assert!(found, "Matrix not symmetric at ({}, {})", i, col);
            }
        }
    }

    #[test]
    fn test_laplacian_3d_7pt() {
        let a = laplacian_3d_7pt(5, 5, 5);
        assert_eq!(a.nrows(), 125);
        assert_eq!(a.ncols(), 125);

        // Check non-zero count (should have ~7 per row on average)
        let nnz = a.nnz();
        assert!(
            (700..=900).contains(&nnz),
            "Expected ~819 non-zeros, got {}",
            nnz
        );
    }

    #[test]
    fn test_random_spd_matrix() {
        let a = random_spd_matrix(50, 0.1, 1.0);
        assert_eq!(a.nrows(), 50);
        assert_eq!(a.ncols(), 50);

        // Verify positive diagonals (necessary for SPD)
        for i in 0..a.nrows() {
            let mut diag_val = 0.0;

            for j in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
                let col = a.col_indices()[j];
                if col == i {
                    diag_val = a.values()[j];
                    break;
                }
            }

            // SPD matrices from A = B^T*B + λI should have positive diagonals
            assert!(diag_val > 0.0, "Diagonal must be positive for SPD matrix");
        }
    }

    #[test]
    fn test_tridiagonal_matrix() {
        let a = tridiagonal_matrix(100, -1.0, 2.0, -1.0);
        assert_eq!(a.nrows(), 100);
        assert_eq!(a.ncols(), 100);

        // Check non-zero count: first row (2) + last row (2) + interior rows (98 * 3) = 298
        let expected_nnz = 2 + 2 + 3 * 98;
        assert_eq!(a.nnz(), expected_nnz);

        // Verify structure
        for i in 0..a.nrows() {
            let row_start = a.row_ptrs()[i];
            let row_end = a.row_ptrs()[i + 1];
            let row_nnz = row_end - row_start;

            if i == 0 || i == a.nrows() - 1 {
                assert_eq!(row_nnz, 2, "Corners should have 2 non-zeros");
            } else {
                assert_eq!(row_nnz, 3, "Interior rows should have 3 non-zeros");
            }
        }
    }

    #[test]
    fn test_banded_matrix() {
        let a = banded_matrix(50, 3, 4.0, -1.0);
        assert_eq!(a.nrows(), 50);
        assert_eq!(a.ncols(), 50);

        // Check bandwidth
        for i in 0..a.nrows() {
            for j in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
                let col = a.col_indices()[j];
                let dist = (i as i32 - col as i32).unsigned_abs() as usize;
                assert!(dist <= 3, "Non-zero at ({}, {}) exceeds bandwidth", i, col);
            }
        }
    }

    #[test]
    fn test_matrix_vector_product_laplacian() {
        let a = laplacian_2d_5pt(8, 8);
        let n = a.nrows();
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        spmv(1.0, &a, &x, 0.0, &mut y);

        // For constant vector, Laplacian * 1 should give boundary effects
        // Interior points: 4 - 4 = 0 (approximately)
        // Boundary points: will be non-zero

        let interior_start = 8 + 1;
        let interior_end = n - 8 - 1;

        // Check that interior points are near zero
        for i in interior_start..interior_end {
            if (i % 8) != 0 && (i % 8) != 7 {
                // Not on left or right boundary
                assert!(
                    y[i].abs() < 1e-10,
                    "Interior point {} should be ~0, got {}",
                    i,
                    y[i]
                );
            }
        }
    }
}
