//! LU Decomposition.
//!
//! LU decomposition factors a matrix A into the product of a lower triangular
//! matrix L and an upper triangular matrix U, with optional row/column permutation:
//!
//! - Partial pivoting: PA = LU (row permutations only)
//! - Rook pivoting: PAQ = LU (alternating row/column searches)
//! - Full pivoting: PAQ = LU (both row and column permutations)
//! - Band matrices: Specialized algorithms for banded matrices
//!
//! # Pivoting Strategies
//!
//! - **Partial pivoting** (default): Permutes rows to select the largest pivot
//!   in the current column. This is numerically stable for most matrices.
//!
//! - **Rook pivoting**: Alternates between row and column searches to find a
//!   locally maximal pivot. Provides better stability than partial pivoting
//!   with typically lower cost than full pivoting.
//!
//! - **Full pivoting**: Permutes both rows and columns to select the largest
//!   pivot in the remaining submatrix. Most stable but slower.
//!
//! # Band Matrices
//!
//! For band matrices with limited bandwidth, specialized algorithms provide
//! significant performance improvements: O(n·k²) vs O(n³) for general matrices,
//! where k is the bandwidth.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::lu::{Lu, LuRook, LuFullPiv, BandLu, dense_to_band};
//! use oxiblas_matrix::Mat;
//!
//! let a: Mat<f64> = Mat::from_rows(&[
//!     &[2.0, 1.0, 1.0],
//!     &[4.0, 3.0, 3.0],
//!     &[8.0, 7.0, 9.0],
//! ]);
//!
//! // Partial pivoting (faster, usually sufficient)
//! let lu = Lu::compute(a.as_ref()).expect("Matrix is singular");
//! let det = lu.determinant();
//!
//! // Rook pivoting (good balance of stability and speed)
//! let lu_rook = LuRook::compute(a.as_ref()).expect("Matrix is singular");
//! let det_rook = lu_rook.determinant();
//!
//! // Full pivoting (most stable for ill-conditioned matrices)
//! let lu_full = LuFullPiv::compute(a.as_ref()).expect("Matrix is singular");
//! let det_full = lu_full.determinant();
//!
//! // Band matrix example (tridiagonal)
//! let a_band: Vec<f64> = vec![
//!     4.0, -1.0, 0.0,
//!     -1.0, 4.0, -1.0,
//!     0.0, -1.0, 4.0,
//! ];
//! let ab = dense_to_band(&a_band, 3, 1, 1);
//! let lu_band = BandLu::compute(3, 1, 1, &ab).expect("Matrix is singular");
//! ```

mod band;
mod full_piv;
mod partial_piv;
mod rook_piv;

pub use band::{BandLu, BandLuError, band_norm_1, band_norm_inf, band_to_dense, dense_to_band};
pub use full_piv::{LuFullPiv, LuFullPivError};
pub use partial_piv::{Lu, LuError};
pub use rook_piv::{LuRook, LuRookError, RookPivotStats};

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_lu_simple() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 3.0], &[6.0, 3.0]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");

        // det(A) = 4*3 - 3*6 = 12 - 18 = -6
        let det = lu.determinant();
        assert!((det + 6.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_lu_solve() {
        // A = [2 1; 4 3]
        // b = [3; 7]
        // x = [1; 1] (since 2*1 + 1*1 = 3, 4*1 + 3*1 = 7)
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0], &[4.0, 3.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[3.0], &[7.0]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-10, "x[1] = {}", x[(1, 0)]);
    }

    #[test]
    fn test_lu_singular() {
        // Singular matrix (second row is 2x first row)
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);

        let result = Lu::compute(a.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_3x3() {
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0, 1.0], &[4.0, 3.0, 3.0], &[8.0, 7.0, 9.0]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");

        // Test solve: Ax = b where b = [4, 10, 24]
        // Solution should be x = [1, 1, 1]
        let b: Mat<f64> = Mat::from_rows(&[&[4.0], &[10.0], &[24.0]]);
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-10, "x[1] = {}", x[(1, 0)]);
        assert!((x[(2, 0)] - 1.0).abs() < 1e-10, "x[2] = {}", x[(2, 0)]);
    }

    #[test]
    fn test_lu_determinant() {
        // A = [1 2 3; 4 5 6; 7 8 10]
        // det = 1*(5*10 - 6*8) - 2*(4*10 - 6*7) + 3*(4*8 - 5*7)
        //     = 1*(50 - 48) - 2*(40 - 42) + 3*(32 - 35)
        //     = 1*2 - 2*(-2) + 3*(-3)
        //     = 2 + 4 - 9 = -3
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let det = lu.determinant();

        assert!((det + 3.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_lu_inverse() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let a_inv = lu.inverse().expect("Should invert");

        // A * A^-1 should be identity
        // det(A) = 24 - 14 = 10
        // A^-1 = [6/10 -7/10; -2/10 4/10] = [0.6 -0.7; -0.2 0.4]
        assert!((a_inv[(0, 0)] - 0.6).abs() < 1e-10);
        assert!((a_inv[(0, 1)] + 0.7).abs() < 1e-10);
        assert!((a_inv[(1, 0)] + 0.2).abs() < 1e-10);
        assert!((a_inv[(1, 1)] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_lu_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[2.0f32, 1.0], &[4.0, 3.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[3.0f32], &[7.0]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-5, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-5, "x[1] = {}", x[(1, 0)]);
    }

    #[test]
    fn test_lu_complex64() {
        use num_complex::Complex64;

        // Complex matrix A = [[2+i, 1], [1, 3-i]]
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 1.0), Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(3.0, -1.0)],
        ]);

        // b = [3+i, 4]
        let b: Mat<Complex64> =
            Mat::from_rows(&[&[Complex64::new(3.0, 1.0)], &[Complex64::new(4.0, 0.0)]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];

        assert!(
            (ax0 - b[(0, 0)]).norm() < 1e-10,
            "ax0 = {:?}, b0 = {:?}",
            ax0,
            b[(0, 0)]
        );
        assert!(
            (ax1 - b[(1, 0)]).norm() < 1e-10,
            "ax1 = {:?}, b1 = {:?}",
            ax1,
            b[(1, 0)]
        );
    }

    #[test]
    fn test_lu_complex64_determinant() {
        use num_complex::Complex64;

        // A = [[1+i, 2], [3, 4-i]]
        // det = (1+i)(4-i) - 2*3 = 4 - i + 4i - i² - 6 = 4 + 3i + 1 - 6 = -1 + 3i
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, -1.0)],
        ]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let det = lu.determinant();

        // det should be -1 + 3i
        assert!((det.re + 1.0).abs() < 1e-10, "det.re = {}", det.re);
        assert!((det.im - 3.0).abs() < 1e-10, "det.im = {}", det.im);
    }

    #[test]
    fn test_lu_complex64_inverse() {
        use num_complex::Complex64;

        // A = [[2, i], [-i, 2]]  (Hermitian matrix)
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            &[Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let a_inv = lu.inverse().expect("Should invert");

        // A * A^-1 should be identity
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + a[(i, k)] * a_inv[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (sum - expected).norm() < 1e-10,
                    "A*A^-1[{},{}] = {:?}",
                    i,
                    j,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_lu_complex32() {
        use num_complex::Complex32;

        // Simple complex solve test
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(4.0, 0.0), Complex32::new(1.0, 1.0)],
            &[Complex32::new(1.0, -1.0), Complex32::new(3.0, 0.0)],
        ]);

        let b: Mat<Complex32> =
            Mat::from_rows(&[&[Complex32::new(5.0, 1.0)], &[Complex32::new(4.0, -1.0)]]);

        let lu = Lu::compute(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];

        assert!(
            (ax0 - b[(0, 0)]).norm() < 1e-5,
            "ax0 = {:?}, b0 = {:?}",
            ax0,
            b[(0, 0)]
        );
        assert!(
            (ax1 - b[(1, 0)]).norm() < 1e-5,
            "ax1 = {:?}, b1 = {:?}",
            ax1,
            b[(1, 0)]
        );
    }

    #[test]
    fn test_lu_blocked_small() {
        // Test blocked algorithm on small matrix (should still work correctly)
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0], &[4.0, 3.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[3.0], &[7.0]]);

        let lu = Lu::compute_blocked(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-10, "x[1] = {}", x[(1, 0)]);
    }

    #[test]
    fn test_lu_blocked_large() {
        // Test blocked algorithm on larger matrix (100x100)
        let n = 100;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        // Create a diagonally dominant matrix
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    a[(i, j)] = (n as f64) + 1.0;
                } else {
                    a[(i, j)] = 0.5;
                }
            }
        }

        // Create RHS such that solution is all ones
        let mut b: Mat<f64> = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[(i, j)];
            }
            b[(i, 0)] = sum;
        }

        let lu = Lu::compute_blocked(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        // Verify solution
        for i in 0..n {
            assert!(
                (x[(i, 0)] - 1.0).abs() < 1e-8,
                "x[{}] = {}, expected 1.0",
                i,
                x[(i, 0)]
            );
        }
    }

    #[test]
    fn test_lu_blocked_determinant() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let lu = Lu::compute_blocked(a.as_ref()).expect("Should not be singular");
        let det = lu.determinant();

        // Expected det = -3
        assert!((det + 3.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_lu_blocked_inverse() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);

        let lu = Lu::compute_blocked(a.as_ref()).expect("Should not be singular");
        let a_inv = lu.inverse().expect("Should invert");

        // Verify A * A^-1 = I
        assert!((a_inv[(0, 0)] - 0.6).abs() < 1e-10);
        assert!((a_inv[(0, 1)] + 0.7).abs() < 1e-10);
        assert!((a_inv[(1, 0)] + 0.2).abs() < 1e-10);
        assert!((a_inv[(1, 1)] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_lu_blocked_vs_unblocked() {
        // Verify blocked and unblocked give same results
        let n = 100;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        // Create a random-ish matrix
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 / 100.0;
                if i == j {
                    a[(i, j)] += 10.0; // Make diagonally dominant
                }
            }
        }

        let lu_unblocked = Lu::compute(a.as_ref()).expect("Unblocked should work");
        let lu_blocked = Lu::compute_blocked(a.as_ref()).expect("Blocked should work");

        let det_unblocked = lu_unblocked.determinant();
        let det_blocked = lu_blocked.determinant();

        // Determinants should match
        let rel_error = ((det_unblocked - det_blocked) / det_unblocked).abs();
        assert!(
            rel_error < 1e-10,
            "det_unblocked = {}, det_blocked = {}, rel_error = {}",
            det_unblocked,
            det_blocked,
            rel_error
        );
    }

    #[test]
    fn test_lu_blocked_with_block_size() {
        // Test with custom block size
        let n = 50;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i + 1) * (j + 1)) as f64 * 0.01;
                if i == j {
                    a[(i, j)] += 5.0;
                }
            }
        }

        let lu_16 = Lu::compute_with_block_size(a.as_ref(), 16).expect("nb=16 should work");
        let lu_32 = Lu::compute_with_block_size(a.as_ref(), 32).expect("nb=32 should work");

        let det_16 = lu_16.determinant();
        let det_32 = lu_32.determinant();

        let rel_error = ((det_16 - det_32) / det_16).abs();
        assert!(
            rel_error < 1e-10,
            "Different block sizes should give same result: det_16 = {}, det_32 = {}",
            det_16,
            det_32
        );
    }
}
