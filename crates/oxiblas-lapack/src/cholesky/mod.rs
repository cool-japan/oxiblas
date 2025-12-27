//! Cholesky Decomposition and Symmetric Factorizations.
//!
//! This module provides symmetric matrix factorizations:
//!
//! - **LL^T (Cholesky)**: For symmetric positive definite matrices.
//!   A = LL^T where L is lower triangular with positive diagonal.
//!
//! - **LDL^T**: For symmetric matrices (may be indefinite).
//!   A = LDL^T where L is unit lower triangular and D is diagonal.
//!
//! - **Bunch-Kaufman**: For symmetric indefinite matrices with pivoting.
//!   A = P*L*D*L^T*P^T where D has 1×1 and 2×2 blocks.
//!
//! - **Aasen's method**: For symmetric indefinite matrices.
//!   A = P*L*T*L^T*P^T where T is symmetric tridiagonal.
//!
//! - **Band Cholesky**: Efficient factorization for SPD band matrices.
//!   O(n·k²) complexity where k is the bandwidth.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::cholesky::{Cholesky, Ldlt, BunchKaufman, Aasen, BandCholesky, dense_to_band_lower};
//! use oxiblas_matrix::Mat;
//!
//! // Symmetric positive definite matrix
//! let a: Mat<f64> = Mat::from_rows(&[
//!     &[4.0, 2.0],
//!     &[2.0, 5.0],
//! ]);
//!
//! // LL^T (requires positive definiteness)
//! let chol = Cholesky::compute(a.as_ref()).expect("Must be positive definite");
//! let det = chol.determinant();
//!
//! // LDL^T (works for indefinite matrices too)
//! let ldlt = Ldlt::compute(a.as_ref()).expect("Must be non-singular");
//! assert!(ldlt.is_positive_definite());
//!
//! // Bunch-Kaufman for symmetric indefinite with pivoting
//! let b: Mat<f64> = Mat::from_rows(&[
//!     &[1.0, 2.0],
//!     &[2.0, 1.0],  // Indefinite matrix
//! ]);
//! let bk = BunchKaufman::compute(b.as_ref()).expect("Pivoted factorization");
//!
//! // Aasen's method for symmetric indefinite (tridiagonal T)
//! let aasen = Aasen::compute(b.as_ref()).expect("Aasen factorization");
//!
//! // Band Cholesky for SPD band matrices
//! let a_band: Vec<f64> = vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0];
//! let ab = dense_to_band_lower(&a_band, 3, 1);
//! let band_chol = BandCholesky::compute(3, 1, &ab).expect("Must be SPD");
//! ```

mod aasen;
mod band;
mod bunch_kaufman;
mod hermitian;
mod ldlt;
mod llt;
mod packed;

pub use aasen::{Aasen, AasenError, aasen};
pub use band::{BandCholesky, BandCholeskyError, band_lower_to_dense, dense_to_band_lower};
pub use bunch_kaufman::{BunchKaufman, BunchKaufmanError, Uplo as BunchKaufmanUplo};
pub use hermitian::{HermitianCholesky, HermitianCholeskyError};
pub use ldlt::{Ldlt, LdltError};
pub use llt::{Cholesky, CholeskyError};
pub use packed::{
    PackedCholesky, PackedCholeskyError, PackedLdlt, PackedLdltError, Uplo, dense_to_packed_lower,
    dense_to_packed_upper, packed_lower_index, packed_lower_to_dense, packed_upper_index,
    packed_upper_to_dense, ppsv, spsv,
};

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_cholesky_simple() {
        // A = [4 2; 2 5] is symmetric positive definite
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let chol = Cholesky::compute(a.as_ref()).expect("Should be positive definite");

        // det(A) = 4*5 - 2*2 = 16
        let det = chol.determinant();
        assert!((det - 16.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_cholesky_solve() {
        // A = [4 2; 2 5]
        // b = [8; 11]
        // Ax = b => x = [1; 1.8] since 4*1 + 2*1.8 = 7.6 ≠ 8
        // Let's verify: 4x + 2y = 8, 2x + 5y = 11
        // x = 2 - 0.5y, substitute: 2(2-0.5y) + 5y = 11 => 4 - y + 5y = 11 => 4y = 7 => y = 1.75
        // x = 2 - 0.875 = 1.125
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let chol = Cholesky::compute(a.as_ref()).expect("Should be positive definite");
        let x = chol.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // A = [1 2; 2 1] has eigenvalues -1, 3, so not positive definite
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);

        let result = Cholesky::compute(a.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_cholesky_3x3() {
        // A = [4 12 -16; 12 37 -43; -16 -43 98]
        // This is a classic test matrix that factors as:
        // L = [2 0 0; 6 1 0; -8 5 3]
        let a: Mat<f64> = Mat::from_rows(&[
            &[4.0, 12.0, -16.0],
            &[12.0, 37.0, -43.0],
            &[-16.0, -43.0, 98.0],
        ]);

        let chol = Cholesky::compute(a.as_ref()).expect("Should be positive definite");
        let l = chol.l_factor();

        // Check L
        assert!((l[(0, 0)] - 2.0).abs() < 1e-10, "L[0,0] = {}", l[(0, 0)]);
        assert!((l[(1, 0)] - 6.0).abs() < 1e-10, "L[1,0] = {}", l[(1, 0)]);
        assert!((l[(1, 1)] - 1.0).abs() < 1e-10, "L[1,1] = {}", l[(1, 1)]);
        assert!((l[(2, 0)] + 8.0).abs() < 1e-10, "L[2,0] = {}", l[(2, 0)]);
        assert!((l[(2, 1)] - 5.0).abs() < 1e-10, "L[2,1] = {}", l[(2, 1)]);
        assert!((l[(2, 2)] - 3.0).abs() < 1e-10, "L[2,2] = {}", l[(2, 2)]);
    }

    #[test]
    fn test_cholesky_identity() {
        let a: Mat<f64> = Mat::eye(3);

        let chol = Cholesky::compute(a.as_ref()).expect("Identity is positive definite");
        let l = chol.l_factor();

        // L should also be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (l[(i, j)] - expected).abs() < 1e-10,
                    "L[{},{}] = {}",
                    i,
                    j,
                    l[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cholesky_inverse() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let chol = Cholesky::compute(a.as_ref()).expect("Should be positive definite");
        let a_inv = chol.inverse().expect("Should invert");

        // Verify A * A^-1 = I (approximately)
        // det(A) = 16
        // A^-1 = [5/16 -2/16; -2/16 4/16] = [0.3125 -0.125; -0.125 0.25]
        assert!((a_inv[(0, 0)] - 0.3125).abs() < 1e-10);
        assert!((a_inv[(0, 1)] + 0.125).abs() < 1e-10);
        assert!((a_inv[(1, 0)] + 0.125).abs() < 1e-10);
        assert!((a_inv[(1, 1)] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[4.0f32, 2.0], &[2.0, 5.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[8.0f32], &[11.0]]);

        let chol = Cholesky::compute(a.as_ref()).expect("Should be positive definite");
        let x = chol.solve(b.as_ref()).expect("Should solve");

        // Verify Ax = b
        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-5, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-5, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_cholesky_blocked_small() {
        // Test blocked algorithm on small matrix
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[8.0], &[11.0]]);

        let chol = Cholesky::compute_blocked(a.as_ref()).expect("Should be positive definite");
        let x = chol.solve(b.as_ref()).expect("Should solve");

        let ax0 = 4.0 * x[(0, 0)] + 2.0 * x[(1, 0)];
        let ax1 = 2.0 * x[(0, 0)] + 5.0 * x[(1, 0)];
        assert!((ax0 - 8.0).abs() < 1e-10, "Ax[0] = {}", ax0);
        assert!((ax1 - 11.0).abs() < 1e-10, "Ax[1] = {}", ax1);
    }

    #[test]
    fn test_cholesky_blocked_large() {
        // Test blocked algorithm on larger matrix (100x100)
        let n = 100;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        // Create SPD matrix: A = B^T B + n*I
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 / 100.0;
            }
        }

        // Make it SPD: A = A^T * A + n*I
        let mut spd: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[(k, i)] * a[(k, j)];
                }
                spd[(i, j)] = sum;
                if i == j {
                    spd[(i, j)] += n as f64; // Ensure positive definiteness
                }
            }
        }

        // Create RHS such that solution is all ones
        let mut b: Mat<f64> = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += spd[(i, j)];
            }
            b[(i, 0)] = sum;
        }

        let chol = Cholesky::compute_blocked(spd.as_ref()).expect("Should be positive definite");
        let x = chol.solve(b.as_ref()).expect("Should solve");

        // Verify solution (relaxed tolerance for larger matrix)
        for i in 0..n {
            assert!(
                (x[(i, 0)] - 1.0).abs() < 1e-6,
                "x[{}] = {}, expected 1.0",
                i,
                x[(i, 0)]
            );
        }
    }

    #[test]
    fn test_cholesky_blocked_determinant() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let chol = Cholesky::compute_blocked(a.as_ref()).expect("Should be positive definite");
        let det = chol.determinant();

        // det = 4*5 - 2*2 = 16
        assert!((det - 16.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_cholesky_blocked_vs_unblocked() {
        // Verify blocked and unblocked give same results
        let n = 100;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        // Create SPD matrix
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = if i == j {
                    (n as f64) + 1.0
                } else {
                    ((i + j) % 10) as f64 * 0.1
                };
            }
        }

        // Make symmetric
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (a[(i, j)] + a[(j, i)]) / 2.0;
                a[(i, j)] = avg;
                a[(j, i)] = avg;
            }
        }

        let chol_unblocked = Cholesky::compute(a.as_ref()).expect("Unblocked should work");
        let chol_blocked = Cholesky::compute_blocked(a.as_ref()).expect("Blocked should work");

        let det_unblocked = chol_unblocked.determinant();
        let det_blocked = chol_blocked.determinant();

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
}
