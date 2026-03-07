//! QR Decomposition and related factorizations.
//!
//! Factorizes a matrix A into Q·R where:
//! - Q is an orthogonal matrix (Q^T·Q = I)
//! - R is an upper triangular matrix
//!
//! This module provides several variants:
//!
//! - **Qr**: Standard QR decomposition using Householder reflections.
//! - **QrPivot**: QR decomposition with column pivoting, useful for
//!   rank-revealing decomposition of rank-deficient matrices.
//! - **Rq**: RQ decomposition (A = R·Q) with R upper trapezoidal.
//! - **Lq**: LQ decomposition (A = L·Q) with L lower trapezoidal.
//! - **CompleteOrthogonalDecomp**: Complete orthogonal decomposition for
//!   rank-deficient matrices.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::qr::{Qr, QrPivot};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 3.0],
//!     &[4.0, 5.0, 6.0],
//!     &[7.0, 8.0, 10.0],
//! ]);
//!
//! // Standard QR (faster, usually sufficient)
//! let qr = Qr::compute(a.as_ref()).unwrap();
//! let q = qr.q();
//! let r = qr.r();
//!
//! // QR with column pivoting (rank-revealing)
//! let qr_pivot = QrPivot::compute(a.as_ref()).unwrap();
//! let rank = qr_pivot.rank();
//! ```

mod col_pivot;
mod complete_orthogonal;
mod householder;
mod lq;
mod ortho;
mod ql;
mod rq;
mod unitary;

pub use col_pivot::{QrPivot, QrPivotError};
pub use complete_orthogonal::CompleteOrthogonalDecomp;
pub use householder::{Qr, QrError};
pub use lq::Lq;
pub use ortho::{OrthoError, Side, Trans, orgqr, ormqr, ungqr, unmqr};
pub use ql::Ql;
pub use rq::Rq;
pub use unitary::{UnitaryQr, UnitaryQrError};

#[cfg(test)]
mod tests_recursive_qr {
    use super::Qr;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_qr_recursive_square_small() {
        // Small square matrix -- exercises the base case path
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 12.0, 13.0],
            &[14.0, 15.0, 16.0, 18.0],
        ]);

        let qr = Qr::compute_recursive(a.as_ref()).expect("recursive QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q is orthogonal: Q^T * Q = I
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "Recursive small: Q not orthogonal at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - expected).abs()
                );
            }
        }

        // Verify R is upper triangular
        for i in 0..4 {
            for j in 0..i {
                assert!(
                    r[(i, j)].abs() < 1e-10,
                    "Recursive small: R not upper tri at ({}, {}): {}",
                    i,
                    j,
                    r[(i, j)]
                );
            }
        }

        // Verify Q * R = A
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-10,
                    "Recursive small: QR[{},{}] = {}, A = {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_recursive_large_square() {
        // Large enough to trigger multiple levels of recursion (n > 48)
        let n = 200;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 7 + j * 11 + 3) % 13 + 1) as f64;
            }
            a[(i, i)] += 50.0; // Well-conditioned
        }

        let qr = Qr::compute_recursive(a.as_ref()).expect("recursive QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q^T * Q = I
        let tol = 1e-8;
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < tol,
                    "Recursive large: Q not orthogonal at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - expected).abs()
                );
            }
        }

        // Verify R is upper triangular
        for i in 0..n {
            for j in 0..i {
                assert!(
                    r[(i, j)].abs() < 1e-10,
                    "Recursive large: R not upper tri at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Verify Q * R = A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Recursive large: reconstruction error at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - a[(i, j)]).abs()
                );
            }
        }
    }

    #[test]
    fn test_qr_recursive_tall_matrix() {
        // Tall matrix: m > n, exercises rectangular path
        let m = 300;
        let n = 80;
        let mut a = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a[(i, j)] = ((i + 2 * j + 5) % 11 + 1) as f64;
            }
        }

        let qr = Qr::compute_recursive(a.as_ref()).expect("recursive QR tall should succeed");
        let q_thin = qr.q_thin();
        let r_thin = qr.r_thin();

        // q_thin: m x n, r_thin: n x n
        assert_eq!(q_thin.nrows(), m);
        assert_eq!(q_thin.ncols(), n);
        assert_eq!(r_thin.nrows(), n);
        assert_eq!(r_thin.ncols(), n);

        // Verify R is upper triangular
        for i in 0..n {
            for j in 0..i {
                assert!(
                    r_thin[(i, j)].abs() < 1e-10,
                    "Recursive tall: R not upper tri at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Verify Q_thin * R_thin = A
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q_thin[(i, k)] * r_thin[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Recursive tall: reconstruction error at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - a[(i, j)]).abs()
                );
            }
        }
    }

    #[test]
    fn test_qr_recursive_wide_matrix() {
        // Wide matrix: m < n
        let m = 60;
        let n = 150;
        let mut a = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a[(i, j)] = ((i * 5 + j * 3 + 2) % 9 + 1) as f64;
            }
        }

        let qr = Qr::compute_recursive(a.as_ref()).expect("recursive QR wide should succeed");
        let q = qr.q();
        let r = qr.r();

        // Q should be m x m
        assert_eq!(q.nrows(), m);
        assert_eq!(q.ncols(), m);

        // Verify Q^T * Q = I
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-9,
                    "Recursive wide: Q not orthogonal at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Verify Q * R = A
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Recursive wide: reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_recursive_matches_blocked() {
        // Verify that recursive and blocked produce equivalent factorizations
        let n = 100;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 3 + j * 7 + 1) % 11) as f64 + 1.0;
            }
            a[(i, i)] += 30.0;
        }

        let qr_recursive = Qr::compute_recursive(a.as_ref()).expect("recursive should succeed");
        let qr_blocked = Qr::compute_blocked(a.as_ref(), 32).expect("blocked should succeed");

        let r_rec = qr_recursive.r();
        let r_blk = qr_blocked.r();

        // R diagonal magnitudes should match (sign may differ due to Householder convention)
        for i in 0..n {
            assert!(
                (r_rec[(i, i)].abs() - r_blk[(i, i)].abs()).abs() < 1e-8,
                "Diagonal mismatch at {}: recursive={}, blocked={}",
                i,
                r_rec[(i, i)],
                r_blk[(i, i)]
            );
        }

        // Both should reconstruct A
        let q_rec = qr_recursive.q();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q_rec[(i, k)] * r_rec[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Recursive reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_recursive_identity() {
        // Identity matrix should decompose trivially
        let n = 64;
        let mut eye: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            eye[(i, i)] = 1.0;
        }

        let qr = Qr::compute_recursive(eye.as_ref()).expect("identity recursive QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Q * R should reconstruct identity
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "Identity recursive reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_recursive_f32() {
        // Test with f32 to ensure GemmKernel bound works for both types
        let n = 64;
        let mut a: Mat<f32> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 3 + j * 5 + 1) % 9 + 1) as f32;
            }
            a[(i, i)] += 20.0;
        }

        let qr = Qr::compute_recursive(a.as_ref()).expect("f32 recursive QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A with f32 tolerance
        for i in 0..n {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-3,
                    "f32 recursive reconstruction error at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - a[(i, j)]).abs()
                );
            }
        }
    }
}
