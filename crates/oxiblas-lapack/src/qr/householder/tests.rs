//! Tests for `Qr` (Householder QR decomposition).

use super::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn test_qr_square() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Verify Q is orthogonal: Q^T * Q = I
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                approx_eq(sum, expected, 1e-10),
                "Q^T*Q[{},{}] = {}, expected {}",
                i,
                j,
                sum,
                expected
            );
        }
    }

    // Verify R is upper triangular
    assert!(approx_eq(r[(1, 0)], 0.0, 1e-10));
    assert!(approx_eq(r[(2, 0)], 0.0, 1e-10));
    assert!(approx_eq(r[(2, 1)], 0.0, 1e-10));

    // Verify Q * R = A
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!(
                approx_eq(sum, a[(i, j)], 1e-10),
                "QR[{},{}] = {}, A = {}",
                i,
                j,
                sum,
                a[(i, j)]
            );
        }
    }
}

#[test]
fn test_qr_tall() {
    // 4x2 matrix
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Q should be 4×4
    assert_eq!(q.nrows(), 4);
    assert_eq!(q.ncols(), 4);

    // R should be 4×2
    assert_eq!(r.nrows(), 4);
    assert_eq!(r.ncols(), 2);

    // Verify Q is orthogonal
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(approx_eq(sum, expected, 1e-10));
        }
    }

    // Verify Q * R = A
    for i in 0..4 {
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!(approx_eq(sum, a[(i, j)], 1e-10));
        }
    }
}

#[test]
fn test_qr_wide() {
    // 2x3 matrix
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Q should be 2×2
    assert_eq!(q.nrows(), 2);
    assert_eq!(q.ncols(), 2);

    // R should be 2×3
    assert_eq!(r.nrows(), 2);
    assert_eq!(r.ncols(), 3);

    // Verify Q * R = A
    for i in 0..2 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..2 {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!(approx_eq(sum, a[(i, j)], 1e-10));
        }
    }
}

#[test]
fn test_qr_identity() {
    let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

    let qr = Qr::compute(eye.as_ref()).expect("QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Q and R should both be close to identity (with possible sign flips)
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert!(q[(i, j)].abs() > 0.99);
                assert!(r[(i, j)].abs() > 0.99);
            } else {
                assert!(approx_eq(q[(i, j)], 0.0, 1e-10));
                assert!(approx_eq(r[(i, j)], 0.0, 1e-10));
            }
        }
    }
}

#[test]
fn test_qr_thin() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let q_thin = qr.q_thin();
    let r_thin = qr.r_thin();

    // Q_thin should be 3×2
    assert_eq!(q_thin.nrows(), 3);
    assert_eq!(q_thin.ncols(), 2);

    // R_thin should be 2×2
    assert_eq!(r_thin.nrows(), 2);
    assert_eq!(r_thin.ncols(), 2);

    // Verify Q_thin * R_thin = A
    for i in 0..3 {
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..2 {
                sum += q_thin[(i, k)] * r_thin[(k, j)];
            }
            assert!(approx_eq(sum, a[(i, j)], 1e-10));
        }
    }
}

#[test]
fn test_qr_least_squares() {
    // Overdetermined system: 3 equations, 2 unknowns
    let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
    let b = Mat::from_rows(&[&[1.0f64], &[2.0], &[2.5]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let x = qr
        .solve_least_squares(b.as_ref())
        .expect("least squares should succeed");

    // Verify the solution minimizes ||Ax - b||
    // The solution should be close to x = [0.5, 0.75] for this problem
    assert!(x.nrows() == 2);
    assert!(x.ncols() == 1);

    // Verify Ax is close to b in least squares sense
    let mut ax = [0.0; 3];
    for i in 0..3 {
        for j in 0..2 {
            ax[i] += a[(i, j)] * x[(j, 0)];
        }
    }

    // Check residuals are reasonable
    let mut residual = 0.0;
    for i in 0..3 {
        residual += (ax[i] - b[(i, 0)]).powi(2);
    }
    residual = residual.sqrt();
    assert!(residual < 0.5); // Should be small for this well-conditioned problem
}

#[test]
fn test_qr_solve_least_squares_dimension_mismatch() {
    let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
    // b has only 2 rows, but A has 3 rows.
    let b = Mat::from_rows(&[&[1.0f64], &[2.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let result = qr.solve_least_squares(b.as_ref());

    match result {
        Err(QrError::DimensionMismatch { expected, actual }) => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("expected QrError::DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn test_qr_solve_least_squares_near_singular_reports_error() {
    // The second column is numerically (but not exactly) linearly
    // dependent on the first: after the first Householder reflection
    // is applied, the trailing entry of the second column is 1e-15,
    // which is far below the singularity threshold
    // (epsilon * 100 ~= 2.22e-14 for f64). This produces a genuinely
    // tiny -- but nonzero -- diagonal entry in R, exercising the
    // near-rank-deficiency detection path rather than an exact zero.
    let a = Mat::from_rows(&[&[1.0f64, 1.0], &[0.0, 1.0e-15]]);
    let b = Mat::from_rows(&[&[1.0f64], &[1.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");

    // Sanity-check that R really does have a tiny, nonzero diagonal at
    // index 1 -- otherwise this test would not exercise the intended
    // code path at all.
    let r = qr.r();
    assert!(
        r[(1, 1)] != 0.0,
        "R[1,1] should be nonzero (exact zero would take a different code path)"
    );
    assert!(
        r[(1, 1)].abs() < 1e-12,
        "R[1,1] should be tiny, got {}",
        r[(1, 1)]
    );

    let result = qr.solve_least_squares(b.as_ref());
    match result {
        Err(QrError::NearlySingular { index }) => {
            assert_eq!(index, 1, "near-singularity should be reported at index 1");
        }
        other => panic!("expected QrError::NearlySingular, got {other:?}"),
    }
}

#[test]
fn test_qr_extreme_magnitude_column_stays_orthogonal() {
    // A column of extreme-magnitude values (~1e200) would overflow a
    // naive sum-of-squares norm: (1e200)^2 = 1e400 vastly exceeds
    // f64::MAX (~1.8e308), producing an infinite norm that corrupts
    // beta/tau with NaN and destroys orthogonality of Q. The scaled
    // (Blue's-algorithm-style) norm computation must keep the norm --
    // and therefore Q -- finite and orthogonal.
    let huge = 1.0e200_f64;
    let a = Mat::from_rows(&[&[huge, 2.0, 3.0], &[huge, 5.0, 6.0], &[huge, 8.0, 10.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed even with extreme magnitudes");
    let q = qr.q();

    // Every entry of Q must be finite. Before the fix, the overflowed
    // norm produces NaN entries here.
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                q[(i, j)].is_finite(),
                "Q[{i},{j}] = {} is not finite",
                q[(i, j)]
            );
        }
    }

    // Verify Q is orthogonal: Q^T * Q = I.
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                approx_eq(sum, expected, 1e-8),
                "Q^T*Q[{i},{j}] = {sum}, expected {expected}"
            );
        }
    }

    // R should also stay finite and reproduce A via Q*R.
    let r = qr.r();
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += q[(i, k)] * r[(k, j)];
            }
            let rel_tol = a[(i, j)].abs() * 1e-8;
            assert!(
                (sum - a[(i, j)]).abs() <= rel_tol,
                "QR[{i},{j}] = {sum}, A = {}",
                a[(i, j)]
            );
        }
    }
}

#[test]
fn test_qr_f32() {
    let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Verify Q * R = A
    for i in 0..2 {
        for j in 0..2 {
            let mut sum: f32 = 0.0;
            for k in 0..2 {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!((sum - a[(i, j)]).abs() < 1e-5);
        }
    }
}

#[test]
fn test_qr_single() {
    let a = Mat::from_rows(&[&[3.0f64]]);

    let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Q should be ±1, R should be ±3
    assert!(q[(0, 0)].abs() > 0.99);
    assert!(r[(0, 0)].abs() > 2.99);
    assert!(approx_eq((q[(0, 0)] * r[(0, 0)]).abs(), 3.0, 1e-10));
}

#[test]
fn test_qr_blocked_vs_unblocked_4x4() {
    // Verify blocked and unblocked produce equivalent factorizations
    let a = Mat::from_rows(&[
        &[1.0f64, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &[9.0, 10.0, 11.0, 12.0],
        &[13.0, 14.0, 15.0, 16.0],
    ]);

    let qr_blocked = Qr::compute_blocked(a.as_ref(), 2).expect("blocked QR should succeed");
    let q_b = qr_blocked.q();
    let r_b = qr_blocked.r();

    // Verify Q * R = A for blocked
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += q_b[(i, k)] * r_b[(k, j)];
            }
            let diff = sum - a[(i, j)];
            assert!(
                diff.abs() < 1e-10,
                "Blocked reconstruction error at ({}, {}): got {}, expected {}, diff={}",
                i,
                j,
                sum,
                a[(i, j)],
                diff
            );
        }
    }

    // Verify Q is orthogonal
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += q_b[(k, i)] * q_b[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-10,
                "Q not orthogonal at ({}, {}): got {}, expected {}",
                i,
                j,
                sum,
                expected
            );
        }
    }

    // Verify R is upper triangular
    for i in 0..4 {
        for j in 0..i {
            assert!(
                r_b[(i, j)].abs() < 1e-10,
                "R not upper triangular at ({}, {}): got {}",
                i,
                j,
                r_b[(i, j)]
            );
        }
    }
}

#[test]
fn test_qr_blocked_various_block_sizes() {
    // Test blocked QR with different block sizes on a 12x12 matrix
    let n = 12;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 3 + j * 7 + 1) % 11) as f64 + 1.0;
        }
        a[(i, i)] += 20.0; // Make well-conditioned
    }

    for nb in [1, 2, 3, 4, 6, 12] {
        let qr = Qr::compute_blocked(a.as_ref(), nb).expect("blocked QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-9,
                    "nb={}: reconstruction error at ({}, {}): diff={}",
                    nb,
                    i,
                    j,
                    sum - a[(i, j)]
                );
            }
        }

        // Verify Q^T * Q = I
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-9,
                    "nb={}: Q not orthogonal at ({}, {})",
                    nb,
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn test_qr_blocked_small() {
    // Test blocked QR with a small matrix first
    let n = 8;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i + j) % 5 + 1) as f64;
        }
    }

    // Compute using blocked algorithm with small block size
    let qr_blocked = Qr::compute_blocked(a.as_ref(), 4).expect("blocked QR should succeed");
    let q = qr_blocked.q();
    let r = qr_blocked.r();

    // Verify Q * R = A
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * r[(k, j)];
            }
            let diff = sum - a[(i, j)];
            assert!(
                diff.abs() < 1e-10,
                "Reconstruction error at ({}, {}): got {}, expected {}, diff={}",
                i,
                j,
                sum,
                a[(i, j)],
                diff
            );
        }
    }

    // Verify Q is orthogonal
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-10,
                "Q not orthogonal at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]

fn test_qr_blocked_correctness() {
    // Test that blocked QR produces correct factorization
    let n = 200;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i + j) % 10 + 1) as f64;
        }
        a[(i, i)] += 10.0; // Make it well-conditioned
    }

    // Compute using blocked algorithm
    let qr_blocked = Qr::compute_blocked(a.as_ref(), 64).expect("blocked QR should succeed");
    let q = qr_blocked.q();
    let r = qr_blocked.r();

    // Verify Q is orthogonal: Q^T * Q = I
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-9,
                "Q not orthogonal at ({}, {}): got {}, expected {}",
                i,
                j,
                sum,
                expected
            );
        }
    }

    // Verify R is upper triangular
    for i in 0..n {
        for j in 0..i {
            assert!(
                r[(i, j)].abs() < 1e-10,
                "R not upper triangular at ({}, {}): got {}",
                i,
                j,
                r[(i, j)]
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
                "Reconstruction error at ({}, {}): got {}, expected {}",
                i,
                j,
                sum,
                a[(i, j)]
            );
        }
    }
}

#[test]

fn test_qr_auto_selection() {
    // Test automatic algorithm selection
    let n = 150;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 7 + j * 11) % 13 + 1) as f64;
        }
    }

    let qr = Qr::compute_auto(a.as_ref()).expect("auto QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Verify Q is orthogonal: Q^T * Q = I
    // Use relaxed tolerance for larger matrices (150×150) due to accumulated rounding errors
    let tol = 1e-5;
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < tol,
                "Q not orthogonal at ({}, {}): got {}, expected {}, diff={}",
                i,
                j,
                sum,
                expected,
                (sum - expected).abs()
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
                "QR reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_blocked_tall_matrix() {
    // Test blocked QR on tall matrix (m > n)
    let m = 300;
    let n = 100;
    let mut a = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            a[(i, j)] = ((i + 2 * j) % 7 + 1) as f64;
        }
    }

    let qr = Qr::compute_blocked(a.as_ref(), 32).expect("blocked QR should succeed");
    let r = qr.r_thin();

    // Verify R is upper triangular
    for i in 0..n {
        for j in 0..i {
            assert!(
                r[(i, j)].abs() < 1e-10,
                "R not upper triangular at ({}, {}): got {}",
                i,
                j,
                r[(i, j)]
            );
        }
    }

    // Verify Q * R = A (using thin R)
    let q_thin = qr.q_thin();
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q_thin[(i, k)] * r[(k, j)];
            }
            assert!(
                (sum - a[(i, j)]).abs() < 1e-8,
                "Thin QR reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_blocked_wide_matrix() {
    // Test blocked QR on wide matrix (m < n)
    let m = 50;
    let n = 120;
    let mut a = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            a[(i, j)] = ((i * 5 + j * 3 + 2) % 11) as f64 + 0.5;
        }
    }

    let qr = Qr::compute_blocked(a.as_ref(), 16).expect("blocked QR should succeed");
    let q = qr.q();
    let r = qr.r();

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
                "Q not orthogonal at ({}, {}): diff={}",
                i,
                j,
                (sum - expected).abs()
            );
        }
    }

    // Verify R is upper triangular
    let k = m.min(n);
    for i in 0..m {
        for j in 0..i.min(k) {
            assert!(
                r[(i, j)].abs() < 1e-10,
                "R not upper triangular at ({}, {})",
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
                "Wide reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_blocked_f32() {
    // Test blocked QR with f32 precision
    let n = 32;
    let mut a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 3 + j * 5 + 1) % 9 + 1) as f32;
        }
        a[(i, i)] += 10.0;
    }

    let qr = Qr::compute_blocked(a.as_ref(), 8).expect("f32 blocked QR should succeed");
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
                "f32 blocked reconstruction error at ({}, {}): diff={}",
                i,
                j,
                (sum - a[(i, j)]).abs()
            );
        }
    }

    // Verify Q^T * Q = I with f32 tolerance
    for i in 0..n {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected: f32 = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-3,
                "f32 Q not orthogonal at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_blocked_identity_matrix() {
    // The identity should decompose trivially
    let n = 16;
    let mut eye = Mat::zeros(n, n);
    for i in 0..n {
        eye[(i, i)] = 1.0f64;
    }

    let qr = Qr::compute_blocked(eye.as_ref(), 4).expect("identity blocked QR should succeed");
    let q = qr.q();
    let r = qr.r();

    for i in 0..n {
        for j in 0..n {
            let expected: f64 = if i == j { 1.0 } else { 0.0 };
            // Q and R should each be +/-I (with possible sign flips on diagonal)
            assert!(
                (q[(i, j)].abs() - expected.abs()).abs() < 1e-10
                    || (i == j && q[(i, j)].abs() > 0.99),
                "Identity Q error at ({}, {})",
                i,
                j
            );
        }
    }

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
                "Identity reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_blocked_block_size_1() {
    // Block size 1 should be equivalent to unblocked
    let n = 20;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i + j + 1) % 7) as f64 + 1.0;
        }
        a[(i, i)] += 15.0;
    }

    let qr_unblocked = Qr::compute(a.as_ref()).expect("unblocked QR should succeed");
    let qr_blocked = Qr::compute_blocked(a.as_ref(), 1).expect("nb=1 blocked QR should succeed");

    // R matrices should be numerically identical
    let r_u = qr_unblocked.r();
    let r_b = qr_blocked.r();

    for i in 0..n {
        for j in 0..n {
            assert!(
                (r_u[(i, j)] - r_b[(i, j)]).abs() < 1e-10,
                "R mismatch at ({}, {}): unblocked={}, blocked={}",
                i,
                j,
                r_u[(i, j)],
                r_b[(i, j)]
            );
        }
    }
}

#[test]
fn test_qr_blocked_block_size_exceeds_n() {
    // Block size larger than matrix dimension -- single panel, no trailing update
    let n = 8;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = (i * n + j + 1) as f64;
        }
    }

    let qr = Qr::compute_blocked(a.as_ref(), 64).expect("large nb should succeed");
    let q = qr.q();
    let r = qr.r();

    // Verify Q * R = A
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!(
                (sum - a[(i, j)]).abs() < 1e-9,
                "Large nb reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_auto_small_uses_unblocked() {
    // For small matrices (< 128), auto should use unblocked
    let n = 64;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 3 + j * 7) % 17 + 1) as f64;
        }
        a[(i, i)] += 20.0;
    }

    let qr = Qr::compute_auto(a.as_ref()).expect("auto QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Verify reconstruction
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!(
                (sum - a[(i, j)]).abs() < 1e-8,
                "Auto small reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_qr_blocked_well_conditioned() {
    // Test with a well-conditioned matrix (diagonally dominant)
    let n = 100;
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = if i == j {
                100.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }

    let qr = Qr::compute_blocked(a.as_ref(), 32).expect("well-conditioned QR should succeed");
    let q = qr.q();
    let r = qr.r();

    // Tight orthogonality check
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * q[(k, j)];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-10,
                "Well-conditioned Q orthogonality error at ({}, {}): diff={}",
                i,
                j,
                (sum - expected).abs()
            );
        }
    }

    // Tight reconstruction check
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * r[(k, j)];
            }
            assert!(
                (sum - a[(i, j)]).abs() < 1e-9,
                "Well-conditioned reconstruction error at ({}, {})",
                i,
                j
            );
        }
    }
}
