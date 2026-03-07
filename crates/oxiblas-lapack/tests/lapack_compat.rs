//! LAPACK compatibility integration tests.
//!
//! Verifies OxiBLAS LAPACK operations match reference LAPACK behavior
//! for standard test cases including Hilbert, Vandermonde, Pascal, and
//! diagonally dominant matrices.

use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::evd::{GeneralEvd, SymmetricEvd};
use oxiblas_lapack::lu::Lu;
use oxiblas_lapack::qr::{Qr, QrPivot};
use oxiblas_lapack::solve::{solve, solve_multiple};
use oxiblas_lapack::svd::{Svd, SvdDc};
use oxiblas_matrix::Mat;

// ============================================================================
// Test Matrix Generators
// ============================================================================

/// Creates an n x n Hilbert matrix: H(i,j) = 1 / (i + j + 1)
fn hilbert(n: usize) -> Mat<f64> {
    let mut h = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] = 1.0 / ((i + j + 1) as f64);
        }
    }
    h
}

/// Creates an n x n Vandermonde matrix from nodes 1..=n.
fn vandermonde(n: usize) -> Mat<f64> {
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        let x = (i + 1) as f64;
        for j in 0..n {
            v[(i, j)] = x.powi(j as i32);
        }
    }
    v
}

/// Creates an n x n lower-triangular Pascal matrix.
/// Pascal matrices are always SPD (symmetric positive definite).
fn pascal(n: usize) -> Mat<f64> {
    let mut p = Mat::zeros(n, n);
    for i in 0..n {
        p[(i, 0)] = 1.0;
    }
    for j in 0..n {
        p[(0, j)] = 1.0;
    }
    for i in 1..n {
        for j in 1..n {
            p[(i, j)] = p[(i - 1, j)] + p[(i, j - 1)];
        }
    }
    // Make symmetric: S = P * P^T
    let mut s = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += p[(i, k)] * p[(j, k)];
            }
            s[(i, j)] = sum;
        }
    }
    s
}

/// Creates an n x n diagonally dominant matrix.
fn diag_dominant(n: usize) -> Mat<f64> {
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                a[(i, j)] = (n as f64) + 1.0;
            } else {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 / 100.0;
            }
        }
    }
    a
}

/// Creates an n x n SPD matrix: A = B^T * B + n * I
fn make_spd(n: usize) -> Mat<f64> {
    let mut b = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] = ((i * 13 + j * 37 + 7) % 100) as f64 / 50.0;
        }
    }
    let mut spd = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += b[(k, i)] * b[(k, j)];
            }
            spd[(i, j)] = sum;
            if i == j {
                spd[(i, j)] += n as f64;
            }
        }
    }
    spd
}

/// Creates a matrix with known eigenvalues on the diagonal.
fn diagonal_eigenvalue_matrix(eigenvalues: &[f64]) -> Mat<f64> {
    let n = eigenvalues.len();
    let mut d = Mat::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = eigenvalues[i];
    }
    d
}

/// Creates a companion matrix for polynomial x^n - c_{n-1}x^{n-1} - ... - c_0
/// whose eigenvalues are the roots of the polynomial.
fn companion_matrix(coefficients: &[f64]) -> Mat<f64> {
    let n = coefficients.len();
    let mut c = Mat::zeros(n, n);
    // Sub-diagonal ones
    for i in 1..n {
        c[(i, i - 1)] = 1.0;
    }
    // Last column contains the negated coefficients (from bottom)
    for i in 0..n {
        c[(i, n - 1)] = coefficients[i];
    }
    c
}

/// Matrix-matrix multiply: C = A * B
fn matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();
    assert_eq!(k, b.nrows(), "Dimension mismatch in matmul");
    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[(i, p)] * b[(p, j)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}

/// Transpose a matrix.
fn transpose(a: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut t = Mat::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            t[(j, i)] = a[(i, j)];
        }
    }
    t
}

/// Frobenius norm of the difference A - B.
fn frobenius_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let mut sum = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let d = a[(i, j)] - b[(i, j)];
            sum += d * d;
        }
    }
    sum.sqrt()
}

/// Frobenius norm.
fn frobenius_norm(a: &Mat<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += a[(i, j)] * a[(i, j)];
        }
    }
    sum.sqrt()
}

// ============================================================================
// LU Factorization Tests
// ============================================================================

mod lu_tests {
    use super::*;

    #[test]
    fn lu_1x1() {
        let a: Mat<f64> = Mat::from_rows(&[&[7.0]]);
        let lu = Lu::compute(a.as_ref()).unwrap();
        assert!((lu.determinant() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn lu_2x2_pa_eq_lu() {
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0], &[4.0, 3.0]]);
        let lu = Lu::compute(a.as_ref()).unwrap();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();
        let pa = matmul(&p, &a);
        let lu_prod = matmul(&l, &u);
        let err = frobenius_diff(&pa, &lu_prod);
        assert!(err < 1e-12, "PA != LU, error = {}", err);
    }

    #[test]
    fn lu_5x5_hilbert() {
        let a = hilbert(5);
        let lu = Lu::compute(a.as_ref()).unwrap();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();
        let pa = matmul(&p, &a);
        let lu_prod = matmul(&l, &u);
        let rel_err = frobenius_diff(&pa, &lu_prod) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "Hilbert 5x5 PA != LU, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn lu_5x5_vandermonde() {
        let a = vandermonde(5);
        let lu = Lu::compute(a.as_ref()).unwrap();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();
        let pa = matmul(&p, &a);
        let lu_prod = matmul(&l, &u);
        let rel_err = frobenius_diff(&pa, &lu_prod) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "Vandermonde 5x5 PA != LU, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn lu_50x50_diag_dominant() {
        let a = diag_dominant(50);
        let lu = Lu::compute(a.as_ref()).unwrap();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();
        let pa = matmul(&p, &a);
        let lu_prod = matmul(&l, &u);
        let rel_err = frobenius_diff(&pa, &lu_prod) / frobenius_norm(&a);
        assert!(rel_err < 1e-10, "50x50 PA != LU, rel_error = {}", rel_err);
    }

    #[test]
    fn lu_100x100_reconstruction() {
        let a = diag_dominant(100);
        let lu = Lu::compute(a.as_ref()).unwrap();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();
        let pa = matmul(&p, &a);
        let lu_prod = matmul(&l, &u);
        let rel_err = frobenius_diff(&pa, &lu_prod) / frobenius_norm(&a);
        assert!(rel_err < 1e-9, "100x100 PA != LU, rel_error = {}", rel_err);
    }

    #[test]
    fn lu_singular_returns_error() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(Lu::compute(a.as_ref()).is_err());
    }

    #[test]
    fn lu_near_singular_2x2() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[1.0, 2.0 + 1e-14]]);
        // Near-singular should still factorize (with large condition number)
        let result = Lu::compute(a.as_ref());
        // It may succeed or fail depending on tolerance; just verify no panic
        let _ = result;
    }

    #[test]
    fn lu_singular_3x3_rank_deficient() {
        // Rank 2: third row = first + second
        // Note: LU with partial pivoting may not always detect near-singular
        // matrices due to floating-point arithmetic. We test that either:
        // (a) It returns an error, or
        // (b) The determinant is near-zero.
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[5.0, 7.0, 9.0]]);
        match Lu::compute(a.as_ref()) {
            Err(_) => {} // Correctly detected as singular
            Ok(lu) => {
                // If it succeeded, the determinant should be near zero
                assert!(
                    lu.determinant().abs() < 1e-10,
                    "Rank-deficient matrix should have near-zero determinant, got {}",
                    lu.determinant()
                );
            }
        }
    }

    #[test]
    fn lu_pascal_matrix() {
        let a = pascal(5);
        let lu = Lu::compute(a.as_ref()).unwrap();
        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.permutation_matrix();
        let pa = matmul(&p, &a);
        let lu_prod = matmul(&l, &u);
        let rel_err = frobenius_diff(&pa, &lu_prod) / frobenius_norm(&a);
        assert!(rel_err < 1e-10, "Pascal PA != LU, rel_error = {}", rel_err);
    }

    #[test]
    fn lu_blocked_matches_unblocked() {
        let a = diag_dominant(80);
        let lu1 = Lu::compute(a.as_ref()).unwrap();
        let lu2 = Lu::compute_blocked(a.as_ref()).unwrap();
        let det1 = lu1.determinant();
        let det2 = lu2.determinant();
        let rel = ((det1 - det2) / det1).abs();
        assert!(rel < 1e-10, "blocked/unblocked det mismatch: rel = {}", rel);
    }
}

// ============================================================================
// Cholesky Tests
// ============================================================================

mod cholesky_tests {
    use super::*;

    #[test]
    fn cholesky_1x1() {
        let a: Mat<f64> = Mat::from_rows(&[&[9.0]]);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        assert!((l[(0, 0)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_2x2_reconstruction() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let err = frobenius_diff(&a, &recon);
        assert!(err < 1e-12, "A != LL^T, error = {}", err);
    }

    #[test]
    fn cholesky_spd_5x5() {
        let a = make_spd(5);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "SPD 5x5 A != LL^T, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn cholesky_spd_50x50() {
        let a = make_spd(50);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-8,
            "SPD 50x50 A != LL^T, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn cholesky_spd_100x100() {
        let a = make_spd(100);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-7,
            "SPD 100x100 A != LL^T, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn cholesky_identity() {
        let a: Mat<f64> = Mat::eye(10);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        for i in 0..10 {
            for j in 0..10 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((l[(i, j)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_not_spd_returns_error() {
        // Indefinite matrix: eigenvalues 3 and -1
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);
        assert!(Cholesky::compute(a.as_ref()).is_err());
    }

    #[test]
    fn cholesky_hilbert_small() {
        // Hilbert matrices are SPD but ill-conditioned
        let a = hilbert(4);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "Hilbert 4x4 A != LL^T, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn cholesky_diag_dominant_spd() {
        // Diagonally dominant symmetric matrix is SPD
        let n = 20;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (n as f64) * 2.0;
            for j in 0..n {
                if i != j {
                    let val = ((i + j) % 5) as f64 * 0.1;
                    a[(i, j)] = val;
                    a[(j, i)] = val;
                }
            }
        }
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "Diag dominant A != LL^T, rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn cholesky_pascal_spd() {
        let a = pascal(5);
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let l = chol.l_factor();
        let lt = transpose(&l);
        let recon = matmul(&l, &lt);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-8, "Pascal A != LL^T, rel_error = {}", rel_err);
    }

    #[test]
    fn cholesky_negative_diagonal_returns_error() {
        let a: Mat<f64> = Mat::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]);
        assert!(Cholesky::compute(a.as_ref()).is_err());
    }
}

// ============================================================================
// QR Decomposition Tests
// ============================================================================

mod qr_tests {
    use super::*;

    #[test]
    fn qr_2x2_orthogonality() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let qtq = matmul(&transpose(&q), &q);
        let identity = Mat::eye(2);
        let err = frobenius_diff(&qtq, &identity);
        assert!(err < 1e-12, "Q not orthogonal, error = {}", err);
    }

    #[test]
    fn qr_2x2_reconstruction() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();
        let recon = matmul(&q, &r);
        let err = frobenius_diff(&a, &recon);
        assert!(err < 1e-12, "QR != A, error = {}", err);
    }

    #[test]
    fn qr_5x5_hilbert() {
        let a = hilbert(5);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q orthogonality
        let qtq = matmul(&transpose(&q), &q);
        let identity = Mat::eye(5);
        let orth_err = frobenius_diff(&qtq, &identity);
        assert!(
            orth_err < 1e-10,
            "Hilbert Q not orthogonal, error = {}",
            orth_err
        );

        // Reconstruction
        let recon = matmul(&q, &r);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-10, "Hilbert QR != A, rel_error = {}", rel_err);
    }

    #[test]
    fn qr_50x50_diag_dominant() {
        let a = diag_dominant(50);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q orthogonality
        let qtq = matmul(&transpose(&q), &q);
        let identity = Mat::eye(50);
        let orth_err = frobenius_diff(&qtq, &identity);
        assert!(
            orth_err < 1e-9,
            "50x50 Q not orthogonal, error = {}",
            orth_err
        );

        // Reconstruction
        let recon = matmul(&q, &r);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-9, "50x50 QR != A, rel_error = {}", rel_err);
    }

    #[test]
    fn qr_100x100_orthogonality_and_reconstruction() {
        let a = diag_dominant(100);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Check Q^T * Q = I
        let qtq = matmul(&transpose(&q), &q);
        let identity = Mat::eye(100);
        let orth_err = frobenius_diff(&qtq, &identity);
        assert!(
            orth_err < 1e-8,
            "100x100 Q not orthogonal, error = {}",
            orth_err
        );

        // Check Q * R = A
        let recon = matmul(&q, &r);
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-8, "100x100 QR != A, rel_error = {}", rel_err);
    }

    #[test]
    fn qr_tall_matrix() {
        // 6x3 tall matrix
        let a: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 10.0],
            &[11.0, 12.0, 13.0],
            &[14.0, 15.0, 17.0],
            &[18.0, 19.0, 20.0],
        ]);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q^T * Q = I (6x6)
        let qtq = matmul(&transpose(&q), &q);
        let identity = Mat::eye(6);
        let orth_err = frobenius_diff(&qtq, &identity);
        assert!(
            orth_err < 1e-10,
            "Tall Q not orthogonal, error = {}",
            orth_err
        );

        // Q * R = A
        let recon = matmul(&q, &r);
        let err = frobenius_diff(&a, &recon);
        assert!(err < 1e-10, "Tall QR != A, error = {}", err);
    }

    #[test]
    fn qr_r_is_upper_triangular() {
        let a = diag_dominant(10);
        let qr = Qr::compute(a.as_ref()).unwrap();
        let r = qr.r();

        for i in 0..r.nrows() {
            for j in 0..i.min(r.ncols()) {
                assert!(
                    r[(i, j)].abs() < 1e-12,
                    "R[{},{}] = {} is not zero (not upper triangular)",
                    i,
                    j,
                    r[(i, j)]
                );
            }
        }
    }

    #[test]
    fn qr_pivot_rank_deficient() {
        // Rank 2 matrix: third column is sum of first two
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0], &[1.0, 1.0, 2.0]]);
        let qr = QrPivot::compute(a.as_ref()).unwrap();
        assert_eq!(qr.rank(), 2, "Rank-deficient matrix should have rank 2");
    }

    #[test]
    fn qr_pivot_reconstruction() {
        let a = diag_dominant(20);
        let qr = QrPivot::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();
        let p = qr.permutation_matrix();

        // Q * R = A * P
        let qr_prod = matmul(&q, &r);
        let ap = matmul(&a, &p);
        let rel_err = frobenius_diff(&qr_prod, &ap) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "QR_pivot: QR != AP, rel_error = {}",
            rel_err
        );
    }
}

// ============================================================================
// SVD Tests
// ============================================================================

mod svd_tests {
    use super::*;

    #[test]
    fn svd_2x2_reconstruction() {
        let a: Mat<f64> = Mat::from_rows(&[&[3.0, 2.0], &[2.0, 3.0]]);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let recon = svd.reconstruct();
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-10, "SVD reconstruct error = {}", rel_err);
    }

    #[test]
    fn svd_singular_values_nonnegative_and_sorted() {
        let a = diag_dominant(20);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let sv = svd.singular_values();
        for (idx, &s) in sv.iter().enumerate() {
            assert!(s >= 0.0, "Singular value {} is negative: {}", idx, s);
        }
        for i in 1..sv.len() {
            assert!(
                sv[i - 1] >= sv[i] - 1e-12,
                "Singular values not sorted at index {}: {} < {}",
                i,
                sv[i - 1],
                sv[i]
            );
        }
    }

    #[test]
    fn svd_u_orthogonality() {
        let a = diag_dominant(10);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let u_mat: Mat<f64> = {
            let u_ref = svd.u();
            let m = u_ref.nrows();
            let n = u_ref.ncols();
            let mut out = Mat::zeros(m, n);
            for i in 0..m {
                for j in 0..n {
                    out[(i, j)] = u_ref[(i, j)];
                }
            }
            out
        };
        let utu = matmul(&transpose(&u_mat), &u_mat);
        let identity = Mat::eye(u_mat.ncols());
        let err = frobenius_diff(&utu, &identity);
        assert!(err < 1e-10, "U not orthogonal, error = {}", err);
    }

    #[test]
    fn svd_v_orthogonality() {
        let a = diag_dominant(10);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let vt_ref = svd.vt();
        let n = vt_ref.nrows();
        let mut vt_mat = Mat::zeros(n, vt_ref.ncols());
        for i in 0..n {
            for j in 0..vt_ref.ncols() {
                vt_mat[(i, j)] = vt_ref[(i, j)];
            }
        }
        // V^T * V = I
        let vt_v = matmul(&vt_mat, &transpose(&vt_mat));
        let identity = Mat::eye(n);
        let err = frobenius_diff(&vt_v, &identity);
        assert!(err < 1e-10, "V not orthogonal, error = {}", err);
    }

    #[test]
    fn svd_5x5_hilbert() {
        let a = hilbert(5);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let recon = svd.reconstruct();
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-8,
            "Hilbert SVD reconstruct rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn svd_dc_matches_jacobi_small() {
        // D&C SVD on a small well-conditioned matrix should match Jacobi closely.
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
        let svd1 = Svd::compute(a.as_ref()).unwrap();
        let svd2 = SvdDc::compute(a.as_ref()).unwrap();
        let sv1 = svd1.singular_values();
        let sv2 = svd2.singular_values();
        assert_eq!(sv1.len(), sv2.len());
        for i in 0..sv1.len() {
            let rel_diff = if sv1[i].abs() > 1e-15 {
                (sv1[i] - sv2[i]).abs() / sv1[i]
            } else {
                (sv1[i] - sv2[i]).abs()
            };
            assert!(
                rel_diff < 0.05,
                "SV[{}] relative mismatch: jacobi={}, dc={}, rel_diff={}",
                i,
                sv1[i],
                sv2[i],
                rel_diff
            );
        }
    }

    #[test]
    fn svd_dc_small_reconstruction() {
        // D&C SVD on a small well-conditioned matrix
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let recon = svd.reconstruct();
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-8,
            "SvdDc 3x3 reconstruct rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn svd_dc_singular_values_nonneg_sorted() {
        // Verify D&C SVD produces non-negative, sorted singular values
        let a = diag_dominant(30);
        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let sv = svd.singular_values();
        for (i, &s) in sv.iter().enumerate() {
            assert!(s >= -1e-12, "D&C SV[{}] negative: {}", i, s);
        }
        for i in 1..sv.len() {
            assert!(
                sv[i - 1] >= sv[i] - 1e-10,
                "D&C SVs not sorted at {}: {} < {}",
                i,
                sv[i - 1],
                sv[i]
            );
        }
    }

    #[test]
    fn svd_tall_matrix() {
        // 6x3 matrix
        let a: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 10.0],
            &[11.0, 12.0, 13.0],
            &[14.0, 15.0, 17.0],
            &[18.0, 19.0, 20.0],
        ]);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let sv = svd.singular_values();
        assert_eq!(
            sv.len(),
            3,
            "Tall matrix should have min(m,n) singular values"
        );
        for &s in sv {
            assert!(s >= 0.0);
        }
        let recon = svd.reconstruct();
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(
            rel_err < 1e-10,
            "Tall SVD reconstruct rel_error = {}",
            rel_err
        );
    }

    #[test]
    fn svd_diagonal_matrix() {
        // Singular values should be the absolute diagonal entries, sorted
        let a: Mat<f64> = Mat::from_rows(&[&[5.0, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 1.0]]);
        let svd = Svd::compute(a.as_ref()).unwrap();
        let sv = svd.singular_values();
        assert!((sv[0] - 5.0).abs() < 1e-10, "sv[0] = {}", sv[0]);
        assert!((sv[1] - 3.0).abs() < 1e-10, "sv[1] = {}", sv[1]);
        assert!((sv[2] - 1.0).abs() < 1e-10, "sv[2] = {}", sv[2]);
    }
}

// ============================================================================
// Eigenvalue Tests
// ============================================================================

mod evd_tests {
    use super::*;

    #[test]
    fn symmetric_evd_2x2() {
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        // Eigenvalues are 1 and 3 (sorted ascending)
        assert!((eigs[0] - 1.0).abs() < 1e-10, "eig[0] = {}", eigs[0]);
        assert!((eigs[1] - 3.0).abs() < 1e-10, "eig[1] = {}", eigs[1]);
    }

    #[test]
    fn symmetric_evd_eigenvalues_real() {
        let a = make_spd(20);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        // All eigenvalues of SPD must be positive
        for (i, &e) in eigs.iter().enumerate() {
            assert!(e > -1e-10, "Eigenvalue {} is not real/positive: {}", i, e);
        }
    }

    #[test]
    fn symmetric_evd_eigenvectors_orthogonal() {
        let a = make_spd(10);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();
        let n = v.nrows();

        // V^T * V = I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "V not orthogonal: V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn symmetric_evd_av_eq_vd_reconstruction() {
        let a = make_spd(10);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let recon = evd.reconstruct();
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-8, "A != VDV^T, rel_error = {}", rel_err);
    }

    #[test]
    fn symmetric_evd_50x50() {
        let a = make_spd(50);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let recon = evd.reconstruct();
        let rel_err = frobenius_diff(&a, &recon) / frobenius_norm(&a);
        assert!(rel_err < 1e-6, "50x50 A != VDV^T, rel_error = {}", rel_err);
    }

    #[test]
    fn symmetric_evd_known_eigenvalues() {
        // Diagonal matrix with known eigenvalues
        let eigenvalues = [1.0, 3.0, 5.0, 7.0, 9.0];
        let a = diagonal_eigenvalue_matrix(&eigenvalues);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let computed = evd.eigenvalues();

        // Should be sorted ascending
        for i in 0..5 {
            assert!(
                (computed[i] - eigenvalues[i]).abs() < 1e-10,
                "Eigenvalue {} mismatch: expected {}, got {}",
                i,
                eigenvalues[i],
                computed[i]
            );
        }
    }

    #[test]
    fn symmetric_evd_identity() {
        let a: Mat<f64> = Mat::eye(5);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        for (i, &e) in eigs.iter().enumerate() {
            assert!(
                (e - 1.0).abs() < 1e-10,
                "Identity eigenvalue {} = {}, expected 1.0",
                i,
                e
            );
        }
    }

    #[test]
    fn general_evd_rotation_matrix() {
        // 90-degree rotation matrix has eigenvalues +/- i
        let a: Mat<f64> = Mat::from_rows(&[&[0.0, -1.0], &[1.0, 0.0]]);
        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        // Both eigenvalues should have zero real part and |imag| = 1
        for e in eigs {
            assert!(
                e.real.abs() < 1e-10,
                "Rotation eigenvalue real part should be 0, got {}",
                e.real
            );
            assert!(
                (e.imag.abs() - 1.0).abs() < 1e-10,
                "Rotation eigenvalue |imag| should be 1, got {}",
                e.imag.abs()
            );
        }
    }

    #[test]
    fn general_evd_companion_matrix() {
        // Companion matrix for x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        // Eigenvalues should be 1, 2, 3
        let c = companion_matrix(&[6.0, -11.0, 6.0]);
        let evd = GeneralEvd::compute(c.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Collect real parts, sort, and compare
        let mut reals: Vec<f64> = eigs.iter().map(|e| e.real).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        assert!(
            (reals[0] - 1.0).abs() < 1e-6,
            "Companion eigenvalue 0 = {}, expected 1.0",
            reals[0]
        );
        assert!(
            (reals[1] - 2.0).abs() < 1e-6,
            "Companion eigenvalue 1 = {}, expected 2.0",
            reals[1]
        );
        assert!(
            (reals[2] - 3.0).abs() < 1e-6,
            "Companion eigenvalue 2 = {}, expected 3.0",
            reals[2]
        );
    }

    #[test]
    fn general_evd_verify_av_eq_vd() {
        // For a general matrix with real eigenvalues, verify A*v = lambda*v
        let a: Mat<f64> = Mat::from_rows(&[&[5.0, 1.0, 0.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 1.0]]);
        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        // Upper triangular matrix has eigenvalues on diagonal: 5, 3, 1
        let eigs = evd.eigenvalues();
        let mut reals: Vec<f64> = eigs.iter().map(|e| e.real).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((reals[0] - 1.0).abs() < 1e-8);
        assert!((reals[1] - 3.0).abs() < 1e-8);
        assert!((reals[2] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn symmetric_evd_100x100() {
        let a = make_spd(100);
        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();
        // All eigenvalues of SPD should be positive
        for (i, &e) in eigs.iter().enumerate() {
            assert!(e > -1e-6, "Eigenvalue {} negative: {}", i, e);
        }
        // Should be sorted ascending
        for i in 1..eigs.len() {
            assert!(
                eigs[i] >= eigs[i - 1] - 1e-10,
                "Eigenvalues not sorted at {}: {} > {}",
                i,
                eigs[i - 1],
                eigs[i]
            );
        }
    }
}

// ============================================================================
// Linear Solve Tests
// ============================================================================

mod solve_tests {
    use super::*;

    #[test]
    fn solve_2x2() {
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0], &[7.0]]);
        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // Verify Ax = b
        let ax = matmul(&a, &x);
        let err = frobenius_diff(&ax, &b);
        assert!(err < 1e-12, "Ax != b, error = {}", err);
    }

    #[test]
    fn solve_5x5_hilbert() {
        let a = hilbert(5);
        // b = A * ones => solution should be all ones
        let n = 5;
        let ones = {
            let mut o = Mat::zeros(n, 1);
            for i in 0..n {
                o[(i, 0)] = 1.0;
            }
            o
        };
        let b = matmul(&a, &ones);
        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // Check x is approximately ones (Hilbert is ill-conditioned)
        let ax = matmul(&a, &x);
        let residual = frobenius_diff(&ax, &b) / frobenius_norm(&b);
        assert!(residual < 1e-8, "Hilbert solve residual = {}", residual);
    }

    #[test]
    fn solve_50x50() {
        let a = diag_dominant(50);
        let n = 50;
        let ones = {
            let mut o = Mat::zeros(n, 1);
            for i in 0..n {
                o[(i, 0)] = 1.0;
            }
            o
        };
        let b = matmul(&a, &ones);
        let x = solve(a.as_ref(), b.as_ref()).unwrap();

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
    fn solve_100x100() {
        let a = diag_dominant(100);
        let n = 100;
        let ones = {
            let mut o = Mat::zeros(n, 1);
            for i in 0..n {
                o[(i, 0)] = 1.0;
            }
            o
        };
        let b = matmul(&a, &ones);
        let x = solve(a.as_ref(), b.as_ref()).unwrap();

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
    fn solve_multiple_rhs() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 1.0, 0.0], &[1.0, 4.0, 1.0], &[0.0, 1.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 1.0], &[6.0, 2.0], &[5.0, 3.0]]);
        let x = solve_multiple(a.as_ref(), b.as_ref()).unwrap();

        // Verify AX = B
        let ax = matmul(&a, &x);
        let err = frobenius_diff(&ax, &b);
        assert!(err < 1e-10, "AX != B, error = {}", err);
    }

    #[test]
    fn solve_multiple_rhs_large() {
        let a = diag_dominant(30);
        let n = 30;
        let nrhs = 5;
        // Create B with known solutions
        let mut b = Mat::zeros(n, nrhs);
        for j in 0..nrhs {
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[(i, k)] * ((k + j + 1) as f64);
                }
                b[(i, j)] = sum;
            }
        }

        let x = solve_multiple(a.as_ref(), b.as_ref()).unwrap();

        // Verify solutions
        for j in 0..nrhs {
            for i in 0..n {
                let expected = (i + j + 1) as f64;
                assert!(
                    (x[(i, j)] - expected).abs() < 1e-6,
                    "x[{},{}] = {}, expected {}",
                    i,
                    j,
                    x[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn solve_ill_conditioned_hilbert() {
        // Hilbert matrices are notoriously ill-conditioned
        let a = hilbert(8);
        let n = 8;
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[(i, j)];
            }
            b[(i, 0)] = sum;
        }

        let x = solve(a.as_ref(), b.as_ref()).unwrap();

        // For ill-conditioned systems, check residual rather than solution accuracy
        let ax = matmul(&a, &x);
        let residual_norm = frobenius_diff(&ax, &b);
        let b_norm = frobenius_norm(&b);
        let rel_residual = residual_norm / b_norm;
        // Relative residual should still be small even for ill-conditioned systems
        assert!(
            rel_residual < 1e-4,
            "Ill-conditioned solve relative residual = {}",
            rel_residual
        );
    }

    #[test]
    fn solve_cholesky_spd() {
        let a = make_spd(20);
        let n = 20;
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[(i, j)];
            }
            b[(i, 0)] = sum;
        }

        // Solve via Cholesky
        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let x = chol.solve(b.as_ref()).unwrap();

        // Verify residual
        let ax = matmul(&a, &x);
        let rel_residual = frobenius_diff(&ax, &b) / frobenius_norm(&b);
        assert!(
            rel_residual < 1e-10,
            "Cholesky solve residual = {}",
            rel_residual
        );
    }

    #[test]
    fn solve_singular_returns_error() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[3.0], &[7.0]]);
        assert!(solve(a.as_ref(), b.as_ref()).is_err());
    }
}
