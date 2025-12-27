//! LAPACK decompositions and solvers demonstration.
//!
//! This example shows how to use LAPACK operations:
//! - LU decomposition and solving linear systems
//! - QR decomposition
//! - Cholesky decomposition (for positive definite matrices)
//! - SVD (Singular Value Decomposition)
//! - Eigenvalue decomposition
//!
//! Run with: cargo run --example lapack_decompositions

use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::evd::SymmetricEvd;
use oxiblas_lapack::lu::Lu;
use oxiblas_lapack::qr::Qr;
use oxiblas_lapack::solve::solve;
use oxiblas_lapack::svd::Svd;
use oxiblas_lapack::utils::{det, inv};
use oxiblas_matrix::Mat;

fn main() {
    println!("=== OxiBLAS LAPACK Decompositions ===\n");

    // ========================================
    // LU Decomposition
    // ========================================
    println!("--- LU Decomposition ---");

    let a_lu = Mat::from_rows(&[&[2.0, 1.0, 1.0], &[4.0, -6.0, 0.0], &[-2.0, 7.0, 2.0]]);

    println!("Matrix A:");
    print_matrix_mat(&a_lu);

    let lu = Lu::compute_auto(a_lu.as_ref()).expect("LU decomposition failed");
    let det_a = lu.determinant();
    println!("Determinant: {}", det_a);
    println!(
        "Expected: 2*(-6)*2 + 1*0*(-2) + 1*4*7 - 1*(-6)*(-2) - 2*0*7 - 1*4*2 = -24 + 28 - 12 - 8 = -16\n"
    );

    // Solve A * x = b
    let b_mat = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
    println!("Solving A * x = b where b = [1, 2, 3]");
    let x_mat = lu.solve(b_mat.as_ref()).expect("Solve failed");
    println!("Solution x:");
    print_matrix_mat(&x_mat);

    // Verify solution
    let b_check = matrix_mult(&a_lu, &x_mat);
    println!("Verification A * x:");
    print_matrix_mat(&b_check);
    println!();

    // Direct solve interface
    let a_solve = Mat::from_rows(&[&[3.0, 1.0], &[1.0, 2.0]]);
    let b_solve = Mat::from_rows(&[&[9.0], &[8.0]]);
    println!("Direct solve example:");
    println!("A = [[3, 1], [1, 2]]");
    println!("b = [[9], [8]]");
    let x_solve = solve(a_solve.as_ref(), b_solve.as_ref()).expect("Solve failed");
    println!("Solution x:");
    print_matrix_mat(&x_solve);
    println!("Expected: x = [[2], [3]] (since 3*2 + 1*3 = 9, 1*2 + 2*3 = 8)\n");

    // ========================================
    // QR Decomposition
    // ========================================
    println!("--- QR Decomposition ---");

    let a_qr = Mat::from_rows(&[&[12.0, -51.0], &[6.0, 167.0], &[-4.0, 24.0]]);

    println!("Matrix A (3x2):");
    print_matrix_mat(&a_qr);

    let qr = Qr::compute(a_qr.as_ref()).expect("QR decomposition failed");
    let q = qr.q();
    let r = qr.r();

    println!("Q matrix (orthogonal):");
    print_matrix_mat(&q);

    println!("R matrix (upper triangular):");
    print_matrix_mat(&r);

    // Verify Q is orthogonal: Q^T * Q = I
    let qtq = matrix_mult_transpose_left(&q, &q);
    println!("Q^T * Q (should be identity):");
    print_matrix_mat(&qtq);

    // Verify A = Q * R
    let qr_product = matrix_mult(&q, &r);
    println!("Q * R (should equal A):");
    print_matrix_mat(&qr_product);
    println!();

    // ========================================
    // Cholesky Decomposition
    // ========================================
    println!("--- Cholesky Decomposition ---");

    // Create a symmetric positive definite matrix
    let a_chol = Mat::from_rows(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

    println!("Symmetric positive definite matrix A:");
    print_matrix_mat(&a_chol);

    let chol = Cholesky::compute_auto(a_chol.as_ref()).expect("Cholesky decomposition failed");
    let l = chol.l_factor();

    println!("L matrix (lower triangular, A = L * L^T):");
    print_matrix_mat(&l);

    // Verify A = L * L^T
    let llt = matrix_mult_transpose_right(&l, &l);
    println!("L * L^T (should equal A):");
    print_matrix_mat(&llt);
    println!();

    // ========================================
    // SVD (Singular Value Decomposition)
    // ========================================
    println!("--- Singular Value Decomposition (SVD) ---");

    let a_svd = Mat::from_rows(&[&[3.0, 1.0], &[1.0, 3.0], &[1.0, 1.0]]);

    println!("Matrix A (3x2):");
    print_matrix_mat(&a_svd);

    let svd = Svd::compute(a_svd.as_ref()).expect("SVD failed");
    let singular_values = svd.singular_values();
    let u_mat = svd.u();
    let vt_mat = svd.vt();

    println!("\nSingular values:");
    for (i, &s) in singular_values.iter().enumerate() {
        println!("  σ_{} = {:.6}", i + 1, s);
    }

    println!("\nU matrix (left singular vectors):");
    print_matrix(u_mat);

    println!("V^T matrix (right singular vectors, transposed):");
    print_matrix(vt_mat);

    // Verify A = U * Σ * V^T
    println!("Verification: reconstructing A from U * Σ * V^T");
    let a_reconstructed = reconstruct_from_svd(&svd);
    println!("Reconstructed A:");
    print_matrix_mat(&a_reconstructed);
    println!();

    // ========================================
    // Symmetric Eigenvalue Decomposition
    // ========================================
    println!("--- Symmetric Eigenvalue Decomposition ---");

    let a_eig = Mat::from_rows(&[&[4.0, 1.0, 0.0], &[1.0, 4.0, 1.0], &[0.0, 1.0, 4.0]]);

    println!("Symmetric matrix A:");
    print_matrix_mat(&a_eig);

    let evd = SymmetricEvd::compute(a_eig.as_ref()).expect("EVD failed");
    let eigenvalues = evd.eigenvalues();
    let eigenvectors = evd.eigenvectors();

    println!("\nEigenvalues:");
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        println!("  λ_{} = {:.6}", i + 1, lambda);
    }

    println!("\nEigenvectors (as columns of V):");
    print_matrix(eigenvectors);

    println!("\nVerification: A * v_i = λ_i * v_i for first eigenvector:");
    let v1 = extract_column_from_ref(&eigenvectors, 0);
    let av1 = matrix_vector_mult(&a_eig, &v1);
    let lambda_v1 = scale_vector(&v1, eigenvalues[0]);
    println!("A * v_1 = {:?}", av1);
    println!("λ_1 * v_1 = {:?}", lambda_v1);
    println!("Error: {:?}\n", vec_diff(&av1, &lambda_v1));

    // ========================================
    // Matrix Utilities
    // ========================================
    println!("--- Matrix Utilities ---");

    let a_util = Mat::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);
    println!("Matrix A:");
    print_matrix_mat(&a_util);

    // Determinant
    let det_val = det(a_util.as_ref()).expect("Determinant failed");
    println!("Determinant: {}", det_val);
    println!("Expected: 4*6 - 7*2 = 24 - 14 = 10\n");

    // Inverse
    let a_inv = inv(a_util.as_ref()).expect("Inverse failed");
    println!("Inverse A^(-1):");
    print_matrix_mat(&a_inv);

    // Verify A * A^(-1) = I
    let identity = matrix_mult(&a_util, &a_inv);
    println!("A * A^(-1) (should be identity):");
    print_matrix_mat(&identity);

    println!("\n=== All LAPACK operations completed successfully! ===");
}

// Helper functions

fn print_matrix_mat(m: &Mat<f64>) {
    for i in 0..m.nrows() {
        print!("  [");
        for j in 0..m.ncols() {
            print!("{:8.4}", m[(i, j)]);
        }
        println!("]");
    }
}

fn print_matrix(m: MatRef<'_, f64>) {
    for i in 0..m.nrows() {
        print!("  [");
        for j in 0..m.ncols() {
            print!("{:8.4}", m[(i, j)]);
        }
        println!("]");
    }
}

fn matrix_vector_mult(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    assert_eq!(n, x.len());
    let mut result = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            result[i] += a[(i, j)] * x[j];
        }
    }
    result
}

fn matrix_mult(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    assert_eq!(k, b.nrows());
    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                let val = c[(i, j)] + a[(i, p)] * b[(p, j)];
                c.set(i, j, val);
            }
        }
    }
    c
}

fn matrix_mult_transpose_left(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let k = a.nrows();
    let m = a.ncols();
    let n = b.ncols();
    assert_eq!(k, b.nrows());
    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                let val = c[(i, j)] + a[(p, i)] * b[(p, j)];
                c.set(i, j, val);
            }
        }
    }
    c
}

fn matrix_mult_transpose_right(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.nrows();
    assert_eq!(k, b.ncols());
    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                let val = c[(i, j)] + a[(i, p)] * b[(j, p)];
                c.set(i, j, val);
            }
        }
    }
    c
}

fn reconstruct_from_svd(svd: &Svd<f64>) -> Mat<f64> {
    let u = svd.u();
    let s = svd.singular_values();
    let vt = svd.vt();

    let m = u.nrows();
    let n = vt.ncols();
    let mut result = Mat::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            for k in 0..s.len() {
                let val = result[(i, j)] + u[(i, k)] * s[k] * vt[(k, j)];
                result.set(i, j, val);
            }
        }
    }
    result
}

use oxiblas_matrix::MatRef;

fn extract_column_from_ref(m: &MatRef<'_, f64>, col: usize) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, col)]).collect()
}

fn scale_vector(v: &[f64], scalar: f64) -> Vec<f64> {
    v.iter().map(|&x| x * scalar).collect()
}

fn vec_diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
