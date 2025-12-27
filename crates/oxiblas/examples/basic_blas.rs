//! Basic BLAS operations demonstration.
//!
//! This example shows how to use fundamental BLAS operations:
//! - Level 1: Vector operations (dot, axpy, nrm2)
//! - Level 2: Matrix-vector operations (gemv)
//! - Level 3: Matrix-matrix operations (gemm)
//!
//! Run with: cargo run --example basic_blas

use oxiblas_blas::level1::{axpy, dot, nrm2, scal};
use oxiblas_blas::level2::{GemvTrans, gemv};
use oxiblas_blas::level3::gemm;
use oxiblas_matrix::Mat;

fn main() {
    println!("=== OxiBLAS Basic Operations ===\n");

    // ========================================
    // BLAS Level 1: Vector Operations
    // ========================================
    println!("--- BLAS Level 1 (Vector Operations) ---");

    // Dot product: x · y
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![5.0, 6.0, 7.0, 8.0];
    let dot_result = dot(&x, &y);
    println!("x = {:?}", x);
    println!("y = {:?}", y);
    println!("dot(x, y) = {}", dot_result);
    println!(
        "Expected: 1*5 + 2*6 + 3*7 + 4*8 = {}\n",
        1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0
    );

    // AXPY: y = alpha * x + y
    let mut y_axpy = vec![1.0, 2.0, 3.0, 4.0];
    let x_axpy = vec![10.0, 20.0, 30.0, 40.0];
    let alpha = 2.5;
    println!("Before AXPY: y = {:?}", y_axpy);
    axpy(alpha, &x_axpy, &mut y_axpy);
    println!("After AXPY(alpha={}, x, y): y = {:?}", alpha, y_axpy);
    println!("Expected: [1 + 2.5*10, 2 + 2.5*20, 3 + 2.5*30, 4 + 2.5*40] = [26, 52, 78, 104]\n");

    // NRM2: Euclidean norm ||x||_2
    let x_norm = vec![3.0, 4.0];
    let norm = nrm2(&x_norm);
    println!("x = {:?}", x_norm);
    println!("||x||_2 = {}", norm);
    println!("Expected: sqrt(3² + 4²) = sqrt(25) = 5.0\n");

    // SCAL: x = alpha * x
    let mut x_scal = vec![1.0, 2.0, 3.0, 4.0];
    println!("Before SCAL: x = {:?}", x_scal);
    scal(0.5, &mut x_scal);
    println!("After SCAL(0.5, x): x = {:?}\n", x_scal);

    // ========================================
    // BLAS Level 2: Matrix-Vector Operations
    // ========================================
    println!("--- BLAS Level 2 (Matrix-Vector Operations) ---");

    // GEMV: y = alpha * A * x + beta * y
    let a = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let x_gemv = vec![1.0, 2.0, 3.0];
    let mut y_gemv = vec![0.0, 0.0];

    println!("Matrix A (2x3):");
    for i in 0..2 {
        print!("  [");
        for j in 0..3 {
            print!("{:4.1}", a[(i, j)]);
        }
        println!("]");
    }
    println!("Vector x: {:?}", x_gemv);

    gemv(
        GemvTrans::NoTrans,
        1.0,
        a.as_ref(),
        &x_gemv,
        0.0,
        &mut y_gemv,
    );
    println!("y = A * x = {:?}", y_gemv);
    println!("Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]\n");

    // GEMV with transpose
    let mut y_trans = vec![0.0, 0.0, 0.0];
    let x_trans = vec![1.0, 2.0];
    gemv(
        GemvTrans::Trans,
        1.0,
        a.as_ref(),
        &x_trans,
        0.0,
        &mut y_trans,
    );
    println!("y = A^T * x = {:?}", y_trans);
    println!("Expected: [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2] = [9, 12, 15]\n");

    // ========================================
    // BLAS Level 3: Matrix-Matrix Operations
    // ========================================
    println!("--- BLAS Level 3 (Matrix-Matrix Operations) ---");

    // GEMM: C = alpha * A * B + beta * C
    let a_gemm = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let b_gemm = Mat::from_rows(&[&[7.0, 8.0], &[9.0, 10.0], &[11.0, 12.0]]);
    let mut c_gemm = Mat::zeros(2, 2);

    println!("Matrix A (2x3):");
    for i in 0..2 {
        print!("  [");
        for j in 0..3 {
            print!("{:4.1}", a_gemm[(i, j)]);
        }
        println!("]");
    }

    println!("Matrix B (3x2):");
    for i in 0..3 {
        print!("  [");
        for j in 0..2 {
            print!("{:4.1}", b_gemm[(i, j)]);
        }
        println!("]");
    }

    gemm(1.0, a_gemm.as_ref(), b_gemm.as_ref(), 0.0, c_gemm.as_mut());

    println!("Matrix C = A * B (2x2):");
    for i in 0..2 {
        print!("  [");
        for j in 0..2 {
            print!("{:6.1}", c_gemm[(i, j)]);
        }
        println!("]");
    }
    println!("Expected:");
    println!("  [ 58.0  64.0]  (1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12)");
    println!("  [139.0 154.0]  (4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12)\n");

    // GEMM with scaling
    let mut c_scaled = Mat::filled(2, 2, 100.0);
    println!("Initial C filled with 100.0");
    gemm(
        2.0,
        a_gemm.as_ref(),
        b_gemm.as_ref(),
        0.5,
        c_scaled.as_mut(),
    );
    println!("C = 2.0 * A * B + 0.5 * C:");
    for i in 0..2 {
        print!("  [");
        for j in 0..2 {
            print!("{:6.1}", c_scaled[(i, j)]);
        }
        println!("]");
    }
    println!("Expected: 2*(A*B) + 0.5*100 = [166, 178, 328, 358]\n");

    println!("=== All BLAS operations completed successfully! ===");
}
