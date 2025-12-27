//! Basic BLAS operations example.
//!
//! Demonstrates fundamental vector and matrix operations.

use oxiblas::blas::{dot, axpy, gemv, gemm};
use oxiblas::matrix::Mat;

fn main() {
    println!("OxiBLAS - Basic BLAS Operations Example\n");

    // ========================================
    // BLAS Level 1: Vector Operations
    // ========================================
    println!("=== BLAS Level 1: Vector Operations ===\n");

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![5.0, 6.0, 7.0, 8.0];

    // Dot product: x^T · y
    let result = dot(&x, &y);
    println!("dot(x, y) = {}", result);
    println!("  x = {:?}", x);
    println!("  y = {:?}", y);
    println!("  Expected: 1*5 + 2*6 + 3*7 + 4*8 = 70\n");

    // AXPY: y := alpha*x + y
    let mut y_axpy = y.clone();
    let alpha = 2.0;
    axpy(alpha, &x, &mut y_axpy);
    println!("axpy({}, x, y) = {:?}", alpha, y_axpy);
    println!("  Expected: 2*[1,2,3,4] + [5,6,7,8] = [7,10,13,16]\n");

    // ========================================
    // BLAS Level 2: Matrix-Vector Operations
    // ========================================
    println!("=== BLAS Level 2: Matrix-Vector Operations ===\n");

    // Create a 3×3 matrix
    let a_data = vec![
        1.0, 2.0, 3.0,  // row 1
        4.0, 5.0, 6.0,  // row 2
        7.0, 8.0, 9.0,  // row 3
    ];
    let a = Mat::from_slice(3, 3, &a_data);
    let x_mv = vec![1.0, 2.0, 3.0];
    let mut y_mv = vec![0.0; 3];

    // GEMV: y := alpha*A*x + beta*y
    gemv(1.0, a.as_ref(), &x_mv, 0.0, &mut y_mv);
    println!("gemv(A, x):");
    println!("  A = [[1, 2, 3],");
    println!("       [4, 5, 6],");
    println!("       [7, 8, 9]]");
    println!("  x = {:?}", x_mv);
    println!("  y = {:?}", y_mv);
    println!("  Expected: [14, 32, 50] = [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3]\n");

    // ========================================
    // BLAS Level 3: Matrix-Matrix Operations
    // ========================================
    println!("=== BLAS Level 3: Matrix-Matrix Operations ===\n");

    // Create two 2×2 matrices
    let a_mm = Mat::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b_mm = Mat::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    let mut c_mm = Mat::from_slice(2, 2, &[0.0; 4]);

    // GEMM: C := alpha*A*B + beta*C
    gemm(1.0, a_mm.as_ref(), b_mm.as_ref(), 0.0, c_mm.as_mut());

    println!("gemm(A, B):");
    println!("  A = [[1, 2],");
    println!("       [3, 4]]");
    println!("  B = [[5, 6],");
    println!("       [7, 8]]");
    println!("  C = [[{}, {}],", c_mm[(0, 0)], c_mm[(0, 1)]);
    println!("       [{}, {}]]", c_mm[(1, 0)], c_mm[(1, 1)]);
    println!("  Expected: [[19, 22], [43, 50]]\n");

    println!("✓ All BLAS operations completed successfully!");
}
