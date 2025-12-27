//! LAPACK linear system solving example.
//!
//! Demonstrates solving linear systems Ax = b using various methods.

use oxiblas::lapack::solve::{solve, solve_triangular, TriangularKind};
use oxiblas::lapack::factorization::lu;
use oxiblas::matrix::Mat;

fn main() {
    println!("OxiBLAS - LAPACK Linear System Solving Example\n");

    // ========================================
    // General Linear System: Ax = b
    // ========================================
    println!("=== General Linear System (LU factorization) ===\n");

    // Create a well-conditioned 3×3 system
    let a_data = vec![
        4.0, 1.0, 0.0,  // Diagonally dominant
        1.0, 4.0, 1.0,
        0.0, 1.0, 4.0,
    ];
    let a = Mat::from_slice(3, 3, &a_data);
    let b = Mat::from_slice(3, 1, &[5.0, 6.0, 5.0]);

    println!("System:");
    println!("  A = [[4, 1, 0],");
    println!("       [1, 4, 1],");
    println!("       [0, 1, 4]]");
    println!("  b = [5, 6, 5]^T\n");

    match solve(a.as_ref(), b.as_ref()) {
        Ok(x) => {
            println!("Solution x:");
            for i in 0..3 {
                println!("  x[{}] = {:.6}", i, x[(i, 0)]);
            }
            println!();

            // Verify: Ax = b
            let mut ax = Mat::zeros(3, 1);
            oxiblas::blas::gemv(1.0, a.as_ref(), &x.as_slice(), 0.0, ax.as_mut_slice());
            println!("Verification (Ax should equal b):");
            for i in 0..3 {
                println!("  (Ax)[{}] = {:.6}, b[{}] = {:.6}", i, ax[(i, 0)], i, b[(i, 0)]);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();

    // ========================================
    // Triangular System
    // ========================================
    println!("=== Triangular System (Upper) ===\n");

    let u_data = vec![
        2.0, 1.0, 3.0,  // Upper triangular
        0.0, 3.0, 1.0,
        0.0, 0.0, 2.0,
    ];
    let u = Mat::from_slice(3, 3, &u_data);
    let b_tri = Mat::from_slice(3, 1, &[14.0, 7.0, 4.0]);

    println!("System:");
    println!("  U = [[2, 1, 3],");
    println!("       [0, 3, 1],");
    println!("       [0, 0, 2]]");
    println!("  b = [14, 7, 4]^T\n");

    match solve_triangular(u.as_ref(), b_tri.as_ref(), TriangularKind::Upper) {
        Ok(x) => {
            println!("Solution x:");
            for i in 0..3 {
                println!("  x[{}] = {:.6}", i, x[(i, 0)]);
            }
            println!("  Expected: x = [2, 1, 2]^T\n");
        }
        Err(e) => println!("Error: {:?}", e),
    }

    // ========================================
    // LU Factorization (manual)
    // ========================================
    println!("=== LU Factorization ===\n");

    let a_lu_data = vec![
        2.0, 1.0, 1.0,
        4.0, 3.0, 3.0,
        8.0, 7.0, 9.0,
    ];
    let mut a_lu = Mat::from_slice(3, 3, &a_lu_data);

    println!("Original matrix A:");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            print!("{:5.1}", a_lu[(i, j)]);
        }
        println!(" ]");
    }

    match lu(a_lu.as_mut()) {
        Ok(ipiv) => {
            println!("\nLU factorization successful!");
            println!("Pivot indices: {:?}", ipiv);
            println!("\nFactored matrix (L below diag, U on and above diag):");
            for i in 0..3 {
                print!("  [");
                for j in 0..3 {
                    print!("{:8.4}", a_lu[(i, j)]);
                }
                println!(" ]");
            }
        }
        Err(e) => println!("LU factorization failed: {:?}", e),
    }

    println!("\n✓ All LAPACK examples completed successfully!");
}
