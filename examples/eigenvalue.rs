//! Eigenvalue computation example.
//!
//! Demonstrates computing eigenvalues and eigenvectors for dense and sparse matrices.

use oxiblas::lapack::eigenvalue::syev;
use oxiblas::matrix::Mat;
use oxiblas_sparse::{CsrMatrix, CooMatrixBuilder};
use oxiblas_sparse::linalg::{Lanczos, LanczosConfig, WhichEigenvalues};

fn main() {
    println!("OxiBLAS - Eigenvalue Computation Example\n");

    // ========================================
    // Dense Symmetric Eigenvalue (LAPACK)
    // ========================================
    println!("=== Dense Symmetric Eigenvalue Problem ===\n");

    // Create a symmetric 3×3 matrix
    let a_data = vec![
        4.0, 1.0, 0.0,
        1.0, 4.0, 1.0,
        0.0, 1.0, 4.0,
    ];
    let mut a = Mat::from_slice(3, 3, &a_data);

    println!("Matrix A (symmetric):");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            print!("{:5.1}", a_data[i * 3 + j]);
        }
        println!(" ]");
    }
    println!();

    match syev(a.as_mut(), true) {  // compute_eigenvectors = true
        Ok((eigenvalues, eigenvectors)) => {
            println!("Eigenvalues: {:?}", eigenvalues);
            if let Some(v) = eigenvectors {
                println!("\nEigenvectors (as columns):");
                for i in 0..3 {
                    print!("  [");
                    for j in 0..3 {
                        print!("{:8.4}", v[(i, j)]);
                    }
                    println!(" ]");
                }
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();

    // ========================================
    // Sparse Symmetric Eigenvalue (Lanczos)
    // ========================================
    println!("=== Sparse Symmetric Eigenvalue (Lanczos Method) ===\n");

    let n = 100;
    let mut builder = CooMatrixBuilder::new(n, n);

    // Create a tridiagonal symmetric matrix
    for i in 0..n {
        builder.add(i, i, 2.0);
        if i > 0 {
            builder.add(i, i - 1, -1.0);
            builder.add(i - 1, i, -1.0);
        }
    }

    let a_sparse = builder.build().to_csr();

    println!("Sparse matrix: {}×{} tridiagonal", n, n);
    println!("Non-zeros: {}", a_sparse.nnz());
    println!("Computing 5 largest eigenvalues...\n");

    let config = LanczosConfig {
        num_eigenvalues: 5,
        max_iterations: 100,
        tolerance: 1e-10,
        which: WhichEigenvalues::LargestMagnitude,
        compute_eigenvectors: false,
        krylov_dimension: 15,
        full_reorthogonalization: true,
    };

    let mut lanczos = Lanczos::new(config);
    match lanczos.compute(&a_sparse, None) {
        Ok(result) => {
            println!("Converged: {}", result.converged);
            println!("Iterations: {}", result.iterations);
            println!("\nTop 5 eigenvalues:");
            for (i, &lambda) in result.eigenvalues.iter().take(5).enumerate() {
                println!("  λ[{}] = {:.10}", i, lambda);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n✓ Eigenvalue computations completed successfully!");
}
