//! Sparse iterative solver example.
//!
//! Demonstrates solving large sparse systems with iterative methods.

use oxiblas_sparse::{CsrMatrix, CooMatrixBuilder};
use oxiblas_sparse::linalg::{cg, gmres, bicgstab};

fn main() {
    println!("OxiBLAS - Sparse Iterative Solvers Example\n");

    // ========================================
    // Create a sparse SPD system (1D Poisson problem)
    // ========================================
    let n = 100;
    let mut builder = CooMatrixBuilder::new(n, n);

    // Tridiagonal: [-1, 2, -1] pattern
    for i in 0..n {
        builder.add(i, i, 2.0);
        if i > 0 {
            builder.add(i, i - 1, -1.0);
            builder.add(i - 1, i, -1.0);
        }
    }

    let a = builder.build().to_csr();
    let b: Vec<f64> = vec![1.0; n];  // Right-hand side
    let x0 = vec![0.0; n];  // Initial guess

    println!("Problem: {}×{} tridiagonal SPD matrix", n, n);
    println!("Non-zeros: {}", a.nnz());
    println!("Sparsity: {:.2}%\n", 100.0 * a.nnz() as f64 / (n * n) as f64);

    // ========================================
    // Conjugate Gradient (CG) - for SPD systems
    // ========================================
    println!("=== Conjugate Gradient (CG) ===");
    match cg(&a, &b, &x0, 1e-10, 1000) {
        Ok(result) => {
            println!("  Converged: {}", result.converged);
            println!("  Iterations: {}", result.iterations);
            println!("  Residual norm: {:.2e}", result.residual_norm);
            println!("  Solution (first 5 elements): {:?}", &result.x[..5.min(n)]);
        }
        Err(e) => println!("  Error: {:?}", e),
    }
    println!();

    // ========================================
    // GMRES - for general systems
    // ========================================
    println!("=== GMRES (Generalized Minimal Residual) ===");
    match gmres(&a, &b, &x0, 30, 1e-10, 1000) {
        Ok(result) => {
            println!("  Converged: {}", result.converged);
            println!("  Iterations: {}", result.iterations);
            println!("  Restarts: {}", result.restarts);
            println!("  Residual norm: {:.2e}", result.residual_norm);
            println!("  Solution (first 5 elements): {:?}", &result.x[..5.min(n)]);
        }
        Err(e) => println!("  Error: {:?}", e),
    }
    println!();

    // ========================================
    // BiCGStab - for non-symmetric systems
    // ========================================
    println!("=== BiCGStab (Bi-Conjugate Gradient Stabilized) ===");
    match bicgstab(&a, &b, &x0, 1e-10, 1000) {
        Ok(result) => {
            println!("  Converged: {}", result.converged);
            println!("  Iterations: {}", result.iterations);
            println!("  Residual norm: {:.2e}", result.residual_norm);
            println!("  Solution (first 5 elements): {:?}", &result.x[..5.min(n)]);
        }
        Err(e) => println!("  Error: {:?}", e),
    }

    println!("\n✓ All iterative solvers completed successfully!");
}
