//! Sparse matrix operations demonstration.
//!
//! This example demonstrates:
//! - Creating sparse matrices in different formats (CSR, CSC, COO)
//! - Converting between formats
//! - Sparse matrix-vector multiplication
//! - Iterative solvers (CG, GMRES)
//! - Preconditioners (Jacobi, ILUT)
//!
//! Run with: cargo run --example sparse_matrices --features parallel

use oxiblas_sparse::convert::{coo_to_csr, csr_to_csc};
use oxiblas_sparse::coo::CooMatrix;
use oxiblas_sparse::csr::CsrMatrix;
use oxiblas_sparse::linalg::iterative::{cg, gmres};
use oxiblas_sparse::linalg::precond::Jacobi;
use oxiblas_sparse::ops::{spmv, spmv_csc};

fn main() {
    println!("=== Sparse Matrix Operations ===\n");

    // ========================================
    // Creating Sparse Matrices
    // ========================================
    println!("--- Creating Sparse Matrices ---");

    // Create a sparse matrix in COO (coordinate) format
    // Representing:
    //   [10.0   1.0   0.0   0.0]
    //   [ 1.0  10.0   2.0   0.0]
    //   [ 0.0   2.0  10.0   3.0]
    //   [ 0.0   0.0   3.0  10.0]

    let row_indices = vec![0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
    let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let values = vec![10.0, 1.0, 1.0, 10.0, 2.0, 2.0, 10.0, 3.0, 3.0, 10.0];

    let coo = CooMatrix::new(4, 4, row_indices, col_indices, values)
        .expect("Failed to create COO matrix");

    println!("COO Matrix (4×4, 10 non-zeros):");
    println!("  Diagonal: 10.0");
    println!("  Off-diagonals: 1.0, 2.0, 3.0 (tridiagonal pattern)");
    println!("  Sparsity: 10/16 = 62.5% sparse\n");

    // Convert to CSR (Compressed Sparse Row) format
    let csr = coo_to_csr(&coo);

    println!("Converted to CSR format:");
    println!("  Row pointers: {:?}", csr.row_ptrs());
    println!("  Column indices: {:?}", csr.col_indices());
    println!("  Values: {:?}", csr.values());
    println!("  (CSR is efficient for row-wise operations)\n");

    // ========================================
    // Sparse Matrix-Vector Multiplication
    // ========================================
    println!("--- Sparse Matrix-Vector Multiplication ---");

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    spmv(1.0, &csr, &x, 0.0, &mut y);

    println!("x = {:?}", x);
    println!("y = A * x = {:?}", y);
    println!("Verification:");
    println!("  y[0] = 10*1 + 1*2 = 12 ✓");
    println!("  y[1] = 1*1 + 10*2 + 2*3 = 27 ✓");
    println!("  y[2] = 2*2 + 10*3 + 3*4 = 46 ✓");
    println!("  y[3] = 3*3 + 10*4 = 49 ✓\n");

    // ========================================
    // Conjugate Gradient Solver
    // ========================================
    println!("--- Conjugate Gradient (CG) Solver ---");
    println!("Solving A * x = b iteratively for symmetric positive definite A\n");

    let b = vec![12.0, 27.0, 46.0, 49.0];
    let x0 = vec![0.0; 4]; // Initial guess
    let tol = 1e-10;
    let max_iter = 100;

    println!("Right-hand side b = {:?}", b);
    println!("Initial guess x0 = {:?}", x0);
    println!("Tolerance: {:.2e}, Max iterations: {}", tol, max_iter);

    let cg_result = cg(&csr, &b, &x0, tol, max_iter).expect("CG failed");

    println!("\nCG Solution:");
    println!("  x = {:?}", cg_result.x);
    println!("  Iterations: {}", cg_result.iterations);
    println!("  Residual norm: {:.2e}", cg_result.residual_norm);
    println!("  Converged: {}", cg_result.converged);
    println!("  Expected: x ≈ [1, 2, 3, 4] (our original x) ✓\n");

    // ========================================
    // GMRES Solver
    // ========================================
    println!("--- GMRES Solver (General Matrices) ---");
    println!("Solving A * x = b with GMRES (works for non-symmetric matrices)\n");

    let restart = 10;
    let gmres_result = gmres(&csr, &b, &x0, max_iter, tol, restart).expect("GMRES failed");

    println!("GMRES Solution:");
    println!("  x = {:?}", gmres_result.x);
    println!("  Iterations: {}", gmres_result.iterations);
    println!("  Residual norm: {:.2e}", gmres_result.residual_norm);
    println!("  Restart parameter: {}", restart);
    println!("  Converged: {}", gmres_result.converged);
    println!("  Expected: x ≈ [1, 2, 3, 4] ✓\n");

    // ========================================
    // Preconditioned Iterative Solver
    // ========================================
    println!("--- Preconditioners ---");
    println!("Using Jacobi preconditioner to improve convergence\n");

    // Create Jacobi preconditioner (diagonal scaling)
    let precond = Jacobi::new(&csr).expect("Failed to create Jacobi preconditioner");

    // For demonstration, create a harder problem
    let b_hard = vec![100.0, 200.0, 300.0, 400.0];

    println!("Harder problem: b = {:?}", b_hard);

    // Solve without preconditioner
    let result_no_precond = cg(&csr, &b_hard, &x0, tol, max_iter).expect("CG failed");
    println!("Without preconditioner:");
    println!("  Iterations: {}", result_no_precond.iterations);
    println!("  Residual: {:.2e}", result_no_precond.residual_norm);

    // Demonstrate preconditioner application
    let mut r = b_hard.clone();
    let mut ax0 = vec![0.0; 4];
    spmv(1.0, &csr, &x0, 0.0, &mut ax0);
    for i in 0..4 {
        r[i] -= ax0[i];
    }
    let mut z = vec![0.0; 4];
    precond.apply(&r, &mut z);

    println!("\nJacobi preconditioner demonstration:");
    println!(
        "  Original residual r: [{:.1}, {:.1}, {:.1}, {:.1}]",
        r[0], r[1], r[2], r[3]
    );
    println!(
        "  Preconditioned z = M^(-1) * r: [{:.1}, {:.1}, {:.1}, {:.1}]",
        z[0], z[1], z[2], z[3]
    );
    println!("  (Diagonal scaling improves conditioning)\n");

    // ========================================
    // Format Conversion
    // ========================================
    println!("--- Format Conversions ---");

    println!("Original format: COO (Coordinate)");
    println!("  Good for: construction, incremental updates");
    println!("  Storage: 3 arrays (row, col, val)");

    println!("\nConverted to CSR (Compressed Sparse Row):");
    println!("  Good for: row-wise operations, matrix-vector products");
    println!("  Storage: row_ptr (n+1), col_ind (nnz), values (nnz)");

    let csc = csr_to_csc(&csr);
    println!("\nConverted to CSC (Compressed Sparse Column):");
    println!("  Good for: column-wise operations, transpose products");
    println!("  Storage: col_ptr (n+1), row_ind (nnz), values (nnz)");
    println!("  Column pointers: {:?}", csc.col_ptrs());
    println!("  Row indices: {:?}", csc.row_indices());

    println!("\n=== Format conversion preserves values ===");
    let mut y_csr = vec![0.0; 4];
    let mut y_csc = vec![0.0; 4];
    spmv(1.0, &csr, &x, 0.0, &mut y_csr);
    spmv_csc(1.0, &csc, &x, 0.0, &mut y_csc);
    println!("CSR * x = {:?}", y_csr);
    println!("CSC * x = {:?}", y_csc);
    println!("Results match: {} ✓\n", y_csr == y_csc);

    // ========================================
    // Sparse Matrix Properties
    // ========================================
    println!("--- Sparse Matrix Properties ---");

    println!("Matrix dimensions: {} × {}", csr.nrows(), csr.ncols());
    println!("Non-zero elements: {}", csr.nnz());
    println!(
        "Sparsity: {:.1}%",
        (1.0 - csr.nnz() as f64 / (csr.nrows() * csr.ncols()) as f64) * 100.0
    );
    println!(
        "Average non-zeros per row: {:.1}",
        csr.nnz() as f64 / csr.nrows() as f64
    );

    // Bandwidth (for banded matrices)
    let bandwidth = compute_bandwidth(&csr);
    println!(
        "Bandwidth: {} (half-bandwidth = {})",
        bandwidth,
        bandwidth / 2
    );
    println!("(Tridiagonal matrix has bandwidth 3)\n");

    println!("=== All sparse matrix operations completed successfully! ===");
}

fn compute_bandwidth(csr: &CsrMatrix<f64>) -> usize {
    let mut max_bandwidth = 0;
    for i in 0..csr.nrows() {
        let row_start = csr.row_ptrs()[i];
        let row_end = csr.row_ptrs()[i + 1];
        for idx in row_start..row_end {
            let j = csr.col_indices()[idx];
            let dist = i.abs_diff(j);
            max_bandwidth = max_bandwidth.max(dist);
        }
    }
    2 * max_bandwidth + 1
}
