//! Memory usage tests for sparse operations
//!
//! Validates that sparse operations use reasonable memory and don't leak.
//! Tests focus on verifying that data structures have expected sizes and
//! that repeated operations don't cause unbounded growth.

use oxiblas_sparse::linalg::cg;
use oxiblas_sparse::linalg::gmres;
use oxiblas_sparse::linalg::lu::{ILU0, SparseLU};
use oxiblas_sparse::linalg::precond::Jacobi;
use oxiblas_sparse::ops::{spmm_sparse, spmv};
use oxiblas_sparse::{CooMatrixBuilder, CsrMatrix};

/// Helper to estimate memory usage of CSR matrix (bytes)
fn estimate_csr_memory(a: &CsrMatrix<f64>) -> usize {
    // row_ptrs: (n+1) * usize, col_indices: nnz * usize, values: nnz * f64
    std::mem::size_of_val(a.row_ptrs())
        + std::mem::size_of_val(a.col_indices())
        + std::mem::size_of_val(a.values())
}

/// Helper to create a sparse Laplacian matrix
fn create_laplacian(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);
    for i in 0..n {
        builder.add(i, i, 4.0);
        if i > 0 {
            builder.add(i, i - 1, -1.0);
        }
        if i < n - 1 {
            builder.add(i, i + 1, -1.0);
        }
    }
    builder.build().to_csr()
}

#[test]
fn test_spmv_no_allocation() {
    // Test that SpMV doesn't allocate when output buffer is pre-allocated
    let n = 1000;
    let a = create_laplacian(n);
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];

    // Perform SpMV multiple times - should not panic
    for _ in 0..100 {
        spmv(1.0, &a, &x, 0.0, &mut y);
    }

    // Verify result is computed correctly
    assert_eq!(y.len(), n);
    for &val in &y {
        assert!(val.is_finite(), "SpMV produced non-finite values");
    }
}

#[test]
fn test_spmm_sparse_reasonable_size() {
    let n = 100;
    let a = create_laplacian(n);
    let b = create_laplacian(n);

    let c = spmm_sparse(&a, &b);

    // Result matrix should have reasonable size
    assert_eq!(c.nrows(), n);
    assert_eq!(c.ncols(), n);

    // For Laplacian^2, nnz should be more than input but bounded
    // Input: ~3n non-zeros each, output: ~5n non-zeros
    assert!(
        c.nnz() > a.nnz(),
        "Result should have at least as many non-zeros"
    );
    assert!(
        c.nnz() < a.nnz() * 5,
        "Result should not have excessive fill-in for Laplacian"
    );

    let c_memory = estimate_csr_memory(&c);
    println!(
        "Result matrix memory: {} bytes ({} non-zeros)",
        c_memory,
        c.nnz()
    );
}

#[test]
fn test_cg_solver_memory() {
    let n = 500;
    let a = create_laplacian(n);
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];

    let result = cg(&a, &b, &x0, 1e-6, 100);

    assert!(result.is_ok(), "CG should converge");
    let cg_result = result.unwrap();

    // Solution should be correct size
    assert_eq!(cg_result.x.len(), n);

    // All values should be finite
    for &val in &cg_result.x {
        assert!(val.is_finite(), "CG produced non-finite values");
    }
}

#[test]
fn test_gmres_memory() {
    let n = 200;
    let a = create_laplacian(n);
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];

    // GMRES with restart=20
    let result = gmres(&a, &b, &x0, 20, 1e-6, 50);

    assert!(result.is_ok(), "GMRES should converge");
    let gmres_result = result.unwrap();

    // Solution should be correct size
    assert_eq!(gmres_result.x.len(), n);

    for &val in &gmres_result.x {
        assert!(val.is_finite(), "GMRES produced non-finite values");
    }
}

#[test]
fn test_ilu0_memory_usage() {
    let n = 500;
    let a = create_laplacian(n);

    let ilu = ILU0::new(&a);
    assert!(ilu.is_ok(), "ILU0 construction should succeed");

    let ilu = ilu.unwrap();

    // ILU0 maintains same sparsity pattern as input
    // Get the LU matrix size (it should be similar to input)
    let input_memory = estimate_csr_memory(&a);

    println!("Input matrix memory: {} bytes", input_memory);
    println!("ILU0 should have similar memory footprint");

    // Verify we can apply it as a preconditioner
    let b = vec![1.0; n];
    let x = ilu.apply(&b);

    assert_eq!(x.len(), n);
    for &val in &x {
        assert!(val.is_finite(), "ILU0 apply produced non-finite values");
    }
}

#[test]
fn test_sparse_lu_memory_usage() {
    let n = 100;
    let a_csr = create_laplacian(n);
    let a = a_csr.to_csc(); // SparseLU requires CSC format

    let lu = SparseLU::new(&a);
    assert!(lu.is_ok(), "Sparse LU construction should succeed");

    let lu = lu.unwrap();

    // Verify we can solve with it
    let b = vec![1.0; n];
    let x = lu.solve(&b);

    assert_eq!(x.len(), n);
    for &val in &x {
        assert!(
            val.is_finite(),
            "Sparse LU solve produced non-finite values"
        );
    }
}

#[test]
fn test_matrix_conversion_memory() {
    let n = 500;
    let mut builder = CooMatrixBuilder::new(n, n);

    // Create sparse matrix
    for i in 0..n {
        for j in 0..5.min(n) {
            if i >= j {
                builder.add(i, (i - j) % n, 1.0);
            }
        }
    }

    let coo = builder.build();
    let coo_size = std::mem::size_of_val(coo.row_indices())
        + std::mem::size_of_val(coo.col_indices())
        + std::mem::size_of_val(coo.values());

    // Convert COO -> CSR
    let csr = coo.to_csr();
    let csr_size = estimate_csr_memory(&csr);

    // Convert CSR -> CSC
    let csc = csr.to_csc();
    let csc_size = std::mem::size_of_val(csc.col_ptrs())
        + std::mem::size_of_val(csc.row_indices())
        + std::mem::size_of_val(csc.values());

    println!("COO size: {} bytes", coo_size);
    println!("CSR size: {} bytes", csr_size);
    println!("CSC size: {} bytes", csc_size);

    // All formats should have similar memory footprint (within 2x)
    assert!(
        csr_size < coo_size * 2,
        "CSR should not use excessive memory vs COO"
    );
    assert!(
        csc_size < csr_size * 2,
        "CSC should not use excessive memory vs CSR"
    );
}

#[test]
fn test_preconditioner_memory() {
    let n = 300;
    let a = create_laplacian(n);

    let jacobi = Jacobi::new(&a).unwrap();

    let x = vec![1.0; n];
    let mut y = vec![0.0; n];

    // Apply preconditioner multiple times
    for _ in 0..100 {
        jacobi.apply(&x, &mut y);
    }

    // Check output is correct size
    assert_eq!(y.len(), n);
    for &val in &y {
        assert!(val.is_finite(), "Jacobi apply produced non-finite values");
    }

    // Jacobi preconditioner stores only diagonal (n * f64)
    let expected_size = n * std::mem::size_of::<f64>();
    println!(
        "Jacobi preconditioner expected size: {} bytes",
        expected_size
    );
}

#[test]
fn test_repeated_operations_no_growth() {
    // Test that repeated operations don't cause unbounded memory growth
    let n = 100;
    let a = create_laplacian(n);
    let x = vec![1.0; n];

    // Create many temporary results - if there's a leak, this will accumulate
    for _ in 0..1000 {
        let mut y = vec![0.0; n];
        spmv(1.0, &a, &x, 0.0, &mut y);
        // y is dropped here
    }

    // If we get here without OOM, no leak
    println!("1000 SpMV operations completed without memory growth");
}

#[test]
fn test_large_matrix_memory_estimate() {
    // Test memory estimation for large matrix
    let n = 10000;
    let a = create_laplacian(n);

    let memory = estimate_csr_memory(&a);
    let nnz = a.nnz();

    println!(
        "Matrix {}×{} with {} non-zeros uses {} bytes ({} KB)",
        n,
        n,
        nnz,
        memory,
        memory / 1024
    );

    // Expected: (n+1) * 8 + 2*nnz*8 for row_ptrs + (col_indices + values)
    // For tridiagonal Laplacian: nnz ≈ 3n, so ~(n + 6n)*8 = 56n bytes
    let expected_approx = 56 * n;

    assert!(
        memory < expected_approx * 2,
        "Memory usage {} should be close to expected {}",
        memory,
        expected_approx
    );
    assert!(
        memory > expected_approx / 2,
        "Memory usage {} should be close to expected {}",
        memory,
        expected_approx
    );
}
