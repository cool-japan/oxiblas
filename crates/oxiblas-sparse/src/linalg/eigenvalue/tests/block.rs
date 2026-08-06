//! Block Lanczos and Block Arnoldi tests for sparse eigenvalue solvers.

use super::super::*;
use super::make_larger_symmetric_matrix;
use crate::csr::CsrMatrix;
// Block Lanczos tests

#[test]
fn test_block_lanczos_basic() {
    // Test Block Lanczos on a symmetric tridiagonal matrix
    let n = 20;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockLanczosConfig {
        num_eigenvalues: 4,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 10,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: false,
        full_reorthogonalization: true,
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should return at least some eigenvalues
    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");

    // All eigenvalues should be positive for this SPD matrix
    for &ev in &result.eigenvalues {
        assert!(
            ev > 0.0,
            "Eigenvalue should be positive for SPD matrix, got {ev}"
        );
    }

    // Eigenvalues should be in valid range for n=20 tridiagonal [~0.02, ~3.98]
    for &ev in &result.eigenvalues {
        assert!(
            ev < 5.0,
            "Eigenvalue should be less than 5 for this matrix, got {ev}"
        );
    }
}

#[test]
fn test_block_lanczos_with_eigenvectors() {
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockLanczosConfig {
        num_eigenvalues: 3,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 8,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: true,
        full_reorthogonalization: true,
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should have eigenvectors
    assert!(result.eigenvectors.is_some(), "Should compute eigenvectors");

    let evecs = result.eigenvectors.unwrap();
    assert!(!evecs.is_empty(), "Should have at least one eigenvector");

    // Check each eigenvector
    for (i, v) in evecs.iter().enumerate() {
        // Should have correct dimension
        assert_eq!(v.len(), n, "Eigenvector should have dimension {n}");

        // Should be normalized (approximately)
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 0.3,
            "Eigenvector {i} should be normalized, got norm^2 = {norm_sq}"
        );

        // Should not be all zeros
        let max_abs: f64 = v.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max_abs > 0.01, "Eigenvector {i} should not be zero");
    }
}

#[test]
fn test_block_lanczos_diagonal_matrix() {
    // Test on diagonal matrix with distinct eigenvalues
    let n = 12;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        values.push((i + 1) as f64);
        col_indices.push(i);
        row_ptrs.push(values.len());
    }
    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    let config = BlockLanczosConfig {
        num_eigenvalues: 4,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 6,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: false,
        full_reorthogonalization: true,
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // LargestMagnitude should return eigenvalues near 12, 11, 10, 9
    let mut sorted_eigs = result.eigenvalues.clone();
    sorted_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Check that we got large eigenvalues
    for &ev in &sorted_eigs {
        assert!(
            ev >= 8.0,
            "LargestMagnitude should return large eigenvalues, got {ev}"
        );
    }
}

#[test]
fn test_block_lanczos_smallest_magnitude() {
    let n = 20;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockLanczosConfig {
        num_eigenvalues: 3,
        block_size: 2,
        which: WhichEigenvalues::SmallestMagnitude,
        num_blocks: 10,
        max_iterations: 150,
        tolerance: 1e-4,
        compute_eigenvectors: false,
        full_reorthogonalization: true,
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should return eigenvalues
    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");

    // For SmallestMagnitude on SPD matrix, should get smallest positive eigenvalues
    // For n=20 tridiagonal, smallest is ~0.024
    let max_returned = result
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // The smallest eigenvalues should be relatively small
    assert!(
        max_returned < 3.0,
        "SmallestMagnitude eigenvalues should be small, got max {max_returned}"
    );
}

#[test]
fn test_block_lanczos_larger_block_size() {
    // Test with larger block size
    let n = 24;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockLanczosConfig {
        num_eigenvalues: 6,
        block_size: 3, // Larger block size
        which: WhichEigenvalues::LargestAlgebraic,
        num_blocks: 8,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: true,
        full_reorthogonalization: true,
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should return 6 eigenvalues
    assert!(
        !result.eigenvalues.is_empty(),
        "Should return at least 1 eigenvalue"
    );

    // LargestAlgebraic for n=24 tridiagonal should give values near 3.95
    let min_returned = result
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_returned > 1.0,
        "LargestAlgebraic eigenvalues should be > 1.0, got {min_returned}"
    );
}

#[test]
fn test_block_lanczos_dimension_mismatch() {
    let n = 10;
    let a = make_larger_symmetric_matrix(n);

    // Create initial block with wrong dimension
    let wrong_block = vec![vec![1.0; 5], vec![1.0; 5]]; // 5 instead of 10

    let config = BlockLanczosConfig {
        num_eigenvalues: 3,
        block_size: 2,
        ..Default::default()
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, Some(&wrong_block));

    assert!(result.is_err(), "Should error on dimension mismatch");
}

#[test]
fn test_block_lanczos_residual_norms() {
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockLanczosConfig {
        num_eigenvalues: 3,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 10,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: false,
        full_reorthogonalization: true,
    };

    let solver = BlockLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should have residual norms for each eigenvalue
    assert_eq!(
        result.residual_norms.len(),
        result.eigenvalues.len(),
        "Should have residual norm for each eigenvalue"
    );

    // Residual norms should be non-negative
    for &res in &result.residual_norms {
        assert!(res >= 0.0, "Residual norm should be non-negative");
    }
}

// Block Arnoldi tests

#[test]
fn test_block_arnoldi_basic() {
    // Test Block Arnoldi on a general matrix
    let n = 20;
    let a = make_larger_symmetric_matrix(n); // Can use symmetric for testing

    let config = BlockArnoldiConfig {
        num_eigenvalues: 4,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 10,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: false,
    };

    let solver = BlockArnoldi::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should return eigenvalues
    assert!(
        !result.eigenvalues_real.is_empty(),
        "Should compute eigenvalues"
    );

    // For symmetric matrix, imaginary parts should be ~0
    for &im in &result.eigenvalues_imag {
        assert!(
            im.abs() < 0.5,
            "Imaginary part should be small for symmetric matrix, got {im}"
        );
    }
}

#[test]
fn test_block_arnoldi_with_eigenvectors() {
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockArnoldiConfig {
        num_eigenvalues: 3,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 8,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: true,
    };

    let solver = BlockArnoldi::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should have eigenvectors
    assert!(result.eigenvectors.is_some(), "Should compute eigenvectors");

    let evecs = result.eigenvectors.unwrap();
    assert!(!evecs.is_empty(), "Should have at least one eigenvector");

    for (i, v) in evecs.iter().enumerate() {
        assert_eq!(v.len(), n, "Eigenvector should have dimension {n}");

        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 0.3,
            "Eigenvector {i} should be normalized, got norm^2 = {norm_sq}"
        );
    }
}

#[test]
fn test_block_arnoldi_non_symmetric() {
    // Test on a larger non-symmetric tridiagonal matrix
    let n = 15;
    let mut values: Vec<f64> = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        if i > 0 {
            values.push(-0.5); // subdiagonal
            col_indices.push(i - 1);
        }
        values.push(2.0); // diagonal
        col_indices.push(i);
        if i < n - 1 {
            values.push(-0.8); // superdiagonal (different from subdiagonal = non-symmetric)
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }
    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    let config = BlockArnoldiConfig {
        num_eigenvalues: 3,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 8,
        max_iterations: 100,
        tolerance: 1e-4,
        compute_eigenvectors: false,
    };

    let solver = BlockArnoldi::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should return eigenvalues
    assert!(
        !result.eigenvalues_real.is_empty(),
        "Should compute eigenvalues for non-symmetric matrix"
    );

    // All computed eigenvalues should be finite
    for (re, im) in result
        .eigenvalues_real
        .iter()
        .zip(result.eigenvalues_imag.iter())
    {
        assert!(
            !re.is_nan() && !re.is_infinite(),
            "Real part should be finite"
        );
        assert!(
            !im.is_nan() && !im.is_infinite(),
            "Imaginary part should be finite"
        );
    }
}

#[test]
fn test_block_arnoldi_larger_matrix() {
    // Test on larger matrix to verify scaling
    let n = 25;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockArnoldiConfig {
        num_eigenvalues: 4,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 12,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: false,
    };

    let solver = BlockArnoldi::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should return eigenvalues
    assert!(
        !result.eigenvalues_real.is_empty(),
        "Should compute eigenvalues"
    );

    // Eigenvalues should have matching real and imaginary parts
    assert_eq!(
        result.eigenvalues_real.len(),
        result.eigenvalues_imag.len(),
        "Real and imaginary eigenvalue vectors should have same length"
    );

    // All eigenvalues should be finite
    for (re, im) in result
        .eigenvalues_real
        .iter()
        .zip(result.eigenvalues_imag.iter())
    {
        assert!(
            !re.is_nan() && !re.is_infinite(),
            "Real part should be finite, got {re}"
        );
        assert!(
            !im.is_nan() && !im.is_infinite(),
            "Imaginary part should be finite, got {im}"
        );
    }
}

#[test]
fn test_block_arnoldi_dimension_mismatch() {
    let n = 10;
    let a = make_larger_symmetric_matrix(n);

    let wrong_block = vec![vec![1.0; 5], vec![1.0; 5]]; // 5 instead of 10

    let config = BlockArnoldiConfig {
        num_eigenvalues: 3,
        block_size: 2,
        ..Default::default()
    };

    let solver = BlockArnoldi::new(config);
    let result = solver.compute(&a, Some(&wrong_block));

    assert!(result.is_err(), "Should error on dimension mismatch");
}

#[test]
fn test_block_arnoldi_residual_norms() {
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    let config = BlockArnoldiConfig {
        num_eigenvalues: 3,
        block_size: 2,
        which: WhichEigenvalues::LargestMagnitude,
        num_blocks: 10,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: false,
    };

    let solver = BlockArnoldi::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should have residual norms for each eigenvalue
    assert_eq!(
        result.residual_norms.len(),
        result.eigenvalues_real.len(),
        "Should have residual norm for each eigenvalue"
    );

    // Residual norms should be non-negative
    for &res in &result.residual_norms {
        assert!(res >= 0.0, "Residual norm should be non-negative");
    }
}
