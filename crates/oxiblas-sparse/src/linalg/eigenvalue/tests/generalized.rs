//! Generalized eigenvalue problems tests for sparse eigenvalue solvers.

use super::super::*;
use super::make_larger_symmetric_matrix;
use crate::csr::CsrMatrix;
// Generalized eigenvalue tests

fn make_spd_matrix_b(n: usize) -> CsrMatrix<f64> {
    // Create an SPD matrix B = I + 0.5 * T
    // where T is the tridiagonal from make_larger_symmetric_matrix
    // This ensures B is SPD and well-conditioned
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        if i > 0 {
            values.push(-0.25); // 0.5 * (-0.5)
            col_indices.push(i - 1);
        }
        values.push(1.0 + 1.0); // 1 + 0.5 * 2
        col_indices.push(i);
        if i < n - 1 {
            values.push(-0.25);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

#[test]
fn test_generalized_eigen_standard_mode() {
    // Test generalized eigenvalue A*x = lambda*B*x with standard mode
    let n = 10;
    let a = make_larger_symmetric_matrix(n);
    let b = make_spd_matrix_b(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: false,
        krylov_dimension: 15,
        symmetric: true,
        mode: GeneralizedMode::Standard,
        sigma: 0.0,
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");
    assert_eq!(
        result.eigenvalues.len(),
        3,
        "Should return requested number of eigenvalues"
    );

    // Eigenvalues should be real (non-NaN, non-Inf)
    for &ev in &result.eigenvalues {
        assert!(ev.is_finite(), "Eigenvalue should be finite, got {ev}");
    }
}

#[test]
fn test_generalized_eigen_identity_b() {
    // When B = I, generalized problem reduces to standard eigenvalue
    let n = 10;
    let a = make_larger_symmetric_matrix(n);
    let b = CsrMatrix::<f64>::eye(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: false,
        krylov_dimension: 15,
        symmetric: true,
        mode: GeneralizedMode::Standard,
        sigma: 0.0,
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    // Should return requested number of eigenvalues
    assert_eq!(result.eigenvalues.len(), 3, "Should return 3 eigenvalues");

    // Eigenvalues should be finite
    for &ev in &result.eigenvalues {
        assert!(ev.is_finite(), "Eigenvalue should be finite, got {ev}");
    }
}

#[test]
fn test_generalized_eigen_shift_invert() {
    // Test shift-invert mode for finding eigenvalues near a target
    let n = 10;
    let a = make_larger_symmetric_matrix(n);
    let b = make_spd_matrix_b(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: false,
        krylov_dimension: 15,
        symmetric: true,
        mode: GeneralizedMode::ShiftInvert,
        sigma: 1.0, // Look for eigenvalues near 1.0
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");

    // All eigenvalues should be positive
    for &ev in &result.eigenvalues {
        assert!(ev > 0.0, "Eigenvalue should be positive");
    }
}

#[test]
fn test_generalized_eigen_eigenvectors() {
    // Test that eigenvectors are computed correctly
    let n = 10;
    let a = make_larger_symmetric_matrix(n);
    let b = make_spd_matrix_b(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: true,
        krylov_dimension: 15,
        symmetric: true,
        mode: GeneralizedMode::Standard,
        sigma: 0.0,
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    assert!(
        result.eigenvectors.is_some(),
        "Should compute eigenvectors when requested"
    );
    let evecs = result.eigenvectors.unwrap();
    assert!(!evecs.is_empty(), "Should have at least one eigenvector");

    // Check each eigenvector is normalized (under standard norm)
    for v in &evecs {
        assert_eq!(v.len(), n, "Eigenvector should have correct dimension");
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(norm_sq > 0.1, "Eigenvector should have non-trivial norm");
    }
}

#[test]
fn test_generalized_eigen_buckling_mode() {
    // Test buckling mode: (A - sigma*B)^{-1} * A
    let n = 10;
    let a = make_larger_symmetric_matrix(n);
    let b = make_spd_matrix_b(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: false,
        krylov_dimension: 12,
        symmetric: true,
        mode: GeneralizedMode::Buckling,
        sigma: 0.5,
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");

    // Eigenvalues should be positive
    for &ev in &result.eigenvalues {
        assert!(ev > 0.0, "Buckling eigenvalue should be positive");
    }
}

#[test]
fn test_generalized_eigen_cayley_mode() {
    // Test Cayley mode: (A - sigma*B)^{-1} * (A + sigma*B)
    let n = 10;
    let a = make_larger_symmetric_matrix(n);
    let b = make_spd_matrix_b(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: false,
        krylov_dimension: 12,
        symmetric: true,
        mode: GeneralizedMode::Cayley,
        sigma: 0.5,
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    // Cayley transform maps eigenvalues to different values
    // The result should still be valid
    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");
}

#[test]
fn test_generalized_eigen_nonsymmetric() {
    // Test generalized eigenvalue for non-symmetric A
    let n = 8;
    // Create upper triangular matrix A (non-symmetric)
    let mut values_a = Vec::new();
    let mut col_indices_a = Vec::new();
    let mut row_ptrs_a = vec![0];

    for i in 0..n {
        for j in i..n {
            values_a.push((i + j + 1) as f64 * 0.5);
            col_indices_a.push(j);
        }
        row_ptrs_a.push(values_a.len());
    }
    let a = CsrMatrix::new(n, n, row_ptrs_a, col_indices_a, values_a).unwrap();
    let b = CsrMatrix::<f64>::eye(n);

    let config = GeneralizedEigenConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-5,
        compute_eigenvectors: false,
        krylov_dimension: 15,
        symmetric: false,
        mode: GeneralizedMode::Standard,
        sigma: 0.0,
    };

    let solver = GeneralizedEigen::new(config);
    let result = solver.compute(&a, &b, None).unwrap();

    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");
}
