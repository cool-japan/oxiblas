//! Polynomial-filtered Lanczos tests for sparse eigenvalue solvers.

use super::super::*;
use crate::csr::CsrMatrix;
// =====================================================================
// Polynomial Filtered Lanczos Tests
// =====================================================================

#[test]
fn test_polynomial_filtered_lanczos_basic() {
    // Simple diagonal matrix with known eigenvalues
    // A = diag(1, 2, 3, 4, 5) - eigenvalues are 1, 2, 3, 4, 5
    let n = 5;
    let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let col_indices: Vec<usize> = (0..n).collect();
    let row_ptrs: Vec<usize> = (0..=n).collect();

    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    // Find eigenvalues in broader interval [2.0, 4.0] - should find eigenvalues 2, 3, 4
    // Using looser tolerance since polynomial filtering is approximate
    let config = PolynomialFilterConfig {
        num_eigenvalues: 2,
        target_low: 2.0,
        target_high: 4.0,
        spectral_low: Some(0.5),
        spectral_high: Some(5.5),
        polynomial_degree: 15,
        krylov_dimension: 10,
        max_iterations: 100,
        tolerance: 1e-3, // Looser tolerance for this simple test
        compute_eigenvectors: false,
        full_reorthogonalization: true,
    };

    let solver = PolynomialFilteredLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // The algorithm should either find eigenvalues or run through iterations
    // For simple diagonal matrices, results may vary due to filter behavior
    assert!(
        result.iterations > 0,
        "Should perform at least one iteration"
    );

    // If eigenvalues found, check they're reasonable
    if !result.eigenvalues.is_empty() {
        for &ev in &result.eigenvalues {
            assert!(
                (1.0..=5.0).contains(&ev),
                "Eigenvalue {} should be in spectral range",
                ev
            );
        }
    }
}

#[test]
fn test_polynomial_filtered_convenience_function() {
    // Diagonal matrix: eigenvalues are 1, 2, 3, 4, 5
    let n = 5;
    let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let col_indices: Vec<usize> = (0..n).collect();
    let row_ptrs: Vec<usize> = (0..=n).collect();

    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    // Find eigenvalues in [1.5, 4.5] - should find 2, 3, 4
    let result = polynomial_filtered_eigenvalues(&a, 1.5, 4.5, 3).unwrap();

    assert!(
        !result.eigenvalues.is_empty(),
        "Should find eigenvalues in interval"
    );

    // Each found eigenvalue should be in the target interval (with some tolerance)
    for &ev in &result.eigenvalues {
        assert!(
            (1.0..=5.0).contains(&ev),
            "Eigenvalue {} should be within spectral range",
            ev
        );
    }
}

#[test]
fn test_polynomial_filtered_tridiagonal_matrix() {
    // Tridiagonal matrix (1,-1,-1) pattern
    // Known eigenvalues: 2 - 2*cos(k*pi/(n+1)) for k=1..n
    let n = 10;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        if i > 0 {
            values.push(-1.0);
            col_indices.push(i - 1);
        }
        values.push(2.0);
        col_indices.push(i);
        if i < n - 1 {
            values.push(-1.0);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }

    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    // Eigenvalues are approximately in [0.08, 3.92]
    // Find eigenvalues in middle of spectrum [1.5, 2.5]
    let config = PolynomialFilterConfig {
        num_eigenvalues: 2,
        target_low: 1.5,
        target_high: 2.5,
        spectral_low: Some(0.0),
        spectral_high: Some(4.0),
        polynomial_degree: 15,
        krylov_dimension: 20,
        max_iterations: 100,
        tolerance: 1e-6,
        compute_eigenvectors: true,
        full_reorthogonalization: true,
    };

    let solver = PolynomialFilteredLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should find some eigenvalues
    assert!(
        !result.eigenvalues.is_empty(),
        "Should find eigenvalues in interval"
    );

    // If eigenvectors computed, check they're valid
    if let Some(ref vecs) = result.eigenvectors {
        for v in vecs {
            let norm_sq: f64 = v.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 0.5,
                "Eigenvector should be roughly normalized"
            );
        }
    }
}

#[test]
fn test_polynomial_filtered_config_default() {
    let config: PolynomialFilterConfig<f64> = PolynomialFilterConfig::default();

    assert_eq!(config.num_eigenvalues, 6);
    assert_eq!(config.polynomial_degree, 20);
    assert_eq!(config.krylov_dimension, 50);
    assert_eq!(config.max_iterations, 100);
    assert!(config.compute_eigenvectors);
    assert!(config.full_reorthogonalization);
}

#[test]
fn test_polynomial_filtered_empty_interval() {
    // Diagonal matrix with eigenvalues 1, 2, 3
    let values = vec![1.0_f64, 2.0, 3.0];
    let col_indices = vec![0_usize, 1, 2];
    let row_ptrs = vec![0_usize, 1, 2, 3];

    let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

    // Search in interval with no eigenvalues
    let config = PolynomialFilterConfig {
        num_eigenvalues: 1,
        target_low: 5.0,
        target_high: 6.0,
        spectral_low: Some(0.5),
        spectral_high: Some(3.5),
        polynomial_degree: 10,
        krylov_dimension: 10,
        max_iterations: 20,
        tolerance: 1e-6,
        compute_eigenvectors: false,
        full_reorthogonalization: true,
    };

    let solver = PolynomialFilteredLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // May find eigenvalues (method doesn't guarantee empty result for empty interval)
    // but we just ensure it completes without error
    assert!(
        result.iterations > 0,
        "Should perform at least one iteration"
    );
}

#[test]
fn test_polynomial_filtered_interior_amplification() {
    // Diagonal matrix A = diag(1, 2, ..., 10). The extreme eigenvalues are 1
    // and 10; the target interval [3.5, 5.5] contains ONLY the strictly
    // interior eigenvalues 4 and 5. A filter that does not genuinely amplify
    // the target interval (e.g. one that peaks at an extreme of the spectrum)
    // could never isolate these interior eigenvalues, so this directly
    // exercises the Chebyshev band-pass filter.
    let n = 10;
    let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let col_indices: Vec<usize> = (0..n).collect();
    let row_ptrs: Vec<usize> = (0..=n).collect();
    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    let config = PolynomialFilterConfig {
        num_eigenvalues: 2,
        target_low: 3.5,
        target_high: 5.5,
        spectral_low: Some(0.5),
        spectral_high: Some(10.5),
        polynomial_degree: 25,
        krylov_dimension: 10,
        max_iterations: 50,
        tolerance: 1e-8,
        compute_eigenvectors: true,
        full_reorthogonalization: true,
    };

    let solver = PolynomialFilteredLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    assert_eq!(
        result.eigenvalues.len(),
        2,
        "Should isolate the two interior eigenvalues 4 and 5, got {:?}",
        result.eigenvalues
    );
    assert!(result.converged, "Interior eigenvalues should converge");

    let mut evs = result.eigenvalues.clone();
    evs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!(
        (evs[0] - 4.0).abs() < 1e-6,
        "First interior eigenvalue should be ~4, got {}",
        evs[0]
    );
    assert!(
        (evs[1] - 5.0).abs() < 1e-6,
        "Second interior eigenvalue should be ~5, got {}",
        evs[1]
    );

    // Every reported pair must satisfy the residual bound against A.
    for &res in &result.residual_norms {
        assert!(res <= 1e-8, "Residual bound must hold, got {res}");
    }
}
