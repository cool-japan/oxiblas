//! Interval eigenvalue search tests for sparse eigenvalue solvers.

use super::super::*;
use super::make_larger_symmetric_matrix;
use crate::csr::CsrMatrix;
// ============================================
// Interval Eigenvalue Tests
// ============================================

#[test]
fn test_interval_eigen_basic() {
    // Simple 5x5 diagonal matrix with known eigenvalues 1, 2, 3, 4, 5
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let col_indices: Vec<usize> = (0..5).collect();
    let row_ptrs = vec![0, 1, 2, 3, 4, 5];

    let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();

    // Find eigenvalues in [1.5, 3.5] - should find 2 and 3
    let config = IntervalEigenConfig {
        low: 1.5,
        high: 3.5,
        max_iterations: 100,
        tolerance: 1e-8,
        compute_eigenvectors: false,
        krylov_dimension: 5,
        full_reorthogonalization: true,
    };

    let solver = IntervalEigen::new(config);
    let result = solver.compute(&a, None).unwrap();

    assert_eq!(result.count, 2, "Should find 2 eigenvalues in [1.5, 3.5]");
    assert_eq!(result.eigenvalues.len(), 2);

    // Eigenvalues should be close to 2 and 3
    let mut eigenvalues_sorted = result.eigenvalues.clone();
    eigenvalues_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (eigenvalues_sorted[0] - 2.0).abs() < 0.1,
        "First eigenvalue should be ~2, got {}",
        eigenvalues_sorted[0]
    );
    assert!(
        (eigenvalues_sorted[1] - 3.0).abs() < 0.1,
        "Second eigenvalue should be ~3, got {}",
        eigenvalues_sorted[1]
    );
}

#[test]
fn test_interval_eigen_tridiagonal() {
    // Tridiagonal matrix with eigenvalues that can be analytically computed
    // Using make_larger_symmetric_matrix which creates tridiagonal with diag=2, off-diag=-1
    let n = 10;
    let a = make_larger_symmetric_matrix(n);

    // For 2 - 2*cos(k*pi/(n+1)), k=1..n
    // With n=10: eigenvalues are approximately 0.081, 0.318, 0.690, 1.169, 1.708, 2.291, 2.831, 3.309, 3.681, 3.918

    // Find eigenvalues in [0.5, 2.0]
    let result = eigenvalues_in_interval(&a, 0.5, 2.0).unwrap();

    assert!(
        result.count >= 2 && result.count <= 4,
        "Expected 2-4 eigenvalues in [0.5, 2.0], got {}",
        result.count
    );
    assert!(result.converged, "Should converge");

    // All returned eigenvalues should be in the interval
    for &ev in &result.eigenvalues {
        assert!(
            (0.5 - 0.1..=2.0 + 0.1).contains(&ev),
            "Eigenvalue {} should be in [0.5, 2.0]",
            ev
        );
    }
}

#[test]
fn test_interval_eigen_no_eigenvalues() {
    // 5x5 diagonal matrix with eigenvalues 1, 2, 3, 4, 5
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let col_indices: Vec<usize> = (0..5).collect();
    let row_ptrs = vec![0, 1, 2, 3, 4, 5];

    let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();

    // Find eigenvalues in [10.0, 20.0] - should find none
    let result = eigenvalues_in_interval(&a, 10.0, 20.0).unwrap();

    assert_eq!(result.count, 0, "Should find 0 eigenvalues in [10.0, 20.0]");
    assert_eq!(result.eigenvalues.len(), 0);
}

#[test]
fn test_interval_eigen_all_eigenvalues() {
    // 5x5 diagonal matrix with eigenvalues 1, 2, 3, 4, 5
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let col_indices: Vec<usize> = (0..5).collect();
    let row_ptrs = vec![0, 1, 2, 3, 4, 5];

    let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();

    // Find eigenvalues in [0.0, 10.0] - should find all 5
    let result = eigenvalues_in_interval(&a, 0.0, 10.0).unwrap();

    assert_eq!(result.count, 5, "Should find all 5 eigenvalues");
    assert_eq!(result.eigenvalues.len(), 5);
}

#[test]
fn test_interval_eigen_with_eigenvectors() {
    // Diagonal matrix for easy verification
    let values = vec![1.0_f64, 3.0, 5.0, 7.0];
    let col_indices: Vec<usize> = (0..4).collect();
    let row_ptrs = vec![0, 1, 2, 3, 4];

    let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();

    let config = IntervalEigenConfig {
        low: 2.0,
        high: 6.0,
        max_iterations: 100,
        tolerance: 1e-8,
        compute_eigenvectors: true,
        krylov_dimension: 4,
        full_reorthogonalization: true,
    };

    let solver = IntervalEigen::new(config);
    let result = solver.compute(&a, None).unwrap();

    assert_eq!(result.count, 2, "Should find eigenvalues 3 and 5");

    // Check eigenvectors were computed
    assert!(result.eigenvectors.is_some(), "Should compute eigenvectors");
    let evecs = result.eigenvectors.unwrap();
    assert_eq!(evecs.len(), 2, "Should have 2 eigenvectors");

    // Each eigenvector should be normalized
    for evec in &evecs {
        let norm: f64 = evec.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.1,
            "Eigenvector should be approximately normalized"
        );
    }
}

#[test]
fn test_count_eigenvalues_in_interval() {
    // 5x5 diagonal matrix with eigenvalues 1, 2, 3, 4, 5
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let col_indices: Vec<usize> = (0..5).collect();
    let row_ptrs = vec![0, 1, 2, 3, 4, 5];

    let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();

    // Count eigenvalues in various intervals
    let count1 = count_eigenvalues_in_interval(&a, 0.0, 10.0, 5).unwrap();
    assert_eq!(count1, 5, "All eigenvalues in [0, 10]");

    let count2 = count_eigenvalues_in_interval(&a, 1.5, 3.5, 5).unwrap();
    assert_eq!(count2, 2, "Eigenvalues 2, 3 in [1.5, 3.5]");

    let count3 = count_eigenvalues_in_interval(&a, 2.5, 4.5, 5).unwrap();
    assert_eq!(count3, 2, "Eigenvalues 3, 4 in [2.5, 4.5]");

    let count4 = count_eigenvalues_in_interval(&a, 10.0, 20.0, 5).unwrap();
    assert_eq!(count4, 0, "No eigenvalues in [10, 20]");
}

#[test]
fn test_interval_eigen_symmetric_matrix() {
    // Create a larger symmetric matrix
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    // For 2 - 2*cos(k*pi/(n+1)) with n=15:
    // Eigenvalues range from ~0.04 to ~3.96
    // Find eigenvalues in middle range
    let result = eigenvalues_in_interval(&a, 1.5, 2.5).unwrap();

    assert!(
        result.count > 0,
        "Should find some eigenvalues in [1.5, 2.5]"
    );
    assert!(result.converged, "Should converge");

    // All returned eigenvalues should be in the interval
    for &ev in &result.eigenvalues {
        assert!(
            (1.5 - 0.15..=2.5 + 0.15).contains(&ev),
            "Eigenvalue {} should be approximately in [1.5, 2.5]",
            ev
        );
    }
}

#[test]
fn test_interval_eigen_residual_norms() {
    let n = 10;
    let a = make_larger_symmetric_matrix(n);

    let config = IntervalEigenConfig {
        low: 1.0,
        high: 3.0,
        max_iterations: 200,
        tolerance: 1e-6,
        compute_eigenvectors: true,
        krylov_dimension: n,
        full_reorthogonalization: true,
    };

    let solver = IntervalEigen::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should have residual norms for each eigenvalue
    assert_eq!(
        result.residual_norms.len(),
        result.eigenvalues.len(),
        "Should have residual norm for each eigenvalue"
    );

    // Residual norms should be non-negative and reasonably small
    for &res in &result.residual_norms {
        assert!(res >= 0.0, "Residual norm should be non-negative");
        assert!(res < 1.0, "Residual norm should be reasonably bounded");
    }
}

#[test]
fn test_interval_eigen_edge_case_single() {
    // Matrix with single eigenvalue at 5.0
    let values = vec![5.0_f64];
    let col_indices = vec![0_usize];
    let row_ptrs = vec![0_usize, 1];

    let a = CsrMatrix::new(1, 1, row_ptrs, col_indices, values).unwrap();

    let result = eigenvalues_in_interval(&a, 4.0, 6.0).unwrap();
    assert_eq!(result.count, 1, "Should find the single eigenvalue");
    assert!(
        (result.eigenvalues[0] - 5.0).abs() < 0.1,
        "Eigenvalue should be ~5.0"
    );
}

#[test]
fn test_interval_eigen_non_tridiagonal_general_path() {
    // Non-tridiagonal symmetric matrix (nonzero at (0,2) and (2,0)):
    //   A = [2 0 1]
    //       [0 5 0]
    //       [1 0 2]
    // The 2x2 block on coords {0,2}, [[2,1],[1,2]], has eigenvalues 1 and 3;
    // combined with the middle entry 5 the spectrum is {1, 3, 5}. This
    // exercises the general Lanczos + verified-Ritz path (NOT the exact
    // tridiagonal fast path).
    let values = vec![2.0_f64, 1.0, 5.0, 1.0, 2.0];
    let col_indices = vec![0_usize, 2, 1, 0, 2];
    let row_ptrs = vec![0_usize, 2, 3, 5];
    let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

    // Full Krylov subspace (krylov_dimension == n) => tridiagonal is
    // orthogonally similar to A, so the verified count is exact.
    let config = IntervalEigenConfig {
        low: 2.0,
        high: 6.0,
        max_iterations: 100,
        tolerance: 1e-8,
        compute_eigenvectors: true,
        krylov_dimension: 3,
        full_reorthogonalization: true,
    };
    let result = IntervalEigen::new(config).compute(&a, None).unwrap();

    // Eigenvalues 3 and 5 lie in [2, 6].
    assert_eq!(
        result.count, 2,
        "Should find eigenvalues 3 and 5 in [2, 6], got {}",
        result.count
    );
    assert!(result.converged, "Full Krylov subspace should converge");

    let mut evs = result.eigenvalues.clone();
    evs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!(
        (evs[0] - 3.0).abs() < 1e-4,
        "First eigenvalue ~3, got {}",
        evs[0]
    );
    assert!(
        (evs[1] - 5.0).abs() < 1e-4,
        "Second eigenvalue ~5, got {}",
        evs[1]
    );

    // The verified residual bound must actually hold for each reported pair.
    for &res in &result.residual_norms {
        assert!(
            res < 1e-8,
            "Reported eigenpair must satisfy the residual bound, got {res}"
        );
    }

    // A disjoint sub-interval should find only eigenvalue 1.
    let low_result = eigenvalues_in_interval(&a, 0.0, 2.0).unwrap();
    assert_eq!(low_result.count, 1, "Only eigenvalue 1 lies in [0, 2]");
    assert!(
        (low_result.eigenvalues[0] - 1.0).abs() < 1e-4,
        "Eigenvalue should be ~1.0, got {}",
        low_result.eigenvalues[0]
    );
}
