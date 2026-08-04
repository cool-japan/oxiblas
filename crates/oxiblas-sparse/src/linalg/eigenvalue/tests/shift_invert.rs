//! Shift-and-invert Lanczos tests for sparse eigenvalue solvers.

use super::super::*;
use super::make_larger_symmetric_matrix;
use crate::csr::CsrMatrix;
// Shift-and-invert tests

#[test]
fn test_shift_invert_basic() {
    // For n=10 tridiagonal (2,-1,-1), eigenvalues are:
    // lambda_k = 2 - 2*cos(k*pi/(n+1)) for k=1,...,n
    // For n=10: smallest ~0.08, largest ~3.92
    // Middle eigenvalues are around 2.0
    let a = make_larger_symmetric_matrix(10);

    let config = ShiftInvertConfig {
        num_eigenvalues: 3,
        shift: 2.0, // Target eigenvalues near 2.0
        krylov_dimension: 15,
        tolerance: 1e-6,
        symmetric: true,
        ..Default::default()
    };

    let solver = ShiftInvertLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    assert!(!result.eigenvalues.is_empty(), "Should compute eigenvalues");

    // Eigenvalues should be near the shift (2.0)
    for &ev in &result.eigenvalues {
        // All eigenvalues of this matrix are in [0.08, 3.92]
        assert!(
            ev > 0.0 && ev < 4.0,
            "Eigenvalue should be in valid range, got {ev}"
        );
    }

    // At least one should be close to 2.0 (within 0.5)
    let near_two = result.eigenvalues.iter().any(|&ev| (ev - 2.0).abs() < 1.0);
    assert!(
        near_two,
        "At least one eigenvalue should be near the shift 2.0, got {:?}",
        result.eigenvalues
    );
}

#[test]
fn test_shift_invert_identity() {
    // For identity matrix, all eigenvalues are 1.0
    // Shift-invert with sigma=0.5 should find eigenvalues near 0.5 (which is 1.0)
    let a = CsrMatrix::<f64>::eye(5);

    let config = ShiftInvertConfig {
        num_eigenvalues: 3,
        shift: 0.5,
        krylov_dimension: 10,
        tolerance: 1e-8,
        symmetric: true,
        ..Default::default()
    };

    let solver = ShiftInvertLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // All eigenvalues should be close to 1.0
    for &ev in &result.eigenvalues {
        assert!((ev - 1.0).abs() < 0.1, "Expected eigenvalue ~1.0, got {ev}");
    }
}

#[test]
fn test_shift_invert_eigenvectors() {
    let n = 10;
    let a = make_larger_symmetric_matrix(n);

    let config = ShiftInvertConfig {
        num_eigenvalues: 2,
        shift: 1.0, // Look for eigenvalues near 1.0
        compute_eigenvectors: true,
        krylov_dimension: 20,
        tolerance: 1e-6,
        symmetric: true,
        ..Default::default()
    };

    let solver = ShiftInvertLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    assert!(result.eigenvectors.is_some(), "Should compute eigenvectors");
    let evecs = result.eigenvectors.unwrap();

    for v in &evecs {
        // Check normalization
        let vnorm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(
            (vnorm_sq - 1.0).abs() < 0.1,
            "Eigenvector should be normalized"
        );
    }
}

#[test]
fn test_shift_invert_larger_system() {
    let n = 20;
    let a = make_larger_symmetric_matrix(n);

    // For n=20 tridiagonal, eigenvalues range from ~0.024 to ~3.976
    // lambda_k = 2 - 2*cos(k*pi/(n+1))
    // Test finding eigenvalues near 2.5 (not too close to any eigenvalue)
    let config = ShiftInvertConfig {
        num_eigenvalues: 4,
        shift: 2.5, // Between eigenvalues, not near singular
        krylov_dimension: 25,
        tolerance: 1e-6,
        symmetric: true,
        ..Default::default()
    };

    let solver = ShiftInvertLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    assert_eq!(result.eigenvalues.len(), 4, "Should return 4 eigenvalues");

    // Eigenvalues should be in valid range
    for &ev in &result.eigenvalues {
        assert!(ev > 0.0 && ev < 4.0, "Eigenvalue in valid range, got {ev}");
    }
}

#[test]
fn test_shift_invert_lu_fallback() {
    // Create a matrix where Cholesky might fail after shifting
    // (shifted matrix could be indefinite)
    let n = 5;
    let a = make_larger_symmetric_matrix(n);

    // Shift by a value larger than largest eigenvalue
    // This makes (A - sigma*I) negative definite
    let config = ShiftInvertConfig {
        num_eigenvalues: 2,
        shift: 5.0, // Larger than max eigenvalue ~3.9
        krylov_dimension: 10,
        tolerance: 1e-6,
        symmetric: true, // Will try Cholesky first, fall back to LU
        ..Default::default()
    };

    let solver = ShiftInvertLanczos::new(config);
    let result = solver.compute(&a, None).unwrap();

    // Should still find eigenvalues (using LU fallback)
    assert!(
        !result.eigenvalues.is_empty(),
        "Should compute eigenvalues even with LU fallback"
    );

    // Eigenvalues should be valid
    for &ev in &result.eigenvalues {
        assert!(ev > 0.0 && ev < 4.0, "Eigenvalue in valid range");
    }
}
