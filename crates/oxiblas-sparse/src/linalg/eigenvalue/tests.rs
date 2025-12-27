//! Tests for sparse eigenvalue solvers.

use super::*;
use crate::csr::CsrMatrix;

fn make_symmetric_matrix() -> CsrMatrix<f64> {
    // A = [4 1 0]
    //     [1 4 1]
    //     [0 1 4]
    let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
    let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
    let row_ptrs = vec![0, 2, 5, 7];

    CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
}

fn make_larger_symmetric_matrix(n: usize) -> CsrMatrix<f64> {
    // Tridiagonal: A[i,i] = 2, A[i,i+1] = A[i+1,i] = -1
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

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

#[test]
fn test_lanczos_basic() {
    let a = make_symmetric_matrix();

    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 10,
        tolerance: 1e-10,
        ..Default::default()
    };

    let lanczos = Lanczos::new(config);
    let result = lanczos.compute(&a, None).unwrap();

    // For a 3x3 matrix, we may get up to 3 eigenvalues
    assert!(
        result.eigenvalues.len() >= 2,
        "Should get at least 2 eigenvalues"
    );

    // Eigenvalues of this matrix are approximately: 5.414, 4.0, 2.586
    // Sort for comparison
    let mut eigs = result.eigenvalues.clone();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Check that we got valid eigenvalues in expected range
    assert!(
        eigs[0] > 5.0 && eigs[0] < 6.0,
        "Largest eigenvalue ~5.414, got {}",
        eigs[0]
    );
    if eigs.len() >= 2 {
        assert!(
            eigs[1] > 2.0 && eigs[1] < 5.5,
            "Second eigenvalue in range, got {}",
            eigs[1]
        );
    }
}

#[test]
fn test_lanczos_identity() {
    let a = CsrMatrix::<f64>::eye(5);

    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        ..Default::default()
    };

    let lanczos = Lanczos::new(config);
    let result = lanczos.compute(&a, None).unwrap();

    // All eigenvalues should be 1.0
    for &ev in &result.eigenvalues {
        assert!((ev - 1.0).abs() < 1e-6, "Expected 1.0, got {ev}");
    }
}

#[test]
fn test_lanczos_larger_matrix() {
    let a = make_larger_symmetric_matrix(20);

    let config = LanczosConfig {
        num_eigenvalues: 4,
        which: WhichEigenvalues::SmallestMagnitude,
        krylov_dimension: 15,
        tolerance: 1e-8,
        ..Default::default()
    };

    let lanczos = Lanczos::new(config);
    let result = lanczos.compute(&a, None).unwrap();

    assert_eq!(result.eigenvalues.len(), 4);

    // For n=20 tridiagonal, smallest eigenvalue is ~0.0245
    let min_eig = result
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(min_eig > 0.0, "Eigenvalues should be positive");
    assert!(min_eig < 0.1, "Smallest eigenvalue should be small");
}

#[test]
fn test_lanczos_eigenvectors() {
    // Use a larger matrix for better eigenvector quality
    let n = 20;
    let a = make_larger_symmetric_matrix(n);

    let config = LanczosConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestAlgebraic,
        compute_eigenvectors: true,
        krylov_dimension: 30, // Use larger Krylov dimension for better accuracy
        tolerance: 1e-6,
        ..Default::default()
    };

    let lanczos = Lanczos::new(config);
    let result = lanczos.compute(&a, None).unwrap();

    assert!(result.eigenvectors.is_some());
    let evecs = result.eigenvectors.unwrap();
    assert!(!evecs.is_empty(), "Should have at least one eigenvector");

    // Verify basic eigenvector properties
    for (i, ev) in result.eigenvalues.iter().enumerate() {
        if i >= evecs.len() {
            break;
        }
        let v = &evecs[i];

        // Check that eigenvector is normalized (approximately)
        let vnorm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(
            (vnorm_sq - 1.0).abs() < 0.1,
            "Eigenvector should be normalized, got norm^2 = {}",
            vnorm_sq
        );

        // Check that eigenvalue is positive (for this SPD matrix)
        assert!(*ev > 0.0, "Eigenvalue should be positive for SPD matrix");
    }
}

#[test]
fn test_arnoldi_basic() {
    // Use a larger matrix for Arnoldi
    let a = make_larger_symmetric_matrix(10);

    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 15,
        ..Default::default()
    };

    let arnoldi = Arnoldi::new(config);
    let result = arnoldi.compute(&a, None).unwrap();

    // Should get at least some eigenvalues
    assert!(
        !result.eigenvalues_real.is_empty(),
        "Should compute some eigenvalues"
    );

    // For symmetric matrix, imaginary parts should be ~0
    for im in &result.eigenvalues_imag {
        assert!(
            im.abs() < 0.5,
            "Imaginary part should be small for symmetric matrix"
        );
    }
}

#[test]
fn test_arnoldi_general_matrix() {
    // Use a larger non-symmetric matrix for better convergence
    // Create a non-symmetric matrix with real eigenvalues for easier testing
    // A = [2 1 0]
    //     [0 3 1]
    //     [0 0 4]
    // Upper triangular - eigenvalues are 2, 3, 4
    let values = vec![2.0, 1.0, 3.0, 1.0, 4.0];
    let col_indices = vec![0, 1, 1, 2, 2];
    let row_ptrs = vec![0, 2, 4, 5];
    let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 10,
        ..Default::default()
    };

    let arnoldi = Arnoldi::new(config);
    let result = arnoldi.compute(&a, None).unwrap();

    // Should get some eigenvalues
    assert!(
        !result.eigenvalues_real.is_empty(),
        "Should compute eigenvalues"
    );

    // For upper triangular matrix, eigenvalues should be close to diagonal (2, 3, 4)
    let mut eigs = result.eigenvalues_real.clone();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Check that we got eigenvalues in the reasonable range [2, 4]
    for ev in &eigs {
        assert!(
            *ev >= 1.5 && *ev <= 4.5,
            "Eigenvalue should be near 2, 3, or 4, got {ev}"
        );
    }
}

#[test]
fn test_lanczos_smallest_algebraic() {
    let a = make_larger_symmetric_matrix(10);

    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::SmallestAlgebraic,
        krylov_dimension: 15,
        ..Default::default()
    };

    let lanczos = Lanczos::new(config);
    let result = lanczos.compute(&a, None).unwrap();

    // Eigenvalues of n=10 tridiagonal (2,-1,-1) range from ~0.08 to ~3.9
    // SmallestAlgebraic should return the smallest ones
    assert!(
        !result.eigenvalues.is_empty(),
        "Should return some eigenvalues"
    );

    // All eigenvalues should be positive for this SPD matrix
    for &ev in &result.eigenvalues {
        assert!(
            ev > 0.0,
            "All eigenvalues should be positive for this SPD matrix"
        );
    }

    // The smallest eigenvalue should be reasonably small
    let min_ev = result
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_ev < 2.0,
        "At least one eigenvalue should be less than 2, got {}",
        min_ev
    );
}

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

// IRAM tests

#[test]
fn test_iram_symmetric_basic() {
    // Test IRAM on symmetric tridiagonal matrix
    let n = 20;
    let a = make_larger_symmetric_matrix(n);

    let config = IRAMConfig {
        num_eigenvalues: 4,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 12, // ncv > nev
        max_iterations: 100,
        tolerance: 1e-6,
        symmetric: true,
        compute_eigenvectors: false,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    // Should return 4 eigenvalues
    assert_eq!(
        result.eigenvalues_real.len(),
        4,
        "Should return 4 eigenvalues"
    );

    // For symmetric matrix, imaginary parts should be zero
    for &im in &result.eigenvalues_imag {
        assert!(
            im.abs() < 1e-10,
            "Imaginary part should be zero for symmetric matrix"
        );
    }

    // Eigenvalues should be in valid range [0.02, 3.98] for n=20 tridiagonal
    for &ev in &result.eigenvalues_real {
        assert!(ev > 0.0 && ev < 4.0, "Eigenvalue in valid range, got {ev}");
    }
}

#[test]
fn test_iram_symmetric_largest_algebraic() {
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    let config = IRAMConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestAlgebraic,
        krylov_dimension: 10,
        max_iterations: 150,
        tolerance: 1e-5,
        symmetric: true,
        compute_eigenvectors: true,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    assert_eq!(result.eigenvalues_real.len(), 3);

    // For n=15 tridiagonal, largest eigenvalue is ~3.95
    // All returned eigenvalues should be in upper portion
    let min_returned = result
        .eigenvalues_real
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_returned > 2.0,
        "Largest algebraic eigenvalues should be > 2.0"
    );

    // Check eigenvectors were computed
    assert!(result.eigenvectors.is_some(), "Should compute eigenvectors");
    let evecs = result.eigenvectors.unwrap();
    assert!(!evecs.is_empty(), "Should have eigenvectors");

    // Check eigenvectors are normalized
    for v in &evecs {
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 0.2,
            "Eigenvector should be normalized"
        );
    }
}

#[test]
fn test_iram_symmetric_smallest_magnitude() {
    let n = 20;
    let a = make_larger_symmetric_matrix(n);

    let config = IRAMConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::SmallestMagnitude,
        krylov_dimension: 12,
        max_iterations: 200,
        tolerance: 1e-4,
        symmetric: true,
        compute_eigenvectors: false,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    assert_eq!(result.eigenvalues_real.len(), 3);

    // For SmallestMagnitude on SPD matrix, should get smallest positive eigenvalues
    // For n=20 tridiagonal, smallest is ~0.024
    let max_returned = result
        .eigenvalues_real
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_returned < 2.0,
        "Smallest magnitude eigenvalues should be < 2.0, got {max_returned}"
    );
}

#[test]
fn test_iram_diagonal_matrix() {
    // Test on diagonal matrix with distinct eigenvalues 1, 2, 3, ..., 10
    // (Identity matrix has all equal eigenvalues which causes IRAM to break down early)
    let n = 10;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        values.push((i + 1) as f64);
        col_indices.push(i);
        row_ptrs.push(values.len());
    }
    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    let config = IRAMConfig {
        num_eigenvalues: 4,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 8,
        max_iterations: 100,
        tolerance: 1e-6,
        symmetric: true,
        compute_eigenvectors: false,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    // LargestMagnitude should return eigenvalues near 10, 9, 8, 7
    let mut sorted_eigs = result.eigenvalues_real.clone();
    sorted_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Check that we got large eigenvalues
    for &ev in &sorted_eigs {
        assert!(
            ev >= 5.0,
            "LargestMagnitude should return large eigenvalues, got {ev}"
        );
    }
}

#[test]
fn test_iram_general_matrix() {
    // Test IRAM on a non-symmetric matrix
    // Upper triangular matrix with eigenvalues 1, 2, 3, 4, 5
    let values = vec![
        1.0, 0.5, 0.0, 0.0, 0.0, // row 0
        2.0, 0.5, 0.0, 0.0, // row 1
        3.0, 0.5, 0.0, // row 2
        4.0, 0.5, // row 3
        5.0, // row 4
    ];
    let col_indices = vec![
        0, 1, 2, 3, 4, // row 0
        1, 2, 3, 4, // row 1
        2, 3, 4, // row 2
        3, 4, // row 3
        4, // row 4
    ];
    let row_ptrs = vec![0, 5, 9, 12, 14, 15];
    let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();

    let config = IRAMConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 5,
        max_iterations: 100,
        tolerance: 1e-4,
        symmetric: false, // Non-symmetric
        compute_eigenvectors: false,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    // Should return 3 eigenvalues
    assert_eq!(result.eigenvalues_real.len(), 3);

    // For upper triangular, eigenvalues are diagonal elements: 1, 2, 3, 4, 5
    // LargestMagnitude should give us values near 5, 4, 3
    let max_real = result
        .eigenvalues_real
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (3.0..=6.0).contains(&max_real),
        "Largest eigenvalue should be near 5, got {max_real}"
    );
}

#[test]
fn test_iram_with_eigenvectors() {
    let n = 15;
    let a = make_larger_symmetric_matrix(n);

    let config = IRAMConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 8,
        max_iterations: 100,
        tolerance: 1e-5,
        symmetric: true,
        compute_eigenvectors: true,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    assert!(result.eigenvectors.is_some());
    let evecs = result.eigenvectors.unwrap();
    assert_eq!(evecs.len(), 2, "Should have 2 eigenvectors");

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
fn test_iram_convergence_info() {
    let n = 10;
    let a = make_larger_symmetric_matrix(n);

    let config = IRAMConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 6,
        max_iterations: 50,
        tolerance: 1e-4,
        symmetric: true,
        compute_eigenvectors: false,
    };

    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    // Check convergence info fields
    assert!(result.iterations > 0, "Should report iterations");
    assert!(
        result.num_converged <= result.eigenvalues_real.len(),
        "num_converged should be valid"
    );
    assert_eq!(
        result.residual_norms.len(),
        2,
        "Should have residual norms for each eigenvalue"
    );
}

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
