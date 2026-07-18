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
fn test_arnoldi_residual_fields_populated() {
    // Every returned eigenpair must carry a residual norm and a convergence flag,
    // with matching lengths and consistent aggregate flag.
    let a = make_larger_symmetric_matrix(15);
    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 12,
        tolerance: 1e-8,
        ..Default::default()
    };
    let result = Arnoldi::new(config).compute(&a, None).unwrap();

    assert_eq!(result.residual_norms.len(), result.eigenvalues_real.len());
    assert_eq!(result.converged_flags.len(), result.eigenvalues_real.len());

    // Residual norms must be finite and non-negative.
    for r in &result.residual_norms {
        assert!(r.is_finite(), "residual must be finite, got {r}");
        assert!(*r >= 0.0, "residual must be non-negative, got {r}");
    }

    // Aggregate flag must agree with the per-pair flags and the requested count.
    let converged_count = result.converged_flags.iter().filter(|&&c| c).count();
    assert_eq!(result.converged, converged_count >= 3);

    // A per-pair flag is set iff its residual meets the tolerance.
    for (r, &flag) in result
        .residual_norms
        .iter()
        .zip(result.converged_flags.iter())
    {
        assert_eq!(flag, *r <= 1e-8, "flag/residual mismatch at r={r}");
    }
}

#[test]
fn test_arnoldi_reports_true_convergence_on_happy_breakdown() {
    // Upper-triangular matrix; the all-ones start vector spans a 2-dimensional
    // invariant subspace, so Arnoldi finds exactly two *exact* eigenpairs.
    // Requesting two eigenvalues, both must be reported as genuinely converged.
    let values = vec![2.0, 1.0, 3.0, 1.0, 4.0];
    let col_indices = vec![0, 1, 1, 2, 2];
    let row_ptrs = vec![0, 2, 4, 5];
    let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

    let config = LanczosConfig {
        num_eigenvalues: 2,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 10,
        tolerance: 1e-8,
        ..Default::default()
    };
    let result = Arnoldi::new(config).compute(&a, None).unwrap();

    assert_eq!(result.eigenvalues_real.len(), 2);
    for r in &result.residual_norms {
        assert!(
            *r <= 1e-8,
            "exact Ritz pair should have tiny residual, got {r}"
        );
    }
    assert!(
        result.converged_flags.iter().all(|&c| c),
        "all returned eigenpairs should be flagged converged"
    );
    assert!(
        result.converged,
        "converged must be true when every requested eigenpair meets tolerance"
    );
}

#[test]
fn test_arnoldi_reports_honest_nonconvergence() {
    // Regression guard: with a Krylov subspace far smaller than the matrix, the
    // Ritz pairs have NOT numerically converged. The old code reported
    // `converged = actual_dim >= m.min(n)` = true unconditionally as soon as the
    // subspace filled up; the residual-based check must report false instead.
    let a = make_larger_symmetric_matrix(40);
    let config = LanczosConfig {
        num_eigenvalues: 4,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 8,
        tolerance: 1e-8,
        ..Default::default()
    };
    let result = Arnoldi::new(config).compute(&a, None).unwrap();

    // The subspace filled to its target size (iterations == krylov dimension)...
    assert_eq!(result.iterations, 8);
    // ...but that alone must NOT mark the run as converged.
    assert!(
        !result.converged,
        "an under-resolved Krylov subspace must not report convergence"
    );
    // At least one residual must genuinely exceed the tolerance.
    assert!(
        result.residual_norms.iter().any(|r| *r > 1e-8),
        "residual norms should reflect the lack of convergence"
    );
    // Residuals stay finite (inverse iteration on the near-singular block is
    // regularized, so no NaN/Inf leaks through).
    for r in &result.residual_norms {
        assert!(r.is_finite(), "residual must be finite, got {r}");
    }
}

#[test]
fn test_arnoldi_complex_path_residuals() {
    // Non-symmetric matrix whose leading 2x2 block [[1,-1],[1,1]] has the complex
    // conjugate eigenpair 1 +/- i, plus a real eigenvalue 3. This drives the
    // complex-conjugate branch of the residual computation (the 2n x 2n real
    // inverse-iteration system).
    // A = [[1,-1, 0],
    //      [1, 1, 0],
    //      [0, 0, 3]]
    let values = vec![1.0, -1.0, 1.0, 1.0, 3.0];
    let col_indices = vec![0, 1, 0, 1, 2];
    let row_ptrs = vec![0, 2, 4, 5];
    let a = CsrMatrix::<f64>::new(3, 3, row_ptrs, col_indices, values).unwrap();

    let config = LanczosConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 10,
        tolerance: 1e-6,
        ..Default::default()
    };
    let result = Arnoldi::new(config).compute(&a, None).unwrap();

    // The complex-conjugate branch must have been exercised.
    let has_complex_pair = result.eigenvalues_imag.iter().any(|im| im.abs() > 0.1);
    assert!(
        has_complex_pair,
        "should surface a complex conjugate pair, imag={:?}",
        result.eigenvalues_imag
    );

    // Every residual (real and complex parts) is finite and non-negative, and
    // each per-pair flag agrees exactly with the tolerance test -- the complex
    // path is a genuine residual computation via a matrix-vector product, so a
    // pair is only flagged converged when it truly meets the tolerance.
    assert_eq!(result.residual_norms.len(), result.eigenvalues_real.len());
    for (r, &flag) in result
        .residual_norms
        .iter()
        .zip(result.converged_flags.iter())
    {
        assert!(
            r.is_finite() && *r >= 0.0,
            "residual must be finite and non-negative, got {r}"
        );
        assert_eq!(flag, *r <= 1e-6, "flag/residual mismatch at r={r}");
    }
    let count = result.converged_flags.iter().filter(|&&c| c).count();
    assert_eq!(result.converged, count >= 3);
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

#[test]
fn test_lanczos_near_target_selects_target_nearest_eigenvalue() {
    // Eigenvalues of this 3x3 matrix are approximately 5.414, 4.0, 2.586.
    // Use krylov_dimension == n so the Krylov subspace spans all of R^3: the
    // Ritz values are then (numerically) exact eigenvalues of A, isolating the
    // *selection* criterion from Lanczos convergence quality.
    //
    // An explicit, asymmetric starting vector is used because the default
    // all-ones starting vector happens to be exactly orthogonal to this
    // matrix's eigenvector for eigenvalue 4.0 (a quirk of this particular
    // Toeplitz-tridiagonal test matrix), which would make that eigenvalue
    // unreachable by *any* Krylov method regardless of selection criterion.
    let a = make_symmetric_matrix();
    let init = [1.0, 0.3, 0.1];

    let near_target_config = LanczosConfig {
        num_eigenvalues: 1,
        which: WhichEigenvalues::NearTarget,
        krylov_dimension: 3,
        tolerance: 1e-10,
        ..Default::default()
    };
    let near_target_result = Lanczos::new(near_target_config)
        .with_target(4.0)
        .compute(&a, Some(&init))
        .unwrap();

    assert_eq!(near_target_result.eigenvalues.len(), 1);
    let selected = near_target_result.eigenvalues[0];
    assert!(
        (selected - 4.0).abs() < 1e-6,
        "NearTarget with target=4.0 should select the eigenvalue nearest 4.0 \
         (expected ~4.0), got {selected}"
    );

    // Prove the fix: the pre-fix code silently fell back to SmallestMagnitude
    // (which would have returned ~2.586 here, not ~4.0).
    let smallest_magnitude_config = LanczosConfig {
        num_eigenvalues: 1,
        which: WhichEigenvalues::SmallestMagnitude,
        krylov_dimension: 3,
        tolerance: 1e-10,
        ..Default::default()
    };
    let smallest_result = Lanczos::new(smallest_magnitude_config)
        .compute(&a, Some(&init))
        .unwrap();
    assert!(
        (smallest_result.eigenvalues[0] - selected).abs() > 1.0,
        "NearTarget(4.0) selection must differ from SmallestMagnitude selection"
    );

    // With no `with_target` call the target defaults to zero, so NearTarget
    // reduces to "nearest the origin", i.e. matches SmallestMagnitude for a
    // matrix with only positive eigenvalues.
    let default_target_config = LanczosConfig {
        num_eigenvalues: 1,
        which: WhichEigenvalues::NearTarget,
        krylov_dimension: 3,
        tolerance: 1e-10,
        ..Default::default()
    };
    let default_result = Lanczos::new(default_target_config)
        .compute(&a, Some(&init))
        .unwrap();
    assert!(
        (default_result.eigenvalues[0] - smallest_result.eigenvalues[0]).abs() < 1e-6,
        "NearTarget with default (unset) target should match SmallestMagnitude"
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
fn test_hessenberg_eigensolver_and_ritz_vector() {
    // H = [[2,1,0],[1,2,1],[0,1,2]] has eigenvalues 2 +/- sqrt(2) and 2, with
    // (unit) eigenvectors [1, sqrt2, 1]/2, [1, 0, -1]/sqrt2 and [1, -sqrt2, 1]/2.
    // This directly exercises the shifted-QR eigenvalue routine and the Ritz
    // eigenvector routine (inverse iteration on (H - lambda*I)) independently of
    // the outer Arnoldi iteration.
    let h = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 2.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let cfg = IRAMConfig::<f64> {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 100,
        tolerance: 1e-12,
        compute_eigenvectors: true,
        krylov_dimension: 3,
        symmetric: false,
    };
    let iram = IRAM::new(cfg);

    let (re, im) = iram.solve_hessenberg_eigenvalues(&h, 3).unwrap();
    let mut sorted = re.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let sqrt2 = 2.0_f64.sqrt();
    let expected = [2.0 + sqrt2, 2.0, 2.0 - sqrt2];
    for (got, want) in sorted.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-9,
            "eigenvalue {got} should equal {want}"
        );
    }
    for imi in &im {
        assert!(
            imi.abs() < 1e-12,
            "eigenvalues must be real, got imag {imi}"
        );
    }

    // Each Ritz eigenvector must satisfy H y = lambda y to machine precision,
    // and vectors for distinct eigenvalues must not be parallel (the old power
    // iteration returned the dominant eigenvector for every eigenvalue).
    let mut vecs = Vec::new();
    for &lam in &re {
        let y = iram.hessenberg_ritz_vector(&h, 3, lam);
        assert_eq!(y.len(), 3);
        let mut hy = [0.0; 3];
        for (i, hyi) in hy.iter_mut().enumerate() {
            for (j, yj) in y.iter().enumerate() {
                *hyi += h[i][j] * yj;
            }
        }
        let res: f64 = (0..3)
            .map(|i| (hy[i] - lam * y[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            res < 1e-9,
            "Ritz vector for lambda={lam} has residual ||Hy - lambda y||={res}"
        );
        vecs.push(y);
    }
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            let d: f64 = vecs[i].iter().zip(vecs[j].iter()).map(|(a, b)| a * b).sum();
            assert!(
                d.abs() < 0.9,
                "Ritz vectors {i},{j} should be distinct (|dot|={})",
                d.abs()
            );
        }
    }
}

#[test]
fn test_iram_general_residual_per_eigenvalue() {
    // With a truncated Krylov basis (ncv < n) the Arnoldi residual beta = ||f||
    // is nonzero, so the per-eigenvalue residual estimate beta*|e_m^T y_i| must
    // differ across the requested Ritz values. The old code reported a single
    // shared Hessenberg entry for every eigenvalue, making them all identical.
    let n = 12usize;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0usize];
    for i in 0..n {
        values.push((n - i) as f64);
        col_indices.push(i);
        if i + 1 < n {
            values.push(0.3);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }
    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    let config = IRAMConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 6, // strictly smaller than n => nonzero residual
        max_iterations: 200,
        tolerance: 1e-10,
        symmetric: false,
        compute_eigenvectors: false,
    };
    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    assert_eq!(result.residual_norms.len(), 3);
    // The estimates must not be all identical (the defining symptom of the bug).
    let r = &result.residual_norms;
    let max_spread = r
        .iter()
        .flat_map(|ri| r.iter().map(move |rj| (ri - rj).abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        max_spread > 0.0,
        "per-eigenvalue residual estimates should differ, got {r:?}"
    );
}

#[test]
fn test_iram_general_eigenvectors_residual() {
    // Non-symmetric upper-bidiagonal matrix: A[i,i] = n-i, A[i,i+1] = 0.5.
    // Eigenvalues are the (distinct, real) diagonal entries n, n-1, ..., 1.
    // This exercises the general (non-symmetric) eigenvector path. A full
    // Krylov basis (ncv == n) makes the Ritz values exact so the test isolates
    // the eigenvector reconstruction rather than the restart convergence.
    let n = 6usize;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0usize];
    for i in 0..n {
        values.push((n - i) as f64);
        col_indices.push(i);
        if i + 1 < n {
            values.push(0.5);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }
    let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

    let config = IRAMConfig {
        num_eigenvalues: 3,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: n,
        max_iterations: 300,
        tolerance: 1e-8,
        symmetric: false,
        compute_eigenvectors: true,
    };
    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    let evecs = result.eigenvectors.expect("eigenvectors requested");
    assert_eq!(evecs.len(), 3, "should return 3 eigenvectors");

    // Each computed eigenpair (lambda, x) must satisfy the eigen relation
    // A*x = lambda*x measured against the ORIGINAL operator A. The old code
    // used power iteration on H (dominant eigenvalue only), which failed this
    // for every non-dominant requested eigenvalue.
    for (k, x) in evecs.iter().enumerate() {
        assert_eq!(x.len(), n, "eigenvector {k} has wrong dimension");
        let xnorm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            xnorm > 0.5,
            "eigenvector {k} must be nonzero (norm {xnorm})"
        );

        let lambda = result.eigenvalues_real[k];
        let mut ax = vec![0.0; n];
        crate::ops::spmv(1.0, &a, x, 0.0, &mut ax);
        let res: f64 = ax
            .iter()
            .zip(x.iter())
            .map(|(axi, xi)| (axi - lambda * xi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            res < 1e-3,
            "eigenpair {k} (lambda={lambda}): ||A x - lambda x|| = {res} too large"
        );
    }

    // Distinct eigenvalues => the eigenvectors must be genuinely different, not
    // all collapsed onto the dominant one.
    for i in 0..evecs.len() {
        for j in (i + 1)..evecs.len() {
            let dot_ij: f64 = evecs[i]
                .iter()
                .zip(evecs[j].iter())
                .map(|(vi, vj)| vi * vj)
                .sum();
            assert!(
                dot_ij.abs() < 0.99,
                "eigenvectors {i} and {j} should not be parallel (|dot|={})",
                dot_ij.abs()
            );
        }
    }
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

// =============================================================================
// IRAM implicit-restart tests: real double-shift QR for non-symmetric problems
// =============================================================================

fn make_upper_bidiagonal_real(n: usize, super_val: f64) -> CsrMatrix<f64> {
    // Upper bidiagonal (hence non-symmetric): A[i,i] = i+1 gives distinct REAL
    // eigenvalues 1,2,...,n; A[i,i+1] = super_val is the superdiagonal.
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for i in 0..n {
        values.push((i + 1) as f64);
        col_indices.push(i);
        if i + 1 < n {
            values.push(super_val);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Block-diagonal matrix of 2x2 rotation-like blocks
/// `[[a_k, b_k], [-b_k, a_k]]`, whose eigenvalues are the genuinely complex-conjugate
/// pairs `a_k +/- i*b_k`. Here `a_k = 0.3` and `b_k = 1.6^k`, giving a well-separated
/// complex spectrum `0.3 +/- i*1.6^k`.
fn make_complex_block_diag(num_blocks: usize) -> CsrMatrix<f64> {
    let n = 2 * num_blocks;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for k in 0..num_blocks {
        let a = 0.3;
        let b = 1.6_f64.powi(k as i32);
        // row 2k
        values.push(a);
        col_indices.push(2 * k);
        values.push(b);
        col_indices.push(2 * k + 1);
        row_ptrs.push(values.len());
        // row 2k+1
        values.push(-b);
        col_indices.push(2 * k);
        values.push(a);
        col_indices.push(2 * k + 1);
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Reproduction of the adversarial finding: n=24 non-symmetric upper-bidiagonal matrix
/// with distinct REAL eigenvalues, ncv=10 << n forcing the IRAM implicit-restart path.
/// Previously returned converged=false with poor residuals.
#[test]
fn test_iram_restart_real_bidiagonal() {
    let n = 24;
    let a = make_upper_bidiagonal_real(n, 0.5);
    let config = IRAMConfig {
        num_eigenvalues: 4,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 10, // ncv = 10 << n = 24 -> restarts are exercised
        max_iterations: 200,
        tolerance: 1e-8,
        symmetric: false,
        compute_eigenvectors: true,
    };
    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    assert!(
        result.converged,
        "restart path must converge (got converged={}, num_converged={})",
        result.converged, result.num_converged
    );
    assert_eq!(result.num_converged, 4);

    // Eigenvalues of an upper-triangular matrix are its diagonal: the four largest are
    // 24, 23, 22, 21. They should all be (essentially) real.
    let mut got: Vec<f64> = result.eigenvalues_real.clone();
    for im in &result.eigenvalues_imag {
        assert!(im.abs() < 1e-6, "eigenvalues should be real, got imag {im}");
    }
    got.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let expected = [24.0, 23.0, 22.0, 21.0];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-3,
            "eigenvalue {g} should be close to {e}"
        );
    }

    // The true residual ||A x - lambda x|| of every returned eigenpair must be small.
    let evecs = result.eigenvectors.as_ref().unwrap();
    for (k, lam) in result.eigenvalues_real.iter().enumerate() {
        let x = &evecs[k];
        let mut ax = vec![0.0; n];
        crate::ops::spmv(1.0, &a, x, 0.0, &mut ax);
        let mut res = 0.0;
        let mut xn = 0.0;
        for i in 0..n {
            let d = ax[i] - lam * x[i];
            res += d * d;
            xn += x[i] * x[i];
        }
        let rel = res.sqrt() / xn.sqrt().max(1.0);
        assert!(
            rel < 1e-5,
            "actual residual for lambda={lam} too large: {rel:.3e}"
        );
    }
}

/// A genuinely complex spectrum forces the Francis double-shift path: the unwanted
/// Ritz values are complex-conjugate pairs, so they must be applied as real double
/// shifts. The computed eigenvalues must recover the correct complex-conjugate pairs
/// with small residuals.
#[test]
fn test_iram_restart_complex_conjugate_spectrum() {
    let num_blocks = 12; // n = 24, ncv = 10 << n -> restarts (and double shifts)
    let a = make_complex_block_diag(num_blocks);

    let config = IRAMConfig {
        num_eigenvalues: 4,
        which: WhichEigenvalues::LargestMagnitude,
        krylov_dimension: 10,
        max_iterations: 400,
        tolerance: 1e-7,
        symmetric: false,
        compute_eigenvectors: false,
    };
    let iram = IRAM::new(config);
    let result = iram.compute(&a, None).unwrap();

    assert!(
        result.converged,
        "complex-spectrum restart must converge (num_converged={})",
        result.num_converged
    );
    assert_eq!(result.num_converged, 4);

    // The block eigenvalues are the genuinely complex pairs 0.3 +/- i*1.6^k.
    let mut pairs: Vec<(f64, f64)> = result
        .eigenvalues_real
        .iter()
        .zip(result.eigenvalues_imag.iter())
        .map(|(re, im)| (*re, im.abs()))
        .collect();
    pairs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    // Each recovered eigenvalue must be an ACTUAL eigenvalue of the matrix: a real part
    // near 0.3 and an imaginary magnitude equal to some genuine 1.6^k. A broken restart
    // (e.g. treating conjugate shifts as single real shifts) corrupts the factorization
    // and yields spurious values, so this pins down the double-shift correctness.
    for (re, imabs) in &pairs {
        assert!(
            *imabs > 1.0,
            "expected genuinely complex eigenvalues (double-shift path), got |imag|={imabs}"
        );
        assert!(
            (re - 0.3).abs() < 1e-3,
            "real part {re} should be close to 0.3"
        );
        let k = imabs.ln() / 1.6_f64.ln();
        assert!(
            (k - k.round()).abs() < 1e-3,
            "|imag|={imabs} should be an exact eigenvalue magnitude 1.6^k"
        );
    }

    // The recovered eigenvalues form two complex-conjugate pairs (two +imag, two -imag),
    // and the two magnitudes are distinct dominant blocks.
    let n_pos = result
        .eigenvalues_imag
        .iter()
        .filter(|im| **im > 0.5)
        .count();
    let n_neg = result
        .eigenvalues_imag
        .iter()
        .filter(|im| **im < -0.5)
        .count();
    assert_eq!(n_pos, 2, "expected two eigenvalues with +imag");
    assert_eq!(n_neg, 2, "expected two eigenvalues with -imag");
    assert!(
        (pairs[0].1 - pairs[2].1).abs() > 1e-3 * pairs[0].1,
        "expected two distinct conjugate pairs"
    );

    // LargestMagnitude must land among the dominant blocks (|imag| >= 1.6^8).
    for (_, imabs) in &pairs {
        assert!(
            *imabs >= 1.6_f64.powi(8) * (1.0 - 1e-3),
            "recovered pair |imag|={imabs} is not among the dominant blocks"
        );
    }

    // Reported Ritz residual estimates must be small (accurate double-shift result).
    for r in &result.residual_norms {
        assert!(*r < 1e-4, "residual estimate too large: {r:.3e}");
    }
}
