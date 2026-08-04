//! Lanczos and Arnoldi tests for sparse eigenvalue solvers.

use super::super::*;
use super::make_larger_symmetric_matrix;
use super::make_symmetric_matrix;
use crate::csr::CsrMatrix;
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
