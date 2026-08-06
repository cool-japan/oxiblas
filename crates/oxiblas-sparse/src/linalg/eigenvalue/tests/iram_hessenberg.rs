//! IRAM (symmetric), Hessenberg, and general IRAM tests for sparse eigenvalue solvers.

use super::super::*;
use super::make_larger_symmetric_matrix;
use crate::csr::CsrMatrix;
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
