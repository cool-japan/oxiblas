//! IRAM implicit-restart (real double-shift QR) tests for sparse eigenvalue solvers.

use super::super::*;
use crate::csr::CsrMatrix;
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
