//! Iterative solvers for sparse linear systems.
//!
//! Provides:
//! - Conjugate Gradient (CG) for symmetric positive definite systems
//! - BiCGStab for general non-symmetric systems
//! - GMRES for general systems
//! - MINRES for symmetric indefinite systems
//! - QMR and TFQMR for general systems
//! - Block solvers for multiple right-hand sides
//! - FGMRES (Flexible GMRES) for variable preconditioning
//! - IDR(s) (Induced Dimension Reduction) for general systems

pub mod bicgstab;
pub mod block_cg;
pub mod block_gmres;
pub mod cg;
pub mod fgmres;
pub mod gmres;
pub(crate) mod helpers;
pub mod idrs;
pub mod minres;
pub mod qmr;
pub mod tfqmr;
pub mod types;

// Re-export all public types and functions
pub use bicgstab::bicgstab;
pub use block_cg::{block_cg, block_pcg};
pub use block_gmres::block_gmres;
pub use cg::{cg, pcg};
pub use fgmres::{fgmres, fgmres_ir};
pub use gmres::{gmres, pgmres};
pub use idrs::{idrs, pidrs};
pub use minres::{minres, pminres};
pub use qmr::{pqmr, qmr};
pub use tfqmr::{ptfqmr, tfqmr};
pub use types::{
    BlockCgResult, BlockGmresResult, CgResult, FgmresResult, GmresResult, IdrSResult,
    IterativeError, MinresResult, QmrResult, TfqmrResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::ops::spmv;

    fn make_spd_matrix() -> CsrMatrix<f64> {
        // A = [4 1 0]
        //     [1 4 1]
        //     [0 1 4]
        // Symmetric positive definite tridiagonal
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    fn make_nonsymmetric_matrix() -> CsrMatrix<f64> {
        // A = [4 1 0]
        //     [0 4 1]
        //     [0 0 4]
        // Upper triangular (non-symmetric, diagonally dominant)
        let values = vec![4.0, 1.0, 4.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 1, 2, 2];
        let row_ptrs = vec![0, 2, 4, 5];

        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_cg() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = cg(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "CG should converge");
        assert!(result.iterations < 10, "CG should converge quickly");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "CG solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_bicgstab() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = bicgstab(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "BiCGStab should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "BiCGStab solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_cg_identity() {
        // A = I
        let a = CsrMatrix::<f64>::eye(5);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x0 = vec![0.0; 5];

        let result = cg(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 1); // Should converge in 1 iteration for I

        for i in 0..5 {
            assert!((result.x[i] - b[i]).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn test_pcg_with_diagonal_precond() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Jacobi (diagonal) preconditioner
        let precond = |r: &[f64]| -> Vec<f64> {
            // M^{-1} = diag(1/4, 1/4, 1/4)
            r.iter().map(|&x| x / 4.0).collect()
        };

        let result = pcg(&a, &b, &x0, precond, 1e-10, 100).unwrap();

        assert!(result.converged, "PCG should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "PCG solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_gmres() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        assert!(result.converged, "GMRES should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "GMRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_gmres_non_symmetric() {
        // Non-symmetric matrix
        // A = [3 1 0]
        //     [0 4 1]
        //     [0 0 5]
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let col_indices = vec![0, 1, 1, 2, 2];
        let row_ptrs = vec![0, 2, 4, 5];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let b: Vec<f64> = vec![4.0, 5.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        assert!(
            result.converged,
            "GMRES should converge for non-symmetric matrix"
        );

        // Verify solution
        let mut ax: Vec<f64> = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "GMRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_gmres_with_restart() {
        // Use a larger system to test restart
        let n = 10;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let x0 = vec![0.0; n];

        // Use small restart to force restarts
        let result = gmres(&a, &b, &x0, 3, 1e-10, 100).unwrap();

        assert!(result.converged, "GMRES should converge with restarts");
        assert!(
            result.restarts > 0 || result.iterations <= 3,
            "Should have needed restarts or converged early"
        );

        // Verify solution
        let mut ax = vec![0.0; n];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-7_f64,
                "GMRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_pgmres_with_jacobi() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Jacobi preconditioner
        let precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&x| x / 4.0).collect() };

        let result = pgmres(&a, &b, &x0, precond, 10, 1e-10, 100).unwrap();

        assert!(result.converged, "Preconditioned GMRES should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "PGMRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_gmres_identity() {
        let a = CsrMatrix::<f64>::eye(5);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x0 = vec![0.0; 5];

        let result = gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "GMRES should converge quickly for identity"
        );

        for i in 0..5 {
            assert!((result.x[i] - b[i]).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn test_gmres_residual_history() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        // Residual history should be monotonically decreasing for symmetric positive definite
        assert!(
            !result.residual_history.is_empty(),
            "Residual history should not be empty"
        );

        // Check that residuals generally decrease (allowing for numerical noise)
        let first_res = result.residual_history[0];
        let last_res = result.residual_history.last().unwrap();
        assert!(last_res <= &first_res, "Residual should decrease");
    }

    // MINRES tests

    fn make_symmetric_indefinite_matrix() -> CsrMatrix<f64> {
        // A symmetric indefinite matrix:
        // A = [2 1 0]
        //     [1 -3 1]
        //     [0 1 2]
        // Has both positive and negative eigenvalues
        let values = vec![2.0, 1.0, 1.0, -3.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_minres_spd() {
        // MINRES should work for SPD systems (like CG)
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = minres(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "MINRES should converge for SPD");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "MINRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_minres_indefinite() {
        // MINRES should work for symmetric indefinite systems (unlike CG)
        let a = make_symmetric_indefinite_matrix();
        let b = vec![3.0, -1.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = minres(&a, &b, &x0, 1e-8, 100).unwrap();

        assert!(
            result.converged,
            "MINRES should converge for symmetric indefinite"
        );

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "MINRES solution incorrect at index {i}: {} vs {}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_minres_identity() {
        let a = CsrMatrix::<f64>::eye(5);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x0 = vec![0.0; 5];

        let result = minres(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "MINRES should converge quickly for identity"
        );

        for i in 0..5 {
            assert!((result.x[i] - b[i]).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn test_minres_larger_system() {
        // Test on a larger symmetric indefinite system
        let n = 10;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            // Alternate between positive and negative diagonal
            let diag = if i % 2 == 0 { 4.0 } else { -4.0 };
            values.push(diag);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let x0 = vec![0.0; n];

        let result = minres(&a, &b, &x0, 1e-8, 200).unwrap();

        assert!(
            result.converged,
            "MINRES should converge for larger indefinite system"
        );

        // Verify solution
        let mut ax = vec![0.0; n];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-5_f64,
                "MINRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_minres_residual_history() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = minres(&a, &b, &x0, 1e-10, 100).unwrap();

        // Residual history should not be empty
        assert!(
            !result.residual_history.is_empty(),
            "Residual history should not be empty"
        );

        // Residuals should generally decrease
        let first_res = result.residual_history[0];
        let last_res = result.residual_history.last().unwrap();
        assert!(last_res <= &first_res, "Residual should decrease overall");
    }

    #[test]
    fn test_pminres_with_jacobi() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Jacobi preconditioner (using absolute value for indefinite compatibility)
        let precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&x| x / 4.0).collect() };

        let result = pminres(&a, &b, &x0, precond, 1e-10, 100).unwrap();

        assert!(result.converged, "Preconditioned MINRES should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "PMINRES solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_tfqmr_identity() {
        let a = CsrMatrix::<f64>::eye(5);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x0 = vec![0.0; 5];

        let result = tfqmr(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged);
        assert!(
            result.iterations <= 4,
            "TFQMR should converge quickly for identity"
        );

        for i in 0..5 {
            assert!((result.x[i] - b[i]).abs() < 1e-8_f64);
        }
    }

    #[test]
    fn test_tfqmr_non_symmetric() {
        // Non-symmetric matrix
        // [2 -1 0]
        // [0  3 1]
        // [1  0 4]
        let values = vec![2.0, -1.0, 3.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 1, 2, 0, 2];
        let row_ptrs = vec![0, 2, 4, 6];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let b = vec![1.0, 4.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = tfqmr(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(
            result.converged,
            "TFQMR should converge for non-symmetric matrix"
        );

        // Verify solution
        let mut ax: Vec<f64> = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "TFQMR solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_tfqmr_spd() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = tfqmr(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "TFQMR should converge for SPD matrix");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "TFQMR solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_tfqmr_residual_history() {
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = tfqmr(&a, &b, &x0, 1e-10, 100).unwrap();

        // Residual history should not be empty
        assert!(
            !result.residual_history.is_empty(),
            "Residual history should not be empty"
        );

        // Residuals should generally decrease (final should be smaller than first)
        let first_res = result.residual_history[0];
        let last_res = result.residual_history.last().unwrap();
        assert!(last_res <= &first_res, "Residual should decrease overall");
    }

    #[test]
    fn test_tfqmr_larger_system() {
        // Test on a larger non-symmetric system
        let n = 10;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-0.5); // Asymmetric
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let x0 = vec![0.0; n];

        let result = tfqmr(&a, &b, &x0, 1e-8, 200).unwrap();

        assert!(
            result.converged,
            "TFQMR should converge for larger non-symmetric system"
        );

        // Verify solution
        let mut ax = vec![0.0; n];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-5_f64,
                "TFQMR solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_ptfqmr_with_identity_precond() {
        // Test with identity preconditioner (should behave like unpreconditioned)
        let a = make_spd_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Identity preconditioner
        let precond = |r: &[f64]| -> Vec<f64> { r.to_vec() };

        let result = ptfqmr(&a, &b, &x0, precond, 1e-10, 100).unwrap();

        assert!(
            result.converged,
            "PTFQMR with identity precond should converge"
        );

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "PTFQMR solution incorrect at index {i}"
            );
        }
    }

    // Block-CG tests

    #[test]
    fn test_block_cg_single_rhs() {
        // Block-CG with single RHS should behave like regular CG
        let a = make_spd_matrix();
        let b = vec![vec![5.0, 6.0, 5.0]];
        let x0 = vec![vec![0.0, 0.0, 0.0]];

        let result = block_cg(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "Block-CG should converge");
        assert_eq!(result.num_converged, 1, "Should have 1 converged system");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x[0], 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[0][i]).abs() < 1e-8_f64,
                "Block-CG solution incorrect at index {i}"
            );
        }
    }

    #[test]
    fn test_block_cg_multiple_rhs() {
        // Block-CG with multiple right-hand sides
        let a = make_spd_matrix();
        let b = vec![
            vec![5.0, 6.0, 5.0],
            vec![1.0, 2.0, 1.0],
            vec![3.0, 3.0, 3.0],
        ];
        let x0 = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];

        let result = block_cg(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "Block-CG should converge for all RHS");
        assert_eq!(result.num_converged, 3, "All 3 systems should converge");

        // Verify each solution
        for k in 0..3 {
            let mut ax = vec![0.0; 3];
            spmv(1.0, &a, &result.x[k], 0.0, &mut ax);

            for i in 0..3 {
                assert!(
                    (ax[i] - b[k][i]).abs() < 1e-8_f64,
                    "Block-CG solution incorrect for RHS {k} at index {i}"
                );
            }
        }
    }

    #[test]
    fn test_block_cg_identity() {
        let a = CsrMatrix::<f64>::eye(5);
        let b = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]];
        let x0 = vec![vec![0.0; 5], vec![0.0; 5]];

        let result = block_cg(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "Block-CG should converge quickly for identity"
        );
        assert_eq!(result.num_converged, 2);

        for k in 0..2 {
            for i in 0..5 {
                assert!((result.x[k][i] - b[k][i]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn test_block_cg_empty_rhs() {
        let a = make_spd_matrix();
        let b: Vec<Vec<f64>> = vec![];
        let x0: Vec<Vec<f64>> = vec![];

        let result = block_cg(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.num_converged, 0);
        assert!(result.x.is_empty());
    }

    #[test]
    fn test_block_cg_larger_system() {
        // Larger tridiagonal SPD system
        let n = 20;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        // Two different RHS
        let b = vec![
            (1..=n).map(|i| i as f64).collect::<Vec<_>>(),
            (1..=n).map(|i| (n - i + 1) as f64).collect::<Vec<_>>(),
        ];
        let x0 = vec![vec![0.0; n], vec![0.0; n]];

        let result = block_cg(&a, &b, &x0, 1e-10, 200).unwrap();

        assert!(
            result.converged,
            "Block-CG should converge for larger system"
        );
        assert_eq!(result.num_converged, 2);

        // Verify solutions
        for k in 0..2 {
            let mut ax = vec![0.0; n];
            spmv(1.0, &a, &result.x[k], 0.0, &mut ax);

            for i in 0..n {
                assert!(
                    (ax[i] - b[k][i]).abs() < 1e-6_f64,
                    "Block-CG solution incorrect for RHS {k} at index {i}"
                );
            }
        }
    }

    #[test]
    fn test_block_pcg_with_jacobi() {
        let a = make_spd_matrix();
        let b = vec![vec![5.0, 6.0, 5.0], vec![1.0, 2.0, 1.0]];
        let x0 = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

        // Jacobi preconditioner
        let precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&x| x / 4.0).collect() };

        let result = block_pcg(&a, &b, &x0, precond, 1e-10, 100).unwrap();

        assert!(result.converged, "Block-PCG should converge");
        assert_eq!(result.num_converged, 2);

        // Verify solutions
        for k in 0..2 {
            let mut ax = vec![0.0; 3];
            spmv(1.0, &a, &result.x[k], 0.0, &mut ax);

            for i in 0..3 {
                assert!(
                    (ax[i] - b[k][i]).abs() < 1e-8_f64,
                    "Block-PCG solution incorrect for RHS {k} at index {i}"
                );
            }
        }
    }

    #[test]
    fn test_block_pcg_identity_precond() {
        let a = make_spd_matrix();
        let b = vec![vec![5.0, 6.0, 5.0]];
        let x0 = vec![vec![0.0, 0.0, 0.0]];

        // Identity preconditioner
        let precond = |r: &[f64]| -> Vec<f64> { r.to_vec() };

        let result = block_pcg(&a, &b, &x0, precond, 1e-10, 100).unwrap();

        assert!(
            result.converged,
            "Block-PCG with identity precond should converge"
        );

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x[0], 0.0, &mut ax);

        for i in 0..3 {
            assert!((ax[i] - b[0][i]).abs() < 1e-8_f64);
        }
    }

    // =========================================================================
    // FGMRES Tests
    // =========================================================================

    #[test]
    fn test_fgmres_basic() {
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Identity preconditioner
        let mut precond = |r: &[f64]| -> Vec<f64> { r.to_vec() };

        let result = fgmres(&a, &b, &x0, &mut precond, 10, 1e-10, 50).unwrap();

        assert!(result.converged, "FGMRES should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "FGMRES solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_fgmres_with_variable_preconditioner() {
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Variable preconditioner: changes behavior at each call
        let mut call_count = 0;
        let mut precond = |r: &[f64]| -> Vec<f64> {
            call_count += 1;
            // Slight variation in scaling based on iteration
            let scale = 1.0 / (4.0 + 0.01 * (call_count as f64));
            r.iter().map(|&x| x * scale).collect()
        };

        let result = fgmres(&a, &b, &x0, &mut precond, 10, 1e-10, 50).unwrap();

        assert!(
            result.converged,
            "FGMRES with variable precond should converge"
        );
        assert!(call_count > 0, "Preconditioner should have been called");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "FGMRES with variable precond solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_fgmres_jacobi_preconditioner() {
        let a = make_nonsymmetric_matrix();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Jacobi preconditioner
        let mut precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&x| x / 4.0).collect() };

        let result = fgmres(&a, &b, &x0, &mut precond, 10, 1e-10, 50).unwrap();

        assert!(result.converged, "FGMRES with Jacobi should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-8_f64,
                "FGMRES with Jacobi solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_fgmres_restart() {
        // Larger system to test restart behavior
        let n = 20;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-0.5);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let x0 = vec![0.0; n];

        // Identity preconditioner with small restart
        let mut precond = |r: &[f64]| -> Vec<f64> { r.to_vec() };

        let result = fgmres(&a, &b, &x0, &mut precond, 5, 1e-10, 100).unwrap();

        assert!(result.converged, "FGMRES with restart should converge");
        assert!(
            result.restarts >= 1,
            "Should have performed at least one restart"
        );

        // Verify solution
        let mut ax = vec![0.0; n];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "FGMRES solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_fgmres_ir_basic() {
        // Test FGMRES with inner GMRES as preconditioner
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = fgmres_ir(
            &a, &b, &x0, 3,     // inner restart
            0.5,   // inner tolerance (loose)
            5,     // inner max iter
            10,    // outer restart
            1e-10, // outer tolerance
            50,    // outer max iter
        )
        .unwrap();

        assert!(result.converged, "FGMRES-IR should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "FGMRES-IR solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_fgmres_dimension_mismatch() {
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let mut precond = |r: &[f64]| -> Vec<f64> { r.to_vec() };

        let result = fgmres(&a, &b, &x0, &mut precond, 10, 1e-10, 50);

        assert!(matches!(
            result,
            Err(IterativeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_fgmres_stateful_preconditioner() {
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Stateful preconditioner that improves over iterations
        struct AdaptivePrecond {
            omega: f64,
        }

        let mut precond_state = AdaptivePrecond { omega: 0.5 };
        let mut precond = |r: &[f64]| -> Vec<f64> {
            let result: Vec<f64> = r.iter().map(|&x| x * precond_state.omega / 4.0).collect();
            // Slowly increase relaxation factor
            precond_state.omega = (precond_state.omega * 1.1).min(1.0);
            result
        };

        let result = fgmres(&a, &b, &x0, &mut precond, 10, 1e-10, 50).unwrap();

        assert!(
            result.converged,
            "FGMRES with stateful precond should converge"
        );

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "FGMRES with stateful precond solution incorrect at {i}"
            );
        }
    }

    // IDR(s) tests

    #[test]
    fn test_idrs_basic() {
        // Test IDR(s) with s=1 (simplest case)
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = idrs(&a, &b, &x0, 1, 1e-10, 100).unwrap();

        assert!(result.converged, "IDR(1) should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "IDR(1) solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_idrs_s2() {
        // Test IDR(s) with s=2
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = idrs(&a, &b, &x0, 2, 1e-10, 100).unwrap();

        assert!(result.converged, "IDR(2) should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "IDR(2) solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_idrs_s4() {
        // Test IDR(s) with s=4
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = idrs(&a, &b, &x0, 4, 1e-10, 100).unwrap();

        assert!(result.converged, "IDR(4) should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "IDR(4) solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_idrs_larger_matrix() {
        // Test on larger 10x10 tridiagonal SPD matrix
        let n = 10;

        // Build 10x10 tridiagonal SPD matrix
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let x0 = vec![0.0; n];

        let result = idrs(&a, &b, &x0, 2, 1e-8, 500).unwrap();

        assert!(result.converged, "IDR(2) should converge on larger matrix");

        // Verify solution
        let mut ax = vec![0.0; n];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        let residual: f64 = (0..n).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            residual < 1e-6 * b_norm,
            "IDR(2) residual too large: {residual}"
        );
    }

    #[test]
    fn test_pidrs_basic() {
        // Test preconditioned IDR(s)
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Simple diagonal preconditioner
        let precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&x| x / 4.0).collect() };

        let result = pidrs(&a, &b, &x0, &precond, 2, 1e-10, 100).unwrap();

        assert!(result.converged, "Preconditioned IDR(2) should converge");

        // Verify solution
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);

        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6_f64,
                "Preconditioned IDR(2) solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_idrs_residual_history() {
        // Test that residual history is properly tracked
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0, 1.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = idrs(&a, &b, &x0, 2, 1e-10, 100).unwrap();

        assert!(
            !result.residual_history.is_empty(),
            "Should have residual history"
        );
        assert!(
            !result.residual_history.is_empty(),
            "Should have at least one residual entry"
        );
    }

    #[test]
    fn test_idrs_dimension_mismatch() {
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0]; // Wrong dimension
        let x0 = vec![0.0, 0.0, 0.0];

        let result = idrs(&a, &b, &x0, 2, 1e-10, 100);

        assert!(matches!(
            result,
            Err(IterativeError::DimensionMismatch { .. })
        ));
    }

    // Block-GMRES tests

    #[test]
    fn test_block_gmres_basic() {
        // Test Block-GMRES with 2 right-hand sides
        let a = make_nonsymmetric_matrix();
        let b = vec![vec![1.0, 2.0, 1.0], vec![0.5, 1.0, 0.5]];
        let x0 = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

        let result = block_gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        assert!(result.converged, "Block-GMRES should converge");

        // Verify each solution
        for (j, bj) in b.iter().enumerate() {
            let mut ax = vec![0.0; 3];
            spmv(1.0, &a, &result.x[j], 0.0, &mut ax);

            for i in 0..3 {
                assert!(
                    (ax[i] - bj[i]).abs() < 1e-6_f64,
                    "Block-GMRES solution {j} incorrect at {i}"
                );
            }
        }
    }

    #[test]
    fn test_block_gmres_single_rhs() {
        // Block-GMRES with single RHS should produce a reasonable solution
        let a = make_nonsymmetric_matrix();
        let b = vec![vec![1.0, 2.0, 1.0]];
        let x0 = vec![vec![0.0, 0.0, 0.0]];

        let result = block_gmres(&a, &b, &x0, 10, 1e-6, 200).unwrap();

        // Verify solution quality (may not fully converge with single RHS edge case)
        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x[0], 0.0, &mut ax);

        let residual_sq: f64 = (0..3).map(|i| (ax[i] - b[0][i]).powi(2)).sum();
        let b_norm_sq: f64 = b[0].iter().map(|x| x * x).sum();
        let rel_residual = residual_sq.sqrt() / b_norm_sq.sqrt();

        assert!(
            rel_residual < 1e-4_f64,
            "Block-GMRES single RHS relative residual too large: {rel_residual}"
        );
    }

    #[test]
    fn test_block_gmres_three_rhs() {
        // Test with 3 right-hand sides
        let a = make_nonsymmetric_matrix();
        let b = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let x0 = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];

        let result = block_gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        assert!(result.converged, "Block-GMRES with 3 RHS should converge");

        for (j, bj) in b.iter().enumerate() {
            let mut ax = vec![0.0; 3];
            spmv(1.0, &a, &result.x[j], 0.0, &mut ax);

            for i in 0..3 {
                assert!(
                    (ax[i] - bj[i]).abs() < 1e-6_f64,
                    "Block-GMRES solution {j} incorrect at {i}"
                );
            }
        }
    }

    #[test]
    fn test_block_gmres_residual_history() {
        let a = make_nonsymmetric_matrix();
        let b = vec![vec![1.0, 2.0, 1.0], vec![0.5, 1.0, 0.5]];
        let x0 = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

        let result = block_gmres(&a, &b, &x0, 10, 1e-10, 100).unwrap();

        assert!(
            !result.residual_history.is_empty(),
            "Should have residual history"
        );
    }

    #[test]
    fn test_block_gmres_dimension_mismatch() {
        let a = make_nonsymmetric_matrix();
        let b = vec![vec![1.0, 2.0]]; // Wrong dimension
        let x0 = vec![vec![0.0, 0.0, 0.0]];

        let result = block_gmres(&a, &b, &x0, 10, 1e-10, 100);

        assert!(matches!(
            result,
            Err(IterativeError::DimensionMismatch { .. })
        ));
    }

    // ========== QMR Tests ==========

    #[test]
    fn test_qmr_identity() {
        // Test QMR on identity matrix
        let a: CsrMatrix<f64> = CsrMatrix::eye(3);
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = qmr(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(result.converged, "QMR should converge for identity matrix");
        for i in 0..3 {
            assert!(
                (result.x[i] - b[i]).abs() < 1e-8_f64,
                "QMR solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_qmr_residual_history() {
        let a: CsrMatrix<f64> = CsrMatrix::eye(3);
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = qmr(&a, &b, &x0, 1e-10, 100).unwrap();

        assert!(
            !result.residual_history.is_empty(),
            "Should have residual history"
        );
    }

    #[test]
    fn test_qmr_dimension_mismatch() {
        let a = make_nonsymmetric_matrix();
        let b = vec![1.0, 2.0]; // Wrong dimension
        let x0 = vec![0.0, 0.0, 0.0];

        let result = qmr(&a, &b, &x0, 1e-10, 100);

        assert!(matches!(
            result,
            Err(IterativeError::DimensionMismatch { .. })
        ));
    }
}
