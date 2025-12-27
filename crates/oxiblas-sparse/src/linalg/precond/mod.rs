//! Preconditioners for iterative solvers.
//!
//! This module provides preconditioners:
//! - Jacobi (diagonal scaling)
//! - Block Jacobi
//! - Gauss-Seidel
//! - SOR (Successive Over-Relaxation)
//! - SSOR (Symmetric SOR)
//! - AMG (Algebraic Multigrid) - classical Ruge-Stüben AMG
//! - SAMG (Smoothed Aggregation AMG)
//! - Polynomial (Neumann series, Chebyshev)
//! - SPAI (Sparse Approximate Inverse)
//! - AINV (Approximate Inverse)
//! - Additive Schwarz (domain decomposition)
//!
//! For more advanced preconditioners, see:
//! - ILU/ILUT/ILUTP in `lu` module
//! - IC/ICT in `cholesky` module

pub mod ainv;
pub mod amg;
pub mod gauss_seidel;
pub mod jacobi;
pub mod polynomial;
pub mod samg;
pub mod schwarz;
pub mod spai;
pub mod types;

pub use ainv::{AINV, AINVConfig};
pub use amg::{AMG, AMGConfig, AMGCycleType};
pub use gauss_seidel::{GaussSeidel, SOR, SSOR};
pub use jacobi::{BlockJacobi, Jacobi};
pub use polynomial::{Polynomial, PolynomialConfig, PolynomialType};
pub use samg::{SAMG, SAMGConfig};
pub use schwarz::{AdditiveSchwarz, AdditiveSchwarzConfig, LocalSolverType};
pub use spai::{SPAI, SPAIConfig};
pub use types::PreconditionerError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    #[test]
    fn test_jacobi_basic() {
        // Simple diagonal matrix
        // [2 0 0]
        // [0 3 0]
        // [0 0 4]
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let jacobi = Jacobi::new(&a).unwrap();

        let r = vec![4.0, 9.0, 16.0];
        let mut z = vec![0.0; 3];
        jacobi.apply(&r, &mut z);

        // z = r ./ diag(A) = [4/2, 9/3, 16/4] = [2, 3, 4]
        assert!((z[0] - 2.0_f64).abs() < 1e-10);
        assert!((z[1] - 3.0_f64).abs() < 1e-10);
        assert!((z[2] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_tridiagonal() {
        // Tridiagonal matrix with strong diagonal
        // [4 1 0]
        // [1 4 1]
        // [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let jacobi = Jacobi::new(&a).unwrap();

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        jacobi.apply(&r, &mut z);

        // z = r ./ diag(A) = [1/4, 1/4, 1/4] = [0.25, 0.25, 0.25]
        for &val in &z {
            assert!((val - 0.25_f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_jacobi_zero_diagonal() {
        // Matrix with zero diagonal
        // [0 1 0]
        // [1 0 1]
        // [0 1 0]
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 0, 2, 1];
        let row_ptrs = vec![0, 1, 3, 4];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let result = Jacobi::new(&a);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), PreconditionerError::ZeroDiagonal(0));
    }

    #[test]
    fn test_block_jacobi_single_block() {
        // 2x2 matrix as a single block
        // [4 1]
        // [1 3]
        let values = vec![4.0, 1.0, 1.0, 3.0];
        let col_indices = vec![0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap();

        let block_jacobi = BlockJacobi::new(&a, &[2]).unwrap();

        let r = vec![1.0, 1.0];
        let mut z = vec![0.0; 2];
        block_jacobi.apply(&r, &mut z);

        // Inverse of [4 1; 1 3] is (1/11) * [3 -1; -1 4]
        // z = A^{-1} * [1; 1] = (1/11) * [2; 3] = [0.181818..., 0.272727...]
        assert!((z[0] - 2.0_f64 / 11.0).abs() < 1e-6);
        assert!((z[1] - 3.0_f64 / 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_jacobi_two_blocks() {
        // 4x4 matrix with two 2x2 blocks
        // [2 1 0 0]
        // [1 2 0 0]
        // [0 0 3 1]
        // [0 0 1 3]
        let values = vec![2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0];
        let col_indices = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let row_ptrs = vec![0, 2, 4, 6, 8];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();

        let block_jacobi = BlockJacobi::new(&a, &[2, 2]).unwrap();

        let r = vec![1.0, 1.0, 2.0, 2.0];
        let mut z = vec![0.0; 4];
        block_jacobi.apply(&r, &mut z);

        // First block inverse: [2 1; 1 2]^{-1} = (1/3) * [2 -1; -1 2]
        // z[0:2] = (1/3) * [2 -1; -1 2] * [1; 1] = (1/3) * [1; 1] = [1/3, 1/3]
        assert!((z[0] - 1.0_f64 / 3.0).abs() < 1e-6);
        assert!((z[1] - 1.0_f64 / 3.0).abs() < 1e-6);

        // Second block inverse: [3 1; 1 3]^{-1} = (1/8) * [3 -1; -1 3]
        // z[2:4] = (1/8) * [3 -1; -1 3] * [2; 2] = (1/8) * [4; 4] = [1/2, 1/2]
        assert!((z[2] - 0.5_f64).abs() < 1e-6);
        assert!((z[3] - 0.5_f64).abs() < 1e-6);
    }

    #[test]
    fn test_block_jacobi_invalid_block_sizes() {
        let values = vec![1.0, 2.0, 3.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        // Block sizes sum to 4, but matrix is 3x3
        let result = BlockJacobi::new(&a, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gauss_seidel_basic() {
        // Simple tridiagonal system
        // [4 1 0]
        // [1 4 1]
        // [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let gs = GaussSeidel::new(&a).unwrap();

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        gs.apply(&r, &mut z);

        // Forward substitution:
        // z[0] = r[0] / a[0,0] = 1/4 = 0.25
        // z[1] = (r[1] - a[1,0]*z[0]) / a[1,1] = (1 - 1*0.25) / 4 = 0.1875
        // z[2] = (r[2] - a[2,1]*z[1]) / a[2,2] = (1 - 1*0.1875) / 4 = 0.203125
        assert!((z[0] - 0.25_f64).abs() < 1e-10);
        assert!((z[1] - 0.1875_f64).abs() < 1e-10);
        assert!((z[2] - 0.203125_f64).abs() < 1e-10);
    }

    #[test]
    fn test_gauss_seidel_diagonal() {
        // Diagonal matrix should give same result as Jacobi
        // [2 0 0]
        // [0 3 0]
        // [0 0 4]
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let gs = GaussSeidel::new(&a).unwrap();

        let r = vec![4.0, 9.0, 16.0];
        let mut z = vec![0.0; 3];
        gs.apply(&r, &mut z);

        // Should be same as Jacobi: z = r ./ diag
        assert!((z[0] - 2.0_f64).abs() < 1e-10);
        assert!((z[1] - 3.0_f64).abs() < 1e-10);
        assert!((z[2] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_sor_omega_one_equals_gauss_seidel() {
        // SOR with ω = 1 should be equivalent to Gauss-Seidel
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let sor = SOR::new(&a, 1.0).unwrap();
        let gs = GaussSeidel::new(&a).unwrap();

        let r = vec![1.0, 1.0, 1.0];
        let mut z_sor = vec![0.0; 3];
        let mut z_gs = vec![0.0; 3];

        sor.apply(&r, &mut z_sor);
        gs.apply(&r, &mut z_gs);

        for i in 0..3 {
            let diff: f64 = z_sor[i] - z_gs[i];
            assert!(diff.abs() < 1e-10);
        }
    }

    #[test]
    fn test_sor_with_relaxation() {
        // Test SOR with ω = 1.5
        // [4 1 0]
        // [1 4 1]
        // [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let sor = SOR::new(&a, 1.5).unwrap();

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        sor.apply(&r, &mut z);

        // Verify that solution is different from Gauss-Seidel
        // and that all entries are computed
        assert!(z[0] > 0.0);
        assert!(z[1] > 0.0);
        assert!(z[2] > 0.0);
    }

    #[test]
    fn test_ssor_basic() {
        // Test SSOR on symmetric matrix
        // [4 1 0]
        // [1 4 1]
        // [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let ssor = SSOR::new(&a, 1.0).unwrap();

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        ssor.apply(&r, &mut z);

        // SSOR should give a valid result
        assert!(z[0] > 0.0);
        assert!(z[1] > 0.0);
        assert!(z[2] > 0.0);
    }

    #[test]
    fn test_ssor_diagonal() {
        // For diagonal matrix, SSOR should reduce to diagonal scaling
        // [2 0 0]
        // [0 3 0]
        // [0 0 4]
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let ssor = SSOR::new(&a, 1.0).unwrap();

        let r = vec![4.0, 9.0, 16.0];
        let mut z = vec![0.0; 3];
        ssor.apply(&r, &mut z);

        // For diagonal matrix with ω=1, SSOR gives z = r ./ diag
        assert!((z[0] - 2.0_f64).abs() < 1e-10);
        assert!((z[1] - 3.0_f64).abs() < 1e-10);
        assert!((z[2] - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_ssor_with_relaxation() {
        // Test SSOR with ω = 1.5
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let ssor = SSOR::new(&a, 1.5).unwrap();

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        ssor.apply(&r, &mut z);

        // Verify computation completes successfully
        assert!(z[0] > 0.0);
        assert!(z[1] > 0.0);
        assert!(z[2] > 0.0);
    }

    #[test]
    fn test_amg_construction() {
        // Create a 1D Laplacian-like matrix (tridiagonal) with strong diagonal
        // 4 on diagonal, -1 on off-diagonals (diagonally dominant)
        let n = 100;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0); // Strong diagonal for stability
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let config = AMGConfig::default();
        let amg = AMG::new(&a, config).unwrap();

        // Check hierarchy was built
        assert!(amg.num_levels() >= 1);
        assert_eq!(amg.size(), n);

        // Grid complexity should be >= 1.0
        let gc = amg.grid_complexity();
        assert!(gc >= 1.0);
    }

    #[test]
    fn test_amg_apply_diagonal() {
        // Simple diagonal matrix
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        // Use small coarse threshold to force single level
        let config = AMGConfig {
            coarse_size_threshold: 10,
            ..Default::default()
        };
        let amg = AMG::new(&a, config).unwrap();

        let r = vec![2.0, 3.0, 4.0];
        let mut z = vec![0.0; 3];
        amg.apply(&r, &mut z);

        // For diagonal matrix, should get good approximation
        assert!(z[0] > 0.0);
        assert!(z[1] > 0.0);
        assert!(z[2] > 0.0);
    }

    #[test]
    fn test_amg_apply_1d_laplacian() {
        // Create a 1D Laplacian-like matrix
        let n = 50;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0); // Strengthen diagonal for stability
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let config = AMGConfig::default();
        let amg = AMG::new(&a, config).unwrap();

        // Test with ones vector as RHS
        let r: Vec<f64> = vec![1.0; n];
        let mut z = vec![0.0; n];
        amg.apply(&r, &mut z);

        // Should produce non-zero result
        let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_amg_w_cycle() {
        // Test W-cycle
        let n = 30;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(3.0);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let config = AMGConfig {
            cycle_type: AMGCycleType::W,
            ..Default::default()
        };
        let amg = AMG::new(&a, config).unwrap();

        let r: Vec<f64> = vec![1.0; n];
        let mut z = vec![0.0; n];
        amg.apply(&r, &mut z);

        // Should produce non-zero result
        let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_amg_small_matrix() {
        // Very small matrix - should work but probably single level
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AMGConfig {
            coarse_size_threshold: 10, // Force single level
            ..Default::default()
        };
        let amg = AMG::new(&a, config).unwrap();

        assert_eq!(amg.num_levels(), 1);

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];
        amg.apply(&r, &mut z);

        // Should produce result (check no NaN or infinity)
        for i in 0..z.len() {
            let z_i: f64 = z[i];
            assert!(!z_i.is_nan() && z_i != f64::INFINITY && z_i != f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_amg_config_default() {
        let config: AMGConfig<f64> = AMGConfig::default();
        assert_eq!(config.max_levels, 25);
        assert_eq!(config.coarse_size_threshold, 50);
        assert!((config.strength_threshold - 0.25).abs() < 1e-10);
        assert_eq!(config.pre_smooths, 1);
        assert_eq!(config.post_smooths, 1);
        assert_eq!(config.cycle_type, AMGCycleType::V);
    }

    #[test]
    fn test_amg_empty_matrix_error() {
        let a: CsrMatrix<f64> = CsrMatrix::new(0, 0, vec![0], vec![], vec![]).unwrap();
        let config = AMGConfig::default();
        let result = AMG::new(&a, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_amg_complexity_metrics() {
        // Create a larger test matrix with strong diagonal
        let n = 100;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_indices.push(i - 1);
            }
            values.push(4.0); // Strong diagonal
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0);
                col_indices.push(i + 1);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();
        let config = AMGConfig::default();
        let amg = AMG::new(&a, config).unwrap();

        let gc = amg.grid_complexity();
        let oc = amg.operator_complexity();

        // Grid and operator complexity should be >= 1.0
        assert!(gc >= 1.0);
        assert!(oc >= 1.0);
    }

    // Polynomial preconditioner tests

    #[test]
    fn test_polynomial_neumann_basic() {
        // Diagonally dominant matrix
        // [4 1 0]
        // [1 4 1]
        // [0 1 4]
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = PolynomialConfig {
            degree: 3,
            lambda_min: Some(0.5),
            lambda_max: Some(1.5),
        };
        let poly = Polynomial::new(&a, PolynomialType::Neumann, config).unwrap();

        let r: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        poly.apply(&r, &mut z);

        // Check that output is finite and non-zero
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
        let norm: f64 = z.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Output should be non-zero");
    }

    #[test]
    fn test_polynomial_chebyshev_basic() {
        // SPD tridiagonal matrix
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = PolynomialConfig {
            degree: 5,
            lambda_min: Some(0.3),
            lambda_max: Some(1.7),
        };
        let poly = Polynomial::new(&a, PolynomialType::Chebyshev, config).unwrap();

        let r: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        poly.apply(&r, &mut z);

        // Check that output is finite and non-zero
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
        let norm: f64 = z.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Output should be non-zero");
    }

    #[test]
    fn test_polynomial_degree_zero() {
        // With degree 0, should just be diagonal scaling
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = PolynomialConfig {
            degree: 0,
            lambda_min: None,
            lambda_max: None,
        };
        let poly = Polynomial::new(&a, PolynomialType::Chebyshev, config).unwrap();

        let r = vec![2.0, 6.0, 8.0];
        let mut z = vec![0.0; 3];
        poly.apply(&r, &mut z);

        // z[i] = r[i] / a[i,i]
        assert!((z[0] - 1.0_f64).abs() < 1e-10, "z[0] should be 1.0");
        assert!((z[1] - 2.0_f64).abs() < 1e-10, "z[1] should be 2.0");
        assert!((z[2] - 2.0_f64).abs() < 1e-10, "z[2] should be 2.0");
    }

    #[test]
    fn test_polynomial_auto_eigenvalue_estimation() {
        // Test automatic eigenvalue estimation via Gershgorin
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = PolynomialConfig::default();
        let poly = Polynomial::new(&a, PolynomialType::Chebyshev, config).unwrap();

        let (lambda_min, lambda_max) = poly.eigenvalue_bounds();
        assert!(lambda_min > 0.0, "Minimum eigenvalue should be positive");
        assert!(
            lambda_max > lambda_min,
            "Maximum should be greater than minimum"
        );
    }

    #[test]
    fn test_polynomial_larger_matrix() {
        // Larger tridiagonal matrix
        let n = 50;
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
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        // Test both polynomial types
        for poly_type in [PolynomialType::Neumann, PolynomialType::Chebyshev] {
            let config = PolynomialConfig {
                degree: 5,
                lambda_min: Some(0.2),
                lambda_max: Some(1.8),
            };
            let poly = Polynomial::new(&a, poly_type, config).unwrap();

            let r: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
            let mut z = vec![0.0; n];
            poly.apply(&r, &mut z);

            // Check that output is finite
            for val in &z {
                assert!(
                    !val.is_nan() && !val.is_infinite(),
                    "Output should be finite for {:?}",
                    poly_type
                );
            }
        }
    }

    #[test]
    fn test_polynomial_accessors() {
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = PolynomialConfig {
            degree: 7,
            lambda_min: Some(0.1),
            lambda_max: Some(2.0),
        };
        let poly = Polynomial::new(&a, PolynomialType::Neumann, config).unwrap();

        assert_eq!(poly.degree(), 7);
        assert_eq!(poly.poly_type(), PolynomialType::Neumann);

        let (lmin, lmax) = poly.eigenvalue_bounds();
        assert!((lmin - 0.1_f64).abs() < 1e-10);
        assert!((lmax - 2.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_zero_diagonal_error() {
        // Matrix with zero diagonal
        let values = vec![0.0, 1.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = PolynomialConfig::default();
        let result = Polynomial::new(&a, PolynomialType::Chebyshev, config);
        assert!(result.is_err());
    }

    // ========================================
    // SPAI Preconditioner Tests
    // ========================================

    #[test]
    fn test_spai_identity_matrix() {
        // For identity matrix, SPAI should produce identity
        let values = vec![1.0, 1.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig::default();
        let spai = SPAI::new(&a, config).unwrap();

        assert_eq!(spai.dim(), 3);

        let r: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        spai.apply(&r, &mut z);

        // M ≈ I, so z ≈ r
        for i in 0..3 {
            assert!(
                (z[i] - r[i]).abs() < 1e-6,
                "z[{}] = {} should be close to {}",
                i,
                z[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_spai_diagonal_matrix() {
        // Diagonal matrix: A = diag(2, 3, 4)
        // Inverse: M = diag(0.5, 1/3, 0.25)
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig::default();
        let spai = SPAI::new(&a, config).unwrap();

        let r: Vec<f64> = vec![4.0, 9.0, 8.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        spai.apply(&r, &mut z);

        // z = M * r ≈ [4/2, 9/3, 8/4] = [2, 3, 2]
        assert!(
            (z[0] - 2.0_f64).abs() < 1e-6,
            "z[0] = {} should be 2.0",
            z[0]
        );
        assert!(
            (z[1] - 3.0_f64).abs() < 1e-6,
            "z[1] = {} should be 3.0",
            z[1]
        );
        assert!(
            (z[2] - 2.0_f64).abs() < 1e-6,
            "z[2] = {} should be 2.0",
            z[2]
        );
    }

    #[test]
    fn test_spai_tridiagonal_matrix() {
        // Tridiagonal SPD matrix
        // [4 -1  0]
        // [-1 4 -1]
        // [0 -1  4]
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig {
            tolerance: 0.1,
            max_nnz_per_col: 3,
            use_a_pattern: true,
            max_iterations: 10,
        };
        let spai = SPAI::new(&a, config).unwrap();

        assert!(spai.nnz() > 0, "SPAI should have non-zero entries");

        let r: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        spai.apply(&r, &mut z);

        // Check that output is reasonable
        for i in 0..3 {
            assert!(
                !z[i].is_nan() && !z[i].is_infinite(),
                "Output should be finite"
            );
            assert!(z[i].abs() < 10.0, "Output should be bounded");
        }
    }

    #[test]
    fn test_spai_empty_matrix() {
        // Empty matrix (0x0)
        let a = CsrMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).unwrap();

        let config = SPAIConfig::default();
        let spai = SPAI::new(&a, config).unwrap();

        assert_eq!(spai.dim(), 0);
        assert_eq!(spai.nnz(), 0);
    }

    #[test]
    fn test_spai_single_element() {
        // Single element matrix [5]
        let values = vec![5.0];
        let col_indices = vec![0];
        let row_ptrs = vec![0, 1];
        let a = CsrMatrix::new(1, 1, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig::default();
        let spai = SPAI::new(&a, config).unwrap();

        assert_eq!(spai.dim(), 1);

        let r: Vec<f64> = vec![10.0];
        let mut z: Vec<f64> = vec![0.0];
        spai.apply(&r, &mut z);

        // M ≈ 1/5, so z ≈ 10/5 = 2
        assert!(
            (z[0] - 2.0_f64).abs() < 1e-6,
            "z[0] = {} should be 2.0",
            z[0]
        );
    }

    #[test]
    fn test_spai_non_square_error() {
        // Non-square matrix should fail
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 2, 0, 1, 2];
        let row_ptrs = vec![0, 3, 6];
        let a = CsrMatrix::new(2, 3, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig::default();
        let result = SPAI::new(&a, config);
        assert!(result.is_err(), "Non-square matrix should fail");
    }

    #[test]
    fn test_spai_diagonal_pattern() {
        // Test with diagonal-only pattern (use_a_pattern = false)
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig {
            tolerance: 0.1,
            max_nnz_per_col: 5,
            use_a_pattern: false, // Use diagonal pattern
            max_iterations: 5,
        };
        let spai = SPAI::new(&a, config).unwrap();

        assert!(spai.nnz() > 0, "SPAI should have entries");

        let r: Vec<f64> = vec![4.0, 4.0, 4.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        spai.apply(&r, &mut z);

        // With diagonal pattern, M is approximately diagonal
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_spai_larger_matrix() {
        // Larger tridiagonal matrix
        let n = 50;
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
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig {
            tolerance: 0.3,
            max_nnz_per_col: 5,
            use_a_pattern: true,
            max_iterations: 3,
        };
        let spai = SPAI::new(&a, config).unwrap();

        assert_eq!(spai.dim(), n);
        assert!(spai.nnz() > 0);

        let r: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let mut z: Vec<f64> = vec![0.0; n];
        spai.apply(&r, &mut z);

        // Check that output is reasonable
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_spai_config_default() {
        let config = SPAIConfig::default();
        assert!((config.tolerance - 0.4).abs() < 1e-10);
        assert_eq!(config.max_nnz_per_col, 10);
        assert!(config.use_a_pattern);
        assert_eq!(config.max_iterations, 5);
    }

    #[test]
    fn test_spai_accessors() {
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = SPAIConfig::default();
        let spai = SPAI::new(&a, config).unwrap();

        assert_eq!(spai.dim(), 3);
        assert!(spai.nnz() >= 3, "Should have at least diagonal entries");
    }

    // ========================================
    // AINV Preconditioner Tests
    // ========================================

    #[test]
    fn test_ainv_identity_matrix() {
        // For identity matrix, AINV should produce identity-like preconditioner
        let values = vec![1.0, 1.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig::default();
        let ainv = AINV::new(&a, config).unwrap();

        assert_eq!(ainv.dim(), 3);

        let r: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        ainv.apply(&r, &mut z);

        // M ≈ I, so z ≈ r
        for i in 0..3 {
            assert!(
                (z[i] - r[i]).abs() < 0.1,
                "z[{}] = {} should be close to {}",
                i,
                z[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_ainv_diagonal_matrix() {
        // Diagonal matrix: A = diag(2, 3, 4)
        // Inverse: M = diag(0.5, 1/3, 0.25)
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig::default();
        let ainv = AINV::new(&a, config).unwrap();

        let r: Vec<f64> = vec![4.0, 9.0, 8.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        ainv.apply(&r, &mut z);

        // z = M * r ≈ [4/2, 9/3, 8/4] = [2, 3, 2]
        assert!(
            (z[0] - 2.0_f64).abs() < 0.1,
            "z[0] = {} should be close to 2.0",
            z[0]
        );
        assert!(
            (z[1] - 3.0_f64).abs() < 0.1,
            "z[1] = {} should be close to 3.0",
            z[1]
        );
        assert!(
            (z[2] - 2.0_f64).abs() < 0.1,
            "z[2] = {} should be close to 2.0",
            z[2]
        );
    }

    #[test]
    fn test_ainv_tridiagonal_matrix() {
        // Tridiagonal SPD matrix
        // [4 -1  0]
        // [-1 4 -1]
        // [0 -1  4]
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig {
            drop_tolerance: 0.01,
            max_nnz_per_col: 10,
            modified_gs: true,
        };
        let ainv = AINV::new(&a, config).unwrap();

        assert!(ainv.nnz() > 0, "AINV should have non-zero entries");

        let r: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        ainv.apply(&r, &mut z);

        // Check that output is reasonable
        for i in 0..3 {
            assert!(
                !z[i].is_nan() && !z[i].is_infinite(),
                "Output should be finite"
            );
            assert!(z[i].abs() < 10.0, "Output should be bounded");
        }
    }

    #[test]
    fn test_ainv_empty_matrix() {
        // Empty matrix (0x0)
        let a = CsrMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).unwrap();

        let config = AINVConfig::default();
        let ainv = AINV::new(&a, config).unwrap();

        assert_eq!(ainv.dim(), 0);
        assert_eq!(ainv.nnz(), 0);
    }

    #[test]
    fn test_ainv_single_element() {
        // Single element matrix [5]
        let values = vec![5.0];
        let col_indices = vec![0];
        let row_ptrs = vec![0, 1];
        let a = CsrMatrix::new(1, 1, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig::default();
        let ainv = AINV::new(&a, config).unwrap();

        assert_eq!(ainv.dim(), 1);

        let r: Vec<f64> = vec![10.0];
        let mut z: Vec<f64> = vec![0.0];
        ainv.apply(&r, &mut z);

        // M ≈ 1/5, so z ≈ 10/5 = 2
        assert!(
            (z[0] - 2.0_f64).abs() < 0.1,
            "z[0] = {} should be close to 2.0",
            z[0]
        );
    }

    #[test]
    fn test_ainv_non_square_error() {
        // Non-square matrix should fail
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 2, 0, 1, 2];
        let row_ptrs = vec![0, 3, 6];
        let a = CsrMatrix::new(2, 3, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig::default();
        let result = AINV::new(&a, config);
        assert!(result.is_err(), "Non-square matrix should fail");
    }

    #[test]
    fn test_ainv_without_modified_gs() {
        // Test without modified Gram-Schmidt (simpler algorithm)
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig {
            drop_tolerance: 0.1,
            max_nnz_per_col: 5,
            modified_gs: false, // Disable modified GS
        };
        let ainv = AINV::new(&a, config).unwrap();

        assert!(ainv.nnz() > 0, "AINV should have entries");

        let r: Vec<f64> = vec![4.0, 4.0, 4.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        ainv.apply(&r, &mut z);

        // Check finite output
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_ainv_larger_matrix() {
        // Larger tridiagonal matrix
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
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig {
            drop_tolerance: 0.2,
            max_nnz_per_col: 5,
            modified_gs: true,
        };
        let ainv = AINV::new(&a, config).unwrap();

        assert_eq!(ainv.dim(), n);
        assert!(ainv.nnz() > 0);

        let r: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let mut z: Vec<f64> = vec![0.0; n];
        ainv.apply(&r, &mut z);

        // Check that output is reasonable
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_ainv_config_default() {
        let config = AINVConfig::default();
        assert!((config.drop_tolerance - 0.1).abs() < 1e-10);
        assert_eq!(config.max_nnz_per_col, 20);
        assert!(config.modified_gs);
    }

    #[test]
    fn test_ainv_accessors() {
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AINVConfig::default();
        let ainv = AINV::new(&a, config).unwrap();

        assert_eq!(ainv.dim(), 3);
        assert!(ainv.z_nnz() >= 3, "Z should have at least diagonal entries");
        assert!(ainv.w_nnz() >= 3, "W should have at least diagonal entries");
        assert_eq!(ainv.nnz(), ainv.z_nnz() + ainv.w_nnz());
    }

    // ========================================
    // Additive Schwarz Preconditioner Tests
    // ========================================

    #[test]
    fn test_additive_schwarz_diagonal_matrix() {
        // Diagonal matrix: A = diag(2, 3, 4, 5)
        let values = vec![2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 1, 2, 3];
        let row_ptrs = vec![0, 1, 2, 3, 4];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig {
            num_subdomains: 2,
            overlap: 0,
            local_solver: LocalSolverType::ILU0,
        };
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.dim(), 4);
        assert_eq!(schwarz.num_subdomains(), 2);

        let r: Vec<f64> = vec![4.0, 9.0, 8.0, 10.0];
        let mut z: Vec<f64> = vec![0.0; 4];
        schwarz.apply(&r, &mut z);

        // With ILU on diagonal, should get close to inverse
        for i in 0..4 {
            assert!(
                !z[i].is_nan() && !z[i].is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_additive_schwarz_tridiagonal() {
        // Tridiagonal SPD matrix
        let n = 8;
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
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig {
            num_subdomains: 2,
            overlap: 1,
            local_solver: LocalSolverType::ILU0,
        };
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.dim(), n);
        assert!(schwarz.num_subdomains() >= 2);

        let r: Vec<f64> = vec![1.0; n];
        let mut z: Vec<f64> = vec![0.0; n];
        schwarz.apply(&r, &mut z);

        // Check finite output
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_additive_schwarz_jacobi_solver() {
        // Test with Jacobi local solver
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig {
            num_subdomains: 2,
            overlap: 0,
            local_solver: LocalSolverType::Jacobi,
        };
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.local_solver(), LocalSolverType::Jacobi);

        let r: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        schwarz.apply(&r, &mut z);

        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_additive_schwarz_with_overlap() {
        // Test with different overlap levels
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
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        for overlap in [0, 1, 2] {
            let config = AdditiveSchwarzConfig {
                num_subdomains: 3,
                overlap,
                local_solver: LocalSolverType::ILU0,
            };
            let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

            assert_eq!(schwarz.dim(), n);

            let r: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
            let mut z: Vec<f64> = vec![0.0; n];
            schwarz.apply(&r, &mut z);

            for val in &z {
                assert!(
                    !val.is_nan() && !val.is_infinite(),
                    "Output should be finite for overlap {}",
                    overlap
                );
            }
        }
    }

    #[test]
    fn test_additive_schwarz_empty_matrix() {
        let a = CsrMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).unwrap();

        let config = AdditiveSchwarzConfig::default();
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.dim(), 0);
        assert_eq!(schwarz.num_subdomains(), 0);
    }

    #[test]
    fn test_additive_schwarz_single_element() {
        let values = vec![5.0];
        let col_indices = vec![0];
        let row_ptrs = vec![0, 1];
        let a = CsrMatrix::new(1, 1, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig::default();
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.dim(), 1);

        let r: Vec<f64> = vec![10.0];
        let mut z: Vec<f64> = vec![0.0];
        schwarz.apply(&r, &mut z);

        // M ≈ 1/5, so z ≈ 10/5 = 2
        assert!(
            (z[0] - 2.0_f64).abs() < 0.5,
            "z[0] = {} should be close to 2.0",
            z[0]
        );
    }

    #[test]
    fn test_additive_schwarz_non_square_error() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 2, 0, 1, 2];
        let row_ptrs = vec![0, 3, 6];
        let a = CsrMatrix::new(2, 3, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig::default();
        let result = AdditiveSchwarz::new(&a, config);
        assert!(result.is_err(), "Non-square matrix should fail");
    }

    #[test]
    fn test_additive_schwarz_many_subdomains() {
        // More subdomains than unknowns
        let values = vec![2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig {
            num_subdomains: 10, // More than n=3
            overlap: 0,
            local_solver: LocalSolverType::ILU0,
        };
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.dim(), 3);
        // Should be capped at n
        assert!(schwarz.num_subdomains() <= 3);

        let r: Vec<f64> = vec![4.0, 6.0, 8.0];
        let mut z: Vec<f64> = vec![0.0; 3];
        schwarz.apply(&r, &mut z);

        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Output should be finite"
            );
        }
    }

    #[test]
    fn test_additive_schwarz_config_default() {
        let config = AdditiveSchwarzConfig::default();
        assert_eq!(config.num_subdomains, 4);
        assert_eq!(config.overlap, 1);
        assert_eq!(config.local_solver, LocalSolverType::ILU0);
    }

    #[test]
    fn test_additive_schwarz_accessors() {
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let config = AdditiveSchwarzConfig {
            num_subdomains: 2,
            overlap: 1,
            local_solver: LocalSolverType::ILU0,
        };
        let schwarz = AdditiveSchwarz::new(&a, config).unwrap();

        assert_eq!(schwarz.dim(), 3);
        assert!(schwarz.num_subdomains() >= 1);
        assert_eq!(schwarz.local_solver(), LocalSolverType::ILU0);
        assert!(
            schwarz.total_subdomain_size() >= 3,
            "Total size should cover all unknowns"
        );
    }

    // ===== SA-AMG Tests =====

    #[test]
    fn test_samg_basic() {
        // Simple 1D Laplacian discretization (SPD)
        // [2 -1  0  0]
        // [-1 2 -1  0]
        // [0 -1  2 -1]
        // [0  0 -1  2]
        let n = 4;
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let row_ptrs = vec![0, 2, 5, 8, 10];
        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = SAMGConfig::default();
        let samg = SAMG::new(&a, config).unwrap();

        assert_eq!(samg.size(), n);
        assert!(samg.num_levels() >= 1);

        let r: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut z: Vec<f64> = vec![0.0; n];
        samg.apply(&r, &mut z);

        // Check output is finite
        for val in &z {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "SA-AMG output should be finite"
            );
        }
    }

    #[test]
    fn test_samg_larger_matrix() {
        // Larger 1D Laplacian
        let n = 20;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = SAMGConfig {
            coarse_size_threshold: 5,
            ..Default::default()
        };
        let samg = SAMG::new(&a, config).unwrap();

        assert_eq!(samg.size(), n);
        assert!(samg.num_levels() >= 2, "Should have multiple levels");

        let r: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mut z: Vec<f64> = vec![0.0; n];
        samg.apply(&r, &mut z);

        for val in &z {
            assert!(!val.is_nan() && !val.is_infinite());
        }

        // Grid complexity should be > 1 (we have coarse grids)
        assert!(samg.grid_complexity() > 1.0);
    }

    #[test]
    fn test_samg_w_cycle() {
        let n = 10;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = SAMGConfig {
            cycle_type: AMGCycleType::W,
            coarse_size_threshold: 3,
            ..Default::default()
        };
        let samg = SAMG::new(&a, config).unwrap();

        let r: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mut z: Vec<f64> = vec![0.0; n];
        samg.apply(&r, &mut z);

        for val in &z {
            assert!(!val.is_nan() && !val.is_infinite());
        }
    }

    #[test]
    fn test_samg_diagonal_matrix() {
        // Pure diagonal matrix - should work but may have trivial hierarchy
        let n = 5;
        let values = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices: Vec<usize> = (0..n).collect();
        let row_ptrs: Vec<usize> = (0..=n).collect();
        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = SAMGConfig::default();
        let samg = SAMG::new(&a, config).unwrap();

        let r: Vec<f64> = vec![4.0, 6.0, 8.0, 10.0, 12.0];
        let mut z: Vec<f64> = vec![0.0; n];
        samg.apply(&r, &mut z);

        // For diagonal matrix, preconditioner should approximate inverse
        for i in 0..n {
            assert!(!z[i].is_nan() && !z[i].is_infinite());
        }
    }

    #[test]
    fn test_samg_complexities() {
        let n = 30;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(2.0);
            if i < n - 1 {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(col_indices.len());
        }

        let a = CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap();

        let config = SAMGConfig {
            coarse_size_threshold: 5,
            ..Default::default()
        };
        let samg = SAMG::new(&a, config).unwrap();

        let gc = samg.grid_complexity();
        let oc = samg.operator_complexity();

        // For tridiagonal Laplacian, complexities should be reasonable
        assert!(gc > 1.0 && gc < 10.0, "Grid complexity = {}", gc);
        assert!(oc > 1.0 && oc < 10.0, "Operator complexity = {}", oc);
    }

    #[test]
    fn test_samg_config_default() {
        let config: SAMGConfig<f64> = SAMGConfig::default();
        assert_eq!(config.max_levels, 25);
        assert_eq!(config.coarse_size_threshold, 50);
        assert_eq!(config.pre_smooths, 1);
        assert_eq!(config.post_smooths, 1);
        assert_eq!(config.prolongation_smoothing_steps, 1);
        assert_eq!(config.cycle_type, AMGCycleType::V);
    }

    #[test]
    fn test_samg_empty_matrix_error() {
        let a = CsrMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).unwrap();
        let config = SAMGConfig::default();
        let result = SAMG::new(&a, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_samg_non_square_error() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 2, 0, 1, 2];
        let row_ptrs = vec![0, 3, 6];
        let a = CsrMatrix::new(2, 3, row_ptrs, col_indices, values).unwrap();

        let config = SAMGConfig::default();
        let result = SAMG::new(&a, config);
        assert!(result.is_err());
    }
}
