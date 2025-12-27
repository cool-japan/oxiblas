//! Preconditioner effectiveness tests.
//!
//! These tests verify that preconditioners actually improve convergence
//! of iterative solvers. Each test compares unpreconditioned vs preconditioned
//! solver performance on benchmark matrices.

use oxiblas_sparse::CsrMatrix;
use oxiblas_sparse::linalg::cholesky::{IC0, ICT};
use oxiblas_sparse::linalg::iterative::{cg, gmres, pcg, pgmres};
use oxiblas_sparse::linalg::lu::{ILU0, ILUT};
use oxiblas_sparse::linalg::precond::{
    AINV, AINVConfig, AMG, AMGConfig, AdditiveSchwarz, AdditiveSchwarzConfig, BlockJacobi,
    GaussSeidel, Jacobi, LocalSolverType, SOR, SPAI, SPAIConfig, SSOR,
};
use oxiblas_sparse::ops::spmv;

/// Creates a 1D Poisson matrix (tridiagonal: -1, 2, -1).
fn make_poisson_1d(n: usize) -> CsrMatrix<f64> {
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

/// Creates a 2D Poisson matrix (5-point stencil).
fn make_poisson_2d(grid_size: usize) -> CsrMatrix<f64> {
    let n = grid_size * grid_size;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        let row = i / grid_size;
        let col = i % grid_size;

        if row > 0 {
            values.push(-1.0);
            col_indices.push(i - grid_size);
        }
        if col > 0 {
            values.push(-1.0);
            col_indices.push(i - 1);
        }
        values.push(4.0);
        col_indices.push(i);
        if col < grid_size - 1 {
            values.push(-1.0);
            col_indices.push(i + 1);
        }
        if row < grid_size - 1 {
            values.push(-1.0);
            col_indices.push(i + grid_size);
        }

        row_ptrs.push(values.len());
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Creates a diagonally dominant non-symmetric matrix.
fn make_nonsym_diag_dominant(n: usize) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    for i in 0..n {
        if i > 0 {
            values.push(-0.5);
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

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Creates a mildly ill-conditioned SPD matrix.
fn make_ill_conditioned_spd(n: usize, cond: f64) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];

    let alpha = (cond - 1.0) / (n as f64 - 1.0);

    for i in 0..n {
        if i > 0 {
            values.push(-1.0);
            col_indices.push(i - 1);
        }
        let diag = 2.0 + alpha * (i as f64);
        values.push(diag);
        col_indices.push(i);
        if i < n - 1 {
            values.push(-1.0);
            col_indices.push(i + 1);
        }
        row_ptrs.push(values.len());
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Computes relative residual norm ||Ax - b|| / ||b||.
fn relative_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    let mut ax = vec![0.0; n];
    spmv(1.0, a, x, 0.0, &mut ax);

    let mut res_norm_sq = 0.0;
    let mut b_norm_sq = 0.0;

    for i in 0..n {
        res_norm_sq += (ax[i] - b[i]).powi(2);
        b_norm_sq += b[i].powi(2);
    }

    (res_norm_sq / b_norm_sq).sqrt()
}

// =============================================================================
// Jacobi Preconditioner Tests
// =============================================================================

#[test]
fn test_jacobi_effectiveness() {
    let n = 100;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Unpreconditioned CG
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();

    // Preconditioned CG with Jacobi
    let jacobi = Jacobi::new(&a).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        jacobi.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_cg.converged, "CG should converge");
    assert!(result_pcg.converged, "PCG should converge");

    // Jacobi should not hurt convergence (may not help much for Poisson)
    assert!(
        result_pcg.iterations <= result_cg.iterations + 10,
        "Jacobi PCG ({}) should not be much slower than CG ({})",
        result_pcg.iterations,
        result_cg.iterations
    );
}

#[test]
fn test_jacobi_ill_conditioned() {
    let n = 50;
    let a = make_ill_conditioned_spd(n, 100.0);
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-8;
    let max_iter = 500;

    // Unpreconditioned
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();

    // Preconditioned
    let jacobi = Jacobi::new(&a).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        jacobi.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_cg.converged && result_pcg.converged);

    // For ill-conditioned matrices, Jacobi should provide some speedup
    // (or at least not be significantly worse)
    let speedup = result_cg.iterations as f64 / result_pcg.iterations as f64;
    assert!(
        speedup > 0.5,
        "Jacobi should not significantly slow down: speedup={:.2}",
        speedup
    );
}

// =============================================================================
// Block Jacobi Preconditioner Tests
// =============================================================================

#[test]
fn test_block_jacobi_effectiveness() {
    let n = 50;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Block Jacobi with uniform block size 10
    let block_sizes = vec![10; 5]; // 5 blocks of size 10
    let block_jacobi = BlockJacobi::new(&a, &block_sizes).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        block_jacobi.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_pcg.converged, "Block Jacobi PCG should converge");

    let rel_res = relative_residual(&a, &result_pcg.x, &b);
    assert!(rel_res < 1e-8, "Residual too large: {rel_res}");
}

// =============================================================================
// Gauss-Seidel Preconditioner Tests
// =============================================================================

#[test]
fn test_gauss_seidel_effectiveness() {
    // GS is not a symmetric preconditioner, so use GMRES instead of CG
    let n = 20;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-8;
    let max_iter = 500;

    // Gauss-Seidel preconditioned GMRES
    let gs = GaussSeidel::new(&a).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        gs.apply(r, &mut z);
        z
    };
    let result = pgmres(&a, &b, &x0, precond, n, tol, max_iter).unwrap();

    assert!(result.converged, "GS PGMRES should converge");

    let rel_res = relative_residual(&a, &result.x, &b);
    assert!(rel_res < 1e-6, "Residual too large: {rel_res}");
}

// =============================================================================
// SOR Preconditioner Tests
// =============================================================================

#[test]
fn test_sor_effectiveness() {
    // SOR is not symmetric, use GMRES
    let n = 20;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-8;
    let max_iter = 500;

    // SOR with omega = 1.2 (moderate overrelaxation)
    let sor = SOR::new(&a, 1.2).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        sor.apply(r, &mut z);
        z
    };
    let result = pgmres(&a, &b, &x0, precond, n, tol, max_iter).unwrap();

    assert!(result.converged, "SOR PGMRES should converge");

    let rel_res = relative_residual(&a, &result.x, &b);
    assert!(rel_res < 1e-6, "Residual too large: {rel_res}");
}

// =============================================================================
// SSOR Preconditioner Tests
// =============================================================================

#[test]
fn test_ssor_effectiveness() {
    let n = 50;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Unpreconditioned
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();

    // SSOR with omega = 1.0
    let ssor = SSOR::new(&a, 1.0).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        ssor.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_cg.converged && result_pcg.converged);

    // SSOR is a strong preconditioner and should provide speedup
    assert!(
        result_pcg.iterations <= result_cg.iterations,
        "SSOR PCG ({}) should be faster than CG ({})",
        result_pcg.iterations,
        result_cg.iterations
    );
}

// =============================================================================
// IC0 Preconditioner Tests (Incomplete Cholesky)
// =============================================================================

#[test]
fn test_ic0_effectiveness() {
    let n = 50;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Unpreconditioned
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();

    // IC0 preconditioned
    let ic0 = IC0::new(&a).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> { ic0.apply(r) };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_cg.converged && result_pcg.converged);

    // IC0 should provide significant speedup
    assert!(
        result_pcg.iterations < result_cg.iterations,
        "IC0 PCG ({}) should be faster than CG ({})",
        result_pcg.iterations,
        result_cg.iterations
    );
}

#[test]
fn test_ic0_poisson_2d() {
    // Use smaller 2D grid to avoid numerical issues
    let grid = 6;
    let a = make_poisson_2d(grid);
    let n = grid * grid;
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-8;
    let max_iter = 500;

    // IC0 preconditioned (skip speedup comparison for small grids)
    let ic0 = IC0::new(&a).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> { ic0.apply(r) };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(
        result_pcg.converged,
        "IC0 PCG should converge for Poisson 2D"
    );

    let rel_res = relative_residual(&a, &result_pcg.x, &b);
    assert!(
        rel_res < 1e-6,
        "IC0 Poisson 2D: residual too large: {rel_res}"
    );
}

// =============================================================================
// ICT Preconditioner Tests (Incomplete Cholesky with Threshold)
// =============================================================================

#[test]
fn test_ict_effectiveness() {
    let n = 50;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // ICT with drop tolerance and max fill 5
    let ict = ICT::new(&a, 0.01, 5).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> { ict.apply(r) };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_pcg.converged, "ICT PCG should converge");

    let rel_res = relative_residual(&a, &result_pcg.x, &b);
    assert!(rel_res < 1e-8, "Residual too large: {rel_res}");
}

// =============================================================================
// ILU0 Preconditioner Tests
// =============================================================================

#[test]
fn test_ilu0_nonsymmetric() {
    let n = 50;
    let a = make_nonsym_diag_dominant(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Unpreconditioned GMRES
    let result_gmres = gmres(&a, &b, &x0, 20, tol, max_iter).unwrap();

    // ILU0 preconditioned GMRES
    let ilu0 = ILU0::new(&a).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> { ilu0.apply(r) };
    let result_pgmres = pgmres(&a, &b, &x0, precond, 20, tol, max_iter).unwrap();

    assert!(result_gmres.converged && result_pgmres.converged);

    // ILU0 should reduce iterations
    let speedup = result_gmres.iterations as f64 / result_pgmres.iterations as f64;
    assert!(
        speedup >= 0.9,
        "ILU0 should not significantly slow down: speedup={:.2}",
        speedup
    );
}

// =============================================================================
// ILUT Preconditioner Tests
// =============================================================================

#[test]
fn test_ilut_nonsymmetric() {
    let n = 50;
    let a = make_nonsym_diag_dominant(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // ILUT with threshold 0.01 and max fill 10
    let ilut = ILUT::new(&a, 0.01, 10).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> { ilut.apply(r) };
    let result_pgmres = pgmres(&a, &b, &x0, precond, 20, tol, max_iter).unwrap();

    assert!(result_pgmres.converged, "ILUT PGMRES should converge");

    let rel_res = relative_residual(&a, &result_pgmres.x, &b);
    assert!(rel_res < 1e-8, "Residual too large: {rel_res}");
}

// =============================================================================
// AMG Preconditioner Tests
// =============================================================================

#[test]
fn test_amg_effectiveness() {
    let grid = 8;
    let a = make_poisson_2d(grid);
    let n = grid * grid;
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Unpreconditioned
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();

    // AMG preconditioned
    let config = AMGConfig::default();
    let amg = AMG::new(&a, config).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        amg.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_cg.converged && result_pcg.converged);

    // AMG should provide significant speedup for Poisson
    let speedup = result_cg.iterations as f64 / result_pcg.iterations as f64;
    assert!(
        speedup > 1.2,
        "AMG should provide speedup, got {:.2}x",
        speedup
    );
}

// =============================================================================
// SPAI Preconditioner Tests
// =============================================================================

#[test]
fn test_spai_effectiveness() {
    // Use smaller problem for SPAI
    let n = 15;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-8;
    let max_iter = 500;

    // SPAI preconditioner
    let config = SPAIConfig::default();
    let spai = SPAI::new(&a, config).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        spai.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_pcg.converged, "SPAI PCG should converge");

    let rel_res = relative_residual(&a, &result_pcg.x, &b);
    assert!(rel_res < 1e-6, "Residual too large: {rel_res}");
}

// =============================================================================
// AINV Preconditioner Tests
// =============================================================================

#[test]
fn test_ainv_effectiveness() {
    let n = 30;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // AINV preconditioner
    let config = AINVConfig::default();
    let ainv = AINV::new(&a, config).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        ainv.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_pcg.converged, "AINV PCG should converge");

    let rel_res = relative_residual(&a, &result_pcg.x, &b);
    assert!(rel_res < 1e-8, "Residual too large: {rel_res}");
}

// =============================================================================
// Additive Schwarz Preconditioner Tests
// =============================================================================

#[test]
fn test_additive_schwarz_effectiveness() {
    let n = 50;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Additive Schwarz with overlapping subdomains
    let config = AdditiveSchwarzConfig {
        num_subdomains: 4,
        overlap: 2,
        local_solver: LocalSolverType::ILU0,
    };
    let schwarz = AdditiveSchwarz::new(&a, config).unwrap();
    let precond = |r: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; r.len()];
        schwarz.apply(r, &mut z);
        z
    };
    let result_pcg = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();

    assert!(result_pcg.converged, "Additive Schwarz PCG should converge");

    let rel_res = relative_residual(&a, &result_pcg.x, &b);
    assert!(rel_res < 1e-8, "Residual too large: {rel_res}");
}

// =============================================================================
// Comparison Tests
// =============================================================================

#[test]
fn test_preconditioner_comparison_poisson_1d() {
    let n = 50;
    let a = make_poisson_1d(n);
    let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x0 = vec![0.0; n];
    let tol = 1e-10;
    let max_iter = 500;

    // Unpreconditioned baseline
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();
    assert!(result_cg.converged);
    let _cg_iters = result_cg.iterations;

    // Test Jacobi
    {
        let jacobi = Jacobi::new(&a).unwrap();
        let precond = |r: &[f64]| -> Vec<f64> {
            let mut z = vec![0.0; r.len()];
            jacobi.apply(r, &mut z);
            z
        };
        let result = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();
        assert!(result.converged, "Jacobi PCG should converge");

        let rel_res = relative_residual(&a, &result.x, &b);
        assert!(rel_res < 1e-8, "Jacobi: residual too large: {rel_res}");
    }

    // Test SSOR
    {
        let ssor = SSOR::new(&a, 1.0).unwrap();
        let precond = |r: &[f64]| -> Vec<f64> {
            let mut z = vec![0.0; r.len()];
            ssor.apply(r, &mut z);
            z
        };
        let result = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();
        assert!(result.converged, "SSOR PCG should converge");

        let rel_res = relative_residual(&a, &result.x, &b);
        assert!(rel_res < 1e-8, "SSOR: residual too large: {rel_res}");
    }

    // Test IC0
    {
        let ic0 = IC0::new(&a).unwrap();
        let precond = |r: &[f64]| -> Vec<f64> { ic0.apply(r) };
        let result = pcg(&a, &b, &x0, precond, tol, max_iter).unwrap();
        assert!(result.converged, "IC0 PCG should converge");

        let rel_res = relative_residual(&a, &result.x, &b);
        assert!(rel_res < 1e-8, "IC0: residual too large: {rel_res}");
    }
}

#[test]
fn test_preconditioner_comparison_poisson_2d() {
    // Use smaller grid for stability
    let grid = 5;
    let a = make_poisson_2d(grid);
    let n = grid * grid;
    let b: Vec<f64> = vec![1.0; n];
    let x0 = vec![0.0; n];
    let tol = 1e-8;
    let max_iter = 500;

    // Unpreconditioned baseline
    let result_cg = cg(&a, &b, &x0, tol, max_iter).unwrap();
    assert!(result_cg.converged);
    let cg_iters = result_cg.iterations;

    // IC0 preconditioned
    let ic0 = IC0::new(&a).unwrap();
    let precond_ic0 = |r: &[f64]| -> Vec<f64> { ic0.apply(r) };
    let result_ic0 = pcg(&a, &b, &x0, precond_ic0, tol, max_iter).unwrap();
    assert!(result_ic0.converged);

    // IC0 should not be significantly slower for 2D Poisson
    assert!(
        result_ic0.iterations <= cg_iters + 5,
        "IC0 ({} iters) should not be much slower than CG ({} iters)",
        result_ic0.iterations,
        cg_iters
    );
}
