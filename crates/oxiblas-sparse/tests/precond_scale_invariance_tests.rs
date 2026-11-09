//! Scale-invariance regression tests for the preconditioners in
//! `oxiblas_sparse::linalg::precond` and the incomplete Cholesky factorizations
//! `IC0` / `ICT`.
//!
//! A preconditioner built from `s*A` must exist whenever the one built from `A`
//! does, and for a power of two `s` it must be exactly `1/s` times the
//! preconditioner of `A` (`M(s*A)^{-1} r = M(A)^{-1} r / s`, bit for bit),
//! because IEEE-754 arithmetic on consistently scaled operands is exactly
//! scaled (absent overflow and underflow). Any absolute threshold left in a
//! setup (such as `|a_ii| < 1e-14`) breaks this: it rejects every diagonal of
//! a matrix scaled by `2^-64`, or changes the setup of one scaled by `2^64`.
//!
//! The PCG runs then check the end-to-end consequence: with the preconditioner
//! rebuilt from the scaled matrix, PCG on `(s*A, s*b)` has exactly the same
//! outcome (iteration count, convergence flag, bit-identical solution, exactly
//! scaled residual, or a breakdown at the same iteration). Gauss-Seidel and SOR
//! (`M = D + omega*L`) are not symmetric, which PCG assumes, so their PCG runs
//! are not required to converge; they also run with right-preconditioned GMRES,
//! which must converge.

use oxiblas_sparse::CsrMatrix;
use oxiblas_sparse::linalg::cholesky::{IC0, ICT, SparseCholError};
use oxiblas_sparse::linalg::iterative::{IterativeError, pcg, pgmres};
use oxiblas_sparse::linalg::precond::{
    AINV, AINVConfig, AMG, AMGConfig, AdditiveSchwarz, AdditiveSchwarzConfig, BlockJacobi,
    GaussSeidel, Jacobi, LocalSolverType, Polynomial, PolynomialConfig, PolynomialType,
    PreconditionerError, SAMG, SAMGConfig, SOR, SPAI, SPAIConfig, SSOR,
};
use oxiblas_sparse::ops::spmv;

const TOL: f64 = 1e-10;
const MAX_ITER: usize = 1000;
const RESTART: usize = 20;
const GRID: usize = 10;
/// Bound on the true relative residual of a PCG run reported as converged.
const TRUE_RESIDUAL_BOUND: f64 = 1e-8;

/// `[2^-64, 2^64]`, both exactly representable.
fn scale_factors() -> [f64; 2] {
    let two_pow_64 = 18_446_744_073_709_551_616.0_f64;
    [1.0 / two_pow_64, two_pow_64]
}

// =============================================================================
// Problem generators
// =============================================================================

/// SPD 5-point operator on a `grid x grid` mesh whose diagonal varies by row
/// (`4 + k/4`, exact in binary) and whose couplings are `-1`.
fn variable_spd(grid: usize) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for i in 0..n {
        let row = i / grid;
        let col = i % grid;
        if row > 0 {
            values.push(-1.0);
            col_indices.push(i - grid);
        }
        if col > 0 {
            values.push(-1.0);
            col_indices.push(i - 1);
        }
        values.push(4.0 + 0.25 * (i % 7) as f64);
        col_indices.push(i);
        if col + 1 < grid {
            values.push(-1.0);
            col_indices.push(i + 1);
        }
        if row + 1 < grid {
            values.push(-1.0);
            col_indices.push(i + grid);
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// Right-hand side with entries `1 + k/8` (exact in binary).
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + (i % 7) as f64 / 8.0).collect()
}

/// Deterministic probe vectors with entries of mixed sign and magnitude.
fn probes(n: usize) -> Vec<Vec<f64>> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut random = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut vectors = vec![rhs(n)];
    vectors.push((0..n).map(|_| random()).collect());
    vectors
}

fn scaled_matrix(a: &CsrMatrix<f64>, s: f64) -> CsrMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|&v| v * s).collect();
    CsrMatrix::new(
        a.nrows(),
        a.ncols(),
        a.row_ptrs().to_vec(),
        a.col_indices().to_vec(),
        values,
    )
    .unwrap()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|e| e * e).sum::<f64>().sqrt()
}

fn true_relative_residual(a: &CsrMatrix<f64>, b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    spmv(1.0, a, x, 0.0, &mut ax);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    norm2(&r) / norm2(b)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn scaled_bits(values: &[f64], s: f64) -> Vec<u64> {
    values.iter().map(|v| (v * s).to_bits()).collect()
}

// =============================================================================
// Uniform view of every preconditioner
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Precond {
    Jacobi,
    BlockJacobi,
    GaussSeidel,
    Sor,
    Ssor,
    Neumann,
    Chebyshev,
    Amg,
    Samg,
    Spai,
    Ainv,
    SchwarzIlu0,
    SchwarzExactLu,
    SchwarzJacobi,
    Ic0,
    /// ICT with an active drop tolerance and fill cap.
    IctDropping,
    /// ICT without dropping (the complete Cholesky factor).
    IctComplete,
}

impl Precond {
    /// Whether `M` is symmetric for symmetric `A` (a requirement of PCG).
    fn symmetric(self) -> bool {
        !matches!(self, Precond::GaussSeidel | Precond::Sor)
    }
}

/// Applies `M^{-1}` to a vector.
type Apply = Box<dyn Fn(&[f64]) -> Vec<f64>>;

/// Wraps an `apply(r, z)` preconditioner as an [`Apply`] closure.
fn apply_into<P: 'static>(p: P, f: fn(&P, &[f64], &mut [f64])) -> Apply {
    Box::new(move |r: &[f64]| {
        let mut z = vec![0.0; r.len()];
        f(&p, r, &mut z);
        z
    })
}

fn precond_error(e: PreconditionerError) -> String {
    e.to_string()
}

fn chol_error(e: SparseCholError) -> String {
    e.to_string()
}

/// Builds the preconditioner `kind` from `a`.
fn build(kind: Precond, a: &CsrMatrix<f64>) -> Result<Apply, String> {
    let n = a.nrows();
    let schwarz = |local_solver| AdditiveSchwarzConfig {
        num_subdomains: 4,
        overlap: 1,
        local_solver,
    };
    Ok(match kind {
        Precond::Jacobi => apply_into(Jacobi::new(a).map_err(precond_error)?, Jacobi::apply),
        Precond::BlockJacobi => {
            let blocks = vec![GRID; n / GRID];
            apply_into(
                BlockJacobi::new(a, &blocks).map_err(precond_error)?,
                BlockJacobi::apply,
            )
        }
        Precond::GaussSeidel => apply_into(
            GaussSeidel::new(a).map_err(precond_error)?,
            GaussSeidel::apply,
        ),
        Precond::Sor => apply_into(SOR::new(a, 1.25).map_err(precond_error)?, SOR::apply),
        Precond::Ssor => apply_into(SSOR::new(a, 1.25).map_err(precond_error)?, SSOR::apply),
        Precond::Neumann => apply_into(
            Polynomial::new(a, PolynomialType::Neumann, PolynomialConfig::default())
                .map_err(precond_error)?,
            Polynomial::apply,
        ),
        Precond::Chebyshev => apply_into(
            Polynomial::new(a, PolynomialType::Chebyshev, PolynomialConfig::default())
                .map_err(precond_error)?,
            Polynomial::apply,
        ),
        Precond::Amg => apply_into(
            AMG::new(a, AMGConfig::default()).map_err(precond_error)?,
            AMG::apply,
        ),
        Precond::Samg => apply_into(
            SAMG::new(a, SAMGConfig::default()).map_err(precond_error)?,
            SAMG::apply,
        ),
        Precond::Spai => apply_into(
            SPAI::new(a, SPAIConfig::default()).map_err(precond_error)?,
            SPAI::apply,
        ),
        Precond::Ainv => apply_into(
            AINV::new(a, AINVConfig::default()).map_err(precond_error)?,
            AINV::apply,
        ),
        Precond::SchwarzIlu0 => apply_into(
            AdditiveSchwarz::new(a, schwarz(LocalSolverType::ILU0)).map_err(precond_error)?,
            AdditiveSchwarz::apply,
        ),
        Precond::SchwarzExactLu => apply_into(
            AdditiveSchwarz::new(a, schwarz(LocalSolverType::ExactLU)).map_err(precond_error)?,
            AdditiveSchwarz::apply,
        ),
        Precond::SchwarzJacobi => apply_into(
            AdditiveSchwarz::new(a, schwarz(LocalSolverType::Jacobi)).map_err(precond_error)?,
            AdditiveSchwarz::apply,
        ),
        Precond::Ic0 => {
            let ic0 = IC0::new(a).map_err(chol_error)?;
            Box::new(move |r: &[f64]| ic0.apply(r))
        }
        Precond::IctDropping => {
            let ict = ICT::new(a, 0.05, 3).map_err(chol_error)?;
            Box::new(move |r: &[f64]| ict.apply(r))
        }
        Precond::IctComplete => {
            let ict = ICT::new(a, 0.0, n).map_err(chol_error)?;
            Box::new(move |r: &[f64]| ict.apply(r))
        }
    })
}

/// Building on `s*A` (`s = 2^-64`, `2^64`) must succeed because building on
/// `A` does, and `M(s*A)^{-1} r` must be exactly `M(A)^{-1} r / s`.
fn check_exactly_scaled(kind: Precond) {
    let a = variable_spd(GRID);
    let vectors = probes(a.nrows());
    let base = build(kind, &a).unwrap_or_else(|e| panic!("{kind:?}: building on A failed: {e}"));
    let reference: Vec<Vec<f64>> = vectors.iter().map(|r| base(r)).collect();
    for z in &reference {
        assert!(
            z.iter().all(|v| v.is_finite()),
            "{kind:?}: non-finite output"
        );
    }
    for s in scale_factors() {
        let scaled = build(kind, &scaled_matrix(&a, s)).unwrap_or_else(|e| {
            panic!("{kind:?}: building on A scaled by {s:e} failed ({e}) although building on A succeeded")
        });
        for (probe, (r, z)) in vectors.iter().zip(&reference).enumerate() {
            assert!(
                bits(&scaled(r)) == scaled_bits(z, 1.0 / s),
                "{kind:?}: preconditioner of A scaled by {s:e} is not exactly 1/s times that of A (probe {probe})"
            );
        }
    }
}

/// Outcome of one preconditioned solve.
#[derive(Debug)]
struct Run {
    x: Vec<f64>,
    iterations: usize,
    converged: bool,
    residual_norm: f64,
    history: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solver {
    Pcg,
    Pgmres,
}

/// Runs `solver` with the preconditioner `m` from `x0 = 0`.
fn solve(solver: Solver, a: &CsrMatrix<f64>, b: &[f64], m: &Apply) -> Result<Run, IterativeError> {
    let x0 = vec![0.0; a.nrows()];
    match solver {
        Solver::Pcg => pcg(a, b, &x0, |r: &[f64]| m(r), TOL, MAX_ITER).map(|r| Run {
            x: r.x,
            iterations: r.iterations,
            converged: r.converged,
            residual_norm: r.residual_norm,
            history: vec![],
        }),
        Solver::Pgmres => {
            pgmres(a, b, &x0, |r: &[f64]| m(r), RESTART, TOL, MAX_ITER).map(|r| Run {
                x: r.x,
                iterations: r.iterations,
                converged: r.converged,
                residual_norm: r.residual_norm,
                history: r.residual_history,
            })
        }
    }
}

/// `solver` with `kind` rebuilt from the scaled matrix has the same outcome at
/// `s = 2^-64, 2^64` as at `s = 1`: identical iteration count and convergence
/// flag, bit-identical solution, exactly scaled residual (and history), or a
/// breakdown at the same iteration. The unscaled run must converge honestly,
/// except PCG with the non-symmetric Gauss-Seidel / SOR preconditioners.
fn check_solver_scale_invariance(kind: Precond, solver: Solver) {
    let a = variable_spd(GRID);
    let b = rhs(a.nrows());

    let m = build(kind, &a).unwrap_or_else(|e| panic!("{kind:?}: building on A failed: {e}"));
    let reference = solve(solver, &a, &b, &m);
    if solver == Solver::Pgmres || kind.symmetric() {
        let run = reference
            .as_ref()
            .unwrap_or_else(|e| panic!("{kind:?} {solver:?}: unscaled solve failed: {e}"));
        assert!(
            run.converged,
            "{kind:?} {solver:?}: unscaled solve did not converge in {} iterations",
            run.iterations
        );
        let rel = true_relative_residual(&a, &b, &run.x);
        assert!(
            rel <= TRUE_RESIDUAL_BOUND,
            "{kind:?} {solver:?}: reported convergence with true relative residual {rel:e}"
        );
    }

    for s in scale_factors() {
        let a_s = scaled_matrix(&a, s);
        let b_s: Vec<f64> = b.iter().map(|v| v * s).collect();
        let m_s = build(kind, &a_s).unwrap_or_else(|e| {
            panic!("{kind:?}: building on A scaled by {s:e} failed ({e}) although building on A succeeded")
        });
        let scaled = solve(solver, &a_s, &b_s, &m_s);
        match (&reference, &scaled) {
            (Ok(reference), Ok(sol)) => {
                assert_eq!(
                    sol.iterations, reference.iterations,
                    "{kind:?} {solver:?}: iteration count changed under scaling by {s:e}"
                );
                assert_eq!(sol.converged, reference.converged);
                assert!(
                    bits(&sol.x) == bits(&reference.x),
                    "{kind:?} {solver:?}: solution is not bit-identical under scaling by {s:e}"
                );
                assert_eq!(
                    sol.residual_norm.to_bits(),
                    (reference.residual_norm * s).to_bits(),
                    "{kind:?} {solver:?}: residual norm is not exactly scaled by {s:e}"
                );
                assert!(
                    bits(&sol.history) == scaled_bits(&reference.history, s),
                    "{kind:?} {solver:?}: residual history is not exactly scaled by {s:e}"
                );
            }
            (
                Err(IterativeError::Breakdown { iteration: i, .. }),
                Err(IterativeError::Breakdown { iteration: j, .. }),
            ) => assert_eq!(
                i, j,
                "{kind:?} {solver:?}: breakdown iteration changed under scaling by {s:e}"
            ),
            (reference, sol) => panic!(
                "{kind:?} {solver:?}: outcome changed under scaling by {s:e}: {reference:?} vs {sol:?}"
            ),
        }
    }
}

macro_rules! precond_scale_tests {
    ($($exact:ident, $pcg:ident => $kind:expr),* $(,)?) => {
        $(
            #[test]
            fn $exact() {
                check_exactly_scaled($kind);
            }

            #[test]
            fn $pcg() {
                check_solver_scale_invariance($kind, Solver::Pcg);
            }
        )*
    };
}

precond_scale_tests! {
    test_jacobi_exactly_scaled, test_pcg_jacobi_scale_invariant => Precond::Jacobi,
    test_block_jacobi_exactly_scaled, test_pcg_block_jacobi_scale_invariant => Precond::BlockJacobi,
    test_gauss_seidel_exactly_scaled, test_pcg_gauss_seidel_scale_invariant => Precond::GaussSeidel,
    test_sor_exactly_scaled, test_pcg_sor_scale_invariant => Precond::Sor,
    test_ssor_exactly_scaled, test_pcg_ssor_scale_invariant => Precond::Ssor,
    test_neumann_exactly_scaled, test_pcg_neumann_scale_invariant => Precond::Neumann,
    test_chebyshev_exactly_scaled, test_pcg_chebyshev_scale_invariant => Precond::Chebyshev,
    test_amg_exactly_scaled, test_pcg_amg_scale_invariant => Precond::Amg,
    test_samg_exactly_scaled, test_pcg_samg_scale_invariant => Precond::Samg,
    test_spai_exactly_scaled, test_pcg_spai_scale_invariant => Precond::Spai,
    test_ainv_exactly_scaled, test_pcg_ainv_scale_invariant => Precond::Ainv,
    test_schwarz_ilu0_exactly_scaled, test_pcg_schwarz_ilu0_scale_invariant => Precond::SchwarzIlu0,
    test_schwarz_exact_lu_exactly_scaled, test_pcg_schwarz_exact_lu_scale_invariant => Precond::SchwarzExactLu,
    test_schwarz_jacobi_exactly_scaled, test_pcg_schwarz_jacobi_scale_invariant => Precond::SchwarzJacobi,
    test_ic0_exactly_scaled, test_pcg_ic0_scale_invariant => Precond::Ic0,
    test_ict_dropping_exactly_scaled, test_pcg_ict_dropping_scale_invariant => Precond::IctDropping,
    test_ict_complete_exactly_scaled, test_pcg_ict_complete_scale_invariant => Precond::IctComplete,
}

#[test]
fn test_pgmres_gauss_seidel_scale_invariant() {
    check_solver_scale_invariance(Precond::GaussSeidel, Solver::Pgmres);
}

#[test]
fn test_pgmres_sor_scale_invariant() {
    check_solver_scale_invariance(Precond::Sor, Solver::Pgmres);
}

// =============================================================================
// Incomplete Cholesky factors
// =============================================================================

/// `L(s*A)` must be exactly `sqrt(s) * L(A)` for `s = 4^k`, with the same
/// sparsity pattern. `2^-128` is included because the old absolute test
/// `|l_jj| <= eps` rejected every pivot of `2^-128 * A` (`l_jj ~ 2^-63`).
#[test]
fn test_incomplete_cholesky_factors_exactly_scaled() {
    let a = variable_spd(GRID);
    let n = a.nrows();
    let two_pow_64 = 18_446_744_073_709_551_616.0_f64;
    let two_pow_32 = 4_294_967_296.0_f64;
    let cases = [
        (1.0 / two_pow_64, 1.0 / two_pow_32),
        (two_pow_64, two_pow_32),
        (1.0 / (two_pow_64 * two_pow_64), 1.0 / two_pow_64),
        (two_pow_64 * two_pow_64, two_pow_64),
    ];
    type Factor = fn(&CsrMatrix<f64>) -> Result<CsrMatrix<f64>, SparseCholError>;
    let factors: [(&str, Factor); 3] = [
        ("IC0", |m| IC0::new(m).map(|f| f.l().clone())),
        ("ICT(0.05, 3)", |m| {
            ICT::new(m, 0.05, 3).map(|f| f.l().clone())
        }),
        ("ICT(0, n)", |m| {
            ICT::new(m, 0.0, m.nrows()).map(|f| f.l().clone())
        }),
    ];
    for (name, factor) in factors {
        let l = factor(&a).unwrap_or_else(|e| panic!("{name}: factorization of A failed: {e}"));
        assert_eq!(l.nrows(), n);
        for (s, sqrt_s) in cases {
            let l_s = factor(&scaled_matrix(&a, s)).unwrap_or_else(|e| {
                panic!("{name}: factorization of A scaled by {s:e} failed ({e})")
            });
            assert_eq!(
                l_s.row_ptrs(),
                l.row_ptrs(),
                "{name}: pattern changed at {s:e}"
            );
            assert_eq!(
                l_s.col_indices(),
                l.col_indices(),
                "{name}: pattern changed at {s:e}"
            );
            assert!(
                bits(l_s.values()) == scaled_bits(l.values(), sqrt_s),
                "{name}: L of A scaled by {s:e} is not exactly {sqrt_s:e} * L(A)"
            );
        }
    }
}

/// Dense `L * L^T` of a CSR lower-triangular factor.
fn llt_dense(l: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = l.nrows();
    let mut dense = vec![vec![0.0; n]; n];
    for (i, row) in dense.iter_mut().enumerate() {
        for idx in l.row_ptrs()[i]..l.row_ptrs()[i + 1] {
            row[l.col_indices()[idx]] = l.values()[idx];
        }
    }
    dense
        .iter()
        .map(|row_i| {
            dense
                .iter()
                .map(|row_j| row_i.iter().zip(row_j).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect()
}

/// ICT without dropping and without a fill cap is the complete Cholesky
/// factorization: `L * L^T = A` to rounding, with fill-in inside the band.
/// (The elimination used to read row `j` of L where it needs column `j`, so
/// every update between off-diagonal entries was skipped and no fill-in was
/// ever created.)
#[test]
fn test_ict_without_dropping_is_complete_cholesky() {
    let a = variable_spd(6);
    let n = a.nrows();
    let ict = ICT::new(&a, 0.0, n).unwrap();
    let product = llt_dense(ict.l());
    let mut worst = 0.0_f64;
    for (i, product_row) in product.iter().enumerate() {
        let mut row = vec![0.0; n];
        for idx in a.row_ptrs()[i]..a.row_ptrs()[i + 1] {
            row[a.col_indices()[idx]] = a.values()[idx];
        }
        for (p, r) in product_row.iter().zip(&row) {
            worst = worst.max((p - r).abs());
        }
    }
    assert!(worst <= 1e-13, "max |(L L^T - A)_ij| = {worst:e}");
    let lower_nnz = (0..n)
        .map(|i| {
            (a.row_ptrs()[i]..a.row_ptrs()[i + 1])
                .filter(|&idx| a.col_indices()[idx] <= i)
                .count()
        })
        .sum::<usize>();
    assert!(
        ict.nnz() > lower_nnz,
        "complete factor has no fill-in: nnz(L) = {}, nnz(tril(A)) = {lower_nnz}",
        ict.nnz()
    );
}

/// The ICT drop rule compares `|l_ij| * l_jj` with `tau * ||row_i(A)||`: with a
/// large `tau` every off-diagonal entry is dropped and `L` is the diagonal
/// `sqrt(a_ii)`, at every scale.
#[test]
fn test_ict_large_tau_drops_every_off_diagonal() {
    let a = variable_spd(GRID);
    let n = a.nrows();
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a_s = scaled_matrix(&a, s);
        let ict = ICT::new(&a_s, 1.0, n).unwrap();
        assert_eq!(ict.nnz(), n, "scale {s:e}");
        for i in 0..n {
            let diag = (a_s.row_ptrs()[i]..a_s.row_ptrs()[i + 1])
                .find(|&idx| a_s.col_indices()[idx] == i)
                .map(|idx| a_s.values()[idx])
                .unwrap();
            assert_eq!(ict.l().values()[i].to_bits(), diag.sqrt().to_bits());
        }
    }
}

/// IC0 and ICT apply the dense Cholesky rule: a radicand at or below
/// `eps * n * a_ii` is not positive definite, at every scale. Here the second
/// radicand is `2^-52 <= eps * 2 * (1 + 2^-52)`; it used to be accepted
/// because only `l_jj <= eps` was tested, and only when a later row used it.
#[test]
fn test_incomplete_cholesky_relative_radicand_rule() {
    let tiny = f64::EPSILON;
    let a = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0 + tiny],
    )
    .unwrap();
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a_s = scaled_matrix(&a, s);
        assert_eq!(
            IC0::new(&a_s).unwrap_err(),
            SparseCholError::NotPositiveDefinite { index: 1 },
            "IC0 at scale {s:e}"
        );
        assert_eq!(
            ICT::new(&a_s, 0.0, 2).unwrap_err(),
            SparseCholError::NotPositiveDefinite { index: 1 },
            "ICT at scale {s:e}"
        );
    }
    // A clearly positive radicand (2^-40 > eps * 2 * a_ii) still factors.
    let b = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0 + 2.0_f64.powi(-40)],
    )
    .unwrap();
    assert!(IC0::new(&b).is_ok());
    assert!(ICT::new(&b, 0.0, 2).is_ok());
}

/// ICT reports a diagonal entry that is not stored, like IC0.
#[test]
fn test_ict_missing_diagonal_reported() {
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]).unwrap();
    assert_eq!(
        ICT::new(&a, 0.0, 2).unwrap_err(),
        SparseCholError::ZeroDiagonal { index: 1 }
    );
    assert_eq!(
        IC0::new(&a).unwrap_err(),
        SparseCholError::ZeroDiagonal { index: 1 }
    );
}

// =============================================================================
// Zero and singular cases keep being reported at every scale
// =============================================================================

#[test]
fn test_exact_zero_diagonal_rejected_at_every_scale() {
    // [[1, 1], [1, 0]]: the second diagonal entry is stored and exactly zero.
    let a = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a_s = scaled_matrix(&a, s);
        let zero = PreconditionerError::ZeroDiagonal(1);
        assert_eq!(Jacobi::new(&a_s).unwrap_err(), zero, "Jacobi at {s:e}");
        assert_eq!(
            GaussSeidel::new(&a_s).err(),
            Some(zero.clone()),
            "GS at {s:e}"
        );
        assert_eq!(SOR::new(&a_s, 1.5).unwrap_err(), zero, "SOR at {s:e}");
        assert_eq!(SSOR::new(&a_s, 1.5).unwrap_err(), zero, "SSOR at {s:e}");
        assert_eq!(
            Polynomial::new(&a_s, PolynomialType::Neumann, PolynomialConfig::default())
                .unwrap_err(),
            zero,
            "Polynomial at {s:e}"
        );
    }
}

#[test]
fn test_tiny_nonzero_diagonal_accepted() {
    // A diagonal entry of 2^-80 is not zero: Jacobi inverts it exactly.
    let tiny = 2.0_f64.powi(-80);
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![tiny, 1.0]).unwrap();
    let jacobi = Jacobi::new(&a).unwrap();
    let mut z = vec![0.0; 2];
    jacobi.apply(&[1.0, 1.0], &mut z);
    assert_eq!(z, vec![2.0_f64.powi(80), 1.0]);
}

#[test]
fn test_singular_blocks_reported_at_every_scale() {
    // Rank-one 2x2 block [[1, 2], [2, 4]] (exactly singular) next to a
    // regular 1x1 block.
    let a = CsrMatrix::new(
        3,
        3,
        vec![0, 2, 4, 5],
        vec![0, 1, 0, 1, 2],
        vec![1.0, 2.0, 2.0, 4.0, 3.0],
    )
    .unwrap();
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a_s = scaled_matrix(&a, s);
        assert_eq!(
            BlockJacobi::new(&a_s, &[2, 1]).unwrap_err(),
            PreconditionerError::SingularBlock(1),
            "BlockJacobi at {s:e}"
        );
        let config = AdditiveSchwarzConfig {
            num_subdomains: 1,
            overlap: 0,
            local_solver: LocalSolverType::ExactLU,
        };
        assert_eq!(
            AdditiveSchwarz::new(&a_s, config).unwrap_err(),
            PreconditionerError::SingularBlock(1),
            "Schwarz ExactLU at {s:e}"
        );
    }
    // Numerically singular: [[1, 1], [1, 1 + 2^-52]] (the second pivot 2^-52
    // is at or below eps * 2 * 1): rejected at every scale, as with the
    // crate's dense LU; [[1, 1], [1, 1 + 2^-20]] is accepted.
    for (perturbation, singular) in [(f64::EPSILON, true), (2.0_f64.powi(-20), false)] {
        let b = CsrMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 1.0, 1.0, 1.0 + perturbation],
        )
        .unwrap();
        for s in [1.0, scale_factors()[0], scale_factors()[1]] {
            let result = BlockJacobi::new(&scaled_matrix(&b, s), &[2]);
            assert_eq!(
                result.is_err(),
                singular,
                "perturbation {perturbation:e} at {s:e}"
            );
        }
    }
}

/// AINV on a matrix with a zero pivot: the first pivot of
/// `[[0, 1], [1, 0]]` is `e_0^T A e_0 = 0`; it is regularized (not reported),
/// at every scale, and the result stays finite and exactly scaled.
#[test]
fn test_ainv_zero_pivot_regularized_scale_invariantly() {
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap();
    let config = || AINVConfig {
        drop_tolerance: 0.0,
        max_nnz_per_col: 2,
        modified_gs: true,
    };
    let base = AINV::new(&a, config()).unwrap();
    let mut z = vec![0.0_f64; 2];
    base.apply(&[1.0, -2.0], &mut z);
    assert!(z.iter().all(|v| v.is_finite()), "{z:?}");
    for s in scale_factors() {
        let scaled = AINV::new(&scaled_matrix(&a, s), config()).unwrap();
        let mut z_s = vec![0.0; 2];
        scaled.apply(&[1.0, -2.0], &mut z_s);
        assert!(
            bits(&z_s) == scaled_bits(&z, 1.0 / s),
            "scale {s:e}: {z_s:?} vs {z:?}"
        );
    }
}
