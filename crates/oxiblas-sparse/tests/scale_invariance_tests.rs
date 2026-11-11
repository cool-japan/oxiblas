//! Scale-invariance and breakdown-semantics regression tests for the iterative
//! solvers in `oxiblas_sparse::linalg::iterative`.
//!
//! Krylov iterates are invariant under the rescaling `(A, b) -> (s*A, s*b)`:
//! the solution and every iterate stay the same while every residual scales by
//! `s`. When `s` is a power of two, IEEE-754 arithmetic reproduces this
//! exactly (absent overflow and underflow), because every product, quotient,
//! sum and square root of consistently scaled operands is the exactly scaled
//! result. A solver that still compares a scaled quantity against a bare
//! `epsilon()`, or mixes in any other absolute constant, loses the property:
//! it reports spurious breakdowns (or false convergence) on tiny problems and
//! misses real breakdowns on huge ones. The tests below therefore demand an
//! identical iteration count, a bit-identical solution and exactly scaled
//! residuals at `s = 2^-64` and `s = 2^64`, and again at `s = 2^-520` and
//! `s = 2^520`, where every unscaled sum of squares (and every product of a
//! residual with `A`) over- or underflows.

use oxiblas_sparse::CsrMatrix;
use oxiblas_sparse::linalg::cholesky::IC0;
use oxiblas_sparse::linalg::iterative::{
    IterativeError, bicgstab, block_cg, block_gmres, block_pcg, cg, fgmres, gmres, idrs, minres,
    pcg, pgmres, pidrs, pminres, pqmr, ptfqmr, qmr, tfqmr,
};
use oxiblas_sparse::ops::spmv;

const TOL: f64 = 1e-10;
const MAX_ITER: usize = 1000;
const RESTART: usize = 20;
const IDR_S: usize = 4;
const GRID: usize = 10;
/// Bound on the true relative residual of a run reported as converged
/// (the recursive / quasi residual is what meets `TOL`).
const TRUE_RESIDUAL_BOUND: f64 = 1e-8;

/// `[2^-64, 2^64]`, both exactly representable.
fn scale_factors() -> [f64; 2] {
    let two_pow_64 = 18_446_744_073_709_551_616.0_f64;
    [1.0 / two_pow_64, two_pow_64]
}

/// `[2^-520, 2^520]`: `‖b‖²` of the scaled right-hand side over- or
/// underflows, and so do `A·r` and `pᵀAp` of an iteration run in the units
/// of the input.
fn extreme_scale_factors() -> [f64; 2] {
    [2.0_f64.powi(-520), 2.0_f64.powi(520)]
}

// =============================================================================
// Problem generators
// =============================================================================

/// 5-point operator on a `grid x grid` mesh with a row-dependent diagonal and
/// the given `[south, west, east, north]` couplings.
fn five_point(grid: usize, diag: impl Fn(usize) -> f64, couplings: [f64; 4]) -> CsrMatrix<f64> {
    let [south, west, east, north] = couplings;
    let n = grid * grid;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptrs = vec![0];
    for i in 0..n {
        let row = i / grid;
        let col = i % grid;
        if row > 0 {
            values.push(south);
            col_indices.push(i - grid);
        }
        if col > 0 {
            values.push(west);
            col_indices.push(i - 1);
        }
        values.push(diag(i));
        col_indices.push(i);
        if col + 1 < grid {
            values.push(east);
            col_indices.push(i + 1);
        }
        if row + 1 < grid {
            values.push(north);
            col_indices.push(i + grid);
        }
        row_ptrs.push(values.len());
    }
    CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
}

/// SPD 5-point operator whose diagonal varies by row, so that Jacobi and IC(0)
/// are not mere multiples of the identity.
fn variable_spd(grid: usize) -> CsrMatrix<f64> {
    five_point(
        grid,
        |i| 4.0 + 0.25 * (i % 7) as f64,
        [-1.0, -1.0, -1.0, -1.0],
    )
}

/// Non-symmetric convection-diffusion-like 5-point operator.
fn variable_nonsymmetric(grid: usize) -> CsrMatrix<f64> {
    five_point(
        grid,
        |i| 4.0 + 0.25 * (i % 5) as f64,
        [-1.1, -1.25, -0.75, -0.9],
    )
}

/// Right-hand side with entries `1 + k/8` (exact in binary).
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + (i % 7) as f64 / 8.0).collect()
}

/// Two right-hand sides for the block solvers: `b` and `b` reversed.
fn block_rhs(b: &[f64]) -> Vec<Vec<f64>> {
    vec![b.to_vec(), b.iter().rev().copied().collect()]
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

fn diagonal(a: &CsrMatrix<f64>) -> Vec<f64> {
    (0..a.nrows())
        .map(|i| {
            (a.row_ptrs()[i]..a.row_ptrs()[i + 1])
                .find(|&k| a.col_indices()[k] == i)
                .map(|k| a.values()[k])
                .unwrap()
        })
        .collect()
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

// =============================================================================
// Uniform view of every solver
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Cg,
    PcgJacobi,
    PcgIc0,
    BlockCg,
    BlockPcg,
    Bicgstab,
    Gmres,
    Pgmres,
    Fgmres,
    Minres,
    Pminres,
    Tfqmr,
    Ptfqmr,
    Idrs,
    Pidrs,
    Qmr,
    Pqmr,
    BlockGmres,
    /// PCG with Jacobi on the negated SPD matrix: `A` and `M` are both
    /// negative definite, so `rᵀz` and `pᵀAp` stay negative throughout.
    PcgNegDef,
    /// Block PCG with Jacobi on the negated SPD matrix.
    BlockPcgNegDef,
}

impl Kind {
    fn symmetric(self) -> bool {
        matches!(
            self,
            Kind::Cg
                | Kind::PcgJacobi
                | Kind::PcgIc0
                | Kind::BlockCg
                | Kind::BlockPcg
                | Kind::Minres
                | Kind::Pminres
                | Kind::PcgNegDef
                | Kind::BlockPcgNegDef
        )
    }

    fn block(self) -> bool {
        matches!(
            self,
            Kind::BlockCg | Kind::BlockPcg | Kind::BlockGmres | Kind::BlockPcgNegDef
        )
    }

    /// The unscaled matrix of the test problem.
    fn matrix(self) -> CsrMatrix<f64> {
        match self {
            Kind::PcgNegDef | Kind::BlockPcgNegDef => scaled_matrix(&variable_spd(GRID), -1.0),
            kind if kind.symmetric() => variable_spd(GRID),
            _ => variable_nonsymmetric(GRID),
        }
    }

    /// Factor by which the residual history scales when `(A, b)` is scaled by
    /// `s`. PQMR records the left-preconditioned quasi-residual
    /// `||M1^{-1} r||`, which is invariant when the Jacobi `M1` is rebuilt from
    /// the scaled matrix; every other history (and PQMR's with a fixed `M1`) is
    /// a residual (estimate) that scales by `s`.
    fn history_scale(self, mode: PrecondMode, s: f64) -> f64 {
        if self == Kind::Pqmr && mode == PrecondMode::Rebuilt {
            1.0
        } else {
            s
        }
    }
}

/// Where the preconditioners of a scaled run come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrecondMode {
    /// Rebuilt from the scaled matrix (the realistic usage). For Jacobi this
    /// makes `A M^{-1}` and `M^{-1} A` exactly invariant.
    Rebuilt,
    /// Built once from the unscaled matrix and reused at every scale, so every
    /// Krylov quantity of the preconditioned solvers scales with `s` and any
    /// surviving absolute threshold is exercised.
    Fixed,
}

/// Outcome of one solve, normalised across the result types.
#[derive(Debug)]
enum Outcome {
    Solved(Solution),
    Failed(IterativeError),
}

#[derive(Debug)]
struct Solution {
    x: Vec<Vec<f64>>,
    iterations: usize,
    converged: bool,
    residual: Vec<f64>,
    history: Vec<f64>,
}

fn solved(
    x: Vec<Vec<f64>>,
    iterations: usize,
    converged: bool,
    residual: Vec<f64>,
    history: Vec<f64>,
) -> Outcome {
    Outcome::Solved(Solution {
        x,
        iterations,
        converged,
        residual,
        history,
    })
}

/// Run `kind` on `A x = b` (block solvers: on `block_rhs(b)`) from `x0 = 0`.
/// Jacobi and IC(0) preconditioners are built from `precond_source`.
fn solve(kind: Kind, a: &CsrMatrix<f64>, b: &[f64], precond_source: &CsrMatrix<f64>) -> Outcome {
    let n = a.nrows();
    let x0 = vec![0.0; n];
    let d = diagonal(precond_source);
    let jacobi = |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };
    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let bb = block_rhs(b);
    let xx = vec![x0.clone(), x0.clone()];

    let outcome = match kind {
        Kind::Cg => cg(a, b, &x0, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                vec![],
            )
        }),
        Kind::PcgJacobi | Kind::PcgNegDef => pcg(a, b, &x0, jacobi, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                vec![],
            )
        }),
        Kind::PcgIc0 => {
            let ic0 = IC0::new(precond_source).unwrap();
            pcg(a, b, &x0, |r: &[f64]| ic0.apply(r), TOL, MAX_ITER).map(|r| {
                solved(
                    vec![r.x],
                    r.iterations,
                    r.converged,
                    vec![r.residual_norm],
                    vec![],
                )
            })
        }
        Kind::BlockCg => block_cg(a, &bb, &xx, TOL, MAX_ITER)
            .map(|r| solved(r.x, r.iterations, r.converged, r.residual_norms, vec![])),
        Kind::BlockPcg | Kind::BlockPcgNegDef => block_pcg(a, &bb, &xx, jacobi, TOL, MAX_ITER)
            .map(|r| solved(r.x, r.iterations, r.converged, r.residual_norms, vec![])),
        Kind::Bicgstab => bicgstab(a, b, &x0, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                vec![],
            )
        }),
        Kind::Gmres => gmres(a, b, &x0, RESTART, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Pgmres => pgmres(a, b, &x0, jacobi, RESTART, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Fgmres => {
            let mut precond =
                |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };
            fgmres(a, b, &x0, &mut precond, RESTART, TOL, MAX_ITER).map(|r| {
                solved(
                    vec![r.x],
                    r.iterations,
                    r.converged,
                    vec![r.residual_norm],
                    r.residual_history,
                )
            })
        }
        Kind::Minres => minres(a, b, &x0, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Pminres => pminres(a, b, &x0, jacobi, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Tfqmr => tfqmr(a, b, &x0, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Ptfqmr => ptfqmr(a, b, &x0, jacobi, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Idrs => idrs(a, b, &x0, IDR_S, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Pidrs => pidrs(a, b, &x0, &jacobi, IDR_S, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Qmr => qmr(a, b, &x0, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::Pqmr => pqmr(a, b, &x0, jacobi, jacobi, identity, identity, TOL, MAX_ITER).map(|r| {
            solved(
                vec![r.x],
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
        Kind::BlockGmres => block_gmres(a, &bb, &xx, RESTART, TOL, MAX_ITER).map(|r| {
            solved(
                r.x,
                r.iterations,
                r.converged,
                vec![r.residual_norm],
                r.residual_history,
            )
        }),
    };
    outcome.unwrap_or_else(Outcome::Failed)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn scaled_bits(values: &[f64], s: f64) -> Vec<u64> {
    values.iter().map(|v| (v * s).to_bits()).collect()
}

/// The unscaled run must converge honestly (true residual small) and the runs
/// at `s = 2^-64` and `s = 2^64` must reproduce it exactly: same iteration
/// count, bit-identical solution, residuals and residual history scaled
/// exactly. `mode` selects whether the scaled runs rebuild their
/// preconditioners from the scaled matrix or reuse the unscaled ones.
fn check_scale_invariance(kind: Kind, mode: PrecondMode) {
    check_scale_invariance_at(kind, mode, &scale_factors());
}

/// [`check_scale_invariance`] at the given scales.
fn check_scale_invariance_at(kind: Kind, mode: PrecondMode, scales: &[f64]) {
    let a = kind.matrix();
    let b = rhs(a.nrows());

    let reference = match solve(kind, &a, &b, &a) {
        Outcome::Solved(sol) => sol,
        Outcome::Failed(e) => panic!("{kind:?}: unscaled run failed: {e}"),
    };
    assert!(
        reference.converged,
        "{kind:?}: unscaled run did not converge in {} iterations",
        reference.iterations
    );
    let columns = if kind.block() {
        block_rhs(&b)
    } else {
        vec![b.clone()]
    };
    assert_eq!(reference.x.len(), columns.len());
    for (col, (x, bc)) in reference.x.iter().zip(&columns).enumerate() {
        let rel = true_relative_residual(&a, bc, x);
        assert!(
            rel <= TRUE_RESIDUAL_BOUND,
            "{kind:?}: column {col} reported converged but true relative residual is {rel:e}"
        );
    }

    for &s in scales {
        let a_s = scaled_matrix(&a, s);
        let b_s: Vec<f64> = b.iter().map(|v| v * s).collect();
        let precond_source = match mode {
            PrecondMode::Rebuilt => &a_s,
            PrecondMode::Fixed => &a,
        };
        let sol = match solve(kind, &a_s, &b_s, precond_source) {
            Outcome::Solved(sol) => sol,
            Outcome::Failed(e) => {
                panic!(
                    "{kind:?} ({mode:?}): run scaled by {s:e} failed ({e}) although the unscaled run converged"
                )
            }
        };
        assert_eq!(
            sol.iterations, reference.iterations,
            "{kind:?}: iteration count changed under scaling by {s:e}"
        );
        assert_eq!(
            sol.converged, reference.converged,
            "{kind:?}: convergence flag changed under scaling by {s:e}"
        );
        assert_eq!(sol.x.len(), reference.x.len());
        for (col, (xs, xr)) in sol.x.iter().zip(&reference.x).enumerate() {
            assert!(
                bits(xs) == bits(xr),
                "{kind:?}: solution column {col} is not bit-identical under scaling by {s:e}"
            );
        }
        assert!(
            bits(&sol.residual) == scaled_bits(&reference.residual, s),
            "{kind:?}: residual norms are not exactly scaled by {s:e}: {:?} vs {:?}",
            sol.residual,
            reference.residual
        );
        let history_scale = kind.history_scale(mode, s);
        assert!(
            bits(&sol.history) == scaled_bits(&reference.history, history_scale),
            "{kind:?}: residual history is not exactly scaled by {history_scale:e}"
        );
    }
}

macro_rules! scale_invariance_tests {
    ($mode:expr; $($name:ident => $kind:expr),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_scale_invariance($kind, $mode);
            }
        )*
    };
}

scale_invariance_tests! {
    PrecondMode::Rebuilt;
    test_scale_invariance_cg => Kind::Cg,
    test_scale_invariance_pcg_jacobi => Kind::PcgJacobi,
    test_scale_invariance_pcg_ic0 => Kind::PcgIc0,
    test_scale_invariance_block_cg => Kind::BlockCg,
    test_scale_invariance_block_pcg => Kind::BlockPcg,
    test_scale_invariance_bicgstab => Kind::Bicgstab,
    test_scale_invariance_gmres => Kind::Gmres,
    test_scale_invariance_pgmres => Kind::Pgmres,
    test_scale_invariance_fgmres => Kind::Fgmres,
    test_scale_invariance_minres => Kind::Minres,
    test_scale_invariance_pminres => Kind::Pminres,
    test_scale_invariance_tfqmr => Kind::Tfqmr,
    test_scale_invariance_ptfqmr => Kind::Ptfqmr,
    test_scale_invariance_idrs => Kind::Idrs,
    test_scale_invariance_pidrs => Kind::Pidrs,
    test_scale_invariance_qmr => Kind::Qmr,
    test_scale_invariance_pqmr => Kind::Pqmr,
    test_scale_invariance_block_gmres => Kind::BlockGmres,
    test_scale_invariance_pcg_negative_definite => Kind::PcgNegDef,
    test_scale_invariance_block_pcg_negative_definite => Kind::BlockPcgNegDef,
}

// The preconditioned solvers again, with the preconditioner built once from the
// unscaled matrix: then `A M^{-1}` and `M^{-1} A` scale with `s` as well.
scale_invariance_tests! {
    PrecondMode::Fixed;
    test_scale_invariance_fixed_precond_pcg_jacobi => Kind::PcgJacobi,
    test_scale_invariance_fixed_precond_pcg_ic0 => Kind::PcgIc0,
    test_scale_invariance_fixed_precond_block_pcg => Kind::BlockPcg,
    test_scale_invariance_fixed_precond_pgmres => Kind::Pgmres,
    test_scale_invariance_fixed_precond_fgmres => Kind::Fgmres,
    test_scale_invariance_fixed_precond_pminres => Kind::Pminres,
    test_scale_invariance_fixed_precond_ptfqmr => Kind::Ptfqmr,
    test_scale_invariance_fixed_precond_pidrs => Kind::Pidrs,
    test_scale_invariance_fixed_precond_pqmr => Kind::Pqmr,
    test_scale_invariance_fixed_precond_pcg_negative_definite => Kind::PcgNegDef,
    test_scale_invariance_fixed_precond_block_pcg_negative_definite => Kind::BlockPcgNegDef,
}

// The same problems at `2^-520` and `2^520`: `‖b‖²` of the scaled right-hand
// side over- or underflows, and so would `A·r`, `pᵀAp` and the Givens and
// Lanczos coefficients of an iteration run in the units of the input. The
// runs must still reproduce the unscaled one exactly.
macro_rules! extreme_scale_invariance_tests {
    ($mode:expr; $($name:ident => $kind:expr),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_scale_invariance_at($kind, $mode, &extreme_scale_factors());
            }
        )*
    };
}

extreme_scale_invariance_tests! {
    PrecondMode::Rebuilt;
    test_extreme_scale_invariance_cg => Kind::Cg,
    test_extreme_scale_invariance_pcg_jacobi => Kind::PcgJacobi,
    test_extreme_scale_invariance_pcg_ic0 => Kind::PcgIc0,
    test_extreme_scale_invariance_block_cg => Kind::BlockCg,
    test_extreme_scale_invariance_block_pcg => Kind::BlockPcg,
    test_extreme_scale_invariance_bicgstab => Kind::Bicgstab,
    test_extreme_scale_invariance_gmres => Kind::Gmres,
    test_extreme_scale_invariance_pgmres => Kind::Pgmres,
    test_extreme_scale_invariance_fgmres => Kind::Fgmres,
    test_extreme_scale_invariance_minres => Kind::Minres,
    test_extreme_scale_invariance_pminres => Kind::Pminres,
    test_extreme_scale_invariance_tfqmr => Kind::Tfqmr,
    test_extreme_scale_invariance_ptfqmr => Kind::Ptfqmr,
    test_extreme_scale_invariance_idrs => Kind::Idrs,
    test_extreme_scale_invariance_pidrs => Kind::Pidrs,
    test_extreme_scale_invariance_qmr => Kind::Qmr,
    test_extreme_scale_invariance_pqmr => Kind::Pqmr,
    test_extreme_scale_invariance_block_gmres => Kind::BlockGmres,
    test_extreme_scale_invariance_pcg_negative_definite => Kind::PcgNegDef,
    test_extreme_scale_invariance_block_pcg_negative_definite => Kind::BlockPcgNegDef,
}

extreme_scale_invariance_tests! {
    PrecondMode::Fixed;
    test_extreme_scale_invariance_fixed_precond_pcg_jacobi => Kind::PcgJacobi,
    test_extreme_scale_invariance_fixed_precond_pcg_ic0 => Kind::PcgIc0,
    test_extreme_scale_invariance_fixed_precond_block_pcg => Kind::BlockPcg,
    test_extreme_scale_invariance_fixed_precond_pgmres => Kind::Pgmres,
    test_extreme_scale_invariance_fixed_precond_fgmres => Kind::Fgmres,
    test_extreme_scale_invariance_fixed_precond_pminres => Kind::Pminres,
    test_extreme_scale_invariance_fixed_precond_ptfqmr => Kind::Ptfqmr,
    test_extreme_scale_invariance_fixed_precond_pidrs => Kind::Pidrs,
    test_extreme_scale_invariance_fixed_precond_pqmr => Kind::Pqmr,
    test_extreme_scale_invariance_fixed_precond_pcg_negative_definite => Kind::PcgNegDef,
    test_extreme_scale_invariance_fixed_precond_block_pcg_negative_definite => Kind::BlockPcgNegDef,
}

/// 2-norm that neither overflows nor underflows (`max · ‖v / max‖`).
fn safe_norm2(v: &[f64]) -> f64 {
    let max = v.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    if max == 0.0 {
        return 0.0;
    }
    max * v.iter().map(|e| (e / max) * (e / max)).sum::<f64>().sqrt()
}

/// A right-hand side of size `1e160` with `A` unscaled: `‖b‖²` overflowed,
/// which made `tol·‖b‖` infinite, so every solver reported convergence at
/// once with `x = x0`. Every solver must converge honestly instead.
#[test]
fn test_huge_right_hand_side_is_not_falsely_converged() {
    let kinds = [
        Kind::Cg,
        Kind::PcgJacobi,
        Kind::PcgIc0,
        Kind::BlockCg,
        Kind::BlockPcg,
        Kind::Bicgstab,
        Kind::Gmres,
        Kind::Pgmres,
        Kind::Fgmres,
        Kind::Minres,
        Kind::Pminres,
        Kind::Tfqmr,
        Kind::Ptfqmr,
        Kind::Idrs,
        Kind::Pidrs,
        Kind::Qmr,
        Kind::Pqmr,
        Kind::BlockGmres,
        Kind::PcgNegDef,
        Kind::BlockPcgNegDef,
    ];
    for kind in kinds {
        let a = kind.matrix();
        let b: Vec<f64> = rhs(a.nrows()).iter().map(|v| v * 1e160).collect();
        let sol = match solve(kind, &a, &b, &a) {
            Outcome::Solved(sol) => sol,
            Outcome::Failed(e) => panic!("{kind:?}: failed on b ~ 1e160: {e}"),
        };
        assert!(sol.converged, "{kind:?}: not converged on b ~ 1e160");
        let columns = if kind.block() {
            block_rhs(&b)
        } else {
            vec![b.clone()]
        };
        for (col, (x, bc)) in sol.x.iter().zip(&columns).enumerate() {
            let mut ax = vec![0.0; bc.len()];
            spmv(1.0, &a, x, 0.0, &mut ax);
            let r: Vec<f64> = bc.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
            let rel = safe_norm2(&r) / safe_norm2(bc);
            assert!(
                rel <= TRUE_RESIDUAL_BOUND,
                "{kind:?}: column {col} reported converged with true relative residual {rel:e}"
            );
        }
        for residual in &sol.residual {
            assert!(
                residual.is_finite() && *residual > 0.0,
                "{kind:?}: reported residual norm {residual:e}"
            );
        }
    }
}

// =============================================================================
// Breakdown semantics
// =============================================================================

/// `A = diag(1, -1)`: with `b = (1, 1)` and `x0 = 0` the first CG step has
/// `p^T A p = 1 - 1 = 0` exactly, a genuine breakdown with `||b - A x|| = sqrt(2)`.
fn indefinite_diag() -> CsrMatrix<f64> {
    CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, -1.0]).unwrap()
}

#[test]
fn test_breakdown_cg_indefinite_reports_error() {
    let a = indefinite_diag();
    let result = cg(&a, &[1.0, 1.0], &[0.0, 0.0], TOL, 100);
    assert!(
        matches!(result, Err(IterativeError::Breakdown { iteration: 0, .. })),
        "CG must report the breakdown honestly"
    );
}

#[test]
fn test_breakdown_block_cg_not_reported_as_converged() {
    let a = indefinite_diag();
    let b = vec![vec![1.0, 1.0]];
    let x0 = vec![vec![0.0, 0.0]];
    let result = block_cg(&a, &b, &x0, TOL, 100).unwrap();
    assert!(
        !result.converged,
        "a broken-down column must not be reported as converged"
    );
    assert_eq!(result.num_converged, 0);
    let honest = 2.0_f64.sqrt();
    assert!(
        (result.residual_norms[0] - honest).abs() <= 1e-12,
        "residual norm should be the true residual sqrt(2), got {}",
        result.residual_norms[0]
    );
}

#[test]
fn test_breakdown_block_pcg_not_reported_as_converged() {
    let a = indefinite_diag();
    let b = vec![vec![1.0, 1.0]];
    let x0 = vec![vec![0.0, 0.0]];
    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let result = block_pcg(&a, &b, &x0, identity, TOL, 100).unwrap();
    assert!(
        !result.converged,
        "a broken-down column must not be reported as converged"
    );
    assert_eq!(result.num_converged, 0);
}

#[test]
fn test_breakdown_block_cg_column_does_not_mask_converged_column() {
    // Column 0 breaks down at the first step; column 1 (b = e_1) converges in
    // one step. The block result must report exactly one converged system.
    let a = indefinite_diag();
    let b = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
    let x0 = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let result = block_cg(&a, &b, &x0, TOL, 100).unwrap();
    assert!(!result.converged);
    assert_eq!(result.num_converged, 1);
    assert!((result.x[1][0] - 1.0).abs() <= 1e-15 && result.x[1][1].abs() <= 1e-15);
    assert!(result.residual_norms[1] <= TOL);
}

// -----------------------------------------------------------------------------
// PCG with a preconditioner that is not positive definite: r^T z <= 0
// -----------------------------------------------------------------------------

/// Diagonal matrix with the given (exactly representable) entries.
fn diag_matrix(d: &[f64]) -> CsrMatrix<f64> {
    let n = d.len();
    CsrMatrix::new(n, n, (0..=n).collect(), (0..n).collect(), d.to_vec()).unwrap()
}

/// `M^{-1} = diag(signs)` applied to `r`.
fn signed_identity(signs: &'static [f64]) -> impl Fn(&[f64]) -> Vec<f64> {
    move |r: &[f64]| r.iter().zip(signs).map(|(ri, si)| ri * si).collect()
}

#[test]
fn test_pcg_indefinite_preconditioner_zero_rz_reports_breakdown() {
    // A = diag(1, 2) is SPD; M^{-1} = diag(1, -1) is indefinite, and with
    // b = (1, 1), x0 = 0 the first r^T z = 1 - 1 = 0. PCG used to divide by it
    // (beta = 0/0) and return NaN; it must report the breakdown, at every scale.
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a = scaled_matrix(&diag_matrix(&[1.0, 2.0]), s);
        let b = [s, s];
        let result = pcg(&a, &b, &[0.0, 0.0], signed_identity(&[1.0, -1.0]), TOL, 100);
        assert!(
            matches!(result, Err(IterativeError::Breakdown { iteration: 0, .. })),
            "scale {s:e}: expected a breakdown at iteration 0, got {:?}",
            result.map(|r| (r.converged, r.residual_norm, r.x))
        );
    }
}

#[test]
fn test_pcg_preconditioner_turning_indefinite_reports_breakdown() {
    // A = diag(1, 2, 3), M^{-1} = diag(1, 1, -1), b = (1, 1, 1): r0^T z0 = 1 > 0,
    // but after the first step r = (5/6, 2/3, 3/2) and r^T z = -10/9 < 0. PCG
    // must stop there with the true-residual verdict (not converged: an error),
    // at every scale.
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a = scaled_matrix(&diag_matrix(&[1.0, 2.0, 3.0]), s);
        let b = [s, s, s];
        let result = pcg(
            &a,
            &b,
            &[0.0; 3],
            signed_identity(&[1.0, 1.0, -1.0]),
            TOL,
            100,
        );
        assert!(
            matches!(result, Err(IterativeError::Breakdown { iteration: 1, .. })),
            "scale {s:e}: expected a breakdown at iteration 1, got {:?}",
            result.map(|r| (r.converged, r.iterations, r.residual_norm))
        );
    }
}

#[test]
fn test_pcg_nan_preconditioner_output_reports_breakdown() {
    // The preconditioner returns NaN from its second application on. r^T z is
    // then NaN, which must be a breakdown rather than NaN iterates.
    let a = diag_matrix(&[1.0, 2.0]);
    let calls = std::cell::Cell::new(0usize);
    let flaky = |r: &[f64]| -> Vec<f64> {
        calls.set(calls.get() + 1);
        if calls.get() == 1 {
            r.to_vec()
        } else {
            vec![f64::NAN; r.len()]
        }
    };
    let result = pcg(&a, &[1.0, 1.0], &[0.0, 0.0], flaky, TOL, 100);
    assert!(
        matches!(result, Err(IterativeError::Breakdown { iteration: 1, .. })),
        "expected a breakdown at iteration 1, got {:?}",
        result.map(|r| (r.converged, r.residual_norm, r.x))
    );
}

#[test]
fn test_pcg_converged_start_precedes_rz_test() {
    // M^{-1} = 0 makes r^T z = 0, but x0 is the exact solution: the
    // convergence test runs first, so r = 0 is reported as converged and the
    // r^T z test never fires.
    let a = diag_matrix(&[1.0, 2.0]);
    let zero = |r: &[f64]| -> Vec<f64> { vec![0.0; r.len()] };
    let result = pcg(&a, &[1.0, 2.0], &[1.0, 1.0], zero, TOL, 100).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 0.0);
}

#[test]
fn test_block_pcg_indefinite_preconditioner_settles_honestly() {
    // System 0 (b = (1, 1)) has r^T z = 0 at once with M^{-1} = diag(1, -1);
    // system 1 (b = (1, 0)) converges in one step. The broken-down system must
    // be reported as not converged with its finite true residual sqrt(2), not
    // as NaN, and must not mask the converged one.
    let a = diag_matrix(&[1.0, 2.0]);
    let b = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
    let x0 = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let result = block_pcg(&a, &b, &x0, signed_identity(&[1.0, -1.0]), TOL, 100).unwrap();
    assert!(!result.converged);
    assert_eq!(result.num_converged, 1);
    assert_eq!(result.x[0], vec![0.0, 0.0]);
    assert_eq!(
        result.residual_norms[0].to_bits(),
        2.0_f64.sqrt().to_bits(),
        "residual norm should be the true residual sqrt(2), got {}",
        result.residual_norms[0]
    );
    assert_eq!(result.x[1], vec![1.0, 0.0]);
    assert_eq!(result.residual_norms[1], 0.0);
    for x in &result.x {
        assert!(x.iter().all(|v| v.is_finite()));
    }
}

// -----------------------------------------------------------------------------
// PCG with consistently signed (definite but not positive) A and M
// -----------------------------------------------------------------------------

/// Jacobi with the diagonal `d`, multiplied by `sign`.
fn signed_jacobi(d: &[f64], sign: f64) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
    move |r: &[f64]| r.iter().zip(d).map(|(ri, di)| sign * ri / di).collect()
}

/// `(σ_A·A, σ_A·b)` with `σ_M·M⁻¹` has the solution of `(A, b)` with `M⁻¹`,
/// and PCG produces the same iterates: every `rᵀz` is multiplied by `σ_M`
/// (times `σ_A²`) and every `pᵀAp` by `σ_A`, and both signs cancel in `α` and
/// `β`. CG converges whenever `rᵀz` and `pᵀAp` keep their signs, so only a
/// sign that changes during the iteration (or a negligible value) is a
/// breakdown; the rule used to demand `rᵀz > 0` and broke down at once.
#[test]
fn test_pcg_consistent_sign_flips_reproduce_the_spd_iterates() {
    let a = variable_spd(GRID);
    let n = a.nrows();
    let b = rhs(n);
    let x0 = vec![0.0; n];
    let d = diagonal(&a);
    let reference = pcg(&a, &b, &x0, signed_jacobi(&d, 1.0), TOL, MAX_ITER).unwrap();
    assert!(reference.converged);
    for (sign_a, sign_m) in [(1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        for s in [1.0, scale_factors()[0], scale_factors()[1]] {
            let a_s = scaled_matrix(&a, sign_a * s);
            let b_s: Vec<f64> = b.iter().map(|v| v * sign_a * s).collect();
            let d_s: Vec<f64> = d.iter().map(|v| v * s).collect();
            let label = format!("sign(A) = {sign_a}, sign(M) = {sign_m}, scale {s:e}");
            let result = pcg(&a_s, &b_s, &x0, signed_jacobi(&d_s, sign_m), TOL, MAX_ITER)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(result.converged, "{label}: not converged");
            assert_eq!(result.iterations, reference.iterations, "{label}");
            assert!(
                bits(&result.x) == bits(&reference.x),
                "{label}: iterates differ"
            );
            assert_eq!(
                result.residual_norm.to_bits(),
                (reference.residual_norm * s).to_bits(),
                "{label}"
            );
        }
    }
}

#[test]
fn test_block_pcg_consistent_sign_flips_reproduce_the_spd_iterates() {
    let a = variable_spd(GRID);
    let n = a.nrows();
    let bb = block_rhs(&rhs(n));
    let xx = vec![vec![0.0; n]; 2];
    let d = diagonal(&a);
    let reference = block_pcg(&a, &bb, &xx, signed_jacobi(&d, 1.0), TOL, MAX_ITER).unwrap();
    assert!(reference.converged);
    for (sign_a, sign_m) in [(1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        let a_s = scaled_matrix(&a, sign_a);
        let bb_s: Vec<Vec<f64>> = bb
            .iter()
            .map(|bk| bk.iter().map(|v| v * sign_a).collect())
            .collect();
        let label = format!("sign(A) = {sign_a}, sign(M) = {sign_m}");
        let result = block_pcg(&a_s, &bb_s, &xx, signed_jacobi(&d, sign_m), TOL, MAX_ITER)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(result.converged, "{label}: not converged");
        assert_eq!(result.num_converged, 2, "{label}");
        assert_eq!(result.iterations, reference.iterations, "{label}");
        for (x, xr) in result.x.iter().zip(&reference.x) {
            assert!(bits(x) == bits(xr), "{label}: iterates differ");
        }
        assert!(bits(&result.residual_norms) == bits(&reference.residual_norms));
    }
}

#[test]
fn test_pcg_preconditioner_turning_positive_reports_breakdown() {
    // A = diag(1, 2, 3), M^{-1} = diag(-1, -1, 1), b = (1, 1, 1): r0^T z0 = -1
    // fixes the sign of r^T z as negative (p0^T A p0 = 6 > 0), and after the
    // first step r^T z = 40/36 > 0. The sign change reveals the indefinite
    // preconditioner: a breakdown at iteration 1 (not NaN, not iteration 0).
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a = scaled_matrix(&diag_matrix(&[1.0, 2.0, 3.0]), s);
        let b = [s, s, s];
        let result = pcg(
            &a,
            &b,
            &[0.0; 3],
            signed_identity(&[-1.0, -1.0, 1.0]),
            TOL,
            100,
        );
        assert!(
            matches!(result, Err(IterativeError::Breakdown { iteration: 1, .. })),
            "scale {s:e}: expected a breakdown at iteration 1, got {:?}",
            result.map(|r| (r.converged, r.iterations, r.residual_norm))
        );
    }
}

#[test]
fn test_pcg_indefinite_operator_reports_breakdown() {
    // A = diag(1, 1, -1) with M = I and b = (1, 1, 1/2): p0^T A p0 = 7/4 > 0,
    // but p1 = (18, 18, 72)/49 gives p1^T A p1 < 0. The sign change of p^T A p
    // reveals the indefinite operator: an honest breakdown at iteration 1.
    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    for s in [1.0, scale_factors()[0], scale_factors()[1]] {
        let a = scaled_matrix(&diag_matrix(&[1.0, 1.0, -1.0]), s);
        let b = [s, s, 0.5 * s];
        let result = pcg(&a, &b, &[0.0; 3], identity, TOL, 100);
        assert!(
            matches!(result, Err(IterativeError::Breakdown { iteration: 1, .. })),
            "scale {s:e}: expected a breakdown at iteration 1, got {:?}",
            result.map(|r| (r.converged, r.iterations, r.residual_norm))
        );
    }
}

/// `A = diag(1, 0)` (the zero stored explicitly): with `b = (1, 1)` the
/// system is inconsistent and `min ||b - A x|| = 1`.
fn singular_diag() -> CsrMatrix<f64> {
    CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 0.0]).unwrap()
}

#[test]
fn test_gmres_family_inconsistent_system_not_reported_as_converged() {
    // When A v_j = 0, or h_{j+1,j} = 0, the new Givens rotation has s = 0,
    // so the least-squares residual estimate |g_{j+1}| is exactly 0 even
    // though the triangular solve has to drop the singular direction. GMRES,
    // PGMRES and FGMRES used to trust that estimate: on this system the
    // second cycle reported `converged: true, residual_norm: 0` while
    // ||b - A x|| = 1. Convergence must be confirmed by the true residual.
    let a = singular_diag();
    let b = [1.0, 1.0];
    let x0 = [0.0, 0.0];
    let (restart, tol, max_iter) = (10, 1e-10, 100);
    let check = |name: &str, x: &[f64], converged: bool, residual_norm: f64, iterations: usize| {
        let true_residual = true_relative_residual(&a, &b, x) * norm2(&b);
        assert!(
            !converged,
            "{name} reported convergence with true residual {true_residual} (residual_norm {residual_norm})"
        );
        assert!(
            (true_residual - 1.0).abs() <= 1e-12,
            "{name}: true residual {true_residual}"
        );
        assert!(
            (residual_norm - true_residual).abs() <= 1e-12,
            "{name}: reported residual {residual_norm}, true residual {true_residual}"
        );
        assert!(iterations <= max_iter, "{name}: {iterations} iterations");
    };

    let result = gmres(&a, &b, &x0, restart, tol, max_iter).unwrap();
    check(
        "GMRES",
        &result.x,
        result.converged,
        result.residual_norm,
        result.iterations,
    );

    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let result = pgmres(&a, &b, &x0, identity, restart, tol, max_iter).unwrap();
    check(
        "PGMRES",
        &result.x,
        result.converged,
        result.residual_norm,
        result.iterations,
    );

    let mut flexible_identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let result = fgmres(&a, &b, &x0, &mut flexible_identity, restart, tol, max_iter).unwrap();
    check(
        "FGMRES",
        &result.x,
        result.converged,
        result.residual_norm,
        result.iterations,
    );
}

#[test]
fn test_gmres_family_consistent_singular_system_converges() {
    // The same singular A with b = (1, 0) in its range: the dropped
    // direction is not needed, and the true residual confirms convergence.
    let a = singular_diag();
    let b = [1.0, 0.0];
    let x0 = [0.0, 0.0];
    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let mut flexible_identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let results = [
        gmres(&a, &b, &x0, 10, TOL, 100).unwrap(),
        pgmres(&a, &b, &x0, identity, 10, TOL, 100).unwrap(),
    ];
    for (name, result) in ["GMRES", "PGMRES"].iter().zip(&results) {
        assert!(result.converged, "{name} did not converge");
        assert_eq!(result.residual_norm, 0.0, "{name}");
        assert_eq!(result.x, vec![1.0, 0.0], "{name}");
    }
    let result = fgmres(&a, &b, &x0, &mut flexible_identity, 10, TOL, 100).unwrap();
    assert!(result.converged, "FGMRES did not converge");
    assert_eq!(result.residual_norm, 0.0);
    assert_eq!(result.x, vec![1.0, 0.0]);
}

// =============================================================================
// Solver defects found while making the breakdown tests scale-invariant
// =============================================================================

#[test]
fn test_gmres_restart_parameter_is_honoured() {
    // GMRES(n) is full GMRES: it must converge within n Arnoldi steps without
    // restarting. (The lucky-breakdown test used to read the sub-diagonal entry
    // after the Givens rotation had zeroed it, so every step "broke down" and
    // restarted, turning GMRES(m) into GMRES(1) for every m.)
    let a = variable_nonsymmetric(6);
    let n = a.nrows();
    let b = rhs(n);
    let x0 = vec![0.0; n];
    let d = diagonal(&a);
    let jacobi = |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };

    let plain = gmres(&a, &b, &x0, n, TOL, MAX_ITER).unwrap();
    assert!(plain.converged);
    assert_eq!(plain.restarts, 0, "GMRES({n}) restarted");
    assert!(plain.iterations <= n);
    assert!(true_relative_residual(&a, &b, &plain.x) <= TRUE_RESIDUAL_BOUND);

    let precond = pgmres(&a, &b, &x0, jacobi, n, TOL, MAX_ITER).unwrap();
    assert!(precond.converged);
    assert_eq!(precond.restarts, 0, "PGMRES({n}) restarted");
    assert!(precond.iterations <= n);
    assert!(true_relative_residual(&a, &b, &precond.x) <= TRUE_RESIDUAL_BOUND);

    let mut flexible_precond =
        |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };
    let flexible = fgmres(&a, &b, &x0, &mut flexible_precond, n, TOL, MAX_ITER).unwrap();
    assert!(flexible.converged);
    assert_eq!(flexible.restarts, 0, "FGMRES({n}) restarted");
    assert!(flexible.iterations <= n);
    assert!(true_relative_residual(&a, &b, &flexible.x) <= TRUE_RESIDUAL_BOUND);
}

#[test]
fn test_block_gmres_restarted_converges() {
    // Restarted block GMRES must assemble the block Hessenberg matrix
    // correctly for block rows >= 2; it used to diverge here.
    let a = variable_nonsymmetric(8);
    let b = rhs(a.nrows());
    let bb = block_rhs(&b);
    let x0 = vec![vec![0.0; a.nrows()]; 2];
    let result = block_gmres(&a, &bb, &x0, 10, TOL, MAX_ITER).unwrap();
    assert!(result.converged, "block GMRES(10) did not converge");
    for (col, (x, bc)) in result.x.iter().zip(&bb).enumerate() {
        let rel = true_relative_residual(&a, bc, x);
        assert!(
            rel <= TRUE_RESIDUAL_BOUND,
            "column {col}: true residual {rel:e}"
        );
    }
}

#[test]
fn test_pminres_jacobi_converges_on_variable_diagonal() {
    // The preconditioned Lanczos coefficient is v^T A v (v = M^{-1} r / beta);
    // using p^T A v = v^T M A v stalls as soon as M is not the identity.
    let a = variable_spd(8);
    let n = a.nrows();
    let b = rhs(n);
    let x0 = vec![0.0; n];
    let d = diagonal(&a);
    let jacobi = |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };
    let result = pminres(&a, &b, &x0, jacobi, TOL, 500).unwrap();
    assert!(
        result.converged,
        "PMINRES with Jacobi did not converge in {} iterations",
        result.iterations
    );
    assert!(true_relative_residual(&a, &b, &result.x) <= TOL);
}

#[test]
fn test_ptfqmr_jacobi_converges_with_true_residual() {
    // Right-preconditioned TFQMR must keep the Krylov vectors in residual space
    // and map only the solution update through M^{-1}.
    let a = variable_nonsymmetric(8);
    let n = a.nrows();
    let b = rhs(n);
    let x0 = vec![0.0; n];
    let d = diagonal(&a);
    let jacobi = |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };
    let result = ptfqmr(&a, &b, &x0, jacobi, TOL, 500).unwrap();
    assert!(
        result.converged,
        "PTFQMR with Jacobi did not converge in {} half-steps",
        result.iterations
    );
    assert!(true_relative_residual(&a, &b, &result.x) <= TRUE_RESIDUAL_BOUND);
}

#[test]
fn test_idrs_reported_convergence_meets_true_residual() {
    // On the 100 x 100 Laplacian with b = 1 the recursively updated IDR(s)
    // residual used to drift away from b - A x (true relative residuals of
    // 4e-9 and 2.3e-10 were reported as converged at tol = 1e-10). Reported
    // convergence must now be confirmed by the true residual.
    let a = oxiblas_sparse::test_matrices::laplacian_2d(100, 100).unwrap();
    let n = a.nrows();
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];
    for s in [4, 8] {
        let result = idrs(&a, &b, &x0, s, TOL, 2000).unwrap();
        assert!(result.converged, "IDR({s}) did not converge");
        let rel = true_relative_residual(&a, &b, &result.x);
        assert!(
            rel <= TOL,
            "IDR({s}) reported convergence with true relative residual {rel:e}"
        );
        assert!((result.residual_norm / norm2(&b) - rel).abs() <= 1e-3 * rel);
    }
}

#[test]
fn test_idrs_history_records_true_residual_after_replacement() {
    // When the recursive residual meets the tolerance, IDR(s) replaces it by
    // the true residual b - A x. The history must record that true norm for
    // the step: it used to keep the recursive norm, so the last entry of a
    // converged run differed from `residual_norm`, and a residual gap (the
    // true residual still above the tolerance) left a value below the
    // tolerance in the middle of the history. Every case below meets at least
    // one such gap before converging.
    let cases: [(usize, f64, &[usize]); 2] = [(100, TOL, &[4, 8]), (40, 1e-12, &[1, 8])];
    for (grid, tol, s_values) in cases {
        let a = oxiblas_sparse::test_matrices::laplacian_2d(grid, grid).unwrap();
        let n = a.nrows();
        let b = if grid == 100 { vec![1.0; n] } else { rhs(n) };
        let x0 = vec![0.0; n];
        let tol_abs = tol * norm2(&b);
        let d = diagonal(&a);
        let jacobi = |r: &[f64]| -> Vec<f64> { r.iter().zip(&d).map(|(ri, di)| ri / di).collect() };
        for &s in s_values {
            let plain = idrs(&a, &b, &x0, s, tol, 4000).unwrap();
            let precond = pidrs(&a, &b, &x0, &jacobi, s, tol, 4000).unwrap();
            for (name, result) in [("IDR", &plain), ("PIDR", &precond)] {
                let label = format!("{name}({s}) on the {grid}x{grid} Laplacian at tol {tol:e}");
                assert!(result.converged, "{label}: not converged");
                let history = &result.residual_history;
                assert_eq!(
                    history.last().map(|h| h.to_bits()),
                    Some(result.residual_norm.to_bits()),
                    "{label}: last history entry is not the reported true residual"
                );
                for (step, h) in history[..history.len() - 1].iter().enumerate() {
                    assert!(
                        *h > tol_abs,
                        "{label}: history[{step}] = {h:e} meets the tolerance before convergence"
                    );
                }
            }
        }
    }
}

/// `A = I` (order 4).
fn identity_matrix() -> CsrMatrix<f64> {
    CsrMatrix::new(4, 4, (0..=4).collect(), (0..4).collect(), vec![1.0; 4]).unwrap()
}

#[test]
fn test_idrs_iterations_count_confirmation_matvec() {
    // On A = I the first IDR step solves the system: one product with A for
    // the step and one for the true-residual confirmation. `iterations`
    // counts both, and never exceeds `max_iter`: with a budget of one
    // product, the confirmation is left to the final report.
    let a = identity_matrix();
    let b = [1.0, 2.0, 3.0, 4.0];
    let x0 = [0.0; 4];
    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    for s in [1, 2] {
        for max_iter in [1usize, 2, 3, 100] {
            let plain = idrs(&a, &b, &x0, s, TOL, max_iter).unwrap();
            let precond = pidrs(&a, &b, &x0, &identity, s, TOL, max_iter).unwrap();
            for (name, result) in [("IDR", &plain), ("PIDR", &precond)] {
                let label = format!("{name}({s}) max_iter={max_iter}");
                assert!(result.converged, "{label}: not converged");
                assert_eq!(result.iterations, max_iter.min(2), "{label}: iterations");
                assert!(true_relative_residual(&a, &b, &result.x) <= TOL, "{label}");
                assert_eq!(
                    result.residual_history.last().map(|h| h.to_bits()),
                    Some(result.residual_norm.to_bits()),
                    "{label}: history does not end with the reported residual"
                );
            }
        }
    }
}

#[test]
fn test_idrs_iterations_count_breakdown_matvec() {
    // A = [0]: every IDR step breaks down (g = A u = 0, so the pivot p^T g is
    // zero) after its product with A, and every stabilization step meets
    // t = A r = 0. Each cycle therefore spends two products and records one
    // residual. The product of a breaking-down step used to go uncounted, so
    // `max_iter` = 10 allowed 20 products.
    let a = CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![0.0]).unwrap();
    let b = [1.0];
    let x0 = [0.0];
    let max_iter = 10;

    let plain = idrs(&a, &b, &x0, 1, TOL, max_iter).unwrap();
    assert!(!plain.converged);
    assert_eq!(plain.iterations, max_iter);
    assert_eq!(plain.residual_norm, 1.0);
    assert_eq!(plain.residual_history.len(), 1 + max_iter / 2);

    // In PIDR every product with A is preceded by exactly one application of
    // the preconditioner, except the confirmation of a converged residual.
    let applications = std::cell::Cell::new(0usize);
    let counting = |r: &[f64]| -> Vec<f64> {
        applications.set(applications.get() + 1);
        r.to_vec()
    };
    let precond = pidrs(&a, &b, &x0, &counting, 1, TOL, max_iter).unwrap();
    assert!(!precond.converged);
    assert_eq!(precond.iterations, max_iter);
    assert_eq!(
        applications.get(),
        max_iter,
        "PIDR performed {} products with A for {} counted iterations",
        applications.get(),
        precond.iterations
    );
}

#[test]
fn test_ptfqmr_identity_preconditioner_matches_tfqmr_bitwise() {
    let a = variable_nonsymmetric(8);
    let n = a.nrows();
    let b = rhs(n);
    let x0 = vec![0.0; n];
    let identity = |r: &[f64]| -> Vec<f64> { r.to_vec() };
    let plain = tfqmr(&a, &b, &x0, TOL, 500).unwrap();
    let precond = ptfqmr(&a, &b, &x0, identity, TOL, 500).unwrap();
    assert_eq!(plain.iterations, precond.iterations);
    assert!(bits(&plain.x) == bits(&precond.x));
    assert!(bits(&plain.residual_history) == bits(&precond.residual_history));
}
