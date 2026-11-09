//! Internal power-of-four rescaling of a linear system.
//!
//! Every Krylov solver in this module is invariant under rescaling the
//! system: replacing `A` by `α·A`, `b` by `β·b` and the preconditioner `M⁻¹`
//! by `γ·M⁻¹` changes the solution to `x·β/α` and leaves every decision of the
//! iteration unchanged. When `α`, `β` and `γ` are powers of four, IEEE-754
//! arithmetic reproduces this exactly (every product, quotient, sum and square
//! root of consistently scaled operands is the exactly scaled result), as
//! long as nothing overflows or underflows.
//!
//! The solvers use this to keep their working quantities in range. Inner
//! products such as `rᵀr`, `pᵀAp` or `tᵀt`, Givens and Lanczos coefficients
//! and the product of a residual with `A` grow with the square (or a higher
//! power) of the scale of the data: for `(2^520·A, 2^520·b)` the products
//! `A·r` and `‖r‖²` overflow even though the problem is perfectly
//! representable. [`SystemScaling`] therefore picks `α` and `β` so that the
//! largest finite magnitudes of `α·A` and `β·b` lie in `[1, 4)`, the solver
//! iterates on that copy (through [`Operator`], which applies `α·A` with the
//! `alpha` argument of `spmv`, so `A` is never copied), and the results are
//! mapped back exactly: `x = x̂·α/β`, residual norms divided by `β`.
//! [`PrecondScaling`] does the same for the output of a preconditioner, with
//! a `γ` fixed at its first application.
//!
//! Magnitudes within `2^±emax/16` of one (`2^±63` for `f64`) are left alone
//! (factor one), so for a reasonably scaled problem nothing is rescaled and
//! the arithmetic is exactly that of the plain iteration. Since the iteration
//! is exactly invariant, the result does not depend on whether a rescaling
//! took place: a problem and its rescaled copy give bit-identical iterates.

use crate::csr::CsrMatrix;
use crate::linalg::precond::tolerance::{
    binary_exponent, exact_pow2, finite_max_abs, finite_max_abs_slice, gain_normalizer, norm2,
    pow4_normalizer,
};
use crate::ops::{spmv, spmv_transpose};
use core::cell::Cell;
use oxiblas_core::scalar::{Field, Real, Scalar};
use std::borrow::Cow;

/// The operator `scale · A`, applied without copying `A`.
pub(super) struct Operator<'a, T: Scalar> {
    matrix: &'a CsrMatrix<T>,
    scale: T,
}

impl<'a, T: Scalar<Real = T> + Clone + Field + Real> Operator<'a, T> {
    /// Number of rows of `A`.
    #[inline]
    pub(super) fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Number of columns of `A`.
    #[inline]
    pub(super) fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    /// `y = scale · A · x` (`spmv` multiplies each finished row sum by the
    /// power of four `scale`, which is exact).
    #[inline]
    pub(super) fn apply(&self, x: &[T], y: &mut [T]) {
        spmv(self.scale, self.matrix, x, T::zero(), y);
    }

    /// `y = scale · Aᵀ · x`.
    #[inline]
    pub(super) fn apply_transpose(&self, x: &[T], y: &mut [T]) {
        spmv_transpose(self.scale, self.matrix, x, T::zero(), y);
    }

    /// `r = b − scale · A · x`.
    pub(super) fn residual(&self, b: &[T], x: &[T], r: &mut [T]) {
        self.apply(x, r);
        for (ri, bi) in r.iter_mut().zip(b) {
            *ri = *bi - *ri;
        }
    }

    /// The overflow-safe norm `‖b − scale · A · x‖`.
    pub(super) fn true_residual_norm(&self, b: &[T], x: &[T]) -> T {
        let mut r = vec![T::zero(); self.nrows()];
        self.residual(b, x, &mut r);
        norm2(r.iter().copied())
    }
}

/// Multiplies `values` by `2^e` exactly, in steps that keep every factor a
/// normal number (one step for `|e| <= emax`, which is every case in which
/// the scaled values are representable and not subnormal).
fn scale_by_pow2<T: Scalar<Real = T> + Clone + Field + Real>(values: &mut [T], e: i32) {
    let limit = binary_exponent(<T as Scalar>::max_value()) - 1;
    let mut remaining = e;
    while remaining != 0 {
        let step = remaining.clamp(-limit, limit);
        let factor = exact_pow2::<T>(step);
        for value in values.iter_mut() {
            *value = *value * factor;
        }
        remaining -= step;
    }
}

/// The power-of-four factors `α` (for `A`) and `β` (for the right-hand side)
/// of the rescaled system `(α·A) x̂ = β·b`, `x = x̂·α/β`, held as exponents.
#[derive(Debug, Clone, Copy)]
pub(super) struct SystemScaling<T> {
    a_exponent: i32,
    b_exponent: i32,
    a_factor: T,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SystemScaling<T> {
    /// Factors for `A` and a right-hand side whose largest finite magnitude
    /// is `rhs_max` (see the module documentation).
    pub(super) fn new(matrix: &CsrMatrix<T>, rhs_max: T) -> Self {
        let a_factor = pow4_normalizer(finite_max_abs_slice(matrix.values()));
        let b_factor = pow4_normalizer(rhs_max);
        Self {
            a_exponent: binary_exponent(a_factor),
            b_exponent: binary_exponent(b_factor),
            a_factor,
        }
    }

    /// Factors for `A` and the right-hand side `b`.
    pub(super) fn for_rhs(matrix: &CsrMatrix<T>, b: &[T]) -> Self {
        Self::new(matrix, finite_max_abs_slice(b))
    }

    /// Factors for `A` and the right-hand sides `b` taken together.
    pub(super) fn for_rhs_block(matrix: &CsrMatrix<T>, b: &[Vec<T>]) -> Self {
        Self::new(
            matrix,
            finite_max_abs(b.iter().map(|bk| finite_max_abs_slice(bk))),
        )
    }

    /// The same `α` with the right-hand side `b`.
    pub(super) fn with_rhs(&self, b: &[T]) -> Self {
        let b_factor = pow4_normalizer(finite_max_abs_slice(b));
        Self {
            b_exponent: binary_exponent(b_factor),
            ..*self
        }
    }

    /// The rescaled operator `α·A`.
    pub(super) fn operator<'a>(&self, matrix: &'a CsrMatrix<T>) -> Operator<'a, T> {
        Operator {
            matrix,
            scale: self.a_factor,
        }
    }

    /// `β·b` (borrowed when `β = 1`).
    pub(super) fn rhs<'v>(&self, b: &'v [T]) -> Cow<'v, [T]> {
        self.scaled(b, self.b_exponent)
    }

    /// The rescaled initial guess `x̂0 = x0·β/α` (borrowed when `β = α`).
    pub(super) fn initial_guess<'v>(&self, x0: &'v [T]) -> Cow<'v, [T]> {
        self.scaled(x0, self.b_exponent - self.a_exponent)
    }

    /// Maps a solution of the rescaled system back: `x = x̂·α/β`.
    pub(super) fn solution(&self, x: &mut [T]) {
        scale_by_pow2(x, self.a_exponent - self.b_exponent);
    }

    /// Maps a norm of a residual of the rescaled system back: `value / β`.
    pub(super) fn residual(&self, value: T) -> T {
        let mut v = [value];
        scale_by_pow2(&mut v, -self.b_exponent);
        v[0]
    }

    /// [`Self::residual`] for every entry of `values`.
    pub(super) fn residuals(&self, values: &mut [T]) {
        scale_by_pow2(values, -self.b_exponent);
    }

    fn scaled<'v>(&self, v: &'v [T], e: i32) -> Cow<'v, [T]> {
        if e == 0 {
            Cow::Borrowed(v)
        } else {
            let mut owned = v.to_vec();
            scale_by_pow2(&mut owned, e);
            Cow::Owned(owned)
        }
    }
}

/// Power-of-four normalization `γ` of a preconditioner's output.
///
/// At the first application whose input and output both have a finite
/// nonzero largest magnitude, `γ` is fixed so that `γ·M⁻¹` has a gain
/// `max|γ·M⁻¹u| / max|u|` close to one (one when the gain is already within
/// `2^±emax/16` of one); until then outputs are passed through unchanged
/// (for a linear preconditioner they are zero then, or the iteration is about
/// to break down on a non-finite value). Every later output is multiplied by
/// the same `γ`, so the solver sees the consistently scaled preconditioner
/// `γ·M⁻¹`, which leaves its decisions unchanged. The preconditioner must be
/// linear (`M⁻¹(c·u) = c·M⁻¹u`): it receives vectors of the rescaled system.
pub(super) struct PrecondScaling<T> {
    gain: Cell<Option<T>>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> PrecondScaling<T> {
    pub(super) fn new() -> Self {
        Self {
            gain: Cell::new(None),
        }
    }

    /// The factor `γ` (one until it is fixed).
    pub(super) fn factor(&self) -> T {
        self.gain.get().unwrap_or_else(T::one)
    }

    /// Applies `γ` to the preconditioner output `output = M⁻¹ input`,
    /// fixing `γ` first if needed.
    pub(super) fn finish(&self, input: &[T], mut output: Vec<T>) -> Vec<T> {
        let gain = match self.gain.get() {
            Some(gain) => gain,
            None => {
                let input_max = finite_max_abs_slice(input);
                let output_max = finite_max_abs_slice(&output);
                if !(input_max > T::zero()) || !(output_max > T::zero()) {
                    return output;
                }
                let gain = gain_normalizer(input_max, output_max);
                self.gain.set(Some(gain));
                gain
            }
        };
        if gain != T::one() {
            for value in output.iter_mut() {
                *value = *value * gain;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(scale: f64) -> CsrMatrix<f64> {
        CsrMatrix::new(
            2,
            2,
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![5.5 * scale, -scale, 2.0 * scale],
        )
        .unwrap()
    }

    #[test]
    fn test_reasonable_scale_is_left_alone() {
        let a = matrix(1.0);
        let scaling = SystemScaling::for_rhs(&a, &[1.75, -0.5]);
        let b = [1.75, -0.5];
        assert!(matches!(scaling.rhs(&b), Cow::Borrowed(_)));
        assert!(matches!(scaling.initial_guess(&b), Cow::Borrowed(_)));
        let mut y = [0.0; 2];
        scaling.operator(&a).apply(&[1.0, 1.0], &mut y);
        assert_eq!(y, [4.5, 2.0]);
    }

    #[test]
    fn test_extreme_scale_round_trip_is_exact() {
        for s in [2.0_f64.powi(520), 2.0_f64.powi(-520), 2.0_f64.powi(900)] {
            let a = matrix(s);
            let b = [1.75 * s, -0.5 * s];
            let scaling = SystemScaling::for_rhs(&a, &b);
            let b_hat = scaling.rhs(&b);
            assert!(b_hat.iter().all(|v| v.abs() < 4.0) && b_hat.iter().any(|v| v.abs() >= 1.0));
            let x = [0.25, -3.0];
            let x_hat = scaling.initial_guess(&x).into_owned();
            let mut back = x_hat.clone();
            scaling.solution(&mut back);
            assert_eq!(back, x);
            // (α·A) x̂ = β·(A x)
            let mut y_hat = [0.0; 2];
            scaling.operator(&a).apply(&x_hat, &mut y_hat);
            let ax = [(5.5 * 0.25 - 1.0 * -3.0) * s, (2.0 * -3.0) * s];
            let scaled_ax = scaling.rhs(&ax);
            assert_eq!(y_hat.to_vec(), scaled_ax.to_vec());
            assert_eq!(scaling.residual(b_hat[0]), b[0]);
        }
    }

    #[test]
    fn test_precond_scaling_fixed_at_first_nonzero_application() {
        let scaling = PrecondScaling::<f64>::new();
        // A zero input does not fix the gain.
        assert_eq!(scaling.finish(&[0.0, 0.0], vec![0.0, 0.0]), vec![0.0, 0.0]);
        assert_eq!(scaling.factor(), 1.0);
        let tiny = 2.0_f64.powi(-520);
        let out = scaling.finish(&[1.0, 2.0], vec![tiny, 2.0 * tiny]);
        let gain = scaling.factor();
        assert_eq!(binary_exponent(gain) % 2, 0);
        assert!((0.5..4.0).contains(&(out[1] / 2.0)));
        // Later outputs use the same factor.
        assert_eq!(scaling.finish(&[3.0], vec![tiny]), vec![tiny * gain]);
        // A gain close to one is left alone.
        let near = PrecondScaling::<f64>::new();
        assert_eq!(near.finish(&[4.0], vec![1.0]), vec![1.0]);
        assert_eq!(near.factor(), 1.0);
    }
}
