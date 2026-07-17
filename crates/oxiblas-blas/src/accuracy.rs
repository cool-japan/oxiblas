#![allow(clippy::doc_markdown)] // Mathematical notation in docs

//! Accuracy and numerical error analysis for BLAS operations.
//!
//! This module provides utilities and tests for measuring the numerical accuracy
//! of BLAS operations, including:
//!
//! - **Forward error**: How close is our result to the exact result?
//! - **Backward error**: What perturbation in the input would make our result exact?
//! - **Relative error**: Error relative to the magnitude of the result
//!
//! # Error Metrics
//!
//! For a computed result ŷ vs exact result y:
//!
//! - **Forward error**: ||ŷ - y||
//! - **Relative forward error**: ||ŷ - y|| / ||y||
//! - **Backward error**: For y = Ax, find smallest ΔA such that ŷ = (A + ΔA)x
//!
//! # Backward-Error Bounds (Higham)
//!
//! The bounds implemented here follow the standard results in Higham,
//! *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002),
//! §3.1. Every bound is built from the constant
//!
//! ```text
//! γ_n = n·u / (1 - n·u)
//! ```
//!
//! where `u` is the *unit roundoff* (`u = ε/2`, with `ε` the machine
//! epsilon returned by [`EPSILON_F64`]/[`EPSILON_F32`]). [`gamma_f64`] and
//! [`gamma_f32`] compute γ_n directly.
//!
//! | Operation | Componentwise backward-error bound | Norm-wise corollary |
//! |-----------|-------------------------------------|----------------------|
//! | DOT (`n` terms) | `\|fl(xᵀy) - xᵀy\| ≤ γ_n Σ\|x_i\|\|y_i\|` ([`dot_backward_error_bound_f64`]) | `≤ γ_n·‖x‖₂·‖y‖₂` ([`dot_error_bound`]) |
//! | GEMV (`k` = inner dim) | `\|fl(Ax) - Ax\| ≤ γ_k \|A\|\|x\|` componentwise ([`gemv_backward_error_bound_f64`]) | `≤ γ_k·‖A‖_F·‖x‖₂` ([`gemv_error_bound`]) |
//! | GEMM (`k` = inner dim) | `\|fl(AB) - AB\| ≤ γ_k \|A\|\|B\|` componentwise ([`gemm_backward_error_bound_f64`]) | `≤ γ_k·‖A‖_F·‖B‖_F` ([`gemm_error_bound`]) |
//!
//! The componentwise bounds are the precise results from Higham §3.1 (the
//! `k` there is the number of terms summed to produce each output entry:
//! the vector length for DOT, and the shared/contraction dimension for
//! GEMV/GEMM). The norm-wise bounds are weaker corollaries obtained via
//! Cauchy-Schwarz (`Σ|x_i||y_i| ≤ ‖x‖₂‖y‖₂`) and submultiplicativity of the
//! Frobenius norm (`‖|A|‖_F = ‖A‖_F` and `‖MN‖_F ≤ ‖M‖_F‖N‖_F`); they are
//! convenient when only aggregate norms (rather than the raw operands) are
//! at hand.

use num_complex::Complex64;
use num_traits::Float;

/// Machine epsilon for f64.
pub const EPSILON_F64: f64 = f64::EPSILON;

/// Machine epsilon for f32.
pub const EPSILON_F32: f32 = f32::EPSILON;

/// Unit roundoff for f64 (`u = ε/2`), as defined in Higham (2002), Table 2.1.
///
/// For IEEE 754 round-to-nearest, every elementary floating-point operation
/// satisfies `fl(a ∘ b) = (a ∘ b)(1 + δ)` with `|δ| ≤ u`. This is the `u`
/// used throughout the γ_n bounds in this module.
pub const UNIT_ROUNDOFF_F64: f64 = f64::EPSILON / 2.0;

/// Unit roundoff for f32 (`u = ε/2`). See [`UNIT_ROUNDOFF_F64`].
pub const UNIT_ROUNDOFF_F32: f32 = f32::EPSILON / 2.0;

/// Error type for accuracy-analysis routines in this module.
///
/// Every public function that previously panicked via `assert!`/`assert_eq!`
/// on malformed input now returns this error instead, so callers can handle
/// mismatched dimensions rather than aborting the whole process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccuracyError {
    /// Two vectors that are required to have equal length do not.
    LengthMismatch {
        /// Length of the first (reference) vector.
        expected: usize,
        /// Length of the second vector.
        found: usize,
    },
    /// A flat buffer's length does not equal `rows * cols` for the matrix
    /// shape it claims to hold.
    MatrixSizeMismatch {
        /// Claimed row count.
        rows: usize,
        /// Claimed column count.
        cols: usize,
        /// Actual length of the supplied buffer.
        found: usize,
    },
}

impl core::fmt::Display for AccuracyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(
                    f,
                    "vectors must have equal length: expected {expected}, found {found}"
                )
            }
            Self::MatrixSizeMismatch { rows, cols, found } => write!(
                f,
                "matrix buffer length {found} does not match {rows} x {cols} = {}",
                rows * cols
            ),
        }
    }
}

impl std::error::Error for AccuracyError {}

/// Computes the maximum absolute error between two vectors.
///
/// max_error = max_i |x_i - y_i|
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `x` and `y` differ in length.
#[inline]
pub fn max_absolute_error<T: Float>(x: &[T], y: &[T]) -> Result<T, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    Ok(x.iter()
        .zip(y.iter())
        .map(|(a, b)| Float::abs(*a - *b))
        .fold(T::zero(), |acc, v| if v > acc { v } else { acc }))
}

/// Computes the L2 (Euclidean) error between two vectors.
///
/// l2_error = sqrt(Σ (x_i - y_i)²)
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `x` and `y` differ in length.
#[inline]
pub fn l2_error<T: Float>(x: &[T], y: &[T]) -> Result<T, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    let sum_sq: T = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let diff = *a - *b;
            diff * diff
        })
        .fold(T::zero(), |acc, v| acc + v);

    Ok(sum_sq.sqrt())
}

/// Computes the L∞ (infinity) norm of a vector.
///
/// l_inf = max_i |x_i|
#[inline]
pub fn linf_norm<T: Float>(x: &[T]) -> T {
    x.iter()
        .map(|v| Float::abs(*v))
        .fold(T::zero(), |acc, v| if v > acc { v } else { acc })
}

/// Computes the L2 (Euclidean) norm of a vector.
///
/// l2_norm = sqrt(Σ x_i²)
#[inline]
pub fn l2_norm<T: Float>(x: &[T]) -> T {
    let sum_sq: T = x.iter().map(|v| *v * *v).fold(T::zero(), |acc, v| acc + v);
    sum_sq.sqrt()
}

/// Computes the relative forward error.
///
/// rel_error = ||computed - exact|| / ||exact||
///
/// Returns infinity if exact is zero (and computed is not).
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `computed` and `exact` differ in length.
#[inline]
pub fn relative_forward_error<T: Float>(computed: &[T], exact: &[T]) -> Result<T, AccuracyError> {
    let error = l2_error(computed, exact)?;
    let exact_norm = l2_norm(exact);

    Ok(if exact_norm > T::zero() {
        error / exact_norm
    } else if error > T::zero() {
        T::infinity()
    } else {
        T::zero()
    })
}

/// Computes componentwise relative error.
///
/// Returns max_i |computed_i - exact_i| / |exact_i|
///
/// Skips components where exact_i is zero.
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `computed` and `exact` differ in length.
#[inline]
pub fn max_relative_error<T: Float>(computed: &[T], exact: &[T]) -> Result<T, AccuracyError> {
    if computed.len() != exact.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: computed.len(),
            found: exact.len(),
        });
    }

    let eps = if std::mem::size_of::<T>() == 4 {
        T::from(EPSILON_F32).unwrap_or_else(|| T::epsilon())
    } else {
        T::from(EPSILON_F64).unwrap_or_else(|| T::epsilon())
    };

    Ok(computed
        .iter()
        .zip(exact.iter())
        .filter(|(_, e)| Float::abs(**e) > eps)
        .map(|(c, e)| Float::abs(*c - *e) / Float::abs(*e))
        .fold(T::zero(), |acc, v| if v > acc { v } else { acc }))
}

/// Complex maximum absolute error.
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `x` and `y` differ in length.
#[inline]
pub fn max_absolute_error_c64(x: &[Complex64], y: &[Complex64]) -> Result<f64, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    Ok(x.iter()
        .zip(y.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max))
}

/// Complex L2 error.
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `x` and `y` differ in length.
#[inline]
pub fn l2_error_c64(x: &[Complex64], y: &[Complex64]) -> Result<f64, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    let sum_sq: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum();

    Ok(sum_sq.sqrt())
}

/// Complex L2 norm.
#[inline]
#[must_use]
pub fn l2_norm_c64(x: &[Complex64]) -> f64 {
    let sum_sq: f64 = x.iter().map(num_complex::Complex::norm_sqr).sum();
    sum_sq.sqrt()
}

/// Complex relative forward error.
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `computed` and `exact` differ in length.
#[inline]
pub fn relative_forward_error_c64(
    computed: &[Complex64],
    exact: &[Complex64],
) -> Result<f64, AccuracyError> {
    let error = l2_error_c64(computed, exact)?;
    let exact_norm = l2_norm_c64(exact);

    Ok(if exact_norm > 0.0 {
        error / exact_norm
    } else if error > 0.0 {
        f64::INFINITY
    } else {
        0.0
    })
}

// =============================================================================
// Reference Implementations for Accuracy Testing
// =============================================================================

/// Reference dot product using Kahan summation for accuracy comparison.
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `x` and `y` differ in length.
pub fn dot_reference_f64(x: &[f64], y: &[f64]) -> Result<f64, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for i in 0..x.len() {
        let y_val = x[i].mul_add(y[i], -c);
        let t = sum + y_val;
        c = (t - sum) - y_val;
        sum = t;
    }

    Ok(sum)
}

/// Reference GEMV using naive triple loop for accuracy comparison.
///
/// Follows the same `beta == 0` contract as the real GEMV implementation:
/// when `beta == 0.0`, `y` is treated as *not referenced* on input (its
/// prior contents, including NaN/Inf, are overwritten rather than
/// multiplied by zero) so that this reference implementation stays a valid
/// ground truth for accuracy comparisons even when callers pass
/// uninitialized or poisoned `y` buffers with `beta = 0`.
///
/// # Errors
///
/// Returns [`AccuracyError::MatrixSizeMismatch`] if `a.len() != a_rows * a_cols`,
/// or [`AccuracyError::LengthMismatch`] if `x` or `y` has the wrong length for `op(A)`.
pub fn gemv_reference_f64(
    trans: bool,
    alpha: f64,
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> Result<(), AccuracyError> {
    if a.len() != a_rows * a_cols {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: a_rows,
            cols: a_cols,
            found: a.len(),
        });
    }

    let (m, n) = if trans {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };

    if x.len() != n {
        return Err(AccuracyError::LengthMismatch {
            expected: n,
            found: x.len(),
        });
    }
    if y.len() != m {
        return Err(AccuracyError::LengthMismatch {
            expected: m,
            found: y.len(),
        });
    }

    // Scale y by beta. beta == 0 means "y is not referenced on input", so we
    // must overwrite rather than multiply -- otherwise NaN/Inf already
    // present in y (a legitimate state for beta == 0 callers) would poison
    // the result via `NaN * 0.0 == NaN`.
    if beta == 0.0 {
        for yi in y.iter_mut() {
            *yi = 0.0;
        }
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }

    // Accumulate A*x with Kahan summation for each output element
    if trans {
        // y = alpha * A^T * x + beta * y
        for j in 0..m {
            let mut sum = 0.0;
            let mut c = 0.0;
            for i in 0..n {
                let prod = a[i + j * a_rows] * x[i];
                let y_val = prod - c;
                let t = sum + y_val;
                c = (t - sum) - y_val;
                sum = t;
            }
            y[j] += alpha * sum;
        }
    } else {
        // y = alpha * A * x + beta * y
        for i in 0..m {
            let mut sum = 0.0;
            let mut c = 0.0;
            for j in 0..n {
                let prod = a[i + j * a_rows] * x[j];
                let y_val = prod - c;
                let t = sum + y_val;
                c = (t - sum) - y_val;
                sum = t;
            }
            y[i] += alpha * sum;
        }
    }

    Ok(())
}

/// Reference GEMM using naive triple loop for accuracy comparison.
///
/// Follows the same `beta == 0` contract as the real GEMM implementation:
/// when `beta == 0.0`, `c` is treated as *not referenced* on input (its
/// prior contents, including NaN/Inf, are overwritten rather than
/// multiplied by zero) so that this reference implementation stays a valid
/// ground truth for accuracy comparisons even when callers pass
/// uninitialized or poisoned `c` buffers with `beta = 0`.
///
/// # Errors
///
/// Returns [`AccuracyError::MatrixSizeMismatch`] if `a`, `b`, or `c` does not
/// match its claimed `rows * cols` size.
pub fn gemm_reference_f64(
    alpha: f64,
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_cols: usize,
    beta: f64,
    c: &mut [f64],
) -> Result<(), AccuracyError> {
    let m = a_rows;
    let k = a_cols;
    let n = b_cols;

    if a.len() != m * k {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: m,
            cols: k,
            found: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: k,
            cols: n,
            found: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: m,
            cols: n,
            found: c.len(),
        });
    }

    // Scale C by beta. beta == 0 means "C is not referenced on input", so we
    // must overwrite rather than multiply -- otherwise NaN/Inf already
    // present in C (a legitimate state for beta == 0 callers) would poison
    // the result via `NaN * 0.0 == NaN`.
    if beta == 0.0 {
        for ci in c.iter_mut() {
            *ci = 0.0;
        }
    } else if beta != 1.0 {
        for ci in c.iter_mut() {
            *ci *= beta;
        }
    }

    // Accumulate A*B with Kahan summation for each output element
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            let mut comp = 0.0;
            for p in 0..k {
                let prod = a[i + p * m] * b[p + j * k];
                let y = prod - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
            c[i + j * m] += alpha * sum;
        }
    }

    Ok(())
}

// =============================================================================
// Error Bound Calculators
// =============================================================================

/// Higham's γ_n constant: `γ_n = n·u / (1 - n·u)`.
///
/// This is the standard bound (Higham 2002, §3.1, eq. 3.4) on the relative
/// error accumulated after combining `n` quantities each individually
/// correct to within the unit roundoff `u` -- the building block for every
/// backward-error bound in this module (dot products, matrix-vector
/// products, matrix-matrix products).
///
/// Returns `f64::INFINITY` if `n·u ≥ 1`, since the bound's denominator would
/// be non-positive; this only happens for `n` far beyond any value a real
/// BLAS call would use.
#[must_use]
pub fn gamma_f64(n: usize) -> f64 {
    let nu = n as f64 * UNIT_ROUNDOFF_F64;
    if nu >= 1.0 {
        f64::INFINITY
    } else {
        nu / (1.0 - nu)
    }
}

/// Higham's γ_n constant for `f32`. See [`gamma_f64`].
#[must_use]
pub fn gamma_f32(n: usize) -> f32 {
    let nu = n as f32 * UNIT_ROUNDOFF_F32;
    if nu >= 1.0 {
        f32::INFINITY
    } else {
        nu / (1.0 - nu)
    }
}

/// Componentwise backward-error bound for an `n`-term dot product
/// (Higham 2002, §3.1, eq. 3.5).
///
/// The computed inner product satisfies `fl(xᵀy) = xᵀy + e` with
/// `|e| ≤ γ_n Σ|x_i||y_i|`, where `γ_n = n·u/(1-n·u)` ([`gamma_f64`]).
/// This returns that exact scalar bound, computed from the operands
/// themselves rather than an aggregate norm.
///
/// # Errors
///
/// Returns [`AccuracyError::LengthMismatch`] if `x` and `y` differ in length.
pub fn dot_backward_error_bound_f64(x: &[f64], y: &[f64]) -> Result<f64, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::LengthMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }

    let abs_sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| xi.abs() * yi.abs())
        .sum();
    Ok(gamma_f64(x.len()) * abs_sum)
}

/// Norm-wise backward-error bound for a dot product.
///
/// A looser corollary of [`dot_backward_error_bound_f64`], obtained via
/// Cauchy-Schwarz (`Σ|x_i||y_i| ≤ ‖x‖₂‖y‖₂`): `|e| ≤ γ_n · x_norm · y_norm`.
/// Useful when only the 2-norms of `x` and `y` are available.
#[must_use]
pub fn dot_error_bound(n: usize, x_norm: f64, y_norm: f64) -> f64 {
    gamma_f64(n) * x_norm * y_norm
}

/// Componentwise backward-error bound for GEMV (Higham 2002, §3.1, applied
/// to each output entry as an inner product).
///
/// The computed `y = op(A)·x` (where `op(A) = A` if `trans` is `false`, or
/// `Aᵀ` if `trans` is `true`) satisfies `fl(y) = y + Δy` with
/// `|Δy| ≤ γ_k |A||x|` componentwise, where `k` is the number of terms
/// summed per output element (`a_cols` when `trans` is `false`, `a_rows`
/// when `trans` is `true`).
///
/// Returns one bound per output element, in the same order that
/// [`gemv_reference_f64`] (and the real `gemv`) produce `y`.
///
/// # Errors
///
/// Returns [`AccuracyError::MatrixSizeMismatch`] if `a.len() != a_rows * a_cols`,
/// or [`AccuracyError::LengthMismatch`] if `x` has the wrong length for `op(A)`.
pub fn gemv_backward_error_bound_f64(
    trans: bool,
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    x: &[f64],
) -> Result<Vec<f64>, AccuracyError> {
    if a.len() != a_rows * a_cols {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: a_rows,
            cols: a_cols,
            found: a.len(),
        });
    }

    let (m, k) = if trans {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };

    if x.len() != k {
        return Err(AccuracyError::LengthMismatch {
            expected: k,
            found: x.len(),
        });
    }

    let gamma_k = gamma_f64(k);
    let mut bound = vec![0.0; m];

    if trans {
        for j in 0..m {
            let mut abs_sum = 0.0;
            for i in 0..k {
                abs_sum += a[i + j * a_rows].abs() * x[i].abs();
            }
            bound[j] = gamma_k * abs_sum;
        }
    } else {
        for i in 0..m {
            let mut abs_sum = 0.0;
            for j in 0..k {
                abs_sum += a[i + j * a_rows].abs() * x[j].abs();
            }
            bound[i] = gamma_k * abs_sum;
        }
    }

    Ok(bound)
}

/// Norm-wise backward-error bound for GEMV.
///
/// A looser corollary of [`gemv_backward_error_bound_f64`], obtained from
/// `‖|A||x|‖₂ ≤ ‖A‖_F‖x‖₂` (any matrix 2-norm bound applied to `|A|`,
/// combined with `‖|A|‖_F = ‖A‖_F`): `|Δy| ≤ γ_k · a_frobenius · x_norm`
/// (as a bound on `‖Δy‖₂`). Useful when only aggregate norms are available.
#[must_use]
pub fn gemv_error_bound(k: usize, a_frobenius: f64, x_norm: f64) -> f64 {
    gamma_f64(k) * a_frobenius * x_norm
}

/// Componentwise backward-error bound for GEMM (Higham 2002, §3.1, applied
/// to each output entry as an inner product of length `k`).
///
/// For `A` (`m×k`) and `B` (`k×n`), the computed `C = A·B` satisfies
/// `fl(C) = C + E` with `|E| ≤ γ_k |A||B|` componentwise, where `k` is the
/// shared (contraction) dimension.
///
/// Returns the `m×n` bound matrix in the same column-major layout that
/// [`gemm_reference_f64`] (and the real `gemm`) use for `C`.
///
/// # Errors
///
/// Returns [`AccuracyError::MatrixSizeMismatch`] if `a` or `b` does not
/// match its claimed `rows * cols` size.
pub fn gemm_backward_error_bound_f64(
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_cols: usize,
) -> Result<Vec<f64>, AccuracyError> {
    let m = a_rows;
    let k = a_cols;
    let n = b_cols;

    if a.len() != m * k {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: m,
            cols: k,
            found: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows: k,
            cols: n,
            found: b.len(),
        });
    }

    let gamma_k = gamma_f64(k);
    let mut bound = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut abs_sum = 0.0;
            for p in 0..k {
                abs_sum += a[i + p * m].abs() * b[p + j * k].abs();
            }
            bound[i + j * m] = gamma_k * abs_sum;
        }
    }

    Ok(bound)
}

/// Norm-wise backward-error bound for GEMM.
///
/// A looser corollary of [`gemm_backward_error_bound_f64`], obtained from
/// submultiplicativity of the Frobenius norm (`‖|A||B|‖_F ≤ ‖A‖_F‖B‖_F`):
/// `‖E‖_F ≤ γ_k · a_frobenius · b_frobenius`. Useful when only aggregate
/// norms are available.
#[must_use]
pub fn gemm_error_bound(k: usize, a_frobenius: f64, b_frobenius: f64) -> f64 {
    gamma_f64(k) * a_frobenius * b_frobenius
}

/// Compute Frobenius norm of a matrix stored in column-major order.
///
/// # Errors
///
/// Returns [`AccuracyError::MatrixSizeMismatch`] if `matrix.len() != rows * cols`.
pub fn frobenius_norm_f64(matrix: &[f64], rows: usize, cols: usize) -> Result<f64, AccuracyError> {
    if matrix.len() != rows * cols {
        return Err(AccuracyError::MatrixSizeMismatch {
            rows,
            cols,
            found: matrix.len(),
        });
    }

    let sum_sq: f64 = matrix.iter().map(|v| v * v).sum();
    Ok(sum_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level1::{dot, nrm2};
    use crate::level2::{GemvTrans, gemv};
    use crate::level3::gemm;
    use oxiblas_matrix::Mat;

    // ==========================================================================
    // Error Metric Tests
    // ==========================================================================

    #[test]
    fn test_max_absolute_error() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.5, 3.0];
        assert!((max_absolute_error(&x, &y).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_l2_error() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 2.0, 3.0];
        assert!((l2_error(&x, &y).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_forward_error() {
        let exact = [1.0, 2.0, 3.0];
        let computed = [1.0, 2.0, 3.0];
        assert!(relative_forward_error(&computed, &exact).unwrap() < 1e-10);

        let computed_with_error = [1.001, 2.0, 3.0];
        let rel_err = relative_forward_error(&computed_with_error, &exact).unwrap();
        assert!(rel_err > 0.0);
        assert!(rel_err < 0.01);
    }

    // ==========================================================================
    // Dimension-Mismatch Error Tests (no-panic policy)
    // ==========================================================================

    #[test]
    fn test_length_mismatch_returns_error_not_panic() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0];

        assert_eq!(
            max_absolute_error(&x, &y),
            Err(AccuracyError::LengthMismatch {
                expected: 3,
                found: 2
            })
        );
        assert_eq!(
            l2_error(&x, &y),
            Err(AccuracyError::LengthMismatch {
                expected: 3,
                found: 2
            })
        );
        assert!(relative_forward_error(&x, &y).is_err());
        assert!(max_relative_error(&x, &y).is_err());
        assert!(dot_reference_f64(&x, &y).is_err());
        assert!(dot_backward_error_bound_f64(&x, &y).is_err());
    }

    #[test]
    fn test_matrix_size_mismatch_returns_error_not_panic() {
        let a = [1.0, 2.0, 3.0]; // only 3 elements, claims 2x2 = 4
        assert_eq!(
            frobenius_norm_f64(&a, 2, 2),
            Err(AccuracyError::MatrixSizeMismatch {
                rows: 2,
                cols: 2,
                found: 3
            })
        );

        let mut y = [0.0; 2];
        let x = [1.0, 2.0];
        assert!(gemv_reference_f64(false, 1.0, &a, 2, 2, &x, 0.0, &mut y).is_err());

        let mut c = [0.0; 4];
        assert!(gemm_reference_f64(1.0, &a, 2, 2, &a, 2, 0.0, &mut c).is_err());
    }

    // ==========================================================================
    // beta == 0 NaN-poisoning tests (reference implementations must match the
    // "C/y not referenced when beta == 0" contract of the real BLAS kernels)
    // ==========================================================================

    #[test]
    fn test_gemv_reference_beta_zero_does_not_propagate_nan() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2, column-major
        let x = [1.0, 1.0];
        let mut y = [f64::NAN, f64::NAN];

        gemv_reference_f64(false, 1.0, &a, 2, 2, &x, 0.0, &mut y).unwrap();

        assert!(
            y.iter().all(|v| v.is_finite()),
            "NaN leaked through beta=0 scaling: {y:?}"
        );
        // y = A*x = [1*1+3*1, 2*1+4*1] = [4, 6]
        assert!((y[0] - 4.0).abs() < 1e-12);
        assert!((y[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_gemm_reference_beta_zero_does_not_propagate_nan() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [1.0, 0.0, 0.0, 1.0]; // identity 2x2
        let mut c = vec![f64::NAN; 4];

        gemm_reference_f64(1.0, &a, 2, 2, &b, 2, 0.0, &mut c).unwrap();

        assert!(
            c.iter().all(|v| v.is_finite()),
            "NaN leaked through beta=0 scaling: {c:?}"
        );
        // C = A * I = A
        assert!((c[0] - 1.0).abs() < 1e-12);
        assert!((c[1] - 2.0).abs() < 1e-12);
        assert!((c[2] - 3.0).abs() < 1e-12);
        assert!((c[3] - 4.0).abs() < 1e-12);
    }

    // ==========================================================================
    // DOT Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_dot_accuracy_random() {
        // Test with pseudo-random values
        let n = 1000;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();

        let computed = dot(&x, &y);
        let reference = dot_reference_f64(&x, &y).unwrap();

        let rel_error = (computed - reference).abs() / reference.abs();

        // Should be within n * epsilon
        let bound = (n as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 10.0, // Allow 10x margin for implementation differences
            "DOT relative error {} exceeds bound {} * 10",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_dot_accuracy_ill_conditioned() {
        // Test with numbers of very different magnitudes
        let mut x = vec![1e15; 100];
        let mut y = vec![1e15; 100];
        x[50] = 1e-15;
        y[50] = 1e-15;

        let computed = dot(&x, &y);
        let reference = dot_reference_f64(&x, &y).unwrap();

        // For ill-conditioned data, we expect larger relative error
        let rel_error = (computed - reference).abs() / reference.abs();

        // Should still be reasonable (within square root of condition number)
        assert!(
            rel_error < 1e-10,
            "DOT ill-conditioned relative error {} too large",
            rel_error
        );
    }

    #[test]
    fn test_dot_accuracy_cancellation() {
        // Test near-cancellation case
        let x: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let y: Vec<f64> = vec![1.0; 100];

        let computed = dot(&x, &y);
        // Should be 0 for even length
        assert!(
            computed.abs() < 1e-10,
            "DOT cancellation result {} should be 0",
            computed
        );
    }

    // ==========================================================================
    // GEMV Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_gemv_accuracy_random() {
        let m = 100;
        let n = 80;

        let a_data: Vec<f64> = (0..m * n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, n, &a_data);

        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let mut y_computed = vec![0.0; m];
        let mut y_reference = vec![0.0; m];

        gemv(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            &x,
            0.0,
            &mut y_computed,
        );
        gemv_reference_f64(false, 1.0, &a_data, m, n, &x, 0.0, &mut y_reference).unwrap();

        let rel_error = relative_forward_error(&y_computed, &y_reference).unwrap();

        // Error bound: (k+1) * epsilon
        let bound = ((n + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0, // Allow generous margin
            "GEMV relative error {} exceeds bound {} * 100",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_gemv_accuracy_transpose() {
        let m = 80;
        let n = 100;

        let a_data: Vec<f64> = (0..m * n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, n, &a_data);

        let x: Vec<f64> = (0..m)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let mut y_computed = vec![0.0; n];
        let mut y_reference = vec![0.0; n];

        gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y_computed);
        gemv_reference_f64(true, 1.0, &a_data, m, n, &x, 0.0, &mut y_reference).unwrap();

        let rel_error = relative_forward_error(&y_computed, &y_reference).unwrap();
        let bound = ((m + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMV transpose relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    // ==========================================================================
    // GEMM Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_gemm_accuracy_random() {
        let m = 64;
        let k = 48;
        let n = 56;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, k, &a_data);

        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let b = Mat::from_slice(k, n, &b_data);

        let mut c = Mat::zeros(m, n);
        let mut c_reference = vec![0.0; m * n];

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        gemm_reference_f64(1.0, &a_data, m, k, &b_data, n, 0.0, &mut c_reference).unwrap();

        // Convert c to slice for comparison
        let c_computed: Vec<f64> = (0..m * n).map(|i| c[(i % m, i / m)]).collect();

        let rel_error = relative_forward_error(&c_computed, &c_reference).unwrap();
        let bound = ((k + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMM relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_gemm_accuracy_large() {
        let m = 128;
        let k = 96;
        let n = 112;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, k, &a_data);

        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let b = Mat::from_slice(k, n, &b_data);

        let mut c = Mat::zeros(m, n);
        let mut c_reference = vec![0.0; m * n];

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        gemm_reference_f64(1.0, &a_data, m, k, &b_data, n, 0.0, &mut c_reference).unwrap();

        let c_computed: Vec<f64> = (0..m * n).map(|i| c[(i % m, i / m)]).collect();

        let rel_error = relative_forward_error(&c_computed, &c_reference).unwrap();
        let bound = ((k + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMM large matrix relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    #[test]
    fn test_gemm_accuracy_with_alpha_beta() {
        let m = 32;
        let k = 24;
        let n = 28;

        let a_data: Vec<f64> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();
        let a = Mat::from_slice(m, k, &a_data);

        let b_data: Vec<f64> = (0..k * n)
            .map(|i| ((i * 11 + 17) % 1000) as f64 / 1000.0)
            .collect();
        let b = Mat::from_slice(k, n, &b_data);

        // Initialize C with non-zero values
        let c_init: Vec<f64> = (0..m * n)
            .map(|i| ((i * 3 + 5) % 1000) as f64 / 1000.0)
            .collect();
        let mut c = Mat::from_slice(m, n, &c_init);
        let mut c_reference = c_init.clone();

        let alpha = 2.5;
        let beta = 0.3;

        gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
        gemm_reference_f64(alpha, &a_data, m, k, &b_data, n, beta, &mut c_reference).unwrap();

        let c_computed: Vec<f64> = (0..m * n).map(|i| c[(i % m, i / m)]).collect();

        let rel_error = relative_forward_error(&c_computed, &c_reference).unwrap();
        let bound = ((k + 1) as f64) * EPSILON_F64;
        assert!(
            rel_error < bound * 100.0,
            "GEMM with alpha/beta relative error {} exceeds bound {}",
            rel_error,
            bound
        );
    }

    // ==========================================================================
    // NRM2 Accuracy Tests
    // ==========================================================================

    #[test]
    fn test_nrm2_accuracy() {
        let n = 1000;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + 13) % 1000) as f64 / 1000.0)
            .collect();

        let computed = nrm2(&x);
        let reference = l2_norm(&x);

        let rel_error = (computed - reference).abs() / reference;
        assert!(
            rel_error < 1e-14,
            "NRM2 relative error {} too large",
            rel_error
        );
    }

    #[test]
    fn test_nrm2_accuracy_overflow_prevention() {
        // Test that we handle large values without overflow
        let x = vec![1e150; 4];
        let computed = nrm2(&x);
        let expected = 2.0 * 1e150;

        let rel_error = (computed - expected).abs() / expected;
        assert!(
            rel_error < 1e-14,
            "NRM2 overflow prevention error {} too large",
            rel_error
        );
    }

    #[test]
    fn test_nrm2_accuracy_underflow_prevention() {
        // Test that we handle small values without underflow
        let x = vec![1e-150; 4];
        let computed = nrm2(&x);
        let expected = 2.0 * 1e-150;

        let rel_error = (computed - expected).abs() / expected;
        assert!(
            rel_error < 1e-14,
            "NRM2 underflow prevention error {} too large",
            rel_error
        );
    }

    // ==========================================================================
    // gamma_n / Backward-Error Bound Tests (hand-computed)
    // ==========================================================================

    #[test]
    fn test_gamma_f64_hand_computed() {
        // gamma_1 = u / (1 - u)
        let u = UNIT_ROUNDOFF_F64;
        let expected_gamma_1 = u / (1.0 - u);
        assert!((gamma_f64(1) - expected_gamma_1).abs() < 1e-30);

        // gamma_0 = 0 (no terms summed, no error)
        assert_eq!(gamma_f64(0), 0.0);

        // gamma_n is monotonically increasing in n
        assert!(gamma_f64(10) > gamma_f64(5));
        assert!(gamma_f64(1000) > gamma_f64(10));

        // gamma_n for a huge n saturates to infinity once n*u >= 1
        let huge_n = (1.0 / UNIT_ROUNDOFF_F64).ceil() as usize + 1;
        assert!(gamma_f64(huge_n).is_infinite());
    }

    #[test]
    fn test_gamma_f32_hand_computed() {
        let u = UNIT_ROUNDOFF_F32;
        let expected_gamma_4 = 4.0 * u / (1.0 - 4.0 * u);
        assert!((gamma_f32(4) - expected_gamma_4).abs() < 1e-12);
    }

    #[test]
    fn test_dot_backward_error_bound_hand_computed() {
        // x = [1, 2], y = [3, 4] => sum|x_i y_i| = 1*3 + 2*4 = 11
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];

        let bound = dot_backward_error_bound_f64(&x, &y).unwrap();
        let expected = gamma_f64(2) * 11.0;
        assert!(
            (bound - expected).abs() < 1e-25,
            "bound = {bound}, expected = {expected}"
        );

        // Sanity: gamma_2 by hand = 2u/(1-2u)
        let u = UNIT_ROUNDOFF_F64;
        let gamma_2 = 2.0 * u / (1.0 - 2.0 * u);
        assert!((gamma_f64(2) - gamma_2).abs() < 1e-30);
        assert!((bound - gamma_2 * 11.0).abs() < 1e-25);
    }

    #[test]
    fn test_gemm_backward_error_bound_hand_computed_ill_conditioned() {
        // A 2x2 matrix with a large-magnitude, near-cancelling inner
        // dimension (k = 2): A = [[1e8, -1e8], [1, 1]] (column-major:
        // column 0 = [1e8, 1], column 1 = [-1e8, 1]).
        // B = [[1, 1], [1, 1]] (column-major: both columns = [1, 1]).
        let a: [f64; 4] = [1e8, 1.0, -1e8, 1.0]; // 2x2, column-major
        let b: [f64; 4] = [1.0, 1.0, 1.0, 1.0]; // 2x2, column-major
        let k = 2;

        let bound = gemm_backward_error_bound_f64(&a, 2, k, &b, 2).unwrap();
        assert_eq!(bound.len(), 4);

        let gamma_k = gamma_f64(k);

        // Entry (0,0) = gamma_k * (|A_00||B_00| + |A_01||B_10|)
        //             = gamma_k * (1e8*1 + 1e8*1) = gamma_k * 2e8
        let expected_00 = gamma_k * 2e8;
        assert!(
            (bound[0] - expected_00).abs() < expected_00 * 1e-10,
            "bound[0] = {}, expected = {expected_00}",
            bound[0]
        );

        // Entry (1,0) = gamma_k * (|A_10||B_00| + |A_11||B_10|) = gamma_k * (1+1) = gamma_k*2
        let expected_10 = gamma_k * 2.0;
        assert!(
            (bound[1] - expected_10).abs() < 1e-20,
            "bound[1] = {}, expected = {expected_10}",
            bound[1]
        );

        // The actual computed C is dominated by catastrophic cancellation in
        // row 0 (1e8 - 1e8 = 0 exactly here, but the *bound* correctly
        // reflects that up to ~gamma_k*2e8 of error could have occurred),
        // demonstrating the bound is meaningful precisely in the
        // ill-conditioned regime where norm-wise bounds would be misleading.
        let mut c = vec![0.0; 4];
        gemm_reference_f64(1.0, &a, 2, k, &b, 2, 0.0, &mut c).unwrap();
        assert!(
            c[0].abs() <= bound[0] + 1e-6,
            "actual error exceeds bound: c[0]={}, bound={}",
            c[0],
            bound[0]
        );
    }

    #[test]
    fn test_gemv_backward_error_bound_matches_gemm_bound_for_single_column() {
        // A GEMV with x as a single column is equivalent to a GEMM with n=1;
        // cross-check the two componentwise bound implementations agree.
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2, column-major
        let x = [5.0, 6.0];

        let gemv_bound = gemv_backward_error_bound_f64(false, &a, 2, 2, &x).unwrap();
        let gemm_bound = gemm_backward_error_bound_f64(&a, 2, 2, &x, 1).unwrap();

        assert_eq!(gemv_bound.len(), gemm_bound.len());
        for (g_v, g_m) in gemv_bound.iter().zip(gemm_bound.iter()) {
            assert!((g_v - g_m).abs() < 1e-25, "{g_v} vs {g_m}");
        }
    }

    // ==========================================================================
    // Error Bound Tests
    // ==========================================================================

    #[test]
    fn test_error_bound_dot() {
        let n = 100;
        let x_norm = 10.0;
        let y_norm = 5.0;

        let bound = dot_error_bound(n, x_norm, y_norm);
        assert!(bound > 0.0);
        assert!(bound < 1.0); // Should be small for reasonable inputs
        assert!((bound - gamma_f64(n) * x_norm * y_norm).abs() < 1e-20);
    }

    #[test]
    fn test_error_bound_gemv() {
        let k = 100;
        let a_frobenius = 50.0;
        let x_norm = 5.0;

        let bound = gemv_error_bound(k, a_frobenius, x_norm);
        assert!(bound > 0.0);
        assert!((bound - gamma_f64(k) * a_frobenius * x_norm).abs() < 1e-15);
    }

    #[test]
    fn test_error_bound_gemm() {
        let k = 100;
        let a_frobenius = 50.0;
        let b_frobenius = 40.0;

        let bound = gemm_error_bound(k, a_frobenius, b_frobenius);
        assert!(bound > 0.0);
        assert!((bound - gamma_f64(k) * a_frobenius * b_frobenius).abs() < 1e-15);
    }

    #[test]
    fn test_frobenius_norm() {
        // 2x2 matrix [[1, 2], [3, 4]] stored column-major as [1, 3, 2, 4]
        let matrix = [1.0, 3.0, 2.0, 4.0];
        let norm = frobenius_norm_f64(&matrix, 2, 2).unwrap();
        let expected = (1.0 + 9.0 + 4.0 + 16.0_f64).sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }
}
