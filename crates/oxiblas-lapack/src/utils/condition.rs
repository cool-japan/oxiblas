//! Condition number estimation.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::norms::{norm_1, norm_inf};
use crate::lu::{Lu, LuError};
use crate::svd::{Svd, SvdError};

/// Error type for condition number computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix is empty.
    EmptyMatrix,
    /// SVD computation failed.
    SvdFailed,
    /// LU computation failed.
    LuFailed,
}

impl core::fmt::Display for CondError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::SvdFailed => write!(f, "SVD computation failed"),
            Self::LuFailed => write!(f, "LU computation failed"),
        }
    }
}

impl std::error::Error for CondError {}

impl From<SvdError> for CondError {
    fn from(e: SvdError) -> Self {
        match e {
            SvdError::EmptyMatrix => Self::EmptyMatrix,
            SvdError::NotConverged => Self::SvdFailed,
        }
    }
}

impl From<LuError> for CondError {
    fn from(e: LuError) -> Self {
        match e {
            LuError::NotSquare { .. } => Self::NotSquare,
            _ => Self::LuFailed,
        }
    }
}

/// Computes the 2-norm condition number of a matrix.
///
/// κ_2(A) = σ_max / σ_min
///
/// This is the most accurate condition number but requires SVD computation.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Returns
///
/// The condition number. Returns infinity if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cond;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[2.0f64, 0.0],
///     &[0.0, 4.0],
/// ]);
///
/// let kappa = cond(a.as_ref()).unwrap();
/// // For diagonal matrix, cond = max/min = 4/2 = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn cond<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let svd = Svd::compute(a)?;
    Ok(svd.condition_number())
}

/// Computes the 1-norm condition number of a square matrix.
///
/// κ_1(A) = ||A||_1 * ||A^(-1)||_1
///
/// Uses LU decomposition to compute the inverse, which is more efficient
/// than SVD for just the 1-norm condition number.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The 1-norm condition number. Returns infinity if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cond_1;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 2.0],
/// ]);
///
/// let kappa = cond_1(a.as_ref()).unwrap();
/// // ||A||_1 = 2, ||A^(-1)||_1 = 1
/// // kappa_1 = 2 * 1 = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn cond_1<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CondError::NotSquare);
    }

    if n == 0 {
        return Err(CondError::EmptyMatrix);
    }

    let norm_a = norm_1(a);

    // Compute inverse via LU
    let lu = match Lu::compute(a) {
        Ok(lu) => lu,
        Err(_) => return Ok(<T as Scalar>::max_value()), // Singular = infinite condition number
    };

    let a_inv = match lu.inverse() {
        Ok(inv) => inv,
        Err(_) => return Ok(<T as Scalar>::max_value()),
    };

    let norm_a_inv = norm_1(a_inv.as_ref());

    Ok(norm_a * norm_a_inv)
}

/// Computes the infinity-norm condition number of a square matrix.
///
/// κ_∞(A) = ||A||_∞ * ||A^(-1)||_∞
///
/// Uses LU decomposition to compute the inverse.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The infinity-norm condition number. Returns infinity if the matrix is singular.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cond_inf;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 2.0],
/// ]);
///
/// let kappa = cond_inf(a.as_ref()).unwrap();
/// // ||A||_inf = 2, ||A^(-1)||_inf = 1
/// // kappa_inf = 2 * 1 = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn cond_inf<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CondError::NotSquare);
    }

    if n == 0 {
        return Err(CondError::EmptyMatrix);
    }

    let norm_a = norm_inf(a);

    // Compute inverse via LU
    let lu = match Lu::compute(a) {
        Ok(lu) => lu,
        Err(_) => return Ok(<T as Scalar>::max_value()),
    };

    let a_inv = match lu.inverse() {
        Ok(inv) => inv,
        Err(_) => return Ok(<T as Scalar>::max_value()),
    };

    let norm_a_inv = norm_inf(a_inv.as_ref());

    Ok(norm_a * norm_a_inv)
}

/// Estimates the reciprocal of the 1-norm condition number.
///
/// rcond(A) = 1 / κ_1(A)
///
/// This is more numerically stable than computing κ directly, and
/// is the form used by LAPACK's DGECON routine.
///
/// Returns 0 if the matrix is singular (infinite condition number).
/// Returns 1 if the matrix is perfectly conditioned (identity-like).
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::rcond;
/// use oxiblas_matrix::Mat;
///
/// let eye = Mat::from_rows(&[
///     &[1.0f64, 0.0],
///     &[0.0, 1.0],
/// ]);
///
/// let rc = rcond(eye.as_ref()).unwrap();
/// assert!((rc - 1.0).abs() < 1e-10);
/// ```
pub fn rcond<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, CondError> {
    let kappa = cond_1(a)?;

    if kappa >= <T as Scalar>::max_value() / T::from_f64(2.0).unwrap_or(T::one()) {
        Ok(T::zero())
    } else {
        Ok(T::one() / kappa)
    }
}

/// Estimates the reciprocal of the 1-norm condition number using the
/// Hager-Higham estimator (equivalent to LAPACK's `DGECON` / `xLACN2`).
///
/// This is a fast estimator (a handful of triangular solves, `O(n²)` work per
/// solve) that avoids forming the full inverse. It obtains a lower bound on
/// `||A^{-1}||_1` by iteratively applying the operator `A^{-1}` and its
/// transpose `(A^{-1})^T = (A^T)^{-1}` to a sequence of sign vectors, following
/// Hager (1984) and Higham (1988).
///
/// Alternating between solves with `A` and `A^T` is an intrinsic part of the
/// algorithm: without the transpose solves the iteration can only ever probe a
/// single direction, systematically *under*-estimating `||A^{-1}||_1` and hence
/// *over*-estimating `rcond`. Over-estimating `rcond` (under-estimating the true
/// condition number) is exactly the failure mode that would make an
/// ill-conditioning warning miss dangerous matrices, so the transpose solves
/// are mandatory.
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// An estimate of `1/κ_1(A)` in `[0, 1]`. Returns `0` for a singular matrix.
pub fn rcond_estimate<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<T, CondError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CondError::NotSquare);
    }

    if n == 0 {
        return Err(CondError::EmptyMatrix);
    }

    let norm_a = norm_1(a);

    // Factorize once; both the `A x = y` and `A^T x = y` solves reuse the
    // same LU factors.
    let lu = match Lu::compute(a) {
        Ok(lu) => lu,
        Err(_) => return Ok(T::zero()), // Singular => rcond = 0.
    };

    // Estimate ||A^{-1}||_1 with the Hager-Higham iteration.
    let norm_a_inv_est = match hager_higham_inv_norm(&lu, n) {
        Some(v) => v,
        None => return Ok(T::zero()), // A solve failed => treat as singular.
    };

    // rcond = 1 / (||A||_1 * ||A^{-1}||_1).
    let kappa_est = norm_a * norm_a_inv_est;

    if kappa_est <= T::zero()
        || kappa_est >= <T as Scalar>::max_value() / T::from_f64(2.0).unwrap_or(T::one())
    {
        Ok(T::zero())
    } else {
        Ok(T::one() / kappa_est)
    }
}

/// One-norm (sum of absolute values) of the first column of `x` (length `n`).
fn column_one_norm<T: Real>(x: &Mat<T>, n: usize) -> T {
    let mut sum = T::zero();
    for i in 0..n {
        sum = sum + Scalar::abs(x[(i, 0)]);
    }
    sum
}

/// Index of the first entry attaining the maximum absolute value in the first
/// column of `x` (length `n`); mirrors LAPACK's `IDAMAX`.
fn column_argmax_abs<T: Real>(x: &Mat<T>, n: usize) -> usize {
    let mut best = 0usize;
    let mut best_val = Scalar::abs(x[(0, 0)]);
    for i in 1..n {
        let val = Scalar::abs(x[(i, 0)]);
        if val > best_val {
            best_val = val;
            best = i;
        }
    }
    best
}

/// Hager-Higham 1-norm estimator of `||A^{-1}||_1`, faithfully following
/// LAPACK's reverse-communication routine `DLACN2`.
///
/// The algorithm alternates between the two operators
/// * `y = A^{-1} x`         (via [`Lu::solve`]) and
/// * `y = (A^{-1})^T x = (A^T)^{-1} x`  (via [`Lu::solve_transpose`]),
///
/// refining a `±1` sign vector until the estimate stops increasing, the sign
/// vector repeats, or the maximiser stabilises (with a hard cap of `ITMAX`
/// iterations). A final estimate using the alternating vector
/// `x_i = (-1)^i (1 + i/(n-1))` is always taken as an additional lower bound.
///
/// Returns `None` if any triangular solve fails.
fn hager_higham_inv_norm<T: Field + Real + bytemuck::Zeroable>(lu: &Lu<T>, n: usize) -> Option<T> {
    const ITMAX: usize = 5;

    // Fortran SIGN(1, val): +1 for val >= 0 (including +0), -1 otherwise.
    let sign_of = |val: T| -> T {
        if val >= T::zero() {
            T::one()
        } else {
            -T::one()
        }
    };

    // Initial probe vector x = (1/n, ..., 1/n)^T.
    let mut x = Mat::<T>::zeros(n, 1);
    let inv_n = T::one() / T::from_f64(n as f64)?;
    for i in 0..n {
        x[(i, 0)] = inv_n;
    }

    // First operator application: x <- A^{-1} x.
    let mut v = lu.solve(x.as_ref()).ok()?;

    if n == 1 {
        // The estimate is exact for a 1×1 matrix.
        return Some(Scalar::abs(v[(0, 0)]));
    }

    let mut est = column_one_norm(&v, n);

    // Sign vector of the first solve; also becomes the next probe.
    let mut isgn = vec![T::one(); n];
    for i in 0..n {
        let s = sign_of(v[(i, 0)]);
        isgn[i] = s;
        x[(i, 0)] = s;
    }

    // x <- (A^{-1})^T x, then locate the dominant component.
    let mut xt = lu.solve_transpose(x.as_ref()).ok()?;
    let mut j = column_argmax_abs(&xt, n);
    let mut iter = 2usize;

    loop {
        // Probe with the j-th unit vector.
        for i in 0..n {
            x[(i, 0)] = T::zero();
        }
        x[(j, 0)] = T::one();

        // x <- A^{-1} x.
        v = lu.solve(x.as_ref()).ok()?;

        let est_old = est;
        est = column_one_norm(&v, n);

        // Convergence test 1: the sign vector repeated exactly.
        let mut sign_changed = false;
        for i in 0..n {
            if sign_of(v[(i, 0)]) != isgn[i] {
                sign_changed = true;
                break;
            }
        }
        if !sign_changed {
            break;
        }

        // Convergence test 2 (anti-cycling): the estimate stopped increasing.
        if est <= est_old {
            break;
        }

        // Adopt the new sign vector as the next transpose probe.
        for i in 0..n {
            let s = sign_of(v[(i, 0)]);
            isgn[i] = s;
            x[(i, 0)] = s;
        }

        // x <- (A^{-1})^T x.
        xt = lu.solve_transpose(x.as_ref()).ok()?;
        let j_last = j;
        j = column_argmax_abs(&xt, n);

        // Convergence test 3: the maximiser did not move (or the cap is hit).
        // `j` maximises |xt|, so |xt[j_last]| <= |xt[j]| always; equality means
        // the previous index still attains the maximum.
        if Scalar::abs(xt[(j_last, 0)]) >= Scalar::abs(xt[(j, 0)]) || iter >= ITMAX {
            break;
        }
        iter += 1;
    }

    // Final refinement with the alternating vector
    //   x_i = (-1)^i * (1 + i/(n-1)),  i = 0..n-1,
    // which frequently exposes a larger lower bound than the sign iterations.
    let mut alt = T::one();
    let denom = T::from_f64((n - 1) as f64)?;
    for i in 0..n {
        let frac = T::from_f64(i as f64)? / denom;
        x[(i, 0)] = alt * (T::one() + frac);
        alt = -alt;
    }
    let vf = lu.solve(x.as_ref()).ok()?;
    let three_n = T::from_f64((3 * n) as f64)?;
    let temp = (T::from_f64(2.0)? * column_one_norm(&vf, n)) / three_n;
    if temp > est {
        est = temp;
    }

    Some(est)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_cond_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        assert!(approx_eq(kappa, 2.0, 1e-10));
    }

    #[test]
    fn test_cond_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let kappa = cond(eye.as_ref()).unwrap();
        assert!(approx_eq(kappa, 1.0, 1e-10));
    }

    #[test]
    fn test_cond_ill_conditioned() {
        // Hilbert matrix is ill-conditioned
        let a = Mat::from_rows(&[&[1.0f64, 1.0 / 2.0], &[1.0 / 2.0, 1.0 / 3.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        // Should be around 19
        assert!(kappa > 15.0);
        assert!(kappa < 25.0);
    }

    #[test]
    fn test_cond_1_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let kappa = cond_1(eye.as_ref()).unwrap();
        assert!(approx_eq(kappa, 1.0, 1e-10));
    }

    #[test]
    fn test_cond_inf_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let kappa = cond_inf(eye.as_ref()).unwrap();
        assert!(approx_eq(kappa, 1.0, 1e-10));
    }

    #[test]
    fn test_cond_1_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let kappa = cond_1(a.as_ref()).unwrap();
        // ||A||_1 = 4, ||A^(-1)||_1 = 0.5
        // kappa = 4 * 0.5 = 2
        assert!(approx_eq(kappa, 2.0, 1e-10));
    }

    #[test]
    fn test_cond_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        // Singular matrix has infinite condition number
        assert!(kappa > 1e10);
    }

    #[test]
    fn test_rcond_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let rc = rcond(eye.as_ref()).unwrap();
        assert!(approx_eq(rc, 1.0, 1e-10));
    }

    #[test]
    fn test_rcond_singular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let rc = cond_1(a.as_ref()).unwrap();
        // Should indicate singular (very large condition number)
        assert!(rc > 1e10);

        let rc2 = rcond(a.as_ref()).unwrap();
        // rcond should be 0 for singular
        assert!(rc2 < 1e-10);
    }

    #[test]
    fn test_rcond_estimate() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);

        let rc_est = rcond_estimate(a.as_ref()).unwrap();
        let rc_exact = rcond(a.as_ref()).unwrap();

        // Estimate should be within a factor of the true value
        // (this is a rough test since estimation is approximate)
        assert!(rc_est > 0.0);
        assert!(rc_est / rc_exact < 10.0);
        assert!(rc_exact / rc_est < 10.0);
    }

    #[test]
    fn test_rcond_estimate_nonsymmetric() {
        // Non-symmetric, moderately ill-conditioned matrix. Capturing
        // ||A^{-1}||_1 here requires alternating A and A^T solves; a single
        // direction (no transpose) would under-estimate it and thereby report
        // an rcond that is too large (too optimistic).
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 1e-2, 5.0], &[4.0, 0.0, 1.0]]);

        let rc_est = rcond_estimate(a.as_ref()).unwrap();
        let rc_exact = rcond(a.as_ref()).unwrap();

        assert!(rc_est > 0.0);
        // The estimator lower-bounds ||A^{-1}||_1, so rc_est >= rc_exact (up to
        // rounding) and must not be far above it.
        assert!(
            rc_est >= rc_exact * (1.0 - 1e-9),
            "rc_est = {rc_est} should not be below rc_exact = {rc_exact}"
        );
        assert!(
            rc_est <= rc_exact * 3.0 + 1e-12,
            "rc_est = {rc_est} over-estimates rcond vs exact = {rc_exact}"
        );
    }

    #[test]
    fn test_rcond_estimate_ill_conditioned_nonsymmetric() {
        // A strongly non-symmetric, badly conditioned matrix: the estimate of
        // rcond must be small (the matrix is near-singular), not close to 1.
        let a = Mat::from_rows(&[&[1.0f64, 1.0, 1.0], &[0.0, 1e-8, 1.0], &[0.0, 0.0, 1e-8]]);

        let rc_est = rcond_estimate(a.as_ref()).unwrap();
        let rc_exact = rcond(a.as_ref()).unwrap();

        assert!(rc_est > 0.0);
        assert!(rc_est < 1e-6, "near-singular matrix should have tiny rcond, got {rc_est}");
        assert!(rc_est >= rc_exact * (1.0 - 1e-9));
        assert!(rc_est <= rc_exact * 5.0 + 1e-12);
    }

    #[test]
    fn test_cond_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = cond_1(a.as_ref());
        assert!(matches!(result, Err(CondError::NotSquare)));

        // cond (2-norm) works for non-square
        let kappa = cond(a.as_ref()).unwrap();
        assert!(kappa > 0.0);
    }

    #[test]
    fn test_cond_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 0.0], &[0.0, 4.0]]);

        let kappa = cond(a.as_ref()).unwrap();
        assert!((kappa - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_cond_relationship() {
        // κ_2(A) <= κ_1(A) <= n * κ_2(A) for n×n matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let k2 = cond(a.as_ref()).unwrap();
        let k1 = cond_1(a.as_ref()).unwrap();

        // Not strict inequalities due to different norms
        // But they should be comparable
        assert!(k1 > 0.0);
        assert!(k2 > 0.0);
    }
}
