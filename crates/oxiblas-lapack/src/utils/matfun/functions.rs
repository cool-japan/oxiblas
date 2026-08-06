//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use num_traits::Float;
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{Mat, MatRef};
/// Computes the matrix exponential e^A using the Padé approximation with scaling and squaring.
///
/// # Arguments
///
/// * `a` - Square matrix
///
/// # Returns
///
/// The matrix exponential e^A
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::expm;
/// use oxiblas_matrix::Mat;
///
/// // For the identity matrix, e^I = e*I
/// let eye = Mat::<f64>::eye(2);
/// let result = expm(eye.as_ref()).unwrap();
///
/// // Diagonal elements should be approximately e
/// let e = std::f64::consts::E;
/// assert!((result[(0, 0)] - e).abs() < 1e-10);
/// ```
pub fn expm<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = Real::exp(a[(0, 0)]);
        return Ok(result);
    }
    let norm = matrix_1_norm(&a);
    let ln2: T = Float::ln(T::one() + T::one());
    let log2_norm: T = Float::ln(norm) / ln2;
    let s = max_int(T::zero(), Float::ceil(log2_norm));
    let two = T::one() + T::one();
    let neg_s = from_f64::<T>(-(s as f64));
    let scale_factor: T = Float::powf(two, neg_s);
    let mut scaled = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            scaled[(i, j)] = a[(i, j)] * scale_factor;
        }
    }
    let pade = pade_approximant(&scaled, 13);
    let mut result = pade;
    for _ in 0..s {
        result = mat_mult(&result, &result);
    }
    Ok(result)
}
/// Computes the principal matrix logarithm log(A) for A with positive eigenvalues.
///
/// Uses the inverse scaling and squaring method.
///
/// # Arguments
///
/// * `a` - Square matrix with positive eigenvalues
///
/// # Returns
///
/// The principal matrix logarithm
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::logm;
/// use oxiblas_matrix::Mat;
///
/// // log(e*I) = I
/// let e = std::f64::consts::E;
/// let a = Mat::from_rows(&[
///     &[e, 0.0],
///     &[0.0, e],
/// ]);
/// let result = logm(a.as_ref()).unwrap();
///
/// // Inverse scaling and squaring recovers the identity to near machine precision.
/// assert!((result[(0, 0)] - 1.0).abs() < 1e-12);
/// ```
pub fn logm<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let val = a[(0, 0)];
        if !(val > T::zero()) {
            return Err(MatFunError::NotDefined);
        }
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = Real::ln(val);
        return Ok(result);
    }
    let mut work = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            work[(i, j)] = a[(i, j)];
        }
    }
    // Inverse scaling and squaring (Higham's algorithm): repeatedly take the
    // matrix square root of `work` until ||work - I|| is small enough that the
    // log(I + X) series converges to near machine precision in a handful of
    // terms. A threshold of ~0.1 keeps the number of square roots modest while
    // limiting the cancellation incurred when forming X = work - I to about one
    // decimal digit, leaving the result accurate to ~10*eps.
    let theta = from_f64::<T>(0.1);
    let mut s = 0;
    while matrix_1_norm_minus_identity(&work) > theta && s < 100 {
        work = sqrtm_newton(&work)?;
        s += 1;
    }
    let mut x = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            x[(i, j)] = work[(i, j)];
        }
        x[(i, i)] = x[(i, i)] - T::one();
    }
    // log(work) = log(I + X). Scaling back by 2^s recovers log(A), since each
    // square root halves the log: log(A) = 2^s * log(A^(1/2^s)).
    let log_x = log1p_taylor(&x);
    let two = T::one() + T::one();
    let s_t = from_f64::<T>(s as f64);
    let scale: T = Float::powf(two, s_t);
    let mut result = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = log_x[(i, j)] * scale;
        }
    }
    Ok(result)
}
/// Computes the principal matrix square root A^(1/2).
///
/// Uses the Denman-Beavers iteration or Newton's method.
///
/// # Arguments
///
/// * `a` - Square matrix
///
/// # Returns
///
/// The principal matrix square root
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::sqrtm;
/// use oxiblas_matrix::Mat;
///
/// // sqrt(4*I) = 2*I
/// let a = Mat::from_rows(&[
///     &[4.0f64, 0.0],
///     &[0.0, 4.0],
/// ]);
/// let result = sqrtm(a.as_ref()).unwrap();
///
/// assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
/// assert!((result[(1, 1)] - 2.0).abs() < 1e-10);
/// ```
pub fn sqrtm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let val = a[(0, 0)];
        if !(val >= T::zero()) {
            return Err(MatFunError::NotDefined);
        }
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = Real::sqrt(val);
        return Ok(result);
    }
    sqrtm_denman_beavers(&a)
}
/// Computes the matrix power A^p for a general real power p.
///
/// For integer powers, uses binary exponentiation for accuracy.
/// For non-integer powers, computes exp(p * log(A)).
///
/// # Arguments
///
/// * `a` - Square matrix with positive eigenvalues (for non-integer powers)
/// * `p` - Real power exponent
///
/// # Returns
///
/// The matrix power A^p
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::powm;
/// use oxiblas_matrix::Mat;
///
/// // For diagonal matrix, A^p has diagonal entries raised to p
/// let a = Mat::from_rows(&[
///     &[4.0f64, 0.0],
///     &[0.0, 9.0],
/// ]);
/// let result = powm(a.as_ref(), 0.5).unwrap();
///
/// // Should be sqrt: 2 and 3 (with numerical tolerance)
/// assert!((result[(0, 0)] - 2.0).abs() < 0.1);
/// assert!((result[(1, 1)] - 3.0).abs() < 0.1);
/// ```
pub fn powm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    p: T,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let val = a[(0, 0)];
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = Float::powf(val, p);
        return Ok(result);
    }
    let p_f64 = p.to_f64().unwrap_or(0.0);
    let p_rounded = p_f64.round();
    let is_integer = (p_f64 - p_rounded).abs() < 1e-14;
    if is_integer {
        let p_int = p_rounded as i64;
        powm_integer(a, p_int)
    } else {
        let log_a = logm(a)?;
        let mut scaled = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                scaled[(i, j)] = p * log_a[(i, j)];
            }
        }
        expm(scaled.as_ref())
    }
}
/// Internal helper: Matrix power for integer exponent using binary exponentiation.
fn powm_integer<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    p: i64,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if p == 0 {
        return Ok(Mat::eye(n));
    }
    let is_negative = p < 0;
    let mut exp = p.unsigned_abs();
    let mut base = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            base[(i, j)] = a[(i, j)];
        }
    }
    if is_negative {
        base = matrix_inverse(&base)?;
    }
    let mut result = Mat::<T>::eye(n);
    while exp > 0 {
        if exp & 1 == 1 {
            result = mat_mult(&result, &base);
        }
        base = mat_mult(&base, &base);
        exp >>= 1;
    }
    Ok(result)
}
/// Computes the matrix sign function sign(A).
///
/// The matrix sign function is defined as:
/// - sign(A) = A * (A^2)^(-1/2) for nonsingular A
/// - Or equivalently computed via Newton iteration
///
/// This function uses the Newton iteration:
/// X_{k+1} = (X_k + X_k^{-1}) / 2
///
/// which converges quadratically to sign(A).
///
/// # Arguments
///
/// * `a` - Square nonsingular matrix
///
/// # Returns
///
/// The matrix sign function
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::signm;
/// use oxiblas_matrix::Mat;
///
/// // For positive definite matrix, sign(A) = I
/// let a = Mat::from_rows(&[
///     &[4.0f64, 0.0],
///     &[0.0, 9.0],
/// ]);
/// let result = signm(a.as_ref()).unwrap();
///
/// // Should be identity for positive definite matrix
/// assert!((result[(0, 0)] - 1.0).abs() < 1e-8);
/// assert!((result[(1, 1)] - 1.0).abs() < 1e-8);
/// ```
pub fn signm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let val = a[(0, 0)];
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = if val > T::zero() {
            T::one()
        } else if val < T::zero() {
            T::zero() - T::one()
        } else {
            return Err(MatFunError::NotDefined);
        };
        return Ok(result);
    }
    signm_newton(&a)
}
/// Internal helper: Newton iteration for matrix sign.
///
/// Uses the iteration X_{k+1} = (X_k + X_k^{-1}) / 2 with scaling
/// (determinantal scaling for numerical stability).
fn signm_newton<T: Field + Real + bytemuck::Zeroable>(
    a: &MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    let max_iter = 100;
    let eps: T = <T as Float>::epsilon();
    let tol = eps * from_f64::<T>(1000.0);
    let mut x = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            x[(i, j)] = a[(i, j)];
        }
    }
    for _ in 0..max_iter {
        let x_inv = match matrix_inverse(&x) {
            Ok(m) => m,
            Err(_) => return Err(MatFunError::NotConverged),
        };
        let det_x = compute_determinant(&x);
        let abs_det = Float::abs(det_x);
        let n_f64 = n as f64;
        let mu = if abs_det > eps {
            let det_f64 = abs_det.to_f64().unwrap_or(1.0);
            let mu_f64 = det_f64.powf(-1.0 / n_f64);
            from_f64::<T>(mu_f64)
        } else {
            T::one()
        };
        let half = T::one() / (T::one() + T::one());
        let mu_inv = T::one() / mu;
        let mut x_new = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                x_new[(i, j)] = (mu * x[(i, j)] + mu_inv * x_inv[(i, j)]) * half;
            }
        }
        let diff = matrix_diff_norm(&x, &x_new);
        let norm_x = matrix_frob_norm(&x);
        x = x_new;
        if diff < tol * norm_x {
            return Ok(x);
        }
    }
    Ok(x)
}
/// Computes the matrix cosine cos(A) using the relation cos(A) = (exp(iA) + exp(-iA))/2.
///
/// For real matrices, this is computed via the real Schur form to avoid complex arithmetic.
///
/// # Arguments
///
/// * `a` - Square matrix
///
/// # Returns
///
/// The matrix cosine cos(A)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::cosm;
/// use oxiblas_matrix::Mat;
///
/// // cos(0) = I
/// let zero = Mat::<f64>::zeros(2, 2);
/// let result = cosm(zero.as_ref()).unwrap();
///
/// assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
/// assert!((result[(1, 1)] - 1.0).abs() < 1e-10);
/// ```
pub fn cosm<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = Real::cos(a[(0, 0)]);
        return Ok(result);
    }
    cosm_pade(&a)
}
/// Computes the matrix sine sin(A) using the relation sin(A) = (exp(iA) - exp(-iA))/(2i).
///
/// # Arguments
///
/// * `a` - Square matrix
///
/// # Returns
///
/// The matrix sine sin(A)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::sinm;
/// use oxiblas_matrix::Mat;
///
/// // sin(0) = 0
/// let zero = Mat::<f64>::zeros(2, 2);
/// let result = sinm(zero.as_ref()).unwrap();
///
/// assert!(result[(0, 0)].abs() < 1e-10);
/// assert!(result[(1, 1)].abs() < 1e-10);
/// ```
pub fn sinm<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n == 1 {
        let mut result = Mat::zeros(1, 1);
        result[(0, 0)] = Real::sin(a[(0, 0)]);
        return Ok(result);
    }
    sinm_pade(&a)
}
/// Internal helper: Padé approximation for matrix cosine.
fn cosm_pade<T: Field + Real + bytemuck::Zeroable>(
    a: &MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    let norm = matrix_1_norm(a);
    let ln2: T = Float::ln(T::one() + T::one());
    let log2_norm: T = Float::ln(norm + T::one()) / ln2;
    let s = max_int(T::zero(), log2_norm);
    let two = T::one() + T::one();
    let neg_s = from_f64::<T>(-(s as f64));
    let scale_factor: T = Float::powf(two, neg_s);
    let mut scaled = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            scaled[(i, j)] = a[(i, j)] * scale_factor;
        }
    }
    let a2 = mat_mult(&scaled, &scaled);
    let a4 = mat_mult(&a2, &a2);
    let a6 = mat_mult(&a4, &a2);
    let a8 = mat_mult(&a6, &a2);
    let mut result = Mat::<T>::eye(n);
    let inv_2 = from_f64::<T>(1.0 / 2.0);
    let inv_24 = from_f64::<T>(1.0 / 24.0);
    let inv_720 = from_f64::<T>(1.0 / 720.0);
    let inv_40320 = from_f64::<T>(1.0 / 40320.0);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = result[(i, j)] - a2[(i, j)] * inv_2 + a4[(i, j)] * inv_24
                - a6[(i, j)] * inv_720
                + a8[(i, j)] * inv_40320;
        }
    }
    for _ in 0..s {
        let result_sq = mat_mult(&result, &result);
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] = two * result_sq[(i, j)];
            }
            result[(i, i)] = result[(i, i)] - T::one();
        }
    }
    Ok(result)
}
/// Internal helper: Padé approximation for matrix sine.
fn sinm_pade<T: Field + Real + bytemuck::Zeroable>(
    a: &MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    let norm = matrix_1_norm(a);
    let ln2: T = Float::ln(T::one() + T::one());
    let log2_norm: T = Float::ln(norm + T::one()) / ln2;
    let s = max_int(T::zero(), log2_norm);
    let two = T::one() + T::one();
    let neg_s = from_f64::<T>(-(s as f64));
    let scale_factor: T = Float::powf(two, neg_s);
    let mut scaled = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            scaled[(i, j)] = a[(i, j)] * scale_factor;
        }
    }
    let a2 = mat_mult(&scaled, &scaled);
    let a3 = mat_mult(&a2, &scaled);
    let a5 = mat_mult(&a3, &a2);
    let a7 = mat_mult(&a5, &a2);
    let inv_6 = from_f64::<T>(1.0 / 6.0);
    let inv_120 = from_f64::<T>(1.0 / 120.0);
    let inv_5040 = from_f64::<T>(1.0 / 5040.0);
    let mut sin_scaled = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sin_scaled[(i, j)] =
                scaled[(i, j)] - a3[(i, j)] * inv_6 + a5[(i, j)] * inv_120 - a7[(i, j)] * inv_5040;
        }
    }
    let mut cos_scaled = Mat::<T>::eye(n);
    let a4 = mat_mult(&a2, &a2);
    let a6 = mat_mult(&a4, &a2);
    let inv_2 = from_f64::<T>(1.0 / 2.0);
    let inv_24 = from_f64::<T>(1.0 / 24.0);
    let inv_720 = from_f64::<T>(1.0 / 720.0);
    for i in 0..n {
        for j in 0..n {
            cos_scaled[(i, j)] = cos_scaled[(i, j)] - a2[(i, j)] * inv_2 + a4[(i, j)] * inv_24
                - a6[(i, j)] * inv_720;
        }
    }
    let mut sin_result = sin_scaled;
    let mut cos_result = cos_scaled;
    for _ in 0..s {
        let new_sin = mat_mult(&sin_result, &cos_result);
        let cos_sq = mat_mult(&cos_result, &cos_result);
        for i in 0..n {
            for j in 0..n {
                sin_result[(i, j)] = two * new_sin[(i, j)];
                cos_result[(i, j)] = two * cos_sq[(i, j)];
            }
            cos_result[(i, i)] = cos_result[(i, i)] - T::one();
        }
    }
    Ok(sin_result)
}
/// Helper: Compute determinant via LU decomposition.
fn compute_determinant<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>) -> T {
    match crate::lu::Lu::compute(a.as_ref()) {
        Ok(lu) => lu.determinant(),
        Err(_) => T::zero(),
    }
}
/// Helper: Frobenius norm.
fn matrix_frob_norm<T: Field + Real>(a: &Mat<T>) -> T {
    let mut sum = T::zero();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum = sum + a[(i, j)] * a[(i, j)];
        }
    }
    Real::sqrt(sum)
}
/// Helper: from_f64 with unwrap
#[inline]
fn from_f64<T: Field>(val: f64) -> T {
    num_traits::FromPrimitive::from_f64(val).unwrap_or_else(T::zero)
}
/// Internal helper: Newton's method for matrix square root.
fn sqrtm_newton<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    let max_iter = 100;
    let eps: T = <T as Float>::epsilon();
    let tol = eps * from_f64::<T>(100.0);
    let trace_a = matrix_trace(a);
    let n_t = from_f64::<T>(n as f64);
    let scale_val: T = Real::sqrt(trace_a / n_t);
    let scale = if scale_val > eps { scale_val } else { T::one() };
    let mut x = Mat::zeros(n, n);
    for i in 0..n {
        x[(i, i)] = scale;
    }
    for _ in 0..max_iter {
        let x_inv_a = match solve_linear(&x, a) {
            Ok(m) => m,
            Err(_) => return Err(MatFunError::NotConverged),
        };
        let mut x_new = Mat::zeros(n, n);
        let half = T::one() / (T::one() + T::one());
        for i in 0..n {
            for j in 0..n {
                x_new[(i, j)] = (x[(i, j)] + x_inv_a[(i, j)]) * half;
            }
        }
        let diff = matrix_diff_norm(&x, &x_new);
        x = x_new;
        if diff < tol {
            return Ok(x);
        }
    }
    Ok(x)
}
/// Internal helper: Denman-Beavers iteration for matrix square root.
fn sqrtm_denman_beavers<T: Field + Real + bytemuck::Zeroable>(
    a: &MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    let max_iter = 100;
    let eps: T = <T as Float>::epsilon();
    let tol = eps * from_f64::<T>(100.0);
    let mut y = Mat::zeros(n, n);
    let mut z = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            y[(i, j)] = a[(i, j)];
        }
        z[(i, i)] = T::one();
    }
    for _ in 0..max_iter {
        let z_inv = match matrix_inverse(&z) {
            Ok(m) => m,
            Err(_) => return Err(MatFunError::NotConverged),
        };
        let y_inv = match matrix_inverse(&y) {
            Ok(m) => m,
            Err(_) => return Err(MatFunError::NotConverged),
        };
        let mut y_new = Mat::zeros(n, n);
        let mut z_new = Mat::zeros(n, n);
        let half = T::one() / (T::one() + T::one());
        for i in 0..n {
            for j in 0..n {
                y_new[(i, j)] = (y[(i, j)] + z_inv[(i, j)]) * half;
                z_new[(i, j)] = (z[(i, j)] + y_inv[(i, j)]) * half;
            }
        }
        let diff = matrix_diff_norm(&y, &y_new);
        y = y_new;
        z = z_new;
        if diff < tol {
            return Ok(y);
        }
    }
    Ok(y)
}
/// Computes the Padé approximant for exp(A).
fn pade_approximant<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>, order: usize) -> Mat<T> {
    let n = a.nrows();
    let mut a_powers = vec![Mat::<T>::eye(n)];
    let mut a_k = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_k[(i, j)] = a[(i, j)];
        }
    }
    a_powers.push(a_k.clone());
    for _ in 2..=order {
        let next = mat_mult(
            a_powers.last().expect("collection should be non-empty"),
            &a_powers[1],
        );
        a_powers.push(next);
    }
    let mut num = Mat::zeros(n, n);
    let mut den = Mat::zeros(n, n);
    for k in 0..=order {
        let coeff_num: T = pade_coeff(order, k);
        let sign = if k % 2 == 0 {
            T::one()
        } else {
            T::zero() - T::one()
        };
        let coeff_den: T = coeff_num * sign;
        for i in 0..n {
            for j in 0..n {
                num[(i, j)] = num[(i, j)] + coeff_num * a_powers[k][(i, j)];
                den[(i, j)] = den[(i, j)] + coeff_den * a_powers[k][(i, j)];
            }
        }
    }
    match solve_linear(&den, &num) {
        Ok(result) => result,
        Err(_) => num,
    }
}
/// Computes Padé coefficient.
/// c_k = (2p - k)! * p! / ((2p)! * (p - k)! * k!)
/// Using recurrence: c_k = c_{k-1} * (p - k + 1) / (k * (2p - k + 1))
fn pade_coeff<T: Field + Real>(p: usize, k: usize) -> T {
    let mut result = T::one();
    for i in 1..=k {
        let num = from_f64::<T>((p - i + 1) as f64);
        let den = from_f64::<T>((i * (2 * p - i + 1)) as f64);
        result = result * num / den;
    }
    result
}
/// Adaptive Taylor series for the matrix logarithm log(I + X), valid for a
/// small argument X.
///
/// Sums the series
///
/// ```text
/// log(I + X) = X - X²/2 + X³/3 - X⁴/4 + ...
/// ```
///
/// accumulating terms X^k * (-1)^(k+1) / k until the incremental contribution
/// falls below machine precision relative to the accumulated result (or a
/// generous safety cap is reached). When the caller keeps ||X|| small via the
/// inverse scaling-and-squaring reduction, the terms decay geometrically and
/// the sum reaches near machine precision after only a handful of iterations —
/// unlike a fixed low-order truncation, which leaves a ~||X||^9 residual.
fn log1p_taylor<T: Field + Real + bytemuck::Zeroable>(x: &Mat<T>) -> Mat<T> {
    let n = x.nrows();
    let eps: T = <T as Float>::epsilon();
    let mut result = Mat::zeros(n, n);
    // First term (k = 1): + X.
    let mut term = x.clone();
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = result[(i, j)] + term[(i, j)];
        }
    }
    // Safety cap: for ||X|| < 1 the series converges long before this bound;
    // the cap only guards against a pathological (non-converging) argument.
    let max_terms = 60usize;
    for k in 2..=max_terms {
        // term := X^k (previous term multiplied by X).
        term = mat_mult(&term, x);
        let sign = if k % 2 == 0 {
            T::zero() - T::one()
        } else {
            T::one()
        };
        let coeff = sign / from_f64::<T>(k as f64);
        let mut add_sq = T::zero();
        for i in 0..n {
            for j in 0..n {
                let contrib = coeff * term[(i, j)];
                result[(i, j)] = result[(i, j)] + contrib;
                add_sq = add_sq + contrib * contrib;
            }
        }
        // Stop once the newly added term is negligible relative to the sum.
        // For ||X|| < 1 the geometric tail beyond this term is bounded by a
        // small multiple of it, so the truncation error is ~eps * ||result||.
        let add_norm = Real::sqrt(add_sq);
        let res_norm = matrix_frob_norm(&result);
        if add_norm <= eps * res_norm {
            break;
        }
    }
    result
}
/// Helper: Matrix 1-norm.
fn matrix_1_norm<T: Field + Real>(a: &MatRef<'_, T>) -> T {
    let n = a.ncols();
    let mut max_col_sum = T::zero();
    for j in 0..n {
        let mut col_sum = T::zero();
        for i in 0..a.nrows() {
            col_sum = col_sum + Float::abs(a[(i, j)]);
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}
/// Helper: ||A - I||_1.
fn matrix_1_norm_minus_identity<T: Field + Real>(a: &Mat<T>) -> T {
    let n = a.ncols();
    let mut max_col_sum = T::zero();
    for j in 0..n {
        let mut col_sum = T::zero();
        for i in 0..a.nrows() {
            let val = if i == j {
                a[(i, j)] - T::one()
            } else {
                a[(i, j)]
            };
            col_sum = col_sum + Float::abs(val);
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}
/// Helper: max(0, x) as integer.
fn max_int<T: Real>(zero: T, x: T) -> usize {
    if x > zero {
        let val: f64 = x.to_f64().unwrap_or(0.0);
        val.ceil() as usize
    } else {
        0
    }
}
/// Helper: Matrix trace.
fn matrix_trace<T: Field>(a: &Mat<T>) -> T {
    let n = a.nrows().min(a.ncols());
    let mut tr = T::zero();
    for i in 0..n {
        tr = tr + a[(i, i)];
    }
    tr
}
/// Helper: Frobenius norm of difference.
fn matrix_diff_norm<T: Field + Real>(a: &Mat<T>, b: &Mat<T>) -> T {
    let mut sum = T::zero();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = a[(i, j)] - b[(i, j)];
            sum = sum + diff * diff;
        }
    }
    Real::sqrt(sum)
}
/// Helper: Matrix multiplication.
fn mat_mult<T: Field + bytemuck::Zeroable>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();
    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}
/// Helper: Solve A*X = B.
fn solve_linear<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    b: &Mat<T>,
) -> Result<Mat<T>, MatFunError> {
    let lu = crate::lu::Lu::compute(a.as_ref()).map_err(|_| MatFunError::NotConverged)?;
    let mut result = Mat::zeros(b.nrows(), b.ncols());
    for j in 0..b.ncols() {
        let mut col = Mat::zeros(b.nrows(), 1);
        for i in 0..b.nrows() {
            col[(i, 0)] = b[(i, j)];
        }
        let sol = lu
            .solve(col.as_ref())
            .map_err(|_| MatFunError::NotConverged)?;
        for i in 0..b.nrows() {
            result[(i, j)] = sol[(i, 0)];
        }
    }
    Ok(result)
}
/// Helper: Matrix inverse via LU.
fn matrix_inverse<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    let eye = Mat::<T>::eye(n);
    solve_linear(a, &eye)
}
/// Computes the Frechet derivative of the matrix exponential.
///
/// The Frechet derivative L_exp(A, E) satisfies:
/// exp(A + ε·E) ≈ exp(A) + ε·L_exp(A, E) + O(ε²)
///
/// This is computed using the block matrix method:
/// ```text
/// expm([A E; 0 A]) = [exp(A)  L_exp(A,E); 0  exp(A)]
/// ```
///
/// # Arguments
///
/// * `a` - Square matrix A
/// * `e` - Direction matrix E (same size as A)
///
/// # Returns
///
/// A tuple (exp(A), L_exp(A, E))
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::frechet_expm;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
/// let e = Mat::from_rows(&[&[0.1f64, 0.0], &[0.0, 0.1]]);
///
/// let (exp_a, frechet) = frechet_expm(a.as_ref(), e.as_ref()).unwrap();
/// ```
pub fn frechet_expm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    e: MatRef<'_, T>,
) -> Result<(Mat<T>, Mat<T>), MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(MatFunError::NotSquare {
            nrows: e.nrows(),
            ncols: e.ncols(),
        });
    }
    let mut block = Mat::<T>::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            block[(i, j)] = a[(i, j)];
        }
    }
    for i in 0..n {
        for j in 0..n {
            block[(i, n + j)] = e[(i, j)];
        }
    }
    for i in 0..n {
        for j in 0..n {
            block[(n + i, n + j)] = a[(i, j)];
        }
    }
    let exp_block = expm(block.as_ref())?;
    let mut exp_a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            exp_a[(i, j)] = exp_block[(i, j)];
        }
    }
    let mut frechet = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            frechet[(i, j)] = exp_block[(i, n + j)];
        }
    }
    Ok((exp_a, frechet))
}
/// Computes the Frechet derivative of the matrix logarithm.
///
/// The Frechet derivative L_log(A, E) satisfies:
/// log(A + ε·E) ≈ log(A) + ε·L_log(A, E) + O(ε²)
///
/// This is computed by solving the Sylvester equation:
/// A·L + L·A = E + A·log(A) - log(A)·A
/// when A is diagonalizable with positive eigenvalues.
///
/// # Arguments
///
/// * `a` - Square matrix A with positive eigenvalues
/// * `e` - Direction matrix E (same size as A)
///
/// # Returns
///
/// A tuple (log(A), L_log(A, E))
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::frechet_logm;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 3.0]]);
/// let e = Mat::from_rows(&[&[0.1f64, 0.0], &[0.0, 0.1]]);
///
/// let (log_a, frechet) = frechet_logm(a.as_ref(), e.as_ref()).unwrap();
/// ```
pub fn frechet_logm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    e: MatRef<'_, T>,
) -> Result<(Mat<T>, Mat<T>), MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(MatFunError::NotSquare {
            nrows: e.nrows(),
            ncols: e.ncols(),
        });
    }
    if n == 1 {
        let a_val = a[(0, 0)];
        let e_val = e[(0, 0)];
        if a_val <= T::zero() {
            return Err(MatFunError::NotDefined);
        }
        let log_a_val = Real::ln(a_val);
        let frechet_val = e_val / a_val;
        let mut log_a = Mat::zeros(1, 1);
        let mut frechet = Mat::zeros(1, 1);
        log_a[(0, 0)] = log_a_val;
        frechet[(0, 0)] = frechet_val;
        return Ok((log_a, frechet));
    }
    let log_a = logm(a)?;
    let eps = Real::sqrt(<T as Scalar>::epsilon());
    let mut a_plus = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a_plus[(i, j)] = a[(i, j)] + eps * e[(i, j)];
        }
    }
    let log_a_plus = logm(a_plus.as_ref())?;
    let mut frechet = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            frechet[(i, j)] = (log_a_plus[(i, j)] - log_a[(i, j)]) / eps;
        }
    }
    Ok((log_a, frechet))
}
/// Computes the Frechet derivative of the matrix square root.
///
/// The Frechet derivative L_sqrt(A, E) satisfies:
/// sqrt(A + ε·E) ≈ sqrt(A) + ε·L_sqrt(A, E) + O(ε²)
///
/// For positive definite A, L_sqrt(A, E) is the unique solution of the
/// Sylvester equation:
/// sqrt(A)·L + L·sqrt(A) = E
///
/// # Arguments
///
/// * `a` - Square positive (semi-)definite matrix A
/// * `e` - Direction matrix E (same size as A)
///
/// # Returns
///
/// A tuple (sqrt(A), L_sqrt(A, E))
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::frechet_sqrtm;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 9.0]]);
/// let e = Mat::from_rows(&[&[0.1f64, 0.0], &[0.0, 0.1]]);
///
/// let (sqrt_a, frechet) = frechet_sqrtm(a.as_ref(), e.as_ref()).unwrap();
/// ```
pub fn frechet_sqrtm<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    e: MatRef<'_, T>,
) -> Result<(Mat<T>, Mat<T>), MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(MatFunError::NotSquare {
            nrows: e.nrows(),
            ncols: e.ncols(),
        });
    }
    if n == 1 {
        let a_val = a[(0, 0)];
        let e_val = e[(0, 0)];
        if a_val < T::zero() {
            return Err(MatFunError::NotDefined);
        }
        let sqrt_a_val = Real::sqrt(a_val);
        let frechet_val = if sqrt_a_val > T::zero() {
            e_val / (T::from_f64(2.0).unwrap_or(T::one()) * sqrt_a_val)
        } else {
            T::zero()
        };
        let mut sqrt_a = Mat::zeros(1, 1);
        let mut frechet = Mat::zeros(1, 1);
        sqrt_a[(0, 0)] = sqrt_a_val;
        frechet[(0, 0)] = frechet_val;
        return Ok((sqrt_a, frechet));
    }
    let sqrt_a = sqrtm(a)?;
    let frechet = solve_sylvester_same(&sqrt_a, e)?;
    Ok((sqrt_a, frechet))
}
/// Solves the Sylvester equation A·X + X·A = C where A and C are n×n.
///
/// This is a special case of the Sylvester equation with B = A.
fn solve_sylvester_same<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    c: MatRef<'_, T>,
) -> Result<Mat<T>, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Mat::zeros(0, 0));
    }
    let n2 = n * n;
    let mut kron_sum = Mat::<T>::zeros(n2, n2);
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                kron_sum[(k * n + i, k * n + j)] = kron_sum[(k * n + i, k * n + j)] + a[(i, j)];
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                kron_sum[(i * n + k, j * n + k)] = kron_sum[(i * n + k, j * n + k)] + a[(i, j)];
            }
        }
    }
    let mut vec_c = Mat::zeros(n2, 1);
    for i in 0..n {
        for j in 0..n {
            vec_c[(j * n + i, 0)] = c[(i, j)];
        }
    }
    let vec_x = solve_linear(&kron_sum, &vec_c)?;
    let mut x = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            x[(i, j)] = vec_x[(j * n + i, 0)];
        }
    }
    Ok(x)
}
/// Computes the condition number of the matrix exponential at A.
///
/// The condition number measures the sensitivity of exp(A) to perturbations in A.
/// cond(exp, A) = ||L_exp(A, ·)|| / ||exp(A)||
///
/// This is approximated by computing the Frobenius norms of exp(A) and L_exp(A, E)
/// for a random direction E with ||E||_F = 1.
///
/// # Arguments
///
/// * `a` - Square matrix A
///
/// # Returns
///
/// The estimated condition number
pub fn cond_expm<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, MatFunError> {
    let n = a.nrows();
    if n == 0 {
        return Err(MatFunError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(MatFunError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    let mut e = Mat::<T>::eye(n);
    let e_norm = T::from_f64((n as f64).sqrt()).unwrap_or(T::one());
    for i in 0..n {
        e[(i, i)] = e[(i, i)] / e_norm;
    }
    let (exp_a, frechet) = frechet_expm(a, e.as_ref())?;
    let exp_norm = frobenius_norm(&exp_a);
    let frechet_norm = frobenius_norm(&frechet);
    if exp_norm > T::zero() {
        Ok(frechet_norm / exp_norm)
    } else {
        Ok(T::one())
    }
}
/// Computes the Frobenius norm of a matrix.
fn frobenius_norm<T: Field + Real>(a: &Mat<T>) -> T {
    let mut sum = T::zero();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let val = a[(i, j)];
            sum = sum + val * val;
        }
    }
    Real::sqrt(sum)
}
use oxiblas_core::scalar::Scalar;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expm_identity() {
        let eye = Mat::<f64>::eye(2);
        let result = expm(eye.as_ref()).unwrap();
        let e = std::f64::consts::E;
        assert!((result[(0, 0)] - e).abs() < 1e-10);
        assert!((result[(1, 1)] - e).abs() < 1e-10);
        assert!(result[(0, 1)].abs() < 1e-10);
        assert!(result[(1, 0)].abs() < 1e-10);
    }
    #[test]
    fn test_expm_zero() {
        let zero = Mat::<f64>::zeros(2, 2);
        let result = expm(zero.as_ref()).unwrap();
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 1.0).abs() < 1e-10);
        assert!(result[(0, 1)].abs() < 1e-10);
    }
    #[test]
    fn test_expm_diagonal() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
        let result = expm(a.as_ref()).unwrap();
        let e1 = std::f64::consts::E;
        let e2 = e1 * e1;
        assert!((result[(0, 0)] - e1).abs() < 1e-10);
        assert!((result[(1, 1)] - e2).abs() < 1e-10);
    }
    #[test]
    fn test_expm_1x1() {
        let a = Mat::from_rows(&[&[2.0f64]]);
        let result = expm(a.as_ref()).unwrap();
        let expected = 2.0_f64.exp();
        assert!((result[(0, 0)] - expected).abs() < 1e-10);
    }
    #[test]
    fn test_sqrtm_identity() {
        let eye = Mat::<f64>::eye(2);
        let result = sqrtm(eye.as_ref()).unwrap();
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_sqrtm_diagonal() {
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 9.0]]);
        let result = sqrtm(a.as_ref()).unwrap();
        assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 3.0).abs() < 1e-10);
    }
    #[test]
    fn test_sqrtm_squared() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 2.0]]);
        let sqrt_a = sqrtm(a.as_ref()).unwrap();
        let a_back = mat_mult(&sqrt_a, &sqrt_a);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[(i, j)] - a_back[(i, j)]).abs() < 1e-8,
                    "sqrtm squared failed at ({}, {}): {} vs {}",
                    i,
                    j,
                    a[(i, j)],
                    a_back[(i, j)]
                );
            }
        }
    }
    #[test]
    fn test_logm_exp() {
        let e = std::f64::consts::E;
        let a = Mat::from_rows(&[&[e, 0.0], &[0.0, e * e]]);
        let result = logm(a.as_ref()).unwrap();
        assert!(
            (result[(0, 0)] - 1.0).abs() < 1e-4,
            "log(e) = {}, expected 1.0",
            result[(0, 0)]
        );
        assert!(
            (result[(1, 1)] - 2.0).abs() < 1e-4,
            "log(e^2) = {}, expected 2.0",
            result[(1, 1)]
        );
    }
    #[test]
    fn test_logm_identity() {
        let eye = Mat::<f64>::eye(2);
        let result = logm(eye.as_ref()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(result[(i, j)].abs() < 1e-8);
            }
        }
    }
    #[test]
    fn test_expm_logm_inverse() {
        let a = Mat::from_rows(&[&[0.5f64, 0.1], &[0.1, 0.3]]);
        let exp_a = expm(a.as_ref()).unwrap();
        let log_exp_a = logm(exp_a.as_ref()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[(i, j)] - log_exp_a[(i, j)]).abs() < 1e-4,
                    "log(exp(A)) != A at ({}, {}): {} vs {}",
                    i,
                    j,
                    a[(i, j)],
                    log_exp_a[(i, j)]
                );
            }
        }
    }
    #[test]
    fn test_powm_zero_power() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let result = powm(a.as_ref(), 0.0).unwrap();
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 1.0).abs() < 1e-10);
        assert!(result[(0, 1)].abs() < 1e-10);
        assert!(result[(1, 0)].abs() < 1e-10);
    }
    #[test]
    fn test_powm_power_one() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let result = powm(a.as_ref(), 1.0).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[(i, j)] - a[(i, j)]).abs() < 1e-10,
                    "A^1 != A at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
    #[test]
    fn test_powm_power_two() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let result = powm(a.as_ref(), 2.0).unwrap();
        let expected = mat_mult(&a, &a);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[(i, j)] - expected[(i, j)]).abs() < 1e-10,
                    "A^2 != A*A at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
    #[test]
    fn test_powm_negative_power() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);
        let result = powm(a.as_ref(), -1.0).unwrap();
        assert!((result[(0, 0)] - 0.5).abs() < 1e-10);
        assert!((result[(1, 1)] - 0.25).abs() < 1e-10);
    }
    #[test]
    fn test_powm_half_power() {
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 9.0]]);
        let result = powm(a.as_ref(), 0.5).unwrap();
        assert!(
            (result[(0, 0)] - 2.0).abs() < 1e-4,
            "sqrt(4) = {}, expected 2.0",
            result[(0, 0)]
        );
        assert!(
            (result[(1, 1)] - 3.0).abs() < 1e-4,
            "sqrt(9) = {}, expected 3.0",
            result[(1, 1)]
        );
    }
    #[test]
    fn test_powm_1x1() {
        let a = Mat::from_rows(&[&[8.0f64]]);
        let result = powm(a.as_ref(), 1.0 / 3.0).unwrap();
        assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
    }
    #[test]
    fn test_signm_positive_diagonal() {
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 9.0]]);
        let result = signm(a.as_ref()).unwrap();
        assert!(
            (result[(0, 0)] - 1.0).abs() < 1e-8,
            "sign of positive matrix should be 1"
        );
        assert!(
            (result[(1, 1)] - 1.0).abs() < 1e-8,
            "sign of positive matrix should be 1"
        );
    }
    #[test]
    fn test_signm_negative_diagonal() {
        let a = Mat::from_rows(&[&[-4.0f64, 0.0], &[0.0, -9.0]]);
        let result = signm(a.as_ref()).unwrap();
        assert!(
            (result[(0, 0)] + 1.0).abs() < 1e-8,
            "sign of negative matrix should be -1"
        );
        assert!(
            (result[(1, 1)] + 1.0).abs() < 1e-8,
            "sign of negative matrix should be -1"
        );
    }
    #[test]
    fn test_signm_1x1() {
        let a = Mat::from_rows(&[&[5.0f64]]);
        let result = signm(a.as_ref()).unwrap();
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        let b = Mat::from_rows(&[&[-3.0f64]]);
        let result_b = signm(b.as_ref()).unwrap();
        assert!((result_b[(0, 0)] + 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_signm_squared() {
        let a = Mat::from_rows(&[&[3.0f64, 1.0], &[0.0, 2.0]]);
        let sign_a = signm(a.as_ref()).unwrap();
        let sign_a_sq = mat_mult(&sign_a, &sign_a);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sign_a_sq[(i, j)] - expected).abs() < 1e-6,
                    "sign(A)^2 != I at ({}, {}): {} vs {}",
                    i,
                    j,
                    sign_a_sq[(i, j)],
                    expected
                );
            }
        }
    }
    #[test]
    fn test_cosm_zero() {
        let zero = Mat::<f64>::zeros(2, 2);
        let result = cosm(zero.as_ref()).unwrap();
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 1.0).abs() < 1e-10);
        assert!(result[(0, 1)].abs() < 1e-10);
        assert!(result[(1, 0)].abs() < 1e-10);
    }
    #[test]
    fn test_cosm_1x1() {
        let a = Mat::from_rows(&[&[std::f64::consts::PI]]);
        let result = cosm(a.as_ref()).unwrap();
        assert!((result[(0, 0)] + 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_cosm_diagonal() {
        let pi = std::f64::consts::PI;
        let a = Mat::from_rows(&[&[0.0f64, 0.0], &[0.0, pi / 2.0]]);
        let result = cosm(a.as_ref()).unwrap();
        assert!(
            (result[(0, 0)] - 1.0).abs() < 1e-10,
            "cos(0) = {}, expected 1.0",
            result[(0, 0)]
        );
        assert!(
            result[(1, 1)].abs() < 1e-8,
            "cos(π/2) = {}, expected 0.0",
            result[(1, 1)]
        );
    }
    #[test]
    fn test_sinm_zero() {
        let zero = Mat::<f64>::zeros(2, 2);
        let result = sinm(zero.as_ref()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(result[(i, j)].abs() < 1e-10);
            }
        }
    }
    #[test]
    fn test_sinm_1x1() {
        let a = Mat::from_rows(&[&[std::f64::consts::PI / 2.0]]);
        let result = sinm(a.as_ref()).unwrap();
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_sinm_diagonal() {
        let pi = std::f64::consts::PI;
        let a = Mat::from_rows(&[&[0.0f64, 0.0], &[0.0, pi / 6.0]]);
        let result = sinm(a.as_ref()).unwrap();
        assert!(
            result[(0, 0)].abs() < 1e-10,
            "sin(0) = {}, expected 0.0",
            result[(0, 0)]
        );
        assert!(
            (result[(1, 1)] - 0.5).abs() < 1e-8,
            "sin(π/6) = {}, expected 0.5",
            result[(1, 1)]
        );
    }
    #[test]
    fn test_sin_cos_identity() {
        let a = Mat::from_rows(&[&[0.5f64, 0.0], &[0.0, 1.0]]);
        let sin_a = sinm(a.as_ref()).unwrap();
        let cos_a = cosm(a.as_ref()).unwrap();
        let sin_sq = mat_mult(&sin_a, &sin_a);
        let cos_sq = mat_mult(&cos_a, &cos_a);
        for i in 0..2 {
            for j in 0..2 {
                let sum = sin_sq[(i, j)] + cos_sq[(i, j)];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-6,
                    "sin^2 + cos^2 != I at ({}, {}): {} vs {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }
    #[test]
    fn test_frechet_expm_1x1() {
        let a = Mat::from_rows(&[&[2.0f64]]);
        let e = Mat::from_rows(&[&[1.0f64]]);
        let (exp_a, frechet) = frechet_expm(a.as_ref(), e.as_ref()).unwrap();
        let expected_exp = 2.0_f64.exp();
        assert!(
            (exp_a[(0, 0)] - expected_exp).abs() < 1e-10,
            "exp(2) = {}, expected {}",
            exp_a[(0, 0)],
            expected_exp
        );
        assert!(
            (frechet[(0, 0)] - expected_exp).abs() < 1e-8,
            "Frechet = {}, expected {}",
            frechet[(0, 0)],
            expected_exp
        );
    }
    #[test]
    fn test_frechet_expm_diagonal() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
        let e = Mat::from_rows(&[&[0.1f64, 0.0], &[0.0, 0.1]]);
        let (exp_a, frechet) = frechet_expm(a.as_ref(), e.as_ref()).unwrap();
        let e1 = std::f64::consts::E;
        let e2 = e1 * e1;
        assert!((exp_a[(0, 0)] - e1).abs() < 1e-10);
        assert!((exp_a[(1, 1)] - e2).abs() < 1e-10);
        assert!(
            (frechet[(0, 0)] - 0.1 * e1).abs() < 1e-8,
            "Frechet[0,0] = {}, expected {}",
            frechet[(0, 0)],
            0.1 * e1
        );
        assert!(
            (frechet[(1, 1)] - 0.1 * e2).abs() < 1e-8,
            "Frechet[1,1] = {}, expected {}",
            frechet[(1, 1)],
            0.1 * e2
        );
    }
    #[test]
    fn test_frechet_expm_finite_diff() {
        let a = Mat::from_rows(&[&[1.0f64, 0.5], &[0.0, 1.0]]);
        let e = Mat::from_rows(&[&[0.1f64, 0.1], &[0.1, 0.1]]);
        let (exp_a, frechet) = frechet_expm(a.as_ref(), e.as_ref()).unwrap();
        let eps = 1e-6;
        let mut a_plus = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                a_plus[(i, j)] = a[(i, j)] + eps * e[(i, j)];
            }
        }
        let exp_a_plus = expm(a_plus.as_ref()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let fd = (exp_a_plus[(i, j)] - exp_a[(i, j)]) / eps;
                assert!(
                    (frechet[(i, j)] - fd).abs() < 1e-4,
                    "Frechet[{},{}] = {}, finite diff = {}",
                    i,
                    j,
                    frechet[(i, j)],
                    fd
                );
            }
        }
    }
    #[test]
    fn test_frechet_sqrtm_1x1() {
        let a = Mat::from_rows(&[&[4.0f64]]);
        let e = Mat::from_rows(&[&[1.0f64]]);
        let (sqrt_a, frechet) = frechet_sqrtm(a.as_ref(), e.as_ref()).unwrap();
        assert!(
            (sqrt_a[(0, 0)] - 2.0).abs() < 1e-10,
            "sqrt(4) = {}, expected 2",
            sqrt_a[(0, 0)]
        );
        assert!(
            (frechet[(0, 0)] - 0.25).abs() < 1e-10,
            "Frechet = {}, expected 0.25",
            frechet[(0, 0)]
        );
    }
    #[test]
    fn test_frechet_sqrtm_diagonal() {
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 9.0]]);
        let e = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let (sqrt_a, frechet) = frechet_sqrtm(a.as_ref(), e.as_ref()).unwrap();
        assert!((sqrt_a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((sqrt_a[(1, 1)] - 3.0).abs() < 1e-10);
        assert!(
            (frechet[(0, 0)] - 0.25).abs() < 1e-8,
            "Frechet[0,0] = {}",
            frechet[(0, 0)]
        );
        assert!(
            (frechet[(1, 1)] - 1.0 / 6.0).abs() < 1e-8,
            "Frechet[1,1] = {}",
            frechet[(1, 1)]
        );
    }
    #[test]
    fn test_frechet_logm_1x1() {
        let a = Mat::from_rows(&[&[2.0f64]]);
        let e = Mat::from_rows(&[&[1.0f64]]);
        let (log_a, frechet) = frechet_logm(a.as_ref(), e.as_ref()).unwrap();
        let expected_log = 2.0_f64.ln();
        assert!(
            (log_a[(0, 0)] - expected_log).abs() < 1e-10,
            "log(2) = {}, expected {}",
            log_a[(0, 0)],
            expected_log
        );
        assert!(
            (frechet[(0, 0)] - 0.5).abs() < 1e-10,
            "Frechet = {}, expected 0.5",
            frechet[(0, 0)]
        );
    }
    /// Verifies logm accuracy via the round-trip exp(logm(A)) == A for a range
    /// of symmetric positive-definite matrices whose deviation ||A - I|| spans
    /// from small (0.01) up to and well beyond the previously-problematic 0.5.
    /// With the inverse scaling-and-squaring logm the relative error is now
    /// close to machine epsilon rather than the ~2e-4 of the old 8-term series.
    #[test]
    fn test_logm_accuracy_exp_of_logm() {
        // (label, matrix). All symmetric positive-definite so the principal
        // real logarithm is well defined.
        let cases: Vec<(&str, Mat<f64>)> = vec![
            // ||A - I||_1 = 0.015 (small argument regime).
            (
                "near_identity",
                Mat::from_rows(&[&[1.01f64, 0.005], &[0.005, 1.01]]),
            ),
            // ||A - I||_1 = 0.4.
            ("mid", Mat::from_rows(&[&[1.3f64, 0.1], &[0.1, 1.2]])),
            // ||A - I||_1 = 0.7 (beyond the old 0.5 threshold).
            (
                "beyond_half",
                Mat::from_rows(&[&[1.5f64, 0.2], &[0.2, 1.4]]),
            ),
            // ||A - I||_1 = 2.5 (needs several square-root reductions).
            ("large", Mat::from_rows(&[&[2.0f64, 0.5], &[0.5, 3.0]])),
            // Diagonal with eigenvalues far from 1 in both directions.
            (
                "wide_spectrum",
                Mat::from_rows(&[&[0.25f64, 0.0], &[0.0, 5.0]]),
            ),
        ];
        for (label, a) in &cases {
            let log_a = logm(a.as_ref()).unwrap();
            let recovered = expm(log_a.as_ref()).unwrap();
            let mut max_abs = 0.0f64;
            let mut a_norm = 0.0f64;
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    max_abs = max_abs.max((recovered[(i, j)] - a[(i, j)]).abs());
                    a_norm = a_norm.max(a[(i, j)].abs());
                }
            }
            let rel = max_abs / a_norm;
            assert!(
                rel < 1e-12,
                "case '{}': exp(logm(A)) relative error {} exceeds 1e-12",
                label,
                rel
            );
        }
    }

    /// Directly checks logm against known closed-form logs of diagonal matrices,
    /// where the old truncated series was ~2e-4 off near the 0.5 boundary.
    #[test]
    fn test_logm_diagonal_exact() {
        // exp(0.5) and exp(-0.5) give a matrix with ||A - I|| near the old
        // boundary; log must recover 0.5 and -0.5 to near machine precision.
        let a = Mat::from_rows(&[&[0.5f64.exp(), 0.0], &[0.0, (-0.5f64).exp()]]);
        let log_a = logm(a.as_ref()).unwrap();
        assert!(
            (log_a[(0, 0)] - 0.5).abs() < 1e-13,
            "log(exp(0.5)) = {}, expected 0.5",
            log_a[(0, 0)]
        );
        assert!(
            (log_a[(1, 1)] + 0.5).abs() < 1e-13,
            "log(exp(-0.5)) = {}, expected -0.5",
            log_a[(1, 1)]
        );
        assert!(log_a[(0, 1)].abs() < 1e-13);
        assert!(log_a[(1, 0)].abs() < 1e-13);
    }

    /// Round-trip in the other direction: logm(expm(A)) == A for a non-normal A
    /// whose exponential lands beyond the old 0.5 threshold.
    #[test]
    fn test_logm_of_expm_tight() {
        let a = Mat::from_rows(&[&[0.5f64, 0.3], &[0.1, 0.4]]);
        let exp_a = expm(a.as_ref()).unwrap();
        let log_exp_a = logm(exp_a.as_ref()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[(i, j)] - log_exp_a[(i, j)]).abs() < 1e-12,
                    "log(exp(A)) != A at ({}, {}): {} vs {}",
                    i,
                    j,
                    a[(i, j)],
                    log_exp_a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cond_expm_identity() {
        let eye = Mat::<f64>::eye(2);
        let cond = cond_expm(eye.as_ref()).unwrap();
        assert!(
            cond > 0.0 && cond < 10.0,
            "Condition number of exp(I) = {}",
            cond
        );
    }
}
