//! Matrix norms and trace.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::MatRef;

use crate::svd::{Svd, SvdError};

/// Computes the 1-norm (maximum column sum) of a matrix.
///
/// ||A||_1 = max_j Σ_i |a_ij|
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::norm_1;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, -2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let n1 = norm_1(a.as_ref());
/// // Column sums: |1|+|3|=4, |-2|+|4|=6
/// assert!((n1 - 6.0).abs() < 1e-10);
/// ```
pub fn norm_1<T: Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return T::zero();
    }

    let mut max_col_sum = T::zero();

    for j in 0..n {
        let mut col_sum = T::zero();
        for i in 0..m {
            col_sum = col_sum + Scalar::abs(a[(i, j)]);
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }

    max_col_sum
}

/// Computes the infinity norm (maximum row sum) of a matrix.
///
/// ||A||_∞ = max_i Σ_j |a_ij|
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::norm_inf;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, -2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let ninf = norm_inf(a.as_ref());
/// // Row sums: |1|+|-2|=3, |3|+|4|=7
/// assert!((ninf - 7.0).abs() < 1e-10);
/// ```
pub fn norm_inf<T: Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return T::zero();
    }

    let mut max_row_sum = T::zero();

    for i in 0..m {
        let mut row_sum = T::zero();
        for j in 0..n {
            row_sum = row_sum + Scalar::abs(a[(i, j)]);
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    max_row_sum
}

/// Computes the Frobenius norm of a matrix.
///
/// ||A||_F = sqrt(Σ_ij |a_ij|²)
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::norm_frobenius;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let nf = norm_frobenius(a.as_ref());
/// // ||A||_F = sqrt(1 + 4 + 9 + 16) = sqrt(30)
/// assert!((nf - 30.0f64.sqrt()).abs() < 1e-10);
/// ```
pub fn norm_frobenius<T: Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return T::zero();
    }

    let mut sum_sq = T::zero();

    for i in 0..m {
        for j in 0..n {
            let val = a[(i, j)];
            sum_sq = sum_sq + val * val;
        }
    }

    Real::sqrt(sum_sq)
}

/// Computes the 2-norm (spectral norm / largest singular value) of a matrix.
///
/// ||A||_2 = σ_max(A)
///
/// This is the most expensive norm to compute as it requires SVD.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::norm_2;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[3.0f64, 0.0],
///     &[0.0, 4.0],
/// ]);
///
/// let n2 = norm_2(a.as_ref()).unwrap();
/// // For diagonal matrix, 2-norm is max |diagonal element| = 4
/// assert!((n2 - 4.0).abs() < 1e-10);
/// ```
pub fn norm_2<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, SvdError> {
    let svd = Svd::compute(a)?;
    Ok(svd.norm2())
}

/// Computes the trace of a square matrix.
///
/// tr(A) = Σ_i a_ii
///
/// # Arguments
///
/// * `a` - Square matrix A (n×n)
///
/// # Returns
///
/// The trace. Returns zero if the matrix is empty.
///
/// # Panics
///
/// Panics if the matrix is not square.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::trace;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let t = trace(a.as_ref());
/// assert!((t - 5.0).abs() < 1e-10); // 1 + 4 = 5
/// ```
pub fn trace<T: Field>(a: MatRef<'_, T>) -> T {
    let n = a.nrows();

    if n != a.ncols() {
        panic!("trace requires a square matrix, got {}×{}", n, a.ncols());
    }

    if n == 0 {
        return T::zero();
    }

    let mut tr = T::zero();
    for i in 0..n {
        tr = tr + a[(i, i)];
    }

    tr
}

/// Computes the nuclear norm (trace norm / sum of singular values).
///
/// ||A||_* = Σ_i σ_i
///
/// Also known as the Schatten 1-norm.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::norm_nuclear;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[3.0f64, 0.0],
///     &[0.0, 4.0],
/// ]);
///
/// let nn = norm_nuclear(a.as_ref()).unwrap();
/// // For diagonal matrix, nuclear norm = sum of |diagonal| = 7
/// assert!((nn - 7.0).abs() < 1e-10);
/// ```
pub fn norm_nuclear<T: Field + Real + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Result<T, SvdError> {
    let svd = Svd::compute(a)?;
    let sigma = svd.singular_values();

    let mut sum = T::zero();
    for &s in sigma {
        sum = sum + s;
    }

    Ok(sum)
}

/// Computes the max norm (maximum absolute entry).
///
/// ||A||_max = max_ij |a_ij|
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::norm_max;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, -5.0],
///     &[3.0, 2.0],
/// ]);
///
/// let nm = norm_max(a.as_ref());
/// assert!((nm - 5.0).abs() < 1e-10);
/// ```
pub fn norm_max<T: Real>(a: MatRef<'_, T>) -> T {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return T::zero();
    }

    let mut max_val = T::zero();

    for i in 0..m {
        for j in 0..n {
            let abs_val = Scalar::abs(a[(i, j)]);
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
    }

    max_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_norm_1() {
        let a = Mat::from_rows(&[&[1.0f64, -2.0, 3.0], &[4.0, 5.0, -6.0]]);

        let n1 = norm_1(a.as_ref());
        // Column sums: 1+4=5, 2+5=7, 3+6=9
        assert!(approx_eq(n1, 9.0, 1e-10));
    }

    #[test]
    fn test_norm_inf() {
        let a = Mat::from_rows(&[&[1.0f64, -2.0, 3.0], &[4.0, 5.0, -6.0]]);

        let ni = norm_inf(a.as_ref());
        // Row sums: 1+2+3=6, 4+5+6=15
        assert!(approx_eq(ni, 15.0, 1e-10));
    }

    #[test]
    fn test_norm_frobenius() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let nf = norm_frobenius(a.as_ref());
        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert!(approx_eq(nf, 30.0f64.sqrt(), 1e-10));
    }

    #[test]
    fn test_norm_2() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let n2 = norm_2(a.as_ref()).unwrap();
        assert!(approx_eq(n2, 4.0, 1e-10));
    }

    #[test]
    fn test_norm_2_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let n2 = norm_2(eye.as_ref()).unwrap();
        assert!(approx_eq(n2, 1.0, 1e-10));
    }

    #[test]
    fn test_trace() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let t = trace(a.as_ref());
        assert!(approx_eq(t, 15.0, 1e-10)); // 1 + 5 + 9
    }

    #[test]
    fn test_trace_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let t = trace(eye.as_ref());
        assert!(approx_eq(t, 3.0, 1e-10));
    }

    #[test]
    fn test_norm_nuclear() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let nn = norm_nuclear(a.as_ref()).unwrap();
        assert!(approx_eq(nn, 7.0, 1e-10));
    }

    #[test]
    fn test_norm_max() {
        let a = Mat::from_rows(&[&[1.0f64, -5.0], &[3.0, 2.0]]);

        let nm = norm_max(a.as_ref());
        assert!(approx_eq(nm, 5.0, 1e-10));
    }

    #[test]
    fn test_norm_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);

        assert!(approx_eq(norm_1(a.as_ref()), 0.0, 1e-10));
        assert!(approx_eq(norm_inf(a.as_ref()), 0.0, 1e-10));
        assert!(approx_eq(norm_frobenius(a.as_ref()), 0.0, 1e-10));
        assert!(approx_eq(norm_max(a.as_ref()), 0.0, 1e-10));
    }

    #[test]
    fn test_norm_relations() {
        // For any matrix: ||A||_2 <= ||A||_F <= sqrt(rank) * ||A||_2
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let n2 = norm_2(a.as_ref()).unwrap();
        let nf = norm_frobenius(a.as_ref());

        assert!(n2 <= nf + 1e-10);
        // For full rank 2x2 matrix: nf <= sqrt(2) * n2
        assert!(nf <= 2.0f64.sqrt() * n2 + 1e-10);
    }

    #[test]
    fn test_norm_1_inf_relation() {
        // ||A||_1 = ||A^T||_inf
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        // Transpose
        let mut at = Mat::zeros(3, 2);
        for i in 0..2 {
            for j in 0..3 {
                at[(j, i)] = a[(i, j)];
            }
        }

        let n1_a = norm_1(a.as_ref());
        let ninf_at = norm_inf(at.as_ref());

        assert!(approx_eq(n1_a, ninf_at, 1e-10));
    }

    #[test]
    fn test_norms_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let n1 = norm_1(a.as_ref());
        let ninf = norm_inf(a.as_ref());
        let nf = norm_frobenius(a.as_ref());

        assert!((n1 - 6.0).abs() < 1e-5); // max(4, 6) = 6
        assert!((ninf - 7.0).abs() < 1e-5); // max(3, 7) = 7
        assert!((nf - 30.0f32.sqrt()).abs() < 1e-5);
    }
}
