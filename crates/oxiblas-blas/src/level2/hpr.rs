//! HPR: Hermitian packed rank-1 update.
//!
//! Performs the Hermitian rank-1 update: A = alpha * x * x^H + A
//! where A is a Hermitian matrix stored in packed format.
//! `alpha` is typed `T::Real`, so the update keeps A Hermitian.
//!
//! # Packed Storage Format
//!
//! For an n×n Hermitian matrix, only n*(n+1)/2 elements are stored.
//! The diagonal elements are real (imaginary part is zero).
//!
//! **Upper triangle (column-major):**
//! ```text
//! ap = [a11, a12, a22, a13, a23, a33, ...]
//! Index: ap[i + j*(j+1)/2] = a(i,j) for 0 <= i <= j
//! ```
//!
//! **Lower triangle (column-major):**
//! ```text
//! ap = [a11, a21, a31, ..., a22, a32, ..., a33, ...]
//! Index: ap[i + (2*n - j - 1)*j/2] = a(i,j) for i >= j
//! ```

use super::her::real_alpha_conj;
use num_traits::Zero;
use oxiblas_core::scalar::Field;

/// Specifies which triangle of the matrix is stored in packed format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HprUplo {
    /// Lower triangle is stored.
    Lower,
    /// Upper triangle is stored.
    Upper,
}

/// Error type for Hermitian packed rank-1 update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HprError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector has wrong dimension.
    DimensionMismatch,
}

impl core::fmt::Display for HprError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPackedSize => {
                write!(f, "Packed array has wrong size (expected n*(n+1)/2)")
            }
            Self::DimensionMismatch => write!(f, "Vector has wrong dimension"),
        }
    }
}

impl std::error::Error for HprError {}

/// Computes the index into upper packed array for element (i, j) where i <= j.
#[inline]
fn upper_packed_index(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    i + j * (j + 1) / 2
}

/// Computes the index into lower packed array for element (i, j) where i >= j.
#[inline]
fn lower_packed_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i >= j);
    i + (2 * n - j - 1) * j / 2
}

/// Performs the Hermitian packed rank-1 update: A = alpha * x * x^H + A
///
/// Only the specified triangle (upper or lower) is stored and updated.
/// For a Hermitian matrix, A\[i,j\] = conj(A\[j,i\]) and diagonal is real.
///
/// This is the packed twin of [`her`](crate::level2::her()) and gives the same
/// results as `her` on the same data (bitwise for NaN-free input; with NaN input
/// the NaN sign/payload may differ). As in reference BLAS `?hpr`, the imaginary
/// part of every diagonal element is *set to zero* by the update
/// (`A[j,j] = real(A[j,j]) + real(temp * x[j])` with `temp = alpha * conj(x[j])`,
/// formed exactly as in `her`), and a column with `x[j] == 0` is skipped entirely
/// (only `A[j,j] = real(A[j,j])` is applied), so an infinite or NaN `x[i]` cannot
/// reach it. If `n == 0` or `alpha == 0` the packed array is left completely
/// untouched (including an imaginary diagonal).
///
/// `alpha` is typed `T::Real`, so the update keeps A Hermitian; the Hermitian
/// update with a complex scalar is the rank-2 update [`hpr2`](crate::level2::hpr2()).
///
/// # Arguments
///
/// * `uplo` - Specifies whether the upper or lower triangle is stored
/// * `n` - Order of the Hermitian matrix A
/// * `alpha` - The real scalar alpha
/// * `x` - The vector x of length n
/// * `ap` - The packed Hermitian matrix (n*(n+1)/2 elements)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{hpr, HprUplo};
/// use num_complex::Complex64;
///
/// fn c(re: f64, im: f64) -> Complex64 {
///     Complex64::new(re, im)
/// }
///
/// // n=2: need 2*3/2 = 3 elements
/// let mut ap = [c(0.0, 0.0); 3];
/// let x = [c(1.0, 1.0), c(2.0, 0.0)];
///
/// // A = A + x * x^H
/// hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();
///
/// // x * x^H = [[|1+i|^2, (1+i)*2], [2*(1-i), |2|^2]]
/// //         = [[2, 2+2i], [2-2i, 4]]
/// // Upper packed: [a00, a01, a11] = [2, 2+2i, 4]
/// assert!((ap[0].re - 2.0).abs() < 1e-10); // |1+i|^2 = 2
/// assert!((ap[0].im).abs() < 1e-10);       // diagonal is real
/// assert!((ap[1].re - 2.0).abs() < 1e-10); // (1+i)*2 = 2+2i
/// assert!((ap[1].im - 2.0).abs() < 1e-10);
/// assert!((ap[2].re - 4.0).abs() < 1e-10); // |2|^2 = 4
/// assert!((ap[2].im).abs() < 1e-10);       // diagonal is real
/// ```
pub fn hpr<T: Field>(
    uplo: HprUplo,
    n: usize,
    alpha: T::Real,
    x: &[T],
    ap: &mut [T],
) -> Result<(), HprError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(HprError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(HprError::DimensionMismatch);
    }

    if n == 0 || alpha == T::Real::zero() {
        return Ok(());
    }

    // The loops mirror `her` operation for operation, so packed and dense storage
    // give the same results (bitwise for NaN-free input; with NaN input the NaN
    // sign/payload may differ).
    match uplo {
        HprUplo::Upper => {
            // Update upper triangle: A[i,j] += alpha * x[i] * conj(x[j]) for i <= j
            for j in 0..n {
                let kjj = upper_packed_index(j, j);
                // Reference ?HPR skips the column update when x(j) == 0 (so an
                // infinite or NaN x(i) cannot reach it as `x(i) * conj(0)`) and only
                // drops the imaginary part of the diagonal.
                if x[j] == T::zero() {
                    ap[kjj] = T::from_real(ap[kjj].real());
                    continue;
                }
                let alpha_xj_conj = real_alpha_conj(alpha, x[j]);

                for i in 0..j {
                    let kij = upper_packed_index(i, j);
                    ap[kij] += alpha_xj_conj * x[i];
                }

                // Diagonal: A[j,j] = real(A[j,j]) + real(alpha * conj(x[j]) * x[j]).
                // Like reference ?HPR the stored imaginary part is dropped, so the
                // diagonal is exactly real afterwards.
                let val = alpha_xj_conj * x[j];
                ap[kjj] = T::from_real(ap[kjj].real() + val.real());
            }
        }
        HprUplo::Lower => {
            // Update lower triangle: A[i,j] += alpha * x[i] * conj(x[j]) for i >= j
            for j in 0..n {
                let kjj = lower_packed_index(j, j, n);
                // Column skip for x(j) == 0, as in the upper branch.
                if x[j] == T::zero() {
                    ap[kjj] = T::from_real(ap[kjj].real());
                    continue;
                }
                let alpha_xj_conj = real_alpha_conj(alpha, x[j]);

                // Diagonal: real(A[j,j]) + real(update); stored imaginary part
                // dropped, as in reference ?HPR.
                let val = alpha_xj_conj * x[j];
                ap[kjj] = T::from_real(ap[kjj].real() + val.real());

                for i in (j + 1)..n {
                    let kij = lower_packed_index(i, j, n);
                    ap[kij] += alpha_xj_conj * x[i];
                }
            }
        }
    }

    Ok(())
}

/// Creates a new packed Hermitian matrix from x * x^H.
///
/// # Arguments
///
/// * `uplo` - Specifies whether to store upper or lower triangle
/// * `alpha` - The real scalar alpha
/// * `x` - The vector x
///
/// # Returns
///
/// A new packed array containing alpha * x * x^H.
pub fn hpr_new<T: Field>(uplo: HprUplo, alpha: T::Real, x: &[T]) -> Vec<T> {
    let n = x.len();
    let packed_size = n * (n + 1) / 2;
    let mut ap = vec![T::zero(); packed_size];

    if n == 0 || alpha == T::Real::zero() {
        return ap;
    }

    let _ = hpr(uplo, n, alpha, x, &mut ap);
    ap
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_hpr_upper_basic() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 1.0), c(2.0, 0.0)];

        hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();

        // x * x^H = [[|1+i|^2, (1+i)*2], [2*(1-i), |2|^2]]
        //         = [[2, 2+2i], [2-2i, 4]]
        // Upper packed: [a00, a01, a11] = [2, 2+2i, 4]
        assert!(approx_eq(ap[0].re, 2.0)); // |1+i|^2 = 2
        assert!(approx_eq(ap[0].im, 0.0)); // diagonal is real
        assert!(approx_eq_c(ap[1], c(2.0, 2.0))); // (1+i)*2
        assert!(approx_eq(ap[2].re, 4.0)); // |2|^2 = 4
        assert!(approx_eq(ap[2].im, 0.0)); // diagonal is real
    }

    #[test]
    fn test_hpr_lower_basic() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 1.0), c(2.0, 0.0)];

        hpr(HprUplo::Lower, 2, 1.0, &x, &mut ap).unwrap();

        // x * x^H = [[2, 2+2i], [2-2i, 4]]
        // Lower packed: [a00, a10, a11] = [2, 2-2i, 4]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(2.0, -2.0))); // (1-i)*2 = 2-2i
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_with_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(0.0, 1.0)];

        hpr(HprUplo::Upper, 2, 2.0, &x, &mut ap).unwrap();

        // x * x^H = [[1, -i], [i, 1]]
        // 2 * x * x^H = [[2, -2i], [2i, 2]]
        // Upper packed: [2, -2i, 2]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(0.0, -2.0)));
        assert!(approx_eq(ap[2].re, 2.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_accumulate() {
        let mut ap = [c(1.0, 0.0), c(2.0, 1.0), c(3.0, 0.0)];
        let x = [c(1.0, 0.0), c(1.0, 0.0)];

        hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap).unwrap();

        // x * x^H = [[1, 1], [1, 1]]
        // New: [2, 3+i, 4]
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0)); // diagonal stays real
        assert!(approx_eq_c(ap[1], c(3.0, 1.0)));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_real_vector() {
        // Real vector - result is symmetric
        let mut ap = [c(0.0, 0.0); 6];
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];

        hpr(HprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();

        // Same as SPR for real vectors
        assert!(approx_eq(ap[0].re, 1.0));
        assert!(approx_eq(ap[1].re, 2.0));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[3].re, 3.0));
        assert!(approx_eq(ap[4].re, 6.0));
        assert!(approx_eq(ap[5].re, 9.0));
    }

    #[test]
    fn test_hpr_zero_alpha() {
        let mut ap = [c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)];
        let original = ap;
        let x = [c(1.0, 1.0), c(2.0, 2.0)];

        hpr(HprUplo::Upper, 2, 0.0, &x, &mut ap).unwrap();

        // Unchanged
        for i in 0..3 {
            assert!(approx_eq_c(ap[i], original[i]));
        }
    }

    #[test]
    fn test_hpr_dimension_errors() {
        // Wrong packed size
        let mut ap = [c(0.0, 0.0); 5]; // Should be 6 for n=3
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        assert_eq!(
            hpr(HprUplo::Upper, 3, 1.0, &x, &mut ap),
            Err(HprError::InvalidPackedSize)
        );

        // Wrong x dimension
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        assert_eq!(
            hpr(HprUplo::Upper, 2, 1.0, &x, &mut ap),
            Err(HprError::DimensionMismatch)
        );
    }

    #[test]
    fn test_hpr_empty() {
        let mut ap: [Complex64; 0] = [];
        let x: [Complex64; 0] = [];

        hpr(HprUplo::Upper, 0, 1.0, &x, &mut ap).unwrap();
    }

    #[test]
    fn test_hpr_1x1() {
        let mut ap = [c(0.0, 0.0)];
        let x = [c(2.0, 1.0)];

        hpr(HprUplo::Upper, 1, 1.0, &x, &mut ap).unwrap();

        // |2+i|^2 = 4 + 1 = 5
        assert!(approx_eq(ap[0].re, 5.0));
        assert!(approx_eq(ap[0].im, 0.0));
    }

    #[test]
    fn test_hpr_new_upper() {
        let x = [c(1.0, 1.0), c(2.0, 0.0)];
        let ap = hpr_new(HprUplo::Upper, 1.0, &x);

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(2.0, 2.0)));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_new_lower() {
        let x = [c(1.0, 1.0), c(2.0, 0.0)];
        let ap = hpr_new(HprUplo::Lower, 1.0, &x);

        assert_eq!(ap.len(), 3);
        assert!(approx_eq(ap[0].re, 2.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(2.0, -2.0)));
        assert!(approx_eq(ap[2].re, 4.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_negative_alpha() {
        let mut ap = [c(0.0, 0.0); 3];
        let x = [c(1.0, 0.0), c(0.0, 1.0)];

        hpr(HprUplo::Upper, 2, -1.0, &x, &mut ap).unwrap();

        // -1 * x * x^H = [[-1, i], [-i, -1]]
        assert!(approx_eq(ap[0].re, -1.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(0.0, 1.0)));
        assert!(approx_eq(ap[2].re, -1.0));
        assert!(approx_eq(ap[2].im, 0.0));
    }

    #[test]
    fn test_hpr_3x3() {
        let mut ap = [c(0.0, 0.0); 6];
        let x = [c(1.0, 0.0), c(0.0, 1.0), c(1.0, 1.0)];

        hpr(HprUplo::Upper, 3, 1.0, &x, &mut ap).unwrap();

        // x * x^H:
        // [0,0] = 1*1 = 1
        // [0,1] = 1*(-i) = -i
        // [0,2] = 1*(1-i) = 1-i
        // [1,1] = i*(-i) = 1
        // [1,2] = i*(1-i) = i+1 = 1+i
        // [2,2] = (1+i)*(1-i) = 1+1 = 2

        // Upper packed: [a00, a01, a11, a02, a12, a22]
        assert!(approx_eq(ap[0].re, 1.0));
        assert!(approx_eq(ap[0].im, 0.0));
        assert!(approx_eq_c(ap[1], c(0.0, -1.0)));
        assert!(approx_eq(ap[2].re, 1.0));
        assert!(approx_eq(ap[2].im, 0.0));
        assert!(approx_eq_c(ap[3], c(1.0, -1.0)));
        assert!(approx_eq_c(ap[4], c(1.0, 1.0)));
        assert!(approx_eq(ap[5].re, 2.0));
        assert!(approx_eq(ap[5].im, 0.0));
    }

    fn bits(z: Complex64) -> (u64, u64) {
        (z.re.to_bits(), z.im.to_bits())
    }

    fn finite(z: Complex64) -> bool {
        z.re.is_finite() && z.im.is_finite()
    }

    /// Issue #14: reference ?HPR skips column `j` when `x(j) == 0` and only sets
    /// `AP(jj) = real(AP(jj))`, so an infinite `x(i)` cannot reach the column as
    /// `x(i) * conj(0) = NaN`.
    #[test]
    fn test_issue_14_hpr_zero_column_skip() {
        let inf = f64::INFINITY;

        // Lower packed [A00, A10, A20, A11, A21, A22] with an imaginary diagonal:
        // column 0 (x0 == 0) holds row 1, where x1 is infinite.
        let ap0 = [
            c(1.0, 0.5),
            c(0.2, 0.1),
            c(0.3, -0.2),
            c(2.0, 0.7),
            c(0.4, 0.6),
            c(3.0, -0.8),
        ];
        let mut ap = ap0;
        let x = [c(0.0, -0.0), c(inf, 0.0), c(1.0, 0.5)];
        hpr(HprUplo::Lower, 3, 1.5, &x, &mut ap).unwrap();
        assert_eq!(bits(ap[1]), bits(ap0[1]));
        assert_eq!(bits(ap[2]), bits(ap0[2]));
        assert_eq!(bits(ap[0]), bits(c(1.0, 0.0)));
        // Column 1 is updated with temp = 1.5 * conj(Inf).
        assert!(!finite(ap[3]) && !finite(ap[4]));

        // Upper packed [A00, A01, A11, A02, A12, A22] with an imaginary diagonal:
        // column 2 (x2 == 0) holds row 1, where x1 is infinite.
        let ap0 = [
            c(1.0, 0.5),
            c(0.2, 0.1),
            c(2.0, 0.7),
            c(0.3, -0.2),
            c(0.4, 0.6),
            c(3.0, -0.8),
        ];
        let mut ap = ap0;
        let x = [c(1.0, 0.5), c(inf, 0.0), c(-0.0, 0.0)];
        hpr(HprUplo::Upper, 3, 1.5, &x, &mut ap).unwrap();
        assert_eq!(bits(ap[3]), bits(ap0[3]));
        assert_eq!(bits(ap[4]), bits(ap0[4]));
        assert_eq!(bits(ap[5]), bits(c(3.0, 0.0)));
        assert!(!finite(ap[1]) && !finite(ap[2]));
    }

    /// Dense `n x n` Hermitian test matrix whose diagonal carries an imaginary part
    /// the update must drop, and its packed `uplo` triangle.
    fn hermitian_pair(uplo: HprUplo, n: usize) -> (Mat<Complex64>, Vec<Complex64>) {
        let mut a: Mat<Complex64> = Mat::filled(n, n, c(f64::NAN, f64::NAN));
        let mut ap = vec![c(0.0, 0.0); n * (n + 1) / 2];
        for j in 0..n {
            for i in 0..n {
                let stored = match uplo {
                    HprUplo::Upper => i <= j,
                    HprUplo::Lower => i >= j,
                };
                if !stored {
                    continue;
                }
                let v = c(
                    0.25 * (i + 1) as f64 - 0.5 * j as f64,
                    0.125 * (i + 2 * j) as f64 + 0.0625,
                );
                a[(i, j)] = v;
                let k = match uplo {
                    HprUplo::Upper => upper_packed_index(i, j),
                    HprUplo::Lower => lower_packed_index(i, j, n),
                };
                ap[k] = v;
            }
        }
        (a, ap)
    }

    /// Issue #13: reference ?HPR forms `TEMP = ALPHA*DCONJG(X(J))` as the complex
    /// product `(alpha, 0) * conj(x(j))`, exactly like ?HER, so an infinite `x(j)`
    /// gives the diagonal `(NaN, +0)` in `hpr` as in `her`, and the two agree
    /// bitwise on every stored element (the input holds no NaN). The expected
    /// values follow zhpr.f; NaN bit patterns are only compared between the twins,
    /// never against a literal.
    #[test]
    fn test_issue_13_her_hpr_infinite_x_diagonal_matches_reference() {
        use crate::level2::{HerUplo, her};
        let inf = f64::INFINITY;
        for alpha in [0.5, -2.0] {
            for xj in [c(inf, 0.0), c(0.0, inf), c(-inf, 1.0)] {
                for (uplo, her_uplo) in [
                    (HprUplo::Lower, HerUplo::Lower),
                    (HprUplo::Upper, HerUplo::Upper),
                ] {
                    let x = [c(0.0, 0.0), xj];
                    let (mut a, mut ap) = hermitian_pair(uplo, 2);
                    // Packed indices of A00, A11 and the stored off-diagonal element.
                    let (k00, k11, koff, off) = match uplo {
                        HprUplo::Upper => (
                            upper_packed_index(0, 0),
                            upper_packed_index(1, 1),
                            upper_packed_index(0, 1),
                            (0, 1),
                        ),
                        HprUplo::Lower => (
                            lower_packed_index(0, 0, 2),
                            lower_packed_index(1, 1, 2),
                            lower_packed_index(1, 0, 2),
                            (1, 0),
                        ),
                    };
                    let (a00, aoff) = (ap[k00], ap[koff]);
                    hpr(uplo, 2, alpha, &x, &mut ap).unwrap();
                    her(her_uplo, alpha, &x, a.as_mut()).unwrap();
                    let what = format!("{uplo:?} alpha={alpha} x1={xj}");
                    // A11 = real(A11) + real(x1 * temp), where temp holds a NaN part.
                    assert!(ap[k11].re.is_nan(), "{what}");
                    assert_eq!(ap[k11].im.to_bits(), 0.0f64.to_bits(), "{what}");
                    // Column 0 (x0 == 0) is skipped: A00 = real(A00).
                    assert_eq!(bits(ap[k00]), bits(c(a00.re, 0.0)), "{what}");
                    match uplo {
                        // A01 += x0 * temp: 0 times a NaN and an infinite part.
                        HprUplo::Upper => {
                            assert!(ap[koff].re.is_nan() && ap[koff].im.is_nan(), "{what}");
                        }
                        // Column 0 is skipped and column 1 has no subdiagonal.
                        HprUplo::Lower => assert_eq!(bits(ap[koff]), bits(aoff), "{what}"),
                    }
                    // `her` gives the same bits on every stored element.
                    for (k, (i, j)) in [(k00, (0, 0)), (k11, (1, 1)), (koff, off)] {
                        assert_eq!(bits(ap[k]), bits(a[(i, j)]), "her {what} ({i},{j})");
                    }
                }
            }
        }
    }

    /// Issue #13 / #14: `hpr` is the packed twin of `her` (same column skip, same
    /// temp, same diagonal expression), so both give bitwise identical results on
    /// this NaN-free data, NaN generated from Inf included.
    #[test]
    fn test_issue_14_hpr_matches_her_bitwise() {
        use crate::level2::{HerUplo, her};
        let inf = f64::INFINITY;
        let xs = [
            [c(0.0, 0.0), c(inf, 0.0), c(1.0, -2.0), c(-0.0, 0.0)],
            [c(0.3, 0.7), c(-1.1, 0.2), c(0.0, inf), c(2.5, -0.5)],
            [c(0.1, -0.2), c(0.3, 0.4), c(-0.5, 0.6), c(0.7, 0.8)],
        ];
        let n = 4;
        for (uplo, her_uplo) in [
            (HprUplo::Lower, HerUplo::Lower),
            (HprUplo::Upper, HerUplo::Upper),
        ] {
            for x in &xs {
                for alpha in [0.75, -1.25] {
                    let (mut a, mut ap) = hermitian_pair(uplo, n);
                    hpr(uplo, n, alpha, x, &mut ap).unwrap();
                    her(her_uplo, alpha, x, a.as_mut()).unwrap();
                    for j in 0..n {
                        let rows = match uplo {
                            HprUplo::Upper => 0..j + 1,
                            HprUplo::Lower => j..n,
                        };
                        for i in rows {
                            let k = match uplo {
                                HprUplo::Upper => upper_packed_index(i, j),
                                HprUplo::Lower => lower_packed_index(i, j, n),
                            };
                            assert_eq!(
                                bits(ap[k]),
                                bits(a[(i, j)]),
                                "{uplo:?} alpha={alpha} x={x:?} ({i},{j})"
                            );
                        }
                    }
                }
            }
        }
    }
}
