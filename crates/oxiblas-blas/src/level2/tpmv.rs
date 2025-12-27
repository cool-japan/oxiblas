//! TPMV: Triangular packed matrix-vector multiply.
//!
//! Computes x = op(A) * x where A is a triangular matrix stored in packed format.
//!
//! # Packed Storage Format
//!
//! For an n×n triangular matrix, only n*(n+1)/2 elements are stored.
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

use oxiblas_core::scalar::Field;

/// Specifies which triangle of the matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmvUplo {
    /// Lower triangular matrix.
    Lower,
    /// Upper triangular matrix.
    Upper,
}

/// Specifies the operation on the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmvTrans {
    /// No transpose: x = A * x
    NoTrans,
    /// Transpose: x = A^T * x
    Trans,
    /// Conjugate transpose: x = A^H * x
    ConjTrans,
}

/// Specifies whether the diagonal is assumed to be unit (all ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmvDiag {
    /// Use the actual diagonal elements from the matrix.
    NonUnit,
    /// Assume diagonal elements are all 1.0 (don't read them).
    Unit,
}

/// Error type for triangular packed matrix-vector multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpmvError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector has wrong dimension.
    DimensionMismatch,
}

impl core::fmt::Display for TpmvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPackedSize => {
                write!(f, "Packed array has wrong size (expected n*(n+1)/2)")
            }
            Self::DimensionMismatch => write!(f, "Vector has wrong dimension"),
        }
    }
}

impl std::error::Error for TpmvError {}

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

/// Triangular packed matrix-vector multiply.
///
/// Computes x = op(A) * x in-place, where A is a triangular matrix stored in packed format.
///
/// # Arguments
///
/// * `uplo` - Specifies whether A is lower or upper triangular
/// * `trans` - The operation to apply (`NoTrans`, Trans, or `ConjTrans`)
/// * `diag` - Whether the diagonal is assumed to be unit
/// * `n` - Order of the triangular matrix A
/// * `ap` - Packed triangular matrix (n*(n+1)/2 elements)
/// * `x` - Vector of length n (overwritten with the result)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{tpmv, TpmvUplo, TpmvTrans, TpmvDiag};
///
/// // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
/// // Lower packed: [2, 1, 2, 3, 1, 4]
/// let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
/// let mut x = [1.0, 2.0, 3.0];
///
/// // x = L * x = [2*1, 1*1+3*2, 2*1+1*2+4*3] = [2, 7, 16]
/// tpmv(TpmvUplo::Lower, TpmvTrans::NoTrans, TpmvDiag::NonUnit, 3, &ap, &mut x).unwrap();
///
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 7.0).abs() < 1e-10);
/// assert!((x[2] - 16.0).abs() < 1e-10);
/// ```
pub fn tpmv<T: Field>(
    uplo: TpmvUplo,
    trans: TpmvTrans,
    diag: TpmvDiag,
    n: usize,
    ap: &[T],
    x: &mut [T],
) -> Result<(), TpmvError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(TpmvError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(TpmvError::DimensionMismatch);
    }

    if n == 0 {
        return Ok(());
    }

    match (uplo, trans) {
        (TpmvUplo::Lower, TpmvTrans::NoTrans) => {
            tpmv_lower_notrans(ap, x, n, diag);
        }
        (TpmvUplo::Upper, TpmvTrans::NoTrans) => {
            tpmv_upper_notrans(ap, x, n, diag);
        }
        (TpmvUplo::Lower, TpmvTrans::Trans | TpmvTrans::ConjTrans) => {
            tpmv_lower_trans(ap, x, n, diag, matches!(trans, TpmvTrans::ConjTrans));
        }
        (TpmvUplo::Upper, TpmvTrans::Trans | TpmvTrans::ConjTrans) => {
            tpmv_upper_trans(ap, x, n, diag, matches!(trans, TpmvTrans::ConjTrans));
        }
    }

    Ok(())
}

/// Lower triangular packed, no transpose: x = L * x
fn tpmv_lower_notrans<T: Field>(ap: &[T], x: &mut [T], n: usize, diag: TpmvDiag) {
    // Process from bottom to top to avoid overwriting x values we need
    for i in (0..n).rev() {
        let mut sum = if diag == TpmvDiag::Unit {
            x[i]
        } else {
            let kii = lower_packed_index(i, i, n);
            ap[kii] * x[i]
        };

        // Off-diagonal elements in column i (elements below diagonal)
        for j in 0..i {
            let kij = lower_packed_index(i, j, n);
            sum += ap[kij] * x[j];
        }
        x[i] = sum;
    }
}

/// Upper triangular packed, no transpose: x = U * x
fn tpmv_upper_notrans<T: Field>(ap: &[T], x: &mut [T], n: usize, diag: TpmvDiag) {
    // Process from top to bottom
    for i in 0..n {
        let mut sum = if diag == TpmvDiag::Unit {
            x[i]
        } else {
            let kii = upper_packed_index(i, i);
            ap[kii] * x[i]
        };

        // Off-diagonal elements in row i (elements to the right of diagonal)
        for j in (i + 1)..n {
            let kij = upper_packed_index(i, j);
            sum += ap[kij] * x[j];
        }
        x[i] = sum;
    }
}

/// Lower triangular packed with transpose: x = L^T * x (or L^H * x)
fn tpmv_lower_trans<T: Field>(ap: &[T], x: &mut [T], n: usize, diag: TpmvDiag, conj: bool) {
    // L^T is upper triangular, so process from top to bottom
    for i in 0..n {
        let mut sum = if diag == TpmvDiag::Unit {
            x[i]
        } else {
            let kii = lower_packed_index(i, i, n);
            let diag_val = if conj { ap[kii].conj() } else { ap[kii] };
            diag_val * x[i]
        };

        // L^T[i,j] = L[j,i] for j > i (elements below in L are to the right in L^T)
        for j in (i + 1)..n {
            let kji = lower_packed_index(j, i, n);
            let val = if conj { ap[kji].conj() } else { ap[kji] };
            sum += val * x[j];
        }
        x[i] = sum;
    }
}

/// Upper triangular packed with transpose: x = U^T * x (or U^H * x)
fn tpmv_upper_trans<T: Field>(ap: &[T], x: &mut [T], n: usize, diag: TpmvDiag, conj: bool) {
    // U^T is lower triangular, so process from bottom to top
    for i in (0..n).rev() {
        let mut sum = if diag == TpmvDiag::Unit {
            x[i]
        } else {
            let kii = upper_packed_index(i, i);
            let diag_val = if conj { ap[kii].conj() } else { ap[kii] };
            diag_val * x[i]
        };

        // U^T[i,j] = U[j,i] for j < i (elements above in U are to the left in U^T)
        for j in 0..i {
            let kji = upper_packed_index(j, i);
            let val = if conj { ap[kji].conj() } else { ap[kji] };
            sum += val * x[j];
        }
        x[i] = sum;
    }
}

/// Non-in-place version: returns a new vector with the result.
pub fn tpmv_new<T: Field>(
    uplo: TpmvUplo,
    trans: TpmvTrans,
    diag: TpmvDiag,
    n: usize,
    ap: &[T],
    x: &[T],
) -> Result<Vec<T>, TpmvError> {
    let mut result = x.to_vec();
    tpmv(uplo, trans, diag, n, ap, &mut result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_tpmv_lower_notrans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // Lower packed: [a00, a10, a20, a11, a21, a22] = [2, 1, 2, 3, 1, 4]
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let mut x = [1.0, 2.0, 3.0];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        // L*x = [2*1, 1*1+3*2, 2*1+1*2+4*3] = [2, 7, 16]
        assert!(approx_eq(x[0], 2.0));
        assert!(approx_eq(x[1], 7.0));
        assert!(approx_eq(x[2], 16.0));
    }

    #[test]
    fn test_tpmv_upper_notrans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // Upper packed: [a00, a01, a11, a02, a12, a22] = [2, 1, 3, 2, 1, 4]
        let ap = [2.0f64, 1.0, 3.0, 2.0, 1.0, 4.0];
        let mut x = [1.0, 2.0, 3.0];

        tpmv(
            TpmvUplo::Upper,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        // U*x = [2*1+1*2+2*3, 3*2+1*3, 4*3] = [10, 9, 12]
        assert!(approx_eq(x[0], 10.0));
        assert!(approx_eq(x[1], 9.0));
        assert!(approx_eq(x[2], 12.0));
    }

    #[test]
    fn test_tpmv_lower_trans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // Lower packed: [2, 1, 2, 3, 1, 4]
        // L^T = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let mut x = [1.0, 2.0, 3.0];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::Trans,
            TpmvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        // L^T*x = [2*1+1*2+2*3, 3*2+1*3, 4*3] = [10, 9, 12]
        assert!(approx_eq(x[0], 10.0));
        assert!(approx_eq(x[1], 9.0));
        assert!(approx_eq(x[2], 12.0));
    }

    #[test]
    fn test_tpmv_upper_trans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // Upper packed: [2, 1, 3, 2, 1, 4]
        // U^T = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        let ap = [2.0f64, 1.0, 3.0, 2.0, 1.0, 4.0];
        let mut x = [1.0, 2.0, 3.0];

        tpmv(
            TpmvUplo::Upper,
            TpmvTrans::Trans,
            TpmvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        // U^T*x = [2*1, 1*1+3*2, 2*1+1*2+4*3] = [2, 7, 16]
        assert!(approx_eq(x[0], 2.0));
        assert!(approx_eq(x[1], 7.0));
        assert!(approx_eq(x[2], 16.0));
    }

    #[test]
    fn test_tpmv_unit_diagonal() {
        // L with unit diagonal (diagonal is ignored)
        // L = [[1, 0, 0], [1, 1, 0], [2, 1, 1]]
        // Lower packed: [99, 1, 2, 88, 1, 77] (diagonals ignored)
        let ap = [99.0f64, 1.0, 2.0, 88.0, 1.0, 77.0];
        let mut x = [1.0, 2.0, 3.0];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::Unit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        // L*x = [1*1, 1*1+1*2, 2*1+1*2+1*3] = [1, 3, 7]
        assert!(approx_eq(x[0], 1.0));
        assert!(approx_eq(x[1], 3.0));
        assert!(approx_eq(x[2], 7.0));
    }

    #[test]
    fn test_tpmv_identity() {
        // Identity matrix (upper packed): [1, 0, 1, 0, 0, 1]
        let ap = [1.0f64, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut x = [1.0, 2.0, 3.0];

        tpmv(
            TpmvUplo::Upper,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!(approx_eq(x[0], 1.0));
        assert!(approx_eq(x[1], 2.0));
        assert!(approx_eq(x[2], 3.0));
    }

    #[test]
    fn test_tpmv_2x2() {
        // L = [[2, 0], [3, 4]]
        // Lower packed: [2, 3, 4]
        let ap = [2.0f64, 3.0, 4.0];
        let mut x = [1.0, 2.0];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        // L*x = [2*1, 3*1+4*2] = [2, 11]
        assert!(approx_eq(x[0], 2.0));
        assert!(approx_eq(x[1], 11.0));
    }

    #[test]
    fn test_tpmv_1x1() {
        let ap = [5.0f64];
        let mut x = [3.0];

        tpmv(
            TpmvUplo::Upper,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            1,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!(approx_eq(x[0], 15.0));
    }

    #[test]
    fn test_tpmv_empty() {
        let ap: [f64; 0] = [];
        let mut x: [f64; 0] = [];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            0,
            &ap,
            &mut x,
        )
        .unwrap();
    }

    #[test]
    fn test_tpmv_dimension_errors() {
        // Wrong packed size
        let ap = [1.0f64, 2.0, 3.0, 4.0]; // Should be 6 for n=3
        let mut x = [1.0, 2.0, 3.0];
        assert_eq!(
            tpmv(
                TpmvUplo::Lower,
                TpmvTrans::NoTrans,
                TpmvDiag::NonUnit,
                3,
                &ap,
                &mut x
            ),
            Err(TpmvError::InvalidPackedSize)
        );

        // Wrong x dimension
        let ap = [1.0f64, 2.0, 3.0];
        let mut x = [1.0, 2.0, 3.0];
        assert_eq!(
            tpmv(
                TpmvUplo::Lower,
                TpmvTrans::NoTrans,
                TpmvDiag::NonUnit,
                2,
                &ap,
                &mut x
            ),
            Err(TpmvError::DimensionMismatch)
        );
    }

    #[test]
    fn test_tpmv_new() {
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let x = [1.0, 2.0, 3.0];

        let result = tpmv_new(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            3,
            &ap,
            &x,
        )
        .unwrap();

        assert!(approx_eq(result[0], 2.0));
        assert!(approx_eq(result[1], 7.0));
        assert!(approx_eq(result[2], 16.0));

        // Original unchanged
        assert_eq!(x, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tpmv_complex() {
        // L = [[2, 0], [1+i, 3]]
        // Lower packed: [2, 1+i, 3]
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let mut x = [c(1.0, 0.0), c(1.0, 0.0)];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        // L*x = [2*1, (1+i)*1+3*1] = [2, 4+i]
        assert!(approx_eq_c(x[0], c(2.0, 0.0)));
        assert!(approx_eq_c(x[1], c(4.0, 1.0)));
    }

    #[test]
    fn test_tpmv_complex_conjtrans() {
        // L = [[2, 0], [1+i, 3]]
        // Lower packed: [2, 1+i, 3]
        // L^H = [[2, 1-i], [0, 3]]
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];
        let mut x = [c(1.0, 0.0), c(1.0, 0.0)];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::ConjTrans,
            TpmvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        // L^H*x = [2*1+(1-i)*1, 3*1] = [3-i, 3]
        assert!(approx_eq_c(x[0], c(3.0, -1.0)));
        assert!(approx_eq_c(x[1], c(3.0, 0.0)));
    }

    #[test]
    fn test_tpmv_f32() {
        let ap = [2.0f32, 1.0, 3.0];
        let mut x = [1.0f32, 2.0];

        tpmv(
            TpmvUplo::Lower,
            TpmvTrans::NoTrans,
            TpmvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        // L*x = [2*1, 1*1+3*2] = [2, 7]
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 7.0).abs() < 1e-5);
    }
}
