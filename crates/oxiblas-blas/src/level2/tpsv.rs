//! TPSV: Triangular packed matrix solve.
//!
//! Solves op(A) * x = b where A is a triangular matrix stored in packed format.
//! The solution overwrites b.
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
pub enum TpsvUplo {
    /// Lower triangular matrix.
    Lower,
    /// Upper triangular matrix.
    Upper,
}

/// Specifies the operation on the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpsvTrans {
    /// No transpose: A * x = b
    NoTrans,
    /// Transpose: A^T * x = b
    Trans,
    /// Conjugate transpose: A^H * x = b
    ConjTrans,
}

/// Specifies whether the diagonal is assumed to be unit (all ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpsvDiag {
    /// Use the actual diagonal elements from the matrix.
    NonUnit,
    /// Assume diagonal elements are all 1.0 (don't read them).
    Unit,
}

/// Error type for triangular packed solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpsvError {
    /// Packed array has wrong size.
    InvalidPackedSize,
    /// Vector has wrong dimension.
    DimensionMismatch,
    /// Matrix is singular (zero on diagonal).
    Singular,
}

impl core::fmt::Display for TpsvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPackedSize => {
                write!(f, "Packed array has wrong size (expected n*(n+1)/2)")
            }
            Self::DimensionMismatch => write!(f, "Vector has wrong dimension"),
            Self::Singular => write!(f, "Matrix is singular (zero on diagonal)"),
        }
    }
}

impl std::error::Error for TpsvError {}

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

/// Triangular packed solve.
///
/// Solves op(A) * x = b in-place, where A is a triangular matrix stored in packed format.
/// The solution x overwrites b.
///
/// # Arguments
///
/// * `uplo` - Specifies whether A is lower or upper triangular
/// * `trans` - The operation to apply (`NoTrans`, Trans, or `ConjTrans`)
/// * `diag` - Whether the diagonal is assumed to be unit
/// * `n` - Order of the triangular matrix A
/// * `ap` - Packed triangular matrix (n*(n+1)/2 elements)
/// * `x` - On entry, the right-hand side b. On exit, the solution x.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{tpsv, TpsvUplo, TpsvTrans, TpsvDiag};
///
/// // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
/// // Lower packed: [2, 1, 2, 3, 1, 4]
/// let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
///
/// // Solve L * x = b where b = [2, 7, 16]
/// // Expected: x = [1, 2, 3]
/// let mut x = [2.0, 7.0, 16.0];
///
/// tpsv(TpsvUplo::Lower, TpsvTrans::NoTrans, TpsvDiag::NonUnit, 3, &ap, &mut x).unwrap();
///
/// assert!((x[0] - 1.0).abs() < 1e-10);
/// assert!((x[1] - 2.0).abs() < 1e-10);
/// assert!((x[2] - 3.0).abs() < 1e-10);
/// ```
pub fn tpsv<T: Field>(
    uplo: TpsvUplo,
    trans: TpsvTrans,
    diag: TpsvDiag,
    n: usize,
    ap: &[T],
    x: &mut [T],
) -> Result<(), TpsvError> {
    // Expected packed size
    let packed_size = n * (n + 1) / 2;

    // Validate dimensions
    if ap.len() != packed_size {
        return Err(TpsvError::InvalidPackedSize);
    }
    if x.len() != n {
        return Err(TpsvError::DimensionMismatch);
    }

    if n == 0 {
        return Ok(());
    }

    match (uplo, trans) {
        (TpsvUplo::Lower, TpsvTrans::NoTrans) => {
            tpsv_lower_notrans(ap, x, n, diag)?;
        }
        (TpsvUplo::Upper, TpsvTrans::NoTrans) => {
            tpsv_upper_notrans(ap, x, n, diag)?;
        }
        (TpsvUplo::Lower, TpsvTrans::Trans | TpsvTrans::ConjTrans) => {
            tpsv_lower_trans(ap, x, n, diag, matches!(trans, TpsvTrans::ConjTrans))?;
        }
        (TpsvUplo::Upper, TpsvTrans::Trans | TpsvTrans::ConjTrans) => {
            tpsv_upper_trans(ap, x, n, diag, matches!(trans, TpsvTrans::ConjTrans))?;
        }
    }

    Ok(())
}

/// Lower triangular packed solve, no transpose: L * x = b
/// Forward substitution
fn tpsv_lower_notrans<T: Field>(
    ap: &[T],
    x: &mut [T],
    n: usize,
    diag: TpsvDiag,
) -> Result<(), TpsvError> {
    for i in 0..n {
        // Subtract known terms
        for j in 0..i {
            let kij = lower_packed_index(i, j, n);
            x[i] -= ap[kij] * x[j];
        }

        // Divide by diagonal
        if diag == TpsvDiag::NonUnit {
            let kii = lower_packed_index(i, i, n);
            if ap[kii].is_zero() {
                return Err(TpsvError::Singular);
            }
            x[i] /= ap[kii];
        }
    }
    Ok(())
}

/// Upper triangular packed solve, no transpose: U * x = b
/// Back substitution
fn tpsv_upper_notrans<T: Field>(
    ap: &[T],
    x: &mut [T],
    n: usize,
    diag: TpsvDiag,
) -> Result<(), TpsvError> {
    for i in (0..n).rev() {
        // Subtract known terms
        for j in (i + 1)..n {
            let kij = upper_packed_index(i, j);
            x[i] -= ap[kij] * x[j];
        }

        // Divide by diagonal
        if diag == TpsvDiag::NonUnit {
            let kii = upper_packed_index(i, i);
            if ap[kii].is_zero() {
                return Err(TpsvError::Singular);
            }
            x[i] /= ap[kii];
        }
    }
    Ok(())
}

/// Lower triangular packed solve with transpose: L^T * x = b (or L^H * x = b)
/// L^T is upper triangular, so use back substitution
fn tpsv_lower_trans<T: Field>(
    ap: &[T],
    x: &mut [T],
    n: usize,
    diag: TpsvDiag,
    conj: bool,
) -> Result<(), TpsvError> {
    for i in (0..n).rev() {
        // Subtract known terms: L^T[i,j] = L[j,i] for j > i
        for j in (i + 1)..n {
            let kji = lower_packed_index(j, i, n);
            let val = if conj { ap[kji].conj() } else { ap[kji] };
            x[i] -= val * x[j];
        }

        // Divide by diagonal
        if diag == TpsvDiag::NonUnit {
            let kii = lower_packed_index(i, i, n);
            let diag_val = if conj { ap[kii].conj() } else { ap[kii] };
            if diag_val.is_zero() {
                return Err(TpsvError::Singular);
            }
            x[i] /= diag_val;
        }
    }
    Ok(())
}

/// Upper triangular packed solve with transpose: U^T * x = b (or U^H * x = b)
/// U^T is lower triangular, so use forward substitution
fn tpsv_upper_trans<T: Field>(
    ap: &[T],
    x: &mut [T],
    n: usize,
    diag: TpsvDiag,
    conj: bool,
) -> Result<(), TpsvError> {
    for i in 0..n {
        // Subtract known terms: U^T[i,j] = U[j,i] for j < i
        for j in 0..i {
            let kji = upper_packed_index(j, i);
            let val = if conj { ap[kji].conj() } else { ap[kji] };
            x[i] -= val * x[j];
        }

        // Divide by diagonal
        if diag == TpsvDiag::NonUnit {
            let kii = upper_packed_index(i, i);
            let diag_val = if conj { ap[kii].conj() } else { ap[kii] };
            if diag_val.is_zero() {
                return Err(TpsvError::Singular);
            }
            x[i] /= diag_val;
        }
    }
    Ok(())
}

/// Non-in-place version: returns a new vector with the solution.
pub fn tpsv_new<T: Field>(
    uplo: TpsvUplo,
    trans: TpsvTrans,
    diag: TpsvDiag,
    n: usize,
    ap: &[T],
    b: &[T],
) -> Result<Vec<T>, TpsvError> {
    let mut result = b.to_vec();
    tpsv(uplo, trans, diag, n, ap, &mut result)?;
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
    fn test_tpsv_lower_notrans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // Lower packed: [2, 1, 2, 3, 1, 4]
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];

        // Solve L * x = b where b = L * [1, 2, 3] = [2, 7, 16]
        let mut x = [2.0, 7.0, 16.0];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
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
    fn test_tpsv_upper_notrans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // Upper packed: [2, 1, 3, 2, 1, 4]
        let ap = [2.0f64, 1.0, 3.0, 2.0, 1.0, 4.0];

        // Solve U * x = b where b = U * [1, 2, 3] = [10, 9, 12]
        let mut x = [10.0, 9.0, 12.0];

        tpsv(
            TpsvUplo::Upper,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
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
    fn test_tpsv_lower_trans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // L^T = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // Lower packed: [2, 1, 2, 3, 1, 4]
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];

        // Solve L^T * x = b where b = L^T * [1, 2, 3] = [10, 9, 12]
        let mut x = [10.0, 9.0, 12.0];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::Trans,
            TpsvDiag::NonUnit,
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
    fn test_tpsv_upper_trans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // U^T = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // Upper packed: [2, 1, 3, 2, 1, 4]
        let ap = [2.0f64, 1.0, 3.0, 2.0, 1.0, 4.0];

        // Solve U^T * x = b where b = U^T * [1, 2, 3] = [2, 7, 16]
        let mut x = [2.0, 7.0, 16.0];

        tpsv(
            TpsvUplo::Upper,
            TpsvTrans::Trans,
            TpsvDiag::NonUnit,
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
    fn test_tpsv_unit_diagonal() {
        // L with unit diagonal
        // L = [[1, 0, 0], [1, 1, 0], [2, 1, 1]]
        // Lower packed: [ignored, 1, 2, ignored, 1, ignored]
        let ap = [99.0f64, 1.0, 2.0, 88.0, 1.0, 77.0];

        // Solve L * x = b where b = L * [1, 2, 3] = [1, 3, 7]
        let mut x = [1.0, 3.0, 7.0];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::Unit,
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
    fn test_tpsv_identity() {
        // Identity matrix (upper packed): [1, 0, 1, 0, 0, 1]
        let ap = [1.0f64, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut x = [1.0, 2.0, 3.0];

        tpsv(
            TpsvUplo::Upper,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
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
    fn test_tpsv_2x2() {
        // L = [[2, 0], [3, 4]]
        // Lower packed: [2, 3, 4]
        let ap = [2.0f64, 3.0, 4.0];

        // Solve L * x = b where b = L * [1, 2] = [2, 11]
        let mut x = [2.0, 11.0];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!(approx_eq(x[0], 1.0));
        assert!(approx_eq(x[1], 2.0));
    }

    #[test]
    fn test_tpsv_1x1() {
        let ap = [5.0f64];
        let mut x = [15.0];

        tpsv(
            TpsvUplo::Upper,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            1,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!(approx_eq(x[0], 3.0));
    }

    #[test]
    fn test_tpsv_empty() {
        let ap: [f64; 0] = [];
        let mut x: [f64; 0] = [];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            0,
            &ap,
            &mut x,
        )
        .unwrap();
    }

    #[test]
    fn test_tpsv_dimension_errors() {
        // Wrong packed size
        let ap = [1.0f64, 2.0, 3.0, 4.0]; // Should be 6 for n=3
        let mut x = [1.0, 2.0, 3.0];
        assert_eq!(
            tpsv(
                TpsvUplo::Lower,
                TpsvTrans::NoTrans,
                TpsvDiag::NonUnit,
                3,
                &ap,
                &mut x
            ),
            Err(TpsvError::InvalidPackedSize)
        );

        // Wrong x dimension
        let ap = [1.0f64, 2.0, 3.0];
        let mut x = [1.0, 2.0, 3.0];
        assert_eq!(
            tpsv(
                TpsvUplo::Lower,
                TpsvTrans::NoTrans,
                TpsvDiag::NonUnit,
                2,
                &ap,
                &mut x
            ),
            Err(TpsvError::DimensionMismatch)
        );
    }

    #[test]
    fn test_tpsv_singular() {
        // L with zero on diagonal
        let ap = [2.0f64, 1.0, 0.0]; // Second diagonal is zero
        let mut x = [2.0, 1.0];

        assert_eq!(
            tpsv(
                TpsvUplo::Lower,
                TpsvTrans::NoTrans,
                TpsvDiag::NonUnit,
                2,
                &ap,
                &mut x
            ),
            Err(TpsvError::Singular)
        );
    }

    #[test]
    fn test_tpsv_new() {
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let b = [2.0, 7.0, 16.0];

        let x = tpsv_new(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            3,
            &ap,
            &b,
        )
        .unwrap();

        assert!(approx_eq(x[0], 1.0));
        assert!(approx_eq(x[1], 2.0));
        assert!(approx_eq(x[2], 3.0));

        // Original unchanged
        assert_eq!(b, [2.0, 7.0, 16.0]);
    }

    #[test]
    fn test_tpsv_complex() {
        // L = [[2, 0], [1+i, 3]]
        // Lower packed: [2, 1+i, 3]
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];

        // Solve L * x = b where b = L * [1, 1] = [2, (1+i)+3] = [2, 4+i]
        let mut x = [c(2.0, 0.0), c(4.0, 1.0)];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!(approx_eq_c(x[0], c(1.0, 0.0)));
        assert!(approx_eq_c(x[1], c(1.0, 0.0)));
    }

    #[test]
    fn test_tpsv_complex_conjtrans() {
        // L = [[2, 0], [1+i, 3]]
        // L^H = [[2, 1-i], [0, 3]]
        // Lower packed: [2, 1+i, 3]
        let ap = [c(2.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)];

        // Solve L^H * x = b where b = L^H * [1, 1] = [2+(1-i), 3] = [3-i, 3]
        let mut x = [c(3.0, -1.0), c(3.0, 0.0)];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::ConjTrans,
            TpsvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!(approx_eq_c(x[0], c(1.0, 0.0)));
        assert!(approx_eq_c(x[1], c(1.0, 0.0)));
    }

    #[test]
    fn test_tpsv_f32() {
        let ap = [2.0f32, 3.0, 4.0];
        let mut x = [2.0f32, 11.0];

        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            2,
            &ap,
            &mut x,
        )
        .unwrap();

        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_tpsv_roundtrip() {
        // Test that tpmv and tpsv are inverses
        let ap = [2.0f64, 1.0, 2.0, 3.0, 1.0, 4.0];
        let original = [1.0, 2.0, 3.0];

        // First apply tpmv
        let mut x = original.to_vec();
        super::super::tpmv::tpmv(
            super::super::tpmv::TpmvUplo::Lower,
            super::super::tpmv::TpmvTrans::NoTrans,
            super::super::tpmv::TpmvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        // Then apply tpsv to get back original
        tpsv(
            TpsvUplo::Lower,
            TpsvTrans::NoTrans,
            TpsvDiag::NonUnit,
            3,
            &ap,
            &mut x,
        )
        .unwrap();

        for i in 0..3 {
            assert!(approx_eq(x[i], original[i]));
        }
    }
}
