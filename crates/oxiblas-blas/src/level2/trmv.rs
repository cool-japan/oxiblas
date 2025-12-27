//! TRMV: Triangular matrix-vector multiply.
//!
//! Computes x = op(A) * x where A is a triangular matrix.

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Specifies which triangle of the matrix is used (Upper/Lower).
///
/// This follows the standard BLAS naming convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmvUplo {
    /// Lower triangular matrix (elements below and on diagonal).
    Lower,
    /// Upper triangular matrix (elements above and on diagonal).
    Upper,
}

/// Specifies the operation on the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmvOp {
    /// No transpose: x = A * x
    NoTrans,
    /// Transpose: x = A^T * x
    Trans,
    /// Conjugate transpose: x = A^H * x (same as Trans for real types)
    ConjTrans,
}

/// Specifies whether the diagonal is assumed to be unit (all ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagKind {
    /// Use the actual diagonal elements from the matrix.
    NonUnit,
    /// Assume diagonal elements are all 1.0 (don't read them).
    Unit,
}

/// Error type for triangular matrix-vector multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrmvError {
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch between matrix and vector.
    DimensionMismatch,
}

impl core::fmt::Display for TrmvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between matrix and vector"),
        }
    }
}

impl std::error::Error for TrmvError {}

/// Computes the triangular matrix-vector product x = op(A) * x in-place.
///
/// # Arguments
///
/// * `a` - The triangular matrix A (only the specified triangle is used)
/// * `x` - The vector x (overwritten with the result)
/// * `uplo` - Specifies whether A is lower or upper triangular
/// * `op` - The operation to apply (no transpose, transpose, or conjugate transpose)
/// * `diag` - Whether the diagonal is assumed to be unit
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{trmv, TrmvUplo, TrmvOp, DiagKind};
/// use oxiblas_matrix::Mat;
///
/// // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
/// let l = Mat::from_rows(&[
///     &[2.0f64, 0.0, 0.0],
///     &[1.0, 3.0, 0.0],
///     &[2.0, 1.0, 4.0],
/// ]);
/// let mut x = [1.0, 2.0, 3.0];
///
/// // x = L * x = [2*1, 1*1+3*2, 2*1+1*2+4*3] = [2, 7, 16]
/// trmv(l.as_ref(), &mut x, TrmvUplo::Lower, TrmvOp::NoTrans, DiagKind::NonUnit).unwrap();
///
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 7.0).abs() < 1e-10);
/// assert!((x[2] - 16.0).abs() < 1e-10);
/// ```
pub fn trmv<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    uplo: TrmvUplo,
    op: TrmvOp,
    diag: DiagKind,
) -> Result<(), TrmvError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(TrmvError::NotSquare);
    }
    if n != x.len() {
        return Err(TrmvError::DimensionMismatch);
    }

    if n == 0 {
        return Ok(());
    }

    match (uplo, op) {
        (TrmvUplo::Lower, TrmvOp::NoTrans) => {
            trmv_lower_notrans(a, x, diag);
        }
        (TrmvUplo::Upper, TrmvOp::NoTrans) => {
            trmv_upper_notrans(a, x, diag);
        }
        (TrmvUplo::Lower, TrmvOp::Trans | TrmvOp::ConjTrans) => {
            // L^T is upper triangular, process as upper
            trmv_lower_trans(a, x, diag, matches!(op, TrmvOp::ConjTrans));
        }
        (TrmvUplo::Upper, TrmvOp::Trans | TrmvOp::ConjTrans) => {
            // U^T is lower triangular, process as lower
            trmv_upper_trans(a, x, diag, matches!(op, TrmvOp::ConjTrans));
        }
    }

    Ok(())
}

/// Lower triangular, no transpose: x = L * x
/// Process from bottom to top to avoid overwriting x values we need.
fn trmv_lower_notrans<T: Field>(a: MatRef<'_, T>, x: &mut [T], diag: DiagKind) {
    let n = x.len();

    for i in (0..n).rev() {
        let mut sum = if diag == DiagKind::Unit {
            x[i]
        } else {
            a[(i, i)] * x[i]
        };

        for j in 0..i {
            sum += a[(i, j)] * x[j];
        }
        x[i] = sum;
    }
}

/// Upper triangular, no transpose: x = U * x
/// Process from top to bottom to avoid overwriting x values we need.
fn trmv_upper_notrans<T: Field>(a: MatRef<'_, T>, x: &mut [T], diag: DiagKind) {
    let n = x.len();

    for i in 0..n {
        let mut sum = if diag == DiagKind::Unit {
            x[i]
        } else {
            a[(i, i)] * x[i]
        };

        for j in (i + 1)..n {
            sum += a[(i, j)] * x[j];
        }
        x[i] = sum;
    }
}

/// Lower triangular with transpose: x = L^T * x (or L^H * x)
/// L^T is upper triangular, so process from top to bottom.
fn trmv_lower_trans<T: Field>(a: MatRef<'_, T>, x: &mut [T], diag: DiagKind, conj: bool) {
    let n = x.len();

    for i in 0..n {
        let mut sum = if diag == DiagKind::Unit {
            x[i]
        } else {
            let diag_val = if conj { a[(i, i)].conj() } else { a[(i, i)] };
            diag_val * x[i]
        };

        // L^T[i,j] = L[j,i] for j > i
        for j in (i + 1)..n {
            let val = if conj { a[(j, i)].conj() } else { a[(j, i)] };
            sum += val * x[j];
        }
        x[i] = sum;
    }
}

/// Upper triangular with transpose: x = U^T * x (or U^H * x)
/// U^T is lower triangular, so process from bottom to top.
fn trmv_upper_trans<T: Field>(a: MatRef<'_, T>, x: &mut [T], diag: DiagKind, conj: bool) {
    let n = x.len();

    for i in (0..n).rev() {
        let mut sum = if diag == DiagKind::Unit {
            x[i]
        } else {
            let diag_val = if conj { a[(i, i)].conj() } else { a[(i, i)] };
            diag_val * x[i]
        };

        // U^T[i,j] = U[j,i] for j < i
        for j in 0..i {
            let val = if conj { a[(j, i)].conj() } else { a[(j, i)] };
            sum += val * x[j];
        }
        x[i] = sum;
    }
}

/// Non-in-place version: returns a new vector with the result.
///
/// # Arguments
///
/// * `a` - The triangular matrix A
/// * `x` - The input vector x
/// * `uplo` - Specifies whether A is lower or upper triangular
/// * `op` - The operation to apply
/// * `diag` - Whether the diagonal is assumed to be unit
///
/// # Returns
///
/// A new vector containing op(A) * x.
pub fn trmv_alloc<T: Field>(
    a: MatRef<'_, T>,
    x: &[T],
    uplo: TrmvUplo,
    op: TrmvOp,
    diag: DiagKind,
) -> Result<Vec<T>, TrmvError> {
    let mut result = x.to_vec();
    trmv(a, &mut result, uplo, op, diag)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_trmv_lower_notrans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // x = [1, 2, 3]
        // L*x = [2*1, 1*1+3*2, 2*1+1*2+4*3] = [2, 7, 16]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let mut x = [1.0, 2.0, 3.0];

        trmv(
            l.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        )
        .unwrap();

        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
        assert!((x[2] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_upper_notrans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // x = [1, 2, 3]
        // U*x = [2*1+1*2+2*3, 3*2+1*3, 4*3] = [10, 9, 12]
        let u = Mat::from_rows(&[&[2.0f64, 1.0, 2.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 4.0]]);
        let mut x = [1.0, 2.0, 3.0];

        trmv(
            u.as_ref(),
            &mut x,
            TrmvUplo::Upper,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        )
        .unwrap();

        assert!((x[0] - 10.0).abs() < 1e-10);
        assert!((x[1] - 9.0).abs() < 1e-10);
        assert!((x[2] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_lower_trans() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // L^T = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // x = [1, 2, 3]
        // L^T*x = [2*1+1*2+2*3, 3*2+1*3, 4*3] = [10, 9, 12]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let mut x = [1.0, 2.0, 3.0];

        trmv(
            l.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::Trans,
            DiagKind::NonUnit,
        )
        .unwrap();

        assert!((x[0] - 10.0).abs() < 1e-10);
        assert!((x[1] - 9.0).abs() < 1e-10);
        assert!((x[2] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_upper_trans() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // U^T = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // x = [1, 2, 3]
        // U^T*x = [2*1, 1*1+3*2, 2*1+1*2+4*3] = [2, 7, 16]
        let u = Mat::from_rows(&[&[2.0f64, 1.0, 2.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 4.0]]);
        let mut x = [1.0, 2.0, 3.0];

        trmv(
            u.as_ref(),
            &mut x,
            TrmvUplo::Upper,
            TrmvOp::Trans,
            DiagKind::NonUnit,
        )
        .unwrap();

        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
        assert!((x[2] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_unit_diagonal() {
        // L with unit diagonal (diagonal is ignored)
        // L = [[*, 0, 0], [1, *, 0], [2, 1, *]]
        // x = [1, 2, 3]
        // L*x = [1*1, 1*1+1*2, 2*1+1*2+1*3] = [1, 3, 7]
        let l = Mat::from_rows(&[
            &[99.0f64, 0.0, 0.0], // diagonal ignored
            &[1.0, 88.0, 0.0],
            &[2.0, 1.0, 77.0],
        ]);
        let mut x = [1.0, 2.0, 3.0];

        trmv(
            l.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::Unit,
        )
        .unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
        assert!((x[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_identity() {
        let eye = Mat::<f64>::eye(3);
        let mut x = [1.0, 2.0, 3.0];

        trmv(
            eye.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        )
        .unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trmv_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let mut x = [1.0, 2.0];

        let result = trmv(
            a.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        );
        assert!(matches!(result, Err(TrmvError::NotSquare)));
    }

    #[test]
    fn test_trmv_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let mut x = [1.0, 2.0, 3.0];

        let result = trmv(
            a.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        );
        assert!(matches!(result, Err(TrmvError::DimensionMismatch)));
    }

    #[test]
    fn test_trmv_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let mut x: [f64; 0] = [];

        trmv(
            a.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        )
        .unwrap();
    }

    #[test]
    fn test_trmv_f32() {
        let l = Mat::from_rows(&[&[2.0f32, 0.0], &[1.0, 3.0]]);
        let mut x = [1.0f32, 2.0];

        trmv(
            l.as_ref(),
            &mut x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        )
        .unwrap();

        // L*x = [2*1, 1*1+3*2] = [2, 7]
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_trmv_alloc() {
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let x = [1.0, 2.0, 3.0];

        let result = trmv_alloc(
            l.as_ref(),
            &x,
            TrmvUplo::Lower,
            TrmvOp::NoTrans,
            DiagKind::NonUnit,
        )
        .unwrap();

        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
        assert!((result[2] - 16.0).abs() < 1e-10);

        // Original x unchanged
        assert_eq!(x, [1.0, 2.0, 3.0]);
    }
}
