//! Triangular system solvers.
//!
//! Solves Ax = b where A is upper or lower triangular.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for triangular solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularSolveError {
    /// Matrix is not square.
    NotSquare,
    /// Matrix and vector dimensions don't match.
    DimensionMismatch,
    /// Matrix has zero diagonal element (singular).
    SingularMatrix,
}

impl core::fmt::Display for TriangularSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatch => write!(f, "Matrix and vector dimensions do not match"),
            Self::SingularMatrix => write!(f, "Matrix is singular (has zero diagonal)"),
        }
    }
}

impl std::error::Error for TriangularSolveError {}

/// Specifies whether the matrix is upper or lower triangular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularKind {
    /// Upper triangular (a_ij = 0 for i > j).
    Upper,
    /// Lower triangular (a_ij = 0 for i < j).
    Lower,
    /// Unit upper triangular (a_ij = 0 for i > j, a_ii = 1).
    UnitUpper,
    /// Unit lower triangular (a_ij = 0 for i < j, a_ii = 1).
    UnitLower,
}

/// Solves a triangular system Ax = b.
///
/// # Arguments
///
/// * `a` - Triangular matrix A (n×n)
/// * `b` - Right-hand side vector b (n×1)
/// * `kind` - Type of triangular matrix
///
/// # Returns
///
/// Solution vector x such that Ax = b.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::solve::{solve_triangular, TriangularKind};
/// use oxiblas_matrix::Mat;
///
/// // Upper triangular system
/// let a = Mat::from_rows(&[
///     &[2.0f64, 3.0],
///     &[0.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[&[11.0], &[8.0]]);
///
/// let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::Upper).unwrap();
/// // x = [2.5, 2.0] because 2*2.5 + 3*2 = 11, 4*2 = 8
/// ```
pub fn solve_triangular<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    kind: TriangularKind,
) -> Result<Mat<T>, TriangularSolveError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(TriangularSolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(TriangularSolveError::DimensionMismatch);
    }

    let eps = <T as Scalar>::epsilon();

    let mut x = Mat::zeros(n, 1);

    match kind {
        TriangularKind::Lower | TriangularKind::UnitLower => {
            // Forward substitution
            for i in 0..n {
                let mut sum = b[(i, 0)];
                for j in 0..i {
                    sum = sum - a[(i, j)] * x[(j, 0)];
                }

                if matches!(kind, TriangularKind::UnitLower) {
                    x[(i, 0)] = sum;
                } else {
                    if Scalar::abs(a[(i, i)]) < eps {
                        return Err(TriangularSolveError::SingularMatrix);
                    }
                    x[(i, 0)] = sum / a[(i, i)];
                }
            }
        }
        TriangularKind::Upper | TriangularKind::UnitUpper => {
            // Back substitution
            for i in (0..n).rev() {
                let mut sum = b[(i, 0)];
                for j in (i + 1)..n {
                    sum = sum - a[(i, j)] * x[(j, 0)];
                }

                if matches!(kind, TriangularKind::UnitUpper) {
                    x[(i, 0)] = sum;
                } else {
                    if Scalar::abs(a[(i, i)]) < eps {
                        return Err(TriangularSolveError::SingularMatrix);
                    }
                    x[(i, 0)] = sum / a[(i, i)];
                }
            }
        }
    }

    Ok(x)
}

/// Solves a triangular system AX = B for multiple right-hand sides.
///
/// # Arguments
///
/// * `a` - Triangular matrix A (n×n)
/// * `b` - Right-hand side matrix B (n×m)
/// * `kind` - Type of triangular matrix
///
/// # Returns
///
/// Solution matrix X (n×m) such that AX = B.
pub fn solve_triangular_multiple<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    kind: TriangularKind,
) -> Result<Mat<T>, TriangularSolveError> {
    let n = a.nrows();
    let m = b.ncols();

    if a.ncols() != n {
        return Err(TriangularSolveError::NotSquare);
    }
    if b.nrows() != n {
        return Err(TriangularSolveError::DimensionMismatch);
    }

    let eps = <T as Scalar>::epsilon();
    let mut x = Mat::zeros(n, m);

    for col in 0..m {
        match kind {
            TriangularKind::Lower | TriangularKind::UnitLower => {
                // Forward substitution
                for i in 0..n {
                    let mut sum = b[(i, col)];
                    for j in 0..i {
                        sum = sum - a[(i, j)] * x[(j, col)];
                    }

                    if matches!(kind, TriangularKind::UnitLower) {
                        x[(i, col)] = sum;
                    } else {
                        if Scalar::abs(a[(i, i)]) < eps {
                            return Err(TriangularSolveError::SingularMatrix);
                        }
                        x[(i, col)] = sum / a[(i, i)];
                    }
                }
            }
            TriangularKind::Upper | TriangularKind::UnitUpper => {
                // Back substitution
                for i in (0..n).rev() {
                    let mut sum = b[(i, col)];
                    for j in (i + 1)..n {
                        sum = sum - a[(i, j)] * x[(j, col)];
                    }

                    if matches!(kind, TriangularKind::UnitUpper) {
                        x[(i, col)] = sum;
                    } else {
                        if Scalar::abs(a[(i, i)]) < eps {
                            return Err(TriangularSolveError::SingularMatrix);
                        }
                        x[(i, col)] = sum / a[(i, i)];
                    }
                }
            }
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_solve_upper_triangular() {
        let a = Mat::from_rows(&[&[2.0f64, 3.0], &[0.0, 4.0]]);
        let b = Mat::from_rows(&[&[11.0], &[8.0]]);

        let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::Upper).unwrap();

        // x = [2.5, 2.0]: 2*2.5 + 3*2 = 11, 4*2 = 8
        assert!(approx_eq(x[(0, 0)], 2.5, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
    }

    #[test]
    fn test_solve_lower_triangular() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[2.0, 4.0]]);
        let b = Mat::from_rows(&[&[6.0], &[12.0]]);

        let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::Lower).unwrap();

        // x = [2, 2]: 3*2 = 6, 2*2 + 4*2 = 12
        assert!(approx_eq(x[(0, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
    }

    #[test]
    fn test_solve_unit_lower() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[2.0, 1.0]]);
        let b = Mat::from_rows(&[&[3.0], &[8.0]]);

        let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::UnitLower).unwrap();

        // x = [3, 2]: 1*3 = 3, 2*3 + 1*2 = 8
        assert!(approx_eq(x[(0, 0)], 3.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
    }

    #[test]
    fn test_solve_unit_upper() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[5.0], &[3.0]]);

        let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::UnitUpper).unwrap();

        // x = [-1, 3]: 1*(-1) + 2*3 = 5, 1*3 = 3
        assert!(approx_eq(x[(0, 0)], -1.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 3.0, 1e-10));
    }

    #[test]
    fn test_solve_triangular_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);
        let b = Mat::from_rows(&[&[14.0], &[23.0], &[18.0]]);

        let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::Upper).unwrap();

        // x = [1, 2, 3]
        assert!(approx_eq(x[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(2, 0)], 3.0, 1e-10));
    }

    #[test]
    fn test_solve_triangular_multiple_rhs() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[4.0, 6.0], &[8.0, 9.0]]);

        let x = solve_triangular_multiple(a.as_ref(), b.as_ref(), TriangularKind::Lower).unwrap();

        // First column: x = [2, 2], second column: x = [3, 2]
        assert!(approx_eq(x[(0, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(0, 1)], 3.0, 1e-10));
        assert!(approx_eq(x[(1, 1)], 2.0, 1e-10));
    }

    #[test]
    fn test_solve_triangular_singular() {
        let a = Mat::from_rows(&[
            &[2.0f64, 3.0],
            &[0.0, 0.0], // Singular
        ]);
        let b = Mat::from_rows(&[&[5.0], &[1.0]]);

        let result = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::Upper);
        assert!(matches!(result, Err(TriangularSolveError::SingularMatrix)));
    }

    #[test]
    fn test_solve_triangular_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 3.0], &[0.0, 4.0]]);
        let b = Mat::from_rows(&[&[11.0f32], &[8.0]]);

        let x = solve_triangular(a.as_ref(), b.as_ref(), TriangularKind::Upper).unwrap();

        assert!((x[(0, 0)] - 2.5).abs() < 1e-5);
        assert!((x[(1, 0)] - 2.0).abs() < 1e-5);
    }
}
