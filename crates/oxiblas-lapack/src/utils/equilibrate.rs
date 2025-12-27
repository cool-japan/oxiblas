//! Matrix Equilibration.
//!
//! Computes row and column scaling factors to equilibrate a matrix,
//! which can improve the conditioning of linear systems.
//!
//! This module provides LAPACK-style equilibration functions:
//! - **geequ**: Row and column scaling for general matrices
//! - **geequb**: Similar but with bounded scaling factors (more robust)
//! - **syequ**: Row and column scaling for symmetric matrices
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::utils::equilibrate::{geequ, EquilibrationInfo};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1e-10f64, 1.0],
//!     &[1.0, 1e10],
//! ]);
//!
//! let info = geequ(a.as_ref()).unwrap();
//! println!("Row scale: {:?}", info.row_scale);
//! println!("Col scale: {:?}", info.col_scale);
//! println!("Row condition: {}", info.row_cond);
//! println!("Col condition: {}", info.col_cond);
//! ```

use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for equilibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquilibrateError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix has a zero row.
    ZeroRow {
        /// Index of the zero row.
        index: usize,
    },
    /// Matrix has a zero column.
    ZeroColumn {
        /// Index of the zero column.
        index: usize,
    },
}

impl core::fmt::Display for EquilibrateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::ZeroRow { index } => write!(f, "Matrix has zero row at index {index}"),
            Self::ZeroColumn { index } => write!(f, "Matrix has zero column at index {index}"),
        }
    }
}

impl std::error::Error for EquilibrateError {}

/// Information returned by equilibration routines.
#[derive(Debug, Clone)]
pub struct EquilibrationInfo<T: Scalar> {
    /// Row scaling factors `R[i]` such that diag(R)*A has equal row norms.
    pub row_scale: Vec<T>,
    /// Column scaling factors `C[j]` such that A*diag(C) has equal column norms.
    pub col_scale: Vec<T>,
    /// Row condition number: max(R)/min(R).
    pub row_cond: T,
    /// Column condition number: max(C)/min(C).
    pub col_cond: T,
    /// Maximum absolute element of the scaled matrix.
    pub amax: T,
}

/// Computes row and column scaling for a general matrix (LAPACK DGEEQU).
///
/// Computes row and column scaling factors R and C such that:
/// - Each row of diag(R)*A has similar norm
/// - Each column of A*diag(C) has similar norm
///
/// The scaling is: `R[i] = 1/max_j |A[i,j]|`, `C[j] = 1/max_i |A[i,j]|`
///
/// # Arguments
///
/// * `a` - Input matrix (m×n)
///
/// # Returns
///
/// `EquilibrationInfo` containing scaling factors and condition information.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::equilibrate::geequ;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[100.0f64, 1.0],
///     &[1.0, 0.01],
/// ]);
///
/// let info = geequ(a.as_ref()).unwrap();
/// // Row scales should be ~0.01 and ~1.0 to balance rows
/// ```
pub fn geequ<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<EquilibrationInfo<T>, EquilibrateError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(EquilibrateError::EmptyMatrix);
    }

    let mut row_scale = vec![T::zero(); m];
    let mut col_scale = vec![T::zero(); n];

    // Compute row scaling: `R[i] = 1/max_j |A[i,j]|`
    let mut row_min = T::one() / <T as Scalar>::epsilon();
    let mut row_max = T::zero();

    for i in 0..m {
        let mut row_norm = T::zero();
        for j in 0..n {
            let abs_val = Scalar::abs(a[(i, j)]);
            if abs_val > row_norm {
                row_norm = abs_val;
            }
        }

        if row_norm == T::zero() {
            return Err(EquilibrateError::ZeroRow { index: i });
        }

        row_scale[i] = T::one() / row_norm;

        if row_scale[i] < row_min {
            row_min = row_scale[i];
        }
        if row_scale[i] > row_max {
            row_max = row_scale[i];
        }
    }

    let row_cond = row_max / row_min;

    // Compute column scaling: `C[j] = 1/max_i |A[i,j]|`
    let mut col_min = T::one() / <T as Scalar>::epsilon();
    let mut col_max = T::zero();

    for j in 0..n {
        let mut col_norm = T::zero();
        for i in 0..m {
            let abs_val = Scalar::abs(a[(i, j)]);
            if abs_val > col_norm {
                col_norm = abs_val;
            }
        }

        if col_norm == T::zero() {
            return Err(EquilibrateError::ZeroColumn { index: j });
        }

        col_scale[j] = T::one() / col_norm;

        if col_scale[j] < col_min {
            col_min = col_scale[j];
        }
        if col_scale[j] > col_max {
            col_max = col_scale[j];
        }
    }

    let col_cond = col_max / col_min;

    // Compute amax (maximum absolute element)
    let mut amax = T::zero();
    for i in 0..m {
        for j in 0..n {
            let abs_val = Scalar::abs(a[(i, j)]);
            if abs_val > amax {
                amax = abs_val;
            }
        }
    }

    Ok(EquilibrationInfo {
        row_scale,
        col_scale,
        row_cond,
        col_cond,
        amax,
    })
}

/// Computes row and column scaling with bounded factors (LAPACK DGEEQUB).
///
/// Similar to `geequ`, but uses a more robust scaling that prevents
/// overflow/underflow by bounding the scaling factors.
///
/// The scaling is designed to be safer for very ill-conditioned matrices.
///
/// # Arguments
///
/// * `a` - Input matrix (m×n)
///
/// # Returns
///
/// `EquilibrationInfo` containing scaling factors and condition information.
pub fn geequb<T: Field + Real + bytemuck::Zeroable + FromPrimitive>(
    a: MatRef<'_, T>,
) -> Result<EquilibrationInfo<T>, EquilibrateError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(EquilibrateError::EmptyMatrix);
    }

    // Compute safe range for scaling factors
    let radix = T::from_f64(2.0).unwrap_or(T::one() + T::one());
    let eps = <T as Scalar>::epsilon();
    let small = eps / radix;
    let big = T::one() / small;

    let mut row_scale = vec![T::zero(); m];
    let mut col_scale = vec![T::zero(); n];

    // Compute row scaling
    let mut row_min = big;
    let mut row_max = T::zero();

    for i in 0..m {
        let mut row_norm = T::zero();
        for j in 0..n {
            let abs_val = Scalar::abs(a[(i, j)]);
            if abs_val > row_norm {
                row_norm = abs_val;
            }
        }

        if row_norm == T::zero() {
            return Err(EquilibrateError::ZeroRow { index: i });
        }

        // Use bounded reciprocal
        let scale = T::one() / T::max(row_norm, small);
        row_scale[i] = T::min(scale, big);

        if row_scale[i] < row_min {
            row_min = row_scale[i];
        }
        if row_scale[i] > row_max {
            row_max = row_scale[i];
        }
    }

    let row_cond = if row_min > T::zero() {
        row_max / row_min
    } else {
        big
    };

    // Compute column scaling on the row-scaled matrix
    let mut col_min = big;
    let mut col_max = T::zero();

    for j in 0..n {
        let mut col_norm = T::zero();
        for i in 0..m {
            let abs_val = Scalar::abs(a[(i, j)]) * row_scale[i];
            if abs_val > col_norm {
                col_norm = abs_val;
            }
        }

        if col_norm == T::zero() {
            return Err(EquilibrateError::ZeroColumn { index: j });
        }

        let scale = T::one() / T::max(col_norm, small);
        col_scale[j] = T::min(scale, big);

        if col_scale[j] < col_min {
            col_min = col_scale[j];
        }
        if col_scale[j] > col_max {
            col_max = col_scale[j];
        }
    }

    let col_cond = if col_min > T::zero() {
        col_max / col_min
    } else {
        big
    };

    // Compute amax
    let mut amax = T::zero();
    for i in 0..m {
        for j in 0..n {
            let abs_val = Scalar::abs(a[(i, j)]);
            if abs_val > amax {
                amax = abs_val;
            }
        }
    }

    Ok(EquilibrationInfo {
        row_scale,
        col_scale,
        row_cond,
        col_cond,
        amax,
    })
}

/// Computes scaling for symmetric matrices (LAPACK DSYEQU).
///
/// For symmetric matrices, the same scaling is applied to rows and columns:
/// `D[i] = 1/sqrt(|A[i,i]|)`
///
/// This preserves symmetry of the scaled matrix.
///
/// # Arguments
///
/// * `a` - Symmetric input matrix (n×n)
///
/// # Returns
///
/// Symmetric scaling factors and condition information.
pub fn syequ<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<EquilibrationInfo<T>, EquilibrateError> {
    let n = a.nrows();

    if n == 0 {
        return Err(EquilibrateError::EmptyMatrix);
    }

    let mut scale = vec![T::zero(); n];
    let mut scale_min = T::one() / <T as Scalar>::epsilon();
    let mut scale_max = T::zero();

    for i in 0..n {
        let diag = Scalar::abs(a[(i, i)]);
        if diag == T::zero() {
            return Err(EquilibrateError::ZeroRow { index: i });
        }

        scale[i] = T::one() / Real::sqrt(diag);

        if scale[i] < scale_min {
            scale_min = scale[i];
        }
        if scale[i] > scale_max {
            scale_max = scale[i];
        }
    }

    let cond = scale_max / scale_min;

    // Compute amax
    let mut amax = T::zero();
    for i in 0..n {
        let abs_val = Scalar::abs(a[(i, i)]);
        if abs_val > amax {
            amax = abs_val;
        }
    }

    Ok(EquilibrationInfo {
        row_scale: scale.clone(),
        col_scale: scale,
        row_cond: cond,
        col_cond: cond,
        amax,
    })
}

/// Applies row scaling to a matrix: B = diag(R) * A.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `row_scale` - Row scaling factors
///
/// # Returns
///
/// Scaled matrix.
#[must_use]
pub fn apply_row_scale<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    row_scale: &[T],
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut result = Mat::zeros(m, n);

    for i in 0..m {
        let r = row_scale[i];
        for j in 0..n {
            result[(i, j)] = r * a[(i, j)];
        }
    }

    result
}

/// Applies column scaling to a matrix: B = A * diag(C).
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `col_scale` - Column scaling factors
///
/// # Returns
///
/// Scaled matrix.
#[must_use]
pub fn apply_col_scale<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    col_scale: &[T],
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut result = Mat::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            result[(i, j)] = a[(i, j)] * col_scale[j];
        }
    }

    result
}

/// Applies both row and column scaling: B = diag(R) * A * diag(C).
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `row_scale` - Row scaling factors
/// * `col_scale` - Column scaling factors
///
/// # Returns
///
/// Scaled matrix.
#[must_use]
pub fn apply_scale<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    row_scale: &[T],
    col_scale: &[T],
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut result = Mat::zeros(m, n);

    for i in 0..m {
        let r = row_scale[i];
        for j in 0..n {
            result[(i, j)] = r * a[(i, j)] * col_scale[j];
        }
    }

    result
}

/// Applies row scaling to a right-hand side vector: b_scaled = diag(R) * b.
#[must_use]
pub fn scale_rhs<T: Field + Real + bytemuck::Zeroable>(
    b: MatRef<'_, T>,
    row_scale: &[T],
) -> Mat<T> {
    apply_row_scale(b, row_scale)
}

/// Unscales a solution vector: x = diag(C) * x_scaled.
///
/// After solving the scaled system (R*A*C) * y = R*b,
/// the original solution is x = C * y.
#[must_use]
pub fn unscale_solution<T: Field + Real + bytemuck::Zeroable>(
    x_scaled: MatRef<'_, T>,
    col_scale: &[T],
) -> Mat<T> {
    let m = x_scaled.nrows();
    let n = x_scaled.ncols();
    let mut result = Mat::zeros(m, n);

    for i in 0..m {
        let c = col_scale[i];
        for j in 0..n {
            result[(i, j)] = c * x_scaled[(i, j)];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_geequ_identity() {
        let a: Mat<f64> = Mat::eye(3);

        let info = geequ(a.as_ref()).unwrap();

        // All scales should be 1 for identity matrix
        for i in 0..3 {
            assert!(approx_eq(info.row_scale[i], 1.0, 1e-10));
            assert!(approx_eq(info.col_scale[i], 1.0, 1e-10));
        }
        assert!(approx_eq(info.row_cond, 1.0, 1e-10));
        assert!(approx_eq(info.col_cond, 1.0, 1e-10));
    }

    #[test]
    fn test_geequ_diagonal() {
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 2.0]]);

        let info = geequ(a.as_ref()).unwrap();

        // Row/col scales should be reciprocal of diagonal
        assert!(approx_eq(info.row_scale[0], 0.25, 1e-10));
        assert!(approx_eq(info.row_scale[1], 0.5, 1e-10));
    }

    #[test]
    fn test_geequ_ill_conditioned() {
        let a = Mat::from_rows(&[&[1e-6f64, 1.0], &[1.0, 1e6]]);

        let info = geequ(a.as_ref()).unwrap();

        // Should have large condition numbers
        assert!(info.row_cond > 1e5);
        assert!(info.col_cond > 1e5);
    }

    #[test]
    fn test_geequ_zero_row() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 0.0]]);

        let result = geequ(a.as_ref());
        assert!(matches!(
            result,
            Err(EquilibrateError::ZeroRow { index: 1 })
        ));
    }

    #[test]
    fn test_geequ_zero_column() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[2.0, 0.0]]);

        let result = geequ(a.as_ref());
        assert!(matches!(
            result,
            Err(EquilibrateError::ZeroColumn { index: 1 })
        ));
    }

    #[test]
    fn test_geequb_basic() {
        let a = Mat::from_rows(&[&[100.0f64, 1.0], &[1.0, 0.01]]);

        let info = geequb(a.as_ref()).unwrap();

        // Should have positive scales
        assert!(info.row_scale[0] > 0.0);
        assert!(info.row_scale[1] > 0.0);
        assert!(info.col_scale[0] > 0.0);
        assert!(info.col_scale[1] > 0.0);
    }

    #[test]
    fn test_syequ_symmetric() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 0.0], &[1.0, 9.0, 2.0], &[0.0, 2.0, 16.0]]);

        let info = syequ(a.as_ref()).unwrap();

        // Scale should be 1/sqrt(diag)
        assert!(approx_eq(info.row_scale[0], 0.5, 1e-10)); // 1/sqrt(4) = 0.5
        assert!(approx_eq(info.row_scale[1], 1.0 / 3.0, 1e-10)); // 1/sqrt(9) = 1/3
        assert!(approx_eq(info.row_scale[2], 0.25, 1e-10)); // 1/sqrt(16) = 0.25

        // Row and col scale should be the same
        for i in 0..3 {
            assert!(approx_eq(info.row_scale[i], info.col_scale[i], 1e-10));
        }
    }

    #[test]
    fn test_apply_scale() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[6.0, 8.0]]);
        let row_scale = vec![0.5, 0.25];
        let col_scale = vec![2.0, 0.5];

        let scaled = apply_scale(a.as_ref(), &row_scale, &col_scale);

        // a_scaled[i,j] = row_scale[i] * a[i,j] * col_scale[j]
        assert!(approx_eq(scaled[(0, 0)], 0.5 * 4.0 * 2.0, 1e-10)); // 4.0
        assert!(approx_eq(scaled[(0, 1)], 0.5 * 2.0 * 0.5, 1e-10)); // 0.5
        assert!(approx_eq(scaled[(1, 0)], 0.25 * 6.0 * 2.0, 1e-10)); // 3.0
        assert!(approx_eq(scaled[(1, 1)], 0.25 * 8.0 * 0.5, 1e-10)); // 1.0
    }

    #[test]
    fn test_equilibration_improves_condition() {
        // Create an ill-conditioned matrix
        let a = Mat::from_rows(&[&[1e-8f64, 1.0], &[1.0, 1e8]]);

        let info = geequ(a.as_ref()).unwrap();
        let scaled = apply_scale(a.as_ref(), &info.row_scale, &info.col_scale);

        // Check that scaled matrix has max row/col norms close to 1
        for i in 0..2 {
            let mut row_max = 0.0f64;
            let mut col_max = 0.0f64;
            for j in 0..2 {
                if scaled[(i, j)].abs() > row_max {
                    row_max = scaled[(i, j)].abs();
                }
                if scaled[(j, i)].abs() > col_max {
                    col_max = scaled[(j, i)].abs();
                }
            }
            // After scaling, max element in each row should be ~1
            assert!(row_max <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_scale_unscale_roundtrip() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[6.0, 8.0]]);
        let info = geequ(a.as_ref()).unwrap();

        // Create a "solution" and scale it
        let x = Mat::from_rows(&[&[1.0], &[2.0]]);
        let x_scaled = Mat::from_rows(&[
            &[x[(0, 0)] / info.col_scale[0]],
            &[x[(1, 0)] / info.col_scale[1]],
        ]);

        // Unscale should recover original
        let x_recovered = unscale_solution(x_scaled.as_ref(), &info.col_scale);

        for i in 0..2 {
            assert!(approx_eq(x_recovered[(i, 0)], x[(i, 0)], 1e-10));
        }
    }

    #[test]
    fn test_geequ_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 2.0], &[1.0, 3.0]]);

        let info = geequ(a.as_ref()).unwrap();

        assert!(info.row_scale[0] > 0.0);
        assert!(info.col_scale[0] > 0.0);
    }
}
