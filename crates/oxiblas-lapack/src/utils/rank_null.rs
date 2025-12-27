//! Rank, nullity, null space, and column space computation.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use crate::svd::{Svd, SvdError};

/// Error type for rank computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankError {
    /// Matrix is empty.
    EmptyMatrix,
    /// SVD computation failed.
    SvdFailed,
}

impl core::fmt::Display for RankError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::SvdFailed => write!(f, "SVD computation failed"),
        }
    }
}

impl std::error::Error for RankError {}

impl From<SvdError> for RankError {
    fn from(e: SvdError) -> Self {
        match e {
            SvdError::EmptyMatrix => Self::EmptyMatrix,
            SvdError::NotConverged => Self::SvdFailed,
        }
    }
}

/// Computes the numerical rank of a matrix.
///
/// The rank is determined by counting singular values above the tolerance.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
/// * `tol` - Tolerance for considering a singular value as zero.
///           If None, uses default tolerance `eps * max(m,n) * σ_max`.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::rank;
/// use oxiblas_matrix::Mat;
///
/// // Full rank 2×2 matrix
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// assert_eq!(rank(a.as_ref(), None).unwrap(), 2);
///
/// // Rank 1 matrix
/// let b = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[2.0, 4.0],
/// ]);
/// assert_eq!(rank(b.as_ref(), None).unwrap(), 1);
/// ```
pub fn rank<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<usize, RankError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Ok(0);
    }

    let svd = Svd::compute(a)?;
    let sigma = svd.singular_values();

    let tolerance = match tol {
        Some(t) => t,
        None => {
            let eps = <T as Scalar>::epsilon();
            let sigma_max = if sigma.is_empty() { T::one() } else { sigma[0] };
            eps * T::from_f64(m.max(n) as f64).unwrap_or(T::one()) * sigma_max
        }
    };

    Ok(svd.rank(tolerance))
}

/// Computes the nullity (dimension of null space) of a matrix.
///
/// nullity(A) = n - rank(A) for an m×n matrix.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
/// * `tol` - Tolerance for rank computation
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::nullity;
/// use oxiblas_matrix::Mat;
///
/// // Rank deficient matrix (rank 1 from 2 columns)
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[2.0, 4.0],
///     &[3.0, 6.0],
/// ]);
///
/// assert_eq!(nullity(a.as_ref(), None).unwrap(), 1);
/// ```
pub fn nullity<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<usize, RankError> {
    let n = a.ncols();
    let r = rank(a, tol)?;
    Ok(n - r)
}

/// Computes an orthonormal basis for the null space of a matrix.
///
/// The null space N(A) consists of all vectors x such that Ax = 0.
/// Returns the columns of V corresponding to zero singular values from SVD.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
/// * `tol` - Tolerance for considering a singular value as zero
///
/// # Returns
///
/// A matrix whose columns form an orthonormal basis for N(A).
/// Returns an empty matrix if the null space is trivial.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::null_space;
/// use oxiblas_matrix::Mat;
///
/// // Rank 1 matrix (2×2), so 1-dimensional null space
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[2.0, 4.0],
/// ]);
///
/// let ns = null_space(a.as_ref(), None).unwrap();
/// assert_eq!(ns.ncols(), 1); // 1-dimensional null space
///
/// // Verify: A * ns ≈ 0
/// let prod_0 = a[(0, 0)] * ns[(0, 0)] + a[(0, 1)] * ns[(1, 0)];
/// let prod_1 = a[(1, 0)] * ns[(0, 0)] + a[(1, 1)] * ns[(1, 0)];
/// assert!(prod_0.abs() < 1e-10);
/// assert!(prod_1.abs() < 1e-10);
/// ```
pub fn null_space<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<Mat<T>, RankError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Ok(Mat::zeros(n, 0));
    }

    let svd = Svd::compute(a)?;
    let sigma = svd.singular_values();
    let vt = svd.vt();

    let tolerance = match tol {
        Some(t) => t,
        None => {
            let eps = <T as Scalar>::epsilon();
            let sigma_max = if sigma.is_empty() { T::one() } else { sigma[0] };
            eps * T::from_f64(m.max(n) as f64).unwrap_or(T::one()) * sigma_max
        }
    };

    let r = svd.rank(tolerance);
    let null_dim = n - r;

    if null_dim == 0 {
        return Ok(Mat::zeros(n, 0));
    }

    // The null space is spanned by the last (n - r) columns of V
    // V = Vt^T, so we want columns r..n of V, which are rows r..n of Vt
    let mut null_basis = Mat::zeros(n, null_dim);

    for i in 0..n {
        for j in 0..null_dim {
            // V[i, r+j] = Vt[r+j, i]
            null_basis[(i, j)] = vt[(r + j, i)];
        }
    }

    Ok(null_basis)
}

/// Computes an orthonormal basis for the column space (range) of a matrix.
///
/// The column space R(A) is the span of A's columns.
/// Returns the columns of U corresponding to non-zero singular values from SVD.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
/// * `tol` - Tolerance for considering a singular value as zero
///
/// # Returns
///
/// A matrix whose columns form an orthonormal basis for R(A).
/// Returns an empty matrix if the column space is trivial.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::col_space;
/// use oxiblas_matrix::Mat;
///
/// // Rank 2 matrix (3×3)
/// let a = Mat::from_rows(&[
///     &[1.0f64, 0.0, 1.0],
///     &[0.0, 1.0, 1.0],
///     &[1.0, 1.0, 2.0],
/// ]);
///
/// let cs = col_space(a.as_ref(), None).unwrap();
/// assert_eq!(cs.ncols(), 2); // 2-dimensional column space (rank 2)
/// ```
pub fn col_space<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<Mat<T>, RankError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Ok(Mat::zeros(m, 0));
    }

    let svd = Svd::compute(a)?;
    let sigma = svd.singular_values();
    let u = svd.u();

    let tolerance = match tol {
        Some(t) => t,
        None => {
            let eps = <T as Scalar>::epsilon();
            let sigma_max = if sigma.is_empty() { T::one() } else { sigma[0] };
            eps * T::from_f64(m.max(n) as f64).unwrap_or(T::one()) * sigma_max
        }
    };

    let r = svd.rank(tolerance);

    if r == 0 {
        return Ok(Mat::zeros(m, 0));
    }

    // The column space is spanned by the first r columns of U
    let mut col_basis = Mat::zeros(m, r);

    for i in 0..m {
        for j in 0..r {
            col_basis[(i, j)] = u[(i, j)];
        }
    }

    Ok(col_basis)
}

/// Computes an orthonormal basis for the row space of a matrix.
///
/// The row space is the column space of A^T.
/// Returns the columns of V corresponding to non-zero singular values.
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
/// * `tol` - Tolerance for considering a singular value as zero
///
/// # Returns
///
/// A matrix whose columns form an orthonormal basis for the row space.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::row_space;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let rs = row_space(a.as_ref(), None).unwrap();
/// assert_eq!(rs.ncols(), 2); // Full row rank
/// assert_eq!(rs.nrows(), 2); // 2 rows
/// ```
pub fn row_space<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<Mat<T>, RankError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Ok(Mat::zeros(n, 0));
    }

    let svd = Svd::compute(a)?;
    let sigma = svd.singular_values();
    let vt = svd.vt();

    let tolerance = match tol {
        Some(t) => t,
        None => {
            let eps = <T as Scalar>::epsilon();
            let sigma_max = if sigma.is_empty() { T::one() } else { sigma[0] };
            eps * T::from_f64(m.max(n) as f64).unwrap_or(T::one()) * sigma_max
        }
    };

    let r = svd.rank(tolerance);

    if r == 0 {
        return Ok(Mat::zeros(n, 0));
    }

    // The row space is spanned by the first r columns of V
    // V[i, j] = Vt[j, i]
    let mut row_basis = Mat::zeros(n, r);

    for i in 0..n {
        for j in 0..r {
            row_basis[(i, j)] = vt[(j, i)];
        }
    }

    Ok(row_basis)
}

/// Computes the left null space (cokernel) of a matrix.
///
/// The left null space N(A^T) consists of all vectors y such that y^T A = 0
/// (equivalently, A^T y = 0).
///
/// # Arguments
///
/// * `a` - Matrix A (m×n)
/// * `tol` - Tolerance for considering a singular value as zero
///
/// # Returns
///
/// A matrix whose columns form an orthonormal basis for N(A^T).
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::left_null_space;
/// use oxiblas_matrix::Mat;
///
/// // Rank 1 matrix (2 rows), so 1-dimensional left null space
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[2.0, 4.0],
/// ]);
///
/// let lns = left_null_space(a.as_ref(), None).unwrap();
/// assert_eq!(lns.ncols(), 1); // 1-dimensional left null space
/// ```
pub fn left_null_space<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<Mat<T>, RankError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Ok(Mat::zeros(m, 0));
    }

    let svd = Svd::compute(a)?;
    let sigma = svd.singular_values();
    let u = svd.u();

    let tolerance = match tol {
        Some(t) => t,
        None => {
            let eps = <T as Scalar>::epsilon();
            let sigma_max = if sigma.is_empty() { T::one() } else { sigma[0] };
            eps * T::from_f64(m.max(n) as f64).unwrap_or(T::one()) * sigma_max
        }
    };

    let r = svd.rank(tolerance);
    let left_null_dim = m - r;

    if left_null_dim == 0 {
        return Ok(Mat::zeros(m, 0));
    }

    // The left null space is spanned by the last (m - r) columns of U
    let mut left_null_basis = Mat::zeros(m, left_null_dim);

    for i in 0..m {
        for j in 0..left_null_dim {
            left_null_basis[(i, j)] = u[(i, r + j)];
        }
    }

    Ok(left_null_basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_rank_full() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        assert_eq!(rank(a.as_ref(), None).unwrap(), 2);
    }

    #[test]
    fn test_rank_deficient() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        assert_eq!(rank(a.as_ref(), None).unwrap(), 1);
    }

    #[test]
    fn test_rank_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        assert_eq!(rank(eye.as_ref(), None).unwrap(), 3);
    }

    #[test]
    fn test_rank_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        assert_eq!(rank(a.as_ref(), None).unwrap(), 2);
    }

    #[test]
    fn test_rank_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        assert_eq!(rank(a.as_ref(), None).unwrap(), 2);
    }

    #[test]
    fn test_nullity() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        assert_eq!(nullity(a.as_ref(), None).unwrap(), 1);
    }

    #[test]
    fn test_nullity_full_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        assert_eq!(nullity(a.as_ref(), None).unwrap(), 0);
    }

    #[test]
    fn test_null_space_rank_1() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let ns = null_space(a.as_ref(), None).unwrap();
        assert_eq!(ns.ncols(), 1);
        assert_eq!(ns.nrows(), 2);

        // Verify A * ns ≈ 0
        let prod_0 = a[(0, 0)] * ns[(0, 0)] + a[(0, 1)] * ns[(1, 0)];
        let prod_1 = a[(1, 0)] * ns[(0, 0)] + a[(1, 1)] * ns[(1, 0)];
        assert!(approx_eq(prod_0, 0.0, 1e-10));
        assert!(approx_eq(prod_1, 0.0, 1e-10));
    }

    #[test]
    fn test_null_space_empty() {
        // Full rank matrix has trivial null space
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let ns = null_space(a.as_ref(), None).unwrap();
        assert_eq!(ns.ncols(), 0);
    }

    #[test]
    fn test_col_space_full_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let cs = col_space(a.as_ref(), None).unwrap();
        assert_eq!(cs.ncols(), 2);
        assert_eq!(cs.nrows(), 2);

        // Verify orthonormality
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += cs[(k, i)] * cs[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(dot, expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_col_space_rank_deficient() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 1.0], &[0.0, 1.0, 1.0], &[1.0, 1.0, 2.0]]);

        let cs = col_space(a.as_ref(), None).unwrap();
        assert_eq!(cs.ncols(), 2); // Rank 2
        assert_eq!(cs.nrows(), 3);
    }

    #[test]
    fn test_row_space() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let rs = row_space(a.as_ref(), None).unwrap();
        assert_eq!(rs.ncols(), 2);
        assert_eq!(rs.nrows(), 2);
    }

    #[test]
    fn test_left_null_space() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 4.0]]);

        let lns = left_null_space(a.as_ref(), None).unwrap();
        assert_eq!(lns.ncols(), 1);
        assert_eq!(lns.nrows(), 2);

        // Verify y^T * A ≈ 0
        // (lns^T * A)[0,0] = lns[0,0]*a[0,0] + lns[1,0]*a[1,0]
        // (lns^T * A)[0,1] = lns[0,0]*a[0,1] + lns[1,0]*a[1,1]
        let prod_0 = lns[(0, 0)] * a[(0, 0)] + lns[(1, 0)] * a[(1, 0)];
        let prod_1 = lns[(0, 0)] * a[(0, 1)] + lns[(1, 0)] * a[(1, 1)];
        assert!(approx_eq(prod_0, 0.0, 1e-10));
        assert!(approx_eq(prod_1, 0.0, 1e-10));
    }

    #[test]
    fn test_four_fundamental_subspaces() {
        // Test the Fundamental Theorem of Linear Algebra
        // For m×n matrix with rank r:
        // - dim(col space) = r
        // - dim(null space) = n - r
        // - dim(row space) = r
        // - dim(left null space) = m - r

        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        // This matrix has rank 2

        let r = rank(a.as_ref(), None).unwrap();
        let m = 3;
        let n = 3;

        let cs = col_space(a.as_ref(), None).unwrap();
        let ns = null_space(a.as_ref(), None).unwrap();
        let rs = row_space(a.as_ref(), None).unwrap();
        let lns = left_null_space(a.as_ref(), None).unwrap();

        assert_eq!(cs.ncols(), r);
        assert_eq!(ns.ncols(), n - r);
        assert_eq!(rs.ncols(), r);
        assert_eq!(lns.ncols(), m - r);
    }

    #[test]
    fn test_rank_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        assert_eq!(rank(a.as_ref(), None).unwrap(), 2);
    }
}
