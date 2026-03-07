//! LAPACK decompositions and operations on ndarray types.
//!
//! This module provides LAPACK decompositions (LU, QR, SVD, EVD, Cholesky)
//! directly on ndarray types, using OxiBLAS-LAPACK as the backend.

use crate::conversions::{array2_to_mat, mat_ref_to_array2, mat_to_array2};
use ndarray::{Array1, Array2};
use oxiblas_core::scalar::Field;
use oxiblas_lapack::{cholesky, evd, lu, qr, solve, svd};
use oxiblas_matrix::Mat;

// Re-export useful types
pub use evd::Eigenvalue;

// =============================================================================
// Error Types
// =============================================================================

/// Error type for LAPACK operations on ndarray.
#[derive(Debug, Clone)]
pub enum LapackError {
    /// Matrix is singular or nearly singular
    Singular(String),
    /// Matrix is not positive definite
    NotPositiveDefinite(String),
    /// Dimension mismatch
    DimensionMismatch(String),
    /// Decomposition did not converge
    NotConverged(String),
    /// Other error
    Other(String),
}

impl std::fmt::Display for LapackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Singular(msg) => write!(f, "Singular matrix: {msg}"),
            Self::NotPositiveDefinite(msg) => write!(f, "Not positive definite: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
            Self::NotConverged(msg) => write!(f, "Did not converge: {msg}"),
            Self::Other(msg) => write!(f, "LAPACK error: {msg}"),
        }
    }
}

impl std::error::Error for LapackError {}

/// Result type for LAPACK operations.
pub type LapackResult<T> = Result<T, LapackError>;

// =============================================================================
// LU Decomposition
// =============================================================================

/// Result of LU decomposition.
#[derive(Debug, Clone)]
pub struct LuResult<T> {
    /// L factor (lower triangular with unit diagonal)
    pub l: Array2<T>,
    /// U factor (upper triangular)
    pub u: Array2<T>,
    /// Permutation vector
    pub perm: Vec<usize>,
}

impl<T: Field + Clone> LuResult<T>
where
    T: bytemuck::Zeroable,
{
    /// Solves Ax = b using the LU decomposition.
    pub fn solve(&self, b: &Array1<T>) -> Array1<T> {
        let n = self.l.dim().0;
        assert_eq!(b.len(), n, "b length must match matrix dimension");

        // Apply permutation to b
        let mut pb: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            pb[i] = b[self.perm[i]];
        }

        // Forward substitution: L * y = pb
        let mut y: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = pb[i];
            for j in 0..i {
                sum -= self.l[[i, j]] * y[j];
            }
            y[i] = sum;
        }

        // Back substitution: U * x = y
        let mut x: Vec<T> = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= self.u[[i, j]] * x[j];
            }
            x[i] = sum / self.u[[i, i]];
        }

        Array1::from_vec(x)
    }

    /// Computes the determinant.
    pub fn det(&self) -> T {
        let n = self.l.dim().0;
        let mut det = T::one();

        // Product of U diagonal elements
        for i in 0..n {
            det *= self.u[[i, i]];
        }

        // Account for permutation sign
        let mut sign_changes = 0;
        let mut visited = vec![false; n];
        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut j = i;
            let mut cycle_len = 0;
            while !visited[j] {
                visited[j] = true;
                j = self.perm[j];
                cycle_len += 1;
            }
            if cycle_len > 1 {
                sign_changes += cycle_len - 1;
            }
        }

        if sign_changes % 2 == 1 {
            det = T::zero() - det;
        }

        det
    }
}

/// Computes the LU decomposition of a matrix.
///
/// A = P * L * U
///
/// # Arguments
/// * `a` - The input matrix (m×n)
///
/// # Returns
/// LU decomposition with L, U, and permutation
pub fn lu_ndarray<T: Field + Clone>(a: &Array2<T>) -> LapackResult<LuResult<T>>
where
    T: bytemuck::Zeroable,
{
    let mat = array2_to_mat(a);

    match lu::Lu::compute(mat.as_ref()) {
        Ok(lu_decomp) => {
            // Extract L and U factors
            let l = mat_to_array2(&lu_decomp.l_factor());
            let u = mat_to_array2(&lu_decomp.u_factor());

            // Get permutation
            let perm = lu_decomp.pivot().to_vec();

            Ok(LuResult { l, u, perm })
        }
        Err(e) => Err(LapackError::Singular(format!("{e:?}"))),
    }
}

// =============================================================================
// QR Decomposition
// =============================================================================

/// Result of QR decomposition.
#[derive(Debug, Clone)]
pub struct QrResult<T> {
    /// Q factor (orthogonal/unitary)
    pub q: Array2<T>,
    /// R factor (upper triangular)
    pub r: Array2<T>,
}

impl<T: Field + Clone> QrResult<T> {
    /// Solves the least squares problem min ||Ax - b||.
    pub fn solve_least_squares(&self, b: &Array1<T>) -> Array1<T> {
        let (m, n) = (self.q.dim().0, self.r.dim().1);
        assert_eq!(b.len(), m, "b length must match matrix rows");

        // Compute Q^T * b (or Q^H for complex)
        let mut qtb: Array1<T> = Array1::from_vec(vec![T::zero(); n]);
        for j in 0..n {
            let mut sum = T::zero();
            for i in 0..m {
                sum += self.q[[i, j]].conj() * b[i];
            }
            qtb[j] = sum;
        }

        // Back substitution: R * x = Q^T * b
        let mut x: Array1<T> = Array1::from_vec(vec![T::zero(); n]);
        for i in (0..n).rev() {
            let mut sum = qtb[i];
            for j in (i + 1)..n {
                sum -= self.r[[i, j]] * x[j];
            }
            x[i] = sum / self.r[[i, i]];
        }

        x
    }
}

/// Computes the QR decomposition of a matrix.
///
/// A = Q * R
///
/// # Arguments
/// * `a` - The input matrix (m×n)
///
/// # Returns
/// QR decomposition with Q and R
pub fn qr_ndarray<T: Field + Clone>(a: &Array2<T>) -> LapackResult<QrResult<T>>
where
    T: bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    match qr::Qr::compute(mat.as_ref()) {
        Ok(qr_decomp) => {
            let q = mat_to_array2(&qr_decomp.q());
            let r = mat_to_array2(&qr_decomp.r());

            Ok(QrResult { q, r })
        }
        Err(e) => Err(LapackError::Other(format!("{e:?}"))),
    }
}

// =============================================================================
// Singular Value Decomposition
// =============================================================================

/// Result of SVD decomposition.
#[derive(Debug, Clone)]
pub struct SvdResult<T> {
    /// Left singular vectors U (m×k where k = min(m,n))
    pub u: Array2<T>,
    /// Singular values σ (sorted in descending order)
    pub s: Array1<T>,
    /// Right singular vectors V^T (k×n)
    pub vt: Array2<T>,
}

impl<T: Field + Clone> SvdResult<T> {
    /// Returns the rank based on a tolerance.
    pub fn rank(&self, tol: T) -> usize {
        self.s.iter().filter(|&s| s.abs() > tol.abs()).count()
    }
}

/// Computes the SVD of a matrix.
///
/// A = U * Σ * V^T
///
/// # Arguments
/// * `a` - The input matrix (m×n)
///
/// # Returns
/// SVD with U, S (singular values), and V^T
pub fn svd_ndarray<T>(a: &Array2<T>) -> LapackResult<SvdResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    match svd::Svd::compute(mat.as_ref()) {
        Ok(svd_decomp) => {
            let u = mat_ref_to_array2(svd_decomp.u());
            let s = Array1::from_vec(svd_decomp.singular_values().to_vec());
            let vt = mat_ref_to_array2(svd_decomp.vt());

            Ok(SvdResult { u, s, vt })
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

/// Computes the truncated SVD of a matrix.
///
/// Returns only the top k singular values and vectors.
pub fn svd_truncated<T>(a: &Array2<T>, k: usize) -> LapackResult<SvdResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let svd_result = svd_ndarray(a)?;

    let actual_k = k.min(svd_result.s.len());

    // Truncate to k components
    let u = svd_result.u.slice(ndarray::s![.., ..actual_k]).to_owned();
    let s = svd_result.s.slice(ndarray::s![..actual_k]).to_owned();
    let vt = svd_result.vt.slice(ndarray::s![..actual_k, ..]).to_owned();

    Ok(SvdResult { u, s, vt })
}

// =============================================================================
// Eigenvalue Decomposition (Symmetric)
// =============================================================================

/// Result of symmetric eigenvalue decomposition.
#[derive(Debug, Clone)]
pub struct SymEvdResult<T> {
    /// Eigenvalues (sorted in ascending order)
    pub eigenvalues: Array1<T>,
    /// Eigenvectors (columns are eigenvectors)
    pub eigenvectors: Array2<T>,
}

/// Computes the eigenvalue decomposition of a symmetric matrix.
///
/// A * V = V * Λ where Λ = diag(eigenvalues)
///
/// # Arguments
/// * `a` - The input symmetric matrix (n×n)
///
/// # Returns
/// Eigenvalues and eigenvectors
pub fn eig_symmetric<T>(a: &Array2<T>) -> LapackResult<SymEvdResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match evd::SymmetricEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = Array1::from_vec(evd_result.eigenvalues().to_vec());
            // Convert MatRef to Array2
            let evec_ref = evd_result.eigenvectors();
            let (rows, cols) = (evec_ref.nrows(), evec_ref.ncols());
            let eigenvectors = Array2::from_shape_fn((rows, cols), |(i, j)| evec_ref[(i, j)]);

            Ok(SymEvdResult {
                eigenvalues,
                eigenvectors,
            })
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

/// Computes only the eigenvalues of a symmetric matrix.
pub fn eigvals_symmetric<T>(a: &Array2<T>) -> LapackResult<Array1<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    eig_symmetric(a).map(|result| result.eigenvalues)
}

// =============================================================================
// Complex-Specific Functions (for Complex<f64>, Complex<f32>)
// =============================================================================

/// Result of complex SVD decomposition.
#[derive(Debug, Clone)]
pub struct ComplexSvdResult<T>
where
    T: oxiblas_core::Scalar,
{
    /// Left singular vectors U (m×k, complex unitary)
    pub u: Array2<T>,
    /// Singular values σ (real, sorted in descending order)
    pub s: Array1<T::Real>,
    /// Right singular vectors V^H (k×n, complex unitary)
    pub vt: Array2<T>,
}

/// Computes the SVD of a complex matrix using ComplexSvd algorithm.
///
/// For complex matrices, this function uses the one-sided Jacobi algorithm
/// specifically designed for complex numbers.
///
/// # Arguments
/// * `a` - The input complex matrix (m×n)
///
/// # Returns
/// U, singular values (real), and V^H
pub fn svd_complex_ndarray<T>(a: &Array2<T>) -> LapackResult<ComplexSvdResult<T>>
where
    T: Field + oxiblas_core::scalar::ComplexScalar + Clone + bytemuck::Zeroable,
    T::Real: oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    match svd::ComplexSvd::compute(mat.as_ref()) {
        Ok(svd_decomp) => {
            let u = mat_ref_to_array2(svd_decomp.u().as_ref());
            let s = Array1::from_vec(svd_decomp.singular_values().to_vec());
            let vh = mat_ref_to_array2(svd_decomp.vh().as_ref());

            Ok(ComplexSvdResult { u, s, vt: vh })
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

/// Computes the QR decomposition of a complex matrix.
///
/// # Arguments
/// * `a` - The input complex matrix (m×n)
///
/// # Returns
/// Q (unitary) and R (upper triangular)
pub fn qr_complex_ndarray<T>(a: &Array2<T>) -> LapackResult<QrResult<T>>
where
    T: Field + oxiblas_core::scalar::ComplexScalar + Clone + bytemuck::Zeroable,
    T::Real: oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    match qr::UnitaryQr::compute(mat.as_ref()) {
        Ok(qr_decomp) => {
            let q_mat = qr_decomp.q();
            let r_mat = qr_decomp.r();
            let q = mat_to_array2(&q_mat);
            let r = mat_to_array2(&r_mat);
            Ok(QrResult { q, r })
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

/// Computes the Cholesky decomposition of a Hermitian positive definite matrix.
///
/// # Arguments
/// * `a` - The input Hermitian positive definite matrix (n×n)
///
/// # Returns
/// Lower triangular factor L such that A = LL^H
pub fn cholesky_hermitian_ndarray<T>(a: &Array2<T>) -> LapackResult<CholeskyResult<T>>
where
    T: Field + oxiblas_core::scalar::ComplexScalar + Clone + bytemuck::Zeroable,
    T::Real: oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match cholesky::HermitianCholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            let l_mat = chol.l_factor();
            let l = mat_to_array2(&l_mat);
            Ok(CholeskyResult { l })
        }
        Err(e) => Err(LapackError::NotPositiveDefinite(format!("{e:?}"))),
    }
}

/// Result of Hermitian eigenvalue decomposition for complex matrices.
#[derive(Debug, Clone)]
pub struct HermitianEvdResult<T>
where
    T: oxiblas_core::Scalar,
{
    /// Eigenvalues (real, sorted in ascending order)
    pub eigenvalues: Array1<T>,
    /// Eigenvectors (complex columns)
    pub eigenvectors: Array2<T>,
}

/// Computes the eigenvalue decomposition of a Hermitian matrix.
///
/// For Hermitian matrices (A = A^H), all eigenvalues are real but eigenvectors are complex.
///
/// # Arguments
/// * `a` - The input Hermitian matrix (n×n, only upper triangle is used)
///
/// # Returns
/// Eigenvalues (real, sorted in ascending order) and eigenvectors (complex columns)
pub fn eig_hermitian_ndarray<T>(a: &Array2<T>) -> LapackResult<(Array1<T::Real>, Array2<T>)>
where
    T: Field + oxiblas_core::scalar::ComplexScalar + Clone + bytemuck::Zeroable,
    T::Real: oxiblas_core::scalar::Real + Clone + bytemuck::Zeroable,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match evd::HermitianEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = Array1::from_vec(evd_result.eigenvalues().to_vec());

            // Convert eigenvectors MatRef<T> to Array2<T>
            let evec_ref = evd_result.eigenvectors();
            let eigenvectors = mat_ref_to_array2(evec_ref);

            Ok((eigenvalues, eigenvectors))
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

// =============================================================================
// Cholesky Decomposition
// =============================================================================

/// Result of Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct CholeskyResult<T> {
    /// Lower triangular factor L such that A = L * L^T
    pub l: Array2<T>,
}

impl<T: Field + Clone> CholeskyResult<T> {
    /// Solves Ax = b using the Cholesky decomposition.
    pub fn solve(&self, b: &Array1<T>) -> Array1<T> {
        let n = self.l.dim().0;
        assert_eq!(b.len(), n, "b length must match matrix dimension");

        // Forward substitution: L * y = b
        let mut y: Array1<T> = Array1::from_vec(vec![T::zero(); n]);
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.l[[i, j]] * y[j];
            }
            y[i] = sum / self.l[[i, i]];
        }

        // Back substitution: L^T * x = y
        let mut x: Array1<T> = Array1::from_vec(vec![T::zero(); n]);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= self.l[[j, i]].conj() * x[j];
            }
            x[i] = sum / self.l[[i, i]].conj();
        }

        x
    }

    /// Computes the determinant.
    pub fn det(&self) -> T {
        let n = self.l.dim().0;
        let mut det = T::one();
        for i in 0..n {
            let diag = self.l[[i, i]];
            det = det * diag * diag;
        }
        det
    }
}

/// Computes the Cholesky decomposition of a positive definite matrix.
///
/// A = L * L^T
///
/// # Arguments
/// * `a` - The input symmetric positive definite matrix (n×n)
///
/// # Returns
/// Cholesky decomposition with lower triangular factor L
pub fn cholesky_ndarray<T>(a: &Array2<T>) -> LapackResult<CholeskyResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match cholesky::Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            let l = mat_to_array2(&chol.l_factor());
            Ok(CholeskyResult { l })
        }
        Err(e) => Err(LapackError::NotPositiveDefinite(format!("{e:?}"))),
    }
}

// =============================================================================
// Linear Solve
// =============================================================================

/// Solves the linear system Ax = b.
///
/// # Arguments
/// * `a` - The coefficient matrix (n×n)
/// * `b` - The right-hand side vector (n)
///
/// # Returns
/// The solution vector x
pub fn solve_ndarray<T>(a: &Array2<T>, b: &Array1<T>) -> LapackResult<Array1<T>>
where
    T: Field + Clone + bytemuck::Zeroable,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LapackError::DimensionMismatch(
            "b length must match matrix dimension".to_string(),
        ));
    }

    let a_mat = array2_to_mat(a);
    // Convert b to a column vector matrix
    let mut b_mat: Mat<T> = Mat::zeros(n, 1);
    for i in 0..n {
        b_mat[(i, 0)] = b[i];
    }

    match solve::solve(a_mat.as_ref(), b_mat.as_ref()) {
        Ok(x_mat) => {
            // Extract column vector from result matrix
            let x: Vec<T> = (0..n).map(|i| x_mat[(i, 0)]).collect();
            Ok(Array1::from_vec(x))
        }
        Err(e) => Err(LapackError::Singular(format!("{e:?}"))),
    }
}

/// Solves multiple linear systems AX = B.
///
/// # Arguments
/// * `a` - The coefficient matrix (n×n)
/// * `b` - The right-hand side matrix (n×k)
///
/// # Returns
/// The solution matrix X (n×k)
pub fn solve_multiple_ndarray<T>(a: &Array2<T>, b: &Array2<T>) -> LapackResult<Array2<T>>
where
    T: Field + Clone + bytemuck::Zeroable,
{
    let (m, n) = a.dim();
    let (b_rows, _b_cols) = b.dim();

    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }
    if b_rows != n {
        return Err(LapackError::DimensionMismatch(
            "b rows must match matrix dimension".to_string(),
        ));
    }

    let a_mat = array2_to_mat(a);
    let b_mat = array2_to_mat(b);

    match solve::solve_multiple(a_mat.as_ref(), b_mat.as_ref()) {
        Ok(x_mat) => Ok(mat_to_array2(&x_mat)),
        Err(e) => Err(LapackError::Singular(format!("{e:?}"))),
    }
}

/// Solves the least squares problem min ||Ax - b||.
pub fn lstsq_ndarray<T>(a: &Array2<T>, b: &Array1<T>) -> LapackResult<Array1<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let m = a.dim().0;
    let a_mat = array2_to_mat(a);
    // Convert b to a column vector matrix
    let mut b_mat: Mat<T> = Mat::zeros(m, 1);
    for i in 0..m {
        b_mat[(i, 0)] = b[i];
    }

    match solve::lstsq(a_mat.as_ref(), b_mat.as_ref()) {
        Ok(result) => {
            // Extract solution column vector
            let n = result.solution.nrows();
            let x: Vec<T> = (0..n).map(|i| result.solution[(i, 0)]).collect();
            Ok(Array1::from_vec(x))
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

// =============================================================================
// Matrix Inverse
// =============================================================================

/// Computes the inverse of a matrix.
pub fn inv_ndarray<T>(a: &Array2<T>) -> LapackResult<Array2<T>>
where
    T: Field + Clone + bytemuck::Zeroable,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let a_mat = array2_to_mat(a);

    match oxiblas_lapack::utils::inv(a_mat.as_ref()) {
        Ok(inv_mat) => Ok(mat_to_array2(&inv_mat)),
        Err(e) => Err(LapackError::Singular(format!("{e:?}"))),
    }
}

/// Computes the Moore-Penrose pseudo-inverse.
pub fn pinv_ndarray<T>(a: &Array2<T>) -> LapackResult<Array2<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let a_mat = array2_to_mat(a);

    match oxiblas_lapack::utils::pinv_default(a_mat.as_ref()) {
        Ok(result) => Ok(mat_to_array2(&result.pinv)),
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

// =============================================================================
// Determinant
// =============================================================================

/// Computes the determinant of a matrix.
pub fn det_ndarray<T>(a: &Array2<T>) -> LapackResult<T>
where
    T: Field + Clone + bytemuck::Zeroable,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let a_mat = array2_to_mat(a);

    match oxiblas_lapack::utils::det(a_mat.as_ref()) {
        Ok(d) => Ok(d),
        Err(e) => Err(LapackError::Other(format!("{e:?}"))),
    }
}

// =============================================================================
// Condition Number
// =============================================================================

/// Computes the condition number of a matrix (using 2-norm).
pub fn cond_ndarray<T>(a: &Array2<T>) -> LapackResult<T>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let svd_result = svd_ndarray(a)?;
    let n = svd_result.s.len();

    if n == 0 {
        return Ok(T::one());
    }

    let sigma_max = svd_result.s[0];
    let sigma_min = svd_result.s[n - 1];

    // Check if sigma_min is very small
    if sigma_min == T::zero() {
        // Return a large number to indicate ill-conditioning
        Ok(T::from_f64(1e15).unwrap_or(T::one()))
    } else {
        Ok(sigma_max / sigma_min)
    }
}

// =============================================================================
// Rank
// =============================================================================

/// Computes the numerical rank of a matrix.
pub fn rank_ndarray<T>(a: &Array2<T>) -> LapackResult<usize>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    let svd_result = svd_ndarray(a)?;

    if svd_result.s.is_empty() {
        return Ok(0);
    }

    // Default tolerance: max(m,n) * eps * sigma_max
    let sigma_max = svd_result.s[0];
    let eps = T::from_f64(1e-14).unwrap_or(T::zero());
    let dim_scale = T::from_f64(m.max(n) as f64).unwrap_or(T::one());
    let tol = dim_scale * eps * sigma_max;

    Ok(svd_result.rank(tol))
}

// =============================================================================
// Randomized SVD
// =============================================================================

/// Result of randomized SVD.
#[derive(Debug, Clone)]
pub struct RandomizedSvdResult<T> {
    /// Left singular vectors U (m × k)
    pub u: Array2<T>,
    /// Singular values σ (k elements, sorted descending)
    pub s: Array1<T>,
    /// Right singular vectors V (n × k), NOT V^T
    pub v: Array2<T>,
}

/// Computes a randomized SVD approximation of a matrix.
///
/// Uses randomized projections to compute a rank-k approximation efficiently,
/// particularly useful for large matrices where only the top singular values
/// are needed.
///
/// # Arguments
/// * `a` - The input matrix (m×n)
/// * `k` - Target rank (number of singular values to compute)
///
/// # Returns
/// Truncated SVD with k singular values and vectors
///
/// # Algorithm
/// Uses the Halko-Martinsson-Tropp randomized algorithm:
/// 1. Generate random test matrix Ω
/// 2. Compute Y = A × Ω to sample column space
/// 3. QR factorize Y to get orthonormal basis Q
/// 4. Project B = Q^T × A
/// 5. Compute full SVD of B
/// 6. Recover U = Q × Ũ
pub fn rsvd_ndarray<T>(a: &Array2<T>, k: usize) -> LapackResult<RandomizedSvdResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    match svd::RandomizedSvd::compute(mat.as_ref(), k) {
        Ok(rsvd) => {
            let u = mat_ref_to_array2(rsvd.u());
            let s = Array1::from_vec(rsvd.singular_values().to_vec());
            let v = mat_ref_to_array2(rsvd.v());

            Ok(RandomizedSvdResult { u, s, v })
        }
        Err(e) => Err(LapackError::Other(format!("{e:?}"))),
    }
}

/// Computes randomized SVD with power iteration for improved accuracy.
///
/// Power iteration emphasizes dominant singular values and improves accuracy
/// for matrices with slowly decaying singular values.
///
/// # Arguments
/// * `a` - The input matrix (m×n)
/// * `k` - Target rank
/// * `power_iterations` - Number of power iterations (typically 1-3)
pub fn rsvd_power_ndarray<T>(
    a: &Array2<T>,
    k: usize,
    power_iterations: usize,
) -> LapackResult<RandomizedSvdResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    let config = svd::RandomizedSvdConfig::new(k).with_power_iterations(power_iterations);

    match svd::RandomizedSvd::compute_with_config(mat.as_ref(), config) {
        Ok(rsvd) => {
            let u = mat_ref_to_array2(rsvd.u());
            let s = Array1::from_vec(rsvd.singular_values().to_vec());
            let v = mat_ref_to_array2(rsvd.v());

            Ok(RandomizedSvdResult { u, s, v })
        }
        Err(e) => Err(LapackError::Other(format!("{e:?}"))),
    }
}

// =============================================================================
// Schur Decomposition
// =============================================================================

/// Result of Schur decomposition.
#[derive(Debug, Clone)]
pub struct SchurResult<T> {
    /// Orthogonal matrix Q (Schur vectors)
    pub q: Array2<T>,
    /// Quasi-upper triangular matrix T (Schur form)
    pub t: Array2<T>,
    /// Eigenvalues (real and complex pairs)
    pub eigenvalues: Vec<Eigenvalue<T>>,
}

/// Computes the real Schur decomposition of a square matrix.
///
/// A = Q T Q^T where:
/// - Q is orthogonal (Q^T Q = I)
/// - T is quasi-upper triangular (upper triangular with possible 2×2 blocks
///   on the diagonal for complex eigenvalue pairs)
///
/// # Arguments
/// * `a` - The input square matrix (n×n)
///
/// # Returns
/// Schur decomposition with Q, T, and eigenvalues
pub fn schur_ndarray<T>(a: &Array2<T>) -> LapackResult<SchurResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match evd::Schur::compute(mat.as_ref()) {
        Ok(schur) => {
            let q = mat_ref_to_array2(schur.q());
            let t = mat_ref_to_array2(schur.t());
            let eigenvalues = schur.eigenvalues().to_vec();

            Ok(SchurResult { q, t, eigenvalues })
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

// =============================================================================
// General Eigenvalue Decomposition
// =============================================================================

/// Result of general eigenvalue decomposition.
#[derive(Debug, Clone)]
pub struct GeneralEvdResult<T> {
    /// Eigenvalues (real and imaginary parts)
    pub eigenvalues: Vec<Eigenvalue<T>>,
    /// Right eigenvectors (real parts), if computed
    pub eigenvectors_real: Option<Array2<T>>,
    /// Right eigenvectors (imaginary parts), if computed
    pub eigenvectors_imag: Option<Array2<T>>,
    /// Left eigenvectors (real parts), if computed
    pub left_eigenvectors_real: Option<Array2<T>>,
    /// Left eigenvectors (imaginary parts), if computed
    pub left_eigenvectors_imag: Option<Array2<T>>,
}

/// Computes eigenvalues of a general (non-symmetric) matrix.
///
/// For a real matrix, eigenvalues may be complex. They are returned as
/// real/imaginary pairs. Eigenvectors are also split into real and
/// imaginary parts.
///
/// # Arguments
/// * `a` - The input square matrix (n×n)
///
/// # Returns
/// Eigenvalues and eigenvectors (split into real/imaginary parts)
pub fn eig_ndarray<T>(a: &Array2<T>) -> LapackResult<GeneralEvdResult<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match evd::GeneralEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues().to_vec();

            // Get right eigenvectors (real and imaginary parts)
            let eigenvectors_real = evd_result
                .eigenvectors_real()
                .map(|vr| mat_ref_to_array2(vr));

            let eigenvectors_imag = evd_result
                .eigenvectors_imag()
                .map(|vi| mat_ref_to_array2(vi));

            // Get left eigenvectors (real and imaginary parts)
            let left_eigenvectors_real = evd_result
                .left_eigenvectors_real()
                .map(|vl| mat_ref_to_array2(vl));

            let left_eigenvectors_imag = evd_result
                .left_eigenvectors_imag()
                .map(|vl| mat_ref_to_array2(vl));

            Ok(GeneralEvdResult {
                eigenvalues,
                eigenvectors_real,
                eigenvectors_imag,
                left_eigenvectors_real,
                left_eigenvectors_imag,
            })
        }
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

/// Computes only the eigenvalues of a general matrix.
pub fn eigvals_ndarray<T>(a: &Array2<T>) -> LapackResult<Vec<Eigenvalue<T>>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LapackError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mat = array2_to_mat(a);

    match evd::GeneralEvd::eigenvalues_only(mat.as_ref()) {
        Ok(evd_result) => Ok(evd_result.eigenvalues().to_vec()),
        Err(e) => Err(LapackError::NotConverged(format!("{e:?}"))),
    }
}

// =============================================================================
// Tridiagonal Solvers
// =============================================================================

/// Solves a tridiagonal system of equations.
///
/// Solves T x = b where T is a tridiagonal matrix.
///
/// # Arguments
/// * `dl` - Lower diagonal (n-1 elements)
/// * `d` - Main diagonal (n elements)
/// * `du` - Upper diagonal (n-1 elements)
/// * `b` - Right-hand side vector (n elements)
///
/// # Returns
/// The solution vector x
pub fn tridiag_solve_ndarray<T>(
    dl: &Array1<T>,
    d: &Array1<T>,
    du: &Array1<T>,
    b: &Array1<T>,
) -> LapackResult<Array1<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let n = d.len();

    if dl.len() != n - 1 || du.len() != n - 1 || b.len() != n {
        return Err(LapackError::DimensionMismatch(
            "Tridiagonal dimensions must be consistent".to_string(),
        ));
    }

    let dl_vec: Vec<T> = dl.iter().cloned().collect();
    let d_vec: Vec<T> = d.iter().cloned().collect();
    let du_vec: Vec<T> = du.iter().cloned().collect();
    let b_vec: Vec<T> = b.iter().cloned().collect();

    match solve::tridiag_solve(&dl_vec, &d_vec, &du_vec, &b_vec) {
        Ok(x) => Ok(Array1::from_vec(x)),
        Err(e) => Err(LapackError::Singular(format!("{e:?}"))),
    }
}

/// Solves a symmetric positive definite tridiagonal system.
///
/// Solves T x = b where T is symmetric positive definite and tridiagonal.
/// Uses specialized algorithm that's more efficient for SPD matrices.
///
/// # Arguments
/// * `d` - Main diagonal (n elements, positive)
/// * `e` - Off-diagonal (n-1 elements)
/// * `b` - Right-hand side vector (n elements)
///
/// # Returns
/// The solution vector x
pub fn tridiag_solve_spd_ndarray<T>(
    d: &Array1<T>,
    e: &Array1<T>,
    b: &Array1<T>,
) -> LapackResult<Array1<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let n = d.len();

    if e.len() != n - 1 || b.len() != n {
        return Err(LapackError::DimensionMismatch(
            "Tridiagonal dimensions must be consistent".to_string(),
        ));
    }

    let d_vec: Vec<T> = d.iter().cloned().collect();
    let e_vec: Vec<T> = e.iter().cloned().collect();
    let b_vec: Vec<T> = b.iter().cloned().collect();

    match solve::tridiag_solve_spd(&d_vec, &e_vec, &b_vec) {
        Ok(x) => Ok(Array1::from_vec(x)),
        Err(e) => Err(LapackError::NotPositiveDefinite(format!("{e:?}"))),
    }
}

/// Solves multiple tridiagonal systems with the same matrix.
///
/// # Arguments
/// * `dl` - Lower diagonal (n-1 elements)
/// * `d` - Main diagonal (n elements)
/// * `du` - Upper diagonal (n-1 elements)
/// * `b` - Right-hand side matrix (n × nrhs)
///
/// # Returns
/// The solution matrix X (n × nrhs)
pub fn tridiag_solve_multiple_ndarray<T>(
    dl: &Array1<T>,
    d: &Array1<T>,
    du: &Array1<T>,
    b: &Array2<T>,
) -> LapackResult<Array2<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let n = d.len();
    let (b_rows, _b_cols) = b.dim();

    if dl.len() != n - 1 || du.len() != n - 1 || b_rows != n {
        return Err(LapackError::DimensionMismatch(
            "Tridiagonal dimensions must be consistent".to_string(),
        ));
    }

    let dl_vec: Vec<T> = dl.iter().cloned().collect();
    let d_vec: Vec<T> = d.iter().cloned().collect();
    let du_vec: Vec<T> = du.iter().cloned().collect();
    let b_mat = array2_to_mat(b);

    match solve::tridiag_solve_multiple(&dl_vec, &d_vec, &du_vec, b_mat.as_ref()) {
        Ok(x_mat) => Ok(mat_to_array2(&x_mat)),
        Err(e) => Err(LapackError::Singular(format!("{e:?}"))),
    }
}

// =============================================================================
// Low-Rank Approximation
// =============================================================================

/// Computes a low-rank approximation of a matrix.
///
/// Returns A_k = U_k Σ_k V_k^T, the best rank-k approximation in Frobenius norm.
///
/// # Arguments
/// * `a` - The input matrix (m×n)
/// * `k` - Target rank
///
/// # Returns
/// The rank-k approximation as a matrix
pub fn low_rank_approx_ndarray<T>(a: &Array2<T>, k: usize) -> LapackResult<Array2<T>>
where
    T: Field + Clone + bytemuck::Zeroable + oxiblas_core::scalar::Real,
{
    let mat = array2_to_mat(a);

    match svd::low_rank_approximation(mat.as_ref(), k) {
        Ok(approx) => Ok(mat_to_array2(&approx)),
        Err(e) => Err(LapackError::Other(format!("{e:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lu_decomposition() {
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let lu = lu_ndarray(&a).unwrap();

        // Verify L * U ≈ P * A
        let n = a.dim().0;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += lu.l[[i, k]] * lu.u[[k, j]];
                }
                let perm_i = lu.perm.iter().position(|&p| p == i).unwrap();
                assert!((sum - a[[perm_i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lu_determinant() {
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let lu = lu_ndarray(&a).unwrap();
        let det = lu.det();
        // det = 2*3 - 1*1 = 5
        assert!((det - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_solve() {
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let b = array![5.0f64, 7.0];
        let lu = lu_ndarray(&a).unwrap();
        let x = lu.solve(&b);

        // Verify A * x ≈ b
        let ax0 = a[[0, 0]] * x[0] + a[[0, 1]] * x[1];
        let ax1 = a[[1, 0]] * x[0] + a[[1, 1]] * x[1];
        assert!((ax0 - b[0]).abs() < 1e-10);
        assert!((ax1 - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let qr = qr_ndarray(&a).unwrap();

        // Q should be orthogonal: Q^T * Q = I
        let qt = qr.q.t();
        let qtq = crate::blas::matmul(&qt.to_owned(), &qr.q);
        for i in 0..qtq.dim().0 {
            for j in 0..qtq.dim().1 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-10,
                    "Q^T Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq[[i, j]],
                    expected
                );
            }
        }

        // Q * R should equal A
        let qr_product = crate::blas::matmul(&qr.q, &qr.r);
        for i in 0..a.dim().0 {
            for j in 0..a.dim().1 {
                assert!(
                    (qr_product[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "QR[{},{}] = {}, A = {}",
                    i,
                    j,
                    qr_product[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_svd() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let svd = svd_ndarray(&a).unwrap();

        // Reconstruct A from SVD: U * S * V^T
        let (m, n) = a.dim();
        let k = svd.s.len();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for l in 0..k {
                    sum += svd.u[[i, l]] * svd.s[l] * svd.vt[[l, j]];
                }
                assert!(
                    (sum - a[[i, j]]).abs() < 1e-10,
                    "Reconstructed[{},{}] = {}, A = {}",
                    i,
                    j,
                    sum,
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_symmetric_evd() {
        // Symmetric matrix
        let a = array![[4.0f64, 1.0], [1.0, 3.0]];
        let evd = eig_symmetric(&a).unwrap();

        // Eigenvalues should be real and positive for this matrix
        assert!(evd.eigenvalues.len() == 2);

        // Verify A * V = V * Λ for each eigenvalue/eigenvector pair
        for (idx, &lambda) in evd.eigenvalues.iter().enumerate() {
            let v = evd.eigenvectors.column(idx);
            let av = crate::blas::matvec(&a, &v.to_owned());
            let lambda_v: Array1<f64> = v.iter().map(|&x| lambda * x).collect();

            for i in 0..2 {
                assert!(
                    (av[i] - lambda_v[i]).abs() < 1e-10,
                    "Av[{}] = {}, λv[{}] = {}",
                    i,
                    av[i],
                    i,
                    lambda_v[i]
                );
            }
        }
    }

    #[test]
    fn test_cholesky() {
        // Positive definite matrix
        let a = array![[4.0f64, 2.0], [2.0, 5.0]];
        let chol = cholesky_ndarray(&a).unwrap();

        // Verify L * L^T = A
        let lt = chol.l.t();
        let llt = crate::blas::matmul(&chol.l, &lt.to_owned());

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (llt[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "LLT[{},{}] = {}, A = {}",
                    i,
                    j,
                    llt[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_solve() {
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let b = array![5.0f64, 7.0];
        let x = solve_ndarray(&a, &b).unwrap();

        // Verify A * x = b
        let ax = crate::blas::matvec(&a, &x);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let a = array![[4.0f64, 7.0], [2.0, 6.0]];
        let a_inv = inv_ndarray(&a).unwrap();

        // A * A^-1 = I
        let product = crate::blas::matmul(&a, &a_inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < 1e-10,
                    "A*A^-1[{},{}] = {}, expected {}",
                    i,
                    j,
                    product[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_determinant() {
        let a = array![[2.0f64, 1.0], [1.0, 3.0]];
        let det = det_ndarray(&a).unwrap();
        // det = 2*3 - 1*1 = 5
        assert!((det - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_condition_number() {
        let a = array![[1.0f64, 0.0], [0.0, 1.0]];
        let cond = cond_ndarray(&a).unwrap();
        // Identity matrix has condition number 1
        assert!((cond - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank() {
        // Full rank matrix
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let r = rank_ndarray(&a).unwrap();
        assert_eq!(r, 2);

        // Rank deficient matrix
        let b = array![[1.0f64, 2.0], [2.0, 4.0]];
        let r2 = rank_ndarray(&b).unwrap();
        assert_eq!(r2, 1);
    }

    // =========================================================================
    // Randomized SVD Tests
    // =========================================================================

    #[test]
    fn test_rsvd_basic() {
        // Create a low-rank matrix
        let a = array![
            [1.0f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];

        let rsvd = rsvd_ndarray(&a, 2).unwrap();

        // Should have 2 singular values
        assert_eq!(rsvd.s.len(), 2);

        // Singular values should be positive and in descending order
        assert!(rsvd.s[0] > rsvd.s[1]);
        assert!(rsvd.s[1] >= 0.0);

        // U should be m×k
        assert_eq!(rsvd.u.dim(), (3, 2));

        // V should be n×k
        assert_eq!(rsvd.v.dim(), (4, 2));
    }

    #[test]
    fn test_rsvd_approximation_quality() {
        // Create a matrix with clear rank structure
        let a = Array2::from_shape_fn((10, 8), |(i, j)| (i as f64) * 0.1 + (j as f64) * 0.2);

        let rsvd = rsvd_ndarray(&a, 2).unwrap();

        // Reconstruct: A ≈ U * S * V^T
        let (m, n) = a.dim();
        let k = rsvd.s.len();

        let mut approx: Array2<f64> = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    approx[[i, j]] += rsvd.u[[i, l]] * rsvd.s[l] * rsvd.v[[j, l]];
                }
            }
        }

        // The approximation should capture most of the matrix (rank-1 for this matrix)
        let mut diff_norm = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let diff = a[[i, j]] - approx[[i, j]];
                diff_norm += diff.powi(2);
            }
        }
        diff_norm = diff_norm.sqrt();

        // Should be reasonably small
        assert!(diff_norm < 1e-10, "Reconstruction error = {}", diff_norm);
    }

    #[test]
    fn test_rsvd_power_iteration() {
        let a = Array2::from_shape_fn((20, 15), |(i, j)| ((i * j) as f64).sin() + 0.1 * (i as f64));

        let rsvd = rsvd_power_ndarray(&a, 3, 2).unwrap();

        assert_eq!(rsvd.s.len(), 3);
        assert!(rsvd.s[0] >= rsvd.s[1]);
        assert!(rsvd.s[1] >= rsvd.s[2]);
    }

    // =========================================================================
    // Schur Decomposition Tests
    // =========================================================================

    #[test]
    fn test_schur_triangular() {
        // Already upper triangular matrix
        let a = array![[1.0f64, 2.0], [0.0, 3.0]];

        let schur = schur_ndarray(&a).unwrap();

        // Eigenvalues should be 1 and 3
        assert_eq!(schur.eigenvalues.len(), 2);

        let evs: Vec<f64> = schur.eigenvalues.iter().map(|e| e.real).collect();
        assert!(evs.contains(&1.0) || evs.iter().any(|&x| (x - 1.0).abs() < 1e-10));
        assert!(evs.contains(&3.0) || evs.iter().any(|&x| (x - 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_schur_reconstruction() {
        let a = array![[4.0f64, 1.0], [2.0, 3.0]];

        let schur = schur_ndarray(&a).unwrap();

        // Verify A = Q * T * Q^T
        let qt = schur.q.t();
        let qt_owned = qt.to_owned();
        let qr_temp = crate::blas::matmul(&schur.q, &schur.t);
        let reconstructed = crate::blas::matmul(&qr_temp, &qt_owned);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Reconstruction failed at [{},{}]: {} vs {}",
                    i,
                    j,
                    reconstructed[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_schur_orthogonality() {
        let a = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];

        let schur = schur_ndarray(&a).unwrap();

        // Q should be orthogonal: Q^T * Q = I
        let qt = schur.q.t();
        let qtq = crate::blas::matmul(&qt.to_owned(), &schur.q);

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-10,
                    "Q^T Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    qtq[[i, j]],
                    expected
                );
            }
        }
    }

    // =========================================================================
    // General Eigenvalue Decomposition Tests
    // =========================================================================

    #[test]
    fn test_eig_real_eigenvalues() {
        // Symmetric matrix has real eigenvalues
        let a = array![[4.0f64, 1.0], [1.0, 3.0]];

        let evd = eig_ndarray(&a).unwrap();

        assert_eq!(evd.eigenvalues.len(), 2);

        // All eigenvalues should be real (imaginary part ≈ 0)
        for ev in &evd.eigenvalues {
            assert!(
                ev.imag.abs() < 1e-10,
                "Expected real eigenvalue, got imag = {}",
                ev.imag
            );
        }
    }

    #[test]
    fn test_eig_complex_eigenvalues() {
        // Rotation matrix has complex eigenvalues (±i)
        let a = array![[0.0f64, -1.0], [1.0, 0.0]];

        let evd = eig_ndarray(&a).unwrap();

        assert_eq!(evd.eigenvalues.len(), 2);

        // Should have eigenvalues with nonzero imaginary parts
        let has_complex = evd.eigenvalues.iter().any(|e| e.imag.abs() > 0.5);
        assert!(has_complex, "Expected complex eigenvalues");

        // Real parts should be close to 0
        for ev in &evd.eigenvalues {
            assert!(ev.real.abs() < 1e-10, "Expected real part ≈ 0");
        }
    }

    #[test]
    fn test_eigvals_only() {
        let a = array![[1.0f64, 2.0], [0.0, 3.0]];

        let evs = eigvals_ndarray(&a).unwrap();

        assert_eq!(evs.len(), 2);

        // Eigenvalues of upper triangular matrix are diagonal elements
        let reals: Vec<f64> = evs.iter().map(|e| e.real).collect();
        assert!(reals.iter().any(|&x| (x - 1.0).abs() < 1e-10));
        assert!(reals.iter().any(|&x| (x - 3.0).abs() < 1e-10));
    }

    // =========================================================================
    // Tridiagonal Solver Tests
    // =========================================================================

    #[test]
    fn test_tridiag_solve() {
        // Tridiagonal matrix:
        // [2  -1  0 ]   [x0]   [1]
        // [-1  2 -1 ] * [x1] = [0]
        // [0  -1  2 ]   [x2]   [1]
        let dl = array![-1.0f64, -1.0];
        let d = array![2.0f64, 2.0, 2.0];
        let du = array![-1.0f64, -1.0];
        let b = array![1.0f64, 0.0, 1.0];

        let x = tridiag_solve_ndarray(&dl, &d, &du, &b).unwrap();

        assert_eq!(x.len(), 3);

        // Verify solution: T * x ≈ b
        let tx0 = d[0] * x[0] + du[0] * x[1];
        let tx1 = dl[0] * x[0] + d[1] * x[1] + du[1] * x[2];
        let tx2 = dl[1] * x[1] + d[2] * x[2];

        assert!((tx0 - b[0]).abs() < 1e-10);
        assert!((tx1 - b[1]).abs() < 1e-10);
        assert!((tx2 - b[2]).abs() < 1e-10);
    }

    #[test]
    fn test_tridiag_solve_spd() {
        // SPD tridiagonal matrix:
        // [4 1 0]
        // [1 4 1]
        // [0 1 4]
        // This is diagonally dominant -> SPD
        let d = array![4.0f64, 4.0, 4.0];
        let e = array![1.0f64, 1.0]; // Off-diagonal elements
        let b = array![5.0f64, 6.0, 5.0];

        let x = tridiag_solve_spd_ndarray(&d, &e, &b).unwrap();

        assert_eq!(x.len(), 3);

        // Verify solution: T * x = b where T is symmetric with d on diagonal, e on off-diagonals
        let tx0 = d[0] * x[0] + e[0] * x[1];
        let tx1 = e[0] * x[0] + d[1] * x[1] + e[1] * x[2];
        let tx2 = e[1] * x[1] + d[2] * x[2];

        assert!((tx0 - b[0]).abs() < 1e-10, "tx0 = {}, b[0] = {}", tx0, b[0]);
        assert!((tx1 - b[1]).abs() < 1e-10, "tx1 = {}, b[1] = {}", tx1, b[1]);
        assert!((tx2 - b[2]).abs() < 1e-10, "tx2 = {}, b[2] = {}", tx2, b[2]);
    }

    #[test]
    fn test_tridiag_solve_multiple() {
        let dl = array![-1.0f64, -1.0];
        let d = array![2.0f64, 2.0, 2.0];
        let du = array![-1.0f64, -1.0];
        let b = array![[1.0f64, 0.0], [0.0, 1.0], [1.0, 0.0]];

        let x = tridiag_solve_multiple_ndarray(&dl, &d, &du, &b).unwrap();

        assert_eq!(x.dim(), (3, 2));

        // Each column should be the solution to T * x_j = b_j
        for j in 0..2 {
            let tx0 = d[0] * x[[0, j]] + du[0] * x[[1, j]];
            let tx1 = dl[0] * x[[0, j]] + d[1] * x[[1, j]] + du[1] * x[[2, j]];
            let tx2 = dl[1] * x[[1, j]] + d[2] * x[[2, j]];

            assert!((tx0 - b[[0, j]]).abs() < 1e-10);
            assert!((tx1 - b[[1, j]]).abs() < 1e-10);
            assert!((tx2 - b[[2, j]]).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Low-Rank Approximation Tests
    // =========================================================================

    #[test]
    fn test_low_rank_approx() {
        // Create a rank-1 matrix: outer product of two vectors
        let u = array![1.0f64, 2.0, 3.0];
        let v = array![4.0, 5.0, 6.0, 7.0];

        let mut a = Array2::zeros((3, 4));
        for i in 0..3 {
            for j in 0..4 {
                a[[i, j]] = u[i] * v[j];
            }
        }

        // Rank-1 approximation should be exact
        let approx = low_rank_approx_ndarray(&a, 1).unwrap();

        assert_eq!(approx.dim(), a.dim());

        for i in 0..3 {
            for j in 0..4 {
                assert!(
                    (approx[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Approximation failed at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_low_rank_approx_truncation() {
        let a = array![
            [1.0f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let approx = low_rank_approx_ndarray(&a, 2).unwrap();

        assert_eq!(approx.dim(), (4, 3));

        // The approximation should not equal the original (rank 2 < rank A)
        // but should be close
        let mut diff_norm = 0.0f64;
        let mut orig_norm = 0.0f64;
        for i in 0..4 {
            for j in 0..3 {
                diff_norm += (a[[i, j]] - approx[[i, j]]).powi(2);
                orig_norm += a[[i, j]].powi(2);
            }
        }

        // Relative error should be small (this matrix has rank 2)
        let rel_error = diff_norm.sqrt() / orig_norm.sqrt();
        assert!(rel_error < 0.1, "Relative error = {}", rel_error);
    }
}
