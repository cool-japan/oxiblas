//! Truncated Singular Value Decomposition.
//!
//! Computes only the k largest singular values and their corresponding
//! singular vectors. This is more efficient than computing the full SVD
//! when only the dominant components are needed.
//!
//! # Variants
//!
//! - **TruncatedSvd**: Exact truncated SVD using bidiagonalization + partial eigendecomposition
//! - **thin_svd**: Economy-size SVD (min(m,n) singular values)
//!
//! # Algorithm
//!
//! For k largest singular values:
//! 1. Reduce A to bidiagonal form: A = U_b * B * V_b^T
//! 2. Form T = B^T * B (symmetric tridiagonal)
//! 3. Compute k largest eigenvalues of T using bisection
//! 4. Recover singular values as sqrt(eigenvalues)
//! 5. Compute singular vectors using inverse iteration
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::svd::{TruncatedSvd, thin_svd};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 3.0, 4.0, 5.0],
//!     &[6.0, 7.0, 8.0, 9.0, 10.0],
//!     &[11.0, 12.0, 13.0, 14.0, 15.0],
//! ]);
//!
//! // Compute only the 2 largest singular values
//! let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
//! assert_eq!(tsvd.singular_values().len(), 2);
//! assert!(tsvd.singular_values()[0] >= tsvd.singular_values()[1]);
//!
//! // Low-rank approximation
//! let approx = tsvd.reconstruct();
//! assert_eq!(approx.nrows(), 3);
//! assert_eq!(approx.ncols(), 5);
//!
//! // Thin SVD: compute all min(m,n) singular values efficiently
//! let thin = thin_svd(a.as_ref()).unwrap();
//! assert_eq!(thin.singular_values().len(), 3);
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::selective::{SelectiveSvd, SelectiveSvdError, SingularValueSelector};
use super::{SvdDc, SvdDcError};

/// Error type for truncated SVD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Requested rank is zero.
    ZeroRank,
    /// Requested rank exceeds matrix dimensions.
    RankTooLarge {
        /// Requested rank.
        requested: usize,
        /// Maximum possible rank.
        max_rank: usize,
    },
    /// Algorithm did not converge.
    NotConverged,
    /// Internal computation failed.
    InternalError,
}

impl core::fmt::Display for TruncatedSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::ZeroRank => write!(f, "Requested rank is zero"),
            Self::RankTooLarge {
                requested,
                max_rank,
            } => {
                write!(
                    f,
                    "Requested rank {} exceeds maximum {}",
                    requested, max_rank
                )
            }
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::InternalError => write!(f, "Internal computation failed"),
        }
    }
}

impl std::error::Error for TruncatedSvdError {}

impl From<SelectiveSvdError> for TruncatedSvdError {
    fn from(e: SelectiveSvdError) -> Self {
        match e {
            SelectiveSvdError::EmptyMatrix => Self::EmptyMatrix,
            SelectiveSvdError::InvalidRange => Self::InternalError,
            SelectiveSvdError::NoSingularValuesInRange => Self::InternalError,
            SelectiveSvdError::NotConverged => Self::NotConverged,
            SelectiveSvdError::InvalidIndexRange => Self::RankTooLarge {
                requested: 0,
                max_rank: 0,
            },
        }
    }
}

impl From<SvdDcError> for TruncatedSvdError {
    fn from(e: SvdDcError) -> Self {
        match e {
            SvdDcError::EmptyMatrix => Self::EmptyMatrix,
            SvdDcError::NotConverged => Self::NotConverged,
            SvdDcError::SecularEquationFailed => Self::NotConverged,
        }
    }
}

/// Truncated SVD result.
///
/// Contains the k largest singular values and their corresponding vectors.
#[derive(Debug, Clone)]
pub struct TruncatedSvd<T: Scalar> {
    /// Left singular vectors U (m × k).
    u: Mat<T>,
    /// Singular values (k elements, descending order).
    sigma: Vec<T>,
    /// Right singular vectors as V^T (k × n).
    vt: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
    /// Number of computed singular values.
    k: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> TruncatedSvd<T> {
    /// Computes the k largest singular values and their vectors.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m × n)
    /// * `k` - Number of singular values to compute (must be ≤ min(m, n))
    ///
    /// # Returns
    ///
    /// Truncated SVD with k singular values and vectors.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::TruncatedSvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    ///     &[7.0, 8.0, 9.0],
    /// ]);
    ///
    /// // Compute the largest singular value only
    /// let tsvd = TruncatedSvd::compute(a.as_ref(), 1).unwrap();
    /// assert_eq!(tsvd.singular_values().len(), 1);
    /// ```
    pub fn compute(a: MatRef<'_, T>, k: usize) -> Result<Self, TruncatedSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(TruncatedSvdError::EmptyMatrix);
        }

        if k == 0 {
            return Err(TruncatedSvdError::ZeroRank);
        }

        let max_rank = m.min(n);
        if k > max_rank {
            return Err(TruncatedSvdError::RankTooLarge {
                requested: k,
                max_rank,
            });
        }

        // For small k relative to matrix size, use SelectiveSvd
        // For k close to max_rank, use full SVD and truncate
        let use_selective = k <= max_rank / 2;

        if use_selective {
            // Use SelectiveSvd for efficiency
            let selector = SingularValueSelector::IndexRange {
                low: 0,
                high: k - 1,
            };
            let svd = SelectiveSvd::compute(a, selector)?;

            let u = svd.u().map_or_else(
                || Mat::zeros(m, k),
                |u_ref| {
                    let mut u = Mat::zeros(m, k);
                    for i in 0..m {
                        for j in 0..k.min(u_ref.ncols()) {
                            u[(i, j)] = u_ref[(i, j)];
                        }
                    }
                    u
                },
            );

            let vt = svd.vt().map_or_else(
                || Mat::zeros(k, n),
                |vt_ref| {
                    let mut vt = Mat::zeros(k, n);
                    for i in 0..k.min(vt_ref.nrows()) {
                        for j in 0..n {
                            vt[(i, j)] = vt_ref[(i, j)];
                        }
                    }
                    vt
                },
            );

            let sigma = svd.singular_values().to_vec();

            Ok(Self {
                u,
                sigma,
                vt,
                m,
                n,
                k,
            })
        } else {
            // Use divide-and-conquer SVD and truncate
            let svd = SvdDc::compute(a)?;

            let u_full = svd.u();
            let vt_full = svd.vt();
            let sigma_full = svd.singular_values();

            // Truncate to k
            let mut u = Mat::zeros(m, k);
            for i in 0..m {
                for j in 0..k {
                    u[(i, j)] = u_full[(i, j)];
                }
            }

            let mut vt = Mat::zeros(k, n);
            for i in 0..k {
                for j in 0..n {
                    vt[(i, j)] = vt_full[(i, j)];
                }
            }

            let sigma = sigma_full[..k].to_vec();

            Ok(Self {
                u,
                sigma,
                vt,
                m,
                n,
                k,
            })
        }
    }

    /// Computes only singular values (no vectors).
    ///
    /// This is faster when only the singular values are needed.
    pub fn singular_values_only(a: MatRef<'_, T>, k: usize) -> Result<Vec<T>, TruncatedSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(TruncatedSvdError::EmptyMatrix);
        }

        if k == 0 {
            return Err(TruncatedSvdError::ZeroRank);
        }

        let max_rank = m.min(n);
        if k > max_rank {
            return Err(TruncatedSvdError::RankTooLarge {
                requested: k,
                max_rank,
            });
        }

        let selector = SingularValueSelector::IndexRange {
            low: 0,
            high: k - 1,
        };
        let svd = SelectiveSvd::singular_values_only(a, selector)?;

        Ok(svd.singular_values().to_vec())
    }

    /// Returns the left singular vectors U (m × k matrix).
    pub fn u(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }

    /// Returns an owned copy of the left singular vectors.
    pub fn u_matrix(&self) -> Mat<T> {
        self.u.clone()
    }

    /// Returns the singular values in descending order.
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns V^T (k × n matrix).
    pub fn vt(&self) -> MatRef<'_, T> {
        self.vt.as_ref()
    }

    /// Returns an owned copy of V^T.
    pub fn vt_matrix(&self) -> Mat<T> {
        self.vt.clone()
    }

    /// Returns V (n × k matrix, transposed from V^T).
    pub fn v(&self) -> Mat<T> {
        let mut v = Mat::zeros(self.n, self.k);
        for i in 0..self.n {
            for j in 0..self.k {
                v[(i, j)] = self.vt[(j, i)];
            }
        }
        v
    }

    /// Returns the number of computed singular values.
    pub fn rank(&self) -> usize {
        self.k
    }

    /// Returns the original matrix dimensions (m, n).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Reconstructs the low-rank approximation: A_k = U × Σ × V^T.
    pub fn reconstruct(&self) -> Mat<T> {
        let mut result = Mat::zeros(self.m, self.n);

        // A_k = U × diag(σ) × V^T
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..self.k {
                    sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                }
                result[(i, j)] = sum;
            }
        }

        result
    }

    /// Computes the Frobenius norm of the approximation error.
    pub fn reconstruction_error(&self, a: MatRef<'_, T>) -> T {
        let approx = self.reconstruct();
        let mut error_sq = T::zero();

        for i in 0..self.m {
            for j in 0..self.n {
                let diff = a[(i, j)] - approx[(i, j)];
                error_sq = error_sq + diff * diff;
            }
        }

        Real::sqrt(error_sq)
    }

    /// Computes the relative Frobenius error ||A - A_k||_F / ||A||_F.
    pub fn relative_error(&self, a: MatRef<'_, T>) -> T {
        let error = self.reconstruction_error(a);

        let mut a_norm_sq = T::zero();
        for i in 0..self.m {
            for j in 0..self.n {
                a_norm_sq = a_norm_sq + a[(i, j)] * a[(i, j)];
            }
        }

        let a_norm = Real::sqrt(a_norm_sq);
        if a_norm > T::zero() {
            error / a_norm
        } else {
            T::zero()
        }
    }

    /// Returns the sum of computed singular values (partial nuclear norm).
    pub fn nuclear_norm(&self) -> T {
        self.sigma.iter().copied().fold(T::zero(), |acc, s| acc + s)
    }

    /// Returns the energy captured by the truncation.
    ///
    /// This is the ratio of the sum of squared singular values in the truncation
    /// to the total (computed from Frobenius norm). Returns 1.0 if original matrix
    /// is not provided.
    pub fn energy_ratio(&self, a: MatRef<'_, T>) -> T {
        let mut a_frob_sq = T::zero();
        for i in 0..self.m {
            for j in 0..self.n {
                a_frob_sq = a_frob_sq + a[(i, j)] * a[(i, j)];
            }
        }

        if a_frob_sq <= T::zero() {
            return T::one();
        }

        let truncated_energy: T = self
            .sigma
            .iter()
            .map(|&s| s * s)
            .fold(T::zero(), |a, b| a + b);

        truncated_energy / a_frob_sq
    }

    /// Returns the explained variance ratio for each component.
    ///
    /// Each ratio represents σ_i² / ||A||_F².
    pub fn explained_variance_ratio(&self, a: MatRef<'_, T>) -> Vec<T> {
        let mut a_frob_sq = T::zero();
        for i in 0..self.m {
            for j in 0..self.n {
                a_frob_sq = a_frob_sq + a[(i, j)] * a[(i, j)];
            }
        }

        if a_frob_sq <= T::zero() {
            return vec![T::zero(); self.k];
        }

        self.sigma.iter().map(|&s| (s * s) / a_frob_sq).collect()
    }

    /// Applies the truncated SVD to project a vector onto the principal subspace.
    ///
    /// Computes V^T * x for a vector x.
    pub fn project(&self, x: &[T]) -> Vec<T> {
        assert_eq!(x.len(), self.n);

        let mut result = vec![T::zero(); self.k];
        for i in 0..self.k {
            for j in 0..self.n {
                result[i] = result[i] + self.vt[(i, j)] * x[j];
            }
        }
        result
    }

    /// Projects a vector back to the original space.
    ///
    /// Computes V * y for a vector y in the reduced space.
    pub fn inverse_project(&self, y: &[T]) -> Vec<T> {
        assert_eq!(y.len(), self.k);

        let mut result = vec![T::zero(); self.n];
        for i in 0..self.n {
            for j in 0..self.k {
                result[i] = result[i] + self.vt[(j, i)] * y[j];
            }
        }
        result
    }
}

/// Computes the thin (economy-size) SVD.
///
/// Returns min(m, n) singular values and the corresponding vectors.
/// This is equivalent to `TruncatedSvd::compute(a, min(m, n))` but
/// optimized for this case.
///
/// # Arguments
///
/// * `a` - Input matrix (m × n)
///
/// # Returns
///
/// Thin SVD with min(m, n) singular values and vectors.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::svd::thin_svd;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0, 3.0, 4.0, 5.0],
///     &[6.0, 7.0, 8.0, 9.0, 10.0],
///     &[11.0, 12.0, 13.0, 14.0, 15.0],
/// ]);
///
/// let thin = thin_svd(a.as_ref()).unwrap();
/// assert_eq!(thin.singular_values().len(), 3); // min(3, 5) = 3
/// ```
pub fn thin_svd<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<TruncatedSvd<T>, TruncatedSvdError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(TruncatedSvdError::EmptyMatrix);
    }

    let k = m.min(n);

    // Use divide-and-conquer for full thin SVD
    let svd = SvdDc::compute(a)?;

    let u_full = svd.u();
    let vt_full = svd.vt();
    let sigma_full = svd.singular_values();

    // Extract thin components (k singular values/vectors)
    let mut u = Mat::zeros(m, k);
    for i in 0..m {
        for j in 0..k {
            u[(i, j)] = u_full[(i, j)];
        }
    }

    let mut vt = Mat::zeros(k, n);
    for i in 0..k {
        for j in 0..n {
            vt[(i, j)] = vt_full[(i, j)];
        }
    }

    let sigma = sigma_full[..k].to_vec();

    Ok(TruncatedSvd {
        u,
        sigma,
        vt,
        m,
        n,
        k,
    })
}

/// Computes the best rank-k approximation to a matrix.
///
/// Returns the matrix A_k = U_k × Σ_k × V_k^T that minimizes
/// ||A - A_k||_F (Frobenius norm) among all rank-k matrices.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `k` - Target rank
///
/// # Returns
///
/// The rank-k approximation matrix.
pub fn rank_k_approximation<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    k: usize,
) -> Result<Mat<T>, TruncatedSvdError> {
    let tsvd = TruncatedSvd::compute(a, k)?;
    Ok(tsvd.reconstruct())
}

/// Estimates the optimal rank for a given target energy ratio.
///
/// Returns the minimum k such that the first k singular values
/// capture at least the specified fraction of total energy.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `target_energy` - Target energy ratio (0 < target_energy ≤ 1)
///
/// # Returns
///
/// The optimal rank k.
pub fn optimal_rank_for_energy<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    target_energy: T,
) -> Result<usize, TruncatedSvdError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(TruncatedSvdError::EmptyMatrix);
    }

    // Compute total Frobenius norm squared
    let mut total_energy = T::zero();
    for i in 0..m {
        for j in 0..n {
            total_energy = total_energy + a[(i, j)] * a[(i, j)];
        }
    }

    if total_energy <= T::zero() {
        return Ok(0);
    }

    // Get all singular values
    let thin = thin_svd(a)?;
    let sigma = thin.singular_values();

    // Find minimum k to reach target energy
    let mut accumulated = T::zero();
    let threshold = target_energy * total_energy;

    for (k, &s) in sigma.iter().enumerate() {
        accumulated = accumulated + s * s;
        if accumulated >= threshold {
            return Ok(k + 1);
        }
    }

    Ok(sigma.len())
}

/// Computes the numerical rank of a matrix using SVD.
///
/// The numerical rank is the number of singular values above
/// the tolerance threshold `tol * σ_max`.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `tol` - Relative tolerance (default: machine epsilon * max(m, n))
///
/// # Returns
///
/// The numerical rank.
pub fn numerical_rank<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    tol: Option<T>,
) -> Result<usize, TruncatedSvdError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(TruncatedSvdError::EmptyMatrix);
    }

    let thin = thin_svd(a)?;
    let sigma = thin.singular_values();

    if sigma.is_empty() || sigma[0] <= T::zero() {
        return Ok(0);
    }

    let eps = <T as Scalar>::epsilon();
    let default_tol = eps * T::from_usize(m.max(n)).unwrap_or(T::one());
    let threshold = tol.unwrap_or(default_tol) * sigma[0];

    let rank = sigma.iter().take_while(|&&s| s > threshold).count();
    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_truncated_svd_basic() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        assert_eq!(tsvd.rank(), 2);
        assert_eq!(tsvd.singular_values().len(), 2);

        // Singular values should be descending
        assert!(tsvd.singular_values()[0] >= tsvd.singular_values()[1]);
    }

    #[test]
    fn test_truncated_svd_rank1() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 1).unwrap();
        assert_eq!(tsvd.rank(), 1);
        assert_eq!(tsvd.u().nrows(), 2);
        assert_eq!(tsvd.u().ncols(), 1);
        assert_eq!(tsvd.vt().nrows(), 1);
        assert_eq!(tsvd.vt().ncols(), 3);
    }

    #[test]
    fn test_truncated_svd_dimensions() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0, 5.0],
            &[6.0, 7.0, 8.0, 9.0, 10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0],
        ]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        assert_eq!(tsvd.dimensions(), (3, 5));
        assert_eq!(tsvd.u().nrows(), 3);
        assert_eq!(tsvd.u().ncols(), 2);
        assert_eq!(tsvd.vt().nrows(), 2);
        assert_eq!(tsvd.vt().ncols(), 5);
    }

    #[test]
    fn test_truncated_svd_reconstruction() {
        // Use a well-conditioned matrix
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[1.0, 3.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        let approx = tsvd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(approx[(i, j)], a[(i, j)], 0.1),
                    "approx[{},{}] = {}, expected {}",
                    i,
                    j,
                    approx[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_thin_svd() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0, 5.0],
            &[6.0, 7.0, 8.0, 9.0, 10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0],
        ]);

        let thin = thin_svd(a.as_ref()).unwrap();
        assert_eq!(thin.rank(), 3); // min(3, 5) = 3
        assert_eq!(thin.singular_values().len(), 3);
    }

    #[test]
    fn test_thin_svd_tall() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ]);

        let thin = thin_svd(a.as_ref()).unwrap();
        assert_eq!(thin.rank(), 2); // min(5, 2) = 2
    }

    #[test]
    fn test_rank_k_approximation() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let approx = rank_k_approximation(a.as_ref(), 1).unwrap();
        assert_eq!(approx.nrows(), 2);
        assert_eq!(approx.ncols(), 3);
    }

    #[test]
    fn test_singular_values_only() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let sigma = TruncatedSvd::singular_values_only(a.as_ref(), 2).unwrap();
        assert_eq!(sigma.len(), 2);
        assert!(sigma[0] >= sigma[1]);
    }

    #[test]
    fn test_truncated_svd_error_empty() {
        let a: Mat<f64> = Mat::zeros(0, 3);
        let result = TruncatedSvd::compute(a.as_ref(), 1);
        assert!(matches!(result, Err(TruncatedSvdError::EmptyMatrix)));
    }

    #[test]
    fn test_truncated_svd_error_zero_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let result = TruncatedSvd::compute(a.as_ref(), 0);
        assert!(matches!(result, Err(TruncatedSvdError::ZeroRank)));
    }

    #[test]
    fn test_truncated_svd_error_rank_too_large() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let result = TruncatedSvd::compute(a.as_ref(), 5);
        assert!(matches!(
            result,
            Err(TruncatedSvdError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn test_truncated_svd_orthogonality() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        let u = tsvd.u();
        let vt = tsvd.vt();

        // Check U^T * U ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += u[(k, i)] * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 0.1),
                    "U^T*U[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Check V * V^T ≈ I (using V^T rows)
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += vt[(i, k)] * vt[(j, k)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 0.1),
                    "V*V^T[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_energy_ratio() {
        // Diagonal matrix with known singular values
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 4.0, 0.0], &[0.0, 0.0, 5.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 1).unwrap();
        let energy = tsvd.energy_ratio(a.as_ref());

        // First singular value is 5, total energy = 9 + 16 + 25 = 50
        // Energy from σ=5: 25/50 = 0.5
        assert!(
            approx_eq(energy, 0.5, 0.1),
            "Energy ratio {} should be ~0.5",
            energy
        );
    }

    #[test]
    fn test_explained_variance_ratio() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        let ratios = tsvd.explained_variance_ratio(a.as_ref());

        // Singular values: 4, 3
        // Total energy: 9 + 16 = 25
        // First: 16/25 = 0.64, Second: 9/25 = 0.36
        assert!(
            approx_eq(ratios[0], 0.64, 0.1),
            "First ratio {} should be ~0.64",
            ratios[0]
        );
        assert!(
            approx_eq(ratios[1], 0.36, 0.1),
            "Second ratio {} should be ~0.36",
            ratios[1]
        );
    }

    #[test]
    fn test_project_and_inverse_project() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 3.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();

        let x = vec![1.0, 1.0, 1.0];
        let projected = tsvd.project(&x);
        assert_eq!(projected.len(), 2);

        let reconstructed = tsvd.inverse_project(&projected);
        assert_eq!(reconstructed.len(), 3);
    }

    #[test]
    fn test_optimal_rank_for_energy() {
        // Diagonal matrix with decreasing singular values
        let a = Mat::from_rows(&[
            &[10.0f64, 0.0, 0.0, 0.0],
            &[0.0, 5.0, 0.0, 0.0],
            &[0.0, 0.0, 2.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
        ]);

        // Total energy = 100 + 25 + 4 + 1 = 130
        // First: 100/130 ≈ 0.77
        // First two: 125/130 ≈ 0.96

        let k = optimal_rank_for_energy(a.as_ref(), 0.9).unwrap();
        assert!(k <= 3, "Should need at most 3 components for 90% energy");

        let k = optimal_rank_for_energy(a.as_ref(), 0.99).unwrap();
        assert!(k >= 2, "Should need at least 2 components for 99% energy");
    }

    #[test]
    fn test_numerical_rank() {
        // Full rank matrix
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 3.0]]);

        let rank = numerical_rank(a.as_ref(), None).unwrap();
        assert_eq!(rank, 3);

        // Rank-deficient matrix (rank 2)
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]);

        let rank = numerical_rank(a.as_ref(), Some(1e-8)).unwrap();
        assert!(rank <= 2, "Rank should be at most 2 for rank-1 matrix");
    }

    #[test]
    fn test_v_matrix() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        let v = tsvd.v();

        assert_eq!(v.nrows(), 3);
        assert_eq!(v.ncols(), 2);

        // V should be transpose of V^T
        let vt = tsvd.vt();
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(v[(i, j)], vt[(j, i)], 1e-10),
                    "V[{},{}] = {} should equal Vt[{},{}] = {}",
                    i,
                    j,
                    v[(i, j)],
                    j,
                    i,
                    vt[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_truncated_svd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        assert_eq!(tsvd.rank(), 2);
    }

    #[test]
    fn test_truncated_svd_relative_error() {
        // Well-conditioned matrix
        let a = Mat::from_rows(&[&[4.0f64, 1.0], &[1.0, 3.0]]);

        // Full reconstruction should have near-zero error
        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        let error = tsvd.relative_error(a.as_ref());
        assert!(
            error < 0.1,
            "Full reconstruction error {} should be small",
            error
        );
    }

    #[test]
    fn test_nuclear_norm() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 3.0]]);

        let tsvd = TruncatedSvd::compute(a.as_ref(), 2).unwrap();
        let nuclear = tsvd.nuclear_norm();

        // Nuclear norm should be 2 + 3 = 5
        assert!(
            approx_eq(nuclear, 5.0, 0.5),
            "Nuclear norm {} should be ~5",
            nuclear
        );
    }
}
