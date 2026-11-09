//! Parallel Singular Value Decomposition.
//!
//! [`ParallelSvdDc`] is the divide-and-conquer SVD of [`SvdDc`] with every
//! Level-3 step (the trailing updates of the bidiagonal reduction, the
//! GEMM back-transforms of the divide-and-conquer merges and the
//! compact-WY back-transformation) run on the Rayon pool:
//! `ParallelSvdDc::compute(a)` is `SvdDc::compute_with_par(a, Par::Rayon)`
//! and returns bitwise the same factors as [`SvdDc::compute`]. The
//! reconstruction and the pseudoinverse of large matrices are evaluated in
//! parallel as well.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "parallel")] {
//! use oxiblas_lapack::svd::ParallelSvdDc;
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0],
//!     &[3.0, 4.0],
//!     &[5.0, 6.0],
//! ]);
//!
//! let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
//! let sigma = svd.singular_values();
//! assert_eq!(sigma.len(), 2);
//! assert!(sigma[0] >= sigma[1]); // sorted in descending order
//! # }
//! ```

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};
use rayon::prelude::*;

use super::divide_conquer::{SvdDc, SvdDcError};

/// Minimum size to use parallel algorithms.
const PARALLEL_THRESHOLD: usize = 64;

/// Error type for parallel SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
}

impl core::fmt::Display for ParallelSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "SVD algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for ParallelSvdError {}

/// Parallel SVD using divide-and-conquer algorithm.
///
/// Computes A = U·Σ·V^T with the algorithm of [`SvdDc`], running its matrix
/// multiplications on the Rayon pool (see the module documentation).
#[derive(Debug, Clone)]
pub struct ParallelSvdDc<T: Scalar> {
    /// Left singular vectors (m×m orthogonal matrix).
    u: Mat<T>,
    /// Singular values (sorted in descending order).
    sigma: Vec<T>,
    /// Right singular vectors (n×n orthogonal matrix, stored as V^T).
    vt: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable + Send + Sync> ParallelSvdDc<T> {
    /// Computes the full SVD using parallel divide-and-conquer.
    ///
    /// Equivalent to `SvdDc::compute_with_par(a, Par::Rayon)`: the factors
    /// are bitwise identical to those of [`SvdDc::compute`].
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use oxiblas_lapack::svd::ParallelSvdDc;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
    /// assert_eq!(svd.singular_values(), &[4.0, 3.0]);
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`SvdDc::compute`], mapped to the variants of the same
    /// name.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ParallelSvdError> {
        let m = a.nrows();
        let n = a.ncols();
        let svd = SvdDc::compute_with_par(a, Par::Rayon).map_err(|e| match e {
            SvdDcError::EmptyMatrix => ParallelSvdError::EmptyMatrix,
            SvdDcError::NotConverged => ParallelSvdError::NotConverged,
            SvdDcError::SecularEquationFailed => ParallelSvdError::SecularEquationFailed,
        })?;
        let (u, sigma, vt) = svd.into_parts();
        Ok(Self { u, sigma, vt, m, n })
    }

    /// Returns the singular values (sorted in descending order).
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the left singular vectors U (m×m orthogonal matrix).
    pub fn u(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }

    /// Returns V^T (n×n orthogonal matrix).
    pub fn vt(&self) -> MatRef<'_, T> {
        self.vt.as_ref()
    }

    /// Returns the thin U matrix (m×k where k = min(m,n)).
    pub fn u_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut u_thin = Mat::zeros(self.m, k);
        for i in 0..self.m {
            for j in 0..k {
                u_thin[(i, j)] = self.u[(i, j)];
            }
        }
        u_thin
    }

    /// Returns the thin V^T matrix (k×n where k = min(m,n)).
    pub fn vt_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut vt_thin = Mat::zeros(k, self.n);
        for i in 0..k {
            for j in 0..self.n {
                vt_thin[(i, j)] = self.vt[(i, j)];
            }
        }
        vt_thin
    }

    /// Computes the rank given a tolerance.
    pub fn rank(&self, tol: T) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }

    /// Computes the 2-norm (largest singular value).
    pub fn norm2(&self) -> T {
        if self.sigma.is_empty() {
            T::zero()
        } else {
            self.sigma[0]
        }
    }

    /// Computes the condition number.
    pub fn condition_number(&self) -> T {
        if self.sigma.is_empty() {
            T::zero()
        } else {
            let max_sv = self.sigma[0];
            let min_sv = self.sigma[self.sigma.len() - 1];
            if min_sv > T::zero() {
                max_sv / min_sv
            } else {
                <T as Scalar>::max_value()
            }
        }
    }

    /// Reconstructs the original matrix: A = U·Σ·V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        if self.m >= PARALLEL_THRESHOLD || self.n >= PARALLEL_THRESHOLD {
            // Parallel reconstruction
            let indices: Vec<(usize, usize)> = (0..self.m)
                .flat_map(|i| (0..self.n).map(move |j| (i, j)))
                .collect();

            let results: Vec<(usize, usize, T)> = indices
                .par_iter()
                .map(|&(i, j)| {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                    }
                    (i, j, sum)
                })
                .collect();

            let mut a = Mat::zeros(self.m, self.n);
            for (i, j, val) in results {
                a[(i, j)] = val;
            }
            a
        } else {
            // Sequential for small matrices
            let mut a = Mat::zeros(self.m, self.n);
            for i in 0..self.m {
                for j in 0..self.n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                    }
                    a[(i, j)] = sum;
                }
            }
            a
        }
    }

    /// Computes the pseudoinverse using SVD.
    pub fn pseudoinverse(&self, tol: T) -> Mat<T> {
        let k = self.m.min(self.n);

        if self.n >= PARALLEL_THRESHOLD || self.m >= PARALLEL_THRESHOLD {
            let indices: Vec<(usize, usize)> = (0..self.n)
                .flat_map(|i| (0..self.m).map(move |j| (i, j)))
                .collect();

            let sigma = &self.sigma;
            let u = &self.u;
            let vt = &self.vt;

            let results: Vec<(usize, usize, T)> = indices
                .par_iter()
                .map(|&(i, j)| {
                    let mut sum = T::zero();
                    for l in 0..k {
                        if sigma[l] > tol {
                            sum = sum + vt[(l, i)] * (T::one() / sigma[l]) * u[(j, l)];
                        }
                    }
                    (i, j, sum)
                })
                .collect();

            let mut pinv = Mat::zeros(self.n, self.m);
            for (i, j, val) in results {
                pinv[(i, j)] = val;
            }
            pinv
        } else {
            let mut pinv = Mat::zeros(self.n, self.m);
            for i in 0..self.n {
                for j in 0..self.m {
                    let mut sum = T::zero();
                    for l in 0..k {
                        if self.sigma[l] > tol {
                            sum =
                                sum + self.vt[(l, i)] * (T::one() / self.sigma[l]) * self.u[(j, l)];
                        }
                    }
                    pinv[(i, j)] = sum;
                }
            }
            pinv
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_parallel_svd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 4.0, 1e-10));
        assert!(approx_eq(sigma[1], 3.0, 1e-10));
    }

    #[test]
    fn test_parallel_svd_2x2() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_parallel_svd_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8));
            }
        }
    }

    #[test]
    fn test_parallel_svd_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_parallel_svd_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let svd = ParallelSvdDc::compute(eye.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for &s in sigma {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_parallel_svd_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 1);
        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_parallel_svd_norm() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.norm2(), 4.0, 1e-10));
    }

    #[test]
    fn test_parallel_svd_condition() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.condition_number(), 2.0, 1e-10));
    }

    #[test]
    fn test_parallel_svd_vs_sequential() {
        use crate::svd::SvdDc;

        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd_seq = SvdDc::compute(a.as_ref()).unwrap();
        let svd_par = ParallelSvdDc::compute(a.as_ref()).unwrap();

        let sigma_seq = svd_seq.singular_values();
        let sigma_par = svd_par.singular_values();

        for i in 0..3 {
            assert!(
                sigma_seq[i] == sigma_par[i],
                "singular value {} mismatch: seq={}, par={}",
                i,
                sigma_seq[i],
                sigma_par[i]
            );
        }
    }

    /// The former stand-alone parallel solver returned `Ok` with residuals of
    /// order one on such inputs; the factors now equal those of `SvdDc`
    /// bitwise, whatever the size, scale or rank.
    #[test]
    fn test_parallel_svd_bitwise_equals_svd_dc() {
        use crate::svd::SvdDc;
        let mut state = 0x1234_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        for &(m, n, f, zero_col) in &[
            (2usize, 2usize, 1.0f64, true),
            (20, 20, 1.0, true),
            (40, 30, 1e-150, false),
            (30, 40, 1e150, false),
            (200, 200, 1.0, false),
        ] {
            let mut a = Mat::<f64>::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    a[(i, j)] = if zero_col && j == 0 { 0.0 } else { f * next() };
                }
            }
            let par = ParallelSvdDc::compute(a.as_ref()).unwrap();
            let seq = SvdDc::compute(a.as_ref()).unwrap();
            assert_eq!(par.singular_values(), seq.singular_values(), "{m}x{n}");
            for j in 0..m {
                for i in 0..m {
                    assert_eq!(par.u()[(i, j)].to_bits(), seq.u()[(i, j)].to_bits());
                }
            }
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(par.vt()[(i, j)].to_bits(), seq.vt()[(i, j)].to_bits());
                }
            }
            // And the reconstruction is exact to the ε level.
            let rec = par.reconstruct();
            let amax = (0..n)
                .flat_map(|j| (0..m).map(move |i| (i, j)))
                .fold(0.0f64, |acc, (i, j)| acc.max(a[(i, j)].abs()));
            for j in 0..n {
                for i in 0..m {
                    let err = (rec[(i, j)] - a[(i, j)]).abs() / amax;
                    assert!(
                        err < 16.0 * (m.max(n) as f64) * f64::EPSILON,
                        "{m}x{n}: {err}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_parallel_svd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let svd = ParallelSvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-4,
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }
}
