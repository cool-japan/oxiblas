//! SVD using one-sided Jacobi algorithm.
//!
//! Computes A = U·Σ·V^T where:
//! - U is m×m orthogonal (left singular vectors)
//! - Σ is m×n diagonal (singular values, non-negative, descending)
//! - V is n×n orthogonal (right singular vectors)

use oxiblas_blas::level1::{axpy, dot};
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for SvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "SVD algorithm did not converge"),
        }
    }
}

impl std::error::Error for SvdError {}

/// Singular Value Decomposition result.
///
/// A = U·Σ·V^T where Σ contains singular values on the diagonal.
#[derive(Debug, Clone)]
pub struct Svd<T: Scalar> {
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

impl<T: Field + Real + bytemuck::Zeroable> Svd<T> {
    /// Maximum sweeps for Jacobi iteration.
    const MAX_SWEEPS: usize = 30;

    /// Computes the full SVD of matrix A using one-sided Jacobi algorithm.
    ///
    /// The Jacobi sweeps rotate pairs of the `n` columns, `O(m · n²)` work per
    /// sweep, and give `V`, the singular values and the columns of `U` for the
    /// singular values above `100 · eps`. The other columns of `U` (those of
    /// the zero singular values, and the `m − n` trailing columns of a tall
    /// matrix) complete it to an `m × m` orthogonal matrix: column `j` is the
    /// normalized component, orthogonal to the columns before it, of the first
    /// standard basis vector `e_b` (in increasing `b`) not already in their
    /// span. The completion is reorthogonalized when needed (the DGKS
    /// criterion), so `U` is orthogonal to working precision, and it costs
    /// `O(m²)` per column. For a tall matrix the completion dominates,
    /// `O(m³)` in total; it needs no further work beyond the Jacobi sweeps
    /// for a square or wide matrix of full rank.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::Svd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = Svd::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// // Singular values of diagonal matrix are absolute values of diagonal
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SvdError::EmptyMatrix);
        }

        // Handle 1x1 case
        if m == 1 && n == 1 {
            let val = a[(0, 0)];
            let sigma = vec![Scalar::abs(val)];
            let mut u = Mat::zeros(1, 1);
            let mut vt = Mat::zeros(1, 1);
            u[(0, 0)] = if val >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            vt[(0, 0)] = T::one();
            return Ok(Self { u, sigma, vt, m, n });
        }

        // Working copies with contiguous columns: B = A (m×n, column j at
        // b[j*m..]) is rotated into U·Σ, and V (n×n, column j at v[j*n..])
        // accumulates the rotations.
        let mut b = vec![T::zero(); m * n];
        for j in 0..n {
            for i in 0..m {
                b[j * m + i] = a[(i, j)];
            }
        }
        let mut v = vec![T::zero(); n * n];
        for i in 0..n {
            v[i * n + i] = T::one();
        }

        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(100.0).unwrap_or(T::one());
        let two = T::from_f64(2.0).unwrap_or(T::one());
        let k = m.min(n);

        // One-sided Jacobi: rotate pairs of columns of B until they are
        // mutually orthogonal (B^T B diagonal); V accumulates the rotations.
        for _sweep in 0..Self::MAX_SWEEPS {
            let mut converged = true;

            for i in 0..n {
                for j in (i + 1)..n {
                    let (bi, bj) = column_pair(&mut b, m, i, j);

                    // B[:, i]^T B[:, j] and the squared norms
                    let mut dot_ij = T::zero();
                    let mut norm_i_sq = T::zero();
                    let mut norm_j_sq = T::zero();
                    for (&x, &y) in bi.iter().zip(bj.iter()) {
                        dot_ij = dot_ij + x * y;
                        norm_i_sq = norm_i_sq + x * x;
                        norm_j_sq = norm_j_sq + y * y;
                    }

                    // Check if rotation is needed
                    let off_diag = Scalar::abs(dot_ij);
                    let threshold = tol * Real::sqrt(norm_i_sq) * Real::sqrt(norm_j_sq);

                    if off_diag > threshold {
                        converged = false;

                        // Jacobi rotation diagonalizing the 2x2 Gram matrix
                        // [[norm_i_sq, dot_ij], [dot_ij, norm_j_sq]]
                        let tau = (norm_j_sq - norm_i_sq) / (two * dot_ij);
                        let t = if tau >= T::zero() {
                            T::one() / (tau + Real::sqrt(T::one() + tau * tau))
                        } else {
                            T::one() / (tau - Real::sqrt(T::one() + tau * tau))
                        };
                        let c = T::one() / Real::sqrt(T::one() + t * t);
                        let s = c * t;

                        rotate(bi, bj, c, s);
                        let (vi, vj) = column_pair(&mut v, n, i, j);
                        rotate(vi, vj, c, s);
                    }
                }
            }

            if converged {
                break;
            }
        }

        // The columns of B are now orthogonal; their norms are the singular
        // values. Sort the columns by norm (descending).
        let mut col_norms: Vec<(usize, T)> = Vec::with_capacity(n);
        for j in 0..n {
            let mut norm_sq = T::zero();
            for &x in &b[j * m..(j + 1) * m] {
                norm_sq = norm_sq + x * x;
            }
            col_norms.push((j, Real::sqrt(norm_sq)));
        }
        col_norms.sort_by(|a, b| {
            if b.1 > a.1 {
                core::cmp::Ordering::Greater
            } else if b.1 < a.1 {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        });

        // Singular values, and the columns of U (column j at u[j*m..]) for
        // the singular values above `tol`.
        let mut sigma = vec![T::zero(); k];
        let mut u = vec![T::zero(); m * m];
        for i in 0..k {
            let (col_idx, norm) = col_norms[i];
            sigma[i] = norm;
            if norm > tol {
                for row in 0..m {
                    u[i * m + row] = b[col_idx * m + row] / norm;
                }
            }
        }

        // Complete U to an orthogonal matrix: first the columns of the zero
        // singular values (against the columns before them), then the m - k
        // trailing columns.
        let mut completion = BasisCompletion::new(m, tol);
        for col in 0..k {
            if sigma[col] <= tol {
                completion.fill(&mut u, col);
            }
        }
        for col in k..m {
            completion.fill(&mut u, col);
        }

        let mut u_mat = Mat::zeros(m, m);
        for j in 0..m {
            for i in 0..m {
                u_mat[(i, j)] = u[j * m + i];
            }
        }

        // V^T with its rows in the order of the sorted singular values
        let mut vt = Mat::zeros(n, n);
        for (i, &(col_idx, _)) in col_norms.iter().enumerate() {
            for j in 0..n {
                vt[(i, j)] = v[col_idx * n + j];
            }
        }

        Ok(Self {
            u: u_mat,
            sigma,
            vt,
            m,
            n,
        })
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
    ///
    /// Note: This returns V^T, not V. To get V, transpose this.
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

    /// Computes the rank of the matrix given a tolerance.
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

    /// Computes the condition number (ratio of largest to smallest singular value).
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
        let mut a = Mat::zeros(self.m, self.n);
        let k = self.m.min(self.n);

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

    /// Computes the pseudoinverse using SVD.
    ///
    /// A^+ = V·Σ^+·U^T where Σ^+ has 1/σ_i on diagonal (for non-zero σ_i).
    pub fn pseudoinverse(&self, tol: T) -> Mat<T> {
        let mut pinv = Mat::zeros(self.n, self.m);
        let k = self.m.min(self.n);

        for i in 0..self.n {
            for j in 0..self.m {
                let mut sum = T::zero();
                for l in 0..k {
                    if self.sigma[l] > tol {
                        // V[i,l] * (1/sigma[l]) * U[j,l]
                        // V = Vt^T, so V[i,l] = Vt[l,i]
                        sum = sum + self.vt[(l, i)] * (T::one() / self.sigma[l]) * self.u[(j, l)];
                    }
                }
                pinv[(i, j)] = sum;
            }
        }

        pinv
    }
}

/// Returns mutable views of the columns `i < j` of the column-major buffer
/// `data`, whose columns have length `len`.
fn column_pair<T>(data: &mut [T], len: usize, i: usize, j: usize) -> (&mut [T], &mut [T]) {
    let (head, tail) = data.split_at_mut(j * len);
    (&mut head[i * len..(i + 1) * len], &mut tail[..len])
}

/// Applies the plane rotation `(x, y) <- (c·x − s·y, s·x + c·y)` to two
/// columns.
fn rotate<T: Field>(x: &mut [T], y: &mut [T], c: T, s: T) {
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let (p, q) = (*xi, *yi);
        *xi = c * p - s * q;
        *yi = s * p + c * q;
    }
}

/// Completes `U` with the columns the Jacobi iteration does not provide: those
/// of the zero singular values, and the `m − k` trailing columns of a tall
/// matrix.
///
/// Column `col` becomes the normalized residual, against the columns
/// `0..col`, of the first standard basis vector `e_b` (in increasing `b`)
/// whose residual norm exceeds `tol` — the basis chosen by the previous
/// releases, which only differs from this one by rounding. The residual is
/// computed by classical Gram–Schmidt; the coefficients `u_jᵀ e_b = u_j[b]` of
/// the first pass are read off row `b`. When that pass removes more than half
/// of the squared norm, a second pass follows (the DGKS criterion used by
/// ARPACK, "twice is enough"), so every new column is orthogonal to the
/// previous ones to working precision.
///
/// A candidate is tried at most once: once taken, or once found inside the
/// span of the columns `0..col` (residual norm at or below `tol`), it lies in
/// the span of every later prefix of columns and can never qualify again. A
/// try costs `O(m)` per column already present, so completing `c` columns
/// costs `O(m² · c)` when few candidates are dependent and `O(m³)` at most.
/// The previous releases restarted the scan at `e_0` for every column and
/// orthogonalized every candidate from scratch (`O(m⁴)` for a tall matrix),
/// in a single pass that left the completed columns orthogonal to the leading
/// ones only to about `1e-11` for a 300 × 31 matrix.
struct BasisCompletion<T> {
    m: usize,
    tol: T,
    /// `1/√2`: a residual norm below it triggers the second pass.
    reorth_threshold: T,
    /// `retired[b]`: `e_b` has been taken or found dependent.
    retired: Vec<bool>,
    /// Residual of the current candidate.
    residual: Vec<T>,
    /// Gram–Schmidt coefficients of the current pass.
    coef: Vec<T>,
}

impl<T: Field + Real> BasisCompletion<T> {
    fn new(m: usize, tol: T) -> Self {
        Self {
            m,
            tol,
            // Without the constant every candidate gets the second pass.
            reorth_threshold: T::from_f64(core::f64::consts::FRAC_1_SQRT_2).unwrap_or(T::one()),
            retired: vec![false; m],
            residual: vec![T::zero(); m],
            coef: Vec::with_capacity(m),
        }
    }

    /// Fills column `col` of the column-major `m × m` buffer `u`, whose
    /// columns `0..col` are final and whose column `col` is zero. The column
    /// stays zero only when no candidate qualifies, which requires non-finite
    /// entries in the columns `0..col`.
    fn fill(&mut self, u: &mut [T], col: usize) {
        let m = self.m;
        let (prev, rest) = u.split_at_mut(col * m);
        let target = &mut rest[..m];
        for b in 0..m {
            if self.retired[b] {
                continue;
            }
            self.retired[b] = true;

            // First pass: r = e_b − Σ_j (u_jᵀ e_b) u_j, with u_jᵀ e_b = u_j[b].
            let r = &mut self.residual;
            for x in r.iter_mut() {
                *x = T::zero();
            }
            r[b] = T::one();
            for u_j in prev.chunks_exact(m) {
                axpy(-u_j[b], u_j, r);
            }

            let mut norm = Real::sqrt(dot(r, r));

            // DGKS criterion (as in ARPACK): when the projection removed more
            // than half of the squared norm (‖r‖ < ‖e_b‖/√2), the first pass
            // may have lost orthogonality, so project once more ("twice is
            // enough"): r −= Σ_j (u_jᵀ r) u_j.
            if !(norm >= self.reorth_threshold) {
                self.coef.clear();
                self.coef
                    .extend(prev.chunks_exact(m).map(|u_j| dot(u_j, r)));
                for (u_j, &c) in prev.chunks_exact(m).zip(&self.coef) {
                    axpy(-c, u_j, r);
                }
                norm = Real::sqrt(dot(r, r));
            }
            if norm > self.tol {
                for (t, &x) in target.iter_mut().zip(r.iter()) {
                    *t = x / norm;
                }
                return;
            }
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
    fn test_svd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 4.0, 1e-10));
        assert!(approx_eq(sigma[1], 3.0, 1e-10));
    }

    #[test]
    fn test_svd_2x2() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
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
    fn test_svd_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9));
            }
        }
    }

    #[test]
    fn test_svd_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
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
    fn test_svd_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let svd = Svd::compute(eye.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for &s in sigma {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_svd_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 1);
        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_svd_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert!(svd.rank(1e-10) == 1);
    }

    #[test]
    fn test_svd_norm() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.norm2(), 4.0, 1e-10));
    }

    #[test]
    fn test_svd_condition() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.condition_number(), 2.0, 1e-10));
    }

    #[test]
    fn test_svd_pseudoinverse() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let pinv = svd.pseudoinverse(1e-10);

        assert!(approx_eq(pinv[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(pinv[(1, 1)], 0.5, 1e-10));
        assert!(approx_eq(pinv[(0, 1)], 0.0, 1e-10));
        assert!(approx_eq(pinv[(1, 0)], 0.0, 1e-10));
    }

    #[test]
    fn test_svd_orthogonality() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let u = svd.u();
        let vt = svd.vt();

        // Verify U^T * U = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += u[(k, i)] * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-9));
            }
        }

        // Verify V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += vt[(i, k)] * vt[(j, k)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-9));
            }
        }
    }

    #[test]
    fn test_svd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
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

#[cfg(test)]
#[path = "bidiagonal_tests.rs"]
mod completion_tests;
