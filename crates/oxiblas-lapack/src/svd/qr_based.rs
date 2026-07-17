//! QR-based SVD algorithm (Golub-Kahan-Reinsch).
//!
//! This implementation uses:
//! 1. Bidiagonal reduction: A = Q * B * P^T
//! 2. Implicit QR iteration on the bidiagonal matrix to compute singular values
//!
//! This is the classical LAPACK approach (DGESVD).

use crate::svd::bidiag_reduce::{BidiagVect, gebrd, orgbr};
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for QR-based SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged {
        /// Number of singular values that did not converge.
        num_unconverged: usize,
    },
    /// Internal error during bidiagonal reduction.
    BidiagError,
}

impl core::fmt::Display for QrSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged { num_unconverged } => {
                write!(
                    f,
                    "SVD did not converge: {} singular values unconverged",
                    num_unconverged
                )
            }
            Self::BidiagError => write!(f, "Bidiagonal reduction failed"),
        }
    }
}

impl std::error::Error for QrSvdError {}

/// QR-based SVD using Golub-Kahan-Reinsch algorithm.
///
/// Computes the economy SVD: A = U·Σ·V^T where:
/// - U is m×k (first k left singular vectors)
/// - Σ is k×k diagonal (singular values, sorted descending)
/// - V^T is k×n (first k right singular vectors transposed)
/// - k = min(m, n)
///
/// This approach is numerically stable and efficient for medium-sized matrices.
#[derive(Debug, Clone)]
pub struct QrSvd<T: Scalar> {
    /// Left singular vectors (m×k matrix where k = min(m, n)).
    u: Mat<T>,
    /// Singular values (sorted in descending order).
    sigma: Vec<T>,
    /// Right singular vectors transposed (k×n matrix where k = min(m, n)).
    vt: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> QrSvd<T> {
    /// Maximum iterations per singular value for bidiagonal QR.
    const MAX_BIDIAG_ITER: usize = 30;

    /// Computes the economy SVD of matrix A using QR-based algorithm.
    ///
    /// Returns U (m×k), Σ (k values), V^T (k×n) where k = min(m, n).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::QrSvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = QrSvd::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, QrSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(QrSvdError::EmptyMatrix);
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

        let k = m.min(n);

        // Step 1: Bidiagonal reduction A = Q * B * P^T
        let factors = gebrd(a).map_err(|_| QrSvdError::BidiagError)?;

        // Extract bidiagonal elements
        let d = factors.d.clone();
        let e = factors.e.clone();

        // Step 2: Compute SVD of the bidiagonal matrix using QR iteration
        let (u_bidiag, sigma, vt_bidiag) = Self::bidiagonal_svd_qr(&d, &e)?;

        // Step 3: Generate Q and P explicitly
        let q = orgbr(&factors, BidiagVect::Q).map_err(|_| QrSvdError::BidiagError)?;
        let p = orgbr(&factors, BidiagVect::P).map_err(|_| QrSvdError::BidiagError)?;

        // Step 4: Combine: U = Q * U_bidiag, V^T = V^T_bidiag * P^T
        // Q: m×k (for tall) or m×m (for wide)
        // U_bidiag: k×k
        // P: n×n (for tall) or k×n (for wide)
        // V^T_bidiag: k×k

        let q_rows = q.nrows();
        let q_cols = q.ncols();
        let p_rows = p.nrows();
        let p_cols = p.ncols();

        // U = Q * U_bidiag
        let mut u = Mat::zeros(q_rows, k);
        for i in 0..q_rows {
            for j in 0..k {
                let mut sum = T::zero();
                for l in 0..k.min(q_cols) {
                    sum = sum + q[(i, l)] * u_bidiag[(l, j)];
                }
                u[(i, j)] = sum;
            }
        }

        // V^T = V^T_bidiag * P^T = (P * V_bidiag)^T
        // First compute P * V_bidiag, then transpose
        let mut vt = Mat::zeros(k, p_cols);
        for i in 0..k {
            for j in 0..p_cols {
                let mut sum = T::zero();
                for l in 0..k.min(p_rows) {
                    // V^T_bidiag[i, l] * P[l, j]^T = V^T_bidiag[i, l] * P^T[j, l]
                    // But P is stored as P, so we want: vt_bidiag[i, l] * p[j, l] (transposed access)
                    // Actually: V^T = (P * V_bidiag)^T but if P is stored in column-major,
                    // we need to be careful.
                    // Simpler: vt[i, j] = sum_l vt_bidiag[i, l] * p^T[l, j] = sum_l vt_bidiag[i, l] * p[j, l]
                    if j < p_rows {
                        sum = sum + vt_bidiag[(i, l)] * p[(j, l)];
                    }
                }
                vt[(i, j)] = sum;
            }
        }

        Ok(Self { u, sigma, vt, m, n })
    }

    /// Compute the SVD of a bidiagonal matrix using implicit-shift QR iteration.
    ///
    /// Returns `(U, sigma, Vᵀ)` where `U` and `Vᵀ` are orthogonal and `sigma`
    /// holds the singular values (sorted descending, all non-negative).
    ///
    /// Non-convergence is reported honestly: if the iteration budget is exhausted
    /// while any super-diagonal of the working bidiagonal is still non-negligible,
    /// the routine returns [`QrSvdError::NotConverged`] carrying the number of
    /// super-diagonals (≈ singular values) that failed to deflate — mirroring the
    /// `INFO > 0` convention of LAPACK `DBDSQR`. It never falls through and returns
    /// a partially-reduced (garbage) factorization as if it had succeeded.
    fn bidiagonal_svd_qr(d: &[T], e: &[T]) -> Result<(Mat<T>, Vec<T>, Mat<T>), QrSvdError> {
        Self::bidiagonal_svd_qr_with_limit(d, e, Self::MAX_BIDIAG_ITER)
    }

    /// Implementation of [`Self::bidiagonal_svd_qr`] with an explicit per-value
    /// iteration budget.
    ///
    /// The number of Golub–Kahan sweeps is bounded by `max_iter_per_value * n`;
    /// the public entry point passes [`Self::MAX_BIDIAG_ITER`]. Factoring the
    /// budget out lets tests drive the non-convergence path deterministically
    /// (e.g. with a budget of zero) without ever weakening the production
    /// tolerance or the production iteration limit.
    fn bidiagonal_svd_qr_with_limit(
        d: &[T],
        e: &[T],
        max_iter_per_value: usize,
    ) -> Result<(Mat<T>, Vec<T>, Mat<T>), QrSvdError> {
        let n = d.len();
        if n == 0 {
            return Ok((Mat::zeros(0, 0), vec![], Mat::zeros(0, 0)));
        }

        let mut d_work: Vec<T> = d.to_vec();
        let mut e_work: Vec<T> = e.to_vec();

        // Initialize U and Vᵀ as identity.
        let mut u = Mat::zeros(n, n);
        let mut vt = Mat::zeros(n, n);
        for i in 0..n {
            u[(i, i)] = T::one();
            vt[(i, i)] = T::one();
        }

        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

        // Implicit-shift Golub–Kahan iteration.
        for _iter in 0..max_iter_per_value.saturating_mul(n) {
            // Deflate the trailing corner: locate the maximal unreduced block
            // spanning diagonal indices [lo, hi] that touches the bottom of the
            // matrix. `hi` is the last diagonal index still coupled by a
            // non-negligible super-diagonal.
            let mut hi = e_work.len();
            while hi > 0 && is_negligible_superdiag(e_work[hi - 1], d_work[hi - 1], d_work[hi], tol)
            {
                hi -= 1;
            }
            if hi == 0 {
                // Every super-diagonal is negligible -> fully converged.
                break;
            }
            let mut lo = hi - 1;
            while lo > 0
                && !is_negligible_superdiag(e_work[lo - 1], d_work[lo - 1], d_work[lo], tol)
            {
                lo -= 1;
            }

            // Apply one Wilkinson-shifted Golub–Kahan step to rows/cols [lo, hi].
            // Isolating the maximal trailing block (rather than always starting at
            // row 0) keeps the Wilkinson shift focused on the sub-problem that is
            // actually converging, which is what makes the shift effective.
            Self::golub_kahan_step(&mut d_work, &mut e_work, &mut u, &mut vt, lo, hi + 1);
        }

        // Honest non-convergence reporting: `NotConverged` was previously dead
        // code (constructed nowhere) and the routine silently returned a
        // not-fully-reduced result. Re-check the working bidiagonal independently
        // of how the loop exited and fail with the real unconverged count.
        let num_unconverged = count_unconverged_superdiags(&d_work, &e_work, tol);
        if num_unconverged > 0 {
            return Err(QrSvdError::NotConverged { num_unconverged });
        }

        // Make all diagonal elements positive by flipping the sign of the
        // corresponding left singular vector (Σ must be non-negative).
        for i in 0..n {
            if d_work[i] < T::zero() {
                d_work[i] = -d_work[i];
                for j in 0..n {
                    u[(j, i)] = -u[(j, i)];
                }
            }
        }

        // Sort singular values in descending order.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            if d_work[b] > d_work[a] {
                core::cmp::Ordering::Greater
            } else if d_work[b] < d_work[a] {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        });

        let mut sigma = vec![T::zero(); n];
        let mut u_sorted = Mat::zeros(n, n);
        let mut vt_sorted = Mat::zeros(n, n);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sigma[new_idx] = d_work[old_idx];
            for j in 0..n {
                u_sorted[(j, new_idx)] = u[(j, old_idx)];
                vt_sorted[(new_idx, j)] = vt[(old_idx, j)];
            }
        }

        Ok((u_sorted, sigma, vt_sorted))
    }

    /// One implicit **Wilkinson-shifted** Golub–Kahan SVD step over the active
    /// block `d[start..end]` / `e[start..end-1]`.
    ///
    /// The step forms the shift μ from the trailing 2×2 of `T = Bᵀ·B`
    /// ([`wilkinson_shift`]) and starts the bulge chase from the shifted vector
    /// `(f, g) = (d[start]² − μ, d[start]·e[start])`. By the implicit-Q theorem the
    /// resulting sequence of Givens rotations is equivalent to one explicitly
    /// shifted symmetric-QR step on `T`, yet it is applied directly to the
    /// bidiagonal `B` (never forming `Bᵀ·B`), which preserves the high relative
    /// accuracy of the small singular values. Setting μ = 0 would recover the
    /// classical zero-shift step; the shift is what restores fast (asymptotically
    /// cubic) convergence on tightly clustered singular values.
    fn golub_kahan_step(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
    ) {
        let n = u.nrows();
        let last = end - 1;

        // Wilkinson shift from the trailing 2×2 of Bᵀ·B, then the shifted start
        // vector (t11 - μ, t12) that seeds the bulge chase.
        let mu = wilkinson_shift(d, e, start, end);
        let mut f = d[start] * d[start] - mu;
        let mut g = d[start] * e[start];

        for k in start..last {
            // Compute Givens rotation to zero g
            let (c, s, r) = givens_rotation(f, g);

            if k > start {
                e[k - 1] = r;
            }

            f = c * d[k] + s * e[k];
            e[k] = -s * d[k] + c * e[k];
            g = s * d[k + 1];
            d[k + 1] = c * d[k + 1];

            // Accumulate V^T rotation
            for j in 0..n {
                let vk = vt[(k, j)];
                let vk1 = vt[(k + 1, j)];
                vt[(k, j)] = c * vk + s * vk1;
                vt[(k + 1, j)] = -s * vk + c * vk1;
            }

            // Compute Givens rotation to zero g
            let (c, s, r) = givens_rotation(f, g);
            d[k] = r;
            f = c * e[k] + s * d[k + 1];
            d[k + 1] = -s * e[k] + c * d[k + 1];

            if k < last - 1 {
                g = s * e[k + 1];
                e[k + 1] = c * e[k + 1];
            }

            // Accumulate U rotation
            for j in 0..n {
                let uk = u[(j, k)];
                let uk1 = u[(j, k + 1)];
                u[(j, k)] = c * uk + s * uk1;
                u[(j, k + 1)] = -s * uk + c * uk1;
            }
        }

        e[last - 1] = f;
    }

    /// Returns the left singular vectors U (m×k matrix).
    pub fn u(&self) -> &Mat<T> {
        &self.u
    }

    /// Returns the singular values in descending order.
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the right singular vectors as V^T (k×n matrix).
    pub fn vt(&self) -> &Mat<T> {
        &self.vt
    }

    /// Returns the original matrix dimensions (m, n).
    pub fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Reconstructs the original matrix A = U·Σ·V^T.
    pub fn reconstruct(&self) -> Mat<T> {
        let mut result = Mat::zeros(self.m, self.n);
        let k = self.sigma.len();

        let u_cols = self.u.ncols();
        let vt_rows = self.vt.nrows();

        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..k.min(u_cols).min(vt_rows) {
                    sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                }
                result[(i, j)] = sum;
            }
        }

        result
    }

    /// Returns the 2-norm condition number κ₂ = σ_max / σ_min.
    ///
    /// For a singular (rank-deficient) matrix σ_min is zero, so the condition
    /// number is mathematically infinite; we return `+∞` (IEEE-754) rather than
    /// panicking or returning a finite sentinel, matching the documented contract
    /// of [`crate::utils::cond`]. An empty spectrum (0×0 map) is reported as `1`.
    ///
    /// The failure case is handled by pattern-matching on the spectrum endpoints
    /// instead of `unwrap`/`expect`, so this production path can never panic.
    pub fn condition_number(&self) -> T {
        match (self.sigma.first(), self.sigma.last()) {
            (Some(&max_sv), Some(&min_sv)) => {
                if min_sv > T::zero() {
                    max_sv / min_sv
                } else {
                    T::infinity()
                }
            }
            _ => T::one(),
        }
    }

    /// Returns the numerical rank with given tolerance.
    pub fn rank(&self, tol: T) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }
}

/// Returns `true` when the super-diagonal entry `e_i` (which couples the diagonal
/// entries `d_i` and `d_ip1`) is negligible relative to its neighbours and may be
/// treated as an exact zero for deflation.
///
/// WHY a plain `<=` comparison: for a NaN super-diagonal `NaN <= x` is `false`, so
/// the entry is reported as *non*-negligible. This is deliberate — it prevents an
/// IEEE-754 NaN from being silently classified as a converged (zero) super-diagonal
/// and dropped from the result. Instead the NaN blocks convergence and is surfaced
/// as an honest [`QrSvdError::NotConverged`] rather than as plausible-looking
/// garbage singular values.
#[inline]
fn is_negligible_superdiag<T: Field + Real>(e_i: T, d_i: T, d_ip1: T, tol: T) -> bool {
    Scalar::abs(e_i) <= tol * (Scalar::abs(d_i) + Scalar::abs(d_ip1))
}

/// Counts the super-diagonal entries that are *not* negligible, i.e. the number of
/// singular values that have not yet deflated. Zero means the bidiagonal has fully
/// converged; a positive count is exactly what LAPACK `DBDSQR` reports in `INFO`.
fn count_unconverged_superdiags<T: Field + Real>(d: &[T], e: &[T], tol: T) -> usize {
    e.iter()
        .enumerate()
        .filter(|&(i, &ev)| !is_negligible_superdiag(ev, d[i], d[i + 1], tol))
        .count()
}

/// Computes the Wilkinson shift μ for one implicit Golub–Kahan SVD step over the
/// active block `d[start..end]` / `e[start..end-1]`.
///
/// μ is the eigenvalue of the trailing 2×2 submatrix of the symmetric tridiagonal
/// `T = Bᵀ·B` (restricted to the block) that lies closest to `T[last,last]`.
///
/// WHY use a shift at all: the zero-shift (μ = 0) step converges only *linearly*
/// on tightly clustered singular values, so a cluster can exhaust the iteration
/// budget — the very failure mode whose (previously silent) mishandling this file
/// also fixes. The Wilkinson shift restores the asymptotically cubic convergence
/// of shifted QR. Because `T = Bᵀ·B` is symmetric positive-semidefinite, its
/// trailing eigenvalue is real and ≥ 0, so μ ≥ 0 and the singular values produced
/// by the step stay real and non-negative.
///
/// The formula is written in the cancellation-avoiding form
/// `μ = t22 − t12² / (δ + sign(δ)·√(δ² + t12²))`, `δ = (t11 − t22)/2`
/// (Golub & Van Loan, *Matrix Computations*, 4th ed., Alg. 8.6.1 and §8.3.5),
/// which is numerically safer than `t22 + δ − sign(δ)·√(δ² + t12²)` when the two
/// trailing eigenvalues are close.
///
/// Precondition: the block has at least two diagonal entries (`end >= start + 2`),
/// which the deflation logic in [`Self::bidiagonal_svd_qr_with_limit`] guarantees.
fn wilkinson_shift<T: Field + Real>(d: &[T], e: &[T], start: usize, end: usize) -> T {
    let last = end - 1;

    // Trailing 2×2 of T = Bᵀ·B over the block:
    //   T[i,i]   = d[i]² + e[i-1]²   (a super-diagonal index below `start` lies
    //                                 outside the block and contributes 0)
    //   T[i,i+1] = d[i] · e[i]
    let e_last = e[last - 1]; // couples d[last-1] and d[last]
    let e_above = if last - 1 > start {
        e[last - 2]
    } else {
        T::zero()
    };
    let t22 = d[last] * d[last] + e_last * e_last;
    let t11 = d[last - 1] * d[last - 1] + e_above * e_above;
    let t12 = d[last - 1] * e_last;

    // Diagonal (or negligibly-coupled) trailing 2×2: the closest eigenvalue is t22.
    if Scalar::abs(t12) <= <T as Scalar>::epsilon() * (Scalar::abs(t11) + Scalar::abs(t22)) {
        return t22;
    }

    let two = T::one() + T::one();
    let delta = (t11 - t22) / two;
    // sign(0) is taken as +1 so the denominator never cancels to zero here.
    let sign_delta = if delta >= T::zero() {
        T::one()
    } else {
        -T::one()
    };
    let denom = delta + sign_delta * Real::sqrt(delta * delta + t12 * t12);
    if Scalar::abs(denom) <= <T as Scalar>::min_positive() {
        // Degenerate denominator (only reachable under extreme underflow):
        // fall back to the unshifted trailing eigenvalue estimate.
        t22
    } else {
        t22 - t12 * t12 / denom
    }
}

/// Compute Givens rotation parameters.
/// Returns (c, s, r) such that [c s; -s c] * [f; g] = [r; 0]
fn givens_rotation<T: Field + Real>(f: T, g: T) -> (T, T, T) {
    let eps = <T as Scalar>::epsilon();

    if Scalar::abs(g) < eps {
        (T::one(), T::zero(), f)
    } else if Scalar::abs(f) < eps {
        (
            T::zero(),
            if g >= T::zero() { T::one() } else { -T::one() },
            Scalar::abs(g),
        )
    } else {
        let h = Real::sqrt(f * f + g * g);
        let c = Scalar::abs(f) / h;
        let s = g / h * (if f >= T::zero() { T::one() } else { -T::one() });
        let r = if f >= T::zero() { h } else { -h };
        (c, s, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_qr_svd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        // Singular values should be 4 and 3 (sorted descending)
        assert!(approx_eq(sigma[0], 4.0, 1e-8), "sigma[0] = {}", sigma[0]);
        assert!(approx_eq(sigma[1], 3.0, 1e-8), "sigma[1] = {}", sigma[1]);
    }

    #[test]
    fn test_qr_svd_identity() {
        let a: Mat<f64> = Mat::eye(3);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for (i, &s) in sigma.iter().enumerate() {
            assert!(approx_eq(s, 1.0, 1e-8), "sigma[{}] = {}", i, s);
        }
    }

    #[test]
    fn test_qr_svd_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_svd_singular_values_descending() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for i in 0..sigma.len() - 1 {
            assert!(
                sigma[i] >= sigma[i + 1],
                "Singular values not descending: sigma[{}]={} < sigma[{}]={}",
                i,
                sigma[i],
                i + 1,
                sigma[i + 1]
            );
        }
    }

    #[test]
    fn test_qr_svd_u_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let u = svd.u();

        // Check U^T * U = I (for thin SVD, this should be k×k identity)
        let k = u.ncols();
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..u.nrows() {
                    dot += u[(l, i)] * u[(l, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-8),
                    "U^T*U not identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_svd_vt_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let vt = svd.vt();

        // Check V^T * V = I (for thin SVD, this should be k×k identity)
        let k = vt.nrows();
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..vt.ncols() {
                    dot += vt[(i, l)] * vt[(j, l)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-8),
                    "V^T*V not identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_svd_condition_number() {
        // Well-conditioned matrix
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 1.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let cond = svd.condition_number();

        assert!(approx_eq(cond, 2.0, 1e-8), "cond = {}", cond);
    }

    #[test]
    fn test_qr_svd_rank() {
        // Rank-1 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let r = svd.rank(1e-8);

        assert_eq!(r, 1, "rank = {}", r);
    }

    #[test]
    fn test_qr_svd_1x1() {
        let a = Mat::from_rows(&[&[-5.0f64]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_qr_svd_wide_matrix() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert!(
            approx_eq(sigma[0], 30.0f64.sqrt(), 1e-8),
            "sigma[0] = {}",
            sigma[0]
        );
    }

    #[test]
    fn test_qr_svd_square_matrix() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    /// Reconstructs a dense matrix from a bidiagonal SVD factorization
    /// `B = U · diag(sigma) · Vᵀ` for verification.
    fn reconstruct_bidiag(u: &Mat<f64>, sigma: &[f64], vt: &Mat<f64>) -> Mat<f64> {
        let n = sigma.len();
        let mut b = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..n {
                    s += u[(i, l)] * sigma[l] * vt[(l, j)];
                }
                b[(i, j)] = s;
            }
        }
        b
    }

    /// Regression for finding #3: the step now applies a real Wilkinson shift.
    /// A bidiagonal with a tight cluster of singular values near 2 is exactly the
    /// case where the previous zero-shift step converges only linearly. With the
    /// shift it converges within the standard budget; we verify the factorization
    /// reconstructs B and that Σ is a valid (non-negative, descending) spectrum.
    #[test]
    fn test_qr_svd_bidiagonal_clustered_converges() {
        let d = vec![2.0f64, 2.0, 2.0, 2.0, 2.0];
        let e = vec![1e-7f64, 1e-7, 1e-7, 1e-7];

        let (u, sigma, vt) = QrSvd::<f64>::bidiagonal_svd_qr(&d, &e)
            .expect("clustered bidiagonal must converge with the Wilkinson shift");

        // Build the original bidiagonal B = diag(d) + superdiag(e).
        let n = d.len();
        let mut b = Mat::zeros(n, n);
        for i in 0..n {
            b[(i, i)] = d[i];
        }
        for i in 0..n - 1 {
            b[(i, i + 1)] = e[i];
        }

        let recon = reconstruct_bidiag(&u, &sigma, &vt);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(recon[(i, j)], b[(i, j)], 1e-10),
                    "reconstruct mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon[(i, j)],
                    b[(i, j)]
                );
            }
        }

        for w in sigma.windows(2) {
            assert!(w[0] >= w[1], "sigma not descending: {:?}", sigma);
        }
        for &s in &sigma {
            assert!(s >= 0.0, "negative singular value: {}", s);
            assert!(
                (s - 2.0).abs() < 1e-3,
                "singular value not near the cluster: {}",
                s
            );
        }
    }

    /// Regression for finding #1: `NotConverged` used to be dead code. With a zero
    /// iteration budget on a genuinely non-diagonal bidiagonal, the routine must
    /// report `NotConverged` with the exact count of non-negligible super-diagonals
    /// instead of silently returning a partially-reduced (garbage) factorization.
    #[test]
    fn test_qr_svd_not_converged_reported() {
        let d = vec![2.0f64, 2.0, 2.0, 2.0];
        let e = vec![1.0f64, 1.0, 1.0];

        let result = QrSvd::<f64>::bidiagonal_svd_qr_with_limit(&d, &e, 0);
        match result {
            Err(QrSvdError::NotConverged { num_unconverged }) => {
                assert_eq!(
                    num_unconverged, 3,
                    "expected all 3 super-diagonals reported unconverged"
                );
            }
            other => panic!("expected NotConverged, got {:?}", other),
        }

        // An already-diagonal bidiagonal has nothing left to reduce, so even a zero
        // budget must NOT produce a false NotConverged.
        let d_diag = vec![3.0f64, 2.0, 1.0];
        let e_diag = vec![0.0f64, 0.0];
        let (_, sigma, _) = QrSvd::<f64>::bidiagonal_svd_qr_with_limit(&d_diag, &e_diag, 0)
            .expect("already-diagonal input must not report NotConverged");
        assert!(approx_eq(sigma[0], 3.0, 1e-12));
        assert!(approx_eq(sigma[1], 2.0, 1e-12));
        assert!(approx_eq(sigma[2], 1.0, 1e-12));
    }

    /// Regression for finding #2: `condition_number()` no longer uses `expect()`,
    /// and a singular matrix (exact zero smallest singular value) yields `+∞`
    /// instead of panicking or returning a finite sentinel.
    #[test]
    fn test_qr_svd_condition_number_singular() {
        // Diagonal rank-deficient matrix -> the bidiagonal is already diagonal with
        // an exact zero, so the smallest singular value is exactly 0.
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 0.0]]);
        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let cond = svd.condition_number();
        assert!(
            cond.is_infinite() && cond > 0.0,
            "condition number of a singular matrix must be +inf, got {}",
            cond
        );

        // The all-zero matrix is likewise singular; must not panic.
        let z = Mat::from_rows(&[&[0.0f64, 0.0], &[0.0, 0.0]]);
        let svd_z = QrSvd::compute(z.as_ref()).unwrap();
        assert!(
            svd_z.condition_number().is_infinite(),
            "condition number of the zero matrix must be +inf"
        );
    }
}
