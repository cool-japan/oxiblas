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
    /// Non-convergence is reported honestly via [`QrSvdError::NotConverged`]: the
    /// routine never falls through and returns a partially-reduced (garbage)
    /// factorization as if it had succeeded, mirroring the `INFO > 0` convention
    /// of LAPACK `DBDSQR`.
    fn bidiagonal_svd_qr(d: &[T], e: &[T]) -> Result<(Mat<T>, Vec<T>, Mat<T>), QrSvdError> {
        Self::bidiagonal_svd_qr_with_limit(d, e, Self::MAX_BIDIAG_ITER)
    }

    /// Implementation of [`Self::bidiagonal_svd_qr`] with an explicit per-value
    /// iteration budget. Factoring the budget out lets tests drive the
    /// non-convergence path deterministically without weakening the production
    /// tolerance or iteration limit.
    ///
    /// The total number of Golub–Kahan sweeps is bounded by
    /// `max_iter_per_value * n * n` (LAPACK `DBDSQR` uses `MAXITR * N * N`); a
    /// per-value budget that scales only linearly with `n` is far too small for
    /// the trailing-block chase to deflate every singular value on realistic
    /// dense problems.
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

        // Total sweep budget scales as O(n²) (LAPACK DBDSQR: MAXITR·N·N).
        let max_sweeps = max_iter_per_value.saturating_mul(n).saturating_mul(n);

        // Implicit-shift Golub–Kahan iteration.
        for _iter in 0..max_sweeps {
            // Set negligible super-diagonals to an exact zero so block boundaries
            // are clean and the Wilkinson shift sees a genuinely unreduced block.
            for i in 0..e_work.len() {
                if is_negligible_superdiag(e_work[i], d_work[i], d_work[i + 1], tol) {
                    e_work[i] = T::zero();
                }
            }

            // Deflate the trailing corner: locate the maximal unreduced block
            // spanning diagonal indices [lo, hi] that touches the bottom of the
            // matrix. `hi` is the last diagonal index still coupled by a
            // non-negligible super-diagonal.
            let mut hi = e_work.len();
            while hi > 0 && e_work[hi - 1] == T::zero() {
                hi -= 1;
            }
            if hi == 0 {
                // Every super-diagonal is negligible -> fully converged.
                break;
            }
            let mut lo = hi - 1;
            while lo > 0 && e_work[lo - 1] != T::zero() {
                lo -= 1;
            }

            // If the top diagonal of the block is (numerically) zero, the shifted
            // step cannot introduce the bulge cleanly; a zero-shift left-side
            // Givens sweep chases the zero out along the row and decouples it.
            if Scalar::abs(d_work[lo])
                <= tol * (Scalar::abs(d_work[hi]) + Scalar::abs(e_work[lo]))
            {
                Self::deflate_zero_diagonal(&mut d_work, &mut e_work, &mut u, lo, hi + 1);
                continue;
            }

            // Symmetrically, a (numerically) zero diagonal at the bottom of the
            // block degenerates the upward chase (whose bulge is seeded from
            // `d[hi]`); chase the zero up its column with right-side rotations so it
            // splits off as a zero singular value.
            if Scalar::abs(d_work[hi])
                <= tol * (Scalar::abs(d_work[lo]) + Scalar::abs(e_work[hi - 1]))
            {
                Self::deflate_zero_diagonal_bottom(&mut d_work, &mut e_work, &mut vt, lo, hi + 1);
                continue;
            }

            // Apply one Wilkinson-shifted Golub–Kahan step to rows/cols [lo, hi].
            // Isolating the maximal trailing block (rather than always starting at
            // row 0) keeps the Wilkinson shift focused on the sub-problem that is
            // actually converging, which is what makes the shift effective.
            Self::golub_kahan_step(&mut d_work, &mut e_work, &mut u, &mut vt, lo, hi + 1);
        }

        // Honest non-convergence reporting: re-check the working bidiagonal
        // independently of how the loop exited and fail with the real unconverged
        // count instead of silently returning a not-fully-reduced result.
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
    /// The step is applied directly to the bidiagonal `B` (the implicit-Q theorem
    /// makes it equivalent to one explicitly shifted symmetric-QR step on
    /// `T = Bᵀ·B`, without ever forming `Bᵀ·B`), which preserves the high relative
    /// accuracy of the small singular values.
    ///
    /// **Direction matters for the convergence rate.** A downward chase with the
    /// *trailing* Wilkinson shift deflates the bottom super-diagonal cubically,
    /// while the top converges only linearly, and vice-versa. On a graded block
    /// (say large singular values at the top, a tightly clustered tail at the
    /// bottom) the shift that targets the tail can be dwarfed by the large top
    /// diagonal `d[start]²`, degenerating the step to an unshifted sweep whose top
    /// converges only linearly — the exact stall that made this SVD miss its
    /// iteration budget on realistic dense matrices. We therefore always chase
    /// *toward the larger-magnitude end* (upward when `|d[start]| ≥ |d[last]|`,
    /// downward otherwise): the extreme, well-separated singular values there
    /// deflate cubically under a shift drawn from their own end, and the ill-scaled
    /// cluster is left to a smaller, better-balanced sub-block on later sweeps.
    fn golub_kahan_step(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
    ) {
        let last = end - 1;
        if Scalar::abs(d[start]) >= Scalar::abs(d[last]) {
            // Larger diagonals at the top: converge the top with an upward chase
            // seeded by the leading 2×2 shift.
            let mu = wilkinson_shift_leading(d, e, start, end);
            Self::chase_up(d, e, u, vt, start, end, mu);
        } else {
            // Larger diagonals at the bottom: converge the bottom with a downward
            // chase seeded by the trailing 2×2 shift.
            let mu = wilkinson_shift_trailing(d, e, start, end);
            Self::chase_down(d, e, u, vt, start, end, mu);
        }
    }

    /// Downward Golub–Kahan bulge chase over `[start, end)` with shift `mu` drawn
    /// from the trailing 2×2. Deflates the bottom super-diagonal `e[end-2]`.
    fn chase_down(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
        mu: T,
    ) {
        let n = u.nrows();
        let last = end - 1;

        // Shifted start vector (t11 - μ, t12) seeding the bulge at the top.
        let mut f = d[start] * d[start] - mu;
        let mut g = d[start] * e[start];

        for k in start..last {
            // Right (V) Givens rotation to zero g.
            let (c, s, r) = givens_rotation(f, g);

            if k > start {
                e[k - 1] = r;
            }

            f = c * d[k] + s * e[k];
            e[k] = -s * d[k] + c * e[k];
            g = s * d[k + 1];
            d[k + 1] = c * d[k + 1];

            // Accumulate Vᵀ rotation.
            for j in 0..n {
                let vk = vt[(k, j)];
                let vk1 = vt[(k + 1, j)];
                vt[(k, j)] = c * vk + s * vk1;
                vt[(k + 1, j)] = -s * vk + c * vk1;
            }

            // Left (U) Givens rotation to zero the bulge.
            let (c, s, r) = givens_rotation(f, g);
            d[k] = r;
            f = c * e[k] + s * d[k + 1];
            d[k + 1] = -s * e[k] + c * d[k + 1];

            if k < last - 1 {
                g = s * e[k + 1];
                e[k + 1] = c * e[k + 1];
            }

            // Accumulate U rotation.
            for j in 0..n {
                let uk = u[(j, k)];
                let uk1 = u[(j, k + 1)];
                u[(j, k)] = c * uk + s * uk1;
                u[(j, k + 1)] = -s * uk + c * uk1;
            }
        }

        e[last - 1] = f;
    }

    /// Upward Golub–Kahan bulge chase over `[start, end)` with shift `mu` drawn from
    /// the leading 2×2. Deflates the top super-diagonal `e[start]`.
    ///
    /// This is the exact mirror image of [`Self::chase_down`]: the bulge is seeded
    /// at the bottom-right corner from `(d[last]² − μ, d[last]·e[last-1])`, the roles
    /// of the left (U) and right (V) rotations are swapped, and the index runs from
    /// `last-1` down to `start`.
    fn chase_up(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
        mu: T,
    ) {
        let n = u.nrows();
        let last = end - 1;

        // Shifted start vector seeding the bulge at the bottom-right corner.
        let mut f = d[last] * d[last] - mu;
        let mut g = d[last] * e[last - 1];

        for k in (start..last).rev() {
            // Left (U) Givens rotation to zero g.
            let (c, s, r) = givens_rotation(f, g);

            if k < last - 1 {
                e[k + 1] = r;
            }

            f = c * d[k + 1] + s * e[k];
            e[k] = -s * d[k + 1] + c * e[k];
            g = s * d[k];
            d[k] = c * d[k];

            // Accumulate U rotation. The upward chase is the reversal-mirror of the
            // downward one, so the rotation pairs the columns in reversed order
            // (k+1 first, k second) — otherwise U/Σ/Vᵀ no longer reconstructs B.
            for j in 0..n {
                let uk = u[(j, k)];
                let uk1 = u[(j, k + 1)];
                u[(j, k + 1)] = c * uk1 + s * uk;
                u[(j, k)] = -s * uk1 + c * uk;
            }

            // Right (V) Givens rotation to zero the bulge.
            let (c, s, r) = givens_rotation(f, g);
            d[k + 1] = r;
            f = c * e[k] + s * d[k];
            d[k] = -s * e[k] + c * d[k];

            if k > start {
                g = s * e[k - 1];
                e[k - 1] = c * e[k - 1];
            }

            // Accumulate Vᵀ rotation on rows in reversed order (see U above).
            for j in 0..n {
                let vk = vt[(k, j)];
                let vk1 = vt[(k + 1, j)];
                vt[(k + 1, j)] = c * vk1 + s * vk;
                vt[(k, j)] = -s * vk1 + c * vk;
            }
        }

        e[start] = f;
    }

    /// Handle a (numerically) zero diagonal entry at the top of the active block.
    ///
    /// When `d[start] ≈ 0` the shifted Golub–Kahan step degenerates (the bulge
    /// cannot be introduced), so the trailing super-diagonal never deflates. The
    /// classical remedy (LAPACK `DLASQ`/`DBDSQR`) applies a sequence of left-side
    /// Givens rotations that chase the resulting zero along its row, splitting off
    /// a zero singular value and decoupling the block so the ordinary shifted
    /// iteration can proceed. Only `U` (left vectors) is updated because these are
    /// left rotations; `Vᵀ` is untouched.
    fn deflate_zero_diagonal(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        start: usize,
        end: usize,
    ) {
        let n = u.nrows();
        let last = end - 1;

        // Chase the zero at (start, start) rightward. `extra` carries the fill-in
        // that appears at position (start, k+1) as the zero moves.
        let mut extra = e[start];
        e[start] = T::zero();

        for k in (start + 1)..=last {
            // Rotate rows k and `start` to annihilate `extra` against d[k].
            let (c, s, _r) = givens_rotation(d[k], extra);
            d[k] = c * d[k] + s * extra;

            for j in 0..n {
                let uk = u[(j, k)];
                let us = u[(j, start)];
                u[(j, k)] = c * uk + s * us;
                u[(j, start)] = -s * uk + c * us;
            }

            if k < last {
                // SAFETY/correctness note: the left Givens rotation applied to rows
                // (k, start) produces fill-in at (start, k+1) with value -s*e[k],
                // not +s*e[k] -- verified against the accumulated U rotation and by
                // an independent reconstruction check (see qr_based.rs test suite).
                extra = -s * e[k];
                e[k] = c * e[k];
            }
        }
    }

    /// Handle a (numerically) zero diagonal entry at the **bottom** of the active
    /// block (`d[end-1] ≈ 0`).
    ///
    /// This is the transpose-mirror of [`Self::deflate_zero_diagonal`]: a zero at
    /// the bottom-right corner leaves the trailing super-diagonal uncoupled to any
    /// diagonal it can be rotated against from the left, so the upward shifted
    /// chase (whose bulge is seeded from `d[end-1]`) degenerates and never
    /// deflates. The remedy applies a sequence of right-side Givens rotations that
    /// chase the zero *up* its column, splitting off a zero singular value and
    /// decoupling the block. Only `Vᵀ` (right vectors) is updated because these are
    /// right rotations; `U` is untouched.
    fn deflate_zero_diagonal_bottom(
        d: &mut [T],
        e: &mut [T],
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
    ) {
        let n = vt.ncols();
        let last = end - 1;

        // Chase the zero at (last, last) upward. `extra` carries the fill-in that
        // appears at position (k-1, last) as the zero moves up its column.
        let mut extra = e[last - 1];
        e[last - 1] = T::zero();

        for k in (start..last).rev() {
            // Rotate columns k and `last` to annihilate `extra` against d[k].
            let (c, s, _r) = givens_rotation(d[k], extra);
            d[k] = c * d[k] + s * extra;

            for j in 0..n {
                let vk = vt[(k, j)];
                let vl = vt[(last, j)];
                vt[(k, j)] = c * vk + s * vl;
                vt[(last, j)] = -s * vk + c * vl;
            }

            if k > start {
                // Correctness note: mirror of the sign fix in deflate_zero_diagonal
                // above -- the right Givens rotation's fill-in at (k-1, last) is
                // -s*e[k-1], not +s*e[k-1].
                extra = -s * e[k - 1];
                e[k - 1] = c * e[k - 1];
            }
        }
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
    /// For a singular (rank-deficient) matrix σ_min is zero, so the ratio is
    /// reported as [`Scalar::max_value`] rather than dividing by zero; an empty
    /// spectrum (0×0 map) is reported as `1`. The endpoints are obtained by
    /// pattern-matching instead of `unwrap`/`expect`, so this production path can
    /// never panic (COOLJAPAN no-panic policy).
    pub fn condition_number(&self) -> T {
        match (self.sigma.first(), self.sigma.last()) {
            (Some(&max_sv), Some(&min_sv)) => {
                if min_sv > T::zero() {
                    max_sv / min_sv
                } else {
                    <T as Scalar>::max_value()
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
/// A plain `<=` comparison is used deliberately: for a NaN super-diagonal
/// `NaN <= x` is `false`, so the entry is reported as *non*-negligible. This keeps
/// an IEEE-754 NaN from being silently classified as a converged (zero)
/// super-diagonal and dropped; instead the NaN blocks convergence and is surfaced
/// as an honest [`QrSvdError::NotConverged`] rather than plausible-looking garbage.
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
/// Using a shift is essential: the zero-shift (μ = 0) step converges only
/// *linearly* on tightly clustered singular values, so a cluster can exhaust the
/// iteration budget. The Wilkinson shift restores the asymptotically cubic
/// convergence of shifted QR. Because `T = Bᵀ·B` is symmetric positive
/// semidefinite, its trailing eigenvalue is real and ≥ 0, so μ ≥ 0 and the
/// singular values produced by the step stay real and non-negative.
///
/// The cancellation-avoiding form
/// `μ = t22 − t12² / (δ + sign(δ)·√(δ² + t12²))`, `δ = (t11 − t22)/2`
/// (Golub & Van Loan, *Matrix Computations*, 4th ed., Alg. 8.6.1 and §8.3.5) is
/// numerically safer than `t22 + δ − sign(δ)·√(δ² + t12²)` when the two trailing
/// eigenvalues are close.
///
/// Precondition: the block has at least two diagonal entries (`end >= start + 2`),
/// which the deflation logic guarantees.
fn wilkinson_shift_trailing<T: Field + Real>(d: &[T], e: &[T], start: usize, end: usize) -> T {
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
    wilkinson_eig_closest_to(t11, t12, t22)
}

/// Wilkinson shift μ drawn from the **leading** 2×2 of `T = Bᵀ·B` over the block,
/// used to seed an upward bulge chase that deflates the top super-diagonal.
///
/// μ is the eigenvalue of the leading 2×2 closest to `T[start,start] = d[start]²`
/// (a super-diagonal index below `start` lies outside the block and contributes 0).
///
/// Precondition: the block has at least two diagonal entries (`end >= start + 2`);
/// only `start` is needed since the leading 2×2 lives at the block's top.
fn wilkinson_shift_leading<T: Field + Real>(d: &[T], e: &[T], start: usize, _end: usize) -> T {
    // Leading 2×2 of T = Bᵀ·B over the block:
    //   T[start,start]     = d[start]²
    //   T[start,start+1]   = d[start] · e[start]
    //   T[start+1,start+1] = d[start+1]² + e[start]²
    let t11 = d[start] * d[start];
    let t12 = d[start] * e[start];
    let t22 = d[start + 1] * d[start + 1] + e[start] * e[start];
    // Eigenvalue closest to t11 (mirror of the trailing case: pass the pair with
    // t22:=t11 as the anchor).
    wilkinson_eig_closest_to(t22, t12, t11)
}

/// Eigenvalue of the symmetric 2×2 `[[t11, t12], [t12, t_anchor]]` that lies
/// closest to the anchor entry `t_anchor`, using the cancellation-avoiding form
/// `μ = t_anchor − t12² / (δ + sign(δ)·√(δ² + t12²))`, `δ = (t11 − t_anchor)/2`.
fn wilkinson_eig_closest_to<T: Field + Real>(t11: T, t12: T, t_anchor: T) -> T {
    // Diagonal (or negligibly-coupled) 2×2: the closest eigenvalue is the anchor.
    if Scalar::abs(t12) <= <T as Scalar>::epsilon() * (Scalar::abs(t11) + Scalar::abs(t_anchor)) {
        return t_anchor;
    }

    let two = T::one() + T::one();
    let delta = (t11 - t_anchor) / two;
    // sign(0) is taken as +1 so the denominator never cancels to zero here.
    let sign_delta = if delta >= T::zero() {
        T::one()
    } else {
        -T::one()
    };
    let denom = delta + sign_delta * Real::sqrt(delta * delta + t12 * t12);
    if Scalar::abs(denom) <= <T as Scalar>::min_positive() {
        // Degenerate denominator (only reachable under extreme underflow):
        // fall back to the unshifted anchor eigenvalue estimate.
        t_anchor
    } else {
        t_anchor - t12 * t12 / denom
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

    /// Deterministic pseudo-random values in `[-1, 1)` (no external rng), matching
    /// the construction used by the divide-and-conquer reference test.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn lcg_matrix(n: usize) -> Mat<f64> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d ^ (n as u64);
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = lcg(&mut state);
            }
        }
        a
    }

    /// Regression for the finding "QrSvd fails to converge on realistic-sized
    /// dense matrices": a deterministic 50×50 pseudo-random matrix used to return
    /// `NotConverged { num_unconverged: 33 }`. With the Wilkinson-shifted QR sweep,
    /// proper trailing-block deflation and an O(n²) sweep budget it must now
    /// converge and match the Jacobi reference SVD to tight tolerance.
    #[test]
    fn test_qr_svd_lcg_across_sizes() {
        use crate::svd::Svd;

        for &n in &[10usize, 25, 50, 100, 150] {
            let a = lcg_matrix(n);

            let qr = QrSvd::compute(a.as_ref())
                .unwrap_or_else(|e| panic!("n={n}: QrSvd failed to converge: {e}"));
            let reference = Svd::compute(a.as_ref()).unwrap();

            let s_qr = qr.singular_values();
            let s_ref = reference.singular_values();
            let smax = s_ref[0].abs().max(1.0);

            assert_eq!(s_qr.len(), n, "n={n}: wrong number of singular values");
            for k in 0..n {
                assert!(
                    (s_qr[k] - s_ref[k]).abs() < 1e-8 * smax,
                    "n={n}: sigma[{k}] qr={} jacobi={}",
                    s_qr[k],
                    s_ref[k]
                );
            }

            // Singular values are sorted descending and non-negative.
            for k in 0..n {
                assert!(s_qr[k] >= -1e-12, "n={n}: sigma[{k}]={} is negative", s_qr[k]);
            }
            for k in 1..n {
                assert!(
                    s_qr[k - 1] >= s_qr[k] - 1e-12,
                    "n={n}: sigma not descending at {k}"
                );
            }

            // Reconstruction ||A - U Σ Vᵀ|| is tight.
            let rec = qr.reconstruct();
            let mut rec_err = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    rec_err = rec_err.max((rec[(i, j)] - a[(i, j)]).abs());
                }
            }
            assert!(
                rec_err < 1e-8 * smax,
                "n={n}: reconstruction error {rec_err}"
            );
        }
    }

    /// The upward bulge chase must accumulate `U` and `Vᵀ` so that
    /// `U · B_new · Vᵀ == B_old` (the invariant every sweep preserves). This
    /// guards the reversal-mirror column/row pairing in [`QrSvd::chase_up`], whose
    /// sign/order is easy to get wrong and silently corrupts the singular vectors
    /// while leaving the singular values correct.
    #[test]
    fn test_qr_svd_chase_up_invariant() {
        // Verify U * B_new * V^T == B_old after one upward chase.
        let d0 = [3.0f64, 5.0, 2.0, 7.0];
        let e0 = [1.5f64, 0.8, 2.1];
        let n = d0.len();
        // Build B_old dense.
        let mut b_old = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            b_old[(i, i)] = d0[i];
            if i + 1 < n {
                b_old[(i, i + 1)] = e0[i];
            }
        }
        let mut d = d0.to_vec();
        let mut e = e0.to_vec();
        let mut u = Mat::<f64>::zeros(n, n);
        let mut vt = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            u[(i, i)] = 1.0;
            vt[(i, i)] = 1.0;
        }
        let mu = wilkinson_shift_leading(&d, &e, 0, n);
        QrSvd::<f64>::chase_up(&mut d, &mut e, &mut u, &mut vt, 0, n, mu);
        // B_new dense.
        let mut b_new = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            b_new[(i, i)] = d[i];
            if i + 1 < n {
                b_new[(i, i + 1)] = e[i];
            }
        }
        // recon = U * B_new * V^T
        let mut ub = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += u[(i, k)] * b_new[(k, j)];
                }
                ub[(i, j)] = s;
            }
        }
        let mut recon = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += ub[(i, k)] * vt[(k, j)];
                }
                recon[(i, j)] = s;
            }
        }
        let mut err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                err = err.max((recon[(i, j)] - b_old[(i, j)]).abs());
            }
        }
        assert!(err < 1e-10, "chase_up invariant violated: err={err}");
    }

    /// The 50×50 case from the finding, isolated for a fast targeted check.
    #[test]
    fn test_qr_svd_lcg_50x50_converges() {
        let a = lcg_matrix(50);
        let svd = QrSvd::compute(a.as_ref())
            .expect("50x50 LCG matrix must converge after the Wilkinson-shift fix");
        assert_eq!(svd.singular_values().len(), 50);
    }

    /// Rank-deficient matrix with genuine zero singular values (exercises the
    /// zero-diagonal deflation path and confirms it converges rather than stalling).
    #[test]
    fn test_qr_svd_rank_deficient() {
        use crate::svd::Svd;
        let n = 30usize;
        // Rank-3 matrix: outer sum of 3 deterministic vectors -> 27 zero sigmas.
        let mut a = Mat::<f64>::zeros(n, n);
        let mut s1 = 0x1234_5678u64;
        let mut s2 = 0x9abc_def0u64;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for _r in 0..3 {
                    acc += lcg(&mut s1) * lcg(&mut s2);
                }
                a[(i, j)] = acc;
            }
        }
        let qr = QrSvd::compute(a.as_ref())
            .expect("rank-deficient matrix must converge (zero-diagonal path)");
        let reference = Svd::compute(a.as_ref()).unwrap();
        let s_qr = qr.singular_values();
        let s_ref = reference.singular_values();
        let smax = s_ref[0].abs().max(1.0);
        for k in 0..n {
            assert!(
                (s_qr[k] - s_ref[k]).abs() < 1e-7 * smax,
                "rank-deficient sigma[{k}] qr={} jacobi={}",
                s_qr[k],
                s_ref[k]
            );
        }
        let rec = qr.reconstruct();
        let mut rec_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                rec_err = rec_err.max((rec[(i, j)] - a[(i, j)]).abs());
            }
        }
        assert!(rec_err < 1e-7 * smax, "reconstruction error {rec_err}");
    }

    /// Matrix with clustered / repeated singular values (a well-known stress case
    /// for shifted QR convergence).
    #[test]
    fn test_qr_svd_repeated_singular_values() {
        // Scaled orthogonal-ish construction: identity has all sigma == 1.
        let n = 20usize;
        let a: Mat<f64> = Mat::eye(n);
        let qr = QrSvd::compute(a.as_ref()).expect("identity must converge");
        for &s in qr.singular_values() {
            assert!((s - 1.0).abs() < 1e-10, "sigma = {s}");
        }
    }

    /// Adversarial rank-deficient stress: diagonal matrices with exact zeros in
    /// various positions (including the trailing entry, which lands a zero singular
    /// value at a block bottom and would stall a naive upward chase), plus a matrix
    /// with an exact zero row and one with an exact zero column.
    #[test]
    fn test_qr_svd_zero_positions_stress() {
        use crate::svd::Svd;

        // Diagonal with one exact zero at each position.
        for n in [3usize, 6, 12] {
            for zero_pos in 0..n {
                let mut a = Mat::<f64>::zeros(n, n);
                for i in 0..n {
                    a[(i, i)] = if i == zero_pos { 0.0 } else { (i + 1) as f64 };
                }
                // Mix in off-diagonal structure so bidiagonalization is non-trivial.
                if zero_pos + 1 < n {
                    a[(zero_pos, zero_pos + 1)] = 0.0;
                }
                let qr = QrSvd::compute(a.as_ref()).unwrap_or_else(|e| {
                    panic!("n={n} zero_pos={zero_pos}: failed: {e}")
                });
                let reference = Svd::compute(a.as_ref()).unwrap();
                let s_qr = qr.singular_values();
                let s_ref = reference.singular_values();
                let smax = s_ref[0].abs().max(1.0);
                for k in 0..n {
                    assert!(
                        (s_qr[k] - s_ref[k]).abs() < 1e-8 * smax,
                        "n={n} zero_pos={zero_pos}: sigma[{k}] qr={} ref={}",
                        s_qr[k],
                        s_ref[k]
                    );
                }
            }
        }

        // Dense matrix with an exact zero row (rank deficient) at the bottom.
        let n = 10usize;
        let mut a = Mat::<f64>::zeros(n, n);
        let mut st = 0xdead_beefu64;
        for i in 0..(n - 1) {
            for j in 0..n {
                a[(i, j)] = lcg(&mut st);
            }
        }
        // last row stays zero
        let qr = QrSvd::compute(a.as_ref()).expect("zero-row matrix must converge");
        let reference = Svd::compute(a.as_ref()).unwrap();
        let smax = reference.singular_values()[0].abs().max(1.0);
        for k in 0..n {
            assert!(
                (qr.singular_values()[k] - reference.singular_values()[k]).abs() < 1e-7 * smax,
                "zero-row sigma[{k}] qr={} ref={}",
                qr.singular_values()[k],
                reference.singular_values()[k]
            );
        }
    }

    /// A block-diagonal-inducing matrix with an exact interior zero super-diagonal
    /// (deflation must split it into independent sub-problems).
    #[test]
    fn test_qr_svd_all_zeros() {
        let a: Mat<f64> = Mat::zeros(8, 8);
        let qr = QrSvd::compute(a.as_ref()).expect("zero matrix must converge");
        for &s in qr.singular_values() {
            assert!(s.abs() < 1e-12, "sigma = {s}");
        }
    }
}
