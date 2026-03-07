//! Selective SVD computation (GESVDX-style).
//!
//! Computes only selected singular values and optionally singular vectors.
//! Selection can be:
//! - By index range: compute singular values with indices [il, iu]
//! - By value range: compute singular values in the interval [vl, vu]
//!
//! This is more efficient than computing the full SVD when only a subset
//! of singular values is needed.
//!
//! # Algorithm
//!
//! 1. Reduce the matrix to bidiagonal form: A = Q * B * P^T
//! 2. Form B^T * B (for singular values of B)
//! 3. Use bisection/inverse iteration on B^T * B to get selected eigenvalues/vectors
//! 4. Singular values of B are sqrt(eigenvalues of B^T * B)
//! 5. Transform back to get singular vectors of A

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for selective SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectiveSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Invalid range specified.
    InvalidRange,
    /// No singular values found in the specified range.
    NoSingularValuesInRange,
    /// Algorithm did not converge.
    NotConverged,
    /// Invalid index range.
    InvalidIndexRange,
}

impl core::fmt::Display for SelectiveSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::InvalidRange => write!(f, "Invalid value range specified"),
            Self::NoSingularValuesInRange => {
                write!(f, "No singular values found in the specified range")
            }
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::InvalidIndexRange => write!(f, "Invalid index range"),
        }
    }
}

impl std::error::Error for SelectiveSvdError {}

/// Specifies which singular values to compute.
#[derive(Debug, Clone)]
pub enum SingularValueSelector<T> {
    /// Compute all singular values.
    All,
    /// Compute singular values in the interval [low, high].
    /// 0 <= low <= high.
    ValueRange {
        /// Lower bound (must be >= 0).
        low: T,
        /// Upper bound.
        high: T,
    },
    /// Compute singular values with indices [low, high] (0-indexed).
    /// Index 0 is the largest singular value.
    IndexRange {
        /// Starting index (0 = largest singular value).
        low: usize,
        /// Ending index (inclusive).
        high: usize,
    },
}

/// Result of selective SVD computation.
#[derive(Debug, Clone)]
pub struct SelectiveSvd<T: Scalar> {
    /// Computed singular values (sorted in descending order).
    sigma: Vec<T>,
    /// Left singular vectors (columns correspond to sigma, may be None).
    u: Option<Mat<T>>,
    /// Right singular vectors as V^T (rows correspond to sigma, may be None).
    vt: Option<Mat<T>>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
    /// Index offset (for index-based selection).
    index_offset: usize,
}

/// Maximum iterations for bisection.
const MAX_BISECTION_ITER: usize = 1000;
/// Maximum iterations for inverse iteration.
const MAX_INVERSE_ITER: usize = 100;

impl<T: Field + Real + bytemuck::Zeroable> SelectiveSvd<T> {
    /// Computes selected singular values and vectors.
    ///
    /// # Arguments
    ///
    /// * `a` - The input matrix
    /// * `selector` - Which singular values to compute
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::{SelectiveSvd, SingularValueSelector};
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    ///     &[7.0, 8.0, 9.0],
    ///     &[10.0, 11.0, 12.0],
    /// ]);
    ///
    /// // Compute only the largest singular value
    /// let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::IndexRange {
    ///     low: 0,
    ///     high: 0,
    /// }).unwrap();
    ///
    /// assert_eq!(svd.singular_values().len(), 1);
    /// ```
    pub fn compute(
        a: MatRef<'_, T>,
        selector: SingularValueSelector<T>,
    ) -> Result<Self, SelectiveSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SelectiveSvdError::EmptyMatrix);
        }

        // For wide matrices, work with transpose
        if m < n {
            let mut at = Mat::zeros(n, m);
            for i in 0..m {
                for j in 0..n {
                    at[(j, i)] = a[(i, j)];
                }
            }
            let svd_t = Self::compute_tall(at.as_ref(), selector)?;

            // Swap U and Vt
            let u = svd_t.vt.map(|vt| {
                let mut u = Mat::zeros(m, vt.ncols());
                for i in 0..vt.nrows().min(m) {
                    for j in 0..vt.ncols() {
                        u[(i, j)] = vt[(i, j)];
                    }
                }
                u
            });

            let vt = svd_t.u.map(|u| {
                let mut vt = Mat::zeros(u.ncols(), n);
                for i in 0..u.ncols() {
                    for j in 0..u.nrows().min(n) {
                        vt[(i, j)] = u[(j, i)];
                    }
                }
                vt
            });

            return Ok(Self {
                sigma: svd_t.sigma,
                u,
                vt,
                m,
                n,
                index_offset: svd_t.index_offset,
            });
        }

        Self::compute_tall(a, selector)
    }

    /// Computes only singular values (no vectors).
    pub fn singular_values_only(
        a: MatRef<'_, T>,
        selector: SingularValueSelector<T>,
    ) -> Result<Self, SelectiveSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SelectiveSvdError::EmptyMatrix);
        }

        // For wide matrices, work with transpose
        if m < n {
            let mut at = Mat::zeros(n, m);
            for i in 0..m {
                for j in 0..n {
                    at[(j, i)] = a[(i, j)];
                }
            }
            let mut svd_t = Self::compute_tall_values_only(at.as_ref(), selector)?;
            svd_t.m = m;
            svd_t.n = n;
            return Ok(svd_t);
        }

        Self::compute_tall_values_only(a, selector)
    }

    /// Computes selected SVD for tall or square matrices (m >= n).
    fn compute_tall(
        a: MatRef<'_, T>,
        selector: SingularValueSelector<T>,
    ) -> Result<Self, SelectiveSvdError> {
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        // Step 1: Reduce to bidiagonal form
        let (u_b, d, e, vt_b) = bidiagonalize(a)?;

        // Step 2: Form the tridiagonal matrix T = B^T * B
        // T has diagonal d[i]^2 + e[i-1]^2 and off-diagonal d[i]*e[i]
        let (t_diag, t_offdiag) = form_btb_tridiagonal(&d, &e);

        // Step 3: Determine which eigenvalues of T to compute
        // Singular values of B = sqrt(eigenvalues of T)
        let (index_low, _index_high, sigma) = match selector {
            SingularValueSelector::All => {
                let sigma = compute_all_singular_values(&t_diag, &t_offdiag)?;
                (0, sigma.len().saturating_sub(1), sigma)
            }
            SingularValueSelector::ValueRange { low, high } => {
                if low < T::zero() || high < low {
                    return Err(SelectiveSvdError::InvalidRange);
                }
                // Singular values in [low, high] correspond to eigenvalues in [low^2, high^2]
                let eig_low = low * low;
                let eig_high = high * high;
                let (sigma, idx_low) =
                    compute_singular_values_in_range(&t_diag, &t_offdiag, eig_low, eig_high)?;
                if sigma.is_empty() {
                    return Err(SelectiveSvdError::NoSingularValuesInRange);
                }
                (idx_low, idx_low + sigma.len() - 1, sigma)
            }
            SingularValueSelector::IndexRange { low, high } => {
                if low > high || high >= k {
                    return Err(SelectiveSvdError::InvalidIndexRange);
                }
                let sigma = compute_singular_values_by_index(&t_diag, &t_offdiag, low, high)?;
                (low, high, sigma)
            }
        };

        let num_sv = sigma.len();

        // Step 4: Compute singular vectors using inverse iteration on B^T*B
        let v_bidiag = compute_right_singular_vectors(&t_diag, &t_offdiag, &sigma)?;

        // Step 5: Compute U vectors: u_i = B * v_i / sigma_i
        let u_bidiag = compute_left_singular_vectors(&d, &e, &sigma, &v_bidiag);

        // Step 6: Transform back to original matrix basis
        // U = U_b * U_bidiag, V^T = V_bidiag^T * Vt_b
        let mut u = Mat::zeros(m, num_sv);
        for j in 0..num_sv {
            for i in 0..m {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + u_b[(i, l)] * u_bidiag[(l, j)];
                }
                u[(i, j)] = sum;
            }
        }

        let mut vt = Mat::zeros(num_sv, n);
        for i in 0..num_sv {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + v_bidiag[(l, i)] * vt_b[(l, j)];
                }
                vt[(i, j)] = sum;
            }
        }

        Ok(Self {
            sigma,
            u: Some(u),
            vt: Some(vt),
            m,
            n,
            index_offset: index_low,
        })
    }

    /// Computes only singular values for tall or square matrices.
    fn compute_tall_values_only(
        a: MatRef<'_, T>,
        selector: SingularValueSelector<T>,
    ) -> Result<Self, SelectiveSvdError> {
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        // Step 1: Reduce to bidiagonal form (we only need d and e)
        let (_u_b, d, e, _vt_b) = bidiagonalize(a)?;

        // Step 2: Form the tridiagonal matrix T = B^T * B
        let (t_diag, t_offdiag) = form_btb_tridiagonal(&d, &e);

        // Step 3: Compute selected singular values
        let (index_offset, sigma) = match selector {
            SingularValueSelector::All => {
                let sigma = compute_all_singular_values(&t_diag, &t_offdiag)?;
                (0, sigma)
            }
            SingularValueSelector::ValueRange { low, high } => {
                if low < T::zero() || high < low {
                    return Err(SelectiveSvdError::InvalidRange);
                }
                let eig_low = low * low;
                let eig_high = high * high;
                let (sigma, idx) =
                    compute_singular_values_in_range(&t_diag, &t_offdiag, eig_low, eig_high)?;
                if sigma.is_empty() {
                    return Err(SelectiveSvdError::NoSingularValuesInRange);
                }
                (idx, sigma)
            }
            SingularValueSelector::IndexRange { low, high } => {
                if low > high || high >= k {
                    return Err(SelectiveSvdError::InvalidIndexRange);
                }
                let sigma = compute_singular_values_by_index(&t_diag, &t_offdiag, low, high)?;
                (low, sigma)
            }
        };

        Ok(Self {
            sigma,
            u: None,
            vt: None,
            m,
            n,
            index_offset,
        })
    }

    /// Returns the computed singular values (sorted in descending order).
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the left singular vectors U (m×k where k is number of computed values).
    /// Returns None if only values were computed.
    pub fn u(&self) -> Option<MatRef<'_, T>> {
        self.u.as_ref().map(|u| u.as_ref())
    }

    /// Returns V^T (k×n where k is number of computed values).
    /// Returns None if only values were computed.
    pub fn vt(&self) -> Option<MatRef<'_, T>> {
        self.vt.as_ref().map(|vt| vt.as_ref())
    }

    /// Returns the index of the first computed singular value.
    /// Index 0 corresponds to the largest singular value.
    pub fn index_offset(&self) -> usize {
        self.index_offset
    }

    /// Returns the number of computed singular values.
    pub fn count(&self) -> usize {
        self.sigma.len()
    }

    /// Reconstructs the low-rank approximation: A_k = U * Sigma * V^T
    pub fn reconstruct(&self) -> Option<Mat<T>> {
        let u = self.u.as_ref()?;
        let vt = self.vt.as_ref()?;

        let mut a = Mat::zeros(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..self.sigma.len() {
                    sum = sum + u[(i, l)] * self.sigma[l] * vt[(l, j)];
                }
                a[(i, j)] = sum;
            }
        }
        Some(a)
    }
}

/// Bidiagonalizes matrix A: A = U * B * V^T.
/// Returns (U, diagonal of B, superdiagonal of B, V^T).
fn bidiagonalize<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<(Mat<T>, Vec<T>, Vec<T>, Mat<T>), SelectiveSvdError> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);

    // Copy A to working matrix
    let mut work = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            work[(i, j)] = a[(i, j)];
        }
    }

    let mut tau_left = vec![T::zero(); k];
    let num_right = k.saturating_sub(1);
    let mut tau_right = vec![T::zero(); num_right];

    let mut d = vec![T::zero(); k];
    let mut e = vec![T::zero(); num_right];

    for j in 0..k {
        // Apply Householder from the left
        let (tau, beta) = householder_left(&mut work, j, m);
        d[j] = beta;
        tau_left[j] = tau;
        apply_householder_left(&mut work, j, m, n, tau);

        // Apply Householder from the right
        if j < n - 1 {
            let (tau, beta) = householder_right(&mut work, j, n);
            if j < e.len() {
                e[j] = beta;
                tau_right[j] = tau;
            }
            apply_householder_right(&mut work, j, m, n, tau);
        }
    }

    // Build U from left reflections
    let mut u = Mat::zeros(m, m);
    for i in 0..m {
        u[(i, i)] = T::one();
    }
    for j in 0..k {
        let tau = tau_left[j];
        if tau != T::zero() {
            for r in 0..m {
                let mut w = u[(r, j)];
                for i in (j + 1)..m {
                    w = w + u[(r, i)] * work[(i, j)];
                }
                let tw = tau * w;
                u[(r, j)] = u[(r, j)] - tw;
                for i in (j + 1)..m {
                    u[(r, i)] = u[(r, i)] - tw * work[(i, j)];
                }
            }
        }
    }

    // Build V from right reflections
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = T::one();
    }
    for j in 0..tau_right.len() {
        let tau = tau_right[j];
        if tau != T::zero() {
            let start = j + 1;
            for r in 0..n {
                let mut w = v[(r, start)];
                for i in (start + 1)..n {
                    w = w + v[(r, i)] * work[(j, i)];
                }
                let tw = tau * w;
                v[(r, start)] = v[(r, start)] - tw;
                for i in (start + 1)..n {
                    v[(r, i)] = v[(r, i)] - tw * work[(j, i)];
                }
            }
        }
    }

    // V^T
    let mut vt = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            vt[(i, j)] = v[(j, i)];
        }
    }

    Ok((u, d, e, vt))
}

/// Householder reflection for zeroing column below diagonal.
fn householder_left<T: Field + Real>(work: &mut Mat<T>, j: usize, m: usize) -> (T, T) {
    let mut norm_sq = T::zero();
    for i in j..m {
        norm_sq = norm_sq + work[(i, j)] * work[(i, j)];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    let x_j = work[(j, j)];
    let beta = if x_j >= T::zero() { -norm } else { norm };
    let tau = (beta - x_j) / beta;

    let scale = T::one() / (x_j - beta);
    for i in (j + 1)..m {
        work[(i, j)] = work[(i, j)] * scale;
    }

    (tau, beta)
}

/// Householder reflection for zeroing row to right of superdiagonal.
fn householder_right<T: Field + Real>(work: &mut Mat<T>, j: usize, n: usize) -> (T, T) {
    let start = j + 1;
    let mut norm_sq = T::zero();
    for i in start..n {
        norm_sq = norm_sq + work[(j, i)] * work[(j, i)];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    let x_j = work[(j, start)];
    let beta = if x_j >= T::zero() { -norm } else { norm };
    let tau = (beta - x_j) / beta;

    let scale = T::one() / (x_j - beta);
    for i in (start + 1)..n {
        work[(j, i)] = work[(j, i)] * scale;
    }

    (tau, beta)
}

/// Applies left Householder reflection.
fn apply_householder_left<T: Field + Real>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }
    for col in (j + 1)..n {
        let mut w = work[(j, col)];
        for i in (j + 1)..m {
            w = w + work[(i, j)] * work[(i, col)];
        }
        let tw = tau * w;
        work[(j, col)] = work[(j, col)] - tw;
        for i in (j + 1)..m {
            work[(i, col)] = work[(i, col)] - tw * work[(i, j)];
        }
    }
}

/// Applies right Householder reflection.
fn apply_householder_right<T: Field + Real>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }
    let start = j + 1;
    for row in (j + 1)..m {
        let mut w = work[(row, start)];
        for i in (start + 1)..n {
            w = w + work[(j, i)] * work[(row, i)];
        }
        let tw = tau * w;
        work[(row, start)] = work[(row, start)] - tw;
        for i in (start + 1)..n {
            work[(row, i)] = work[(row, i)] - tw * work[(j, i)];
        }
    }
}

/// Forms the tridiagonal matrix T = B^T * B from bidiagonal B.
/// B has diagonal d and superdiagonal e.
/// T\[i,i\] = d[i]^2 + e\[i-1\]^2 (where e\[-1\] = 0)
/// T\[i,i+1\] = T\[i+1,i\] = d[i] * e\[i\]
fn form_btb_tridiagonal<T: Field + Real>(d: &[T], e: &[T]) -> (Vec<T>, Vec<T>) {
    let n = d.len();
    let mut t_diag = vec![T::zero(); n];
    let mut t_offdiag = vec![T::zero(); n.saturating_sub(1)];

    for i in 0..n {
        // Diagonal: d[i]^2 + e[i-1]^2
        t_diag[i] = d[i] * d[i];
        if i > 0 && i - 1 < e.len() {
            t_diag[i] = t_diag[i] + e[i - 1] * e[i - 1];
        }
    }

    for i in 0..t_offdiag.len() {
        // Off-diagonal: d[i] * e[i]
        if i < e.len() {
            t_offdiag[i] = d[i] * e[i];
        }
    }

    (t_diag, t_offdiag)
}

/// Computes all singular values from the tridiagonal T = B^T * B.
fn compute_all_singular_values<T: Field + Real>(
    t_diag: &[T],
    t_offdiag: &[T],
) -> Result<Vec<T>, SelectiveSvdError> {
    let n = t_diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Get bounds for eigenvalues
    let (low, high) = gershgorin_bounds(t_diag, t_offdiag);

    // Compute all eigenvalues using bisection
    let eigenvalues = bisection_eigenvalues(t_diag, t_offdiag, 0, n - 1, low, high)?;

    // Convert to singular values (sqrt of eigenvalues)
    let mut sigma: Vec<T> = eigenvalues
        .iter()
        .map(|&eig| {
            if eig > T::zero() {
                Real::sqrt(eig)
            } else {
                T::zero()
            }
        })
        .collect();

    // Sort in descending order
    sigma.sort_by(|a, b| {
        if *b > *a {
            core::cmp::Ordering::Greater
        } else if *b < *a {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Equal
        }
    });

    Ok(sigma)
}

/// Computes singular values in a value range.
fn compute_singular_values_in_range<T: Field + Real>(
    t_diag: &[T],
    t_offdiag: &[T],
    eig_low: T,
    eig_high: T,
) -> Result<(Vec<T>, usize), SelectiveSvdError> {
    let n = t_diag.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }

    // Count eigenvalues below each bound
    let count_below_low = sturm_count(t_diag, t_offdiag, eig_low);
    let count_below_high = sturm_count(t_diag, t_offdiag, eig_high);

    if count_below_high <= count_below_low {
        return Ok((Vec::new(), 0));
    }

    let index_low = count_below_low;
    let index_high = count_below_high - 1;

    // Get bounds for all eigenvalues
    let (global_low, global_high) = gershgorin_bounds(t_diag, t_offdiag);
    let search_low = if eig_low < global_low {
        global_low
    } else {
        eig_low
    };
    let search_high = if eig_high > global_high {
        global_high
    } else {
        eig_high
    };

    // Compute eigenvalues in the range
    let eigenvalues = bisection_eigenvalues(
        t_diag,
        t_offdiag,
        index_low,
        index_high,
        search_low,
        search_high,
    )?;

    // Convert to singular values
    let mut sigma: Vec<T> = eigenvalues
        .iter()
        .map(|&eig| {
            if eig > T::zero() {
                Real::sqrt(eig)
            } else {
                T::zero()
            }
        })
        .collect();

    // Sort in descending order
    sigma.sort_by(|a, b| {
        if *b > *a {
            core::cmp::Ordering::Greater
        } else if *b < *a {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Equal
        }
    });

    // The index_offset is the position in the full sorted list
    // Since we're computing eigenvalues in a middle range, we need to figure out
    // how many singular values are larger than the ones we computed
    let all_sigma = compute_all_singular_values(t_diag, t_offdiag)?;
    let mut idx_offset = 0;
    if !sigma.is_empty() && !all_sigma.is_empty() {
        let max_computed = sigma[0];
        for s in &all_sigma {
            if *s > max_computed + <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one())
            {
                idx_offset += 1;
            } else {
                break;
            }
        }
    }

    Ok((sigma, idx_offset))
}

/// Computes singular values by index range.
fn compute_singular_values_by_index<T: Field + Real>(
    t_diag: &[T],
    t_offdiag: &[T],
    index_low: usize,
    index_high: usize,
) -> Result<Vec<T>, SelectiveSvdError> {
    let n = t_diag.len();
    if n == 0 || index_high >= n {
        return Err(SelectiveSvdError::InvalidIndexRange);
    }

    // Get all singular values first
    let all_sigma = compute_all_singular_values(t_diag, t_offdiag)?;

    // Extract the requested range
    let sigma: Vec<T> = all_sigma
        .iter()
        .skip(index_low)
        .take(index_high - index_low + 1)
        .copied()
        .collect();

    Ok(sigma)
}

/// Gershgorin bounds for eigenvalues of symmetric tridiagonal matrix.
fn gershgorin_bounds<T: Field + Real>(diag: &[T], offdiag: &[T]) -> (T, T) {
    let n = diag.len();
    if n == 0 {
        return (T::zero(), T::zero());
    }

    let mut low = diag[0];
    let mut high = diag[0];

    // First row
    if !offdiag.is_empty() {
        let r = Scalar::abs(offdiag[0]);
        low = if diag[0] - r < low { diag[0] - r } else { low };
        high = if diag[0] + r > high {
            diag[0] + r
        } else {
            high
        };
    }

    // Middle rows
    for i in 1..n - 1 {
        let r = Scalar::abs(offdiag[i - 1]) + Scalar::abs(offdiag[i]);
        let center = diag[i];
        if center - r < low {
            low = center - r;
        }
        if center + r > high {
            high = center + r;
        }
    }

    // Last row
    if n > 1 {
        let r = Scalar::abs(offdiag[n - 2]);
        let center = diag[n - 1];
        if center - r < low {
            low = center - r;
        }
        if center + r > high {
            high = center + r;
        }
    }

    // Ensure non-negative for eigenvalues of B^T*B
    if low < T::zero() {
        low = T::zero();
    }

    // Add small margin
    let margin = (high - low) * T::from_f64(0.01).unwrap_or(T::one());
    (low - margin, high + margin)
}

/// Counts eigenvalues less than or equal to x using Sturm sequence.
fn sturm_count<T: Field + Real>(diag: &[T], offdiag: &[T], x: T) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }

    let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());
    let mut count = 0;
    let mut d_prev = diag[0] - x;

    if d_prev <= T::zero() {
        count += 1;
    }

    for i in 1..n {
        let e_sq = offdiag[i - 1] * offdiag[i - 1];

        // d_i = a_i - x - e_{i-1}^2 / d_{i-1}
        let d_curr = if Scalar::abs(d_prev) < eps {
            // Avoid division by zero
            diag[i] - x - e_sq / (if d_prev >= T::zero() { eps } else { -eps })
        } else {
            diag[i] - x - e_sq / d_prev
        };

        if d_curr <= T::zero() {
            count += 1;
        }
        d_prev = d_curr;
    }

    count
}

/// Computes eigenvalues using bisection.
fn bisection_eigenvalues<T: Field + Real>(
    diag: &[T],
    offdiag: &[T],
    index_low: usize,
    index_high: usize,
    value_low: T,
    value_high: T,
) -> Result<Vec<T>, SelectiveSvdError> {
    let num_eigenvalues = index_high - index_low + 1;
    let mut eigenvalues = vec![T::zero(); num_eigenvalues];
    let eps = <T as Scalar>::epsilon();
    let tol = eps
        * (Scalar::abs(value_low) + Scalar::abs(value_high))
        * T::from_f64(100.0).unwrap_or(T::one());

    for k in 0..num_eigenvalues {
        let target_index = index_low + k;
        let mut lo = value_low;
        let mut hi = value_high;

        for _ in 0..MAX_BISECTION_ITER {
            let mid = (lo + hi) / T::from_f64(2.0).unwrap_or_else(T::zero);
            let count = sturm_count(diag, offdiag, mid);

            if count <= target_index {
                lo = mid;
            } else {
                hi = mid;
            }

            if hi - lo < tol {
                break;
            }
        }

        eigenvalues[k] = (lo + hi) / T::from_f64(2.0).unwrap_or_else(T::zero);
    }

    Ok(eigenvalues)
}

/// Computes right singular vectors from T = B^T * B eigenvalues using inverse iteration.
fn compute_right_singular_vectors<T: Field + Real + bytemuck::Zeroable>(
    t_diag: &[T],
    t_offdiag: &[T],
    sigma: &[T],
) -> Result<Mat<T>, SelectiveSvdError> {
    let n = t_diag.len();
    let k = sigma.len();

    if n == 0 || k == 0 {
        return Ok(Mat::zeros(0, 0));
    }

    let mut v = Mat::zeros(n, k);
    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    for (col, &s) in sigma.iter().enumerate() {
        // Eigenvalue of T is s^2
        let lambda = s * s;

        // Initial vector
        let mut x = vec![T::one(); n];
        let norm_init = Real::sqrt(T::from_usize(n).unwrap_or(T::one()));
        for xi in &mut x {
            *xi = *xi / norm_init;
        }

        // Inverse iteration: solve (T - lambda*I) * x_new = x
        for _ in 0..MAX_INVERSE_ITER {
            // Solve tridiagonal system
            match solve_tridiagonal_shifted(t_diag, t_offdiag, &x, lambda) {
                Ok(x_new) => {
                    // Normalize
                    let mut norm_sq = T::zero();
                    for &xi in &x_new {
                        norm_sq = norm_sq + xi * xi;
                    }
                    let norm = Real::sqrt(norm_sq);

                    if norm < tol {
                        break;
                    }

                    // Check convergence
                    let mut diff = T::zero();
                    for i in 0..n {
                        let new_val = x_new[i] / norm;
                        diff = diff + (new_val - x[i]) * (new_val - x[i]);
                        x[i] = new_val;
                    }

                    if Real::sqrt(diff) < tol {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        // Orthogonalize against previous vectors (Gram-Schmidt)
        for prev in 0..col {
            let mut dot = T::zero();
            for i in 0..n {
                dot = dot + x[i] * v[(i, prev)];
            }
            for i in 0..n {
                x[i] = x[i] - dot * v[(i, prev)];
            }
        }

        // Final normalization
        let mut norm_sq = T::zero();
        for &xi in &x {
            norm_sq = norm_sq + xi * xi;
        }
        let norm = Real::sqrt(norm_sq);
        if norm > tol {
            for i in 0..n {
                v[(i, col)] = x[i] / norm;
            }
        } else {
            v[(col.min(n - 1), col)] = T::one();
        }
    }

    Ok(v)
}

/// Computes left singular vectors: u_i = B * v_i / sigma_i
fn compute_left_singular_vectors<T: Field + Real + bytemuck::Zeroable>(
    d: &[T],
    e: &[T],
    sigma: &[T],
    v: &Mat<T>,
) -> Mat<T> {
    let n = d.len();
    let k = sigma.len();
    let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

    // B is n×n bidiagonal with diagonal d and superdiagonal e
    // u = B * v / sigma
    let mut u = Mat::zeros(n, k);

    for col in 0..k {
        let s = sigma[col];
        if s < eps {
            // Zero singular value, set u to standard basis vector
            u[(col.min(n - 1), col)] = T::one();
            continue;
        }

        // Multiply B * v[:, col]
        for i in 0..n {
            let mut sum = d[i] * v[(i, col)];
            if i < e.len() {
                sum = sum + e[i] * v[(i + 1, col)];
            }
            u[(i, col)] = sum / s;
        }

        // Normalize
        let mut norm_sq = T::zero();
        for i in 0..n {
            norm_sq = norm_sq + u[(i, col)] * u[(i, col)];
        }
        let norm = Real::sqrt(norm_sq);
        if norm > eps {
            for i in 0..n {
                u[(i, col)] = u[(i, col)] / norm;
            }
        }
    }

    u
}

/// Solves (T - lambda*I) * x = b for tridiagonal T using Thomas algorithm.
fn solve_tridiagonal_shifted<T: Field + Real>(
    diag: &[T],
    offdiag: &[T],
    b: &[T],
    lambda: T,
) -> Result<Vec<T>, SelectiveSvdError> {
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let eps = <T as Scalar>::epsilon() * T::from_f64(1000.0).unwrap_or(T::one());

    // Forward elimination
    let mut c_prime = vec![T::zero(); n];
    let mut d_prime = vec![T::zero(); n];

    // First row
    let diag_shifted = diag[0] - lambda;
    if Scalar::abs(diag_shifted) < eps {
        // Near-singular, use regularization
        let reg = if diag_shifted >= T::zero() { eps } else { -eps };
        c_prime[0] = if !offdiag.is_empty() {
            offdiag[0] / reg
        } else {
            T::zero()
        };
        d_prime[0] = b[0] / reg;
    } else {
        c_prime[0] = if !offdiag.is_empty() {
            offdiag[0] / diag_shifted
        } else {
            T::zero()
        };
        d_prime[0] = b[0] / diag_shifted;
    }

    // Forward sweep
    for i in 1..n {
        let a_i = offdiag[i - 1];
        let diag_shifted = diag[i] - lambda;
        let denom = diag_shifted - a_i * c_prime[i - 1];

        if Scalar::abs(denom) < eps {
            let reg = if denom >= T::zero() { eps } else { -eps };
            c_prime[i] = if i < offdiag.len() {
                offdiag[i] / reg
            } else {
                T::zero()
            };
            d_prime[i] = (b[i] - a_i * d_prime[i - 1]) / reg;
        } else {
            c_prime[i] = if i < offdiag.len() {
                offdiag[i] / denom
            } else {
                T::zero()
            };
            d_prime[i] = (b[i] - a_i * d_prime[i - 1]) / denom;
        }
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Counts singular values greater than a threshold.
pub fn count_singular_values_above<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    threshold: T,
) -> Result<usize, SelectiveSvdError> {
    if a.nrows() == 0 || a.ncols() == 0 {
        return Err(SelectiveSvdError::EmptyMatrix);
    }
    if threshold < T::zero() {
        return Err(SelectiveSvdError::InvalidRange);
    }

    // Form B^T * B tridiagonal
    let (_, d, e, _) = bidiagonalize(a)?;
    let (t_diag, t_offdiag) = form_btb_tridiagonal(&d, &e);

    // Count eigenvalues above threshold^2
    let threshold_sq = threshold * threshold;
    let (_, _high) = gershgorin_bounds(&t_diag, &t_offdiag);

    let count_below = sturm_count(&t_diag, &t_offdiag, threshold_sq);
    let total = t_diag.len();

    Ok(total - count_below)
}

/// Returns bounds for all singular values.
pub fn singular_value_bounds<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<(T, T), SelectiveSvdError> {
    if a.nrows() == 0 || a.ncols() == 0 {
        return Err(SelectiveSvdError::EmptyMatrix);
    }

    let (_, d, e, _) = bidiagonalize(a)?;
    let (t_diag, t_offdiag) = form_btb_tridiagonal(&d, &e);
    let (low, high) = gershgorin_bounds(&t_diag, &t_offdiag);

    // Convert eigenvalue bounds to singular value bounds
    let sv_low = if low > T::zero() {
        Real::sqrt(low)
    } else {
        T::zero()
    };
    let sv_high = if high > T::zero() {
        Real::sqrt(high)
    } else {
        T::zero()
    };

    Ok((sv_low, sv_high))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_selective_svd_all() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        // Should be sorted descending
        let sigma = svd.singular_values();
        assert!(sigma[0] >= sigma[1]);
    }

    #[test]
    fn test_selective_svd_index_range() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        // Get only the largest singular value
        let svd = SelectiveSvd::compute(
            a.as_ref(),
            SingularValueSelector::IndexRange { low: 0, high: 0 },
        )
        .unwrap();

        assert_eq!(svd.count(), 1);
        assert_eq!(svd.index_offset(), 0);
    }

    #[test]
    fn test_selective_svd_index_range_middle() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        // Get only the second singular value
        let svd = SelectiveSvd::compute(
            a.as_ref(),
            SingularValueSelector::IndexRange { low: 1, high: 1 },
        )
        .unwrap();

        assert_eq!(svd.count(), 1);
    }

    #[test]
    fn test_selective_svd_values_only() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd =
            SelectiveSvd::singular_values_only(a.as_ref(), SingularValueSelector::All).unwrap();

        assert_eq!(svd.count(), 2);
        assert!(svd.u().is_none());
        assert!(svd.vt().is_none());
    }

    #[test]
    fn test_selective_svd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 5.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        let sigma = svd.singular_values();

        // Diagonal matrix has singular values equal to absolute diagonal values
        assert!(approx_eq(sigma[0], 5.0, 1e-8));
        assert!(approx_eq(sigma[1], 3.0, 1e-8));
    }

    #[test]
    fn test_selective_svd_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();

        for &s in svd.singular_values() {
            assert!(approx_eq(s, 1.0, 1e-8));
        }
    }

    #[test]
    fn test_selective_svd_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        assert_eq!(svd.count(), 2);

        // Check U and Vt dimensions
        let u = svd.u().unwrap();
        let vt = svd.vt().unwrap();
        assert_eq!(u.nrows(), 4);
        assert_eq!(u.ncols(), 2);
        assert_eq!(vt.nrows(), 2);
        assert_eq!(vt.ncols(), 2);
    }

    #[test]
    fn test_selective_svd_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        assert_eq!(svd.count(), 2);
    }

    #[test]
    fn test_selective_svd_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        let reconstructed = svd.reconstruct().unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-6),
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
    fn test_selective_svd_1x1() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        assert_eq!(svd.count(), 1);
        assert!(approx_eq(svd.singular_values()[0], 5.0, 1e-8));
    }

    #[test]
    fn test_count_singular_values() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 5.0]]);

        // Count singular values > 4
        let count = count_singular_values_above(a.as_ref(), 4.0).unwrap();
        assert_eq!(count, 1); // Only 5 > 4

        // Count singular values > 2
        let count = count_singular_values_above(a.as_ref(), 2.0).unwrap();
        assert_eq!(count, 2); // Both 3 and 5 > 2

        // Count singular values > 6
        let count = count_singular_values_above(a.as_ref(), 6.0).unwrap();
        assert_eq!(count, 0); // Neither > 6
    }

    #[test]
    fn test_singular_value_bounds() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 5.0]]);

        let (low, high) = singular_value_bounds(a.as_ref()).unwrap();

        // Bounds should contain the actual singular values (3 and 5)
        assert!(low <= 3.0 + 0.5); // With margin
        assert!(high >= 5.0 - 0.5);
    }

    #[test]
    fn test_selective_svd_invalid_index() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        // Index 5 is out of range for 2 singular values
        let result = SelectiveSvd::compute(
            a.as_ref(),
            SingularValueSelector::IndexRange { low: 0, high: 5 },
        );

        assert!(matches!(result, Err(SelectiveSvdError::InvalidIndexRange)));
    }

    #[test]
    fn test_selective_svd_orthogonal_vectors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        let u = svd.u().unwrap();
        let vt = svd.vt().unwrap();

        // Check U^T * U is approximately identity
        for i in 0..u.ncols() {
            for j in 0..u.ncols() {
                let mut dot = 0.0;
                for k in 0..u.nrows() {
                    dot += u[(k, i)] * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-6),
                    "U^T*U[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }

        // Check V^T * V is approximately identity (Vt rows are V columns)
        for i in 0..vt.nrows() {
            for j in 0..vt.nrows() {
                let mut dot = 0.0;
                for k in 0..vt.ncols() {
                    dot += vt[(i, k)] * vt[(j, k)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-6),
                    "V*V^T[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_form_btb_tridiagonal() {
        // B with d = [2, 3, 1] and e = [1, 2]
        // B = [2 1 0]
        //     [0 3 2]
        //     [0 0 1]
        // B^T * B = [4   2   0]
        //          [2  10   6]
        //          [0   6   5]
        let d = vec![2.0f64, 3.0, 1.0];
        let e = vec![1.0f64, 2.0];

        let (t_diag, t_offdiag) = form_btb_tridiagonal(&d, &e);

        // Diagonal: d[i]^2 + e[i-1]^2
        assert!(approx_eq(t_diag[0], 4.0, 1e-10)); // 2^2 = 4
        assert!(approx_eq(t_diag[1], 10.0, 1e-10)); // 3^2 + 1^2 = 10
        assert!(approx_eq(t_diag[2], 5.0, 1e-10)); // 1^2 + 2^2 = 5

        // Off-diagonal: d[i] * e[i]
        assert!(approx_eq(t_offdiag[0], 2.0, 1e-10)); // 2 * 1 = 2
        assert!(approx_eq(t_offdiag[1], 6.0, 1e-10)); // 3 * 2 = 6
    }

    #[test]
    fn test_selective_svd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let svd = SelectiveSvd::compute(a.as_ref(), SingularValueSelector::All).unwrap();
        assert_eq!(svd.count(), 2);
    }
}
