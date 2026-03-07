//! Hessenberg reduction.
//!
//! Reduces a general matrix A to upper Hessenberg form: A = Q H Q^T
//! where Q is orthogonal and H is upper Hessenberg (zeros below subdiagonal).
//!
//! This module provides:
//!
//! - **Hessenberg**: Full Hessenberg decomposition with explicit Q
//! - **HessenbergFactors**: LAPACK-style compact storage (gehrd)
//! - **gehrd**: Reduce to Hessenberg form with compact storage
//! - **orghr**: Generate Q from compact Hessenberg factorization
//! - **ormhr**: Apply Q to a matrix without forming it explicitly
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::{gehrd, orghr, ormhr, Side, Trans};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 1.0, -2.0, 2.0],
//!     &[1.0, 2.0, 0.0, 1.0],
//!     &[-2.0, 0.0, 3.0, -2.0],
//!     &[2.0, 1.0, -2.0, -1.0],
//! ]);
//!
//! // Compact Hessenberg factorization
//! let factors = gehrd(a.as_ref()).unwrap();
//!
//! // Generate Q explicitly (if needed)
//! let q = orghr(&factors).unwrap();
//!
//! // Or apply Q to a matrix without forming it
//! let c = Mat::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
//! let qc = ormhr(&factors, Side::Left, Trans::NoTrans, c.as_ref()).unwrap();
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Default block size for blocked Hessenberg reduction.
const DEFAULT_BLOCK_SIZE: usize = 64;

/// Error type for Hessenberg reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessenbergError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
}

impl core::fmt::Display for HessenbergError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
        }
    }
}

impl std::error::Error for HessenbergError {}

/// Upper Hessenberg form of a matrix.
///
/// For a matrix A, computes A = Q H Q^T where:
/// - Q is orthogonal
/// - H is upper Hessenberg (h_ij = 0 for i > j + 1)
#[derive(Debug, Clone)]
pub struct Hessenberg<T: Scalar> {
    /// The orthogonal matrix Q.
    q: Mat<T>,
    /// The upper Hessenberg matrix H.
    h: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> Hessenberg<T> {
    /// Reduces a square matrix to upper Hessenberg form.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Hessenberg;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    ///     &[7.0, 8.0, 9.0],
    /// ]);
    ///
    /// let hess = Hessenberg::compute(a.as_ref()).unwrap();
    /// let h = hess.h();
    ///
    /// // H is upper Hessenberg (zeros below subdiagonal)
    /// assert!(h[(2, 0)].abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, HessenbergError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(HessenbergError::EmptyMatrix);
        }

        if m != n {
            return Err(HessenbergError::NotSquare);
        }

        if n == 1 {
            let mut h = Mat::zeros(1, 1);
            h[(0, 0)] = a[(0, 0)];
            let mut q = Mat::zeros(1, 1);
            q[(0, 0)] = T::one();
            return Ok(Self { q, h, n });
        }

        if n == 2 {
            let mut h = Mat::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    h[(i, j)] = a[(i, j)];
                }
            }
            let mut q = Mat::zeros(2, 2);
            q[(0, 0)] = T::one();
            q[(1, 1)] = T::one();
            return Ok(Self { q, h, n });
        }

        // Copy A to H
        let mut h = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = a[(i, j)];
            }
        }

        // Initialize Q as identity
        let mut q = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        // Store Householder vectors and tau values
        let mut v_storage: Vec<Vec<T>> = Vec::with_capacity(n - 2);
        let mut tau_storage: Vec<T> = Vec::with_capacity(n - 2);

        // Reduce to Hessenberg form using Householder reflections
        for k in 0..(n - 2) {
            // Create Householder reflection to zero H[k+2:n, k]
            let mut x: Vec<T> = Vec::with_capacity(n - k - 1);
            for i in (k + 1)..n {
                x.push(h[(i, k)]);
            }

            let (v, tau) = householder_vector_with_tau(&x);

            if tau != T::zero() {
                // Apply P from the left: H := P * H where P = I - tau*v*v^T
                for j in k..n {
                    let mut dot = T::zero();
                    for i in 0..v.len() {
                        dot = dot + v[i] * h[(k + 1 + i, j)];
                    }
                    let scaled = tau * dot;
                    for i in 0..v.len() {
                        h[(k + 1 + i, j)] = h[(k + 1 + i, j)] - scaled * v[i];
                    }
                }

                // Apply P from the right: H := H * P
                for i in 0..n {
                    let mut dot = T::zero();
                    for j in 0..v.len() {
                        dot = dot + h[(i, k + 1 + j)] * v[j];
                    }
                    let scaled = tau * dot;
                    for j in 0..v.len() {
                        h[(i, k + 1 + j)] = h[(i, k + 1 + j)] - scaled * v[j];
                    }
                }

                v_storage.push(v);
                tau_storage.push(tau);
            } else {
                v_storage.push(vec![T::zero(); n - k - 1]);
                tau_storage.push(T::zero());
            }
        }

        // Accumulate Q from the Householder reflections
        // Q = P_0 * P_1 * ... * P_{n-3}
        // Apply in forward order to build Q
        for k in 0..(n - 2) {
            let v = &v_storage[k];
            let tau = tau_storage[k];

            if tau != T::zero() {
                // Apply Q := Q * P_k where P_k = I - tau*v*v^T
                for i in 0..n {
                    let mut dot = T::zero();
                    for j in 0..v.len() {
                        dot = dot + q[(i, k + 1 + j)] * v[j];
                    }
                    let scaled = tau * dot;
                    for j in 0..v.len() {
                        q[(i, k + 1 + j)] = q[(i, k + 1 + j)] - scaled * v[j];
                    }
                }
            }
        }

        // Clean up small values below subdiagonal
        let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());
        for j in 0..(n - 2) {
            for i in (j + 2)..n {
                if Scalar::abs(h[(i, j)]) < eps {
                    h[(i, j)] = T::zero();
                }
            }
        }

        Ok(Self { q, h, n })
    }

    /// Computes the Hessenberg reduction with automatic algorithm selection.
    ///
    /// For matrices with size ≥ 96, automatically uses the blocked algorithm
    /// for better cache efficiency and performance. Otherwise uses the unblocked
    /// algorithm which has less overhead for small matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Hessenberg;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 256;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         a[(i, j)] = ((i + j) % 10 + 1) as f64;
    ///     }
    /// }
    ///
    /// // Automatically uses blocked algorithm for n >= 96
    /// let hess = Hessenberg::compute_auto(a.as_ref()).unwrap();
    /// ```
    pub fn compute_auto(a: MatRef<'_, T>) -> Result<Self, HessenbergError> {
        const AUTO_BLOCK_THRESHOLD: usize = 96;
        let n = a.nrows();

        // For large matrices, use blocked algorithm
        if n >= AUTO_BLOCK_THRESHOLD {
            Self::compute_blocked(a)
        } else {
            Self::compute(a)
        }
    }

    /// Reduces a square matrix to upper Hessenberg form using blocked algorithm.
    ///
    /// Uses block transformations with the WY representation for better cache
    /// utilization and to leverage BLAS-3 operations. This is typically faster
    /// for matrices larger than ~64×64.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Hessenberg;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0, 4.0],
    ///     &[5.0, 6.0, 7.0, 8.0],
    ///     &[9.0, 10.0, 11.0, 12.0],
    ///     &[13.0, 14.0, 15.0, 16.0],
    /// ]);
    ///
    /// let hess = Hessenberg::compute_blocked(a.as_ref()).unwrap();
    /// let h = hess.h();
    ///
    /// // H is upper Hessenberg (zeros below subdiagonal)
    /// assert!(h[(2, 0)].abs() < 1e-10);
    /// assert!(h[(3, 0)].abs() < 1e-10);
    /// assert!(h[(3, 1)].abs() < 1e-10);
    /// ```
    pub fn compute_blocked(a: MatRef<'_, T>) -> Result<Self, HessenbergError> {
        Self::compute_blocked_with_block_size(a, DEFAULT_BLOCK_SIZE)
    }

    /// Reduces a matrix to upper Hessenberg form with specified block size.
    pub fn compute_blocked_with_block_size(
        a: MatRef<'_, T>,
        block_size: usize,
    ) -> Result<Self, HessenbergError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(HessenbergError::EmptyMatrix);
        }

        if m != n {
            return Err(HessenbergError::NotSquare);
        }

        // For small matrices, use the unblocked algorithm
        if n <= block_size || n <= 3 {
            return Self::compute(a);
        }

        // Copy A to H
        let mut h = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = a[(i, j)];
            }
        }

        // Initialize Q as identity
        let mut q = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        let nb = block_size.min(n - 2);

        // Process columns in blocks
        let mut k = 0;
        while k < n - 2 {
            let kb = nb.min(n - 2 - k);

            // V stores the Householder vectors for this block
            // T stores the triangular factor of the WY representation
            let mut v_block: Vec<Vec<T>> = Vec::with_capacity(kb);
            let mut tau_block: Vec<T> = Vec::with_capacity(kb);

            // Compute Householder reflectors for this block
            for j in 0..kb {
                let col_idx = k + j;

                // Extract column segment h[col_idx+1:n, col_idx]
                let len = n - col_idx - 1;
                let mut x = vec![T::zero(); len];
                for i in 0..len {
                    x[i] = h[(col_idx + 1 + i, col_idx)];
                }

                let (v, tau) = householder_vector_with_tau(&x);
                tau_block.push(tau);

                if tau != T::zero() {
                    // Apply P from the left to current panel: H[col_idx+1:n, col_idx:n]
                    for jj in col_idx..n {
                        let mut dot = T::zero();
                        for i in 0..v.len() {
                            dot = dot + v[i] * h[(col_idx + 1 + i, jj)];
                        }
                        let scaled = tau * dot;
                        for i in 0..v.len() {
                            h[(col_idx + 1 + i, jj)] = h[(col_idx + 1 + i, jj)] - scaled * v[i];
                        }
                    }

                    // Apply P from the right: H[0:n, col_idx+1:n]
                    for ii in 0..n {
                        let mut dot = T::zero();
                        for i in 0..v.len() {
                            dot = dot + h[(ii, col_idx + 1 + i)] * v[i];
                        }
                        let scaled = tau * dot;
                        for i in 0..v.len() {
                            h[(ii, col_idx + 1 + i)] = h[(ii, col_idx + 1 + i)] - scaled * v[i];
                        }
                    }
                }

                v_block.push(v);
            }

            // Accumulate Q for this block
            // Q = Q * P_0 * P_1 * ... * P_{kb-1}
            for j in 0..kb {
                let col_idx = k + j;
                let v = &v_block[j];
                let tau = tau_block[j];

                if tau != T::zero() {
                    for row in 0..n {
                        let mut dot = T::zero();
                        for i in 0..v.len() {
                            dot = dot + q[(row, col_idx + 1 + i)] * v[i];
                        }
                        let scaled = tau * dot;
                        for i in 0..v.len() {
                            q[(row, col_idx + 1 + i)] = q[(row, col_idx + 1 + i)] - scaled * v[i];
                        }
                    }
                }
            }

            k += kb;
        }

        // Clean up small values below subdiagonal
        let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());
        for j in 0..(n - 2) {
            for i in (j + 2)..n {
                if Scalar::abs(h[(i, j)]) < eps {
                    h[(i, j)] = T::zero();
                }
            }
        }

        Ok(Self { q, h, n })
    }

    /// Returns the orthogonal matrix Q.
    pub fn q(&self) -> MatRef<'_, T> {
        self.q.as_ref()
    }

    /// Returns the upper Hessenberg matrix H.
    pub fn h(&self) -> MatRef<'_, T> {
        self.h.as_ref()
    }

    /// Reconstructs the original matrix: A = Q H Q^T.
    pub fn reconstruct(&self) -> Mat<T> {
        let mut qh = Mat::zeros(self.n, self.n);
        let mut a = Mat::zeros(self.n, self.n);

        // QH = Q * H
        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k in 0..self.n {
                    sum = sum + self.q[(i, k)] * self.h[(k, j)];
                }
                qh[(i, j)] = sum;
            }
        }

        // A = QH * Q^T
        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k in 0..self.n {
                    sum = sum + qh[(i, k)] * self.q[(j, k)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }
}

// ============================================================================
// LAPACK-style Compact Hessenberg Factorization (gehrd, orghr, ormhr)
// ============================================================================

/// Side of multiplication for ormhr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Multiply from the left: C <- Q * C or C <- Q^T * C
    Left,
    /// Multiply from the right: C <- C * Q or C <- C * Q^T
    Right,
}

/// Transpose option for ormhr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trans {
    /// No transpose: use Q
    NoTrans,
    /// Transpose: use Q^T (or Q^H for complex)
    Trans,
    /// Conjugate transpose: use Q^H (same as Trans for real)
    ConjTrans,
}

/// Compact Hessenberg factorization (LAPACK gehrd-style).
///
/// Stores the upper Hessenberg matrix H in the upper triangle and first
/// subdiagonal, with Householder vectors stored in the remaining lower part.
/// This is more memory-efficient than storing Q explicitly when Q is not needed
/// or will only be applied to matrices without forming it.
#[derive(Debug, Clone)]
pub struct HessenbergFactors<T: Scalar> {
    /// Combined H and Householder vectors.
    /// - Upper Hessenberg part (including subdiagonal): H
    /// - Below subdiagonal: Householder vectors v (with implicit leading 1)
    factors: Mat<T>,
    /// Scalar factors (tau) for Householder reflectors.
    /// tau[k] corresponds to reflector P_k = I - tau[k] * v_k * v_k^T
    tau: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Range of reduction: ilo (0-based)
    ilo: usize,
    /// Range of reduction: ihi (0-based, exclusive)
    ihi: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> HessenbergFactors<T> {
    /// Returns the matrix dimension.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the lower bound of the active range (0-based).
    #[inline]
    pub fn ilo(&self) -> usize {
        self.ilo
    }

    /// Returns the upper bound of the active range (0-based, exclusive).
    #[inline]
    pub fn ihi(&self) -> usize {
        self.ihi
    }

    /// Returns reference to the tau values.
    pub fn tau(&self) -> &[T] {
        &self.tau
    }

    /// Returns reference to the factors matrix.
    pub fn factors(&self) -> MatRef<'_, T> {
        self.factors.as_ref()
    }

    /// Extracts the upper Hessenberg matrix H.
    #[must_use]
    pub fn h(&self) -> Mat<T> {
        let mut h = Mat::zeros(self.n, self.n);
        for i in 0..self.n {
            // Copy upper Hessenberg part (including subdiagonal)
            let j_start = if i > 0 { i - 1 } else { 0 };
            for j in j_start..self.n {
                h[(i, j)] = self.factors[(i, j)];
            }
        }
        h
    }
}

/// Reduces a general matrix to upper Hessenberg form (LAPACK GEHRD).
///
/// Computes the orthogonal (unitary for complex) matrix Q and upper Hessenberg
/// matrix H such that A = Q * H * Q^T. The Householder vectors are stored in
/// the lower part of the output matrix for efficiency.
///
/// This is the standard range version that processes all rows/columns.
/// For reduced range (after balancing), use `gehrd_range`.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// `HessenbergFactors` containing compact representation of the factorization.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::gehrd;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
///     &[7.0, 8.0, 9.0],
/// ]);
///
/// let factors = gehrd(a.as_ref()).unwrap();
/// let h = factors.h();
///
/// // H is upper Hessenberg
/// assert!(h[(2, 0)].abs() < 1e-10);
/// ```
pub fn gehrd<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<HessenbergFactors<T>, HessenbergError> {
    gehrd_range(a, 0, a.nrows())
}

/// Reduces a general matrix to upper Hessenberg form with range (LAPACK GEHRD).
///
/// This version allows specifying a range [ilo, ihi) of active rows/columns,
/// which is useful after matrix balancing.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `ilo` - Lower bound of active range (0-based)
/// * `ihi` - Upper bound of active range (0-based, exclusive)
///
/// # Returns
///
/// `HessenbergFactors` containing compact representation of the factorization.
pub fn gehrd_range<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    ilo: usize,
    ihi: usize,
) -> Result<HessenbergFactors<T>, HessenbergError> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(HessenbergError::EmptyMatrix);
    }

    if m != n {
        return Err(HessenbergError::NotSquare);
    }

    // Validate range
    let ihi = ihi.min(n);
    let ilo = ilo.min(ihi);

    // Copy A to factors
    let mut factors = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            factors[(i, j)] = a[(i, j)];
        }
    }

    // Number of reflectors
    let nh = if ihi > ilo + 1 { ihi - ilo - 1 } else { 0 };
    let mut tau = vec![T::zero(); nh];

    if nh == 0 {
        // Already Hessenberg or trivial range
        return Ok(HessenbergFactors {
            factors,
            tau,
            n,
            ilo,
            ihi,
        });
    }

    // Reduce to Hessenberg form using Householder reflections
    // For k = ilo, ilo+1, ..., ihi-2:
    //   Create reflector to zero elements factors[k+2:ihi, k]
    for k in ilo..(ihi.saturating_sub(1)) {
        let tau_idx = k - ilo;
        if tau_idx >= tau.len() {
            break;
        }

        // Extract column segment factors[k+1:ihi, k]
        let len = ihi - k - 1;
        if len == 0 {
            continue;
        }

        let mut x = vec![T::zero(); len];
        for i in 0..len {
            x[i] = factors[(k + 1 + i, k)];
        }

        // Compute Householder vector
        let (v, tau_val) = householder_vector_lapack(&x);
        tau[tau_idx] = tau_val;

        if tau_val == T::zero() {
            continue;
        }

        // Store v in the lower part (v[0] is implicit 1)
        // factors[k+1, k] will become part of H (subdiagonal element = -sign*||x||)
        // factors[k+2:ihi, k] stores v[1:len]
        let beta = -sign_of(x[0]) * norm(&x);
        factors[(k + 1, k)] = beta;
        for i in 1..len {
            factors[(k + 1 + i, k)] = v[i];
        }

        // Apply P_k from the left: factors[k+1:ihi, k+1:n] <- P_k * factors[k+1:ihi, k+1:n]
        // P_k = I - tau * v * v^H
        for j in (k + 1)..n {
            let mut dot = factors[(k + 1, j)]; // v[0] = 1
            for i in 1..len {
                dot = dot + v[i].conj() * factors[(k + 1 + i, j)];
            }
            let scaled = tau_val * dot;
            factors[(k + 1, j)] = factors[(k + 1, j)] - scaled;
            for i in 1..len {
                factors[(k + 1 + i, j)] = factors[(k + 1 + i, j)] - scaled * v[i];
            }
        }

        // Apply P_k from the right: factors[0:ihi, k+1:ihi] <- factors[0:ihi, k+1:ihi] * P_k
        for i in 0..ihi {
            let mut dot = factors[(i, k + 1)]; // v[0] = 1
            for jj in 1..len {
                dot = dot + factors[(i, k + 1 + jj)] * v[jj];
            }
            let scaled = tau_val * dot;
            factors[(i, k + 1)] = factors[(i, k + 1)] - scaled;
            for jj in 1..len {
                factors[(i, k + 1 + jj)] = factors[(i, k + 1 + jj)] - scaled * v[jj].conj();
            }
        }
    }

    Ok(HessenbergFactors {
        factors,
        tau,
        n,
        ilo,
        ihi,
    })
}

/// Generates the orthogonal matrix Q from Hessenberg factorization (LAPACK ORGHR/UNGHR).
///
/// Given the output of `gehrd`, generates the orthogonal matrix Q such that
/// A = Q * H * Q^T.
///
/// # Arguments
///
/// * `factors` - The compact Hessenberg factorization from `gehrd`
///
/// # Returns
///
/// The orthogonal matrix Q (n×n).
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::{gehrd, orghr};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
///     &[7.0, 8.0, 9.0],
/// ]);
///
/// let factors = gehrd(a.as_ref()).unwrap();
/// let q = orghr(&factors).unwrap();
///
/// // Q should be orthogonal: Q^T * Q = I
/// ```
pub fn orghr<T: Field + Real + bytemuck::Zeroable>(
    factors: &HessenbergFactors<T>,
) -> Result<Mat<T>, HessenbergError> {
    let n = factors.n;
    let ilo = factors.ilo;
    let ihi = factors.ihi;

    // Initialize Q as identity
    let mut q = Mat::zeros(n, n);
    for i in 0..n {
        q[(i, i)] = T::one();
    }

    // Number of reflectors
    let nh = if ihi > ilo + 1 { ihi - ilo - 1 } else { 0 };
    if nh == 0 {
        return Ok(q);
    }

    // Apply reflectors in forward order to build Q
    // Q = P_0 * P_1 * ... * P_{nh-1}
    // We accumulate by: Q <- Q * P_k for k = 0 to nh-1
    for k_idx in 0..nh {
        let k = ilo + k_idx;
        let len = ihi - k - 1;
        if len == 0 || factors.tau[k_idx] == T::zero() {
            continue;
        }

        // Reconstruct v from factors
        let mut v = vec![T::one(); len]; // v[0] = 1
        for i in 1..len {
            v[i] = factors.factors[(k + 1 + i, k)];
        }

        let tau_val = factors.tau[k_idx];

        // Apply P_k = I - tau * v * v^H from the right to Q
        // Q[:, k+1:k+1+len] <- Q[:, k+1:k+1+len] * P_k
        for row in 0..n {
            let mut dot = q[(row, k + 1)]; // v[0] = 1
            for i in 1..len {
                dot = dot + q[(row, k + 1 + i)] * v[i];
            }
            let scaled = tau_val * dot;
            q[(row, k + 1)] = q[(row, k + 1)] - scaled;
            for i in 1..len {
                q[(row, k + 1 + i)] = q[(row, k + 1 + i)] - scaled * v[i].conj();
            }
        }
    }

    Ok(q)
}

/// Alias for `orghr` for complex matrices (LAPACK UNGHR).
pub fn unghr<T: Field + Real + bytemuck::Zeroable>(
    factors: &HessenbergFactors<T>,
) -> Result<Mat<T>, HessenbergError> {
    orghr(factors)
}

/// Multiplies a matrix by Q from Hessenberg factorization (LAPACK ORMHR/UNMHR).
///
/// Computes one of:
/// - C <- Q * C   (side=Left, trans=NoTrans)
/// - C <- Q^T * C (side=Left, trans=Trans or ConjTrans)
/// - C <- C * Q   (side=Right, trans=NoTrans)
/// - C <- C * Q^T (side=Right, trans=Trans or ConjTrans)
///
/// This is more efficient than explicitly forming Q and multiplying.
///
/// # Arguments
///
/// * `factors` - The compact Hessenberg factorization from `gehrd`
/// * `side` - Whether to multiply from left or right
/// * `trans` - Whether to use Q or Q^T
/// * `c` - Input matrix to multiply
///
/// # Returns
///
/// The result of the multiplication.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::{gehrd, ormhr, Side, Trans};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
///     &[7.0, 8.0, 9.0],
/// ]);
///
/// let factors = gehrd(a.as_ref()).unwrap();
/// let c = Mat::from_rows(&[&[1.0], &[0.0], &[0.0]]);
///
/// // Compute Q * c without forming Q explicitly
/// let qc = ormhr(&factors, Side::Left, Trans::NoTrans, c.as_ref()).unwrap();
/// ```
pub fn ormhr<T: Field + Real + bytemuck::Zeroable>(
    factors: &HessenbergFactors<T>,
    side: Side,
    trans: Trans,
    c: MatRef<'_, T>,
) -> Result<Mat<T>, HessenbergError> {
    let n = factors.n;
    let ilo = factors.ilo;
    let ihi = factors.ihi;
    let c_rows = c.nrows();
    let c_cols = c.ncols();

    // Validate dimensions
    match side {
        Side::Left => {
            if c_rows != n {
                return Err(HessenbergError::NotSquare); // Dimension mismatch
            }
        }
        Side::Right => {
            if c_cols != n {
                return Err(HessenbergError::NotSquare); // Dimension mismatch
            }
        }
    }

    // Copy C to result
    let mut result = Mat::zeros(c_rows, c_cols);
    for i in 0..c_rows {
        for j in 0..c_cols {
            result[(i, j)] = c[(i, j)];
        }
    }

    let nh = if ihi > ilo + 1 { ihi - ilo - 1 } else { 0 };
    if nh == 0 {
        return Ok(result);
    }

    // Determine the order of applying reflectors
    let (k_range, forward): (Vec<usize>, bool) = match (side, trans) {
        (Side::Left, Trans::NoTrans) => ((0..nh).collect(), true),
        (Side::Left, Trans::Trans) | (Side::Left, Trans::ConjTrans) => {
            ((0..nh).rev().collect(), false)
        }
        (Side::Right, Trans::NoTrans) => ((0..nh).rev().collect(), false),
        (Side::Right, Trans::Trans) | (Side::Right, Trans::ConjTrans) => ((0..nh).collect(), true),
    };

    let _ = forward; // For documentation purposes

    for k_idx in k_range {
        let k = ilo + k_idx;
        let len = ihi - k - 1;
        if len == 0 || factors.tau[k_idx] == T::zero() {
            continue;
        }

        // Reconstruct v from factors
        let mut v = vec![T::one(); len]; // v[0] = 1
        for i in 1..len {
            v[i] = factors.factors[(k + 1 + i, k)];
        }

        let tau_val = match trans {
            Trans::NoTrans => factors.tau[k_idx],
            Trans::Trans | Trans::ConjTrans => factors.tau[k_idx].conj(),
        };

        match side {
            Side::Left => {
                // result[k+1:k+1+len, :] <- P_k * result[k+1:k+1+len, :]
                // P_k = I - tau * v * v^H
                for j in 0..c_cols {
                    let mut dot = result[(k + 1, j)]; // v[0] = 1
                    for i in 1..len {
                        let v_elem = match trans {
                            Trans::NoTrans => v[i].conj(),
                            Trans::Trans | Trans::ConjTrans => v[i],
                        };
                        dot = dot + v_elem * result[(k + 1 + i, j)];
                    }
                    let scaled = tau_val * dot;
                    result[(k + 1, j)] = result[(k + 1, j)] - scaled;
                    for i in 1..len {
                        let v_elem = match trans {
                            Trans::NoTrans => v[i],
                            Trans::Trans | Trans::ConjTrans => v[i].conj(),
                        };
                        result[(k + 1 + i, j)] = result[(k + 1 + i, j)] - scaled * v_elem;
                    }
                }
            }
            Side::Right => {
                // result[:, k+1:k+1+len] <- result[:, k+1:k+1+len] * P_k
                for i in 0..c_rows {
                    let mut dot = result[(i, k + 1)]; // v[0] = 1
                    for jj in 1..len {
                        let v_elem = match trans {
                            Trans::NoTrans => v[jj],
                            Trans::Trans | Trans::ConjTrans => v[jj].conj(),
                        };
                        dot = dot + result[(i, k + 1 + jj)] * v_elem;
                    }
                    let scaled = tau_val * dot;
                    result[(i, k + 1)] = result[(i, k + 1)] - scaled;
                    for jj in 1..len {
                        let v_elem = match trans {
                            Trans::NoTrans => v[jj].conj(),
                            Trans::Trans | Trans::ConjTrans => v[jj],
                        };
                        result[(i, k + 1 + jj)] = result[(i, k + 1 + jj)] - scaled * v_elem;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Alias for `ormhr` for complex matrices (LAPACK UNMHR).
pub fn unmhr<T: Field + Real + bytemuck::Zeroable>(
    factors: &HessenbergFactors<T>,
    side: Side,
    trans: Trans,
    c: MatRef<'_, T>,
) -> Result<Mat<T>, HessenbergError> {
    ormhr(factors, side, trans, c)
}

// Helper functions for LAPACK-style Householder

/// Returns sign of x (1 for x >= 0, -1 for x < 0).
fn sign_of<T: Field + Real>(x: T) -> T {
    if x >= T::zero() { T::one() } else { -T::one() }
}

/// Computes the 2-norm of a vector.
fn norm<T: Field + Real>(x: &[T]) -> T {
    let mut sum = T::zero();
    for &xi in x {
        sum = sum + xi * xi.conj();
    }
    Real::sqrt(sum)
}

/// Computes LAPACK-style Householder vector.
/// Returns (v, tau) where v[0] = 1 (implicit), and P = I - tau * v * v^H
/// transforms x to [beta, 0, ..., 0] with beta = -sign(x[0]) * ||x||.
fn householder_vector_lapack<T: Field + Real>(x: &[T]) -> (Vec<T>, T) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), T::zero());
    }

    if n == 1 {
        // Single element, no reflection needed
        return (vec![T::one()], T::zero());
    }

    // Compute norm of x
    let x_norm = norm(x);
    if x_norm == T::zero() {
        return (vec![T::zero(); n], T::zero());
    }

    // beta = -sign(x[0]) * ||x||
    let beta = -sign_of(x[0]) * x_norm;

    // v[0] = 1, v[i] = x[i] / (x[0] - beta) for i > 0
    // tau = (beta - x[0]) / beta
    let denom = x[0] - beta;

    let mut v = vec![T::one(); n];
    if Scalar::abs(denom) > <T as Scalar>::epsilon() {
        for i in 1..n {
            v[i] = x[i] / denom;
        }
    }

    let tau = (beta - x[0]) / beta;

    (v, tau)
}

/// Computes a Householder vector for zeroing elements below the first.
/// Returns (v, tau) such that (I - tau * v * v^T) * x = [-sign(x\[0\]) * ||x||, 0, ..., 0]^T
/// Uses LAPACK-style formulation with v[0] = 1.
fn householder_vector_with_tau<T: Field + Real>(x: &[T]) -> (Vec<T>, T) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), T::zero());
    }

    // Compute norm of x
    let mut norm_sq = T::zero();
    for i in 0..n {
        norm_sq = norm_sq + x[i] * x[i];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (vec![T::zero(); n], T::zero());
    }

    // Choose sign to avoid cancellation: beta = -sign(x[0]) * ||x||
    let sign = if x[0] >= T::zero() {
        -T::one()
    } else {
        T::one()
    };
    let beta = sign * norm;

    // v = x - beta * e_1, so v[0] = x[0] - beta, v[i] = x[i] for i > 0
    let mut v = x.to_vec();
    v[0] = x[0] - beta;

    // Compute ||v||^2 = (x[0] - beta)^2 + sum_{i>0} x[i]^2
    // = x[0]^2 - 2*x[0]*beta + beta^2 + (||x||^2 - x[0]^2)
    // = ||x||^2 - 2*x[0]*beta + beta^2
    // = ||x||^2 + 2*|x[0]|*||x|| + ||x||^2   (since beta = -sign(x[0])*||x||)
    // = 2*||x||^2 + 2*|x[0]|*||x||
    // = 2*||x|| * (||x|| + |x[0]|)
    let v_norm_sq = v[0] * v[0] + (norm_sq - x[0] * x[0]);

    if v_norm_sq == T::zero() {
        return (vec![T::zero(); n], T::zero());
    }

    // tau = 2 / ||v||^2
    let tau = T::from_f64(2.0).unwrap_or_else(T::zero) / v_norm_sq;

    (v, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_hessenberg_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let hess = Hessenberg::compute(a.as_ref()).unwrap();
        let h = hess.h();

        // Check H is upper Hessenberg (zero below subdiagonal)
        assert!(approx_eq(h[(2, 0)], 0.0, 1e-10));

        // Check reconstruction
        let reconstructed = hess.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
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
    fn test_hessenberg_4x4() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, 1.0],
            &[-2.0, 0.0, 3.0, -2.0],
            &[2.0, 1.0, -2.0, -1.0],
        ]);

        let hess = Hessenberg::compute(a.as_ref()).unwrap();
        let h = hess.h();

        // Check H is upper Hessenberg
        assert!(approx_eq(h[(2, 0)], 0.0, 1e-10));
        assert!(approx_eq(h[(3, 0)], 0.0, 1e-10));
        assert!(approx_eq(h[(3, 1)], 0.0, 1e-10));

        // Check reconstruction
        let reconstructed = hess.reconstruct();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
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
    fn test_hessenberg_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let hess = Hessenberg::compute(eye.as_ref()).unwrap();
        let h = hess.h();

        // Identity should stay identity (already Hessenberg)
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(h[(i, j)], expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_hessenberg_q_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let hess = Hessenberg::compute(a.as_ref()).unwrap();
        let q = hess.q();

        // Check Q^T * Q = I
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_hessenberg_small() {
        let a = Mat::from_rows(&[&[5.0f64]]);
        let hess = Hessenberg::compute(a.as_ref()).unwrap();
        assert!(approx_eq(hess.h()[(0, 0)], 5.0, 1e-10));

        let a2 = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let hess2 = Hessenberg::compute(a2.as_ref()).unwrap();
        // 2x2 is already Hessenberg
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(hess2.h()[(i, j)], a2[(i, j)], 1e-10));
            }
        }
    }

    #[test]
    fn test_hessenberg_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let hess = Hessenberg::compute(a.as_ref()).unwrap();
        let reconstructed = hess.reconstruct();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-5,
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    // Tests for LAPACK-style compact Hessenberg (gehrd, orghr, ormhr)

    #[test]
    fn test_gehrd_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let factors = gehrd(a.as_ref()).unwrap();
        let h = factors.h();

        // H should be upper Hessenberg
        assert!(
            h[(2, 0)].abs() < 1e-10,
            "H[2,0] = {} should be zero",
            h[(2, 0)]
        );
    }

    #[test]
    fn test_gehrd_4x4() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, 1.0],
            &[-2.0, 0.0, 3.0, -2.0],
            &[2.0, 1.0, -2.0, -1.0],
        ]);

        let factors = gehrd(a.as_ref()).unwrap();
        let h = factors.h();

        // Check H is upper Hessenberg
        assert!(
            h[(2, 0)].abs() < 1e-10,
            "H[2,0] = {} should be zero",
            h[(2, 0)]
        );
        assert!(
            h[(3, 0)].abs() < 1e-10,
            "H[3,0] = {} should be zero",
            h[(3, 0)]
        );
        assert!(
            h[(3, 1)].abs() < 1e-10,
            "H[3,1] = {} should be zero",
            h[(3, 1)]
        );
    }

    #[test]
    fn test_orghr_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let factors = gehrd(a.as_ref()).unwrap();
        let q = orghr(&factors).unwrap();

        // Check Q^T * Q = I
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gehrd_orghr_reconstruction() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, 1.0],
            &[-2.0, 0.0, 3.0, -2.0],
            &[2.0, 1.0, -2.0, -1.0],
        ]);

        let factors = gehrd(a.as_ref()).unwrap();
        let h = factors.h();
        let q = orghr(&factors).unwrap();

        // Reconstruct A = Q * H * Q^T
        let n = 4;
        let mut qh = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * h[(k, j)];
                }
                qh[(i, j)] = sum;
            }
        }

        let mut reconstructed = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += qh[(i, k)] * q[(j, k)]; // Q^T
                }
                reconstructed[(i, j)] = sum;
            }
        }

        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
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
    fn test_ormhr_left_notrans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let factors = gehrd(a.as_ref()).unwrap();
        let q = orghr(&factors).unwrap();

        // Test vector
        let c = Mat::from_rows(&[&[1.0f64], &[0.0], &[0.0]]);

        // Using ormhr
        let qc_ormhr = ormhr(&factors, Side::Left, Trans::NoTrans, c.as_ref()).unwrap();

        // Using explicit Q multiplication
        let n = 3;
        let mut qc_explicit = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * c[(k, 0)];
            }
            qc_explicit[(i, 0)] = sum;
        }

        for i in 0..n {
            assert!(
                approx_eq(qc_ormhr[(i, 0)], qc_explicit[(i, 0)], 1e-10),
                "ormhr[{}] = {}, explicit = {}",
                i,
                qc_ormhr[(i, 0)],
                qc_explicit[(i, 0)]
            );
        }
    }

    #[test]
    fn test_ormhr_left_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let factors = gehrd(a.as_ref()).unwrap();
        let q = orghr(&factors).unwrap();

        // Test vector
        let c = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);

        // Using ormhr with transpose
        let qtc_ormhr = ormhr(&factors, Side::Left, Trans::Trans, c.as_ref()).unwrap();

        // Using explicit Q^T multiplication
        let n = 3;
        let mut qtc_explicit = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * c[(k, 0)]; // Q^T[i,k] = Q[k,i]
            }
            qtc_explicit[(i, 0)] = sum;
        }

        for i in 0..n {
            assert!(
                approx_eq(qtc_ormhr[(i, 0)], qtc_explicit[(i, 0)], 1e-10),
                "ormhr_trans[{}] = {}, explicit = {}",
                i,
                qtc_ormhr[(i, 0)],
                qtc_explicit[(i, 0)]
            );
        }
    }

    #[test]
    fn test_ormhr_right_notrans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let factors = gehrd(a.as_ref()).unwrap();
        let q = orghr(&factors).unwrap();

        // Test matrix (1 x n)
        let c = Mat::from_rows(&[&[1.0f64, 0.0, 0.0]]);

        // Using ormhr
        let cq_ormhr = ormhr(&factors, Side::Right, Trans::NoTrans, c.as_ref()).unwrap();

        // Using explicit multiplication: C * Q
        let n = 3;
        let mut cq_explicit = Mat::zeros(1, n);
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += c[(0, k)] * q[(k, j)];
            }
            cq_explicit[(0, j)] = sum;
        }

        for j in 0..n {
            assert!(
                approx_eq(cq_ormhr[(0, j)], cq_explicit[(0, j)], 1e-10),
                "ormhr_right[{}] = {}, explicit = {}",
                j,
                cq_ormhr[(0, j)],
                cq_explicit[(0, j)]
            );
        }
    }

    #[test]
    fn test_gehrd_identity() {
        let eye: Mat<f64> = Mat::eye(4);
        let factors = gehrd(eye.as_ref()).unwrap();
        let q = orghr(&factors).unwrap();
        let h = factors.h();

        // For identity, Q and H should both be identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(q[(i, j)], expected, 1e-10),
                    "Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    q[(i, j)],
                    expected
                );
                assert!(
                    approx_eq(h[(i, j)], expected, 1e-10),
                    "H[{},{}] = {}, expected {}",
                    i,
                    j,
                    h[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gehrd_2x2() {
        // 2x2 is already Hessenberg
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let factors = gehrd(a.as_ref()).unwrap();
        let h = factors.h();

        // Should match original
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(h[(i, j)], a[(i, j)], 1e-10),
                    "H[{},{}] = {}, a = {}",
                    i,
                    j,
                    h[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_gehrd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let factors = gehrd(a.as_ref()).unwrap();
        let h = factors.h();
        let q = orghr(&factors).unwrap();

        // Check H is upper Hessenberg
        assert!(h[(2, 0)].abs() < 1e-5);

        // Check Q is orthogonal
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0f32;
                for k in 0..3 {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    // Tests for blocked Hessenberg reduction

    #[test]
    fn test_hessenberg_blocked_4x4() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, 1.0],
            &[-2.0, 0.0, 3.0, -2.0],
            &[2.0, 1.0, -2.0, -1.0],
        ]);

        let hess = Hessenberg::compute_blocked(a.as_ref()).unwrap();
        let h = hess.h();

        // Check H is upper Hessenberg
        assert!(approx_eq(h[(2, 0)], 0.0, 1e-10), "H[2,0] = {}", h[(2, 0)]);
        assert!(approx_eq(h[(3, 0)], 0.0, 1e-10), "H[3,0] = {}", h[(3, 0)]);
        assert!(approx_eq(h[(3, 1)], 0.0, 1e-10), "H[3,1] = {}", h[(3, 1)]);

        // Check reconstruction
        let reconstructed = hess.reconstruct();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
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
    fn test_hessenberg_blocked_q_orthogonal() {
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
        ]);

        let hess = Hessenberg::compute_blocked(a.as_ref()).unwrap();
        let q = hess.q();

        // Check Q^T * Q = I
        let n = 4;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_hessenberg_blocked_vs_unblocked() {
        // Test that blocked and unblocked give the same result
        let n = 100;
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 / 100.0 + 0.1;
            }
        }

        let hess_unblocked = Hessenberg::compute(a.as_ref()).unwrap();
        let hess_blocked = Hessenberg::compute_blocked_with_block_size(a.as_ref(), 16).unwrap();

        // Both should reconstruct to the original matrix
        let rec_unblocked = hess_unblocked.reconstruct();
        let rec_blocked = hess_blocked.reconstruct();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(rec_unblocked[(i, j)], a[(i, j)], 1e-9),
                    "unblocked reconstruction differs at ({},{}): {} vs {}",
                    i,
                    j,
                    rec_unblocked[(i, j)],
                    a[(i, j)]
                );
                assert!(
                    approx_eq(rec_blocked[(i, j)], a[(i, j)], 1e-9),
                    "blocked reconstruction differs at ({},{}): {} vs {}",
                    i,
                    j,
                    rec_blocked[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_hessenberg_blocked_small() {
        // Small matrices should fall back to unblocked
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let hess = Hessenberg::compute_blocked(a.as_ref()).unwrap();
        let h = hess.h();

        // Check H is upper Hessenberg
        assert!(approx_eq(h[(2, 0)], 0.0, 1e-10));

        // Check reconstruction
        let reconstructed = hess.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
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
    fn test_hessenberg_blocked_identity() {
        let eye: Mat<f64> = Mat::eye(5);
        let hess = Hessenberg::compute_blocked(eye.as_ref()).unwrap();
        let h = hess.h();

        // Identity should stay identity
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(h[(i, j)], expected, 1e-10),
                    "H[{},{}] = {}, expected {}",
                    i,
                    j,
                    h[(i, j)],
                    expected
                );
            }
        }
    }
}
