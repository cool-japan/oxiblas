//! Complex Hessenberg reduction.
//!
//! Reduces a general complex matrix A to upper Hessenberg form: A = Q H Q^H
//! where Q is unitary and H is upper Hessenberg (zeros below subdiagonal).
//!
//! This module provides:
//!
//! - **ComplexHessenberg**: Full Hessenberg decomposition with explicit Q
//! - **ComplexHessenbergFactors**: LAPACK-style compact storage (zgehrd)
//! - **zgehrd**: Reduce to Hessenberg form with compact storage
//! - **zunhhr**: Generate Q from compact Hessenberg factorization (unghr for complex)
//! - **zunmhr**: Apply Q to a matrix without forming it explicitly (unmhr for complex)
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::{ComplexHessenberg, zgehrd, zunhhr, zunmhr, Side, Trans};
//! use oxiblas_matrix::Mat;
//! use num_complex::Complex64;
//!
//! let a = Mat::from_rows(&[
//!     &[Complex64::new(4.0, 1.0), Complex64::new(1.0, 0.0), Complex64::new(-2.0, 1.0)],
//!     &[Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0), Complex64::new(0.0, 0.0)],
//!     &[Complex64::new(-2.0, -1.0), Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
//! ]);
//!
//! let hess = ComplexHessenberg::compute(a.as_ref()).unwrap();
//! let h = hess.h();
//! let q = hess.q();
//!
//! // H is upper Hessenberg (zeros below subdiagonal)
//! assert!(h[(2, 0)].norm() < 1e-10);
//! ```

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::hessenberg::{Side, Trans};

/// Error type for complex Hessenberg reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexHessenbergError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
}

impl core::fmt::Display for ComplexHessenbergError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for ComplexHessenbergError {}

/// Upper Hessenberg form of a complex matrix.
///
/// For a complex matrix A, computes A = Q H Q^H where:
/// - Q is unitary (Q^H Q = I)
/// - H is upper Hessenberg (h_ij = 0 for i > j + 1)
#[derive(Debug, Clone)]
pub struct ComplexHessenberg<T: Scalar> {
    /// The unitary matrix Q.
    q: Mat<T>,
    /// The upper Hessenberg matrix H.
    h: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexHessenberg<T>
where
    T::Real: Real,
{
    /// Reduces a square complex matrix to upper Hessenberg form.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::ComplexHessenberg;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0), Complex64::new(3.0, 0.0)],
    ///     &[Complex64::new(4.0, -1.0), Complex64::new(5.0, 0.0), Complex64::new(6.0, 1.0)],
    ///     &[Complex64::new(7.0, 0.0), Complex64::new(8.0, -1.0), Complex64::new(9.0, 0.0)],
    /// ]);
    ///
    /// let hess = ComplexHessenberg::compute(a.as_ref()).unwrap();
    /// let h = hess.h();
    ///
    /// // H is upper Hessenberg (zeros below subdiagonal)
    /// assert!(h[(2, 0)].norm() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ComplexHessenbergError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexHessenbergError::EmptyMatrix);
        }

        if m != n {
            return Err(ComplexHessenbergError::NotSquare);
        }

        if n == 1 {
            let mut h: Mat<T> = Mat::zeros(1, 1);
            h[(0, 0)] = a[(0, 0)];
            let mut q: Mat<T> = Mat::zeros(1, 1);
            q[(0, 0)] = T::one();
            return Ok(Self { q, h, n });
        }

        if n == 2 {
            let mut h: Mat<T> = Mat::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    h[(i, j)] = a[(i, j)];
                }
            }
            let mut q: Mat<T> = Mat::zeros(2, 2);
            q[(0, 0)] = T::one();
            q[(1, 1)] = T::one();
            return Ok(Self { q, h, n });
        }

        // Copy A to H
        let mut h: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = a[(i, j)];
            }
        }

        // Initialize Q as identity
        let mut q: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        // Reduce to Hessenberg form using Householder reflections
        for k in 0..(n - 2) {
            // Create Householder reflection to zero H[k+2:n, k]
            let m_size = n - k - 1;
            let mut x: Vec<T> = vec![T::zero(); m_size];
            for i in 0..m_size {
                x[i] = h[(k + 1 + i, k)];
            }

            let (v, tau) = complex_householder_vector(&x);

            if tau.abs() > T::Real::epsilon() {
                // Apply P from the left: H := P * H where P = I - tau*v*v^H
                for j in k..n {
                    let mut dot = T::zero();
                    for i in 0..v.len() {
                        // dot = v^H * h[:, j] = sum conj(v[i]) * h[k+1+i, j]
                        dot = dot + v[i].conj() * h[(k + 1 + i, j)];
                    }
                    let scaled = tau * dot;
                    for i in 0..v.len() {
                        h[(k + 1 + i, j)] = h[(k + 1 + i, j)] - scaled * v[i];
                    }
                }

                // Apply P from the right: H := H * P^H = H * (I - conj(tau)*v*v^H)
                // Since tau is real for standard Householder, P^H = P
                for i in 0..n {
                    let mut dot = T::zero();
                    for j in 0..v.len() {
                        // dot = h[i, :] * v = sum h[i, k+1+j] * v[j]
                        dot = dot + h[(i, k + 1 + j)] * v[j];
                    }
                    let scaled = tau * dot;
                    for j in 0..v.len() {
                        h[(i, k + 1 + j)] = h[(i, k + 1 + j)] - scaled * v[j].conj();
                    }
                }

                // Accumulate Q: Q := Q * P
                for i in 0..n {
                    let mut dot = T::zero();
                    for j in 0..v.len() {
                        dot = dot + q[(i, k + 1 + j)] * v[j];
                    }
                    let scaled = tau * dot;
                    for j in 0..v.len() {
                        q[(i, k + 1 + j)] = q[(i, k + 1 + j)] - scaled * v[j].conj();
                    }
                }
            }
        }

        // Clean up small values below subdiagonal
        let hundred: T::Real =
            <T::Real as FromPrimitive>::from_f64(100.0).unwrap_or(<T::Real as One>::one());
        let eps = <T::Real as Scalar>::epsilon() * hundred;
        for j in 0..(n - 2) {
            for i in (j + 2)..n {
                if h[(i, j)].abs() < eps {
                    h[(i, j)] = T::zero();
                }
            }
        }

        Ok(Self { q, h, n })
    }

    /// Returns the unitary matrix Q.
    ///
    /// Q satisfies Q^H Q = I and A = Q H Q^H.
    pub fn q(&self) -> MatRef<'_, T> {
        self.q.as_ref()
    }

    /// Returns the upper Hessenberg matrix H.
    pub fn h(&self) -> MatRef<'_, T> {
        self.h.as_ref()
    }

    /// Returns the matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = Q H Q^H.
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a: Mat<T> = Mat::zeros(n, n);

        // First compute H * Q^H
        let mut hqh: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    // H * Q^H = H * conj(Q)^T
                    sum = sum + self.h[(i, k)] * self.q[(j, k)].conj();
                }
                hqh[(i, j)] = sum;
            }
        }

        // Then compute Q * (H * Q^H)
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + self.q[(i, k)] * hqh[(k, j)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }
}

/// Compact storage for complex Hessenberg factorization (LAPACK zgehrd style).
#[derive(Debug, Clone)]
pub struct ComplexHessenbergFactors<T: Scalar> {
    /// Upper Hessenberg matrix with Householder vectors stored below subdiagonal.
    qr: Mat<T>,
    /// Householder scalars (tau values).
    tau: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Lower index for reduction range.
    ilo: usize,
    /// Upper index for reduction range.
    ihi: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexHessenbergFactors<T>
where
    T::Real: Real,
{
    /// Returns the matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the lower index of the reduction range.
    pub fn ilo(&self) -> usize {
        self.ilo
    }

    /// Returns the upper index of the reduction range.
    pub fn ihi(&self) -> usize {
        self.ihi
    }

    /// Returns the tau values.
    pub fn tau(&self) -> &[T] {
        &self.tau
    }

    /// Extracts the upper Hessenberg matrix H.
    pub fn h(&self) -> Mat<T> {
        let n = self.n;
        let mut h: Mat<T> = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if i <= j + 1 {
                    h[(i, j)] = self.qr[(i, j)];
                }
            }
        }

        h
    }

    /// Generates the unitary matrix Q explicitly.
    ///
    /// This is equivalent to LAPACK's ZUNGHR.
    pub fn q(&self) -> Mat<T> {
        let n = self.n;
        let mut q: Mat<T> = Mat::zeros(n, n);

        // Initialize as identity
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        if n <= 2 {
            return q;
        }

        // Apply Householder reflections in reverse order
        // k ranges from ilo to ihi-2 (same as in zgehrd_range)
        for k in (self.ilo..(self.ihi.saturating_sub(1))).rev() {
            let tau = self.tau[k - self.ilo];
            if tau.abs() < T::Real::epsilon() {
                continue;
            }

            // Extract Householder vector from column k
            // The Householder vector has m_size elements:
            // - v[0] = 1 (implicit)
            // - v[1:m_size-1] stored in qr[k+2:ihi-1, k]
            let m_size = self.ihi - k - 1;
            if m_size == 0 {
                continue;
            }
            let mut v: Vec<T> = vec![T::zero(); m_size];
            v[0] = T::one();
            for i in 1..m_size {
                v[i] = self.qr[(k + 1 + i, k)];
            }

            // Apply P = I - tau * v * v^H to Q from the right
            for i in 0..n {
                let mut dot = T::zero();
                for j in 0..v.len() {
                    dot = dot + q[(i, k + 1 + j)] * v[j];
                }
                let scaled = tau * dot;
                for j in 0..v.len() {
                    q[(i, k + 1 + j)] = q[(i, k + 1 + j)] - scaled * v[j].conj();
                }
            }
        }

        q
    }

    /// Applies the unitary matrix Q to a matrix C.
    ///
    /// This is equivalent to LAPACK's ZUNMHR.
    ///
    /// - `Side::Left`: Computes Q * C or Q^H * C
    /// - `Side::Right`: Computes C * Q or C * Q^H
    pub fn apply(
        &self,
        side: Side,
        trans: Trans,
        c: MatRef<'_, T>,
    ) -> Result<Mat<T>, ComplexHessenbergError> {
        let n = self.n;
        let m = c.nrows();
        let k = c.ncols();

        // Check dimensions
        match side {
            Side::Left => {
                if m != n {
                    return Err(ComplexHessenbergError::DimensionMismatch);
                }
            }
            Side::Right => {
                if k != n {
                    return Err(ComplexHessenbergError::DimensionMismatch);
                }
            }
        }

        // Copy C to result
        let mut result: Mat<T> = Mat::zeros(m, k);
        for i in 0..m {
            for j in 0..k {
                result[(i, j)] = c[(i, j)];
            }
        }

        if n <= 2 {
            return Ok(result);
        }

        // Determine iteration order based on side and trans
        let apply_conj = matches!(trans, Trans::ConjTrans);
        let forward = matches!(
            (side, trans),
            (Side::Left, Trans::ConjTrans) | (Side::Right, Trans::NoTrans)
        );

        let range: Vec<usize> = if forward {
            (self.ilo..(self.ihi.saturating_sub(1))).collect()
        } else {
            (self.ilo..(self.ihi.saturating_sub(1))).rev().collect()
        };

        for &idx in &range {
            let tau = if apply_conj {
                self.tau[idx].conj()
            } else {
                self.tau[idx]
            };

            if tau.abs() < T::Real::epsilon() {
                continue;
            }

            let m_size = self.ihi - idx - 1;
            let mut v: Vec<T> = vec![T::zero(); m_size + 1];
            v[0] = T::one();
            for i in 1..=m_size {
                v[i] = self.qr[(idx + 1 + i, idx)];
            }

            match side {
                Side::Left => {
                    // Apply P from left: C := P * C = C - tau * v * (v^H * C)
                    for j in 0..k {
                        let mut dot = T::zero();
                        for i in 0..v.len() {
                            dot = dot + v[i].conj() * result[(idx + 1 + i, j)];
                        }
                        let scaled = tau * dot;
                        for i in 0..v.len() {
                            result[(idx + 1 + i, j)] = result[(idx + 1 + i, j)] - scaled * v[i];
                        }
                    }
                }
                Side::Right => {
                    // Apply P from right: C := C * P = C - tau * (C * v) * v^H
                    for i in 0..m {
                        let mut dot = T::zero();
                        for j in 0..v.len() {
                            dot = dot + result[(i, idx + 1 + j)] * v[j];
                        }
                        let scaled = tau * dot;
                        for j in 0..v.len() {
                            result[(i, idx + 1 + j)] =
                                result[(i, idx + 1 + j)] - scaled * v[j].conj();
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Reduces a complex matrix to upper Hessenberg form (LAPACK zgehrd).
///
/// Returns the factorization in compact form with Householder vectors
/// stored below the subdiagonal.
///
/// # Arguments
///
/// * `a` - Square complex matrix to reduce
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::{zgehrd, zunhhr};
/// use oxiblas_matrix::Mat;
/// use num_complex::Complex64;
///
/// let a = Mat::from_rows(&[
///     &[Complex64::new(4.0, 0.0), Complex64::new(1.0, 1.0), Complex64::new(-2.0, 0.0)],
///     &[Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
///     &[Complex64::new(-2.0, 0.0), Complex64::new(0.0, -1.0), Complex64::new(3.0, 0.0)],
/// ]);
///
/// let factors = zgehrd(a.as_ref()).unwrap();
/// let h = factors.h();
///
/// // H is upper Hessenberg
/// assert!(h[(2, 0)].norm() < 1e-10);
/// ```
pub fn zgehrd<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<ComplexHessenbergFactors<T>, ComplexHessenbergError>
where
    T::Real: Real,
{
    zgehrd_range(a, 0, a.nrows())
}

/// Reduces a complex matrix to upper Hessenberg form over a specified range.
///
/// This is useful when the matrix has been balanced and only a submatrix
/// needs to be reduced.
///
/// # Arguments
///
/// * `a` - Square complex matrix to reduce
/// * `ilo` - Lower index of the range (0-based)
/// * `ihi` - Upper index of the range (exclusive, 0-based)
pub fn zgehrd_range<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    ilo: usize,
    ihi: usize,
) -> Result<ComplexHessenbergFactors<T>, ComplexHessenbergError>
where
    T::Real: Real,
{
    let n = a.nrows();

    if n == 0 {
        return Err(ComplexHessenbergError::EmptyMatrix);
    }

    if n != a.ncols() {
        return Err(ComplexHessenbergError::NotSquare);
    }

    // Copy A to QR storage
    let mut qr: Mat<T> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            qr[(i, j)] = a[(i, j)];
        }
    }

    let tau_len = if ihi > ilo + 1 { ihi - ilo - 1 } else { 0 };
    let mut tau: Vec<T> = vec![T::zero(); tau_len];

    // Reduce columns ilo to ihi-2
    for k in ilo..(ihi.saturating_sub(1)) {
        let m_size = ihi - k - 1;
        let mut x: Vec<T> = vec![T::zero(); m_size];
        for i in 0..m_size {
            x[i] = qr[(k + 1 + i, k)];
        }

        let (v, tau_k) = complex_householder_vector(&x);
        tau[k - ilo] = tau_k;

        if tau_k.abs() > T::Real::epsilon() {
            // Store Householder vector in lower part of column k
            for i in 1..v.len() {
                qr[(k + 1 + i, k)] = v[i];
            }

            // Apply P from the left to columns k to n-1
            for j in (k + 1)..n {
                let mut dot = T::zero();
                dot = dot + qr[(k + 1, j)]; // v[0] = 1
                for i in 1..v.len() {
                    dot = dot + v[i].conj() * qr[(k + 1 + i, j)];
                }
                let scaled = tau_k * dot;
                qr[(k + 1, j)] = qr[(k + 1, j)] - scaled;
                for i in 1..v.len() {
                    qr[(k + 1 + i, j)] = qr[(k + 1 + i, j)] - scaled * v[i];
                }
            }

            // Apply P from the right to rows 0 to ihi-1
            for i in 0..ihi {
                let mut dot = T::zero();
                dot = dot + qr[(i, k + 1)]; // v[0] = 1
                for j in 1..v.len() {
                    dot = dot + qr[(i, k + 1 + j)] * v[j];
                }
                let scaled = tau_k * dot;
                qr[(i, k + 1)] = qr[(i, k + 1)] - scaled;
                for j in 1..v.len() {
                    qr[(i, k + 1 + j)] = qr[(i, k + 1 + j)] - scaled * v[j].conj();
                }
            }
        }

        // Set the subdiagonal element
        qr[(k + 1, k)] = T::from_real(-compute_householder_beta(&x));
    }

    Ok(ComplexHessenbergFactors {
        qr,
        tau,
        n,
        ilo,
        ihi,
    })
}

/// Generates the unitary matrix Q from compact Hessenberg factorization (LAPACK zunghr).
pub fn zunhhr<T: Field + ComplexScalar + bytemuck::Zeroable>(
    factors: &ComplexHessenbergFactors<T>,
) -> Result<Mat<T>, ComplexHessenbergError>
where
    T::Real: Real,
{
    Ok(factors.q())
}

/// Applies the unitary matrix Q from Hessenberg factorization (LAPACK zunmhr).
pub fn zunmhr<T: Field + ComplexScalar + bytemuck::Zeroable>(
    factors: &ComplexHessenbergFactors<T>,
    side: Side,
    trans: Trans,
    c: MatRef<'_, T>,
) -> Result<Mat<T>, ComplexHessenbergError>
where
    T::Real: Real,
{
    factors.apply(side, trans, c)
}

/// Computes the complex Householder vector for zeroing elements.
/// Returns (v, tau) where P = I - tau * v * v^H zeros x[1:] and v[0] = 1 (LAPACK convention).
fn complex_householder_vector<T: Field + ComplexScalar>(x: &[T]) -> (Vec<T>, T)
where
    T::Real: Real,
{
    let n = x.len();
    if n == 0 {
        return (Vec::new(), T::zero());
    }

    // Compute ||x||
    let mut norm_sq = T::Real::zero();
    for i in 0..n {
        norm_sq = norm_sq + x[i].abs_sq();
    }
    let norm = <T::Real as Real>::sqrt(norm_sq);

    if norm < T::Real::epsilon() {
        return (vec![T::zero(); n], T::zero());
    }

    let x0 = x[0];
    let x0_abs = x0.abs();

    // Compute alpha = -sign(x0) * ||x||
    let alpha = if x0_abs > T::Real::epsilon() {
        let sign = T::from_real_imag(x0.real() / x0_abs, x0.imag() / x0_abs);
        T::zero() - sign * T::from_real(norm)
    } else {
        T::from_real(-norm)
    };

    // LAPACK convention: v[0] = 1 (implicit), v[1:] scaled
    // v = [1, x[1:]/(x0 - alpha)]
    let v0_unscaled = x0 - alpha;

    if v0_unscaled.abs() < T::Real::epsilon() {
        return (vec![T::zero(); n], T::zero());
    }

    let mut v: Vec<T> = vec![T::zero(); n];
    v[0] = T::one();
    for i in 1..n {
        v[i] = x[i] / v0_unscaled;
    }

    // tau = (alpha - x0) / alpha = -v0_unscaled / alpha
    let tau = (T::zero() - v0_unscaled) / alpha;

    (v, tau)
}

/// Computes the beta value for Householder transformation.
fn compute_householder_beta<T: Field + ComplexScalar>(x: &[T]) -> T::Real
where
    T::Real: Real,
{
    let mut norm_sq = T::Real::zero();
    for val in x {
        norm_sq = norm_sq + val.abs_sq();
    }
    <T::Real as Real>::sqrt(norm_sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn test_complex_hessenberg_simple() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 0.0),
            ],
            &[
                Complex64::new(4.0, -1.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 1.0),
            ],
            &[
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, -1.0),
                Complex64::new(9.0, 0.0),
            ],
        ]);

        let hess = ComplexHessenberg::compute(a.as_ref()).expect("Should compute");
        let h = hess.h();
        let q = hess.q();

        // H should be upper Hessenberg (zeros below subdiagonal)
        assert!(
            h[(2, 0)].norm() < 1e-10,
            "H[2,0] = {:?} should be zero",
            h[(2, 0)]
        );

        // Q should be unitary: Q^H * Q = I
        let n = q.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + q[(k, i)].conj() * q[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(
                    diff < 1e-10,
                    "Q^H*Q[{},{}] = {:?}, expected {:?}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_complex_hessenberg_reconstruction() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 1.0), Complex64::new(1.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(3.0, -1.0)],
        ]);

        let hess = ComplexHessenberg::compute(a.as_ref()).expect("Should compute");
        let reconstructed = hess.reconstruct();

        // A = Q H Q^H should give back original A
        for i in 0..2 {
            for j in 0..2 {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "Reconstruction[{},{}] = {:?}, A = {:?}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_complex_hessenberg_4x4() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, 1.0),
                Complex64::new(2.0, 0.0),
            ],
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 1.0),
            ],
            &[
                Complex64::new(-2.0, -1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ],
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(-1.0, 1.0),
            ],
        ]);

        let hess = ComplexHessenberg::compute(a.as_ref()).expect("Should compute");
        let h = hess.h();

        // H should be upper Hessenberg
        for j in 0..2 {
            for i in (j + 2)..4 {
                assert!(
                    h[(i, j)].norm() < 1e-10,
                    "H[{},{}] = {:?} should be zero",
                    i,
                    j,
                    h[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_complex_hessenberg_hermitian() {
        // Hermitian matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(0.0, 2.0),
            ],
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            &[
                Complex64::new(0.0, -2.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        ]);

        let hess = ComplexHessenberg::compute(a.as_ref()).expect("Should compute");
        let h = hess.h();

        // H is upper Hessenberg
        assert!(h[(2, 0)].norm() < 1e-10);

        // For Hermitian input, H should be Hermitian tridiagonal
        // Check that H is nearly Hermitian
        let diff = (h[(1, 0)] - h[(0, 1)].conj()).norm();
        assert!(
            diff < 1e-10,
            "H should be Hermitian tridiagonal: H[1,0]={:?}, H[0,1]={:?}",
            h[(1, 0)],
            h[(0, 1)]
        );
    }

    #[test]
    fn test_complex_hessenberg_f32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 1.0),
                Complex32::new(3.0, 0.0),
            ],
            &[
                Complex32::new(4.0, -1.0),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 1.0),
            ],
            &[
                Complex32::new(7.0, 0.0),
                Complex32::new(8.0, -1.0),
                Complex32::new(9.0, 0.0),
            ],
        ]);

        let hess = ComplexHessenberg::compute(a.as_ref()).expect("Should compute");
        let h = hess.h();

        assert!(h[(2, 0)].norm() < 1e-5);
    }

    #[test]
    fn test_zgehrd() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(-2.0, 0.0),
            ],
            &[
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 1.0),
            ],
            &[
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        ]);

        let factors = zgehrd(a.as_ref()).expect("Should compute");
        let h = factors.h();

        // H is upper Hessenberg
        assert!(h[(2, 0)].norm() < 1e-10);
    }

    #[test]
    fn test_zunhhr() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(-2.0, 0.0),
            ],
            &[
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 1.0),
            ],
            &[
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        ]);

        let factors = zgehrd(a.as_ref()).expect("Should compute");
        let q = zunhhr(&factors).expect("Should generate Q");

        // Q should be unitary
        let n = q.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + q[(k, i)].conj() * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (sum.re - expected).abs() + sum.im.abs();
                assert!(
                    diff < 1e-10,
                    "Q^H*Q[{},{}] = {:?}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_complex_hessenberg_identity() {
        let eye: Mat<Complex64> = Mat::eye(4);

        let hess = ComplexHessenberg::compute(eye.as_ref()).expect("Should compute");
        let h = hess.h();
        let q = hess.q();

        // H should be identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (h[(i, j)] - expected).norm();
                assert!(diff < 1e-10, "H[{},{}] = {:?}", i, j, h[(i, j)]);
            }
        }

        // Q should be identity (or close to it with possible sign flips)
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!(q[(i, j)].norm() > 0.99);
                } else {
                    assert!(q[(i, j)].norm() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_complex_hessenberg_single() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(5.0, 2.0)]]);

        let hess = ComplexHessenberg::compute(a.as_ref()).expect("Should compute");
        let h = hess.h();
        let q = hess.q();

        assert_eq!(h[(0, 0)], Complex64::new(5.0, 2.0));
        assert_eq!(q[(0, 0)], Complex64::new(1.0, 0.0));
    }
}
