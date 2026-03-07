//! Complex general eigenvalue decomposition.
//!
//! Computes eigenvalues and eigenvectors of general complex matrices.
//! For complex matrices, eigenvalues are complex numbers stored directly.

use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for complex general eigenvalue decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexGeneralEvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for ComplexGeneralEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::NotConverged => write!(f, "Eigenvalue algorithm did not converge"),
        }
    }
}

impl std::error::Error for ComplexGeneralEvdError {}

/// Complex general eigenvalue decomposition.
///
/// For a general complex matrix A, computes eigenvalues and optionally eigenvectors.
/// Eigenvalues are directly complex numbers (not paired as in real EVD).
///
/// Right eigenvectors satisfy: A * v = λ * v
/// Left eigenvectors satisfy: u^H * A = λ * u^H
#[derive(Debug, Clone)]
pub struct ComplexGeneralEvd<T: Scalar> {
    /// Complex eigenvalues.
    eigenvalues: Vec<T>,
    /// Right eigenvectors (stored column-wise).
    eigenvectors: Option<Mat<T>>,
    /// Left eigenvectors (stored column-wise).
    left_eigenvectors: Option<Mat<T>>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexGeneralEvd<T>
where
    T::Real: Real,
{
    /// Maximum iterations for QR iteration.
    const MAX_ITERATIONS: usize = 100;

    /// Computes only eigenvalues of a general complex matrix (no eigenvectors).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::ComplexGeneralEvd;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
    ///     &[Complex64::new(0.0, -1.0), Complex64::new(3.0, 0.0)],
    /// ]);
    ///
    /// let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
    /// let eigenvalues = evd.eigenvalues();
    ///
    /// // Sum of eigenvalues equals trace
    /// let sum = eigenvalues[0] + eigenvalues[1];
    /// let trace = a[(0, 0)] + a[(1, 1)];
    /// assert!((sum - trace).norm() < 1e-10);
    /// ```
    pub fn eigenvalues_only(a: MatRef<'_, T>) -> Result<Self, ComplexGeneralEvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexGeneralEvdError::EmptyMatrix);
        }

        if m != n {
            return Err(ComplexGeneralEvdError::NotSquare);
        }

        // Handle 1×1 case
        if n == 1 {
            return Ok(Self {
                eigenvalues: vec![a[(0, 0)]],
                eigenvectors: None,
                left_eigenvectors: None,
                n,
            });
        }

        // Handle 2×2 case directly
        if n == 2 {
            let eigenvalues = compute_2x2_eigenvalues(a);
            return Ok(Self {
                eigenvalues,
                eigenvectors: None,
                left_eigenvectors: None,
                n,
            });
        }

        // Reduce to upper Hessenberg form
        let (mut h, _q) = complex_hessenberg(a);

        // Apply QR iteration to reduce to upper triangular
        let eigenvalues = complex_qr_iteration(&mut h, Self::MAX_ITERATIONS * n)?;

        Ok(Self {
            eigenvalues,
            eigenvectors: None,
            left_eigenvectors: None,
            n,
        })
    }

    /// Computes eigenvalues and right eigenvectors of a general complex matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::ComplexGeneralEvd;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     &[Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
    /// ]);
    ///
    /// let evd = ComplexGeneralEvd::compute(a.as_ref()).unwrap();
    /// let v = evd.eigenvectors().unwrap();
    ///
    /// // Diagonal matrix: eigenvectors are standard basis vectors
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ComplexGeneralEvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexGeneralEvdError::EmptyMatrix);
        }

        if m != n {
            return Err(ComplexGeneralEvdError::NotSquare);
        }

        // Handle 1×1 case
        if n == 1 {
            let mut v = Mat::zeros(1, 1);
            v[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues: vec![a[(0, 0)]],
                eigenvectors: Some(v),
                left_eigenvectors: None,
                n,
            });
        }

        // Handle 2×2 case directly
        if n == 2 {
            let eigenvalues = compute_2x2_eigenvalues(a);
            let eigenvectors = compute_2x2_eigenvectors(a, &eigenvalues);
            return Ok(Self {
                eigenvalues,
                eigenvectors: Some(eigenvectors),
                left_eigenvectors: None,
                n,
            });
        }

        // Reduce to upper Hessenberg form
        let (mut h, q) = complex_hessenberg(a);

        // Apply QR iteration with Schur vectors
        let (eigenvalues, schur_vectors) =
            complex_qr_iteration_with_vectors(&mut h, q, Self::MAX_ITERATIONS * n)?;

        // Compute eigenvectors from Schur form
        let eigenvectors = compute_eigenvectors_from_schur(&h, &schur_vectors, &eigenvalues);

        Ok(Self {
            eigenvalues,
            eigenvectors: Some(eigenvectors),
            left_eigenvectors: None,
            n,
        })
    }

    /// Computes eigenvalues, right eigenvectors, and left eigenvectors.
    ///
    /// Right eigenvectors satisfy: A * v = λ * v
    /// Left eigenvectors satisfy: u^H * A = λ * u^H
    pub fn compute_full(a: MatRef<'_, T>) -> Result<Self, ComplexGeneralEvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexGeneralEvdError::EmptyMatrix);
        }

        if m != n {
            return Err(ComplexGeneralEvdError::NotSquare);
        }

        // Compute right eigenvectors
        let right_evd = Self::compute(a)?;
        let eigenvalues = right_evd.eigenvalues;
        let eigenvectors = right_evd.eigenvectors;

        // Compute left eigenvectors from A^H
        // Left eigenvectors of A are conjugate of right eigenvectors of A^H
        let mut ah = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                ah[(i, j)] = Scalar::conj(a[(j, i)]);
            }
        }

        let (mut h_ah, q_ah) = complex_hessenberg(ah.as_ref());
        let (_, schur_vectors_ah) =
            complex_qr_iteration_with_vectors(&mut h_ah, q_ah, Self::MAX_ITERATIONS * n)?;
        let left_vecs = compute_eigenvectors_from_schur(&h_ah, &schur_vectors_ah, &eigenvalues);

        // Conjugate the left eigenvectors
        let mut left_eigenvectors = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                left_eigenvectors[(i, j)] = Scalar::conj(left_vecs[(i, j)]);
            }
        }

        Ok(Self {
            eigenvalues,
            eigenvectors,
            left_eigenvectors: Some(left_eigenvectors),
            n,
        })
    }

    /// Returns the eigenvalues.
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the right eigenvectors (if computed).
    pub fn eigenvectors(&self) -> Option<MatRef<'_, T>> {
        self.eigenvectors.as_ref().map(|m| m.as_ref())
    }

    /// Returns the left eigenvectors (if computed).
    pub fn left_eigenvectors(&self) -> Option<MatRef<'_, T>> {
        self.left_eigenvectors.as_ref().map(|m| m.as_ref())
    }

    /// Returns the matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Verifies the eigenvalue decomposition: A * v ≈ λ * v.
    /// Returns the maximum residual norm.
    pub fn verify(&self, a: MatRef<'_, T>) -> T::Real {
        if self.eigenvectors.is_none() {
            return real_zero::<T::Real>();
        }

        let v = self.eigenvectors.as_ref().expect("value should be present");
        let mut max_residual = real_zero::<T::Real>();

        for (col, &lambda) in self.eigenvalues.iter().enumerate() {
            if col >= self.n {
                break;
            }

            // Compute ||A*v - λ*v||
            let mut residual_sq = real_zero::<T::Real>();

            for i in 0..self.n {
                let mut av_i = T::zero();
                for j in 0..self.n {
                    av_i = av_i + a[(i, j)] * v[(j, col)];
                }
                let diff = av_i - lambda * v[(i, col)];
                residual_sq = residual_sq + diff.abs_sq();
            }

            let residual = Real::sqrt(residual_sq);
            if residual > max_residual {
                max_residual = residual;
            }
        }

        max_residual
    }
}

/// Helper function to get zero for a Real type
#[inline]
fn real_zero<R: Real>() -> R {
    R::from_f64(0.0).unwrap_or_else(R::zero)
}

/// Helper function to get one for a Real type
#[inline]
fn real_one<R: Real>() -> R {
    R::from_f64(1.0).unwrap_or_else(R::zero)
}

/// Helper function to get epsilon for a Real type
#[inline]
fn real_eps<R: Real>() -> R {
    <R as Scalar>::epsilon()
}

/// Helper function to get a constant for a Real type
#[inline]
fn real_const<R: Real>(val: f64) -> R {
    R::from_f64(val).unwrap_or_else(R::zero)
}

/// Computes eigenvalues of a 2x2 complex matrix directly using quadratic formula.
fn compute_2x2_eigenvalues<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Vec<T>
where
    T::Real: Real,
{
    let a00 = a[(0, 0)];
    let a01 = a[(0, 1)];
    let a10 = a[(1, 0)];
    let a11 = a[(1, 1)];

    // Characteristic polynomial: λ² - (a00+a11)λ + (a00*a11 - a01*a10) = 0
    let trace = a00 + a11;
    let det = a00 * a11 - a01 * a10;

    // λ = (trace ± sqrt(trace² - 4*det)) / 2
    let four = T::from_real(real_const(4.0));
    let two = T::from_real(real_const(2.0));
    let disc = trace * trace - four * det;

    // Compute square root of discriminant
    let disc_sqrt = complex_sqrt(disc);

    let lambda1 = (trace + disc_sqrt) / two;
    let lambda2 = (trace - disc_sqrt) / two;

    vec![lambda1, lambda2]
}

/// Computes eigenvectors of a 2x2 complex matrix given eigenvalues.
fn compute_2x2_eigenvectors<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    eigenvalues: &[T],
) -> Mat<T>
where
    T::Real: Real,
{
    let eps: T::Real = real_eps::<T::Real>() * real_const(100.0);
    let mut v = Mat::zeros(2, 2);

    for (col, &lambda) in eigenvalues.iter().enumerate() {
        // Solve (A - λI)x = 0
        // For 2x2: [[a00-λ, a01], [a10, a11-λ]] * [x, y]^T = 0

        let a00_l = a[(0, 0)] - lambda;
        let a01 = a[(0, 1)];
        let a10 = a[(1, 0)];
        let a11_l = a[(1, 1)] - lambda;

        // Choose the row with larger magnitude for numerical stability
        let row0_mag = a00_l.abs_sq() + a01.abs_sq();
        let row1_mag = a10.abs_sq() + a11_l.abs_sq();

        let (x, y) = if row0_mag >= row1_mag && a00_l.abs_sq() > eps * eps {
            // Use first row: x * (a00-λ) + y * a01 = 0  =>  y = -x * (a00-λ) / a01
            if a01.abs_sq() > eps * eps {
                (T::one(), -a00_l / a01)
            } else {
                (T::zero(), T::one())
            }
        } else if row1_mag > eps * eps && a10.abs_sq() > eps * eps {
            // Use second row: x * a10 + y * (a11-λ) = 0  =>  x = -y * (a11-λ) / a10
            if a10.abs_sq() > eps * eps {
                (-a11_l / a10, T::one())
            } else {
                (T::one(), T::zero())
            }
        } else {
            // Nearly zero matrix, use standard basis
            (T::one(), T::zero())
        };

        // Normalize
        let norm_sq = x.abs_sq() + y.abs_sq();
        let norm = Real::sqrt(norm_sq);
        if norm > eps {
            v[(0, col)] = x / T::from_real(norm);
            v[(1, col)] = y / T::from_real(norm);
        } else {
            v[(0, col)] = T::one();
            v[(1, col)] = T::zero();
        }
    }

    v
}

/// Reduces a complex matrix to upper Hessenberg form using Householder reflections.
/// Returns (H, Q) where A = Q * H * Q^H.
fn complex_hessenberg<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> (Mat<T>, Mat<T>)
where
    T::Real: Real,
{
    let n = a.nrows();

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

    if n <= 2 {
        return (h, q);
    }

    // Reduce to Hessenberg form
    for k in 0..(n - 2) {
        // Create Householder reflection to zero H[k+2:n, k]
        let m = n - k - 1;
        let mut x = vec![T::zero(); m];
        for i in 0..m {
            x[i] = h[(k + 1 + i, k)];
        }

        let (v, tau) = complex_householder_vector(&x);

        if tau != real_zero::<T::Real>() {
            let tau_c = T::from_real(tau);

            // Apply P from the left: H := P * H where P = I - tau*v*v^H
            for j in k..n {
                let mut dot = T::zero();
                for i in 0..m {
                    dot = dot + Scalar::conj(v[i]) * h[(k + 1 + i, j)];
                }
                let scaled = tau_c * dot;
                for i in 0..m {
                    h[(k + 1 + i, j)] = h[(k + 1 + i, j)] - scaled * v[i];
                }
            }

            // Apply P from the right: H := H * P^H = H * (I - tau*v*v^H)^H = H * (I - conj(tau)*v*v^H)
            // Since tau is real, P^H = P
            for i in 0..n {
                let mut dot = T::zero();
                for j in 0..m {
                    dot = dot + h[(i, k + 1 + j)] * v[j];
                }
                let scaled = tau_c * dot;
                for j in 0..m {
                    h[(i, k + 1 + j)] = h[(i, k + 1 + j)] - scaled * Scalar::conj(v[j]);
                }
            }

            // Accumulate Q := Q * P
            for i in 0..n {
                let mut dot = T::zero();
                for j in 0..m {
                    dot = dot + q[(i, k + 1 + j)] * v[j];
                }
                let scaled = tau_c * dot;
                for j in 0..m {
                    q[(i, k + 1 + j)] = q[(i, k + 1 + j)] - scaled * Scalar::conj(v[j]);
                }
            }
        }
    }

    // Clean up small values below subdiagonal
    let eps: T::Real = real_eps::<T::Real>() * real_const::<T::Real>(100.0);
    for j in 0..(n - 2) {
        for i in (j + 2)..n {
            if h[(i, j)].abs_sq() < eps * eps {
                h[(i, j)] = T::zero();
            }
        }
    }

    (h, q)
}

/// Computes a complex Householder vector.
/// Returns (v, tau) such that (I - tau * v * v^H) * x = [alpha, 0, ..., 0]^T
fn complex_householder_vector<T: Field + ComplexScalar>(x: &[T]) -> (Vec<T>, T::Real)
where
    T::Real: Real,
{
    let n = x.len();
    if n == 0 {
        return (Vec::new(), real_zero::<T::Real>());
    }

    // Compute ||x||
    let mut norm_sq: T::Real = real_zero();
    for i in 0..n {
        norm_sq = norm_sq + x[i].abs_sq();
    }
    let norm = Real::sqrt(norm_sq);

    if norm == real_zero::<T::Real>() {
        return (vec![T::zero(); n], real_zero::<T::Real>());
    }

    // Compute alpha = -exp(i*arg(x[0])) * ||x||
    let x0_norm = Real::sqrt(x[0].abs_sq());
    let phase = if x0_norm > real_eps::<T::Real>() {
        x[0] / T::from_real(x0_norm)
    } else {
        T::one()
    };
    let alpha = -phase * T::from_real(norm);

    // v = x - alpha * e_1
    let mut v = x.to_vec();
    v[0] = x[0] - alpha;

    // ||v||^2
    let mut v_norm_sq: T::Real = real_zero();
    for i in 0..n {
        v_norm_sq = v_norm_sq + v[i].abs_sq();
    }

    if v_norm_sq == real_zero::<T::Real>() {
        return (vec![T::zero(); n], real_zero::<T::Real>());
    }

    // Normalize v
    let v_norm = Real::sqrt(v_norm_sq);
    for i in 0..n {
        v[i] = v[i] / T::from_real(v_norm);
    }

    // tau = 2
    let tau: T::Real = real_const(2.0);

    (v, tau)
}

/// Applies QR iteration to reduce upper Hessenberg matrix to upper triangular (Schur form).
/// Returns the eigenvalues (diagonal elements).
fn complex_qr_iteration<T: Field + ComplexScalar + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    max_iter: usize,
) -> Result<Vec<T>, ComplexGeneralEvdError>
where
    T::Real: Real,
{
    let n = h.nrows();
    let eps: T::Real = real_eps::<T::Real>() * real_const(100.0);

    let mut p = n;
    let mut iter_count = 0;
    let mut stagnation_count = 0;

    while p > 1 && iter_count < max_iter {
        iter_count += 1;

        // Find the active block
        let mut q_idx = p - 1;
        while q_idx > 0 {
            let sub = Real::sqrt(h[(q_idx, q_idx - 1)].abs_sq());
            let diag_sum = Real::sqrt(h[(q_idx - 1, q_idx - 1)].abs_sq())
                + Real::sqrt(h[(q_idx, q_idx)].abs_sq());
            if sub <= eps * (diag_sum + real_one::<T::Real>()) {
                h[(q_idx, q_idx - 1)] = T::zero();
                break;
            }
            q_idx -= 1;
        }

        if q_idx == p - 1 {
            // 1×1 block converged
            p -= 1;
            stagnation_count = 0;
        } else {
            // Compute Wilkinson shift from bottom 2x2 block
            let shift = compute_wilkinson_shift(h, p);

            // Check for stagnation
            stagnation_count += 1;
            let actual_shift = if stagnation_count > 10 {
                // Exceptional shift to break stagnation
                stagnation_count = 0;
                let mag =
                    Real::sqrt(h[(p - 1, p - 2)].abs_sq()) + Real::sqrt(h[(p - 1, p - 1)].abs_sq());
                T::from_real(mag)
            } else {
                shift
            };

            complex_qr_step(h, q_idx, p, actual_shift);
        }
    }

    // Clean up small subdiagonal elements
    for i in 1..n {
        let sub = Real::sqrt(h[(i, i - 1)].abs_sq());
        let diag_sum = Real::sqrt(h[(i - 1, i - 1)].abs_sq()) + Real::sqrt(h[(i, i)].abs_sq());
        if sub <= eps * (diag_sum + real_one::<T::Real>()) * real_const(10.0) {
            h[(i, i - 1)] = T::zero();
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(h[(i, i)]);
    }

    Ok(eigenvalues)
}

/// Computes the Wilkinson shift from the bottom 2x2 block.
fn compute_wilkinson_shift<T: Field + ComplexScalar + bytemuck::Zeroable>(h: &Mat<T>, p: usize) -> T
where
    T::Real: Real,
{
    if p < 2 {
        return h[(p - 1, p - 1)];
    }

    let h11 = h[(p - 2, p - 2)];
    let h12 = h[(p - 2, p - 1)];
    let h21 = h[(p - 1, p - 2)];
    let h22 = h[(p - 1, p - 1)];

    // Compute eigenvalues of 2x2 block
    let trace = h11 + h22;
    let det = h11 * h22 - h12 * h21;

    // λ = (trace ± sqrt(trace² - 4*det)) / 2
    let disc = trace * trace - T::from_real(real_const(4.0)) * det;

    // Use square root of complex number
    let disc_sqrt = complex_sqrt(disc);

    let half = T::from_real(real_const(0.5));
    let lambda1 = half * (trace + disc_sqrt);
    let lambda2 = half * (trace - disc_sqrt);

    // Choose eigenvalue closer to h22
    let diff1 = lambda1 - h22;
    let diff2 = lambda2 - h22;

    if diff1.abs_sq() < diff2.abs_sq() {
        lambda1
    } else {
        lambda2
    }
}

/// Computes the square root of a complex number.
fn complex_sqrt<T: Field + ComplexScalar>(z: T) -> T
where
    T::Real: Real,
{
    let (re, im) = (z.real(), z.imag());
    let mag = Real::sqrt(re * re + im * im);

    if mag == real_zero::<T::Real>() {
        return T::zero();
    }

    let two = real_const::<T::Real>(2.0);
    let r = Real::sqrt((mag + re) / two);
    let i = if im >= real_zero::<T::Real>() {
        Real::sqrt((mag - re) / two)
    } else {
        -Real::sqrt((mag - re) / two)
    };

    T::from_real_imag(r, i)
}

/// Applies QR iteration with accumulation of Schur vectors.
fn complex_qr_iteration_with_vectors<T: Field + ComplexScalar + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    mut q: Mat<T>,
    max_iter: usize,
) -> Result<(Vec<T>, Mat<T>), ComplexGeneralEvdError>
where
    T::Real: Real,
{
    let n = h.nrows();
    let eps: T::Real = real_eps::<T::Real>() * real_const(100.0);

    let mut p = n;
    let mut iter_count = 0;
    let mut stagnation_count = 0;

    while p > 1 && iter_count < max_iter {
        iter_count += 1;

        // Find the active block
        let mut q_idx = p - 1;
        while q_idx > 0 {
            let sub = Real::sqrt(h[(q_idx, q_idx - 1)].abs_sq());
            let diag_sum = Real::sqrt(h[(q_idx - 1, q_idx - 1)].abs_sq())
                + Real::sqrt(h[(q_idx, q_idx)].abs_sq());
            if sub <= eps * (diag_sum + real_one::<T::Real>()) {
                h[(q_idx, q_idx - 1)] = T::zero();
                break;
            }
            q_idx -= 1;
        }

        if q_idx == p - 1 {
            // 1×1 block converged
            p -= 1;
            stagnation_count = 0;
        } else {
            // Compute Wilkinson shift from bottom 2x2 block
            let shift = compute_wilkinson_shift(h, p);

            // Check for stagnation
            stagnation_count += 1;
            let actual_shift = if stagnation_count > 10 {
                // Exceptional shift to break stagnation
                stagnation_count = 0;
                let mag =
                    Real::sqrt(h[(p - 1, p - 2)].abs_sq()) + Real::sqrt(h[(p - 1, p - 1)].abs_sq());
                T::from_real(mag)
            } else {
                shift
            };

            complex_qr_step_with_q(h, &mut q, q_idx, p, actual_shift);
        }
    }

    // Clean up small subdiagonal elements
    for i in 1..n {
        let sub = Real::sqrt(h[(i, i - 1)].abs_sq());
        let diag_sum = Real::sqrt(h[(i - 1, i - 1)].abs_sq()) + Real::sqrt(h[(i, i)].abs_sq());
        if sub <= eps * (diag_sum + real_one::<T::Real>()) * real_const(10.0) {
            h[(i, i - 1)] = T::zero();
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(h[(i, i)]);
    }

    Ok((eigenvalues, q))
}

/// Applies one QR step with shift.
fn complex_qr_step<T: Field + ComplexScalar + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    start: usize,
    end: usize,
    shift: T,
) where
    T::Real: Real,
{
    let n = h.nrows();

    for k in start..(end - 1) {
        // Apply Givens rotation to zero out h[k+1, k]
        let a = h[(k, k)] - shift;
        let b = h[(k + 1, k)];

        let (c, s) = givens_rotation_complex(a, b);

        // Apply from left: H := G^H * H
        for j in k..n {
            let t1 = h[(k, j)];
            let t2 = h[(k + 1, j)];
            h[(k, j)] = Scalar::conj(c) * t1 + Scalar::conj(s) * t2;
            h[(k + 1, j)] = -s * t1 + c * t2;
        }

        // Apply from right: H := H * G
        let row_end = (k + 3).min(end);
        for i in 0..row_end {
            let t1 = h[(i, k)];
            let t2 = h[(i, k + 1)];
            h[(i, k)] = t1 * c + t2 * s;
            h[(i, k + 1)] = -t1 * Scalar::conj(s) + t2 * Scalar::conj(c);
        }
    }
}

/// Applies one QR step with Q accumulation.
fn complex_qr_step_with_q<T: Field + ComplexScalar + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    q: &mut Mat<T>,
    start: usize,
    end: usize,
    shift: T,
) where
    T::Real: Real,
{
    let n = h.nrows();

    for k in start..(end - 1) {
        // Apply Givens rotation to zero out h[k+1, k]
        let a = h[(k, k)] - shift;
        let b = h[(k + 1, k)];

        let (c, s) = givens_rotation_complex(a, b);

        // Apply from left: H := G^H * H
        for j in k..n {
            let t1 = h[(k, j)];
            let t2 = h[(k + 1, j)];
            h[(k, j)] = Scalar::conj(c) * t1 + Scalar::conj(s) * t2;
            h[(k + 1, j)] = -s * t1 + c * t2;
        }

        // Apply from right: H := H * G
        let row_end = (k + 3).min(end);
        for i in 0..row_end {
            let t1 = h[(i, k)];
            let t2 = h[(i, k + 1)];
            h[(i, k)] = t1 * c + t2 * s;
            h[(i, k + 1)] = -t1 * Scalar::conj(s) + t2 * Scalar::conj(c);
        }

        // Accumulate Q := Q * G
        for i in 0..n {
            let t1 = q[(i, k)];
            let t2 = q[(i, k + 1)];
            q[(i, k)] = t1 * c + t2 * s;
            q[(i, k + 1)] = -t1 * Scalar::conj(s) + t2 * Scalar::conj(c);
        }
    }
}

/// Computes Givens rotation for complex numbers.
/// Returns (c, s) such that G^H * [a; b] = [r; 0] where G = [[c, s], [-conj(s), conj(c)]]
fn givens_rotation_complex<T: Field + ComplexScalar>(a: T, b: T) -> (T, T)
where
    T::Real: Real,
{
    let b_norm = Real::sqrt(b.abs_sq());

    if b_norm == real_zero::<T::Real>() {
        return (T::one(), T::zero());
    }

    let a_norm = Real::sqrt(a.abs_sq());
    let r = Real::sqrt(a_norm * a_norm + b_norm * b_norm);

    if r == real_zero::<T::Real>() {
        return (T::one(), T::zero());
    }

    let c = T::from_real(a_norm / r);
    let phase_a = if a_norm > real_eps::<T::Real>() {
        a / T::from_real(a_norm)
    } else {
        T::one()
    };
    let s = Scalar::conj(phase_a) * b / T::from_real(r);

    (c * phase_a, s)
}

/// Computes eigenvectors from Schur form by back-substitution.
fn compute_eigenvectors_from_schur<T: Field + ComplexScalar + bytemuck::Zeroable>(
    t: &Mat<T>,
    q: &Mat<T>,
    eigenvalues: &[T],
) -> Mat<T>
where
    T::Real: Real,
{
    let n = t.nrows();
    let mut v = Mat::zeros(n, n);
    let eps: T::Real = real_eps::<T::Real>() * real_const(100.0);

    for (col, &lambda) in eigenvalues.iter().enumerate() {
        // Solve (T - λI)x = 0 by back substitution
        let mut x = vec![T::zero(); n];
        x[col] = T::one();

        // Back substitution for upper triangular system
        for i in (0..col).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..=col {
                sum = sum + t[(i, j)] * x[j];
            }
            let diag = t[(i, i)] - lambda;
            if diag.abs_sq() > eps * eps {
                x[i] = -sum / diag;
            }
        }

        // Normalize
        let mut norm_sq: T::Real = real_zero();
        for i in 0..n {
            norm_sq = norm_sq + x[i].abs_sq();
        }
        let norm = Real::sqrt(norm_sq);
        if norm > eps {
            for i in 0..n {
                x[i] = x[i] / T::from_real(norm);
            }
        }

        // Transform back: v = Q * x
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + q[(i, j)] * x[j];
            }
            v[(i, col)] = sum;
        }
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn approx_eq_complex(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    #[test]
    fn test_complex_general_evd_diagonal() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Eigenvalues should be 1 and 2
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.re).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_general_evd_trace() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 1.0)],
            &[Complex64::new(-1.0, 1.0), Complex64::new(4.0, -1.0)],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Sum of eigenvalues equals trace
        let sum = eigenvalues[0] + eigenvalues[1];
        let trace = a[(0, 0)] + a[(1, 1)];
        assert!(
            approx_eq_complex(sum, trace, 1e-10),
            "sum = {:?}, trace = {:?}",
            sum,
            trace
        );
    }

    #[test]
    fn test_complex_general_evd_determinant() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Product of eigenvalues equals determinant
        let prod = eigenvalues[0] * eigenvalues[1];
        let det = a[(0, 0)] * a[(1, 1)] - a[(0, 1)] * a[(1, 0)]; // = 1*4 - 2*3 = -2
        assert!(
            approx_eq_complex(prod, det, 1e-10),
            "prod = {:?}, det = {:?}",
            prod,
            det
        );
    }

    #[test]
    fn test_complex_general_evd_with_eigenvectors() {
        let a = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
        ]);

        let evd = ComplexGeneralEvd::compute(a.as_ref()).unwrap();
        assert!(evd.eigenvectors().is_some());

        // Verify eigenvalue-eigenvector relationship
        let residual = evd.verify(a.as_ref());
        assert!(residual < 1e-10, "Residual too large: {}", residual);
    }

    #[test]
    fn test_complex_general_evd_3x3() {
        // Upper triangular matrix
        let a = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 0.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, -1.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // For upper triangular, eigenvalues are diagonal elements
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.re).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 4.0).abs() < 1e-10);
        assert!((eigs[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_general_evd_single() {
        let a = Mat::from_rows(&[&[Complex64::new(5.0, 2.0)]]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        assert_eq!(eigenvalues.len(), 1);
        assert!(approx_eq_complex(
            eigenvalues[0],
            Complex64::new(5.0, 2.0),
            1e-10
        ));
    }

    #[test]
    fn test_complex_general_evd_hermitian() {
        // Hermitian matrix - eigenvalues should be real
        let a = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Eigenvalues of Hermitian matrix are real
        for e in eigenvalues {
            assert!(e.im.abs() < 1e-10, "Expected real eigenvalue, got {:?}", e);
        }
    }

    #[test]
    fn test_complex_general_evd_f32() {
        let a = Mat::from_rows(&[
            &[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
            &[Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Sum of eigenvalues equals trace
        let sum = eigenvalues[0] + eigenvalues[1];
        let trace = a[(0, 0)] + a[(1, 1)];
        assert!(
            (sum - trace).norm() < 1e-4,
            "sum = {:?}, trace = {:?}",
            sum,
            trace
        );
    }

    #[test]
    fn test_complex_general_evd_rotation() {
        // Rotation matrix - eigenvalues should be e^{±iθ}
        let theta = std::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();

        let a = Mat::from_rows(&[
            &[Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
            &[Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
        ]);

        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Eigenvalues should be e^{iθ} and e^{-iθ}
        // Sum of eigenvalues = trace = 2*cos(θ)
        let sum = eigenvalues[0] + eigenvalues[1];
        let expected_trace = Complex64::new(2.0 * c, 0.0);
        assert!(
            (sum - expected_trace).norm() < 1e-10,
            "Expected trace {}, got {}",
            expected_trace,
            sum
        );

        // Product of eigenvalues = determinant = cos²θ + sin²θ = 1
        let prod = eigenvalues[0] * eigenvalues[1];
        let expected_det = Complex64::new(1.0, 0.0);
        assert!(
            (prod - expected_det).norm() < 1e-10,
            "Expected det {}, got {}",
            expected_det,
            prod
        );
    }
}
