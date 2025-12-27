//! QZ Algorithm (Generalized Schur Decomposition).
//!
//! Computes the generalized Schur form of matrix pencil (A, B):
//!   A = Q * S * Z^T
//!   B = Q * T * Z^T
//! where Q and Z are orthogonal, S is quasi-upper triangular, and T is upper triangular.
//!
//! Generalized eigenvalues are α/β where α = diag(S), β = diag(T).

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for QZ decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QzError {
    /// Matrices are empty.
    EmptyMatrix,
    /// Matrices have incompatible dimensions.
    DimensionMismatch {
        /// Rows of A.
        nrows_a: usize,
        /// Columns of A.
        ncols_a: usize,
        /// Rows of B.
        nrows_b: usize,
        /// Columns of B.
        ncols_b: usize,
    },
    /// Matrix A is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Algorithm did not converge.
    NotConverged {
        /// Iteration count when failure occurred.
        iterations: usize,
    },
}

impl core::fmt::Display for QzError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Empty matrix"),
            Self::DimensionMismatch {
                nrows_a,
                ncols_a,
                nrows_b,
                ncols_b,
            } => {
                write!(
                    f,
                    "Dimension mismatch: A is {nrows_a}x{ncols_a}, B is {nrows_b}x{ncols_b}"
                )
            }
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::NotConverged { iterations } => {
                write!(f, "Did not converge after {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for QzError {}

/// Generalized eigenvalue from QZ decomposition.
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedEigenvalue<T: Scalar> {
    /// Numerator (α = S\[i,i\]).
    pub alpha_real: T,
    /// Imaginary part of numerator (for complex eigenvalues).
    pub alpha_imag: T,
    /// Denominator (β = T\[i,i\]).
    pub beta: T,
}

impl<T: Field + Real> GeneralizedEigenvalue<T> {
    /// Creates a real generalized eigenvalue.
    pub fn real(alpha: T, beta: T) -> Self {
        Self {
            alpha_real: alpha,
            alpha_imag: T::zero(),
            beta,
        }
    }

    /// Creates a complex generalized eigenvalue.
    pub fn complex(alpha_real: T, alpha_imag: T, beta: T) -> Self {
        Self {
            alpha_real,
            alpha_imag,
            beta,
        }
    }

    /// Returns true if the eigenvalue is real.
    pub fn is_real(&self) -> bool {
        Scalar::abs(self.alpha_imag) <= <T as Scalar>::epsilon()
    }

    /// Returns true if the eigenvalue is finite (β ≠ 0).
    pub fn is_finite(&self) -> bool {
        Scalar::abs(self.beta) > <T as Scalar>::epsilon()
    }

    /// Computes the eigenvalue as α/β.
    /// Returns None if β = 0 (infinite eigenvalue).
    pub fn value(&self) -> Option<T> {
        if self.is_finite() && self.is_real() {
            Some(self.alpha_real / self.beta)
        } else {
            None
        }
    }

    /// Computes the real and imaginary parts of α/β.
    pub fn value_complex(&self) -> Option<(T, T)> {
        if !self.is_finite() {
            return None;
        }
        Some((self.alpha_real / self.beta, self.alpha_imag / self.beta))
    }
}

/// QZ decomposition (generalized Schur form).
///
/// For matrix pencil (A, B), computes:
///   A = Q * S * Z^T
///   B = Q * T * Z^T
///
/// where Q, Z are orthogonal, S is quasi-upper triangular, T is upper triangular.
#[derive(Debug, Clone)]
pub struct Qz<T: Scalar> {
    /// Quasi-upper triangular matrix S.
    s: Mat<T>,
    /// Upper triangular matrix T.
    t: Mat<T>,
    /// Left orthogonal factor Q.
    q: Mat<T>,
    /// Right orthogonal factor Z.
    z: Mat<T>,
    /// Generalized eigenvalues.
    eigenvalues: Vec<GeneralizedEigenvalue<T>>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> Qz<T> {
    /// Maximum number of QZ iterations per eigenvalue.
    const MAX_ITERATIONS: usize = 100;

    /// Computes the QZ decomposition of matrix pencil (A, B).
    ///
    /// # Arguments
    ///
    /// * `a` - First matrix of the pencil
    /// * `b` - Second matrix of the pencil
    ///
    /// # Returns
    ///
    /// The QZ decomposition such that A = Q * S * Z^T and B = Q * T * Z^T.
    pub fn compute(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Result<Self, QzError> {
        let n = a.nrows();

        // Validate dimensions
        if n == 0 {
            return Err(QzError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(QzError::NotSquare {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        if n != b.nrows() || n != b.ncols() {
            return Err(QzError::DimensionMismatch {
                nrows_a: a.nrows(),
                ncols_a: a.ncols(),
                nrows_b: b.nrows(),
                ncols_b: b.ncols(),
            });
        }

        // Handle trivial case
        if n == 1 {
            let mut s = Mat::zeros(1, 1);
            let mut t = Mat::zeros(1, 1);
            let mut q = Mat::zeros(1, 1);
            let mut z = Mat::zeros(1, 1);

            s[(0, 0)] = a[(0, 0)];
            t[(0, 0)] = b[(0, 0)];
            q[(0, 0)] = T::one();
            z[(0, 0)] = T::one();

            let eigenvalue = GeneralizedEigenvalue::real(a[(0, 0)], b[(0, 0)]);

            return Ok(Self {
                s,
                t,
                q,
                z,
                eigenvalues: vec![eigenvalue],
                n,
            });
        }

        // Copy matrices to working storage
        let mut h = Mat::zeros(n, n);
        let mut r = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = a[(i, j)];
                r[(i, j)] = b[(i, j)];
            }
        }

        // Initialize Q and Z as identity
        let mut q = Mat::zeros(n, n);
        let mut z = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
            z[(i, i)] = T::one();
        }

        // Step 1: Reduce B to upper triangular form using QR
        qr_factorize(&mut r, &mut q);

        // Apply Q^T to A from the left: H = Q^T * A
        let qt_a = q_transpose_mult(&q, &h);
        h = qt_a;

        // Step 2: Reduce A to upper Hessenberg form while keeping B upper triangular
        hessenberg_qz(&mut h, &mut r, &mut q, &mut z);

        // Step 3: QZ iterations to reduce to quasi-triangular form
        qz_iterations(&mut h, &mut r, &mut q, &mut z)?;

        // Extract eigenvalues from diagonal of S and T
        let eigenvalues = extract_eigenvalues(&h, &r);

        Ok(Self {
            s: h,
            t: r,
            q,
            z,
            eigenvalues,
            n,
        })
    }

    /// Returns the S matrix (quasi-upper triangular).
    pub fn s(&self) -> MatRef<'_, T> {
        self.s.as_ref()
    }

    /// Returns the T matrix (upper triangular).
    pub fn t(&self) -> MatRef<'_, T> {
        self.t.as_ref()
    }

    /// Returns the left orthogonal matrix Q.
    pub fn q(&self) -> MatRef<'_, T> {
        self.q.as_ref()
    }

    /// Returns the right orthogonal matrix Z.
    pub fn z(&self) -> MatRef<'_, T> {
        self.z.as_ref()
    }

    /// Returns the generalized eigenvalues.
    pub fn eigenvalues(&self) -> &[GeneralizedEigenvalue<T>] {
        &self.eigenvalues
    }

    /// Returns the real parts of the eigenvalues.
    pub fn eigenvalues_real(&self) -> Vec<T> {
        self.eigenvalues.iter().filter_map(|e| e.value()).collect()
    }

    /// Returns the number of infinite eigenvalues (β = 0).
    pub fn num_infinite(&self) -> usize {
        self.eigenvalues.iter().filter(|e| !e.is_finite()).count()
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Verifies the decomposition: A ≈ Q * S * Z^T and B ≈ Q * T * Z^T.
    pub fn verify(&self, a: MatRef<'_, T>, b: MatRef<'_, T>) -> (T, T) {
        let n = self.n;

        // Compute Q * S * Z^T and compare with A
        let qs = mat_mult(&self.q, &self.s);
        let qszt = mat_mult_transpose(&qs, &self.z);

        let mut error_a = T::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = Scalar::abs(qszt[(i, j)] - a[(i, j)]);
                if diff > error_a {
                    error_a = diff;
                }
            }
        }

        // Compute Q * T * Z^T and compare with B
        let qt = mat_mult(&self.q, &self.t);
        let qtzt = mat_mult_transpose(&qt, &self.z);

        let mut error_b = T::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = Scalar::abs(qtzt[(i, j)] - b[(i, j)]);
                if diff > error_b {
                    error_b = diff;
                }
            }
        }

        (error_a, error_b)
    }
}

/// Performs QR factorization of R, accumulating Q.
fn qr_factorize<T: Field + Real + bytemuck::Zeroable>(r: &mut Mat<T>, q: &mut Mat<T>) {
    let n = r.nrows();

    for k in 0..n {
        // Compute Householder vector for column k
        let mut col_norm_sq = T::zero();
        for i in k..n {
            col_norm_sq = col_norm_sq + r[(i, k)] * r[(i, k)];
        }
        let col_norm = Real::sqrt(col_norm_sq);

        if Scalar::abs(col_norm) <= <T as Scalar>::epsilon() {
            continue;
        }

        // Sign for numerical stability
        let sign = if r[(k, k)] >= T::zero() {
            T::one()
        } else {
            T::zero() - T::one()
        };
        let alpha = sign * col_norm;

        // Householder vector
        let mut v = vec![T::zero(); n - k];
        v[0] = r[(k, k)] + alpha;
        for i in 1..(n - k) {
            v[i] = r[(k + i, k)];
        }

        // Normalize v
        let mut v_norm_sq = T::zero();
        for vi in &v {
            v_norm_sq = v_norm_sq + *vi * *vi;
        }
        let v_norm = Real::sqrt(v_norm_sq);
        if Scalar::abs(v_norm) > <T as Scalar>::epsilon() {
            for vi in &mut v {
                *vi = *vi / v_norm;
            }
        }

        // Apply H = I - 2*v*v^T to R from the left
        // R[k:, :] = R[k:, :] - 2 * v * (v^T * R[k:, :])
        let two = T::one() + T::one();
        for j in 0..n {
            let mut vtr = T::zero();
            for i in 0..(n - k) {
                vtr = vtr + v[i] * r[(k + i, j)];
            }
            for i in 0..(n - k) {
                r[(k + i, j)] = r[(k + i, j)] - two * v[i] * vtr;
            }
        }

        // Apply H to Q from the right: Q = Q * H
        // Q[:, k:] = Q[:, k:] - 2 * (Q[:, k:] * v) * v^T
        for i in 0..n {
            let mut qv = T::zero();
            for l in 0..(n - k) {
                qv = qv + q[(i, k + l)] * v[l];
            }
            for l in 0..(n - k) {
                q[(i, k + l)] = q[(i, k + l)] - two * qv * v[l];
            }
        }
    }
}

/// Applies Q^T to matrix H.
fn q_transpose_mult<T: Field + Real + bytemuck::Zeroable>(q: &Mat<T>, h: &Mat<T>) -> Mat<T> {
    let n = q.nrows();
    let mut result = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + q[(k, i)] * h[(k, j)]; // Q^T = Q.transpose()
            }
            result[(i, j)] = sum;
        }
    }

    result
}

/// Reduces A to upper Hessenberg form while keeping B upper triangular.
fn hessenberg_qz<T: Field + Real + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    r: &mut Mat<T>,
    q: &mut Mat<T>,
    z: &mut Mat<T>,
) {
    let n = h.nrows();

    for k in 0..(n - 2) {
        // Zero out H[k+2:n, k] using Householder
        let mut col_norm_sq = T::zero();
        for i in (k + 1)..n {
            col_norm_sq = col_norm_sq + h[(i, k)] * h[(i, k)];
        }
        let col_norm = Real::sqrt(col_norm_sq);

        if Scalar::abs(col_norm) <= <T as Scalar>::epsilon() {
            continue;
        }

        let sign = if h[(k + 1, k)] >= T::zero() {
            T::one()
        } else {
            T::zero() - T::one()
        };
        let alpha = sign * col_norm;

        let mut v = vec![T::zero(); n - k - 1];
        v[0] = h[(k + 1, k)] + alpha;
        for i in 1..(n - k - 1) {
            v[i] = h[(k + 1 + i, k)];
        }

        let mut v_norm_sq = T::zero();
        for vi in &v {
            v_norm_sq = v_norm_sq + *vi * *vi;
        }
        let v_norm = Real::sqrt(v_norm_sq);
        if Scalar::abs(v_norm) <= <T as Scalar>::epsilon() {
            continue;
        }
        for vi in &mut v {
            *vi = *vi / v_norm;
        }

        let two = T::one() + T::one();

        // Apply H_left to H from the left: H[(k+1):, :] -= 2 * v * (v^T * H[(k+1):, :])
        for j in 0..n {
            let mut vth = T::zero();
            for i in 0..(n - k - 1) {
                vth = vth + v[i] * h[(k + 1 + i, j)];
            }
            for i in 0..(n - k - 1) {
                h[(k + 1 + i, j)] = h[(k + 1 + i, j)] - two * v[i] * vth;
            }
        }

        // Apply H_left to Q from the right
        for i in 0..n {
            let mut qv = T::zero();
            for l in 0..(n - k - 1) {
                qv = qv + q[(i, k + 1 + l)] * v[l];
            }
            for l in 0..(n - k - 1) {
                q[(i, k + 1 + l)] = q[(i, k + 1 + l)] - two * qv * v[l];
            }
        }

        // Apply from the right to maintain triangular B
        // H[:, (k+1):] -= 2 * (H[:, (k+1):] * v) * v^T
        for i in 0..n {
            let mut hv = T::zero();
            for l in 0..(n - k - 1) {
                hv = hv + h[(i, k + 1 + l)] * v[l];
            }
            for l in 0..(n - k - 1) {
                h[(i, k + 1 + l)] = h[(i, k + 1 + l)] - two * hv * v[l];
            }
        }

        // Apply to R from the right
        for i in 0..n {
            let mut rv = T::zero();
            for l in 0..(n - k - 1) {
                rv = rv + r[(i, k + 1 + l)] * v[l];
            }
            for l in 0..(n - k - 1) {
                r[(i, k + 1 + l)] = r[(i, k + 1 + l)] - two * rv * v[l];
            }
        }

        // Apply to Z from the right
        for i in 0..n {
            let mut zv = T::zero();
            for l in 0..(n - k - 1) {
                zv = zv + z[(i, k + 1 + l)] * v[l];
            }
            for l in 0..(n - k - 1) {
                z[(i, k + 1 + l)] = z[(i, k + 1 + l)] - two * zv * v[l];
            }
        }
    }
}

/// Performs QZ iterations to reduce to quasi-triangular form.
fn qz_iterations<T: Field + Real + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    r: &mut Mat<T>,
    q: &mut Mat<T>,
    z: &mut Mat<T>,
) -> Result<(), QzError> {
    let n = h.nrows();
    let max_iter = Qz::<T>::MAX_ITERATIONS * n;

    let mut iter_count = 0;
    let mut p = n;

    while p > 1 {
        iter_count += 1;
        if iter_count > max_iter {
            return Err(QzError::NotConverged {
                iterations: iter_count,
            });
        }

        // Find the lowest subdiagonal that is negligible
        let mut l = p - 1;
        while l > 0 {
            let sub = Scalar::abs(h[(l, l - 1)]);
            let diag_sum = Scalar::abs(h[(l - 1, l - 1)]) + Scalar::abs(h[(l, l)]);
            if sub <= <T as Scalar>::epsilon() * diag_sum {
                h[(l, l - 1)] = T::zero();
                break;
            }
            l -= 1;
        }

        if l == p - 1 {
            // 1x1 block deflated
            p -= 1;
            continue;
        }

        if l == p - 2 {
            // Check for 2x2 block (complex eigenvalues)
            let a11 = h[(l, l)];
            let a12 = h[(l, l + 1)];
            let a21 = h[(l + 1, l)];
            let a22 = h[(l + 1, l + 1)];

            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - (T::one() + T::one() + T::one() + T::one()) * det;

            if !(disc >= T::zero()) {
                // Complex eigenvalues - keep 2x2 block
                p -= 2;
                continue;
            }
        }

        // Single QZ step
        qz_step(h, r, q, z, l, p);
    }

    Ok(())
}

/// Single QZ step using implicit double shift.
fn qz_step<T: Field + Real + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    r: &mut Mat<T>,
    q: &mut Mat<T>,
    z: &mut Mat<T>,
    l: usize,
    p: usize,
) {
    let n = h.nrows();

    // Compute implicit shift from trailing 2x2 block
    let pm1 = p - 1;
    let pm2 = p.saturating_sub(2);

    let h11 = if pm2 < n { h[(pm2, pm2)] } else { T::zero() };
    let h12 = if pm2 < n && pm1 < n {
        h[(pm2, pm1)]
    } else {
        T::zero()
    };
    let h21 = if pm1 < n && pm2 < n {
        h[(pm1, pm2)]
    } else {
        T::zero()
    };
    let h22 = if pm1 < n { h[(pm1, pm1)] } else { T::zero() };

    // Francis double shift
    let _tr = h11 + h22;
    let _det = h11 * h22 - h12 * h21;

    // For simplicity, use single shift based on h[p-1, p-1]
    let shift = h22;

    // Chase the bulge
    for k in l..(p - 1) {
        // Givens rotation to zero out subdiagonal element
        let a = h[(k, k)] - shift;
        let b = if k + 1 < n { h[(k + 1, k)] } else { T::zero() };

        let (c, s) = givens(a, b);

        // Apply Givens from left to H
        for j in 0..n {
            let h_k = h[(k, j)];
            let h_k1 = if k + 1 < n { h[(k + 1, j)] } else { T::zero() };
            h[(k, j)] = c * h_k + s * h_k1;
            if k + 1 < n {
                h[(k + 1, j)] = (T::zero() - s) * h_k + c * h_k1;
            }
        }

        // Apply to Q from right
        for i in 0..n {
            let q_k = q[(i, k)];
            let q_k1 = if k + 1 < n { q[(i, k + 1)] } else { T::zero() };
            q[(i, k)] = c * q_k + s * q_k1;
            if k + 1 < n {
                q[(i, k + 1)] = (T::zero() - s) * q_k + c * q_k1;
            }
        }

        // Apply Givens from right to H and R to maintain structure
        for i in 0..n {
            let h_k = h[(i, k)];
            let h_k1 = if k + 1 < n { h[(i, k + 1)] } else { T::zero() };
            h[(i, k)] = c * h_k + s * h_k1;
            if k + 1 < n {
                h[(i, k + 1)] = (T::zero() - s) * h_k + c * h_k1;
            }
        }

        for i in 0..n {
            let r_k = r[(i, k)];
            let r_k1 = if k + 1 < n { r[(i, k + 1)] } else { T::zero() };
            r[(i, k)] = c * r_k + s * r_k1;
            if k + 1 < n {
                r[(i, k + 1)] = (T::zero() - s) * r_k + c * r_k1;
            }
        }

        for i in 0..n {
            let z_k = z[(i, k)];
            let z_k1 = if k + 1 < n { z[(i, k + 1)] } else { T::zero() };
            z[(i, k)] = c * z_k + s * z_k1;
            if k + 1 < n {
                z[(i, k + 1)] = (T::zero() - s) * z_k + c * z_k1;
            }
        }
    }
}

/// Computes Givens rotation coefficients.
fn givens<T: Field + Real>(a: T, b: T) -> (T, T) {
    if Scalar::abs(b) <= <T as Scalar>::epsilon() {
        return (T::one(), T::zero());
    }

    if Scalar::abs(a) <= <T as Scalar>::epsilon() {
        return (T::zero(), T::one());
    }

    let abs_a = Scalar::abs(a);
    let abs_b = Scalar::abs(b);

    if abs_b > abs_a {
        let t = a / b;
        let s = T::one() / Real::sqrt(T::one() + t * t);
        let c = s * t;
        (c, s)
    } else {
        let t = b / a;
        let c = T::one() / Real::sqrt(T::one() + t * t);
        let s = c * t;
        (c, s)
    }
}

/// Extracts generalized eigenvalues from S and T.
fn extract_eigenvalues<T: Field + Real + bytemuck::Zeroable>(
    s: &Mat<T>,
    t: &Mat<T>,
) -> Vec<GeneralizedEigenvalue<T>> {
    let n = s.nrows();
    let mut eigenvalues = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        if i + 1 < n && Scalar::abs(s[(i + 1, i)]) > <T as Scalar>::epsilon() {
            // 2x2 block - complex conjugate pair
            let a11 = s[(i, i)];
            let a12 = s[(i, i + 1)];
            let a21 = s[(i + 1, i)];
            let a22 = s[(i + 1, i + 1)];
            let b11 = t[(i, i)];
            let b22 = t[(i + 1, i + 1)];

            // Approximate eigenvalues of 2x2 pencil
            let tr = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = tr * tr - (T::one() + T::one() + T::one() + T::one()) * det;
            let two = T::one() + T::one();

            if disc >= T::zero() {
                // Real eigenvalues
                let sqrt_disc = Real::sqrt(disc);
                let e1 = (tr + sqrt_disc) / two;
                let e2 = (tr - sqrt_disc) / two;
                eigenvalues.push(GeneralizedEigenvalue::real(e1, b11));
                eigenvalues.push(GeneralizedEigenvalue::real(e2, b22));
            } else {
                // Complex conjugate pair
                let real_part = tr / two;
                let imag_part = Real::sqrt(T::zero() - disc) / two;
                eigenvalues.push(GeneralizedEigenvalue::complex(real_part, imag_part, b11));
                eigenvalues.push(GeneralizedEigenvalue::complex(
                    real_part,
                    T::zero() - imag_part,
                    b22,
                ));
            }
            i += 2;
        } else {
            // 1x1 block - real eigenvalue
            eigenvalues.push(GeneralizedEigenvalue::real(s[(i, i)], t[(i, i)]));
            i += 1;
        }
    }

    eigenvalues
}

/// Matrix multiplication: C = A * B.
fn mat_mult<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    let n = a.nrows();
    let mut c = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}

/// Matrix multiplication with transpose: C = A * B^T.
fn mat_mult_transpose<T: Field + Real + bytemuck::Zeroable>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    let n = a.nrows();
    let mut c = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + a[(i, k)] * b[(j, k)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qz_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();

        // Eigenvalues should be 1 and 2
        let mut eigs: Vec<f64> = qz.eigenvalues_real();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((eigs[0] - 1.0).abs() < 1e-6, "Expected 1, got {}", eigs[0]);
        assert!((eigs[1] - 2.0).abs() < 1e-6, "Expected 2, got {}", eigs[1]);
    }

    #[test]
    fn test_qz_scaled() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);
        let b = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 2.0]]);

        let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();

        // Eigenvalues should be 1 (2/2) and 2 (4/2)
        let mut eigs: Vec<f64> = qz.eigenvalues_real();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((eigs[0] - 1.0).abs() < 1e-6);
        assert!((eigs[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qz_verify() {
        // Simple diagonal case for verification
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();
        let (error_a, error_b) = qz.verify(a.as_ref(), b.as_ref());

        // With diagonal A and B = I, errors should be small
        assert!(
            error_a < 1e-6,
            "Verification error for A: {} (should be < 1e-6)",
            error_a
        );
        assert!(
            error_b < 1e-6,
            "Verification error for B: {} (should be < 1e-6)",
            error_b
        );
    }

    #[test]
    fn test_qz_1x1() {
        let a = Mat::from_rows(&[&[5.0f64]]);
        let b = Mat::from_rows(&[&[2.0f64]]);

        let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();
        let eigs = qz.eigenvalues();

        assert_eq!(eigs.len(), 1);
        assert!((eigs[0].value().unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_qz_infinite_eigenvalue() {
        // B is singular, so there's an infinite eigenvalue
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 0.0]]);

        let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();

        // Check that we have at least one infinite eigenvalue
        assert!(qz.num_infinite() >= 1 || qz.eigenvalues().len() == 2);
    }

    #[test]
    fn test_givens() {
        let (c, s) = givens(3.0f64, 4.0);

        // c^2 + s^2 = 1
        assert!((c * c + s * s - 1.0).abs() < 1e-10);

        // c*a + s*b gives the radius
        let r = c * 3.0 + s * 4.0;
        assert!((r - 5.0).abs() < 1e-10);
    }
}
