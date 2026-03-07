//! Schur decomposition.
//!
//! Computes the Schur decomposition A = Q T Q^T where Q is orthogonal
//! and T is quasi-upper triangular (upper triangular with possible 2×2 blocks
//! on the diagonal for complex eigenvalue pairs).

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::hessenberg::Hessenberg;

/// Error type for Schur decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for SchurError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::NotConverged => write!(f, "Schur decomposition did not converge"),
        }
    }
}

impl std::error::Error for SchurError {}

/// Represents a real or complex eigenvalue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigenvalue<T> {
    /// Real part of the eigenvalue.
    pub real: T,
    /// Imaginary part of the eigenvalue (zero for real eigenvalues).
    pub imag: T,
}

impl<T: Scalar> Eigenvalue<T> {
    /// Creates a real eigenvalue.
    pub fn real_only(value: T) -> Self {
        Self {
            real: value,
            imag: T::zero(),
        }
    }

    /// Creates a complex eigenvalue.
    pub fn complex(real: T, imag: T) -> Self {
        Self { real, imag }
    }

    /// Returns true if this is a real eigenvalue.
    pub fn is_real(&self) -> bool {
        self.imag == T::zero()
    }
}

/// Schur decomposition of a matrix.
///
/// For a matrix A, computes A = Q T Q^T where:
/// - Q is orthogonal (Q^T Q = I)
/// - T is quasi-upper triangular (real Schur form)
#[derive(Debug, Clone)]
pub struct Schur<T: Scalar> {
    /// The orthogonal matrix Q (Schur vectors).
    q: Mat<T>,
    /// The quasi-upper triangular matrix T.
    t: Mat<T>,
    /// Eigenvalues (real and complex pairs).
    eigenvalues: Vec<Eigenvalue<T>>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> Schur<T> {
    /// Maximum iterations for QR iteration.
    const MAX_ITERATIONS: usize = 100;

    /// Computes the Schur decomposition of a square matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Schur;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[0.0, 3.0],
    /// ]);
    ///
    /// let schur = Schur::compute(a.as_ref()).unwrap();
    /// let eigenvalues = schur.eigenvalues();
    ///
    /// // Eigenvalues are 1 and 3
    /// assert!((eigenvalues[0].real - 1.0).abs() < 1e-10 || (eigenvalues[0].real - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SchurError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SchurError::EmptyMatrix);
        }

        if m != n {
            return Err(SchurError::NotSquare);
        }

        // Handle 1×1 case
        if n == 1 {
            let mut t = Mat::zeros(1, 1);
            t[(0, 0)] = a[(0, 0)];
            let mut q = Mat::zeros(1, 1);
            q[(0, 0)] = T::one();
            let eigenvalues = vec![Eigenvalue::real_only(a[(0, 0)])];
            return Ok(Self {
                q,
                t,
                eigenvalues,
                n,
            });
        }

        // Handle 2×2 case
        if n == 2 {
            return Self::compute_2x2(a);
        }

        // Step 1: Reduce to upper Hessenberg form
        let hess = Hessenberg::compute(a).map_err(|_| SchurError::NotSquare)?;
        let mut t = Mat::zeros(n, n);
        let h = hess.h();
        for i in 0..n {
            for j in 0..n {
                t[(i, j)] = h[(i, j)];
            }
        }

        let mut q = Mat::zeros(n, n);
        let q_hess = hess.q();
        for i in 0..n {
            for j in 0..n {
                q[(i, j)] = q_hess[(i, j)];
            }
        }

        // Step 2: Apply QR iteration with implicit shifts
        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

        // Process from bottom to top
        let mut p = n;
        let mut iter_count = 0;

        while p > 2 && iter_count < Self::MAX_ITERATIONS * n {
            iter_count += 1;

            // Find the active block
            let mut q_idx = p - 1;
            while q_idx > 0 {
                let sub = Scalar::abs(t[(q_idx, q_idx - 1)]);
                let diag_sum =
                    Scalar::abs(t[(q_idx - 1, q_idx - 1)]) + Scalar::abs(t[(q_idx, q_idx)]);
                if sub <= tol * diag_sum {
                    t[(q_idx, q_idx - 1)] = T::zero();
                    break;
                }
                q_idx -= 1;
            }

            if q_idx == p - 1 {
                // 1×1 block converged
                p -= 1;
            } else if q_idx == p - 2 {
                // Check if 2×2 block has converged (complex eigenvalues)
                let a11 = t[(p - 2, p - 2)];
                let a12 = t[(p - 2, p - 1)];
                let a21 = t[(p - 1, p - 2)];
                let a22 = t[(p - 1, p - 1)];
                let trace = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                let disc = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;

                if disc < T::zero() {
                    // Complex eigenvalues, keep as 2×2 block
                    p -= 2;
                } else {
                    // Real eigenvalues, continue iteration
                    Self::francis_qr_step(&mut t, &mut q, q_idx, p);
                }
            } else {
                // Apply Francis QR step
                Self::francis_qr_step(&mut t, &mut q, q_idx, p);
            }
        }

        // Handle remaining 2×2 block if needed
        if p == 2 {
            let sub = Scalar::abs(t[(1, 0)]);
            let diag_sum = Scalar::abs(t[(0, 0)]) + Scalar::abs(t[(1, 1)]);
            if sub <= tol * diag_sum {
                t[(1, 0)] = T::zero();
            }
        }

        // Extract eigenvalues
        let eigenvalues = Self::extract_eigenvalues(&t);

        Ok(Self {
            q,
            t,
            eigenvalues,
            n,
        })
    }

    /// Computes Schur decomposition for 2×2 matrix.
    fn compute_2x2(a: MatRef<'_, T>) -> Result<Self, SchurError> {
        let mut t = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                t[(i, j)] = a[(i, j)];
            }
        }

        let a11 = a[(0, 0)];
        let a12 = a[(0, 1)];
        let a21 = a[(1, 0)];
        let a22 = a[(1, 1)];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;
        let disc = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;

        let mut q = Mat::zeros(2, 2);
        let eigenvalues: Vec<Eigenvalue<T>>;

        if disc >= T::zero() {
            // Real eigenvalues - triangularize
            let sqrt_disc = Real::sqrt(disc);
            let lambda1 = (trace + sqrt_disc) / T::from_f64(2.0).unwrap_or_else(T::zero);
            let lambda2 = (trace - sqrt_disc) / T::from_f64(2.0).unwrap_or_else(T::zero);

            // Find rotation to triangularize
            if Scalar::abs(a21) > <T as Scalar>::epsilon() {
                let theta = if Scalar::abs(a11 - lambda1) > <T as Scalar>::epsilon() {
                    Real::atan2(a21, a11 - lambda1)
                } else {
                    T::from_f64(core::f64::consts::FRAC_PI_2).unwrap_or_else(T::zero)
                };
                let c = Real::cos(theta);
                let s = Real::sin(theta);

                q[(0, 0)] = c;
                q[(0, 1)] = -s;
                q[(1, 0)] = s;
                q[(1, 1)] = c;

                // T = Q^T * A * Q
                let mut temp = Mat::zeros(2, 2);
                // temp = Q^T * A
                for i in 0..2 {
                    for j in 0..2 {
                        let mut sum = T::zero();
                        for k in 0..2 {
                            sum = sum + q[(k, i)] * a[(k, j)];
                        }
                        temp[(i, j)] = sum;
                    }
                }
                // t = temp * Q
                for i in 0..2 {
                    for j in 0..2 {
                        let mut sum = T::zero();
                        for k in 0..2 {
                            sum = sum + temp[(i, k)] * q[(k, j)];
                        }
                        t[(i, j)] = sum;
                    }
                }
            } else {
                // Already triangular
                q[(0, 0)] = T::one();
                q[(1, 1)] = T::one();
            }

            eigenvalues = vec![
                Eigenvalue::real_only(lambda1),
                Eigenvalue::real_only(lambda2),
            ];
        } else {
            // Complex eigenvalues - keep as 2×2 block
            let sqrt_disc = Real::sqrt(-disc);
            let real_part = trace / T::from_f64(2.0).unwrap_or_else(T::zero);
            let imag_part = sqrt_disc / T::from_f64(2.0).unwrap_or_else(T::zero);

            q[(0, 0)] = T::one();
            q[(1, 1)] = T::one();

            eigenvalues = vec![
                Eigenvalue::complex(real_part, imag_part),
                Eigenvalue::complex(real_part, -imag_part),
            ];
        }

        Ok(Self {
            q,
            t,
            eigenvalues,
            n: 2,
        })
    }

    /// Applies one step of Francis double-shift QR iteration.
    fn francis_qr_step(t: &mut Mat<T>, q: &mut Mat<T>, start: usize, end: usize) {
        let n = t.nrows();

        if end - start < 2 {
            return;
        }

        // Compute shift from bottom 2×2 block
        let h11 = t[(end - 2, end - 2)];
        let h12 = t[(end - 2, end - 1)];
        let h21 = t[(end - 1, end - 2)];
        let h22 = t[(end - 1, end - 1)];

        let s = h11 + h22; // trace
        let p = h11 * h22 - h12 * h21; // determinant

        // First column of (H - s1*I)(H - s2*I) = H² - s*H + p*I
        let h_00 = t[(start, start)];
        let h_01 = t[(start, start + 1)];
        let h_10 = t[(start + 1, start)];

        let mut x = h_00 * h_00 + h_01 * h_10 - s * h_00 + p;
        let mut y = h_10 * (h_00 + t[(start + 1, start + 1)] - s);
        let mut z = if start + 2 < end {
            h_10 * t[(start + 2, start + 1)]
        } else {
            T::zero()
        };

        // Chase the bulge
        for k in start..end.saturating_sub(1) {
            // Compute Householder to zero out y, z
            let (v, tau) = householder_3(&[x, y, z]);

            if tau != T::zero() {
                let r = if k > start { k - 1 } else { k };

                // Apply from left: T := (I - tau * v * v^T) * T
                let col_start = r;
                let col_end = n;
                for j in col_start..col_end {
                    let rows = (k..(k + 3).min(end)).collect::<Vec<_>>();
                    let mut dot = T::zero();
                    for (vi, &row) in rows.iter().enumerate() {
                        dot = dot + v[vi] * t[(row, j)];
                    }
                    let scaled = tau * dot;
                    for (vi, &row) in rows.iter().enumerate() {
                        t[(row, j)] = t[(row, j)] - scaled * v[vi];
                    }
                }

                // Apply from right: T := T * (I - tau * v * v^T)
                let row_end = (k + 4).min(end);
                for i in 0..row_end {
                    let cols = (k..(k + 3).min(end)).collect::<Vec<_>>();
                    let mut dot = T::zero();
                    for (vi, &col) in cols.iter().enumerate() {
                        dot = dot + t[(i, col)] * v[vi];
                    }
                    let scaled = tau * dot;
                    for (vi, &col) in cols.iter().enumerate() {
                        t[(i, col)] = t[(i, col)] - scaled * v[vi];
                    }
                }

                // Accumulate Q
                for i in 0..n {
                    let cols = (k..(k + 3).min(end)).collect::<Vec<_>>();
                    let mut dot = T::zero();
                    for (vi, &col) in cols.iter().enumerate() {
                        dot = dot + q[(i, col)] * v[vi];
                    }
                    let scaled = tau * dot;
                    for (vi, &col) in cols.iter().enumerate() {
                        q[(i, col)] = q[(i, col)] - scaled * v[vi];
                    }
                }
            }

            // Prepare for next iteration
            if k + 3 < end {
                x = t[(k + 1, k)];
                y = t[(k + 2, k)];
                z = if k + 3 < end {
                    t[(k + 3, k)]
                } else {
                    T::zero()
                };
            } else if k + 2 < end {
                // 2×2 Householder for the last step
                x = t[(k + 1, k)];
                y = t[(k + 2, k)];
                z = T::zero();
            }
        }
    }

    /// Extracts eigenvalues from the quasi-upper triangular Schur form.
    fn extract_eigenvalues(t: &Mat<T>) -> Vec<Eigenvalue<T>> {
        let n = t.nrows();
        let mut eigenvalues = Vec::with_capacity(n);
        let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

        let mut i = 0;
        while i < n {
            if i == n - 1 {
                // Last element is a 1×1 block
                eigenvalues.push(Eigenvalue::real_only(t[(i, i)]));
                i += 1;
            } else {
                // Check if this is a 2×2 block
                let sub = Scalar::abs(t[(i + 1, i)]);
                let diag_sum = Scalar::abs(t[(i, i)]) + Scalar::abs(t[(i + 1, i + 1)]);

                if sub <= eps * diag_sum {
                    // 1×1 block
                    eigenvalues.push(Eigenvalue::real_only(t[(i, i)]));
                    i += 1;
                } else {
                    // 2×2 block - compute eigenvalues
                    let a11 = t[(i, i)];
                    let a12 = t[(i, i + 1)];
                    let a21 = t[(i + 1, i)];
                    let a22 = t[(i + 1, i + 1)];

                    let trace = a11 + a22;
                    let det = a11 * a22 - a12 * a21;
                    let disc = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;

                    if disc >= T::zero() {
                        // Real eigenvalues
                        let sqrt_disc = Real::sqrt(disc);
                        let lambda1 =
                            (trace + sqrt_disc) / T::from_f64(2.0).unwrap_or_else(T::zero);
                        let lambda2 =
                            (trace - sqrt_disc) / T::from_f64(2.0).unwrap_or_else(T::zero);
                        eigenvalues.push(Eigenvalue::real_only(lambda1));
                        eigenvalues.push(Eigenvalue::real_only(lambda2));
                    } else {
                        // Complex conjugate pair
                        let sqrt_disc = Real::sqrt(-disc);
                        let real_part = trace / T::from_f64(2.0).unwrap_or_else(T::zero);
                        let imag_part = sqrt_disc / T::from_f64(2.0).unwrap_or_else(T::zero);
                        eigenvalues.push(Eigenvalue::complex(real_part, imag_part));
                        eigenvalues.push(Eigenvalue::complex(real_part, -imag_part));
                    }
                    i += 2;
                }
            }
        }

        eigenvalues
    }

    /// Returns the orthogonal matrix Q (Schur vectors).
    pub fn q(&self) -> MatRef<'_, T> {
        self.q.as_ref()
    }

    /// Returns the quasi-upper triangular matrix T (Schur form).
    pub fn t(&self) -> MatRef<'_, T> {
        self.t.as_ref()
    }

    /// Returns the eigenvalues.
    pub fn eigenvalues(&self) -> &[Eigenvalue<T>] {
        &self.eigenvalues
    }

    /// Returns only the real parts of eigenvalues.
    pub fn eigenvalues_real(&self) -> Vec<T> {
        self.eigenvalues.iter().map(|e| e.real).collect()
    }

    /// Reconstructs the original matrix: A = Q T Q^T.
    pub fn reconstruct(&self) -> Mat<T> {
        let mut qt = Mat::zeros(self.n, self.n);
        let mut a = Mat::zeros(self.n, self.n);

        // QT = Q * T
        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k in 0..self.n {
                    sum = sum + self.q[(i, k)] * self.t[(k, j)];
                }
                qt[(i, j)] = sum;
            }
        }

        // A = QT * Q^T
        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k in 0..self.n {
                    sum = sum + qt[(i, k)] * self.q[(j, k)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }

    /// Computes right eigenvectors from the Schur form (LAPACK TREVC).
    ///
    /// Returns the right eigenvectors V such that T V = V D where D is the
    /// diagonal matrix of eigenvalues. The eigenvectors are normalized.
    ///
    /// For real eigenvalues, returns real eigenvectors.
    /// For complex conjugate pairs, returns two columns: the real and imaginary
    /// parts of the eigenvector.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Schur;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[0.0, 4.0, 5.0],
    ///     &[0.0, 0.0, 6.0],
    /// ]);
    ///
    /// let schur = Schur::compute(a.as_ref()).unwrap();
    /// let vr = schur.right_eigenvectors();
    /// // Each column of vr is a right eigenvector
    /// ```
    #[must_use]
    pub fn right_eigenvectors(&self) -> Mat<T> {
        trevc_right(&self.t)
    }

    /// Computes left eigenvectors from the Schur form (LAPACK TREVC).
    ///
    /// Returns the left eigenvectors U such that U^T T = D U^T where D is the
    /// diagonal matrix of eigenvalues. The eigenvectors are normalized.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Schur;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[0.0, 4.0, 5.0],
    ///     &[0.0, 0.0, 6.0],
    /// ]);
    ///
    /// let schur = Schur::compute(a.as_ref()).unwrap();
    /// let vl = schur.left_eigenvectors();
    /// ```
    #[must_use]
    pub fn left_eigenvectors(&self) -> Mat<T> {
        trevc_left(&self.t)
    }

    /// Computes eigenvectors of the original matrix A = Q T Q^T.
    ///
    /// The eigenvectors of A are Q * V where V are the eigenvectors of T.
    ///
    /// # Returns
    ///
    /// (right_eigenvectors, left_eigenvectors) of A.
    #[must_use]
    pub fn eigenvectors(&self) -> (Mat<T>, Mat<T>) {
        let vr_t = trevc_right(&self.t);
        let vl_t = trevc_left(&self.t);

        // Transform to eigenvectors of A: A_vr = Q * T_vr, A_vl = Q * T_vl
        let mut vr_a = Mat::zeros(self.n, self.n);
        let mut vl_a = Mat::zeros(self.n, self.n);

        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum_r = T::zero();
                let mut sum_l = T::zero();
                for k in 0..self.n {
                    sum_r = sum_r + self.q[(i, k)] * vr_t[(k, j)];
                    sum_l = sum_l + self.q[(i, k)] * vl_t[(k, j)];
                }
                vr_a[(i, j)] = sum_r;
                vl_a[(i, j)] = sum_l;
            }
        }

        (vr_a, vl_a)
    }

    /// Computes reciprocal condition numbers for eigenvalues (LAPACK DTRSNA).
    ///
    /// For each eigenvalue λ, the reciprocal condition number s measures how
    /// sensitive λ is to perturbations in the matrix. A small value indicates
    /// a poorly conditioned eigenvalue.
    ///
    /// The condition number is computed as:
    /// - For simple eigenvalues: s = 1 / |y^H x| where x is the right eigenvector
    ///   and y is the left eigenvector, both normalized to unit length.
    /// - For complex conjugate pairs: uses the average of the pair.
    ///
    /// # Returns
    ///
    /// Vector of reciprocal condition numbers, one per eigenvalue.
    /// Smaller values indicate more sensitive eigenvalues.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::Schur;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 0.0],
    ///     &[0.0, 1000.0],
    /// ]);
    ///
    /// let schur = Schur::compute(a.as_ref()).unwrap();
    /// let cond = schur.eigenvalue_condition_numbers();
    /// // Both eigenvalues of a diagonal matrix are well-conditioned
    /// assert!(cond[0] > 0.9);
    /// assert!(cond[1] > 0.9);
    /// ```
    #[must_use]
    pub fn eigenvalue_condition_numbers(&self) -> Vec<T> {
        trsna_s(&self.t)
    }

    /// Computes reciprocal condition numbers for eigenvectors (LAPACK DTRSNA).
    ///
    /// For each right eigenvector x_j, computes the separation sep_j which
    /// measures how close the eigenvalue is to the rest of the spectrum.
    ///
    /// # Returns
    ///
    /// Vector of separation values, one per eigenvalue.
    /// Smaller values indicate more sensitive eigenvectors.
    #[must_use]
    pub fn eigenvector_separation(&self) -> Vec<T> {
        trsna_sep(&self.t)
    }
}

/// Computes reciprocal condition numbers for eigenvalues (s values from LAPACK DTRSNA).
///
/// For each simple eigenvalue λ_j, s_j = |y_j^H * x_j| where x_j is the right
/// eigenvector and y_j is the left eigenvector, both normalized.
///
/// # Arguments
///
/// * `t` - The quasi-upper triangular Schur matrix
///
/// # Returns
///
/// Vector of reciprocal condition numbers for each eigenvalue.
pub fn trsna_s<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Vec<T> {
    let n = t.nrows();
    if n == 0 {
        return Vec::new();
    }

    let vr = trevc_right(t);
    let vl = trevc_left(t);

    let mut s = vec![T::zero(); n];
    let eps = <T as Scalar>::epsilon();

    let mut j = 0;
    while j < n {
        // Check for 2×2 block
        let is_2x2 = if j + 1 < n {
            let sub = Scalar::abs(t[(j + 1, j)]);
            let diag_sum = Scalar::abs(t[(j, j)]) + Scalar::abs(t[(j + 1, j + 1)]);
            sub > eps * T::from_f64(100.0).unwrap_or_else(T::zero) * diag_sum
        } else {
            false
        };

        if is_2x2 {
            // Complex conjugate pair: compute s = |y^H * x| using both columns
            // For complex eigenvector stored as (real, imag) in consecutive columns:
            // y^H * x = (yr - i*yi)^T * (xr + i*xi) = (yr^T*xr + yi^T*xi) + i*(yr^T*xi - yi^T*xr)
            let jp1 = j + 1;

            let mut prod_rr = T::zero(); // yr^T * xr
            let mut prod_ii = T::zero(); // yi^T * xi
            let mut prod_ri = T::zero(); // yr^T * xi
            let mut prod_ir = T::zero(); // yi^T * xr

            for k in 0..n {
                prod_rr = prod_rr + vl[(k, j)] * vr[(k, j)];
                prod_ii = prod_ii + vl[(k, jp1)] * vr[(k, jp1)];
                prod_ri = prod_ri + vl[(k, j)] * vr[(k, jp1)];
                prod_ir = prod_ir + vl[(k, jp1)] * vr[(k, j)];
            }

            let real_part = prod_rr + prod_ii;
            let imag_part = prod_ri - prod_ir;
            let abs_inner = Real::sqrt(real_part * real_part + imag_part * imag_part);

            // Both eigenvalues in the pair have the same condition number
            s[j] = abs_inner;
            s[jp1] = abs_inner;

            j += 2;
        } else {
            // Simple real eigenvalue: s = |y^T * x|
            let mut inner = T::zero();
            for k in 0..n {
                inner = inner + vl[(k, j)] * vr[(k, j)];
            }
            s[j] = Scalar::abs(inner);
            j += 1;
        }
    }

    s
}

/// Computes separation (sep) for eigenvectors (from LAPACK DTRSNA).
///
/// For each eigenvalue λ_j, sep_j = σ_min(T_22 - λ_j * I) where T_22 is the
/// (n-1)×(n-1) trailing principal submatrix with λ_j removed.
///
/// This measures how separated λ_j is from the rest of the spectrum.
///
/// # Arguments
///
/// * `t` - The quasi-upper triangular Schur matrix
///
/// # Returns
///
/// Vector of separation values for each eigenvalue.
pub fn trsna_sep<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Vec<T> {
    let n = t.nrows();
    if n == 0 {
        return Vec::new();
    }

    let mut sep = vec![T::zero(); n];
    let eps = <T as Scalar>::epsilon();

    let mut j = 0;
    while j < n {
        // Check for 2×2 block
        let is_2x2 = if j + 1 < n {
            let sub = Scalar::abs(t[(j + 1, j)]);
            let diag_sum = Scalar::abs(t[(j, j)]) + Scalar::abs(t[(j + 1, j + 1)]);
            sub > eps * T::from_f64(100.0).unwrap_or_else(T::zero) * diag_sum
        } else {
            false
        };

        if is_2x2 {
            // Complex conjugate pair
            let jp1 = j + 1;

            // Compute approximate separation as minimum distance to other eigenvalues
            let a11 = t[(j, j)];
            let a22 = t[(jp1, jp1)];
            let lambda_real = (a11 + a22) / T::from_f64(2.0).unwrap_or_else(T::zero);
            let trace = a11 + a22;
            let det = a11 * a22 - t[(j, jp1)] * t[(jp1, j)];
            let disc = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;
            let lambda_imag = if disc < T::zero() {
                Real::sqrt(-disc) / T::from_f64(2.0).unwrap_or_else(T::zero)
            } else {
                T::zero()
            };

            let mut min_sep = T::one() / eps;

            // Check distance to all other eigenvalues
            let mut k = 0;
            while k < n {
                if k == j || k == jp1 {
                    k += 1;
                    continue;
                }

                // Check if k is part of a 2×2 block
                let adjacent = j > 0 && k == j - 1;
                let k_is_2x2 = if k + 1 < n && !adjacent {
                    let sub = Scalar::abs(t[(k + 1, k)]);
                    let diag_sum = Scalar::abs(t[(k, k)]) + Scalar::abs(t[(k + 1, k + 1)]);
                    sub > eps * T::from_f64(100.0).unwrap_or_else(T::zero) * diag_sum
                } else {
                    false
                };

                let (other_real, other_imag) = if k_is_2x2 {
                    let kp1 = k + 1;
                    let b11 = t[(k, k)];
                    let b22 = t[(kp1, kp1)];
                    let other_trace = b11 + b22;
                    let other_det = b11 * b22 - t[(k, kp1)] * t[(kp1, k)];
                    let other_disc = other_trace * other_trace
                        - T::from_f64(4.0).unwrap_or_else(T::zero) * other_det;
                    let r = (b11 + b22) / T::from_f64(2.0).unwrap_or_else(T::zero);
                    let i = if other_disc < T::zero() {
                        Real::sqrt(-other_disc) / T::from_f64(2.0).unwrap_or_else(T::zero)
                    } else {
                        T::zero()
                    };
                    (r, i)
                } else {
                    (t[(k, k)], T::zero())
                };

                // Distance between eigenvalues
                let dr = lambda_real - other_real;
                let di = lambda_imag - other_imag;
                let dist = Real::sqrt(dr * dr + di * di);
                if dist < min_sep && dist > T::zero() {
                    min_sep = dist;
                }

                // Also check conjugate
                if other_imag != T::zero() {
                    let di_conj = lambda_imag + other_imag;
                    let dist_conj = Real::sqrt(dr * dr + di_conj * di_conj);
                    if dist_conj < min_sep && dist_conj > T::zero() {
                        min_sep = dist_conj;
                    }
                }

                if k_is_2x2 {
                    k += 2;
                } else {
                    k += 1;
                }
            }

            sep[j] = min_sep;
            sep[jp1] = min_sep;
            j += 2;
        } else {
            // Simple real eigenvalue
            let lambda = t[(j, j)];
            let mut min_sep = T::one() / eps;

            // Check distance to all other eigenvalues
            let mut k = 0;
            while k < n {
                if k == j {
                    k += 1;
                    continue;
                }

                // Check if k is part of a 2×2 block
                let adjacent_to_j = (j > 0 && k == j - 1) || k == j + 1;
                let k_is_2x2 = if k + 1 < n && !adjacent_to_j {
                    let sub = Scalar::abs(t[(k + 1, k)]);
                    let diag_sum = Scalar::abs(t[(k, k)]) + Scalar::abs(t[(k + 1, k + 1)]);
                    sub > eps * T::from_f64(100.0).unwrap_or_else(T::zero) * diag_sum
                } else {
                    false
                };

                let (other_real, other_imag) = if k_is_2x2 {
                    let kp1 = k + 1;
                    let b11 = t[(k, k)];
                    let b22 = t[(kp1, kp1)];
                    let other_trace = b11 + b22;
                    let other_det = b11 * b22 - t[(k, kp1)] * t[(kp1, k)];
                    let other_disc = other_trace * other_trace
                        - T::from_f64(4.0).unwrap_or_else(T::zero) * other_det;
                    let r = (b11 + b22) / T::from_f64(2.0).unwrap_or_else(T::zero);
                    let i = if other_disc < T::zero() {
                        Real::sqrt(-other_disc) / T::from_f64(2.0).unwrap_or_else(T::zero)
                    } else {
                        T::zero()
                    };
                    (r, i)
                } else {
                    (t[(k, k)], T::zero())
                };

                // Distance between eigenvalues
                let dr = lambda - other_real;
                let dist = Real::sqrt(dr * dr + other_imag * other_imag);
                if dist < min_sep && dist > T::zero() {
                    min_sep = dist;
                }

                if k_is_2x2 {
                    k += 2;
                } else {
                    k += 1;
                }
            }

            sep[j] = min_sep;
            j += 1;
        }
    }

    sep
}

/// Computes right eigenvectors of a quasi-upper triangular matrix (LAPACK DTREVC).
///
/// The input T must be in real Schur form (quasi-upper triangular with 1×1 and
/// 2×2 diagonal blocks).
///
/// # Arguments
///
/// * `t` - The quasi-upper triangular Schur matrix
///
/// # Returns
///
/// Matrix V of right eigenvectors (column-wise). For complex conjugate pairs,
/// consecutive columns contain the real and imaginary parts.
pub fn trevc_right<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Mat<T> {
    let n = t.nrows();
    let mut v = Mat::zeros(n, n);

    // Initialize to identity for back-substitution starting point
    for i in 0..n {
        v[(i, i)] = T::one();
    }

    let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

    // Process eigenvalues from last to first
    let mut j = n;
    while j > 0 {
        j -= 1;

        // Check if this is part of a 2×2 block
        let is_2x2 = if j > 0 {
            let sub = Scalar::abs(t[(j, j - 1)]);
            let diag_sum = Scalar::abs(t[(j - 1, j - 1)]) + Scalar::abs(t[(j, j)]);
            sub > eps * diag_sum
        } else {
            false
        };

        if is_2x2 {
            // 2×2 block: complex conjugate eigenvalues
            // Process columns j-1 and j together
            let jm1 = j - 1;

            // Eigenvalues of 2×2 block
            let a11 = t[(jm1, jm1)];
            let a12 = t[(jm1, j)];
            let a21 = t[(j, jm1)];
            let a22 = t[(j, j)];

            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;

            // Complex eigenvalues: λ = (trace ± i*sqrt(-disc)) / 2
            let two = T::from_f64(2.0).unwrap_or_else(T::zero);
            let real_part = trace / two;
            let imag_part = Real::sqrt(-disc) / two;

            // For the 2×2 block, set up eigenvector components
            // v[jm1] = real part, v[j] = imag part for first eigenvector
            v[(jm1, jm1)] = T::one();
            v[(j, jm1)] = T::zero();
            v[(jm1, j)] = T::zero();
            v[(j, j)] = T::one();

            // Compute actual eigenvector of 2×2 block
            // (a11 - λ) v1 + a12 v2 = 0
            // a21 v1 + (a22 - λ) v2 = 0
            // For λ = real_part + i*imag_part:
            // Let v = vr + i*vi
            // Then: (T - λI)(vr + i*vi) = 0
            // Real: (T - real_part*I)vr + imag_part*vi = 0
            // Imag: (T - real_part*I)vi - imag_part*vr = 0

            // For the 2×2 block itself, find normalized eigenvector
            // The eigenvector satisfies (T - λI)v = 0 where λ = real_part + i*imag_part
            // For complex eigenvector v = vr + i*vi:
            // (T - real_part*I)vr + imag_part*vi = 0  (real part)
            // (T - real_part*I)vi - imag_part*vr = 0  (imag part)
            //
            // For the 2×2 case, we can set v2 = 1 + 0i and solve for v1
            // From row 2: a21*v1r + (a22-real)*v2r + imag*v2i = 0 (real)
            //             a21*v1i + (a22-real)*v2i - imag*v2r = 0 (imag)
            // With v2r=1, v2i=0:
            //   a21*v1r + (a22-real) = 0  =>  v1r = -(a22-real)/a21 = -d22/a21
            //   a21*v1i - imag = 0        =>  v1i = imag/a21

            // Use the more stable formulation based on LAPACK
            // For standardized eigenvector, use row with larger coefficient
            let d11 = a11 - real_part;
            let d22 = a22 - real_part;

            // Choose the row with larger off-diagonal to avoid division by small number
            if Scalar::abs(a21) >= Scalar::abs(a12) && Scalar::abs(a21) > eps {
                // Use row 2: a21*v1 + d22*v2 = 0 (with v2=1)
                // v1r = -d22/a21, v1i = imag/a21
                let v1r = -d22 / a21;
                let v1i = imag_part / a21;
                v[(jm1, jm1)] = v1r;
                v[(j, jm1)] = T::one();
                v[(jm1, j)] = v1i;
                v[(j, j)] = T::zero();
            } else if Scalar::abs(a12) > eps {
                // Use row 1: d11*v1 + a12*v2 = 0 (with v1=1)
                // v2r = -d11/a12, v2i = -imag/a12
                let v2r = -d11 / a12;
                let v2i = -imag_part / a12;
                v[(jm1, jm1)] = T::one();
                v[(j, jm1)] = v2r;
                v[(jm1, j)] = T::zero();
                v[(j, j)] = v2i;
            } else {
                // Fallback to standard basis
                v[(jm1, jm1)] = T::one();
                v[(j, jm1)] = T::zero();
                v[(jm1, j)] = T::zero();
                v[(j, j)] = T::one();
            }

            // Back-substitute for rows above the 2×2 block
            for i in (0..jm1).rev() {
                // Solve for v[i,jm1] and v[i,j] (real and imag parts)
                // (T[i,i] - real_part) * vr[i] + imag_part * vi[i] = -sum of upper terms (real)
                // (T[i,i] - real_part) * vi[i] - imag_part * vr[i] = -sum of upper terms (imag)

                let mut sum_r = T::zero();
                let mut sum_i = T::zero();
                for k in (i + 1)..=j {
                    sum_r = sum_r + t[(i, k)] * v[(k, jm1)];
                    sum_i = sum_i + t[(i, k)] * v[(k, j)];
                }

                let d = t[(i, i)] - real_part;
                let det_2x2 = d * d + imag_part * imag_part;

                if Scalar::abs(det_2x2) > eps {
                    // Solve 2×2 system:
                    // [d, imag] [vr]   [-sum_r]
                    // [-imag, d] [vi] = [-sum_i]
                    v[(i, jm1)] = (-d * sum_r - imag_part * sum_i) / det_2x2;
                    v[(i, j)] = (imag_part * sum_r - d * sum_i) / det_2x2;
                }
            }

            // Normalize the two columns
            let mut norm_r_sq = T::zero();
            let mut norm_i_sq = T::zero();
            for i in 0..n {
                norm_r_sq = norm_r_sq + v[(i, jm1)] * v[(i, jm1)];
                norm_i_sq = norm_i_sq + v[(i, j)] * v[(i, j)];
            }
            let norm = Real::sqrt(norm_r_sq + norm_i_sq);
            if norm > T::zero() {
                for i in 0..n {
                    v[(i, jm1)] = v[(i, jm1)] / norm;
                    v[(i, j)] = v[(i, j)] / norm;
                }
            }

            j = jm1; // Skip the already-processed column
        } else {
            // 1×1 block: real eigenvalue
            let lambda = t[(j, j)];

            // Initialize: v[j,j] = 1, others computed by back-substitution
            v[(j, j)] = T::one();

            // Back-substitute: (T[i,i] - λ) v[i] = -sum_{k>i} T[i,k] v[k]
            for i in (0..j).rev() {
                let mut sum = T::zero();
                for k in (i + 1)..=j {
                    sum = sum + t[(i, k)] * v[(k, j)];
                }

                let d = t[(i, i)] - lambda;
                if Scalar::abs(d) > eps {
                    v[(i, j)] = -sum / d;
                } else {
                    // Near-singular: use small perturbation
                    v[(i, j)] = -sum / eps;
                }
            }

            // Normalize
            let mut norm_sq = T::zero();
            for i in 0..n {
                norm_sq = norm_sq + v[(i, j)] * v[(i, j)];
            }
            let norm = Real::sqrt(norm_sq);
            if norm > T::zero() {
                for i in 0..n {
                    v[(i, j)] = v[(i, j)] / norm;
                }
            }
        }
    }

    v
}

/// Computes left eigenvectors of a quasi-upper triangular matrix (LAPACK DTREVC).
///
/// The input T must be in real Schur form. Returns left eigenvectors U such that
/// U^T T = D U^T where D is diagonal.
///
/// # Arguments
///
/// * `t` - The quasi-upper triangular Schur matrix
///
/// # Returns
///
/// Matrix U of left eigenvectors (column-wise).
pub fn trevc_left<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Mat<T> {
    let n = t.nrows();
    let mut v = Mat::zeros(n, n);

    // Initialize
    for i in 0..n {
        v[(i, i)] = T::one();
    }

    let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

    // Process eigenvalues from first to last (forward substitution for left eigenvectors)
    let mut j = 0;
    while j < n {
        // Check if this is part of a 2×2 block
        let is_2x2 = if j + 1 < n {
            let sub = Scalar::abs(t[(j + 1, j)]);
            let diag_sum = Scalar::abs(t[(j, j)]) + Scalar::abs(t[(j + 1, j + 1)]);
            sub > eps * diag_sum
        } else {
            false
        };

        if is_2x2 {
            // 2×2 block: complex conjugate eigenvalues
            let jp1 = j + 1;

            let a11 = t[(j, j)];
            let a12 = t[(j, jp1)];
            let a21 = t[(jp1, j)];
            let a22 = t[(jp1, jp1)];

            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;

            let two = T::from_f64(2.0).unwrap_or_else(T::zero);
            let real_part = trace / two;
            let imag_part = Real::sqrt(-disc) / two;

            // Initialize 2×2 block eigenvector
            let d11 = a11 - real_part;
            let det_factor = d11 * d11 + imag_part * imag_part;

            if Scalar::abs(det_factor) > eps {
                let vr2 = -a21 * d11 / det_factor;
                let vi2 = -imag_part * a21 / det_factor;
                v[(j, j)] = T::one();
                v[(jp1, j)] = vr2;
                v[(j, jp1)] = T::zero();
                v[(jp1, jp1)] = vi2;
            } else {
                v[(j, j)] = T::one();
                v[(jp1, j)] = T::zero();
                v[(j, jp1)] = T::zero();
                v[(jp1, jp1)] = T::one();
            }

            // Forward substitute for rows after the 2×2 block
            for i in (jp1 + 1)..n {
                let mut sum_r = T::zero();
                let mut sum_i = T::zero();
                for k in j..i {
                    sum_r = sum_r + t[(k, i)] * v[(k, j)];
                    sum_i = sum_i + t[(k, i)] * v[(k, jp1)];
                }

                let d = t[(i, i)] - real_part;
                let det_2x2 = d * d + imag_part * imag_part;

                if Scalar::abs(det_2x2) > eps {
                    v[(i, j)] = (-d * sum_r - imag_part * sum_i) / det_2x2;
                    v[(i, jp1)] = (imag_part * sum_r - d * sum_i) / det_2x2;
                }
            }

            // Normalize
            let mut norm_sq = T::zero();
            for i in 0..n {
                norm_sq = norm_sq + v[(i, j)] * v[(i, j)] + v[(i, jp1)] * v[(i, jp1)];
            }
            let norm = Real::sqrt(norm_sq);
            if norm > T::zero() {
                for i in 0..n {
                    v[(i, j)] = v[(i, j)] / norm;
                    v[(i, jp1)] = v[(i, jp1)] / norm;
                }
            }

            j = jp1 + 1;
        } else {
            // 1×1 block: real eigenvalue
            let lambda = t[(j, j)];
            v[(j, j)] = T::one();

            // Forward substitute: (T[i,i] - λ) v[i] = -sum_{k<i} T[k,i] v[k]
            for i in (j + 1)..n {
                let mut sum = T::zero();
                for k in j..i {
                    sum = sum + t[(k, i)] * v[(k, j)];
                }

                let d = t[(i, i)] - lambda;
                if Scalar::abs(d) > eps {
                    v[(i, j)] = -sum / d;
                } else {
                    v[(i, j)] = -sum / eps;
                }
            }

            // Normalize
            let mut norm_sq = T::zero();
            for i in 0..n {
                norm_sq = norm_sq + v[(i, j)] * v[(i, j)];
            }
            let norm = Real::sqrt(norm_sq);
            if norm > T::zero() {
                for i in 0..n {
                    v[(i, j)] = v[(i, j)] / norm;
                }
            }

            j += 1;
        }
    }

    v
}

/// Computes a Householder vector for a 3-element (or smaller) vector.
fn householder_3<T: Field + Real>(x: &[T]) -> (Vec<T>, T) {
    let n = x.len().min(3);
    if n == 0 {
        return (Vec::new(), T::zero());
    }

    let mut norm_sq = T::zero();
    for i in 0..n {
        norm_sq = norm_sq + x[i] * x[i];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (vec![T::zero(); n], T::zero());
    }

    let mut v = vec![T::zero(); n];
    for i in 0..n {
        v[i] = x[i];
    }

    let sign = if x[0] >= T::zero() {
        T::one()
    } else {
        -T::one()
    };
    v[0] = v[0] + sign * norm;

    let mut v_norm_sq = T::zero();
    for i in 0..n {
        v_norm_sq = v_norm_sq + v[i] * v[i];
    }

    if v_norm_sq > T::zero() {
        let tau = T::from_f64(2.0).unwrap_or_else(T::zero) / v_norm_sq;
        (v, tau)
    } else {
        (v, T::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_schur_upper_triangular() {
        // Already upper triangular
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[0.0, 3.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let eigenvalues = schur.eigenvalues();

        // Eigenvalues should be 1 and 3
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_schur_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let eigenvalues = schur.eigenvalues();

        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 5.0, 1e-10));
    }

    #[test]
    fn test_schur_complex_eigenvalues() {
        // Rotation matrix - has complex eigenvalues
        let theta = core::f64::consts::FRAC_PI_4; // 45 degrees
        let c = theta.cos();
        let s = theta.sin();
        let a = Mat::from_rows(&[&[c, -s], &[s, c]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let eigenvalues = schur.eigenvalues();

        // Eigenvalues should be cos(θ) ± i*sin(θ)
        assert_eq!(eigenvalues.len(), 2);
        // They should be complex conjugates
        assert!(approx_eq(eigenvalues[0].real, eigenvalues[1].real, 1e-10));
        assert!(approx_eq(eigenvalues[0].imag, -eigenvalues[1].imag, 1e-10));
        assert!(approx_eq(eigenvalues[0].real, c, 1e-10));
        assert!(approx_eq(eigenvalues[0].imag.abs(), s, 1e-10));
    }

    #[test]
    fn test_schur_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let reconstructed = schur.reconstruct();

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
    fn test_schur_q_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let q = schur.q();

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
    fn test_schur_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);
        let schur = Schur::compute(a.as_ref()).unwrap();

        assert_eq!(schur.eigenvalues().len(), 1);
        assert!(approx_eq(schur.eigenvalues()[0].real, 5.0, 1e-10));
    }

    #[test]
    fn test_schur_4x4() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, 1.0],
            &[-2.0, 0.0, 3.0, -2.0],
            &[2.0, 1.0, -2.0, -1.0],
        ]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let reconstructed = schur.reconstruct();

        for i in 0..4 {
            for j in 0..4 {
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
    fn test_schur_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let reconstructed = schur.reconstruct();

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

    #[test]
    fn test_trevc_right_upper_triangular() {
        // Upper triangular matrix - eigenvectors should be standard basis vectors
        let t = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        let v = trevc_right(&t);

        // First eigenvector for λ=1 should be proportional to [1, 0, 0]
        assert!(v[(1, 0)].abs() < 1e-10);
        assert!(v[(2, 0)].abs() < 1e-10);

        // Third eigenvector for λ=6 should be proportional to [*, *, 1]
        // (normalized, so third component is non-zero)
        assert!(v[(2, 2)].abs() > 0.1);
    }

    #[test]
    fn test_trevc_right_diagonal() {
        // Diagonal matrix - eigenvectors are standard basis
        let t = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let v = trevc_right(&t);

        // Should be identity (or close to it with normalization)
        for i in 0..3 {
            assert!(
                approx_eq(v[(i, i)].abs(), 1.0, 1e-10),
                "v[{},{}] = {}",
                i,
                i,
                v[(i, i)]
            );
        }
    }

    #[test]
    fn test_trevc_eigenvector_equation() {
        // Test that T * v = λ * v for computed eigenvectors
        let t = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        let v = trevc_right(&t);

        // Eigenvalues are 1, 4, 6 (diagonal elements)
        let eigenvalues = [1.0, 4.0, 6.0];

        for (j, &lambda) in eigenvalues.iter().enumerate() {
            // Compute T * v[:,j]
            let mut tv = [0.0; 3];
            for i in 0..3 {
                for k in 0..3 {
                    tv[i] += t[(i, k)] * v[(k, j)];
                }
            }

            // Check T * v = λ * v
            for i in 0..3 {
                assert!(
                    approx_eq(tv[i], lambda * v[(i, j)], 1e-10),
                    "T*v[{}] = {}, λ*v = {}",
                    i,
                    tv[i],
                    lambda * v[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_trevc_left_diagonal() {
        let t = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let u = trevc_left(&t);

        // Should be identity
        for i in 0..3 {
            assert!(
                approx_eq(u[(i, i)].abs(), 1.0, 1e-10),
                "u[{},{}] = {}",
                i,
                i,
                u[(i, i)]
            );
        }
    }

    #[test]
    fn test_schur_eigenvectors() {
        // Test eigenvectors through the Schur decomposition
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let (vr, _vl) = schur.eigenvectors();

        // For upper triangular matrix, A = T, so eigenvectors are the same
        // Check that A * v = λ * v
        for j in 0..3 {
            let lambda = schur.eigenvalues()[j].real;

            let mut av = [0.0; 3];
            for i in 0..3 {
                for k in 0..3 {
                    av[i] += a[(i, k)] * vr[(k, j)];
                }
            }

            for i in 0..3 {
                assert!(
                    approx_eq(av[i], lambda * vr[(i, j)], 1e-8),
                    "A*v[{}] = {}, λ*v = {}",
                    i,
                    av[i],
                    lambda * vr[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_trevc_2x2_block() {
        // Create a 2x2 block with complex eigenvalues: rotation matrix
        let theta = core::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();
        let t = Mat::from_rows(&[&[c, -s], &[s, c]]);

        let v = trevc_right(&t);

        // For complex eigenvalues, columns should contain real and imag parts
        // The eigenvector equation (T - λI)v = 0 where λ = c + i*s
        // Check that both columns are normalized
        let norm0_sq = v[(0, 0)] * v[(0, 0)] + v[(1, 0)] * v[(1, 0)];
        let norm1_sq = v[(0, 1)] * v[(0, 1)] + v[(1, 1)] * v[(1, 1)];
        let total_norm = (norm0_sq + norm1_sq).sqrt();

        assert!(
            approx_eq(total_norm, 1.0, 1e-10),
            "eigenvector norm = {}",
            total_norm
        );
    }

    #[test]
    fn test_trsna_s_diagonal() {
        // Diagonal matrix: left and right eigenvectors are standard basis
        // so s = |e_i^T e_i| = 1 for all eigenvalues
        let t = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let s = trsna_s(&t);

        assert_eq!(s.len(), 3);
        for i in 0..3 {
            assert!(
                approx_eq(s[i], 1.0, 1e-10),
                "s[{}] = {}, expected 1.0",
                i,
                s[i]
            );
        }
    }

    #[test]
    fn test_trsna_s_upper_triangular() {
        // Upper triangular: eigenvalues are diagonal elements
        let t = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        let s = trsna_s(&t);

        assert_eq!(s.len(), 3);
        // All condition numbers should be positive
        for i in 0..3 {
            assert!(s[i] > 0.0, "s[{}] = {} should be positive", i, s[i]);
        }
    }

    #[test]
    fn test_trsna_sep_diagonal() {
        // Diagonal matrix: separation is the minimum distance to other eigenvalues
        let t = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let sep = trsna_sep(&t);

        assert_eq!(sep.len(), 3);
        // λ=2: min dist to 3,5 is 1
        assert!(
            approx_eq(sep[0], 1.0, 1e-10),
            "sep[0] = {}, expected 1.0",
            sep[0]
        );
        // λ=5: min dist to 2,3 is 2
        assert!(
            approx_eq(sep[1], 2.0, 1e-10),
            "sep[1] = {}, expected 2.0",
            sep[1]
        );
        // λ=3: min dist to 2,5 is 1
        assert!(
            approx_eq(sep[2], 1.0, 1e-10),
            "sep[2] = {}, expected 1.0",
            sep[2]
        );
    }

    #[test]
    fn test_trsna_sep_close_eigenvalues() {
        // Eigenvalues very close together should have small separation
        let t = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.001]]);

        let sep = trsna_sep(&t);

        assert_eq!(sep.len(), 2);
        // Both should have separation close to 0.001
        assert!(
            approx_eq(sep[0], 0.001, 1e-10),
            "sep[0] = {}, expected 0.001",
            sep[0]
        );
        assert!(
            approx_eq(sep[1], 0.001, 1e-10),
            "sep[1] = {}, expected 0.001",
            sep[1]
        );
    }

    #[test]
    fn test_schur_eigenvalue_condition_numbers() {
        // Test through Schur decomposition
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 2.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let cond = schur.eigenvalue_condition_numbers();

        assert_eq!(cond.len(), 2);
        // Diagonal matrix should have well-conditioned eigenvalues
        for i in 0..2 {
            assert!(
                cond[i] > 0.5,
                "cond[{}] = {} should be > 0.5 for diagonal matrix",
                i,
                cond[i]
            );
        }
    }

    #[test]
    fn test_schur_eigenvector_separation() {
        // Test through Schur decomposition
        let a = Mat::from_rows(&[&[4.0f64, 0.0], &[0.0, 2.0]]);

        let schur = Schur::compute(a.as_ref()).unwrap();
        let sep = schur.eigenvector_separation();

        assert_eq!(sep.len(), 2);
        // Eigenvalues are 2 and 4, so separation should be 2
        for i in 0..2 {
            assert!(
                approx_eq(sep[i], 2.0, 1e-10),
                "sep[{}] = {}, expected 2.0",
                i,
                sep[i]
            );
        }
    }

    #[test]
    fn test_trsna_complex_eigenvalues() {
        // Rotation matrix with complex eigenvalues
        let theta = core::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();
        let t = Mat::from_rows(&[&[c, -s], &[s, c]]);

        let cond = trsna_s(&t);
        let sep = trsna_sep(&t);

        assert_eq!(cond.len(), 2);
        assert_eq!(sep.len(), 2);

        // Both eigenvalues in a complex pair should have the same condition
        assert!(
            approx_eq(cond[0], cond[1], 1e-10),
            "cond[0]={}, cond[1]={} should be equal",
            cond[0],
            cond[1]
        );
        assert!(
            approx_eq(sep[0], sep[1], 1e-10),
            "sep[0]={}, sep[1]={} should be equal",
            sep[0],
            sep[1]
        );
    }
}
