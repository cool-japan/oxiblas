//! Schur decomposition.
//!
//! Computes the Schur decomposition A = Q T Q^T where Q is orthogonal
//! and T is quasi-upper triangular (upper triangular with possible 2×2 blocks
//! on the diagonal for complex eigenvalue pairs).
//!
//! `T` is returned in *standardized* real Schur form (as LAPACK `DHSEQR`):
//! every entry below the first subdiagonal is zero, a subdiagonal entry is
//! non-zero exactly where a 2×2 diagonal block starts, and every 2×2 block
//! holds a complex-conjugate pair with equal diagonal entries and off-diagonal
//! entries of opposite sign. The eigenvalues are listed in the order of the
//! diagonal of `T`, a complex pair with its positive imaginary part first.

mod qr;
mod trevc;

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::hessenberg::Hessenberg;

pub(crate) use qr::{big_number, scale_by_pow2, scaling_exponent, small_number};
pub(crate) use trevc::{ladiv, left_eigenvectors_raw, right_eigenvectors_raw, starts_block};
pub use trevc::{trevc_left, trevc_right};

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
/// - T is quasi-upper triangular (real Schur form) in standardized form: every
///   entry below the first subdiagonal is zero, a subdiagonal entry is
///   non-zero exactly where a 2×2 diagonal block starts, and every 2×2 block
///   holds a complex-conjugate pair `a ± i·sqrt(|b·c|)` as `[a b; c a]` with
///   `b·c < 0` (as LAPACK `DHSEQR`)
///
/// `eigenvalues()[k]` is the eigenvalue at diagonal position `k` of `T`; a
/// complex pair is listed with its positive imaginary part first.
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
    /// // Eigenvalues are 1 and 3, listed in the order of T's diagonal
    /// let t = schur.t();
    /// assert_eq!(eigenvalues[0].real, t[(0, 0)]);
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

        // The iteration budget scales with n — `MAX_ITERATIONS` sweeps per row
        // on average (`saturating_mul` guards the pathological huge-`n` case
        // against usize overflow).
        Self::compute_with_budget(a, Self::MAX_ITERATIONS.saturating_mul(n))
    }

    /// Reduces the square, non-empty `a` to standardized real Schur form:
    /// exact power-of-two scaling into a safe range, Hessenberg reduction
    /// (`n ≥ 3`), and the Francis double-shift QR iteration with deflation,
    /// exceptional shifts and 2×2 block standardization (LAPACK `DLAHQR`).
    ///
    /// `max_total_iterations` bounds the *total* number of QR sweeps across all
    /// deflation blocks. If that budget is exhausted while an unreduced block
    /// of order `>= 3` remains, this returns [`SchurError::NotConverged`]
    /// instead of silently handing back a partially reduced `T`/`Q` as if the
    /// decomposition had succeeded — matching the `INFO > 0` failure contract
    /// of LAPACK's DHSEQR. 1×1 and 2×2 blocks never need a sweep.
    fn compute_with_budget(
        a: MatRef<'_, T>,
        max_total_iterations: usize,
    ) -> Result<Self, SchurError> {
        let n = a.nrows();

        // Scale by an exact power of two so that max|a_ij| ∈ [1, 2): every
        // threshold below then acts relative to the matrix norm, and the
        // intermediate quantities can neither overflow nor underflow early.
        let mut anrm = T::zero();
        for j in 0..n {
            for i in 0..n {
                anrm = Float::max(anrm, Scalar::abs(a[(i, j)]));
            }
        }
        let exponent = scaling_exponent(anrm);
        let mut scaled = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                scaled[(i, j)] = scale_by_pow2(a[(i, j)], exponent);
            }
        }

        let (mut t, mut q) = if n >= 3 {
            let hess = Hessenberg::compute(scaled.as_ref()).map_err(|_| SchurError::NotSquare)?;
            let mut t = Mat::zeros(n, n);
            let mut q = Mat::zeros(n, n);
            let (h, qh) = (hess.h(), hess.q());
            for j in 0..n {
                for i in 0..n {
                    t[(i, j)] = h[(i, j)];
                    q[(i, j)] = qh[(i, j)];
                }
            }
            (t, q)
        } else {
            let mut q = Mat::zeros(n, n);
            for i in 0..n {
                q[(i, i)] = T::one();
            }
            (scaled, q)
        };

        qr::hessenberg_qr(&mut t, &mut q, max_total_iterations)?;

        if exponent != 0 {
            for j in 0..n {
                for i in 0..n {
                    t[(i, j)] = scale_by_pow2(t[(i, j)], -exponent);
                }
            }
        }

        let eigenvalues = Self::extract_eigenvalues(&t);

        Ok(Self {
            q,
            t,
            eigenvalues,
            n,
        })
    }

    /// Test-only hook: runs the Schur decomposition with a custom QR iteration
    /// budget so the [`SchurError::NotConverged`] path can be exercised
    /// deterministically (a genuine slow-converging matrix would otherwise need to
    /// be enormous to defeat the default `MAX_ITERATIONS * n` budget).
    #[cfg(test)]
    pub(crate) fn compute_with_iteration_budget(
        a: MatRef<'_, T>,
        max_total_iterations: usize,
    ) -> Result<Self, SchurError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SchurError::EmptyMatrix);
        }
        if m != n {
            return Err(SchurError::NotSquare);
        }
        // Tiny cases deflate without any sweep; the budget is irrelevant there.
        Self::compute_with_budget(a, max_total_iterations)
    }

    /// Reads the eigenvalues off a standardized quasi-triangular `t`, in the
    /// order of its diagonal: `t[k, k]` for a 1×1 block, and
    /// `t[k, k] ± i·sqrt(|t[k, k+1]|)·sqrt(|t[k+1, k]|)` (positive imaginary
    /// part first) for a 2×2 block at `(k, k + 1)`, exactly as LAPACK `DLANV2`
    /// reports them.
    fn extract_eigenvalues(t: &Mat<T>) -> Vec<Eigenvalue<T>> {
        let n = t.nrows();
        let mut eigenvalues = Vec::with_capacity(n);
        let mut k = 0;
        while k < n {
            if starts_block(t, k) {
                let re = t[(k, k)];
                let im =
                    Real::sqrt(Scalar::abs(t[(k, k + 1)])) * Real::sqrt(Scalar::abs(t[(k + 1, k)]));
                eigenvalues.push(Eigenvalue::complex(re, im));
                eigenvalues.push(Eigenvalue::complex(re, -im));
                k += 2;
            } else {
                eigenvalues.push(Eigenvalue::real_only(t[(k, k)]));
                k += 1;
            }
        }
        eigenvalues
    }

    /// Returns the orthogonal matrix Q (Schur vectors).
    pub fn q(&self) -> MatRef<'_, T> {
        self.q.as_ref()
    }

    /// The Schur vectors as an owned-matrix reference (crate-internal).
    pub(crate) fn q_mat(&self) -> &Mat<T> {
        &self.q
    }

    /// The standardized Schur factor as an owned-matrix reference
    /// (crate-internal).
    pub(crate) fn t_mat(&self) -> &Mat<T> {
        &self.t
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

/// Returns the standardized form of `t` (see [`trevc_right`]); `t` itself when
/// it is already standardized.
fn standardized_view<T: Field + Real + bytemuck::Zeroable>(
    t: &Mat<T>,
) -> std::borrow::Cow<'_, Mat<T>> {
    match trevc::standardize(t) {
        Some((ts, _)) => std::borrow::Cow::Owned(ts),
        None => std::borrow::Cow::Borrowed(t),
    }
}

/// Eigenvalue `(re, im)` of the diagonal block of a standardized `t` that
/// starts at `k` (the positive-imaginary member for a 2×2 block).
fn block_eigenvalue<T: Field + Real>(t: &Mat<T>, k: usize) -> (T, T) {
    if starts_block(t, k) {
        let im = Real::sqrt(Scalar::abs(t[(k, k + 1)])) * Real::sqrt(Scalar::abs(t[(k + 1, k)]));
        (t[(k, k)], im)
    } else {
        (t[(k, k)], T::zero())
    }
}

/// Computes reciprocal condition numbers for eigenvalues (s values from LAPACK DTRSNA).
///
/// For each simple eigenvalue λ_j, s_j = |y_j^H * x_j| where x_j is the right
/// eigenvector and y_j is the left eigenvector, both normalized.
///
/// # Arguments
///
/// * `t` - The quasi-upper triangular Schur matrix (see [`trevc_right`] for
///   the accepted forms; a non-standardized 2×2 block is standardized first,
///   which leaves the condition numbers unchanged)
///
/// # Returns
///
/// Vector of reciprocal condition numbers for each eigenvalue.
pub fn trsna_s<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Vec<T> {
    let n = t.nrows();
    if n == 0 {
        return Vec::new();
    }

    let t = standardized_view(t);
    let t = t.as_ref();
    let vr = trevc_right(t);
    let vl = trevc_left(t);

    let mut s = vec![T::zero(); n];

    let mut j = 0;
    while j < n {
        if starts_block(t, j) {
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
            let abs_inner = Real::hypot(real_part, imag_part);

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
/// * `t` - The quasi-upper triangular Schur matrix (see [`trevc_right`] for
///   the accepted forms)
///
/// # Returns
///
/// Vector of separation values for each eigenvalue.
pub fn trsna_sep<T: Field + Real + bytemuck::Zeroable>(t: &Mat<T>) -> Vec<T> {
    let n = t.nrows();
    if n == 0 {
        return Vec::new();
    }

    let t = standardized_view(t);
    let t = t.as_ref();
    let mut sep = vec![T::zero(); n];
    let eps = <T as Scalar>::epsilon();

    let mut j = 0;
    while j < n {
        let width = if starts_block(t, j) { 2 } else { 1 };
        let (lambda_real, lambda_imag) = block_eigenvalue(t, j);
        let mut min_sep = T::one() / eps;

        // Distance to every other eigenvalue (both members of other pairs).
        let mut k = 0;
        while k < n {
            let k_width = if starts_block(t, k) { 2 } else { 1 };
            if k != j {
                let (other_real, other_imag) = block_eigenvalue(t, k);
                let dr = lambda_real - other_real;
                let dist = Real::hypot(dr, lambda_imag - other_imag);
                if dist < min_sep && dist > T::zero() {
                    min_sep = dist;
                }
                if other_imag != T::zero() {
                    let dist_conj = Real::hypot(dr, lambda_imag + other_imag);
                    if dist_conj < min_sep && dist_conj > T::zero() {
                        min_sep = dist_conj;
                    }
                }
            }
            k += k_width;
        }

        for s in &mut sep[j..j + width] {
            *s = min_sep;
        }
        j += width;
    }

    sep
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

    /// A 5×5 non-symmetric matrix with tightly clustered eigenvalues. It is
    /// perfectly solvable, but a single QR sweep cannot possibly deflate it down
    /// to a size-2 window, so an artificially tiny iteration budget must be
    /// reported as a convergence failure — not silently returned as garbage.
    ///
    /// The eigenvalues cluster near 5 (this is `5*I` plus a small nilpotent-ish
    /// bidiagonal + coupling perturbation), which is exactly the kind of spectrum
    /// that makes the QR iteration work hardest.
    fn slow_converging_5x5() -> Mat<f64> {
        Mat::from_rows(&[
            &[5.0, 1.0, 0.2, 0.0, 0.1],
            &[0.3, 5.0, 1.0, 0.15, 0.0],
            &[0.0, 0.25, 5.0, 1.0, 0.2],
            &[0.1, 0.0, 0.3, 5.0, 1.0],
            &[0.2, 0.1, 0.0, 0.35, 5.0],
        ])
    }

    #[test]
    fn test_schur_reports_non_convergence_when_budget_exhausted() {
        let a = slow_converging_5x5();

        // A total budget of 1 QR sweep is nowhere near enough to reduce a 5×5
        // active window (which needs several sweeps and multiple deflations).
        // Before the fix this fell through and returned a partially-reduced T as
        // if valid; now it must surface NotConverged.
        let result = Schur::compute_with_iteration_budget(a.as_ref(), 1);
        assert_eq!(
            result.err(),
            Some(SchurError::NotConverged),
            "tiny iteration budget must report NotConverged, not fabricate a result"
        );
    }

    #[test]
    fn test_schur_converges_with_full_budget() {
        // Control for the non-convergence test: the SAME matrix must converge
        // (and reconstruct as A = Q T Q^T) under the real default budget, proving
        // the error above is caused purely by the artificial cap, not a defect in
        // the matrix or the algorithm.
        let a = slow_converging_5x5();

        let schur = Schur::compute(a.as_ref()).expect("full budget must converge");
        let rec = schur.reconstruct();
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    approx_eq(rec[(i, j)], a[(i, j)], 1e-8),
                    "reconstruction mismatch at ({i},{j}): {} vs {}",
                    rec[(i, j)],
                    a[(i, j)]
                );
            }
        }

        // Trace is preserved by a similarity transform: sum of eigenvalue real
        // parts must equal the trace (= 25 here).
        let trace: f64 = (0..5).map(|i| a[(i, i)]).sum();
        let eig_sum: f64 = schur.eigenvalues().iter().map(|e| e.real).sum();
        assert!(
            approx_eq(eig_sum, trace, 1e-8),
            "eigenvalue real-part sum {eig_sum} != trace {trace}"
        );
    }
}
