//! General (non-symmetric) eigenvalue decomposition.
//!
//! Computes eigenvalues and eigenvectors of general (possibly non-symmetric) matrices.
//! Unlike symmetric EVD, eigenvalues may be complex even for real matrices.
//!
//! The decomposition follows LAPACK `DGEEV`: the matrix is scaled by an exact
//! power of two, reduced to standardized real Schur form `A = Q·T·Qᵀ`, the
//! eigenvectors of `T` are computed by back-substitution (`DTREVC`, with
//! perturbation of near-singular pivots and overflow-safe rescaling), and
//! transformed back with `Q`.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::schur::{
    Eigenvalue, Schur, SchurError, left_eigenvectors_raw, right_eigenvectors_raw, scale_by_pow2,
    scaling_exponent, starts_block,
};

/// Error type for general eigenvalue decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralEvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for GeneralEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::NotConverged => write!(f, "Eigenvalue algorithm did not converge"),
        }
    }
}

impl std::error::Error for GeneralEvdError {}

impl From<SchurError> for GeneralEvdError {
    fn from(e: SchurError) -> Self {
        match e {
            SchurError::EmptyMatrix => Self::EmptyMatrix,
            SchurError::NotSquare => Self::NotSquare,
            SchurError::NotConverged => Self::NotConverged,
        }
    }
}

/// General eigenvalue decomposition.
///
/// For a general matrix A, computes eigenvalues and optionally eigenvectors.
/// Note that for non-symmetric real matrices, eigenvalues may be complex,
/// and eigenvectors are computed from the Schur decomposition.
///
/// Right eigenvectors satisfy: A * v = λ * v
/// Left eigenvectors satisfy: u^H * A = λ * u^H (for a real matrix this is
/// A^T * u = conj(λ) * u, i.e. A^T * u = λ * u when λ is real)
///
/// # Eigenvector layout
///
/// Eigenvalue `k` (`eigenvalues()[k]`) owns column `k` of the eigenvector
/// matrices: its right eigenvector is
/// `v_k = eigenvectors_real()[:, k] + i·eigenvectors_imag()[:, k]` and its
/// left eigenvector is
/// `u_k = left_eigenvectors_real()[:, k] + i·left_eigenvectors_imag()[:, k]`.
///
/// - A real eigenvalue has a real eigenvector (the imaginary column is zero).
/// - A complex-conjugate pair occupies two consecutive positions `k, k + 1`,
///   listed as `a + bi` (`b > 0`) then `a − bi`. The eigenvectors of the
///   second member are the conjugates of those of the first: columns `k` and
///   `k + 1` of the real parts are equal, and the imaginary parts are negated.
///
/// Every eigenvector has unit Euclidean norm, and a complex eigenvector is
/// scaled so that its component of largest modulus is real and positive (as
/// LAPACK `DGEEV`). This layout differs from LAPACK's packed `VR`/`VL`
/// storage, where a pair's real and imaginary parts share one matrix.
///
/// Eigenvalues are listed in the order of the diagonal of the real Schur
/// form computed by [`Schur`].
#[derive(Debug, Clone)]
pub struct GeneralEvd<T: Scalar> {
    /// Eigenvalues (may be complex).
    eigenvalues: Vec<Eigenvalue<T>>,
    /// Real parts of the right eigenvectors, one column per eigenvalue.
    eigenvectors_real: Option<Mat<T>>,
    /// Imaginary parts of the right eigenvectors, one column per eigenvalue.
    eigenvectors_imag: Option<Mat<T>>,
    /// Real parts of the left eigenvectors, one column per eigenvalue.
    left_eigenvectors_real: Option<Mat<T>>,
    /// Imaginary parts of the left eigenvectors, one column per eigenvalue.
    left_eigenvectors_imag: Option<Mat<T>>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> GeneralEvd<T> {
    /// Computes only eigenvalues of a general matrix (no eigenvectors).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::GeneralEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    ///
    /// let evd = GeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
    /// let eigenvalues = evd.eigenvalues();
    ///
    /// // Sum of eigenvalues equals trace
    /// let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
    /// assert!((sum - 5.0).abs() < 1e-10); // trace = 1 + 4 = 5
    /// ```
    pub fn eigenvalues_only(a: MatRef<'_, T>) -> Result<Self, GeneralEvdError> {
        let (schur, exponent) = Self::scaled_schur(a)?;
        Ok(Self {
            eigenvalues: Self::unscaled_eigenvalues(&schur, exponent),
            eigenvectors_real: None,
            eigenvectors_imag: None,
            left_eigenvectors_real: None,
            left_eigenvectors_imag: None,
            n: a.nrows(),
        })
    }

    /// Computes eigenvalues and right eigenvectors of a general matrix.
    ///
    /// The eigenvector of `eigenvalues()[k]` is
    /// `eigenvectors_real()[:, k] + i·eigenvectors_imag()[:, k]` (see the
    /// [type-level documentation](GeneralEvd#eigenvector-layout)). For a
    /// complex pair `λ = a ± bi` at positions `k, k + 1` the eigenvector of
    /// `a + bi` is `v_real + i·v_imag` and that of `a − bi` is
    /// `v_real − i·v_imag`.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::GeneralEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    ///
    /// let evd = GeneralEvd::compute(a.as_ref()).unwrap();
    /// let vr = evd.eigenvectors_real().unwrap();
    ///
    /// // A * v_k = λ_k * v_k for every eigenpair (both eigenvalues are real here).
    /// for (k, lambda) in evd.eigenvalues().iter().enumerate() {
    ///     for i in 0..2 {
    ///         let av = a[(i, 0)] * vr[(0, k)] + a[(i, 1)] * vr[(1, k)];
    ///         assert!((av - lambda.real * vr[(i, k)]).abs() < 1e-12);
    ///     }
    /// }
    /// assert!(evd.verify(a.as_ref()) < 1e-12);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, GeneralEvdError> {
        Self::compute_with_vectors(a, false)
    }

    /// Computes eigenvalues, right eigenvectors, and left eigenvectors of a general matrix.
    ///
    /// Right eigenvectors satisfy: A * v = λ * v
    /// Left eigenvectors satisfy: u^H * A = λ * u^H (for a real matrix,
    /// A^T * u = conj(λ) * u)
    ///
    /// Both are paired with the eigenvalue of the same index (see the
    /// [type-level documentation](GeneralEvd#eigenvector-layout)).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::GeneralEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    ///
    /// let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();
    /// let vr = evd.eigenvectors_real().unwrap();
    /// let vl = evd.left_eigenvectors_real().unwrap();
    ///
    /// // u_k^T * A = λ_k * u_k^T (real eigenvalues)
    /// for (k, lambda) in evd.eigenvalues().iter().enumerate() {
    ///     for j in 0..2 {
    ///         let ua = vl[(0, k)] * a[(0, j)] + vl[(1, k)] * a[(1, j)];
    ///         assert!((ua - lambda.real * vl[(j, k)]).abs() < 1e-12);
    ///     }
    /// }
    /// # let _ = vr;
    /// ```
    pub fn compute_full(a: MatRef<'_, T>) -> Result<Self, GeneralEvdError> {
        Self::compute_with_vectors(a, true)
    }

    /// Validates `a`, scales it by an exact power of two so that
    /// `max|a_ij| ∈ [1, 2)` (no-op for a zero or non-finite maximum) and
    /// computes the Schur decomposition of the scaled matrix. Returns the
    /// decomposition and the scaling exponent `k` (the matrix decomposed is
    /// `2^k·A`).
    fn scaled_schur(a: MatRef<'_, T>) -> Result<(Schur<T>, i32), GeneralEvdError> {
        let n = a.nrows();
        if n == 0 || a.ncols() == 0 {
            return Err(GeneralEvdError::EmptyMatrix);
        }
        if a.ncols() != n {
            return Err(GeneralEvdError::NotSquare);
        }
        let mut anrm = T::zero();
        for j in 0..n {
            for i in 0..n {
                anrm = Float::max(anrm, Scalar::abs(a[(i, j)]));
            }
        }
        let exponent = scaling_exponent(anrm);
        if exponent == 0 {
            return Ok((Schur::compute(a)?, 0));
        }
        let mut scaled = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                scaled[(i, j)] = scale_by_pow2(a[(i, j)], exponent);
            }
        }
        Ok((Schur::compute(scaled.as_ref())?, exponent))
    }

    /// Eigenvalues of `A` from the Schur decomposition of `2^exponent·A`.
    fn unscaled_eigenvalues(schur: &Schur<T>, exponent: i32) -> Vec<Eigenvalue<T>> {
        schur
            .eigenvalues()
            .iter()
            .map(|e| {
                Eigenvalue::complex(
                    scale_by_pow2(e.real, -exponent),
                    scale_by_pow2(e.imag, -exponent),
                )
            })
            .collect()
    }

    /// Shared implementation of [`compute`](Self::compute) and
    /// [`compute_full`](Self::compute_full).
    fn compute_with_vectors(a: MatRef<'_, T>, want_left: bool) -> Result<Self, GeneralEvdError> {
        let (schur, exponent) = Self::scaled_schur(a)?;
        let t = schur.t_mat();
        let q = schur.q_mat();

        let right = right_eigenvectors_raw(t);
        let (vr, vi) = back_transform_split(q, t, &right);

        let (left_re, left_im) = if want_left {
            let left = left_eigenvectors_raw(t);
            let (ur, ui) = back_transform_split(q, t, &left);
            (Some(ur), Some(ui))
        } else {
            (None, None)
        };

        Ok(Self {
            eigenvalues: Self::unscaled_eigenvalues(&schur, exponent),
            eigenvectors_real: Some(vr),
            eigenvectors_imag: Some(vi),
            left_eigenvectors_real: left_re,
            left_eigenvectors_imag: left_im,
            n: a.nrows(),
        })
    }

    /// Returns the eigenvalues.
    pub fn eigenvalues(&self) -> &[Eigenvalue<T>] {
        &self.eigenvalues
    }

    /// Returns the real parts of eigenvalues.
    pub fn eigenvalues_real(&self) -> Vec<T> {
        self.eigenvalues.iter().map(|e| e.real).collect()
    }

    /// Returns true if all eigenvalues are real.
    pub fn all_eigenvalues_real(&self) -> bool {
        self.eigenvalues.iter().all(|e| e.is_real())
    }

    /// Returns the real parts of the right eigenvectors (if computed).
    ///
    /// Column `k` belongs to `eigenvalues()[k]` (see the
    /// [type-level documentation](GeneralEvd#eigenvector-layout)).
    pub fn eigenvectors_real(&self) -> Option<MatRef<'_, T>> {
        self.eigenvectors_real.as_ref().map(|m| m.as_ref())
    }

    /// Returns the imaginary parts of the right eigenvectors (if computed).
    ///
    /// Column `k` belongs to `eigenvalues()[k]`; it is zero for a real
    /// eigenvalue.
    pub fn eigenvectors_imag(&self) -> Option<MatRef<'_, T>> {
        self.eigenvectors_imag.as_ref().map(|m| m.as_ref())
    }

    /// Returns the real parts of left eigenvectors (if computed).
    ///
    /// Left eigenvectors satisfy: u^H * A = λ * u^H; column `k` belongs to
    /// `eigenvalues()[k]`.
    pub fn left_eigenvectors_real(&self) -> Option<MatRef<'_, T>> {
        self.left_eigenvectors_real.as_ref().map(|m| m.as_ref())
    }

    /// Returns the imaginary parts of left eigenvectors (if computed).
    pub fn left_eigenvectors_imag(&self) -> Option<MatRef<'_, T>> {
        self.left_eigenvectors_imag.as_ref().map(|m| m.as_ref())
    }

    /// Returns the number of complex eigenvalue pairs.
    pub fn num_complex_pairs(&self) -> usize {
        self.eigenvalues
            .iter()
            .filter(|e| !e.is_real() && e.imag > T::zero())
            .count()
    }

    /// Verifies the eigenvalue decomposition: A * v ≈ λ * v for every
    /// eigenpair, real or complex (with `v = v_real + i·v_imag`).
    ///
    /// Returns the maximum residual norm `‖A·v_k − λ_k·v_k‖₂` (the
    /// eigenvectors have unit norm), or zero when no eigenvectors were
    /// computed.
    pub fn verify(&self, a: MatRef<'_, T>) -> T {
        // Every constructor sets the real and imaginary eigenvector parts
        // together, so either both are present or neither is.
        let (Some(vr), Some(vi)) = (
            self.eigenvectors_real.as_ref(),
            self.eigenvectors_imag.as_ref(),
        ) else {
            return T::zero();
        };
        let mut max_residual = T::zero();

        for (col, eigenvalue) in self.eigenvalues.iter().enumerate() {
            if col >= self.n {
                break;
            }

            if eigenvalue.is_real() {
                // For real eigenvalue: ||A*v - λ*v||
                let lambda = eigenvalue.real;
                let mut residual_sq = T::zero();

                for i in 0..self.n {
                    let mut av_i = T::zero();
                    for j in 0..self.n {
                        av_i = av_i + a[(i, j)] * vr[(j, col)];
                    }
                    let diff = av_i - lambda * vr[(i, col)];
                    residual_sq = residual_sq + diff * diff;
                }

                let residual = Real::sqrt(residual_sq);
                if residual > max_residual {
                    max_residual = residual;
                }
            } else {
                // For complex eigenvalue: need to check with complex arithmetic
                // A * (vr + i*vi) = (λr + i*λi) * (vr + i*vi)
                // Real part: A*vr = λr*vr - λi*vi
                // Imag part: A*vi = λr*vi + λi*vr
                let lambda_r = eigenvalue.real;
                let lambda_i = eigenvalue.imag;
                let mut residual_sq = T::zero();

                for i in 0..self.n {
                    // A * vr
                    let mut avr_i = T::zero();
                    let mut avi_i = T::zero();
                    for j in 0..self.n {
                        avr_i = avr_i + a[(i, j)] * vr[(j, col)];
                        avi_i = avi_i + a[(i, j)] * vi[(j, col)];
                    }

                    // Expected: λr*vr - λi*vi, λr*vi + λi*vr
                    let expected_r = lambda_r * vr[(i, col)] - lambda_i * vi[(i, col)];
                    let expected_i = lambda_r * vi[(i, col)] + lambda_i * vr[(i, col)];

                    let diff_r = avr_i - expected_r;
                    let diff_i = avi_i - expected_i;
                    residual_sq = residual_sq + diff_r * diff_r + diff_i * diff_i;
                }

                let residual = Real::sqrt(residual_sq);
                if residual > max_residual {
                    max_residual = residual;
                }
            }
        }

        max_residual
    }
}

/// Back-transforms the raw eigenvectors `x` of the standardized Schur factor
/// `t` (LAPACK column layout, see `trevc_right`) to eigenvectors `Q·x` of the
/// original matrix, normalizes them (unit Euclidean norm; a complex
/// eigenvector is rotated so that its largest component is real and
/// positive), and returns them in the per-eigenvalue split layout of
/// [`GeneralEvd`]: `(real parts, imaginary parts)`.
fn back_transform_split<T: Field + Real + bytemuck::Zeroable>(
    q: &Mat<T>,
    t: &Mat<T>,
    x: &Mat<T>,
) -> (Mat<T>, Mat<T>) {
    let n = t.nrows();
    // v = Q·x, skipping the structurally zero entries of x.
    let mut v = Mat::zeros(n, n);
    for col in 0..n {
        for j in 0..n {
            let xj = x[(j, col)];
            if xj == T::zero() {
                continue;
            }
            for i in 0..n {
                v[(i, col)] = v[(i, col)] + q[(i, j)] * xj;
            }
        }
    }

    let mut re = Mat::zeros(n, n);
    let mut im = Mat::zeros(n, n);
    let mut part_re = vec![T::zero(); n];
    let mut part_im = vec![T::zero(); n];
    let mut k = 0;
    while k < n {
        if starts_block(t, k) {
            for i in 0..n {
                part_re[i] = v[(i, k)];
                part_im[i] = v[(i, k + 1)];
            }
            normalize_complex(&mut part_re, &mut part_im);
            for i in 0..n {
                re[(i, k)] = part_re[i];
                im[(i, k)] = part_im[i];
                re[(i, k + 1)] = part_re[i];
                im[(i, k + 1)] = -part_im[i];
            }
            k += 2;
        } else {
            for i in 0..n {
                part_re[i] = v[(i, k)];
            }
            normalize_real(&mut part_re);
            for i in 0..n {
                re[(i, k)] = part_re[i];
            }
            k += 1;
        }
    }
    (re, im)
}

/// Scales `v` to unit Euclidean norm (no-op for a zero or non-finite vector).
fn normalize_real<T: Field + Real>(v: &mut [T]) {
    let big = v
        .iter()
        .fold(T::zero(), |acc, &x| Float::max(acc, Scalar::abs(x)));
    if !(big > T::zero()) || !Float::is_finite(big) {
        return;
    }
    let sum = v.iter().fold(T::zero(), |acc, &x| {
        let s = x / big;
        acc + s * s
    });
    let inv = T::one() / (big * Real::sqrt(sum));
    for x in v.iter_mut() {
        *x = *x * inv;
    }
}

/// Scales the complex vector `re + i·im` to unit Euclidean norm and rotates
/// it so that its component of largest modulus is real and positive (no-op
/// for a zero or non-finite vector).
fn normalize_complex<T: Field + Real>(re: &mut [T], im: &mut [T]) {
    let big = re
        .iter()
        .chain(im.iter())
        .fold(T::zero(), |acc, &x| Float::max(acc, Scalar::abs(x)));
    if !(big > T::zero()) || !Float::is_finite(big) {
        return;
    }
    let mut sum = T::zero();
    for (&r, &i) in re.iter().zip(im.iter()) {
        let (sr, si) = (r / big, i / big);
        sum = sum + sr * sr + si * si;
    }
    let inv = T::one() / (big * Real::sqrt(sum));
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = *r * inv;
        *i = *i * inv;
    }
    // Rotate the largest component onto the positive real axis.
    let mut kmax = 0;
    let mut best = T::zero();
    for (k, (&r, &i)) in re.iter().zip(im.iter()).enumerate() {
        let mag = r * r + i * i;
        if mag > best {
            best = mag;
            kmax = k;
        }
    }
    let modulus = Real::hypot(re[kmax], im[kmax]);
    if !(modulus > T::zero()) {
        return;
    }
    let (c, s) = (re[kmax] / modulus, im[kmax] / modulus);
    // z ← z·(c − i·s)
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        let (zr, zi) = (*r, *i);
        *r = zr * c + zi * s;
        *i = zi * c - zr * s;
    }
    re[kmax] = modulus;
    im[kmax] = T::zero();
}

#[cfg(test)]
impl<T: Field + Real + bytemuck::Zeroable> GeneralEvd<T> {
    /// Test-only hook mirroring [`GeneralEvd::eigenvalues_only`] but with a custom
    /// QR iteration budget for the underlying Schur decomposition.
    ///
    /// Its sole purpose is to exercise the `Schur -> GeneralEvd` non-convergence
    /// propagation path: the `?` below routes a [`SchurError::NotConverged`]
    /// through `From<SchurError> for GeneralEvdError`, which must surface as
    /// [`GeneralEvdError::NotConverged`] rather than being swallowed or mislabelled.
    pub(crate) fn eigenvalues_only_with_budget(
        a: MatRef<'_, T>,
        max_total_iterations: usize,
    ) -> Result<Self, GeneralEvdError> {
        let schur = Schur::compute_with_iteration_budget(a, max_total_iterations)?;
        let n = a.nrows();
        let eigenvalues = schur.eigenvalues().to_vec();

        Ok(Self {
            eigenvalues,
            eigenvectors_real: None,
            eigenvectors_imag: None,
            left_eigenvectors_real: None,
            left_eigenvectors_imag: None,
            n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_general_evd_eigenvalues_only() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Sum of eigenvalues = trace
        let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
        assert!(approx_eq(sum, 5.0, 1e-10)); // trace = 1 + 4 = 5

        // Product of eigenvalues = determinant
        let mut prod: f64 = 1.0;
        for e in eigenvalues {
            prod *= e.real;
        }
        // det = 1*4 - 2*3 = -2
        assert!(approx_eq(prod, -2.0, 1e-10));
    }

    #[test]
    fn test_general_evd_with_eigenvectors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        assert!(evd.eigenvectors_real().is_some());

        // Eigenvalues should still be correct
        let eigenvalues = evd.eigenvalues();
        let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
        assert!(approx_eq(sum, 5.0, 1e-10)); // trace = 1 + 4 = 5

        // Product of eigenvalues = determinant
        let mut prod: f64 = 1.0;
        for e in eigenvalues {
            prod *= e.real;
        }
        assert!(approx_eq(prod, -2.0, 1e-10)); // det = 1*4 - 2*3 = -2
    }

    #[test]
    fn test_general_evd_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 5.0, 1e-10));

        assert!(evd.all_eigenvalues_real());
    }

    #[test]
    fn test_general_evd_complex_eigenvalues() {
        // Rotation matrix has complex eigenvalues
        let theta = core::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();
        let a = Mat::from_rows(&[&[c, -s], &[s, c]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        assert_eq!(eigenvalues.len(), 2);
        assert!(!evd.all_eigenvalues_real());
        assert_eq!(evd.num_complex_pairs(), 1);

        // Complex conjugate pair
        assert!(approx_eq(eigenvalues[0].real, eigenvalues[1].real, 1e-10));
        assert!(approx_eq(eigenvalues[0].imag, -eigenvalues[1].imag, 1e-10));
    }

    #[test]
    fn test_general_evd_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Upper triangular: eigenvalues are diagonal elements
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 4.0, 1e-10));
        assert!(approx_eq(eigs[2], 6.0, 1e-10));
    }

    #[test]
    fn test_general_evd_single() {
        let a = Mat::from_rows(&[&[7.0f64]]);
        let evd = GeneralEvd::compute(a.as_ref()).unwrap();

        assert_eq!(evd.eigenvalues().len(), 1);
        assert!(approx_eq(evd.eigenvalues()[0].real, 7.0, 1e-10));
    }

    #[test]
    fn test_general_evd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        let sum: f32 = eigenvalues.iter().map(|e| e.real).sum();
        assert!((sum - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_general_evd_full_with_left_eigenvectors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();

        // Should have both right and left eigenvectors
        assert!(evd.eigenvectors_real().is_some());
        assert!(evd.left_eigenvectors_real().is_some());

        let vr = evd.eigenvectors_real().unwrap();
        let vl = evd.left_eigenvectors_real().unwrap();

        // Both should have the same dimensions
        assert_eq!(vr.nrows(), 2);
        assert_eq!(vr.ncols(), 2);
        assert_eq!(vl.nrows(), 2);
        assert_eq!(vl.ncols(), 2);

        // Eigenvalues should be correct
        let eigenvalues = evd.eigenvalues();
        let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
        assert!(approx_eq(sum, 5.0, 1e-10)); // trace = 1 + 4 = 5
    }

    #[test]
    fn test_general_evd_full_diagonal() {
        // For a diagonal matrix, left and right eigenvectors should be identity columns
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 5.0]]);

        let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();

        assert!(evd.eigenvectors_real().is_some());
        assert!(evd.left_eigenvectors_real().is_some());

        // Eigenvalues should be 2 and 5
        let eigenvalues = evd.eigenvalues();
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 5.0, 1e-10));
    }

    #[test]
    fn test_general_evd_full_verify_left_eigenvector_property() {
        // Left eigenvectors satisfy u^T * A = λ * u^T (real eigenvalues here),
        // and column k of the left eigenvectors belongs to eigenvalues()[k] —
        // the same index as the right eigenvector, because both are computed
        // from one Schur decomposition of A.
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();
        let vl = evd.left_eigenvectors_real().unwrap();
        let eigenvalues = evd.eigenvalues();

        assert!(
            vl.nrows() == 2 && vl.ncols() == 2,
            "Left eigenvector matrix should be 2x2"
        );

        for (col, lambda) in eigenvalues.iter().enumerate() {
            assert_eq!(lambda.imag, 0.0);
            let u0 = vl[(0, col)];
            let u1 = vl[(1, col)];
            let norm = (u0 * u0 + u1 * u1).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "left eigenvector {col} norm {norm}"
            );

            // (A^T * u)_j = (u^T * A)_j
            let at_u0 = a[(0, 0)] * u0 + a[(1, 0)] * u1;
            let at_u1 = a[(0, 1)] * u0 + a[(1, 1)] * u1;
            let residual =
                ((at_u0 - lambda.real * u0).powi(2) + (at_u1 - lambda.real * u1).powi(2)).sqrt();
            assert!(
                residual < 1e-13,
                "Left eigenvector column {col} (λ = {}) has residual {residual}",
                lambda.real
            );
        }
    }

    /// A 5×5 non-symmetric matrix with a tightly clustered spectrum (near 5). It
    /// is solvable, but a single QR sweep cannot deflate it, so a starved budget
    /// exposes the non-convergence propagation path.
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
    fn test_general_evd_reports_non_convergence() {
        // Regression for the dead `NotConverged` variant: when the internal Schur
        // decomposition fails to converge, GeneralEvd must report NotConverged
        // rather than returning fabricated eigenvalues from a partial reduction.
        let a = slow_converging_5x5();

        let result = GeneralEvd::eigenvalues_only_with_budget(a.as_ref(), 1);
        assert_eq!(
            result.err(),
            Some(GeneralEvdError::NotConverged),
            "SchurError::NotConverged must propagate to GeneralEvdError::NotConverged"
        );
    }

    #[test]
    fn test_general_evd_converges_with_full_budget() {
        // Control: the same matrix converges under the real default budget, so the
        // error above is purely the artificial cap — not a broken matrix.
        let a = slow_converging_5x5();

        let evd = GeneralEvd::eigenvalues_only(a.as_ref()).expect("full budget must converge");
        let trace: f64 = (0..5).map(|i| a[(i, i)]).sum();
        let eig_sum: f64 = evd.eigenvalues().iter().map(|e| e.real).sum();
        assert!(
            approx_eq(eig_sum, trace, 1e-8),
            "eigenvalue real-part sum {eig_sum} != trace {trace}"
        );
    }

    /// `verify` reads both eigenvector parts through one `let … else` (no
    /// `expect`): without eigenvectors it reports a zero residual, with them
    /// it evaluates the residual. A multiple of the identity is used so that
    /// every unit eigenvector is exact for every eigenvalue; this pins the
    /// branch structure of `verify`, not the accuracy of the eigensolver.
    #[test]
    fn test_general_evd_verify_with_and_without_eigenvectors() {
        let a = Mat::from_rows(&[&[2.5f64, 0.0], &[0.0, 2.5]]);

        let values_only = GeneralEvd::eigenvalues_only(a.as_ref()).expect("evd");
        assert!(values_only.eigenvectors_real().is_none());
        assert!(values_only.eigenvectors_imag().is_none());
        assert_eq!(values_only.verify(a.as_ref()), 0.0);

        let full = GeneralEvd::compute(a.as_ref()).expect("evd");
        assert!(full.eigenvectors_real().is_some());
        assert!(full.eigenvectors_imag().is_some());
        let residual = full.verify(a.as_ref());
        assert!(residual < 1e-12, "residual too large: {residual}");
    }
}
