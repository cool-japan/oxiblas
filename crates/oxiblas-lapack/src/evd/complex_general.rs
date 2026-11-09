//! Complex general eigenvalue decomposition.
//!
//! Computes eigenvalues and eigenvectors of general complex matrices.
//! For complex matrices, eigenvalues are complex numbers stored directly.
//!
//! The decomposition follows LAPACK `ZGEEV`: the matrix is scaled by an exact
//! power of two, reduced to complex Schur form `A = Q·T·Qᴴ` by
//! [`ComplexSchur`], the eigenvectors of the triangular `T` are computed by
//! back-substitution (`ZTREVC`, with perturbation of near-singular pivots and
//! overflow-safe rescaling), and transformed back with `Q`.

use num_traits::{Float, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::complex_schur::{
    ComplexSchur, ComplexSchurError, complex_left_eigenvectors, complex_right_eigenvectors,
    scale_complex,
};
use super::schur::scaling_exponent;

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

impl From<ComplexSchurError> for ComplexGeneralEvdError {
    fn from(e: ComplexSchurError) -> Self {
        match e {
            ComplexSchurError::EmptyMatrix => Self::EmptyMatrix,
            ComplexSchurError::NotSquare => Self::NotSquare,
            ComplexSchurError::NotConverged => Self::NotConverged,
        }
    }
}

/// Complex general eigenvalue decomposition.
///
/// For a general complex matrix A, computes eigenvalues and optionally eigenvectors.
/// Eigenvalues are directly complex numbers (not paired as in real EVD).
///
/// Right eigenvectors satisfy: A * v = λ * v
/// Left eigenvectors satisfy: u^H * A = λ * u^H
///
/// Column `k` of [`eigenvectors`](Self::eigenvectors) and of
/// [`left_eigenvectors`](Self::left_eigenvectors) belongs to
/// `eigenvalues()[k]`. Every eigenvector has unit Euclidean norm and its
/// component of largest modulus is real and positive (as LAPACK `ZGEEV`).
/// Eigenvalues are listed in the order of the diagonal of the complex Schur
/// form computed by [`ComplexSchur`].
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
        let (schur, exponent) = Self::scaled_schur(a)?;
        Ok(Self {
            eigenvalues: Self::unscaled_eigenvalues(&schur, exponent),
            eigenvectors: None,
            left_eigenvectors: None,
            n: a.nrows(),
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
    /// assert!(evd.verify(a.as_ref()) < 1e-12);
    /// # let _ = v;
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ComplexGeneralEvdError> {
        Self::compute_with_vectors(a, false)
    }

    /// Computes eigenvalues, right eigenvectors, and left eigenvectors.
    ///
    /// Right eigenvectors satisfy: A * v = λ * v
    /// Left eigenvectors satisfy: u^H * A = λ * u^H
    ///
    /// Both are paired with the eigenvalue of the same index.
    pub fn compute_full(a: MatRef<'_, T>) -> Result<Self, ComplexGeneralEvdError> {
        Self::compute_with_vectors(a, true)
    }

    /// Validates `a`, scales it by an exact power of two so that
    /// `max|a_ij| ∈ [1, 2)` (no-op for a zero or non-finite maximum) and
    /// computes the complex Schur decomposition of the scaled matrix. Returns
    /// the decomposition and the scaling exponent `k` (the matrix decomposed
    /// is `2^k·A`).
    fn scaled_schur(a: MatRef<'_, T>) -> Result<(ComplexSchur<T>, i32), ComplexGeneralEvdError> {
        let n = a.nrows();
        if n == 0 || a.ncols() == 0 {
            return Err(ComplexGeneralEvdError::EmptyMatrix);
        }
        if a.ncols() != n {
            return Err(ComplexGeneralEvdError::NotSquare);
        }
        let mut anrm = T::Real::zero();
        for j in 0..n {
            for i in 0..n {
                anrm = Float::max(anrm, Scalar::abs(a[(i, j)]));
            }
        }
        let exponent = scaling_exponent(anrm);
        if exponent == 0 {
            return Ok((ComplexSchur::compute(a)?, 0));
        }
        let mut scaled: Mat<T> = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                scaled[(i, j)] = scale_complex(a[(i, j)], exponent);
            }
        }
        Ok((ComplexSchur::compute(scaled.as_ref())?, exponent))
    }

    /// Eigenvalues of `A` from the Schur decomposition of `2^exponent·A`.
    fn unscaled_eigenvalues(schur: &ComplexSchur<T>, exponent: i32) -> Vec<T> {
        schur
            .eigenvalues()
            .iter()
            .map(|&z| scale_complex(z, -exponent))
            .collect()
    }

    /// Shared implementation of [`compute`](Self::compute) and
    /// [`compute_full`](Self::compute_full).
    fn compute_with_vectors(
        a: MatRef<'_, T>,
        want_left: bool,
    ) -> Result<Self, ComplexGeneralEvdError> {
        let (schur, exponent) = Self::scaled_schur(a)?;
        let t = schur.t_mat();
        let q = schur.q_mat();

        let eigenvectors = back_transform(q, &complex_right_eigenvectors(t));
        let left_eigenvectors = if want_left {
            Some(back_transform(q, &complex_left_eigenvectors(t)))
        } else {
            None
        };

        Ok(Self {
            eigenvalues: Self::unscaled_eigenvalues(&schur, exponent),
            eigenvectors: Some(eigenvectors),
            left_eigenvectors,
            n: a.nrows(),
        })
    }

    /// Returns the eigenvalues.
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the right eigenvectors (if computed).
    ///
    /// Column `k` is the eigenvector of `eigenvalues()[k]`.
    pub fn eigenvectors(&self) -> Option<MatRef<'_, T>> {
        self.eigenvectors.as_ref().map(|m| m.as_ref())
    }

    /// Returns the left eigenvectors (if computed).
    ///
    /// Column `k` is the left eigenvector (`u^H * A = λ * u^H`) of
    /// `eigenvalues()[k]`.
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
        let Some(v) = self.eigenvectors.as_ref() else {
            return real_zero::<T::Real>();
        };
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

/// Back-transforms the eigenvectors `x` of the triangular Schur factor to
/// eigenvectors `Q·x` of the original matrix and normalizes each to unit
/// Euclidean norm with its component of largest modulus real and positive.
fn back_transform<T: Field + ComplexScalar + bytemuck::Zeroable>(q: &Mat<T>, x: &Mat<T>) -> Mat<T>
where
    T::Real: Real,
{
    let n = q.nrows();
    let mut v: Mat<T> = Mat::zeros(n, n);
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
    let mut column = vec![T::zero(); n];
    for col in 0..n {
        for (i, c) in column.iter_mut().enumerate() {
            *c = v[(i, col)];
        }
        normalize_complex_vector(&mut column);
        for (i, c) in column.iter().enumerate() {
            v[(i, col)] = *c;
        }
    }
    v
}

/// Scales `z` to unit Euclidean norm and rotates it so that its component of
/// largest modulus is real and positive (no-op for a zero or non-finite
/// vector).
fn normalize_complex_vector<T: Field + ComplexScalar>(z: &mut [T])
where
    T::Real: Real,
{
    let big = z.iter().fold(T::Real::zero(), |acc, w| {
        Float::max(
            acc,
            Float::max(Scalar::abs(w.real()), Scalar::abs(w.imag())),
        )
    });
    if !(big > T::Real::zero()) || !Float::is_finite(big) {
        return;
    }
    let mut sum = T::Real::zero();
    for w in z.iter() {
        let (r, i) = (w.real() / big, w.imag() / big);
        sum = sum + r * r + i * i;
    }
    let inv = T::from_real(T::Real::one() / (big * Real::sqrt(sum)));
    for w in z.iter_mut() {
        *w = *w * inv;
    }
    let mut kmax = 0;
    let mut best = T::Real::zero();
    for (k, w) in z.iter().enumerate() {
        let mag = w.abs_sq();
        if mag > best {
            best = mag;
            kmax = k;
        }
    }
    let modulus = Scalar::abs(z[kmax]);
    if !(modulus > T::Real::zero()) {
        return;
    }
    let phase = T::from_real_imag(z[kmax].real() / modulus, -(z[kmax].imag() / modulus));
    for w in z.iter_mut() {
        *w = *w * phase;
    }
    z[kmax] = T::from_real(modulus);
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

    /// `verify` without eigenvectors reports a zero residual (the `let … else`
    /// path that replaced an `expect`).
    #[test]
    fn test_complex_general_evd_verify_without_eigenvectors() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.5), Complex64::new(0.0, 1.0)],
            &[Complex64::new(2.0, -1.0), Complex64::new(-0.5, 0.0)],
        ]);
        let evd = ComplexGeneralEvd::eigenvalues_only(a.as_ref()).expect("evd");
        assert!(evd.eigenvectors().is_none());
        assert_eq!(evd.verify(a.as_ref()), 0.0);
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
