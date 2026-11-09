//! Complex Schur decomposition.
//!
//! Computes the Schur decomposition A = Q T Q^H where Q is unitary
//! and T is upper triangular (complex Schur form).
//!
//! Unlike the real Schur form which may have 2×2 blocks on the diagonal
//! for complex eigenvalue pairs, the complex Schur form is always
//! upper triangular with eigenvalues on the diagonal.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::ComplexSchur;
//! use oxiblas_matrix::Mat;
//! use num_complex::Complex64;
//!
//! let a = Mat::from_rows(&[
//!     &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
//!     &[Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
//! ]);
//!
//! let schur = ComplexSchur::compute(a.as_ref()).unwrap();
//! let eigenvalues = schur.eigenvalues();
//!
//! // Eigenvalues are 1+0i and 3+0i
//! ```

mod qr;
mod trevc;

use num_traits::{Float, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::complex_hessenberg::ComplexHessenberg;
use super::schur::{scale_by_pow2, scaling_exponent};

pub(crate) use trevc::{complex_left_eigenvectors, complex_right_eigenvectors};

/// Multiplies both components of `z` by `2^k` exactly (barring underflow).
#[inline]
pub(crate) fn scale_complex<T: ComplexScalar>(z: T, k: i32) -> T {
    if k == 0 {
        return z;
    }
    T::from_real_imag(scale_by_pow2(z.real(), k), scale_by_pow2(z.imag(), k))
}

/// Error type for complex Schur decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexSchurError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for ComplexSchurError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::NotConverged => write!(f, "Schur decomposition did not converge"),
        }
    }
}

impl std::error::Error for ComplexSchurError {}

/// Complex Schur decomposition of a matrix.
///
/// For a complex matrix A, computes A = Q T Q^H where:
/// - Q is unitary (Q^H Q = I)
/// - T is upper triangular (complex Schur form)
/// - Eigenvalues are on the diagonal of T
#[derive(Debug, Clone)]
pub struct ComplexSchur<T: Scalar> {
    /// The unitary matrix Q (Schur vectors).
    q: Mat<T>,
    /// The upper triangular matrix T (Schur form).
    t: Mat<T>,
    /// Eigenvalues (diagonal of T).
    eigenvalues: Vec<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexSchur<T>
where
    T::Real: Real,
{
    /// Maximum iterations for QR iteration.
    const MAX_ITERATIONS: usize = 100;

    /// Computes the complex Schur decomposition of a square matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::ComplexSchur;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    ///     &[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
    /// ]);
    ///
    /// let schur = ComplexSchur::compute(a.as_ref()).unwrap();
    /// let t = schur.t();
    ///
    /// // T is upper triangular
    /// assert!(t[(1, 0)].norm() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ComplexSchurError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexSchurError::EmptyMatrix);
        }

        if m != n {
            return Err(ComplexSchurError::NotSquare);
        }

        // Scale by an exact power of two so that max|a_ij| ∈ [1, 2): every
        // threshold below then acts relative to the matrix norm, and the
        // intermediate quantities can neither overflow nor underflow early.
        let mut anrm = T::Real::zero();
        for j in 0..n {
            for i in 0..n {
                anrm = Float::max(anrm, Scalar::abs(a[(i, j)]));
            }
        }
        let exponent = scaling_exponent(anrm);
        let mut scaled: Mat<T> = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                scaled[(i, j)] = scale_complex(a[(i, j)], exponent);
            }
        }

        // Reduce to upper Hessenberg form (a matrix of order <= 2 already is).
        let (mut t, mut q) = if n >= 3 {
            let hess = ComplexHessenberg::compute(scaled.as_ref())
                .map_err(|_| ComplexSchurError::NotSquare)?;
            let mut t: Mat<T> = Mat::zeros(n, n);
            let mut q: Mat<T> = Mat::zeros(n, n);
            let (h, qh) = (hess.h(), hess.q());
            for j in 0..n {
                for i in 0..n {
                    t[(i, j)] = h[(i, j)];
                    q[(i, j)] = qh[(i, j)];
                }
            }
            (t, q)
        } else {
            let mut q: Mat<T> = Mat::zeros(n, n);
            for i in 0..n {
                q[(i, i)] = T::one();
            }
            (scaled, q)
        };

        // Single-shift QR iteration down to upper triangular form. The budget
        // allows `MAX_ITERATIONS` sweeps per row on average.
        qr::complex_hessenberg_qr(&mut t, &mut q, Self::MAX_ITERATIONS.saturating_mul(n))?;

        if exponent != 0 {
            for j in 0..n {
                for i in 0..=j {
                    t[(i, j)] = scale_complex(t[(i, j)], -exponent);
                }
            }
        }

        let eigenvalues = (0..n).map(|i| t[(i, i)]).collect();

        Ok(Self {
            q,
            t,
            eigenvalues,
            n,
        })
    }

    /// The Schur vectors as an owned-matrix reference (crate-internal).
    pub(crate) fn q_mat(&self) -> &Mat<T> {
        &self.q
    }

    /// The triangular Schur factor as an owned-matrix reference
    /// (crate-internal).
    pub(crate) fn t_mat(&self) -> &Mat<T> {
        &self.t
    }

    /// Returns the unitary matrix Q (Schur vectors).
    pub fn q(&self) -> MatRef<'_, T> {
        self.q.as_ref()
    }

    /// Returns the upper triangular Schur matrix T.
    pub fn t(&self) -> MatRef<'_, T> {
        self.t.as_ref()
    }

    /// Returns the eigenvalues (diagonal elements of T).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = Q T Q^H.
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a: Mat<T> = Mat::zeros(n, n);

        // First compute T * Q^H
        let mut tqh: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + self.t[(i, k)] * self.q[(j, k)].conj();
                }
                tqh[(i, j)] = sum;
            }
        }

        // Then compute Q * (T * Q^H)
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + self.q[(i, k)] * tqh[(k, j)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }

    /// Verifies the Schur decomposition: ||A - Q T Q^H||_F / ||A||_F
    /// (the absolute residual ||A - Q T Q^H||_F when A is zero).
    pub fn residual(&self, a: MatRef<'_, T>) -> T::Real {
        let n = self.n;
        let reconstructed = self.reconstruct();

        // ||A - reconstructed||_F and ||A||_F, accumulated with `hypot` so that
        // neither squares of tiny entries underflow nor squares of huge ones
        // overflow: the ratio is then scale-invariant.
        let mut diff_norm = T::Real::zero();
        let mut a_norm = T::Real::zero();

        for i in 0..n {
            for j in 0..n {
                let diff = a[(i, j)] - reconstructed[(i, j)];
                diff_norm = <T::Real as Real>::hypot(diff_norm, diff.abs());
                a_norm = <T::Real as Real>::hypot(a_norm, a[(i, j)].abs());
            }
        }

        // A zero matrix has no relative scale: report the absolute residual.
        if a_norm == T::Real::zero() {
            return diff_norm;
        }

        diff_norm / a_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn approx_eq_complex(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    #[test]
    fn test_complex_schur_upper_triangular() {
        // Upper triangular matrix - already in Schur form
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let t = schur.t();
        let eigenvalues = schur.eigenvalues();

        // T should be upper triangular
        assert!(
            t[(1, 0)].norm() < 1e-10,
            "T[1,0] = {:?} should be zero",
            t[(1, 0)]
        );

        // Eigenvalues should be 1 and 3
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.re).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (eigs[0] - 1.0).abs() < 1e-10,
            "eig[0] = {}, expected 1",
            eigs[0]
        );
        assert!(
            (eigs[1] - 3.0).abs() < 1e-10,
            "eig[1] = {}, expected 3",
            eigs[1]
        );
    }

    #[test]
    fn test_complex_schur_general() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let t = schur.t();
        let q = schur.q();

        // T should be upper triangular
        assert!(
            t[(1, 0)].norm() < 1e-10,
            "T[1,0] = {:?} should be zero",
            t[(1, 0)]
        );

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
    fn test_complex_schur_reconstruction() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 1.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let residual = schur.residual(a.as_ref());

        assert!(
            residual < 1e-10,
            "Reconstruction residual = {} is too large",
            residual
        );
    }

    #[test]
    fn test_complex_schur_3x3() {
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

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let t = schur.t();

        // T should be upper triangular
        for j in 0..2 {
            for i in (j + 1)..3 {
                assert!(
                    t[(i, j)].norm() < 1e-10,
                    "T[{},{}] = {:?} should be zero",
                    i,
                    j,
                    t[(i, j)]
                );
            }
        }

        // Verify reconstruction
        let residual = schur.residual(a.as_ref());
        assert!(residual < 1e-10);
    }

    #[test]
    fn test_complex_schur_trace_determinant() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 1.0)],
            &[Complex64::new(-1.0, 1.0), Complex64::new(4.0, -1.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let eigenvalues = schur.eigenvalues();

        // Sum of eigenvalues equals trace
        let sum = eigenvalues[0] + eigenvalues[1];
        let trace = a[(0, 0)] + a[(1, 1)];
        assert!(
            approx_eq_complex(sum, trace, 1e-10),
            "sum = {:?}, trace = {:?}",
            sum,
            trace
        );

        // Product of eigenvalues equals determinant
        let prod = eigenvalues[0] * eigenvalues[1];
        let det = a[(0, 0)] * a[(1, 1)] - a[(0, 1)] * a[(1, 0)];
        assert!(
            approx_eq_complex(prod, det, 1e-10),
            "prod = {:?}, det = {:?}",
            prod,
            det
        );
    }

    #[test]
    fn test_complex_schur_hermitian() {
        // Hermitian matrix - eigenvalues should be real
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let eigenvalues = schur.eigenvalues();

        // Eigenvalues of Hermitian matrix are real
        for e in eigenvalues {
            assert!(e.im.abs() < 1e-10, "Expected real eigenvalue, got {:?}", e);
        }
    }

    #[test]
    fn test_complex_schur_single() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(5.0, 2.0)]]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let eigenvalues = schur.eigenvalues();

        assert_eq!(eigenvalues.len(), 1);
        assert!(approx_eq_complex(
            eigenvalues[0],
            Complex64::new(5.0, 2.0),
            1e-10
        ));
    }

    #[test]
    fn test_complex_schur_f32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
            &[Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let t = schur.t();

        // T should be upper triangular
        assert!(
            t[(1, 0)].norm() < 1e-5,
            "T[1,0] = {:?} should be zero",
            t[(1, 0)]
        );
    }

    #[test]
    fn test_complex_schur_rotation() {
        // Rotation matrix - eigenvalues should be e^{±iθ}
        let theta = std::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();

        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
            &[Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
        ]);

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let eigenvalues = schur.eigenvalues();

        // Eigenvalues should be e^{iθ} and e^{-iθ} (magnitude = 1)
        for e in eigenvalues {
            let mag = e.norm();
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "Expected |eigenvalue| = 1, got {}",
                mag
            );
        }
    }

    #[test]
    fn test_complex_schur_identity() {
        let eye: Mat<Complex64> = Mat::eye(3);

        let schur = ComplexSchur::compute(eye.as_ref()).expect("Should compute");
        let t = schur.t();

        // T should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (t[(i, j)] - expected).norm();
                assert!(diff < 1e-10, "T[{},{}] = {:?}", i, j, t[(i, j)]);
            }
        }
    }

    #[test]
    fn test_complex_schur_4x4() {
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

        let schur = ComplexSchur::compute(a.as_ref()).expect("Should compute");
        let t = schur.t();

        // T should be upper triangular
        for j in 0..3 {
            for i in (j + 1)..4 {
                assert!(
                    t[(i, j)].norm() < 1e-9,
                    "T[{},{}] = {:?} should be zero",
                    i,
                    j,
                    t[(i, j)]
                );
            }
        }

        // Verify reconstruction
        let residual = schur.residual(a.as_ref());
        assert!(residual < 1e-9, "Residual {} is too large", residual);
    }
}
