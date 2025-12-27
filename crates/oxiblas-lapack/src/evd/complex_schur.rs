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

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::complex_hessenberg::ComplexHessenberg;

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

        // Handle 1×1 case
        if n == 1 {
            let mut t: Mat<T> = Mat::zeros(1, 1);
            t[(0, 0)] = a[(0, 0)];
            let mut q: Mat<T> = Mat::zeros(1, 1);
            q[(0, 0)] = T::one();
            let eigenvalues = vec![a[(0, 0)]];
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
        let hess = ComplexHessenberg::compute(a).map_err(|_| ComplexSchurError::NotSquare)?;
        let mut t: Mat<T> = Mat::zeros(n, n);
        let h = hess.h();
        for i in 0..n {
            for j in 0..n {
                t[(i, j)] = h[(i, j)];
            }
        }

        let mut q: Mat<T> = Mat::zeros(n, n);
        let q_hess = hess.q();
        for i in 0..n {
            for j in 0..n {
                q[(i, j)] = q_hess[(i, j)];
            }
        }

        // Step 2: Apply QR iteration with shifts to reduce to upper triangular
        let eps = <T::Real as Scalar>::epsilon();
        let tol = eps * T::Real::from_f64(100.0).unwrap_or(T::Real::one());

        let mut p = n;
        let mut iter_count = 0;
        let mut stagnation_count = 0;

        while p > 1 && iter_count < Self::MAX_ITERATIONS * n {
            iter_count += 1;

            // Find the active block (find largest q such that t[q, q-1] is negligible)
            let mut q_idx = p - 1;
            while q_idx > 0 {
                let sub = t[(q_idx, q_idx - 1)].abs();
                let diag_sum = t[(q_idx - 1, q_idx - 1)].abs() + t[(q_idx, q_idx)].abs();
                if sub <= tol * (diag_sum + T::Real::one()) {
                    t[(q_idx, q_idx - 1)] = T::zero();
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
                let shift = compute_wilkinson_shift(&t, p);

                // Check for stagnation
                stagnation_count += 1;
                let actual_shift = if stagnation_count > 10 {
                    // Exceptional shift to break stagnation
                    stagnation_count = 0;
                    let mag = t[(p - 1, p - 2)].abs() + t[(p - 1, p - 1)].abs();
                    T::from_real(mag)
                } else {
                    shift
                };

                // Apply one QR step with shift
                complex_qr_step(&mut t, &mut q, q_idx, p, actual_shift);
            }
        }

        if iter_count >= Self::MAX_ITERATIONS * n && p > 1 {
            return Err(ComplexSchurError::NotConverged);
        }

        // Clean up small subdiagonal elements
        for i in 1..n {
            let sub = t[(i, i - 1)].abs();
            let diag_sum = t[(i - 1, i - 1)].abs() + t[(i, i)].abs();
            if sub
                <= tol
                    * (diag_sum + T::Real::one())
                    * T::Real::from_f64(10.0).unwrap_or(T::Real::one())
            {
                t[(i, i - 1)] = T::zero();
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(t[(i, i)]);
        }

        Ok(Self {
            q,
            t,
            eigenvalues,
            n,
        })
    }

    /// Computes Schur decomposition for a 2×2 matrix.
    fn compute_2x2(a: MatRef<'_, T>) -> Result<Self, ComplexSchurError> {
        let a00 = a[(0, 0)];
        let a01 = a[(0, 1)];
        let a10 = a[(1, 0)];
        let a11 = a[(1, 1)];

        // Compute eigenvalues of 2×2 matrix
        let trace = a00 + a11;
        let det = a00 * a11 - a01 * a10;

        // λ = (trace ± sqrt(trace² - 4*det)) / 2
        let two = T::one() + T::one();
        let four = two + two;
        let disc = trace * trace - four * det;
        let disc_sqrt = complex_sqrt(disc);

        let lambda1 = (trace + disc_sqrt) / two;
        let _lambda2 = (trace - disc_sqrt) / two;

        // For Schur decomposition, we need to find an eigenvector for lambda1
        // Then extend to a unitary basis
        let eps = T::Real::epsilon();

        // If a10 is negligible, matrix is already upper triangular
        if a10.abs() < eps {
            let mut q: Mat<T> = Mat::zeros(2, 2);
            q[(0, 0)] = T::one();
            q[(1, 1)] = T::one();

            let mut t: Mat<T> = Mat::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    t[(i, j)] = a[(i, j)];
                }
            }

            let eigenvalues = vec![a00, a11];
            return Ok(Self {
                q,
                t,
                eigenvalues,
                n: 2,
            });
        }

        // Find eigenvector for lambda1: (A - lambda1*I) * v = 0
        // From first row: (a00 - lambda1) * v0 + a01 * v1 = 0
        // From second row: a10 * v0 + (a11 - lambda1) * v1 = 0
        let d0 = a00 - lambda1;
        let d1 = a11 - lambda1;

        // Choose the row with larger magnitude for numerical stability
        let row0_mag = d0.abs_sq() + a01.abs_sq();
        let row1_mag = a10.abs_sq() + d1.abs_sq();

        let (v0, v1) = if row0_mag >= row1_mag && a01.abs() > eps {
            // v0 * d0 + v1 * a01 = 0  =>  v1 = -v0 * d0 / a01
            (T::one(), T::zero() - d0 / a01)
        } else if a10.abs() > eps {
            // v0 * a10 + v1 * d1 = 0  =>  v0 = -v1 * d1 / a10
            (T::zero() - d1 / a10, T::one())
        } else {
            // Degenerate case: use standard basis
            (T::one(), T::zero())
        };

        // Normalize first eigenvector
        let norm_sq = v0.abs_sq() + v1.abs_sq();
        let norm = <T::Real as Real>::sqrt(norm_sq);
        let u0 = if norm > eps {
            (v0 / T::from_real(norm), v1 / T::from_real(norm))
        } else {
            (T::one(), T::zero())
        };

        // Second column of Q is orthogonal to first: u1 = [-conj(u0.1), conj(u0.0)]
        let u1 = (T::zero() - u0.1.conj(), u0.0.conj());

        // Build Q (columns are eigenvectors)
        let mut q: Mat<T> = Mat::zeros(2, 2);
        q[(0, 0)] = u0.0;
        q[(1, 0)] = u0.1;
        q[(0, 1)] = u1.0;
        q[(1, 1)] = u1.1;

        // Build T = Q^H * A * Q
        let mut t: Mat<T> = Mat::zeros(2, 2);

        // Compute Q^H * A
        let mut qha: Mat<T> = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = T::zero();
                for k in 0..2 {
                    sum = sum + q[(k, i)].conj() * a[(k, j)];
                }
                qha[(i, j)] = sum;
            }
        }

        // Compute (Q^H * A) * Q
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = T::zero();
                for k in 0..2 {
                    sum = sum + qha[(i, k)] * q[(k, j)];
                }
                t[(i, j)] = sum;
            }
        }

        // Clean up small subdiagonal
        let hundred: T::Real =
            <T::Real as FromPrimitive>::from_f64(100.0).unwrap_or(<T::Real as One>::one());
        if t[(1, 0)].abs() < eps * hundred {
            t[(1, 0)] = T::zero();
        }

        let eigenvalues = vec![t[(0, 0)], t[(1, 1)]];

        Ok(Self {
            q,
            t,
            eigenvalues,
            n: 2,
        })
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

    /// Verifies the Schur decomposition: ||A - Q T Q^H|| / ||A||.
    pub fn residual(&self, a: MatRef<'_, T>) -> T::Real {
        let n = self.n;
        let reconstructed = self.reconstruct();

        // Compute ||A - reconstructed||_F
        let mut diff_norm_sq = T::Real::zero();
        let mut a_norm_sq = T::Real::zero();

        for i in 0..n {
            for j in 0..n {
                let diff = a[(i, j)] - reconstructed[(i, j)];
                diff_norm_sq = diff_norm_sq + diff.abs_sq();
                a_norm_sq = a_norm_sq + a[(i, j)].abs_sq();
            }
        }

        if a_norm_sq < T::Real::epsilon() {
            return diff_norm_sq;
        }

        <T::Real as Real>::sqrt(diff_norm_sq / a_norm_sq)
    }
}

/// Computes the square root of a complex number.
fn complex_sqrt<T: Field + ComplexScalar>(z: T) -> T
where
    T::Real: Real,
{
    let (re, im) = (z.real(), z.imag());
    let mag = <T::Real as Real>::sqrt(re * re + im * im);

    if mag < T::Real::epsilon() {
        return T::zero();
    }

    let two = T::Real::one() + T::Real::one();
    let r = <T::Real as Real>::sqrt((mag + re) / two);
    let i = if im >= T::Real::zero() {
        <T::Real as Real>::sqrt((mag - re) / two)
    } else {
        -<T::Real as Real>::sqrt((mag - re) / two)
    };

    T::from_real_imag(r, i)
}

/// Computes the Wilkinson shift from the bottom 2x2 block.
fn compute_wilkinson_shift<T: Field + ComplexScalar + bytemuck::Zeroable>(t: &Mat<T>, p: usize) -> T
where
    T::Real: Real,
{
    if p < 2 {
        return t[(p - 1, p - 1)];
    }

    let h11 = t[(p - 2, p - 2)];
    let h12 = t[(p - 2, p - 1)];
    let h21 = t[(p - 1, p - 2)];
    let h22 = t[(p - 1, p - 1)];

    // Compute eigenvalues of 2x2 block
    let trace = h11 + h22;
    let det = h11 * h22 - h12 * h21;

    let two = T::one() + T::one();
    let four = two + two;
    let disc = trace * trace - four * det;
    let disc_sqrt = complex_sqrt(disc);

    let half = T::one() / two;
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

/// Applies one implicit QR step with shift (Francis single-shift algorithm).
///
/// For a Hessenberg matrix H[start:end, start:end], this performs one iteration
/// of the implicit shifted QR algorithm with shift σ.
fn complex_qr_step<T: Field + ComplexScalar + bytemuck::Zeroable>(
    h: &mut Mat<T>,
    q: &mut Mat<T>,
    start: usize,
    end: usize,
    shift: T,
) where
    T::Real: Real,
{
    let n = h.nrows();

    // Step 1: Create initial rotation to introduce the bulge
    // Compute first column of (H - σI): x = [h[start,start] - σ, h[start+1,start], 0, ...]
    let x0 = h[(start, start)] - shift;
    let x1 = h[(start + 1, start)];

    let (c, s) = givens_rotation_complex(x0, x1);

    // Apply G_0 from left and right
    apply_givens_left(h, c, s, start, start, n);
    apply_givens_right(h, c, s, start, 0, end.min(start + 3));
    apply_givens_right_to_q(q, c, s, start, n);

    // Step 2: Chase the bulge down the subdiagonal
    for k in start..(end - 2) {
        // The bulge is at position (k+2, k)
        // Create rotation to zero h[k+2, k] using rows k+1 and k+2
        let a = h[(k + 1, k)];
        let b = h[(k + 2, k)];

        let (c, s) = givens_rotation_complex(a, b);

        // Apply rotation from left to rows k+1 and k+2
        apply_givens_left(h, c, s, k + 1, k, n);

        // Apply rotation from right to columns k+1 and k+2
        let row_end = end.min(k + 5);
        apply_givens_right(h, c, s, k + 1, 0, row_end);

        // Accumulate Q
        apply_givens_right_to_q(q, c, s, k + 1, n);

        // Explicitly zero the bulge (clean up numerical noise)
        h[(k + 2, k)] = T::zero();
    }
}

/// Apply Givens rotation from the left: H := G^H * H
/// G^H = [[c, s*], [-s, c]] (c is real)
#[inline]
fn apply_givens_left<T: Field + ComplexScalar>(
    h: &mut Mat<T>,
    c: T,
    s: T,
    row: usize,
    col_start: usize,
    col_end: usize,
) where
    T::Real: Real,
{
    for j in col_start..col_end {
        let t1 = h[(row, j)];
        let t2 = h[(row + 1, j)];
        h[(row, j)] = c * t1 + s.conj() * t2;
        h[(row + 1, j)] = (T::zero() - s) * t1 + c * t2;
    }
}

/// Apply Givens rotation from the right: H := H * G
/// G = [[c, -s*], [s, c]] (c is real)
#[inline]
fn apply_givens_right<T: Field + ComplexScalar>(
    h: &mut Mat<T>,
    c: T,
    s: T,
    col: usize,
    row_start: usize,
    row_end: usize,
) where
    T::Real: Real,
{
    for i in row_start..row_end {
        let t1 = h[(i, col)];
        let t2 = h[(i, col + 1)];
        h[(i, col)] = t1 * c + t2 * s;
        h[(i, col + 1)] = (T::zero() - t1) * s.conj() + t2 * c;
    }
}

/// Apply Givens rotation to Q: Q := Q * G
#[inline]
fn apply_givens_right_to_q<T: Field + ComplexScalar>(
    q: &mut Mat<T>,
    c: T,
    s: T,
    col: usize,
    n: usize,
) where
    T::Real: Real,
{
    for i in 0..n {
        let t1 = q[(i, col)];
        let t2 = q[(i, col + 1)];
        q[(i, col)] = t1 * c + t2 * s;
        q[(i, col + 1)] = (T::zero() - t1) * s.conj() + t2 * c;
    }
}

/// Computes Givens rotation for complex numbers.
/// Returns (c, s) such that G^H * [a; b] = [r; 0] where G^H = [[c, s*], [-s, c]].
/// Here c is real (stored as complex with zero imaginary part).
///
/// The rotation satisfies: -s * a + c * b = 0, which gives s = c * b / a.
/// With c = |a| / r, we get s = |a| * b / (r * a) = conj(a) * b / (|a| * r).
fn givens_rotation_complex<T: Field + ComplexScalar>(a: T, b: T) -> (T, T)
where
    T::Real: Real,
{
    let b_norm = b.abs();

    if b_norm < T::Real::epsilon() {
        return (T::one(), T::zero());
    }

    let a_norm = a.abs();
    let r = <T::Real as Real>::sqrt(a_norm * a_norm + b_norm * b_norm);

    if r < T::Real::epsilon() {
        return (T::one(), T::zero());
    }

    // c is real: c = |a| / r
    let c = T::from_real(a_norm / r);

    // s = conj(a) * b / (|a| * r) = conj(a/|a|) * b / r
    let s = if a_norm > T::Real::epsilon() {
        a.conj() * b / T::from_real(a_norm * r)
    } else {
        b / T::from_real(r)
    };

    (c, s)
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
