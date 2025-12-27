//! Complex SVD using one-sided Jacobi algorithm.
//!
//! Computes A = U·Σ·V^H where:
//! - U is m×m unitary (left singular vectors)
//! - Σ is m×n diagonal (singular values, real non-negative, descending)
//! - V is n×n unitary (right singular vectors)

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for complex SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for ComplexSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "Complex SVD algorithm did not converge"),
        }
    }
}

impl std::error::Error for ComplexSvdError {}

/// Complex Singular Value Decomposition result.
///
/// A = U·Σ·V^H where Σ contains real singular values on the diagonal.
#[derive(Debug, Clone)]
pub struct ComplexSvd<T: Scalar> {
    /// Left singular vectors (m×m unitary matrix).
    u: Mat<T>,
    /// Singular values (real, sorted in descending order).
    sigma: Vec<T::Real>,
    /// Right singular vectors (n×n unitary matrix, stored as V^H).
    vh: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexSvd<T>
where
    T::Real: Real,
{
    /// Maximum sweeps for Jacobi iteration.
    const MAX_SWEEPS: usize = 30;

    /// Computes the full SVD of a complex matrix A using one-sided Jacobi algorithm.
    ///
    /// Returns A = U·Σ·V^H where U and V are unitary and Σ has real non-negative values.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ComplexSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexSvdError::EmptyMatrix);
        }

        // Handle 1x1 case
        if m == 1 && n == 1 {
            let val = a[(0, 0)];
            let abs_val = val.abs();
            let sigma = vec![abs_val];
            let mut u: Mat<T> = Mat::zeros(1, 1);
            let mut vh: Mat<T> = Mat::zeros(1, 1);

            if abs_val > T::Real::zero() {
                // u = val / |val|, v = 1
                u[(0, 0)] = T::from_real_imag(val.real() / abs_val, val.imag() / abs_val);
            } else {
                u[(0, 0)] = T::one();
            }
            vh[(0, 0)] = T::one();
            return Ok(Self { u, sigma, vh, m, n });
        }

        // Copy A into working matrix B (we'll work on B to get V, then compute U)
        let mut b: Mat<T> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                b[(i, j)] = a[(i, j)];
            }
        }

        // Initialize V as identity
        let mut v: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = T::one();
        }

        let eps = <T::Real as Scalar>::epsilon();
        let tol = eps * T::Real::from_f64(100.0).unwrap_or(T::Real::one());

        // One-sided Jacobi: Apply Jacobi rotations to columns of B to diagonalize B^H*B
        // For complex matrices, the Gram matrix is B^H*B (conjugate transpose)
        for _sweep in 0..Self::MAX_SWEEPS {
            let mut converged = true;

            // Sweep through all column pairs (i, j) with i < j
            for i in 0..n {
                for j in (i + 1)..n {
                    // Compute B[:, i]^H * B[:, j] and norms (complex inner products)
                    // <b_i, b_j> = sum conj(b_i[k]) * b_j[k]
                    let mut dot_ij = T::zero();
                    let mut norm_i_sq = T::Real::zero();
                    let mut norm_j_sq = T::Real::zero();

                    for row in 0..m {
                        let bi = b[(row, i)];
                        let bj = b[(row, j)];
                        // Hermitian inner product: conj(bi) * bj
                        dot_ij = dot_ij + bi.conj() * bj;
                        norm_i_sq = norm_i_sq + bi.abs_sq();
                        norm_j_sq = norm_j_sq + bj.abs_sq();
                    }

                    // Check if rotation is needed
                    let off_diag = dot_ij.abs();
                    let threshold = tol
                        * <T::Real as Real>::sqrt(norm_i_sq)
                        * <T::Real as Real>::sqrt(norm_j_sq);

                    if off_diag > threshold {
                        converged = false;

                        // Compute complex Jacobi rotation to zero dot_ij
                        // For complex, we use a 2x2 unitary matrix
                        let (c, s) = complex_jacobi_rotation(norm_i_sq, norm_j_sq, dot_ij);

                        // Apply rotation to columns of B: [b_i, b_j] = [b_i, b_j] * [[c, -s*], [s, c]]
                        // b_i' = c*b_i + s*b_j
                        // b_j' = -s**b_i + c*b_j
                        for row in 0..m {
                            let bi = b[(row, i)];
                            let bj = b[(row, j)];
                            b[(row, i)] = T::from_real(c) * bi + s * bj;
                            b[(row, j)] = T::from_real(c) * bj - s.conj() * bi;
                        }

                        // Apply same rotation to columns of V
                        for row in 0..n {
                            let vi = v[(row, i)];
                            let vj = v[(row, j)];
                            v[(row, i)] = T::from_real(c) * vi + s * vj;
                            v[(row, j)] = T::from_real(c) * vj - s.conj() * vi;
                        }
                    }
                }
            }

            if converged {
                break;
            }
        }

        // Extract singular values from columns of B (now orthogonal)
        // and compute U
        let k = m.min(n);
        let mut sigma = vec![T::Real::zero(); k];
        let mut u: Mat<T> = Mat::zeros(m, m);

        // First k columns of U come from normalizing columns of B
        for j in 0..k {
            let mut norm_sq = T::Real::zero();
            for i in 0..m {
                norm_sq = norm_sq + b[(i, j)].abs_sq();
            }
            let norm = <T::Real as Real>::sqrt(norm_sq);
            sigma[j] = norm;

            if norm > eps {
                for i in 0..m {
                    u[(i, j)] = b[(i, j)] / T::from_real(norm);
                }
            } else {
                u[(j, j)] = T::one();
            }
        }

        // Complete U to a full unitary matrix (for m > n, add orthogonal columns)
        if m > k {
            // Use Gram-Schmidt to complete U
            complete_unitary_matrix(&mut u, m, k);
        }

        // Sort singular values in descending order and reorder U, V accordingly
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap());

        let mut sigma_sorted = vec![T::Real::zero(); k];
        let mut u_sorted: Mat<T> = Mat::zeros(m, m);
        let mut v_sorted: Mat<T> = Mat::zeros(n, n);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sigma_sorted[new_idx] = sigma[old_idx];
            for i in 0..m {
                u_sorted[(i, new_idx)] = u[(i, old_idx)];
            }
            for i in 0..n {
                v_sorted[(i, new_idx)] = v[(i, old_idx)];
            }
        }

        // Copy remaining columns for U (if m > k)
        for j in k..m {
            for i in 0..m {
                u_sorted[(i, j)] = u[(i, j)];
            }
        }

        // Copy remaining columns for V (if n > k)
        for j in k..n {
            for i in 0..n {
                v_sorted[(i, j)] = v[(i, j)];
            }
        }

        // Compute V^H
        let mut vh: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                vh[(i, j)] = v_sorted[(j, i)].conj();
            }
        }

        Ok(Self {
            u: u_sorted,
            sigma: sigma_sorted,
            vh,
            m,
            n,
        })
    }

    /// Returns the left singular vectors U (m×m unitary matrix).
    pub fn u(&self) -> &Mat<T> {
        &self.u
    }

    /// Returns the singular values (real, non-negative, descending).
    pub fn singular_values(&self) -> &[T::Real] {
        &self.sigma
    }

    /// Returns V^H (the conjugate transpose of right singular vectors).
    pub fn vh(&self) -> &Mat<T> {
        &self.vh
    }

    /// Returns the original matrix dimensions (m, n).
    pub fn shape(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Computes the rank using a given tolerance.
    pub fn rank(&self, tol: T::Real) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }

    /// Computes the 2-norm (largest singular value).
    pub fn norm_2(&self) -> T::Real {
        self.sigma.first().copied().unwrap_or(T::Real::zero())
    }

    /// Computes the condition number (ratio of largest to smallest singular value).
    pub fn cond(&self) -> T::Real {
        if self.sigma.is_empty() {
            return T::Real::zero();
        }
        let max = self.sigma[0];
        let min = self.sigma.last().copied().unwrap_or(T::Real::zero());
        if min > T::Real::zero() {
            max / min
        } else {
            T::Real::max_value()
        }
    }
}

/// Computes complex Jacobi rotation parameters to zero off-diagonal element.
/// Returns (c, s) where the rotation matrix is [[c, -conj(s)], [s, c]].
/// After rotation: b_i' = c*b_i + s*b_j, b_j' = -conj(s)*b_i + c*b_j
fn complex_jacobi_rotation<T: Field + ComplexScalar>(
    norm_i_sq: T::Real,
    norm_j_sq: T::Real,
    dot_ij: T,
) -> (T::Real, T)
where
    T::Real: Real,
{
    let eps = <T::Real as Scalar>::epsilon();
    let off_diag = dot_ij.abs();

    // Handle the case where dot_ij is essentially zero
    if off_diag < eps {
        return (T::Real::one(), T::zero());
    }

    // For complex Jacobi, we first compute a phase to make the problem real
    // phase = conj(dot_ij) / |dot_ij|, so dot_ij * phase = |dot_ij| (real and positive)
    let phase = T::from_real_imag(dot_ij.real() / off_diag, dot_ij.imag() / off_diag);

    // Now solve the real 2x2 eigenvalue problem for the Gram matrix:
    // [[norm_i_sq, |dot_ij|], [|dot_ij|, norm_j_sq]]
    // The rotation angle theta satisfies: tan(2*theta) = 2*|dot_ij| / (norm_i_sq - norm_j_sq)
    let two = T::Real::one() + T::Real::one();
    let diff = norm_i_sq - norm_j_sq;

    let (c, s_real) = if diff.abs() < eps * (norm_i_sq + norm_j_sq) {
        // When norms are equal, use 45 degree rotation
        let sqrt2_inv = T::Real::one() / <T::Real as Real>::sqrt(two);
        (sqrt2_inv, sqrt2_inv)
    } else {
        // Standard Jacobi rotation formula
        let tau = two * off_diag / diff;
        let t = if diff >= T::Real::zero() {
            tau / (T::Real::one() + <T::Real as Real>::sqrt(T::Real::one() + tau * tau))
        } else {
            -tau / (T::Real::one() + <T::Real as Real>::sqrt(T::Real::one() + tau * tau))
        };
        let c = T::Real::one() / <T::Real as Real>::sqrt(T::Real::one() + t * t);
        let s = c * t;
        (c, s)
    };

    // The complex s includes the phase to account for the complex off-diagonal
    // s = s_real * conj(phase) to zero the off-diagonal
    let s = T::from_real(s_real) * phase.conj();

    (c, s)
}

/// Completes a partial unitary matrix using Gram-Schmidt.
fn complete_unitary_matrix<T: Field + ComplexScalar>(u: &mut Mat<T>, m: usize, k: usize)
where
    T::Real: Real,
{
    let eps = <T::Real as Scalar>::epsilon();

    // Start with column k and try to add orthogonal columns
    let mut col = k;
    let mut candidate = 0;

    while col < m && candidate < m {
        // Try using a standard basis vector as starting point
        for i in 0..m {
            u[(i, col)] = if i == candidate { T::one() } else { T::zero() };
        }

        // Gram-Schmidt: orthogonalize against all previous columns
        for j in 0..col {
            // Compute dot product <u_j, u_col>
            let mut dot = T::zero();
            for i in 0..m {
                dot = dot + u[(i, j)].conj() * u[(i, col)];
            }

            // Subtract projection
            for i in 0..m {
                u[(i, col)] = u[(i, col)] - dot * u[(i, j)];
            }
        }

        // Normalize
        let mut norm_sq = T::Real::zero();
        for i in 0..m {
            norm_sq = norm_sq + u[(i, col)].abs_sq();
        }
        let norm = <T::Real as Real>::sqrt(norm_sq);

        if norm > eps {
            for i in 0..m {
                u[(i, col)] = u[(i, col)] / T::from_real(norm);
            }
            col += 1;
        }

        candidate += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn test_complex_svd_simple() {
        // Simple complex matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();

        // Singular values should be positive real numbers
        assert!(sigma[0] >= sigma[1], "Singular values should be descending");
        assert!(sigma[0] > 0.0, "First singular value should be positive");

        // Verify reconstruction: A = U Σ V^H
        let u = svd.u();
        let vh = svd.vh();
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..k {
                    sum = sum + u[(i, l)] * Complex64::new(sigma[l], 0.0) * vh[(l, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "A[{},{}] reconstruction error: {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_complex_svd_complex_entries() {
        // Complex matrix with imaginary parts
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(0.0, 2.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();
        let u = svd.u();
        let vh = svd.vh();

        // Verify U is unitary: U^H U = I
        let m = a.nrows();
        for i in 0..m {
            for j in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    sum = sum + u[(k, i)].conj() * u[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(diff < 1e-10, "U^H*U[{},{}] error: {}", i, j, diff);
            }
        }

        // Verify V^H V^H^H = V^H V = I (since we store V^H)
        let n = a.ncols();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + vh[(i, k)] * vh[(j, k)].conj();
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(diff < 1e-10, "V^H*V[{},{}] error: {}", i, j, diff);
            }
        }

        // Verify reconstruction
        let k = m.min(n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..k {
                    sum = sum + u[(i, l)] * Complex64::new(sigma[l], 0.0) * vh[(l, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "A[{},{}] reconstruction error: {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_complex_svd_hermitian() {
        // Hermitian positive definite matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();

        // For Hermitian positive definite, singular values equal eigenvalues
        // eigenvalues of [[4, 1-i], [1+i, 3]] are positive
        assert!(sigma[0] > 0.0);
        assert!(sigma[1] > 0.0);
    }

    #[test]
    fn test_complex_svd_tall() {
        // 3x2 complex matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0)],
            &[Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();
        let u = svd.u();
        let vh = svd.vh();

        assert_eq!(sigma.len(), 2); // min(3, 2) = 2 singular values
        assert_eq!(u.nrows(), 3);
        assert_eq!(u.ncols(), 3);
        assert_eq!(vh.nrows(), 2);
        assert_eq!(vh.ncols(), 2);

        // Verify reconstruction
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..k {
                    sum = sum + u[(i, l)] * Complex64::new(sigma[l], 0.0) * vh[(l, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(diff < 1e-9, "A[{},{}] reconstruction error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_wide() {
        // 2x3 complex matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, -1.0),
            ],
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 2); // min(2, 3) = 2 singular values
        assert!(sigma[0] >= sigma[1]);
    }

    #[test]
    fn test_complex_svd_identity() {
        // Complex identity matrix
        let a: Mat<Complex64> = Mat::eye(3);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();

        // All singular values should be 1
        for &s in sigma {
            assert!((s - 1.0).abs() < 1e-10, "Singular value = {}", s);
        }
    }

    #[test]
    fn test_complex_svd_complex32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(1.0, 0.0), Complex32::new(2.0, 1.0)],
            &[Complex32::new(3.0, -1.0), Complex32::new(4.0, 0.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();
        let u = svd.u();
        let vh = svd.vh();

        // Verify reconstruction
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex32::new(0.0, 0.0);
                for l in 0..k {
                    sum = sum + u[(i, l)] * Complex32::new(sigma[l], 0.0) * vh[(l, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(diff < 1e-4, "A[{},{}] reconstruction error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_1x1() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(3.0, 4.0)]]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();

        // Singular value should be |3+4i| = 5
        assert!((sigma[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_svd_diagonal() {
        // Diagonal complex matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        let svd = ComplexSvd::compute(a.as_ref()).expect("Should compute");
        let sigma = svd.singular_values();

        // Singular values should be |3+4i| = 5 and |1| = 1
        assert!((sigma[0] - 5.0).abs() < 1e-10, "sigma[0] = {}", sigma[0]);
        assert!((sigma[1] - 1.0).abs() < 1e-10, "sigma[1] = {}", sigma[1]);
    }
}
