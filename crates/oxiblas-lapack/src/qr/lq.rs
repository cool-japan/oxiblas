//! LQ factorization.
//!
//! The LQ factorization decomposes a matrix A (m×n) into:
//! A = L * Q
//!
//! where:
//! - L is an m×n lower trapezoidal matrix
//! - Q is an n×n orthogonal (real) / unitary (complex) matrix
//!
//! This is the row-oriented dual of QR: instead of annihilating the sub-diagonal
//! entries of each *column* from the left, it annihilates the super-diagonal
//! entries of each *row* from the right.
//!
//! ## Real and complex scalars
//!
//! The implementation is generic over any [`Field`] whose real component type is
//! a [`Real`] (i.e. `f32`, `f64`, `Complex32`, `Complex64`). The reflector math
//! follows LAPACK's complex convention (`zgelqf`/`zlarfg`); for real scalars the
//! complex conjugations collapse to identities and the routine is numerically
//! identical to the classic real LQ.

use crate::error::LapackError;
use num_traits::Zero;
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{Mat, MatRef};

/// Generates the elementary Householder reflector used at one row of the LQ
/// factorization, following LAPACK `zlarfg`.
///
/// LQ is row-oriented: at row `i` we need a reflector applied *from the right*
/// that annihilates the trailing entries of the row. `zlarfg` generates a
/// reflector `H = I - tau * v * v^H` (with `v[0] = 1`) satisfying
///
/// ```text
///     H^H * conj(row)^T = beta * e_0        (beta is real)
/// ```
///
/// Conjugate-transposing that identity gives `row * H = beta * e_0^T`, i.e. the
/// *same* `H` annihilates the row when multiplied from the right. That is the
/// non-obvious trick: we feed the **conjugated** row into the standard column
/// reflector generator and then apply the resulting `H` (not its conjugate)
/// from the right, so the conjugation is folded into the generator and the
/// application stays a plain `I - tau * v * v^H`.
///
/// Returns `(v, tau, beta)` where `v[0] == 1`, `beta` (real, embedded in `T`)
/// becomes the diagonal entry `L[i,i]`, and `tau` is the (possibly complex)
/// reflector scalar. When the row tail is already zero *and* the head is
/// already real, the row is in final form: `tau == 0` and no reflection is
/// applied. Because `beta = -sign(Re(head)) * ||row||` is chosen with the
/// opposite sign of the head, the denominator `head - beta` never cancels, so
/// no division-by-zero guard is needed once we are past the early return.
fn lq_reflector<T: Field>(row: &[T]) -> (Vec<T>, T, T)
where
    T::Real: Real,
{
    let p = row.len();
    let mut v = vec![T::zero(); p];
    if p == 0 {
        return (v, T::zero(), T::zero());
    }

    // v[0] is implicitly 1 for every reflector.
    v[0] = T::one();

    // Tail norm^2 in the conjugated coordinate equals the tail norm^2 of the
    // original row: |conj(row[j])|^2 == |row[j]|^2.
    let mut tail_norm_sq = T::Real::zero();
    for &entry in &row[1..] {
        tail_norm_sq = tail_norm_sq + entry.abs_sq();
    }

    // Head of the conjugated row.
    let c0 = row[0].conj();

    // If the tail is already zero and the head is already real, the row is
    // already `(beta, 0, ..., 0)`: no reflector is required.
    if tail_norm_sq == T::Real::zero() && c0.imag() == T::Real::zero() {
        return (v, T::zero(), c0);
    }

    // beta = -sign(Re(c0)) * ||row||, real. The sign is chosen opposite the head
    // to maximize |c0 - beta| and avoid catastrophic cancellation.
    let norm = <T::Real as Real>::sqrt(c0.abs_sq() + tail_norm_sq);
    let beta_real = if c0.real() >= T::Real::zero() {
        -norm
    } else {
        norm
    };
    let beta = T::from_real(beta_real);

    // tau = (beta - c0) / beta  (complex when c0 is complex; beta is real).
    let tau = (beta - c0) / beta;

    // v[j] = conj(row[j]) / (c0 - beta) for j >= 1.
    let scale = T::one() / (c0 - beta);
    for j in 1..p {
        v[j] = row[j].conj() * scale;
    }

    (v, tau, beta)
}

/// LQ factorization result.
///
/// Contains the L factor and Householder reflector data to reconstruct Q.
#[derive(Debug, Clone)]
pub struct Lq<T: Field> {
    /// Combined L and Householder reflectors in compact storage.
    /// - Lower trapezoidal part contains L
    /// - Above diagonal contains the (conjugated) Householder vectors
    pub(crate) factors: Mat<T>,
    /// Scalar factors (tau) for Householder reflectors.
    pub(crate) tau: Vec<T>,
}

impl<T: Field + bytemuck::Zeroable> Lq<T>
where
    T::Real: Real,
{
    /// Computes the LQ factorization of a matrix.
    ///
    /// Works for both real (`f32`, `f64`) and complex (`Complex32`,
    /// `Complex64`) scalars. For complex input, `Q` is unitary (`Q Q^H = I`)
    /// and the factorization satisfies `A = L Q`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m×n)
    ///
    /// # Returns
    ///
    /// LQ factorization containing L factor and Q reflectors.
    ///
    /// # Errors
    ///
    /// Returns error if the factorization fails.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Lq;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    /// ]);
    ///
    /// let lq = Lq::compute(a.as_ref()).expect("LQ should succeed");
    /// let l = lq.l_factor();
    /// ```
    pub fn compute(a: MatRef<T>) -> Result<Self, LapackError> {
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        // Copy A to working matrix
        let mut factors = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                factors[(i, j)] = a[(i, j)];
            }
        }

        let mut tau = vec![T::zero(); k];

        // Perform Householder reflections row by row.
        for i in 0..k {
            let row_len = n - i;

            // Extract the current row segment [i, i..n].
            let mut row = vec![T::zero(); row_len];
            for j in 0..row_len {
                row[j] = factors[(i, i + j)];
            }

            // Compute the reflector H = I - tau * v * v^H such that
            // `row * H = beta * e_0^T` (see `lq_reflector`).
            let (v, tau_val, beta) = lq_reflector(&row);
            tau[i] = tau_val;

            // L[i,i] = beta; store the reflector tail v[1..] over the
            // (now annihilated) super-diagonal entries of row i.
            factors[(i, i)] = beta;
            for j in 1..row_len {
                factors[(i, i + j)] = v[j];
            }

            // Apply H from the RIGHT to the trailing rows i+1..m:
            //   b * H = b - tau * (b · v) * v^H
            // component-wise: b[j] -= tau * (sum_l b[l] * v[l]) * conj(v[j]).
            if tau_val != T::zero() {
                for r in (i + 1)..m {
                    let mut s = T::zero();
                    for j in 0..row_len {
                        s = s + factors[(r, i + j)] * v[j];
                    }
                    let ts = tau_val * s;
                    for j in 0..row_len {
                        factors[(r, i + j)] = factors[(r, i + j)] - ts * v[j].conj();
                    }
                }
            }
        }

        Ok(Self { factors, tau })
    }

    /// Extracts the L factor (lower trapezoidal).
    ///
    /// Returns an appropriately-shaped empty matrix when the factorization has
    /// zero rows or zero columns (avoids the `n - 1` underflow on empty input).
    #[must_use]
    pub fn l_factor(&self) -> Mat<T> {
        let m = self.factors.nrows();
        let n = self.factors.ncols();

        let mut l = Mat::zeros(m, n);

        // Empty factor: nothing to copy. This guard is required because the
        // `i.min(n - 1)` bound below would underflow a usize when `n == 0`.
        if m == 0 || n == 0 {
            return l;
        }

        for i in 0..m {
            for j in 0..=i.min(n - 1) {
                l[(i, j)] = self.factors[(i, j)];
            }
        }

        l
    }

    /// Returns the dimensions of the factorization.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.factors.nrows(), self.factors.ncols())
    }

    /// Extracts Q as an explicit matrix.
    ///
    /// This generates the full orthogonal/unitary matrix Q by applying the
    /// stored Householder reflections.
    ///
    /// `Q = H(k-1)^H · … · H(0)^H`, built by applying `M(i) = H(i)^H =
    /// I - conj(tau_i) * v_i * v_i^H` from the right in reverse order. For real
    /// scalars `conj(tau_i) == tau_i`, so this reduces to the classic real
    /// construction.
    #[must_use]
    pub fn q_factor(&self) -> Mat<T> {
        let m = self.factors.nrows();
        let n = self.factors.ncols();
        let k = m.min(n);

        // Start with identity.
        let mut q = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        for i in (0..k).rev() {
            let tau_i = self.tau[i];
            if tau_i == T::zero() {
                continue;
            }
            // Applying M(i) = I - conj(tau) * v * v^H from the right requires
            // conj(tau); v[0] = 1 and v[j] = factors[(i, i+j)] for j >= 1.
            let ctau = tau_i.conj();
            let row_len = n - i;

            for c in 0..n {
                // s = sum_j q[(c, i+j)] * v[j]
                let mut s = q[(c, i)];
                for j in 1..row_len {
                    s = s + q[(c, i + j)] * self.factors[(i, i + j)];
                }
                let ts = ctau * s;
                // q[(c, i+j)] -= conj(tau) * s * conj(v[j])
                q[(c, i)] = q[(c, i)] - ts;
                for j in 1..row_len {
                    q[(c, i + j)] = q[(c, i + j)] - ts * self.factors[(i, i + j)].conj();
                }
            }
        }

        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_lq_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();

        // L should be lower triangular
        assert!(l[(0, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_lq_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        assert_eq!(l.nrows(), 2);
        assert_eq!(l.ncols(), 3);
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 3);

        // Verify L is lower trapezoidal
        assert!(l[(0, 1)].abs() < 1e-10);
        assert!(l[(0, 2)].abs() < 1e-10);
        assert!(l[(1, 2)].abs() < 1e-10);
    }

    #[test]
    fn test_lq_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        // Reconstruct A = L * Q
        let mut reconstructed = Mat::zeros(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += l[(i, k)] * q[(k, j)];
                }
                reconstructed[(i, j)] = sum;
            }
        }

        // Check reconstruction
        for i in 0..2 {
            for j in 0..3 {
                assert!((reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lq_q_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let q = lq.q_factor();

        // Q^T * Q should be identity
        let mut qtq = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(k, i)] * q[(k, j)];
                }
                qtq[(i, j)] = sum;
            }
        }

        // Check orthogonality
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lq_tall_reconstruction() {
        // Tall matrix (m > n): L is m×n lower-triangular in its top n rows,
        // Q is n×n. Exercises the k = n < m branch.
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 9.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        assert_eq!(l.nrows(), 4);
        assert_eq!(l.ncols(), 2);
        assert_eq!(q.nrows(), 2);
        assert_eq!(q.ncols(), 2);

        for i in 0..4 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += l[(i, k)] * q[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-10,
                    "reconstruction mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_lq_empty_l_factor_does_not_panic() {
        // Empty matrices must not underflow / panic in l_factor(). Both the
        // 0-column and 0-row shapes are exercised (the 0-column case is the one
        // that triggered the `n - 1` usize underflow).
        let a_cols: Mat<f64> = Mat::zeros(3, 0);
        let lq_cols = Lq::compute(a_cols.as_ref()).unwrap();
        let l_cols = lq_cols.l_factor();
        assert_eq!(l_cols.nrows(), 3);
        assert_eq!(l_cols.ncols(), 0);
        assert_eq!(lq_cols.dims(), (3, 0));

        let a_rows: Mat<f64> = Mat::zeros(0, 4);
        let lq_rows = Lq::compute(a_rows.as_ref()).unwrap();
        let l_rows = lq_rows.l_factor();
        assert_eq!(l_rows.nrows(), 0);
        assert_eq!(l_rows.ncols(), 4);

        let a_empty: Mat<f64> = Mat::zeros(0, 0);
        let lq_empty = Lq::compute(a_empty.as_ref()).unwrap();
        let l_empty = lq_empty.l_factor();
        assert_eq!(l_empty.nrows(), 0);
        assert_eq!(l_empty.ncols(), 0);
    }

    #[test]
    fn test_lq_complex_square_reconstruction() {
        // Genuinely complex matrix (nonzero imaginary parts everywhere).
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.0, 2.0),
            ],
            &[
                Complex64::new(3.0, 0.5),
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 2.0),
            ],
            &[
                Complex64::new(0.0, -1.0),
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, -3.0),
            ],
        ]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        // A ≈ L * Q
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    sum += l[(i, k)] * q[(k, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "LQ[{i},{j}] = {sum:?}, A = {:?}, diff = {diff}",
                    a[(i, j)]
                );
            }
        }

        // L must be lower triangular (super-diagonal entries ≈ 0).
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    l[(i, j)].norm() < 1e-10,
                    "L not lower triangular at ({i}, {j}): {:?}",
                    l[(i, j)]
                );
            }
        }

        // Q must be unitary: Q Q^H = I.
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    // (Q Q^H)[i,j] = sum_k Q[i,k] * conj(Q[j,k])
                    sum += q[(i, k)] * q[(j, k)].conj();
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (sum - expected).norm() < 1e-10,
                    "Q not unitary at ({i}, {j}): {sum:?}"
                );
            }
        }
    }

    #[test]
    fn test_lq_complex_wide_reconstruction() {
        // Wide complex matrix (m < n): L is 2×3 lower trapezoidal, Q is 3×3.
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(2.0, 1.0),
                Complex64::new(-1.0, 3.0),
                Complex64::new(0.0, -2.0),
            ],
            &[
                Complex64::new(1.0, -1.0),
                Complex64::new(4.0, 0.5),
                Complex64::new(-2.0, 1.0),
            ],
        ]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        assert_eq!(l.nrows(), 2);
        assert_eq!(l.ncols(), 3);
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 3);

        // A ≈ L * Q
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    sum += l[(i, k)] * q[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).norm() < 1e-10,
                    "wide LQ reconstruction mismatch at ({i}, {j})"
                );
            }
        }

        // Trapezoidal structure: entries strictly right of the diagonal are 0.
        assert!(l[(0, 1)].norm() < 1e-10);
        assert!(l[(0, 2)].norm() < 1e-10);
        assert!(l[(1, 2)].norm() < 1e-10);

        // Q^H Q = I.
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    sum += q[(k, i)].conj() * q[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (sum - expected).norm() < 1e-10,
                    "Q^H Q not identity at ({i}, {j}): {sum:?}"
                );
            }
        }
    }

    #[test]
    fn test_lq_complex_tall_reconstruction() {
        // Tall complex matrix (m > n): exercises trailing-row reflector
        // application below the diagonal with complex arithmetic.
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
            &[Complex64::new(-2.0, 1.0), Complex64::new(0.0, 4.0)],
            &[Complex64::new(5.0, -3.0), Complex64::new(1.0, 1.0)],
        ]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 2);
        assert_eq!(q.nrows(), 2);
        assert_eq!(q.ncols(), 2);

        for i in 0..3 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum += l[(i, k)] * q[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).norm() < 1e-10,
                    "tall complex LQ reconstruction mismatch at ({i}, {j})"
                );
            }
        }

        // Q unitary.
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum += q[(i, k)] * q[(j, k)].conj();
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!((sum - expected).norm() < 1e-10);
            }
        }
    }
}
