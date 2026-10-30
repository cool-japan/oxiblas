//! QZ Algorithm (Generalized Schur Decomposition).
//!
//! Computes the generalized real Schur form of the matrix pencil (A, B):
//!   A = Q * S * Z^T
//!   B = Q * T * Z^T
//! where Q and Z are orthogonal, S is upper quasi-triangular (1×1 and 2×2
//! diagonal blocks) and T is upper triangular.
//!
//! [`Qz::compute`] is a Pure Rust equivalent of LAPACK `DGGES` with
//! `JOBVSL = JOBVSR = 'V'` and `SORT = 'N'` (Netlib LAPACK 3.12 control
//! flow and constants):
//!
//! 1. norm scaling of A and B when their largest entry is outside
//!    `[SMLNUM, BIGNUM]` (`DLANGE`/`DLASCL`),
//! 2. permutation balancing (`DGGBAL('P')`) and back-permutation of Q and Z
//!    (`DGGBAK('P')`),
//! 3. QR factorization of B and Hessenberg–triangular reduction (`DGGHRD`),
//! 4. the QZ iteration `DHGEQZ` (single and Francis double shifts,
//!    exceptional shifts, relative deflation criteria, deflation of
//!    infinite eigenvalues by zero chasing, `DLAG2`/`DLASV2` 2×2
//!    standardization).
//!
//! [`Qz::compute_sorted`] is the `DGGES` equivalent with `SORT = 'S'`: the
//! eigenvalues chosen by a selector are moved to the leading diagonal
//! positions by orthogonal swaps of adjacent diagonal blocks (`DTGSEN` with
//! `IJOB = 0`: `DTGEXC`, `DTGEX2`, `DTGSY2`, `DLAGV2`), so the leading
//! columns of `Q` and `Z` span the corresponding deflating subspaces;
//! [`Qz::reorder`] applies the same reordering to an existing decomposition.
//! Both report `SDIM` and the `DGGES` `INFO = N+2` condition in
//! [`QzOrdering`], and a rejected swap (`INFO = N+3`) as
//! [`QzReorderError::SwapFailed`].
//!
//! The generalized eigenvalues are reported in `DGGES` form
//! `λ = (alpha_real + i·alpha_imag) / beta` with `beta ≥ 0`; see
//! [`GeneralizedEigenvalue`]. For a 1×1 diagonal block, `alpha_real = S[j,j]`
//! and `beta = T[j,j]` exactly; for a 2×2 block (complex conjugate pair) the
//! triples come from the standardized block and are not ratios of diagonal
//! entries.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::Qz;
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);
//! let b = Mat::from_rows(&[&[2.0f64, 1.0, 0.0], &[0.0, 3.0, 1.0], &[1.0, 0.0, 4.0]]);
//! let qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();
//!
//! // A = Q S Zᵀ and B = Q T Zᵀ hold to working precision.
//! let (err_a, err_b) = qz.verify(a.as_ref(), b.as_ref());
//! assert!(err_a < 1e-12 && err_b < 1e-12);
//!
//! // T is upper triangular (exact zeros below the diagonal).
//! let t = qz.t();
//! assert!(t[(1, 0)] == 0.0 && t[(2, 0)] == 0.0 && t[(2, 1)] == 0.0);
//!
//! // Every eigenvalue is finite here (B is nonsingular) and β ≥ 0.
//! assert_eq!(qz.num_infinite(), 0);
//! assert!(qz.eigenvalues().iter().all(|e| e.beta >= 0.0));
//! ```

mod balance;
mod hessenberg;
mod hgeqz;
mod hgeqz_sweep;
mod lag2;
mod lagv2;
mod pipeline;
mod reorder;
mod scaling;
mod swap;
mod sylvester;
mod types;
mod util;

#[cfg(test)]
mod test_util;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_issue6;
#[cfg(test)]
mod tests_reorder;

pub use reorder::{QzOrdering, QzReorderError};
pub use types::{GeneralizedEigenvalue, QzError};

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// QZ decomposition (generalized Schur form).
///
/// For matrix pencil (A, B), computes:
///   A = Q * S * Z^T
///   B = Q * T * Z^T
///
/// where Q, Z are orthogonal, S is quasi-upper triangular, T is upper triangular.
///
/// This is the LAPACK `DGGES` decomposition (`JOBVSL = JOBVSR = 'V'`,
/// `SORT = 'N'`, or `SORT = 'S'` with [`Qz::compute_sorted`]): 2×2
/// diagonal blocks of S carry complex conjugate eigenvalue pairs and have a
/// diagonal, positive 2×2 block in T; every 1×1 block has `T[j,j] ≥ 0`.
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
    ///
    /// # Errors
    ///
    /// * [`QzError::EmptyMatrix`] if `a` has no rows,
    /// * [`QzError::NotSquare`] if `a` is not square,
    /// * [`QzError::DimensionMismatch`] if `b` is not the same size as `a`,
    /// * [`QzError::NotConverged`] if the QZ iteration exhausts its budget
    ///   of `30·(ihi − ilo + 1)` passes (`DGGES` `INFO > 0`), or with
    ///   `iterations: 0` if an input entry is NaN or infinite.
    pub fn compute(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Result<Self, QzError> {
        // Steps 0–6 of DGGES (validation … DHGEQZ). Step 7, the optional
        // eigenvalue reordering (DTGSEN), is not part of the unsorted
        // decomposition; see `pipeline` for the seam.
        let work = pipeline::run(a, b)?;
        // Steps 8–10: back-permutation, unscaling, eigenvalue triples.
        Ok(work.finish())
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
    ///
    /// More precisely, the values `α/β` of the finite real eigenvalues (see
    /// [`GeneralizedEigenvalue::value`]); complex and infinite eigenvalues
    /// are skipped.
    pub fn eigenvalues_real(&self) -> Vec<T> {
        self.eigenvalues.iter().filter_map(|e| e.value()).collect()
    }

    /// Returns the number of infinite eigenvalues (β = 0).
    ///
    /// Counts exact zeros of β (see [`GeneralizedEigenvalue::is_finite`]):
    /// after [`Qz::compute_sorted`] or [`Qz::reorder`], an infinite
    /// eigenvalue that took part in a swap usually carries a rounding-level
    /// β and is not counted.
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
