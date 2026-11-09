//! Symmetric Eigenvalue Decomposition.
//!
//! Uses Householder tridiagonalization followed by the implicit QR algorithm.
//!
//! This module provides:
//!
//! - **SymmetricEvd**: Full eigendecomposition with explicit eigenvectors
//! - **TridiagFactors**: LAPACK-style compact storage for tridiagonal reduction (sytrd)
//! - **sytrd**: Reduce symmetric matrix to tridiagonal form with compact storage
//! - **orgtr**: Generate Q from compact tridiagonal factorization
//! - **ormtr**: Apply Q to a matrix without forming it explicitly
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::{sytrd, orgtr, ormtr, Uplo, TridiagSide, TridiagTrans};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 1.0, 1.0],
//!     &[1.0, 3.0, 2.0],
//!     &[1.0, 2.0, 3.0],
//! ]);
//!
//! // Compact tridiagonal factorization
//! let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
//!
//! // Generate Q explicitly (if needed)
//! let q = orgtr(&factors).unwrap();
//!
//! // Or apply Q to a matrix without forming it
//! let c = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
//! let qc = ormtr(&factors, TridiagSide::Left, TridiagTrans::NoTrans, c.as_ref()).unwrap();
//! ```

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::sym_kernels::{
    MatMul, Pipeline, argsort, lower_from_upper, permute_columns, scale_into_safe_range,
    with_gemm_backend,
};
use super::sym_steqr::{Deflation, RotTarget, RotationLog, steqr};
use super::sym_tridiag::{apply_q, reduce};

/// Error type for symmetric eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricEvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for SymmetricEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::NotConverged => write!(f, "Algorithm did not converge"),
        }
    }
}

impl std::error::Error for SymmetricEvdError {}

/// Symmetric eigenvalue decomposition.
///
/// Computes A = V·D·V^T where V contains eigenvectors and D is diagonal.
///
/// [`compute`](Self::compute) runs on the calling thread. With the `parallel`
/// feature, `compute_with_par` runs the GEMMs of the tridiagonal reduction
/// and of the back-transformation with a given [`Par`] and returns bitwise
/// the same result.
#[derive(Debug, Clone)]
pub struct SymmetricEvd<T: Scalar> {
    /// Eigenvalues (sorted in ascending order).
    eigenvalues: Vec<T>,
    /// Eigenvectors (columns of V).
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> SymmetricEvd<T> {
    /// Maximum number of QR iterations.
    const MAX_ITERATIONS: usize = 100;

    /// Computes the eigendecomposition of a symmetric matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (only upper triangle is used)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::SymmetricEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[2.0f64, 1.0],
    ///     &[1.0, 2.0],
    /// ]);
    ///
    /// let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
    /// let eigs = evd.eigenvalues();
    ///
    /// // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    /// assert!((eigs[0] - 1.0).abs() < 1e-10);
    /// assert!((eigs[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SymmetricEvdError> {
        Self::compute_impl(a, Par::Seq)
    }

    /// Same as [`Self::compute`], with the GEMMs of the tridiagonal reduction
    /// and of the back-transformation run with `par`.
    ///
    /// The result is bitwise identical to [`Self::compute`] for every `Par`:
    /// the parallel GEMM only partitions its output, and every other stage
    /// runs on the calling thread.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::compute`].
    #[cfg(feature = "parallel")]
    pub fn compute_with_par(a: MatRef<'_, T>, par: Par) -> Result<Self, SymmetricEvdError> {
        Self::compute_impl(a, par)
    }

    /// [`Self::compute`] with the GEMMs run with `par`.
    fn compute_impl(a: MatRef<'_, T>, par: Par) -> Result<Self, SymmetricEvdError> {
        let n = a.nrows();

        if n == 0 {
            return Err(SymmetricEvdError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(SymmetricEvdError::NotSquare);
        }

        // Handle trivial case
        if n == 1 {
            let eigenvalues = vec![a[(0, 0)]];
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues,
                eigenvectors,
                n,
            });
        }

        // The upper triangle of `a` becomes the lower triangle of the work
        // matrix, which the reduction overwrites with its reflectors.
        let mut work = lower_from_upper(a);
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Mat::zeros(n, n);
        with_gemm_backend(
            &mut work,
            &mut eigenvalues,
            &mut eigenvectors,
            QrPipeline {
                max_iter: Self::MAX_ITERATIONS,
                par,
            },
        )?;

        Ok(Self {
            eigenvalues,
            eigenvectors,
            n,
        })
    }

    /// Returns the eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix V.
    ///
    /// Column i contains the eigenvector corresponding to eigenvalue i.
    pub fn eigenvectors(&self) -> MatRef<'_, T> {
        self.eigenvectors.as_ref()
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = V * D * V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a = Mat::zeros(n, n);

        // A = V^T * D * V = sum_i lambda_i * v_i^T * v_i (if eigenvectors are rows)
        // OR A = V * D * V^T = sum_i lambda_i * v_i * v_i^T (if eigenvectors are columns)
        //
        // Current implementation assumes eigenvectors are columns
        for k in 0..n {
            let lambda = self.eigenvalues[k];
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] =
                        a[(i, j)] + lambda * self.eigenvectors[(i, k)] * self.eigenvectors[(j, k)];
                }
            }
        }

        a
    }
}

// ============================================================================
// LAPACK-style Tridiagonal Reduction (sytrd, orgtr, ormtr)
// ============================================================================

/// Upper or lower triangular storage for symmetric matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uplo {
    /// Use upper triangular part
    Upper,
    /// Use lower triangular part
    Lower,
}

/// Side of multiplication for ormtr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagSide {
    /// Multiply from the left: C <- Q * C or C <- Q^T * C
    Left,
    /// Multiply from the right: C <- C * Q or C <- C * Q^T
    Right,
}

/// Transpose option for ormtr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagTrans {
    /// No transpose: use Q
    NoTrans,
    /// Transpose: use Q^T (or Q^H for complex)
    Trans,
    /// Conjugate transpose: use Q^H (same as Trans for real)
    ConjTrans,
}

/// Compact tridiagonal factorization (LAPACK sytrd-style).
///
/// Stores the tridiagonal matrix T in diagonal and off-diagonal vectors,
/// with Householder vectors stored in the off-diagonal part of the original
/// matrix. This is more memory-efficient when eigenvectors are not needed
/// or will only be applied to matrices without forming them.
#[derive(Debug, Clone)]
pub struct TridiagFactors<T: Scalar> {
    /// Matrix with Householder vectors stored in the triangular part.
    /// - If uplo=Upper: vectors are stored above the superdiagonal
    /// - If uplo=Lower: vectors are stored below the subdiagonal
    factors: Mat<T>,
    /// Diagonal elements of the tridiagonal matrix T.
    pub(crate) diag: Vec<T>,
    /// Off-diagonal elements of T (length n-1).
    pub(crate) off_diag: Vec<T>,
    /// Scalar factors (tau) for Householder reflectors.
    tau: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Which triangular part was used.
    uplo: Uplo,
}

impl<T: Field + Real + bytemuck::Zeroable> TridiagFactors<T> {
    /// Returns the matrix dimension.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the upper/lower indicator.
    #[inline]
    pub fn uplo(&self) -> Uplo {
        self.uplo
    }

    /// Returns the diagonal of T.
    pub fn diag(&self) -> &[T] {
        &self.diag
    }

    /// Returns the off-diagonal of T.
    pub fn off_diag(&self) -> &[T] {
        &self.off_diag
    }

    /// Returns the tau values.
    pub fn tau(&self) -> &[T] {
        &self.tau
    }

    /// Returns reference to the factors matrix.
    pub fn factors(&self) -> MatRef<'_, T> {
        self.factors.as_ref()
    }

    /// Extracts the tridiagonal matrix T as a dense matrix.
    #[must_use]
    pub fn t(&self) -> Mat<T> {
        let mut t = Mat::zeros(self.n, self.n);
        for i in 0..self.n {
            t[(i, i)] = self.diag[i];
            if i + 1 < self.n {
                t[(i, i + 1)] = self.off_diag[i];
                t[(i + 1, i)] = self.off_diag[i];
            }
        }
        t
    }
}

/// Reduces a symmetric matrix to tridiagonal form (LAPACK SYTRD).
///
/// Computes the orthogonal matrix Q and symmetric tridiagonal matrix T
/// such that A = Q * T * Q^T. The Householder vectors are stored in
/// the triangular part of the output matrix for efficiency.
///
/// # Arguments
///
/// * `a` - Symmetric matrix (only the specified triangular part is used)
/// * `uplo` - Whether to use upper or lower triangular part
///
/// # Returns
///
/// `TridiagFactors` containing compact representation of the factorization.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::{sytrd, Uplo};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 1.0, 1.0],
///     &[1.0, 3.0, 2.0],
///     &[1.0, 2.0, 3.0],
/// ]);
///
/// let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
/// let t = factors.t();
///
/// // T is tridiagonal
/// assert!(t[(0, 2)].abs() < 1e-10);
/// assert!(t[(2, 0)].abs() < 1e-10);
/// ```
pub fn sytrd<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    uplo: Uplo,
) -> Result<TridiagFactors<T>, SymmetricEvdError> {
    let n = a.nrows();

    if n == 0 {
        return Err(SymmetricEvdError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(SymmetricEvdError::NotSquare);
    }

    // Copy symmetric matrix
    let mut factors = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            factors[(i, j)] = a[(i, j)];
        }
    }

    let mut diag = vec![T::zero(); n];
    let mut off_diag = vec![T::zero(); n.saturating_sub(1)];
    let mut tau = vec![T::zero(); n.saturating_sub(1)];

    if n <= 1 {
        if n == 1 {
            diag[0] = factors[(0, 0)];
        }
        return Ok(TridiagFactors {
            factors,
            diag,
            off_diag,
            tau,
            n,
            uplo,
        });
    }

    match uplo {
        Uplo::Upper => {
            // Reduce from top to bottom using upper triangular part
            for i in (1..n).rev() {
                // Householder to zero out row i-1, columns 0 to i-2
                // We work on column i-1 (elements 0 to i-1)
                let mut norm_sq = T::zero();
                for k in 0..i {
                    norm_sq = norm_sq + factors[(k, i)] * factors[(k, i)].conj();
                }
                let x_norm = Real::sqrt(norm_sq);

                if x_norm == T::zero() {
                    off_diag[i - 1] = T::zero();
                    tau[i - 1] = T::zero();
                    continue;
                }

                let x_pivot = factors[(i - 1, i)];
                let beta = -sign_of_real(x_pivot) * x_norm;
                off_diag[i - 1] = beta;

                // tau = (beta - x_pivot) / beta
                let tau_val = (beta - x_pivot) / beta;
                tau[i - 1] = tau_val;

                if tau_val == T::zero() {
                    continue;
                }

                // Scale v: v[k] = x[k] / (x_pivot - beta) for k < i-1
                // v[i-1] = 1 (implicit)
                let denom = x_pivot - beta;
                if Scalar::abs(denom) > <T as Scalar>::epsilon() {
                    for k in 0..(i - 1) {
                        factors[(k, i)] = factors[(k, i)] / denom;
                    }
                }
                // Store v[i-1] position as beta (will be the off-diagonal)
                factors[(i - 1, i)] = beta;

                // Apply symmetric update: A = A - tau * (v * w^T + w * v^T)
                // where w = A * v - (tau/2) * (v^T * A * v) * v

                // Compute A * v (using upper triangular storage)
                let mut av = vec![T::zero(); i];
                for row in 0..i {
                    let v_row = if row == i - 1 {
                        T::one()
                    } else {
                        factors[(row, i)]
                    };
                    for col in row..i {
                        let v_col = if col == i - 1 {
                            T::one()
                        } else {
                            factors[(col, i)]
                        };
                        let a_val = factors[(row, col)];
                        av[row] = av[row] + a_val * v_col;
                        if col != row {
                            av[col] = av[col] + a_val * v_row;
                        }
                    }
                }

                // Compute v^T * A * v
                let mut vtav = T::zero();
                for k in 0..i {
                    let v_k = if k == i - 1 {
                        T::one()
                    } else {
                        factors[(k, i)]
                    };
                    vtav = vtav + v_k.conj() * av[k];
                }

                // w = tau * av - (tau^2/2) * vtav * v
                let half_tau_sq_vtav = tau_val * tau_val * vtav / (T::one() + T::one());
                let mut w = vec![T::zero(); i];
                for k in 0..i {
                    let v_k = if k == i - 1 {
                        T::one()
                    } else {
                        factors[(k, i)]
                    };
                    w[k] = tau_val * av[k] - half_tau_sq_vtav * v_k;
                }

                // Update upper triangular part: A -= v * w^T + w * v^T
                for row in 0..i {
                    let v_row = if row == i - 1 {
                        T::one()
                    } else {
                        factors[(row, i)]
                    };
                    for col in row..i {
                        let v_col = if col == i - 1 {
                            T::one()
                        } else {
                            factors[(col, i)]
                        };
                        factors[(row, col)] =
                            factors[(row, col)] - v_row * w[col].conj() - w[row] * v_col.conj();
                    }
                }
            }

            // Extract diagonal
            for i in 0..n {
                diag[i] = factors[(i, i)];
            }
        }
        Uplo::Lower => {
            // Reduce from bottom to top using lower triangular part
            for i in 0..(n - 1) {
                // Householder to zero out column i, rows i+2 to n-1
                let mut norm_sq = T::zero();
                for k in (i + 1)..n {
                    norm_sq = norm_sq + factors[(k, i)] * factors[(k, i)].conj();
                }
                let x_norm = Real::sqrt(norm_sq);

                if x_norm == T::zero() {
                    off_diag[i] = T::zero();
                    tau[i] = T::zero();
                    continue;
                }

                let x_pivot = factors[(i + 1, i)];
                let beta = -sign_of_real(x_pivot) * x_norm;
                off_diag[i] = beta;

                // tau = (beta - x_pivot) / beta
                let tau_val = (beta - x_pivot) / beta;
                tau[i] = tau_val;

                if tau_val == T::zero() {
                    continue;
                }

                // Scale v: v[k] = x[k] / (x_pivot - beta) for k > i+1
                // v[i+1] = 1 (implicit)
                let denom = x_pivot - beta;
                if Scalar::abs(denom) > <T as Scalar>::epsilon() {
                    for k in (i + 2)..n {
                        factors[(k, i)] = factors[(k, i)] / denom;
                    }
                }
                // Store beta in the off-diagonal position
                factors[(i + 1, i)] = beta;

                // Apply symmetric update using lower triangular storage
                let len = n - i - 1;
                let mut av = vec![T::zero(); len];
                for row_idx in 0..len {
                    let row = i + 1 + row_idx;
                    let v_row = if row_idx == 0 {
                        T::one()
                    } else {
                        factors[(row, i)]
                    };
                    for col_idx in 0..=row_idx {
                        let col = i + 1 + col_idx;
                        let v_col = if col_idx == 0 {
                            T::one()
                        } else {
                            factors[(col, i)]
                        };
                        let a_val = factors[(row, col)];
                        av[row_idx] = av[row_idx] + a_val * v_col;
                        if col_idx != row_idx {
                            av[col_idx] = av[col_idx] + a_val * v_row;
                        }
                    }
                }

                // Compute v^T * A * v
                let mut vtav = T::zero();
                for k in 0..len {
                    let v_k = if k == 0 {
                        T::one()
                    } else {
                        factors[(i + 1 + k, i)]
                    };
                    vtav = vtav + v_k.conj() * av[k];
                }

                // w = tau * av - (tau^2/2) * vtav * v
                let half_tau_sq_vtav = tau_val * tau_val * vtav / (T::one() + T::one());
                let mut w = vec![T::zero(); len];
                for k in 0..len {
                    let v_k = if k == 0 {
                        T::one()
                    } else {
                        factors[(i + 1 + k, i)]
                    };
                    w[k] = tau_val * av[k] - half_tau_sq_vtav * v_k;
                }

                // Update lower triangular part
                for row_idx in 0..len {
                    let row = i + 1 + row_idx;
                    let v_row = if row_idx == 0 {
                        T::one()
                    } else {
                        factors[(row, i)]
                    };
                    for col_idx in 0..=row_idx {
                        let col = i + 1 + col_idx;
                        let v_col = if col_idx == 0 {
                            T::one()
                        } else {
                            factors[(col, i)]
                        };
                        factors[(row, col)] = factors[(row, col)]
                            - v_row * w[col_idx].conj()
                            - w[row_idx] * v_col.conj();
                    }
                }
            }

            // Extract diagonal
            for i in 0..n {
                diag[i] = factors[(i, i)];
            }
        }
    }

    Ok(TridiagFactors {
        factors,
        diag,
        off_diag,
        tau,
        n,
        uplo,
    })
}

/// Generates the orthogonal matrix Q from tridiagonal factorization (LAPACK ORGTR/UNGTR).
///
/// Given the output of `sytrd`, generates the orthogonal matrix Q such that
/// A = Q * T * Q^T.
///
/// # Arguments
///
/// * `factors` - The compact tridiagonal factorization from `sytrd`
///
/// # Returns
///
/// The orthogonal matrix Q (n×n).
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::{sytrd, orgtr, Uplo};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 1.0, 1.0],
///     &[1.0, 3.0, 2.0],
///     &[1.0, 2.0, 3.0],
/// ]);
///
/// let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
/// let q = orgtr(&factors).unwrap();
///
/// // Q should be orthogonal: Q^T * Q = I
/// ```
pub fn orgtr<T: Field + Real + bytemuck::Zeroable>(
    factors: &TridiagFactors<T>,
) -> Result<Mat<T>, SymmetricEvdError> {
    let n = factors.n;

    // Initialize Q as identity
    let mut q = Mat::zeros(n, n);
    for i in 0..n {
        q[(i, i)] = T::one();
    }

    if n <= 1 {
        return Ok(q);
    }

    match factors.uplo {
        Uplo::Upper => {
            // Apply reflectors in forward order: Q = H(n-2) * H(n-3) * ... * H(0)
            // To form Q, apply from left: Q <- H(0) * I, Q <- H(1) * Q, ..., Q <- H(n-2) * Q
            // Reflector H_k zeros column k+1, rows 0 to k
            for k in 0..(n - 1) {
                let i = k + 1; // The column we zeroed
                let len = i; // Number of elements in reflector (0 to i-1, plus implicit 1 at i-1)

                if factors.tau[k] == T::zero() || len == 0 {
                    continue;
                }

                // Reconstruct v from factors
                // v is stored in factors[(0..i-1, i)], with v[i-1] = 1 implicit
                let mut v = vec![T::zero(); i];
                for j in 0..(i - 1) {
                    v[j] = factors.factors[(j, i)];
                }
                v[i - 1] = T::one();

                let tau_val = factors.tau[k];

                // Apply H_k = I - tau * v * v^H from the left to Q
                // Q[0:i, :] <- H_k * Q[0:i, :]
                for col in 0..n {
                    let mut dot = T::zero();
                    for j in 0..i {
                        dot = dot + v[j].conj() * q[(j, col)];
                    }
                    let scaled = tau_val * dot;
                    for j in 0..i {
                        q[(j, col)] = q[(j, col)] - scaled * v[j];
                    }
                }
            }
        }
        Uplo::Lower => {
            // Apply reflectors: Q = P_0 * P_1 * ... * P_{n-2}
            // Reflector P_k zeros column k, rows k+2 to n-1
            for k in 0..(n - 1) {
                let len = n - k - 1; // Number of elements affected

                if factors.tau[k] == T::zero() || len == 0 {
                    continue;
                }

                // Reconstruct v from factors
                // v is stored in factors[(k+2..n, k)], with v[0] = 1 implicit
                let mut v = vec![T::one(); len];
                for j in 1..len {
                    v[j] = factors.factors[(k + 1 + j, k)];
                }

                let tau_val = factors.tau[k];

                // Apply P_k = I - tau * v * v^H from the right to Q
                // Q[:, k+1:n] <- Q[:, k+1:n] * P_k
                for row in 0..n {
                    let mut dot = T::zero();
                    for j in 0..len {
                        dot = dot + q[(row, k + 1 + j)] * v[j];
                    }
                    let scaled = tau_val * dot;
                    for j in 0..len {
                        q[(row, k + 1 + j)] = q[(row, k + 1 + j)] - scaled * v[j].conj();
                    }
                }
            }
        }
    }

    Ok(q)
}

/// Alias for `orgtr` for complex matrices (LAPACK UNGTR).
pub fn ungtr<T: Field + Real + bytemuck::Zeroable>(
    factors: &TridiagFactors<T>,
) -> Result<Mat<T>, SymmetricEvdError> {
    orgtr(factors)
}

/// Multiplies a matrix by Q from tridiagonal factorization (LAPACK ORMTR/UNMTR).
///
/// Computes one of:
/// - C <- Q * C   (side=Left, trans=NoTrans)
/// - C <- Q^T * C (side=Left, trans=Trans or ConjTrans)
/// - C <- C * Q   (side=Right, trans=NoTrans)
/// - C <- C * Q^T (side=Right, trans=Trans or ConjTrans)
///
/// # Arguments
///
/// * `factors` - The compact tridiagonal factorization from `sytrd`
/// * `side` - Whether to multiply from left or right
/// * `trans` - Whether to use Q or Q^T
/// * `c` - Input matrix to multiply
///
/// # Returns
///
/// The result of the multiplication.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::evd::{sytrd, ormtr, Uplo, TridiagSide, TridiagTrans};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[4.0f64, 1.0, 1.0],
///     &[1.0, 3.0, 2.0],
///     &[1.0, 2.0, 3.0],
/// ]);
///
/// let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
/// let c = Mat::from_rows(&[&[1.0], &[0.0], &[0.0]]);
///
/// // Compute Q * c without forming Q explicitly
/// let qc = ormtr(&factors, TridiagSide::Left, TridiagTrans::NoTrans, c.as_ref()).unwrap();
/// ```
pub fn ormtr<T: Field + Real + bytemuck::Zeroable>(
    factors: &TridiagFactors<T>,
    side: TridiagSide,
    trans: TridiagTrans,
    c: MatRef<'_, T>,
) -> Result<Mat<T>, SymmetricEvdError> {
    let n = factors.n;
    let c_rows = c.nrows();
    let c_cols = c.ncols();

    // Validate dimensions
    match side {
        TridiagSide::Left => {
            if c_rows != n {
                return Err(SymmetricEvdError::NotSquare);
            }
        }
        TridiagSide::Right => {
            if c_cols != n {
                return Err(SymmetricEvdError::NotSquare);
            }
        }
    }

    // Copy C to result
    let mut result = Mat::zeros(c_rows, c_cols);
    for i in 0..c_rows {
        for j in 0..c_cols {
            result[(i, j)] = c[(i, j)];
        }
    }

    if n <= 1 {
        return Ok(result);
    }

    // Determine iteration order based on side, trans, and uplo
    // For Upper: Q = H(n-2) * H(n-3) * ... * H(0)
    //   Left + NoTrans (Q*C): Apply H(0), H(1), ..., H(n-2) from left -> forward
    //   Left + Trans (Q^T*C): Apply H(n-2), H(n-3), ..., H(0) from left -> reverse
    // For Lower: Q = H(0) * H(1) * ... * H(n-2)
    //   Left + NoTrans (Q*C): Apply H(n-2), ..., H(0) from left -> reverse
    //   Left + Trans (Q^T*C): Apply H(0), ..., H(n-2) from left -> forward
    let forward = match (factors.uplo, side, trans) {
        (Uplo::Upper, TridiagSide::Left, TridiagTrans::NoTrans) => true,
        (Uplo::Upper, TridiagSide::Left, _) => false,
        (Uplo::Upper, TridiagSide::Right, TridiagTrans::NoTrans) => false,
        (Uplo::Upper, TridiagSide::Right, _) => true,
        (Uplo::Lower, TridiagSide::Left, TridiagTrans::NoTrans) => false,
        (Uplo::Lower, TridiagSide::Left, _) => true,
        (Uplo::Lower, TridiagSide::Right, TridiagTrans::NoTrans) => true,
        (Uplo::Lower, TridiagSide::Right, _) => false,
    };

    let k_range: Vec<usize> = if forward {
        (0..(n - 1)).collect()
    } else {
        (0..(n - 1)).rev().collect()
    };

    for k in k_range {
        let tau_val = match trans {
            TridiagTrans::NoTrans => factors.tau[k],
            TridiagTrans::Trans | TridiagTrans::ConjTrans => factors.tau[k].conj(),
        };

        if tau_val == T::zero() {
            continue;
        }

        match factors.uplo {
            Uplo::Upper => {
                let i = k + 1; // Column that was zeroed
                if i == 0 {
                    continue;
                }

                // Reconstruct v
                let mut v = vec![T::zero(); i];
                for j in 0..(i - 1) {
                    v[j] = factors.factors[(j, i)];
                }
                v[i - 1] = T::one();

                match side {
                    TridiagSide::Left => {
                        // result[0:i, :] <- P_k * result[0:i, :]
                        for col in 0..c_cols {
                            let mut dot = T::zero();
                            for j in 0..i {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j].conj(),
                                    _ => v[j],
                                };
                                dot = dot + v_elem * result[(j, col)];
                            }
                            let scaled = tau_val * dot;
                            for j in 0..i {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j],
                                    _ => v[j].conj(),
                                };
                                result[(j, col)] = result[(j, col)] - scaled * v_elem;
                            }
                        }
                    }
                    TridiagSide::Right => {
                        // result[:, 0:i] <- result[:, 0:i] * P_k
                        for row in 0..c_rows {
                            let mut dot = T::zero();
                            for j in 0..i {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j],
                                    _ => v[j].conj(),
                                };
                                dot = dot + result[(row, j)] * v_elem;
                            }
                            let scaled = tau_val * dot;
                            for j in 0..i {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j].conj(),
                                    _ => v[j],
                                };
                                result[(row, j)] = result[(row, j)] - scaled * v_elem;
                            }
                        }
                    }
                }
            }
            Uplo::Lower => {
                let len = n - k - 1;
                if len == 0 {
                    continue;
                }

                // Reconstruct v
                let mut v = vec![T::one(); len];
                for j in 1..len {
                    v[j] = factors.factors[(k + 1 + j, k)];
                }

                match side {
                    TridiagSide::Left => {
                        // result[k+1:n, :] <- P_k * result[k+1:n, :]
                        for col in 0..c_cols {
                            let mut dot = T::zero();
                            for j in 0..len {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j].conj(),
                                    _ => v[j],
                                };
                                dot = dot + v_elem * result[(k + 1 + j, col)];
                            }
                            let scaled = tau_val * dot;
                            for j in 0..len {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j],
                                    _ => v[j].conj(),
                                };
                                result[(k + 1 + j, col)] =
                                    result[(k + 1 + j, col)] - scaled * v_elem;
                            }
                        }
                    }
                    TridiagSide::Right => {
                        // result[:, k+1:n] <- result[:, k+1:n] * P_k
                        for row in 0..c_rows {
                            let mut dot = T::zero();
                            for j in 0..len {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j],
                                    _ => v[j].conj(),
                                };
                                dot = dot + result[(row, k + 1 + j)] * v_elem;
                            }
                            let scaled = tau_val * dot;
                            for j in 0..len {
                                let v_elem = match trans {
                                    TridiagTrans::NoTrans => v[j].conj(),
                                    _ => v[j],
                                };
                                result[(row, k + 1 + j)] =
                                    result[(row, k + 1 + j)] - scaled * v_elem;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Alias for `ormtr` for complex matrices (LAPACK UNMTR).
pub fn unmtr<T: Field + Real + bytemuck::Zeroable>(
    factors: &TridiagFactors<T>,
    side: TridiagSide,
    trans: TridiagTrans,
    c: MatRef<'_, T>,
) -> Result<Mat<T>, SymmetricEvdError> {
    ormtr(factors, side, trans, c)
}

/// Returns sign of x (1 for x >= 0, -1 for x < 0).
fn sign_of_real<T: Field + Real>(x: T) -> T {
    if x >= T::zero() { T::one() } else { -T::one() }
}

/// Tridiagonal reduction, implicit QR on the tridiagonal, eigenvector
/// assembly (LAPACK `DSTEQR` with `COMPZ = 'I'` followed by `DORMTR`).
///
/// The rotations are accumulated into the identity rather than into an
/// explicitly formed `Q`: the batched application then skips the rows each
/// column has not filled in yet, and the final `Q·Z` is GEMM-rich.
#[derive(Debug, Clone, Copy)]
struct QrPipeline {
    /// Iteration limit per row of the QR iteration.
    max_iter: usize,
    /// Parallelism of the GEMMs.
    par: Par,
}

impl Pipeline for QrPipeline {
    type Output = Result<(), SymmetricEvdError>;

    fn run<T, G>(self, work: &mut Mat<T>, values: &mut Vec<T>, vectors: &mut Mat<T>) -> Self::Output
    where
        T: Field + Real + bytemuck::Zeroable,
        G: MatMul<T>,
    {
        let n = work.nrows();
        let sigma = scale_into_safe_range(work);
        let tri = reduce::<T, G>(work, self.par);
        let mut d = tri.d;
        let mut e = tri.e;
        for i in 0..n {
            vectors[(i, i)] = T::one();
        }

        let ld = vectors.row_stride();
        let mut log = RotationLog::new();
        let target = RotTarget {
            data: vectors.raw_data_mut(),
            off: 0,
            ld,
            nrows: n,
            identity: true,
        };
        steqr(
            &mut d,
            &mut e,
            Some(target),
            self.max_iter,
            Deflation::Relaxed,
            &mut log,
        )
        .map_err(|_| SymmetricEvdError::NotConverged)?;

        // Ascending order: sort indices, then one in-place column permutation.
        let mut perm = Vec::with_capacity(n);
        argsort(&d, &mut perm);
        values.clear();
        values.extend(perm.iter().map(|&p| d[p] / sigma));
        let (mut saved, mut visited) = (Vec::with_capacity(n), Vec::with_capacity(n));
        permute_columns(
            vectors.raw_data_mut(),
            0,
            ld,
            n,
            &perm,
            &mut saved,
            &mut visited,
        );

        apply_q::<T, G>(work, &tri.tau, vectors, self.par);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_evd_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 2.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));

        // Verify eigenvectors
        let v = evd.eigenvectors();

        // V^T * V should be identity
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_evd_3x3() {
        // Symmetric 3x3 matrix
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();

        // Test A*v = λ*v for each eigenvector
        for col in 0..3 {
            let lambda = evd.eigenvalues()[col];
            let mut v_col = [0.0; 3];
            for row in 0..3 {
                v_col[row] = v[(row, col)];
            }

            let mut av = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    av[i] += a[(i, j)] * v_col[j];
                }
            }

            // Check A*v ≈ λ*v
            for i in 0..3 {
                assert!(
                    approx_eq(av[i], lambda * v_col[i], 1e-9),
                    "A*v[{}] = {}, λ*v[{}] = {}",
                    i,
                    av[i],
                    i,
                    lambda * v_col[i]
                );
            }
        }

        // Reconstruct and verify
        let reconstructed = evd.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
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
    fn test_evd_diagonal() {
        // Diagonal matrix - eigenvalues are the diagonal elements
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 2.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues sorted: 1, 2, 3
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let evd = SymmetricEvd::compute(eye.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // All eigenvalues should be 1
        for &e in eigs {
            assert!(approx_eq(e, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_evd_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 1);
        assert!(approx_eq(eigs[0], 5.0, 1e-10));
    }

    #[test]
    fn test_evd_negative_eigenvalues() {
        // Matrix with negative eigenvalues
        let a = Mat::from_rows(&[&[-2.0f64, 1.0], &[1.0, -2.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues: -3 and -1
        assert!(approx_eq(eigs[0], -3.0, 1e-10));
        assert!(approx_eq(eigs[1], -1.0, 1e-10));
    }

    #[test]
    fn test_evd_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[1.0, 2.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_evd_repeated_eigenvalues() {
        // Matrix with repeated eigenvalue (3 appears twice)
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 1.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_orthogonality() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();

        // Verify V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-9),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    // Tests for LAPACK-style tridiagonal reduction (sytrd, orgtr, ormtr)

    #[test]
    fn test_sytrd_upper_3x3() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
        let t = factors.t();

        // T should be tridiagonal
        assert!(
            t[(0, 2)].abs() < 1e-10,
            "T[0,2] = {} should be zero",
            t[(0, 2)]
        );
        assert!(
            t[(2, 0)].abs() < 1e-10,
            "T[2,0] = {} should be zero",
            t[(2, 0)]
        );
    }

    #[test]
    fn test_sytrd_lower_3x3() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Lower).unwrap();
        let t = factors.t();

        // T should be tridiagonal
        assert!(
            t[(0, 2)].abs() < 1e-10,
            "T[0,2] = {} should be zero",
            t[(0, 2)]
        );
        assert!(
            t[(2, 0)].abs() < 1e-10,
            "T[2,0] = {} should be zero",
            t[(2, 0)]
        );
    }

    #[test]
    fn test_orgtr_upper_orthogonal() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
        let q = orgtr(&factors).unwrap();

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
    fn test_orgtr_lower_orthogonal() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Lower).unwrap();
        let q = orgtr(&factors).unwrap();

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
    fn test_sytrd_orgtr_reconstruction_upper() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
        let t = factors.t();
        let q = orgtr(&factors).unwrap();

        // Reconstruct A = Q * T * Q^T
        let n = 3;
        let mut qt = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * t[(k, j)];
                }
                qt[(i, j)] = sum;
            }
        }

        let mut reconstructed = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += qt[(i, k)] * q[(j, k)]; // Q^T
                }
                reconstructed[(i, j)] = sum;
            }
        }

        for i in 0..n {
            for j in 0..n {
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
    fn test_sytrd_orgtr_reconstruction_lower() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Lower).unwrap();
        let t = factors.t();
        let q = orgtr(&factors).unwrap();

        // Reconstruct A = Q * T * Q^T
        let n = 3;
        let mut qt = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * t[(k, j)];
                }
                qt[(i, j)] = sum;
            }
        }

        let mut reconstructed = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += qt[(i, k)] * q[(j, k)]; // Q^T
                }
                reconstructed[(i, j)] = sum;
            }
        }

        for i in 0..n {
            for j in 0..n {
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
    fn test_ormtr_left_notrans() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Lower).unwrap();
        let q = orgtr(&factors).unwrap();

        // Test vector
        let c = Mat::from_rows(&[&[1.0f64], &[0.0], &[0.0]]);

        // Using ormtr
        let qc_ormtr = ormtr(
            &factors,
            TridiagSide::Left,
            TridiagTrans::NoTrans,
            c.as_ref(),
        )
        .unwrap();

        // Using explicit Q multiplication
        let n = 3;
        let mut qc_explicit = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(i, k)] * c[(k, 0)];
            }
            qc_explicit[(i, 0)] = sum;
        }

        for i in 0..n {
            assert!(
                approx_eq(qc_ormtr[(i, 0)], qc_explicit[(i, 0)], 1e-10),
                "ormtr[{}] = {}, explicit = {}",
                i,
                qc_ormtr[(i, 0)],
                qc_explicit[(i, 0)]
            );
        }
    }

    #[test]
    fn test_ormtr_left_trans() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Lower).unwrap();
        let q = orgtr(&factors).unwrap();

        // Test vector
        let c = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);

        // Using ormtr with transpose
        let qtc_ormtr =
            ormtr(&factors, TridiagSide::Left, TridiagTrans::Trans, c.as_ref()).unwrap();

        // Using explicit Q^T multiplication
        let n = 3;
        let mut qtc_explicit = Mat::zeros(n, 1);
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[(k, i)] * c[(k, 0)]; // Q^T[i,k] = Q[k,i]
            }
            qtc_explicit[(i, 0)] = sum;
        }

        for i in 0..n {
            assert!(
                approx_eq(qtc_ormtr[(i, 0)], qtc_explicit[(i, 0)], 1e-10),
                "ormtr_trans[{}] = {}, explicit = {}",
                i,
                qtc_ormtr[(i, 0)],
                qtc_explicit[(i, 0)]
            );
        }
    }

    #[test]
    fn test_sytrd_identity() {
        let eye: Mat<f64> = Mat::eye(4);
        let factors = sytrd(eye.as_ref(), Uplo::Upper).unwrap();
        let q = orgtr(&factors).unwrap();
        let t = factors.t();

        // For identity, Q and T should both be identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(q[(i, j)], expected, 1e-10),
                    "Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    q[(i, j)],
                    expected
                );
                assert!(
                    approx_eq(t[(i, j)], expected, 1e-10),
                    "T[{},{}] = {}, expected {}",
                    i,
                    j,
                    t[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_sytrd_2x2() {
        // 2x2 is already tridiagonal
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let factors = sytrd(a.as_ref(), Uplo::Lower).unwrap();
        let t = factors.t();

        // T should match original symmetric matrix
        assert!(approx_eq(t[(0, 0)], a[(0, 0)], 1e-10));
        assert!(approx_eq(t[(1, 1)], a[(1, 1)], 1e-10));
        // Off-diagonal might have different sign due to reflector
        assert!(approx_eq(t[(0, 1)].abs(), a[(0, 1)].abs(), 1e-10));
    }

    #[test]
    fn test_sytrd_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let factors = sytrd(a.as_ref(), Uplo::Upper).unwrap();
        let q = orgtr(&factors).unwrap();

        // Check Q is orthogonal
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0f32;
                for k in 0..3 {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
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
    fn test_sytrd_4x4() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, 2.0, 3.0],
            &[1.0, 5.0, 1.0, 2.0],
            &[2.0, 1.0, 6.0, 1.0],
            &[3.0, 2.0, 1.0, 7.0],
        ]);

        for uplo in [Uplo::Upper, Uplo::Lower] {
            let factors = sytrd(a.as_ref(), uplo).unwrap();
            let t = factors.t();
            let q = orgtr(&factors).unwrap();

            // T should be tridiagonal
            for i in 0..4 {
                for j in 0..4 {
                    if (i as isize - j as isize).abs() > 1 {
                        assert!(
                            t[(i, j)].abs() < 1e-10,
                            "T[{},{}] = {} should be zero (uplo={:?})",
                            i,
                            j,
                            t[(i, j)],
                            uplo
                        );
                    }
                }
            }

            // Verify reconstruction
            let n = 4;
            let mut qt = Mat::zeros(n, n);
            for i in 0..n {
                for jj in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += q[(i, k)] * t[(k, jj)];
                    }
                    qt[(i, jj)] = sum;
                }
            }

            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += qt[(i, k)] * q[(j, k)];
                    }
                    assert!(
                        approx_eq(sum, a[(i, j)], 1e-9),
                        "reconstructed[{},{}] = {}, a = {} (uplo={:?})",
                        i,
                        j,
                        sum,
                        a[(i, j)],
                        uplo
                    );
                }
            }
        }
    }

    /// The generic (plain-loop) pipeline used for scalar types without a GEMM
    /// kernel agrees with the GEMM pipeline, including the blocked paths.
    #[test]
    fn test_generic_pipeline_matches_gemm_pipeline() {
        use super::super::sym_kernels::{BlasMatMul, NaiveMatMul, lower_from_upper};

        for (idx, &n) in [2usize, 9, 70, 190].iter().enumerate() {
            let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (idx as u64 + 1);
            let mut a = Mat::zeros(n, n);
            for j in 0..n {
                for i in 0..=j {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let v = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                    a[(i, j)] = v;
                    a[(j, i)] = v;
                }
            }
            let run = |naive: bool| {
                let mut work = lower_from_upper(a.as_ref());
                let mut vals = Vec::new();
                let mut vecs = Mat::zeros(n, n);
                let p = QrPipeline {
                    max_iter: 100,
                    par: Par::Seq,
                };
                let r = if naive {
                    p.run::<f64, NaiveMatMul>(&mut work, &mut vals, &mut vecs)
                } else {
                    p.run::<f64, BlasMatMul>(&mut work, &mut vals, &mut vecs)
                };
                r.unwrap();
                (vals, vecs)
            };
            let (v_blas, z_blas) = run(false);
            let (v_gen, z_gen) = run(true);
            let scale = v_blas.iter().fold(1.0f64, |m, &x| m.max(x.abs()));
            for i in 0..n {
                assert!(
                    approx_eq(v_blas[i], v_gen[i], 1e-12 * scale),
                    "n={}: eigenvalue {} differs: {} vs {}",
                    n,
                    i,
                    v_blas[i],
                    v_gen[i]
                );
                let dot: f64 = (0..n).map(|r| z_blas[(r, i)] * z_gen[(r, i)]).sum();
                assert!(
                    approx_eq(dot.abs(), 1.0, 1e-10),
                    "n={}: eigenvector {} differs (|dot| = {})",
                    n,
                    i,
                    dot
                );
            }
            // Residual of the generic result.
            for col in 0..n {
                for row in 0..n {
                    let av: f64 = (0..n).map(|k| a[(row, k)] * z_gen[(k, col)]).sum();
                    assert!(
                        approx_eq(av, v_gen[col] * z_gen[(row, col)], 1e-11 * scale),
                        "n={}: residual at ({}, {})",
                        n,
                        row,
                        col
                    );
                }
            }
        }
    }

    #[test]
    fn test_evd_numrs2_failing_matrix() {
        // Matrix that previously failed due to off-diagonal sign bug in QR algorithm
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 1.0], &[1.0, 1.0, 2.0]]);

        let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();

        // Verify eigenvectors are orthogonal
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-9),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Verify A*v = λ*v for each eigenvector
        for col in 0..3 {
            let lambda = evd.eigenvalues()[col];
            for i in 0..3 {
                let mut av_i = 0.0;
                for j in 0..3 {
                    av_i += a[(i, j)] * v[(j, col)];
                }
                assert!(
                    approx_eq(av_i, lambda * v[(i, col)], 1e-9),
                    "A*v[{}] = {}, λ*v[{}] = {}",
                    i,
                    av_i,
                    i,
                    lambda * v[(i, col)]
                );
            }
        }

        // Verify reconstruction: A = V * D * V^T
        let reconstructed = evd.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }
}
