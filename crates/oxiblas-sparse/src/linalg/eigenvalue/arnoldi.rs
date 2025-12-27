//! Arnoldi iteration for sparse general (non-symmetric) matrices.
//!
//! This module provides the Arnoldi algorithm for computing eigenvalues of
//! general (non-symmetric) sparse matrices. The Arnoldi method builds an
//! orthonormal basis for the Krylov subspace and reduces A to upper Hessenberg
//! form H = Q^H A Q.
//!
//! # Algorithm Overview
//!
//! The Arnoldi iteration is a projection method that:
//! 1. Constructs an orthonormal basis {v_1, v_2, ..., v_m} for the Krylov subspace
//!    K_m(A, v_1) = span{v_1, Av_1, A^2 v_1, ..., A^(m-1) v_1}
//! 2. Projects the matrix A onto this subspace to obtain an upper Hessenberg matrix H
//! 3. Computes eigenvalues of H (Ritz values) as approximations to eigenvalues of A
//!
//! Unlike the Lanczos method (which is for symmetric matrices), the Arnoldi method
//! can handle non-symmetric matrices and may produce complex eigenvalues.

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::{EigenvalueError, WhichEigenvalues};
use super::lanczos::LanczosConfig;
use super::utils::{dot, givens_rotation, norm};

/// Result of Arnoldi eigenvalue computation.
#[derive(Debug, Clone)]
pub struct ArnoldiResult<T> {
    /// Real parts of computed eigenvalues.
    pub eigenvalues_real: Vec<T>,
    /// Imaginary parts of computed eigenvalues.
    pub eigenvalues_imag: Vec<T>,
    /// Number of Arnoldi iterations performed.
    pub iterations: usize,
    /// Whether the Arnoldi process completed.
    pub converged: bool,
}

/// Arnoldi iteration for sparse general (non-symmetric) matrices.
///
/// Computes eigenvalues of a general square matrix using the Arnoldi algorithm,
/// which builds an orthonormal basis for the Krylov subspace and reduces A
/// to upper Hessenberg form H = Q^H A Q.
pub struct Arnoldi<T> {
    config: LanczosConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> Arnoldi<T> {
    /// Create a new Arnoldi solver with the given configuration.
    pub fn new(config: LanczosConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues of a general matrix using Arnoldi iteration.
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix in CSR format
    /// * `initial_vector` - Optional starting vector
    ///
    /// # Returns
    ///
    /// Computed eigenvalues (may be complex even for real matrices).
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
    ) -> Result<ArnoldiResult<T>, EigenvalueError> {
        let n = a.nrows();

        if a.ncols() != n {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        let k = self.config.num_eigenvalues;
        let m = self.config.krylov_dimension.max(k + 1).min(n);

        if k > n {
            return Err(EigenvalueError::TooManyEigenvalues {
                requested: k,
                max_allowed: n,
            });
        }

        // Initialize starting vector
        let mut v = if let Some(v0) = initial_vector {
            if v0.len() != n {
                return Err(EigenvalueError::DimensionMismatch {
                    expected: n,
                    actual: v0.len(),
                });
            }
            v0.to_vec()
        } else {
            let n_t = T::from_usize(n).unwrap_or_else(T::one);
            let scale = T::one() / Real::sqrt(n_t);
            vec![scale; n]
        };

        // Normalize initial vector
        let v_norm = norm(&v);
        if v_norm <= <T as Scalar>::epsilon() {
            return Err(EigenvalueError::Breakdown {
                iteration: 0,
                description: "Initial vector is zero".to_string(),
            });
        }
        for vi in &mut v {
            *vi = vi.clone() / v_norm.clone();
        }

        // Storage for Arnoldi vectors (Q matrix)
        let mut arnoldi_vectors: Vec<Vec<T>> = Vec::with_capacity(m + 1);
        arnoldi_vectors.push(v.clone());

        // Upper Hessenberg matrix H (m+1 x m)
        let mut h: Vec<Vec<T>> = vec![vec![T::zero(); m]; m + 1];

        // Working vector
        let mut w = vec![T::zero(); n];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        // Arnoldi iteration
        let mut actual_dim = 0;
        for j in 0..m {
            // w = A * v_j
            spmv(T::one(), a, &arnoldi_vectors[j], T::zero(), &mut w);

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot(&arnoldi_vectors[i], &w);
                for idx in 0..n {
                    w[idx] = w[idx].clone() - h[i][j].clone() * arnoldi_vectors[i][idx].clone();
                }
            }

            // Compute norm of w
            let h_next = norm(&w);
            h[j + 1][j] = h_next.clone();

            // Check for breakdown
            if h_next <= tol_breakdown {
                actual_dim = j + 1;
                break;
            }

            // Normalize to get next Arnoldi vector
            let mut v_next = vec![T::zero(); n];
            for idx in 0..n {
                v_next[idx] = w[idx].clone() / h_next.clone();
            }
            arnoldi_vectors.push(v_next);
            actual_dim = j + 1;
        }

        // Compute eigenvalues of the Hessenberg matrix
        let (eigenvalues_real, eigenvalues_imag) = self.solve_hessenberg(&h, actual_dim)?;

        // Select eigenvalues based on configuration
        let (selected_eigenvalues_real, selected_eigenvalues_imag) =
            self.select_eigenvalues_complex(&eigenvalues_real, &eigenvalues_imag, k);

        Ok(ArnoldiResult {
            eigenvalues_real: selected_eigenvalues_real,
            eigenvalues_imag: selected_eigenvalues_imag,
            iterations: actual_dim,
            converged: actual_dim >= m.min(n),
        })
    }

    /// Solve eigenvalue problem for upper Hessenberg matrix using QR iteration.
    fn solve_hessenberg(
        &self,
        h: &[Vec<T>],
        n: usize,
    ) -> Result<(Vec<T>, Vec<T>), EigenvalueError> {
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        // Copy Hessenberg matrix (n x n upper left part)
        let mut a: Vec<Vec<T>> = (0..n).map(|i| h[i][..n].to_vec()).collect();

        let mut eigenvalues_real = vec![T::zero(); n];
        let mut eigenvalues_imag = vec![T::zero(); n];

        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let four = T::from_f64(4.0).unwrap_or_else(T::one);
        let max_iter = 30 * n;

        let mut p = n;

        for _iter in 0..max_iter {
            if p <= 1 {
                if p == 1 {
                    eigenvalues_real[0] = a[0][0].clone();
                }
                break;
            }

            // Check for convergence at bottom of matrix
            let l = p - 1;
            if Scalar::abs(a[l][l - 1].clone())
                <= tol.clone()
                    * (Scalar::abs(a[l - 1][l - 1].clone()) + Scalar::abs(a[l][l].clone()))
            {
                eigenvalues_real[l] = a[l][l].clone();
                p = l;
                continue;
            }

            // Check for 2x2 block at bottom
            // Need l >= 2 to access a[l-1][l-2] and a[l-2][l-2]
            if p >= 2
                && l >= 2
                && Scalar::abs(a[l - 1][l - 2].clone())
                    <= tol.clone()
                        * (Scalar::abs(a[l - 2][l - 2].clone())
                            + Scalar::abs(a[l - 1][l - 1].clone()))
            {
                // Extract 2x2 block
                let a11 = a[l - 1][l - 1].clone();
                let a12 = a[l - 1][l].clone();
                let a21 = a[l][l - 1].clone();
                let a22 = a[l][l].clone();

                // Compute eigenvalues of 2x2 block
                let trace = a11.clone() + a22.clone();
                let det = a11 * a22 - a12 * a21;
                let disc = trace.clone() * trace.clone() / four.clone() - det;

                if disc >= T::zero() {
                    // Real eigenvalues
                    let sqrt_disc = Real::sqrt(disc);
                    eigenvalues_real[l - 1] = trace.clone() / two.clone() + sqrt_disc.clone();
                    eigenvalues_real[l] = trace / two.clone() - sqrt_disc;
                } else {
                    // Complex conjugate pair
                    let sqrt_disc = Real::sqrt(T::zero() - disc);
                    eigenvalues_real[l - 1] = trace.clone() / two.clone();
                    eigenvalues_real[l] = trace / two.clone();
                    eigenvalues_imag[l - 1] = sqrt_disc.clone();
                    eigenvalues_imag[l] = T::zero() - sqrt_disc;
                }

                p = l - 1;
                continue;
            }

            // Wilkinson shift
            let shift = self.compute_wilkinson_shift(&a, p);

            // Apply shifted QR step
            self.qr_step(&mut a, p, shift);
        }

        Ok((eigenvalues_real, eigenvalues_imag))
    }

    /// Compute Wilkinson shift for QR iteration.
    fn compute_wilkinson_shift(&self, a: &[Vec<T>], p: usize) -> T {
        if p < 2 {
            return a[p - 1][p - 1].clone();
        }

        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let four = T::from_f64(4.0).unwrap_or_else(T::one);

        let n = p;
        let a11 = a[n - 2][n - 2].clone();
        let a12 = a[n - 2][n - 1].clone();
        let a21 = a[n - 1][n - 2].clone();
        let a22 = a[n - 1][n - 1].clone();

        let trace = a11.clone() + a22.clone();
        let det = a11 * a22 - a12 * a21;
        let disc = trace.clone() * trace.clone() / four.clone() - det;

        if disc >= T::zero() {
            let sqrt_disc = Real::sqrt(disc);
            let lambda1 = trace.clone() / two.clone() + sqrt_disc.clone();
            let lambda2 = trace / two.clone() - sqrt_disc;

            // Choose eigenvalue closer to a[n-1][n-1]
            let corner = a[n - 1][n - 1].clone();
            if Scalar::abs(lambda1.clone() - corner.clone()) < Scalar::abs(lambda2.clone() - corner)
            {
                lambda1
            } else {
                lambda2
            }
        } else {
            trace / two
        }
    }

    /// Apply QR step with shift to upper Hessenberg matrix.
    fn qr_step(&self, a: &mut [Vec<T>], p: usize, shift: T) {
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        // Apply shift
        for i in 0..p {
            a[i][i] = a[i][i].clone() - shift.clone();
        }

        // QR factorization using Givens rotations
        for i in 0..p - 1 {
            if Scalar::abs(a[i + 1][i].clone()) <= <T as Scalar>::epsilon() {
                continue;
            }

            // Compute Givens rotation
            let (c, s, r) = givens_rotation(a[i][i].clone(), a[i + 1][i].clone());

            // Apply rotation to rows i and i+1
            a[i][i] = r;
            a[i + 1][i] = T::zero();

            for j in i + 1..p {
                let temp = c.clone() * a[i][j].clone() + s.clone() * a[i + 1][j].clone();
                a[i + 1][j] =
                    T::zero() - s.clone() * a[i][j].clone() + c.clone() * a[i + 1][j].clone();
                a[i][j] = temp;
            }

            // Apply rotation to columns (for RQ product)
            let col_end = (i + 3).min(p);
            for j in 0..col_end {
                let temp = c.clone() * a[j][i].clone() + s.clone() * a[j][i + 1].clone();
                a[j][i + 1] =
                    T::zero() - s.clone() * a[j][i].clone() + c.clone() * a[j][i + 1].clone();
                a[j][i] = temp;
            }
        }

        // Remove shift
        for i in 0..p {
            a[i][i] = a[i][i].clone() + shift.clone();
        }

        // Suppress unused warning
        let _ = two;
    }

    /// Select eigenvalues based on magnitude.
    fn select_eigenvalues_complex(&self, real: &[T], imag: &[T], k: usize) -> (Vec<T>, Vec<T>) {
        let n = real.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let k = k.min(n);

        // Compute magnitudes
        let mut indexed: Vec<(usize, T)> = real
            .iter()
            .zip(imag.iter())
            .enumerate()
            .map(|(i, (r, im))| {
                let mag = Real::sqrt(r.clone() * r.clone() + im.clone() * im.clone());
                (i, mag)
            })
            .collect();

        // Sort by magnitude (largest first for LargestMagnitude)
        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // For algebraic, sort by real part
                indexed = real
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (i, r.clone()))
                    .collect();
                match self.config.which {
                    WhichEigenvalues::LargestAlgebraic => {
                        indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    _ => {
                        indexed.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }
        }

        let selected_real: Vec<T> = indexed
            .iter()
            .take(k)
            .map(|(i, _)| real[*i].clone())
            .collect();
        let selected_imag: Vec<T> = indexed
            .iter()
            .take(k)
            .map(|(i, _)| imag[*i].clone())
            .collect();

        (selected_real, selected_imag)
    }
}
