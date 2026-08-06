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
    /// Relative residual norm `||A x - lambda x||_2 / ||x||_2` for each returned
    /// Ritz pair `(lambda, x)`, computed directly from a sparse matrix-vector
    /// product. Same length/order as `eigenvalues_real`.
    pub residual_norms: Vec<T>,
    /// Per-eigenpair convergence status: `true` iff the residual norm is at or
    /// below the configured tolerance. Same length/order as `eigenvalues_real`.
    pub converged_flags: Vec<bool>,
    /// Number of Arnoldi iterations performed (dimension of the Krylov subspace built).
    pub iterations: usize,
    /// Whether all requested eigenpairs actually converged to the configured
    /// tolerance (i.e. every returned residual norm meets it and at least the
    /// requested count converged). This is a genuine numerical-convergence flag,
    /// not merely an indication that the Krylov subspace reached its target size.
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

        // Compute eigenvalues (Ritz values) of the m x m Hessenberg matrix.
        let (eigenvalues_real, eigenvalues_imag) = self.solve_hessenberg(&h, actual_dim)?;

        // Select eigenvalues based on configuration.
        let (selected_eigenvalues_real, selected_eigenvalues_imag) =
            self.select_eigenvalues_complex(&eigenvalues_real, &eigenvalues_imag, k);

        // Genuine per-eigenpair residual check. For each returned Ritz value we
        // recover the corresponding eigenvector `y` of the m x m Hessenberg
        // matrix (real or complex) via regularized inverse iteration, lift it
        // to the full-space Ritz vector `x = Q_m y`, and directly evaluate the
        // relative residual `||A x - lambda x||_2 / ||x||_2`. This is the true
        // residual of the eigenpair and therefore stays honest even when the
        // Hessenberg QR delivers only an approximate Ritz value. A pair is
        // reported converged only when its residual meets the configured
        // tolerance.
        let tolerance = self.config.tolerance.clone();
        let mut residual_norms = Vec::with_capacity(selected_eigenvalues_real.len());
        let mut converged_flags = Vec::with_capacity(selected_eigenvalues_real.len());
        let mut converged_count = 0usize;

        for (lambda_re, lambda_im) in selected_eigenvalues_real
            .iter()
            .zip(selected_eigenvalues_imag.iter())
        {
            let residual = if actual_dim == 0 {
                T::zero()
            } else {
                Self::ritz_pair_residual(
                    a,
                    &arnoldi_vectors,
                    &h,
                    actual_dim,
                    lambda_re.clone(),
                    lambda_im.clone(),
                )
            };
            let is_converged = residual <= tolerance;
            if is_converged {
                converged_count += 1;
            }
            residual_norms.push(residual);
            converged_flags.push(is_converged);
        }

        // All requested eigenpairs must actually meet the tolerance for the
        // aggregate flag to be true.
        let converged = converged_count >= k;

        Ok(ArnoldiResult {
            eigenvalues_real: selected_eigenvalues_real,
            eigenvalues_imag: selected_eigenvalues_imag,
            residual_norms,
            converged_flags,
            iterations: actual_dim,
            converged,
        })
    }

    /// Compute the true relative residual `||A x - lambda x||_2 / ||x||_2` of a
    /// single Ritz pair.
    ///
    /// The eigenvector `y` of the `dim x dim` upper-Hessenberg block for the Ritz
    /// value `lambda = lambda_re + i * lambda_im` is recovered by regularized
    /// inverse iteration (handling both real values and complex-conjugate pairs).
    /// It is then lifted to the full-space Ritz vector `x = Q_m y` (with real and
    /// imaginary parts) and the residual of `A x - lambda x` is formed directly
    /// with a sparse matrix-vector product, so the estimate reflects the actual
    /// quality of the eigenpair rather than assuming `lambda` is an exact
    /// eigenvalue of the Hessenberg matrix.
    fn ritz_pair_residual(
        a: &CsrMatrix<T>,
        arnoldi_vectors: &[Vec<T>],
        h: &[Vec<T>],
        dim: usize,
        lambda_re: T,
        lambda_im: T,
    ) -> T {
        let n = a.nrows();
        let complex_pair = Scalar::abs(lambda_im.clone()) > <T as Scalar>::epsilon();

        // Eigenvector of the projected Hessenberg matrix (unit complex norm).
        let (y_re, y_im) = Self::ritz_eigenvector(h, dim, lambda_re.clone(), lambda_im.clone());

        // Lift to the original space: x = Q_m y (real and imaginary parts).
        let mut x_re = vec![T::zero(); n];
        let mut x_im = vec![T::zero(); n];
        for j in 0..dim {
            let vj = &arnoldi_vectors[j];
            let yr = y_re[j].clone();
            let yi = y_im[j].clone();
            for idx in 0..n {
                x_re[idx] = x_re[idx].clone() + yr.clone() * vj[idx].clone();
                x_im[idx] = x_im[idx].clone() + yi.clone() * vj[idx].clone();
            }
        }

        let alpha = Real::sqrt(dot(&x_re, &x_re) + dot(&x_im, &x_im));

        // A x (imaginary part only needed for a genuine complex pair).
        let mut ax_re = vec![T::zero(); n];
        spmv(T::one(), a, &x_re, T::zero(), &mut ax_re);
        let mut ax_im = vec![T::zero(); n];
        if complex_pair {
            spmv(T::one(), a, &x_im, T::zero(), &mut ax_im);
        }

        // Residual r = A x - lambda x, split into real and imaginary components:
        //   Re(r) = A x_re - lambda_re x_re + lambda_im x_im
        //   Im(r) = A x_im - lambda_re x_im - lambda_im x_re
        let mut res_sq = T::zero();
        for idx in 0..n {
            let rr = ax_re[idx].clone() - lambda_re.clone() * x_re[idx].clone()
                + lambda_im.clone() * x_im[idx].clone();
            res_sq = res_sq + rr.clone() * rr;
            if complex_pair {
                let ri = ax_im[idx].clone()
                    - lambda_re.clone() * x_im[idx].clone()
                    - lambda_im.clone() * x_re[idx].clone();
                res_sq = res_sq + ri.clone() * ri;
            }
        }
        let res = Real::sqrt(res_sq);

        if alpha > <T as Scalar>::epsilon() {
            res / alpha
        } else {
            res
        }
    }

    /// Recover the (unit-norm) eigenvector of the `n x n` upper-Hessenberg block
    /// for the Ritz value `lambda_re + i * lambda_im` via regularized inverse
    /// iteration. Returns the real and imaginary parts `(y_re, y_im)`; `y_im` is
    /// all zeros for a real Ritz value.
    fn ritz_eigenvector(h: &[Vec<T>], n: usize, lambda_re: T, lambda_im: T) -> (Vec<T>, Vec<T>) {
        if Scalar::abs(lambda_im.clone()) <= <T as Scalar>::epsilon() {
            // Real case: inverse iteration on B = H - lambda I.
            let mut b: Vec<Vec<T>> = (0..n).map(|i| h[i][..n].to_vec()).collect();
            for (i, row) in b.iter_mut().enumerate().take(n) {
                row[i] = row[i].clone() - lambda_re.clone();
            }
            let piv = Self::lu_factor(&mut b, n);

            let mut y = vec![T::one(); n];
            Self::normalize_in_place(&mut y);
            for _ in 0..3 {
                y = Self::lu_solve(&b, &piv, &y, n);
                Self::normalize_in_place(&mut y);
            }
            (y, vec![T::zero(); n])
        } else {
            // Complex case: the system (H - lambda I) y = 0 with lambda = a + i b
            // and y = u + i v is equivalent to the 2n x 2n real system
            // [[H-aI, bI], [-bI, H-aI]] [u; v] = 0. The real 2-norm of [u; v]
            // equals the complex 2-norm of y, so a unit-normalized real solution
            // yields unit-norm (u, v) directly.
            let two_n = 2 * n;
            let mut mmat: Vec<Vec<T>> = vec![vec![T::zero(); two_n]; two_n];
            for i in 0..n {
                for j in 0..n {
                    let hij = h[i][j].clone();
                    mmat[i][j] = hij.clone();
                    mmat[n + i][n + j] = hij;
                }
                mmat[i][i] = mmat[i][i].clone() - lambda_re.clone();
                mmat[n + i][n + i] = mmat[n + i][n + i].clone() - lambda_re.clone();
                mmat[i][n + i] = lambda_im.clone();
                mmat[n + i][i] = T::zero() - lambda_im.clone();
            }
            let piv = Self::lu_factor(&mut mmat, two_n);

            let mut w = vec![T::one(); two_n];
            Self::normalize_in_place(&mut w);
            for _ in 0..3 {
                w = Self::lu_solve(&mmat, &piv, &w, two_n);
                Self::normalize_in_place(&mut w);
            }
            let y_re = w[..n].to_vec();
            let y_im = w[n..].to_vec();
            (y_re, y_im)
        }
    }

    /// Normalize a vector to unit 2-norm in place (no-op for a (near-)zero vector).
    fn normalize_in_place(v: &mut [T]) {
        let nrm = norm(v);
        if nrm > <T as Scalar>::epsilon() {
            for vi in v.iter_mut() {
                *vi = vi.clone() / nrm.clone();
            }
        }
    }

    /// In-place LU factorization with partial pivoting and tiny-pivot regularization.
    ///
    /// Returns the row permutation. Pivots whose magnitude drops to (or below)
    /// `eps * ||A||_max` are bumped up to that floor so that the factorization of
    /// the deliberately (near-)singular matrix used for inverse iteration stays
    /// well defined instead of dividing by zero.
    fn lu_factor(a: &mut [Vec<T>], n: usize) -> Vec<usize> {
        let mut piv: Vec<usize> = (0..n).collect();

        // Largest-magnitude entry, used to scale the pivot regularization floor.
        let mut scale = T::zero();
        for row in a.iter().take(n) {
            for val in row.iter().take(n) {
                let av = Scalar::abs(val.clone());
                if av > scale {
                    scale = av;
                }
            }
        }
        if scale <= T::zero() {
            scale = T::one();
        }
        let min_pivot = scale * <T as Scalar>::epsilon();

        for k in 0..n {
            // Partial pivoting: pick the largest-magnitude entry in column k.
            let mut pivot_row = k;
            let mut max_val = Scalar::abs(a[k][k].clone());
            for i in (k + 1)..n {
                let av = Scalar::abs(a[i][k].clone());
                if av > max_val {
                    max_val = av;
                    pivot_row = i;
                }
            }
            if pivot_row != k {
                a.swap(pivot_row, k);
                piv.swap(pivot_row, k);
            }

            // Regularize a vanishing pivot to keep the factorization finite.
            if Scalar::abs(a[k][k].clone()) <= min_pivot {
                let sign = if a[k][k] < T::zero() {
                    T::zero() - T::one()
                } else {
                    T::one()
                };
                a[k][k] = sign * min_pivot.clone();
            }

            let pivot = a[k][k].clone();
            for i in (k + 1)..n {
                let factor = a[i][k].clone() / pivot.clone();
                a[i][k] = factor.clone();
                for j in (k + 1)..n {
                    a[i][j] = a[i][j].clone() - factor.clone() * a[k][j].clone();
                }
            }
        }

        piv
    }

    /// Solve `A x = b` from the LU factors produced by [`Self::lu_factor`].
    fn lu_solve(lu: &[Vec<T>], piv: &[usize], b: &[T], n: usize) -> Vec<T> {
        // Apply the row permutation to the right-hand side.
        let mut x: Vec<T> = piv.iter().take(n).map(|&p| b[p].clone()).collect();

        // Forward substitution with the unit-lower-triangular factor L.
        for i in 0..n {
            let mut sum = x[i].clone();
            for j in 0..i {
                sum = sum - lu[i][j].clone() * x[j].clone();
            }
            x[i] = sum;
        }

        // Back substitution with the upper-triangular factor U.
        for i in (0..n).rev() {
            let mut sum = x[i].clone();
            for j in (i + 1)..n {
                sum = sum - lu[i][j].clone() * x[j].clone();
            }
            x[i] = sum / lu[i][i].clone();
        }

        x
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
