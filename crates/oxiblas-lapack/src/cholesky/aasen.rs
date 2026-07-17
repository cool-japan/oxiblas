//! Aasen's Method for Symmetric Indefinite Matrices.
//!
//! Computes P * A * P^T = L * T * L^T where:
//! - P is a permutation matrix
//! - L is unit lower triangular whose *first column is e_0* (the identity's
//!   first column), so only columns 1..n carry information
//! - T is symmetric tridiagonal
//!
//! This is an alternative to Bunch-Kaufman for symmetric indefinite matrices.
//! While Bunch-Kaufman produces a block diagonal D (with 1×1 and 2×2 blocks),
//! Aasen's method produces a genuinely tridiagonal T.
//!
//! # Algorithm
//!
//! This is the real Aasen (1971) recurrence, matching the LAPACK reference
//! `DSYTRF_AA`. It is *not* an eigenvalue-style reduction: instead it factors
//! the (permuted) matrix column by column.
//!
//! The key object is the upper-Hessenberg matrix `H = T * L^T`. Because
//! `A = L * H`, isolating the last nonzero of column `j` of `H`
//! (`H[j+1][j] = e_j`, the sub-diagonal of `T`) gives the residual
//!
//! ```text
//!   w = A[:, j] - sum_{k=0}^{j} L[:, k] * H[k][j] = e_j * L[:, j+1].
//! ```
//!
//! So at step `j` we
//!   1. form the already-known part `H[0..j][j]` from `L` and the tridiagonal
//!      entries computed so far,
//!   2. set `H[j][j] = w'[j]` (a *consistency* condition: row `j` of `w` must
//!      vanish) and recover the diagonal `T[j][j] = H[j][j] - e_{j-1} L[j][j-1]`,
//!   3. form the residual `w[i] = w'[i] - L[i][j] H[j][j]` for `i > j`, which
//!      equals `e_j * L[i][j+1]`,
//!   4. pivot the largest `|w[i]|` into position `j+1` (this is what bounds the
//!      multipliers `|L[i][j+1]| <= 1`, giving Aasen its stability), record the
//!      swap, set `e_j = w[j+1]`, and read off `L[i][j+1] = w[i] / e_j`.
//!
//! The pivot swaps a trailing row/column of the working matrix *and* the
//! already-computed rows of `L`, keeping the identity `P A P^T = L T L^T` exact.
//!
//! # References
//!
//! - Aasen, J.O. (1971). "On the reduction of a symmetric matrix to tridiagonal
//!   form". BIT Numerical Mathematics, 11(3), 233-242.
//! - Golub & Van Loan, "Matrix Computations", 4th ed., §4.4 (Aasen's method).
//! - LAPACK reference `dsytrf_aa.f`, `dlasyf_aa.f`, `dgtsv.f`.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::cholesky::Aasen;
//! use oxiblas_matrix::Mat;
//!
//! // Symmetric matrix
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 2.0, 1.0],
//!     &[2.0, 5.0, 2.0],
//!     &[1.0, 2.0, 6.0],
//! ]);
//!
//! let aasen = Aasen::compute(a.as_ref()).expect("factorization");
//!
//! // Solve Ax = b
//! let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
//! let x = aasen.solve(b.as_ref()).expect("solve");
//! ```

use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for Aasen's method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AasenError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Matrix is singular.
    Singular {
        /// Index where singularity was detected.
        index: usize,
    },
    /// Dimension mismatch in solve.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Error in tridiagonal solve.
    TridiagonalSolveError,
}

impl core::fmt::Display for AasenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}×{ncols}")
            }
            Self::Singular { index } => {
                write!(f, "Matrix is singular at index {index}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
            Self::TridiagonalSolveError => {
                write!(f, "Error solving tridiagonal system")
            }
        }
    }
}

impl std::error::Error for AasenError {}

/// Aasen's factorization of a symmetric indefinite matrix.
///
/// For a symmetric matrix A, computes the factorization:
/// P * A * P^T = L * T * L^T
///
/// where P is a permutation, L is unit lower triangular (with first column
/// `e_0`), and T is symmetric tridiagonal.
#[derive(Clone, Debug)]
pub struct Aasen<T: Scalar> {
    /// Unit lower triangular factor L (first column is `e_0`).
    l: Mat<T>,
    /// Tridiagonal matrix T: diagonal elements (length n).
    t_diag: Vec<T>,
    /// Tridiagonal matrix T: sub-diagonal elements (length n-1).
    t_subdiag: Vec<T>,
    /// Pivot indices: `piv[k]` is the row swapped with row `k` at step `k-1`
    /// (applied in increasing `k`). `piv[0] == 0` always.
    piv: Vec<usize>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable + FromPrimitive> Aasen<T> {
    /// Computes Aasen's factorization of a symmetric matrix.
    ///
    /// Uses Aasen's algorithm to compute `P * A * P^T = L * T * L^T` with
    /// partial pivoting for numerical stability. Only the lower triangle of
    /// `a` is read; the matrix is assumed symmetric.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (only lower triangle is used)
    ///
    /// # Returns
    ///
    /// Aasen's factorization or error.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, AasenError> {
        let n = a.nrows();

        if n == 0 {
            return Err(AasenError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(AasenError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        // 1×1: T = [a00], L = [1].
        if n == 1 {
            return Ok(Self {
                l: Mat::eye(1),
                t_diag: vec![a[(0, 0)]],
                t_subdiag: vec![],
                piv: vec![0],
                n: 1,
            });
        }

        // Working copy of A, fully symmetrized from the lower triangle. The
        // pivot at each step applies a symmetric row/column swap here, so at
        // the start of step j this holds the leading part of `P_{<j} A P_{<j}^T`.
        let mut a_work = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = a[(i, j)];
                a_work[(i, j)] = v;
                a_work[(j, i)] = v;
            }
        }

        // L starts as the identity (its first column stays e_0 throughout).
        let mut l = Mat::<T>::eye(n);

        // Tridiagonal T.
        let mut t_diag = vec![T::zero(); n];
        let mut t_subdiag = vec![T::zero(); n - 1];

        // Permutation, identity initially. `piv[j+1]` is filled at step j.
        let mut piv: Vec<usize> = (0..n).collect();

        // Scratch vectors reused every step.
        // hcol[k]   = H[k][j]                 (the known part of column j of H)
        // wprime[i] = A[i][j] - sum_k L[i][k] H[k][j]
        // wvec[i]   = wprime[i] - L[i][j] H[j][j]  (= e_j * L[i][j+1])
        let mut hcol = vec![T::zero(); n];
        let mut wprime = vec![T::zero(); n];
        let mut wvec = vec![T::zero(); n];

        for j in 0..n {
            // Step 1: known part of column j of H = T L^T.
            //   H[k][j] = e[k-1] L[j][k-1] + d[k] L[j][k] + e[k] L[j][k+1]
            // for k = 0..j (all quantities on the right are already computed;
            // L[j][j] = 1 supplies the e[k] L[j][k+1] term at k = j-1).
            for k in 0..j {
                let mut hkj = t_diag[k] * l[(j, k)] + t_subdiag[k] * l[(j, k + 1)];
                if k >= 1 {
                    hkj += t_subdiag[k - 1] * l[(j, k - 1)];
                }
                hcol[k] = hkj;
            }

            // Step 2: residual using the known part of H (rows i >= j only).
            for i in j..n {
                let mut s = a_work[(i, j)];
                for k in 0..j {
                    s -= l[(i, k)] * hcol[k];
                }
                wprime[i] = s;
            }

            // Step 3: the consistency condition w[j] = 0 forces H[j][j] = w'[j];
            // peel off the (known) e_{j-1} L[j][j-1] contribution to get T[j][j].
            let h_jj = wprime[j];
            let mut d_jj = h_jj;
            if j >= 1 {
                d_jj -= t_subdiag[j - 1] * l[(j, j - 1)];
            }
            t_diag[j] = d_jj;

            if j + 1 >= n {
                continue;
            }

            // Step 4: residual w[i] = e_j * L[i][j+1] for i > j.
            for i in (j + 1)..n {
                wvec[i] = wprime[i] - l[(i, j)] * h_jj;
            }

            // Partial pivot: bring the largest |w[i]| to position j+1. This is
            // what keeps every multiplier |L[i][j+1]| <= 1.
            let mut p = j + 1;
            let mut pmax = Scalar::abs(wvec[j + 1]);
            for i in (j + 2)..n {
                let cand = Scalar::abs(wvec[i]);
                if cand > pmax {
                    pmax = cand;
                    p = i;
                }
            }

            if p != j + 1 {
                Self::swap_sym(&mut a_work, j + 1, p, n);
                // Swap the already-computed rows of L (columns 0..=j).
                for k in 0..=j {
                    let tmp = l[(j + 1, k)];
                    l[(j + 1, k)] = l[(p, k)];
                    l[(p, k)] = tmp;
                }
                wvec.swap(j + 1, p);
            }
            piv[j + 1] = p;

            let e_j = wvec[j + 1];
            t_subdiag[j] = e_j;

            // Column j+1 of L. If e_j == 0 the trailing block has decoupled
            // (all residuals are zero after pivoting), so the column is zero.
            if e_j != T::zero() {
                for i in (j + 2)..n {
                    l[(i, j + 1)] = wvec[i] / e_j;
                }
            }
        }

        Ok(Self {
            l,
            t_diag,
            t_subdiag,
            piv,
            n,
        })
    }

    /// Symmetrically swap rows and columns `i` and `j` of a full `n×n` matrix.
    fn swap_sym(a: &mut Mat<T>, i: usize, j: usize, n: usize) {
        if i == j {
            return;
        }
        for k in 0..n {
            let tmp = a[(i, k)];
            a[(i, k)] = a[(j, k)];
            a[(j, k)] = tmp;
        }
        for k in 0..n {
            let tmp = a[(k, i)];
            a[(k, i)] = a[(k, j)];
            a[(k, j)] = tmp;
        }
    }

    /// Solves `A x = b` using the factorization.
    ///
    /// Since `P A P^T = L T L^T`, i.e. `A = P^T L T L^T P`, we solve in five
    /// stages: apply `P`, forward-solve the unit lower triangular `L`, solve
    /// the tridiagonal `T`, back-solve `L^T`, then apply `P^T`.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, AasenError> {
        if b.nrows() != self.n {
            return Err(AasenError::DimensionMismatch {
                expected: self.n,
                actual: b.nrows(),
            });
        }

        let nrhs = b.ncols();
        let n = self.n;

        let mut x = Mat::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                x[(i, j)] = b[(i, j)];
            }
        }

        // Step 1: apply permutation P (transpositions in increasing order).
        for k in 0..n {
            let pk = self.piv[k];
            if pk != k {
                for j in 0..nrhs {
                    let tmp = x[(k, j)];
                    x[(k, j)] = x[(pk, j)];
                    x[(pk, j)] = tmp;
                }
            }
        }

        // Step 2: forward substitution, unit lower triangular L y = Pb.
        for k in 0..n {
            for i in (k + 1)..n {
                let lik = self.l[(i, k)];
                if lik != T::zero() {
                    for j in 0..nrhs {
                        let xk = x[(k, j)];
                        x[(i, j)] -= lik * xk;
                    }
                }
            }
        }

        // Step 3: tridiagonal solve T z = y.
        self.solve_tridiagonal(&mut x, nrhs)?;

        // Step 4: back substitution, unit upper triangular L^T w = z.
        for k in (0..n).rev() {
            for i in (k + 1)..n {
                let lik = self.l[(i, k)];
                if lik != T::zero() {
                    for j in 0..nrhs {
                        let xi = x[(i, j)];
                        x[(k, j)] -= lik * xi;
                    }
                }
            }
        }

        // Step 5: apply P^T (transpositions in decreasing order).
        for k in (0..n).rev() {
            let pk = self.piv[k];
            if pk != k {
                for j in 0..nrhs {
                    let tmp = x[(k, j)];
                    x[(k, j)] = x[(pk, j)];
                    x[(pk, j)] = tmp;
                }
            }
        }

        Ok(x)
    }

    /// Solve the symmetric tridiagonal system `T x = b` in place.
    ///
    /// Implements LAPACK's `xGTSV`: Gaussian elimination on the general
    /// tridiagonal system with partial pivoting. Pivoting is essential because
    /// `T` is symmetric *indefinite* — a zero or tiny diagonal pivot is normal
    /// (e.g. `T = [[0, e], [e, 0]]`), and a row interchange there introduces a
    /// second super-diagonal (`du2`) fill-in that back-substitution must carry.
    ///
    /// Exact zeros are compared as in the reference (`.NE.ZERO`) so a genuinely
    /// singular `T` is reported rather than silently perturbed, and any NaN/Inf
    /// propagates through the arithmetic per IEEE-754.
    fn solve_tridiagonal(&self, x: &mut Mat<T>, nrhs: usize) -> Result<(), AasenError> {
        let n = self.n;
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            let d0 = self.t_diag[0];
            if d0 == T::zero() {
                return Err(AasenError::Singular { index: 0 });
            }
            for j in 0..nrhs {
                x[(0, j)] = x[(0, j)] / d0;
            }
            return Ok(());
        }

        // d: diagonal, du: super-diagonal, dl: sub-diagonal (T symmetric so
        // du and dl start equal), du2: second super-diagonal fill from pivoting.
        let mut d = self.t_diag.clone();
        let mut du = self.t_subdiag.clone();
        let dl = self.t_subdiag.clone();
        let mut du2 = vec![T::zero(); n];

        for i in 0..(n - 1) {
            if Scalar::abs(d[i]) >= Scalar::abs(dl[i]) {
                // No interchange required.
                if d[i] == T::zero() {
                    return Err(AasenError::Singular { index: i });
                }
                let fact = dl[i] / d[i];
                d[i + 1] -= fact * du[i];
                for j in 0..nrhs {
                    let xi = x[(i, j)];
                    x[(i + 1, j)] -= fact * xi;
                }
                // du2[i] stays zero (no fill-in on this row).
            } else {
                // Interchange rows i and i+1; |dl[i]| > |d[i]| guarantees the
                // multiplier is bounded by 1 in magnitude.
                let fact = d[i] / dl[i];
                d[i] = dl[i];
                let temp = d[i + 1];
                d[i + 1] = du[i] - fact * temp;
                du[i] = temp;
                if i < n - 2 {
                    du2[i] = du[i + 1];
                    du[i + 1] = -fact * du2[i];
                }
                for j in 0..nrhs {
                    let temp_b = x[(i, j)];
                    let xip = x[(i + 1, j)];
                    x[(i, j)] = xip;
                    x[(i + 1, j)] = temp_b - fact * xip;
                }
            }
        }

        if d[n - 1] == T::zero() {
            return Err(AasenError::Singular { index: n - 1 });
        }

        // Back substitution over the upper triangular factor (bidiagonal plus
        // the du2 fill-in produced by any interchanges).
        for j in 0..nrhs {
            x[(n - 1, j)] = x[(n - 1, j)] / d[n - 1];
            let val = x[(n - 2, j)] - du[n - 2] * x[(n - 1, j)];
            x[(n - 2, j)] = val / d[n - 2];
            for i in (0..(n - 2)).rev() {
                let val = x[(i, j)] - du[i] * x[(i + 1, j)] - du2[i] * x[(i + 2, j)];
                x[(i, j)] = val / d[i];
            }
        }

        Ok(())
    }

    /// Returns the L factor.
    pub fn l_factor(&self) -> &Mat<T> {
        &self.l
    }

    /// Returns the diagonal of T.
    pub fn t_diagonal(&self) -> &[T] {
        &self.t_diag
    }

    /// Returns the sub-diagonal of T.
    pub fn t_subdiagonal(&self) -> &[T] {
        &self.t_subdiag
    }

    /// Returns the pivot indices.
    pub fn pivot(&self) -> &[usize] {
        &self.piv
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Constructs the tridiagonal matrix T explicitly.
    pub fn t_matrix(&self) -> Mat<T> {
        let n = self.n;
        let mut t = Mat::zeros(n, n);

        for i in 0..n {
            t[(i, i)] = self.t_diag[i];
            if i + 1 < n {
                t[(i, i + 1)] = self.t_subdiag[i];
                t[(i + 1, i)] = self.t_subdiag[i];
            }
        }

        t
    }

    /// Computes the inertia of the matrix: (n_positive, n_negative, n_zero)
    /// eigenvalues.
    ///
    /// By Sylvester's law of inertia the congruence `P A P^T = L T L^T` (with
    /// `L` nonsingular) preserves inertia, so `inertia(A) = inertia(T)`. The
    /// inertia of the symmetric tridiagonal `T` is obtained from the signs of
    /// the pivots of a congruence (root-free `L D L^T`) reduction of `T`,
    /// using 1×1 pivots normally and a 2×2 pivot when a diagonal entry is
    /// (near) zero — the latter is required for indefinite blocks such as
    /// `[[0, e], [e, 0]]`, whose inertia is `(1, 1, 0)` even though both
    /// diagonal entries vanish.
    ///
    /// Zero eigenvalues are detected with a tolerance relative to `‖T‖_∞`, so
    /// the `n_zero` count is meaningful only up to that scale.
    pub fn inertia(&self) -> (usize, usize, usize) {
        let n = self.n;
        if n == 0 {
            return (0, 0, 0);
        }

        // Working diagonal (Schur-complement updated during elimination).
        let mut a = self.t_diag.clone();
        let b = &self.t_subdiag; // super/sub-diagonal, length n-1

        // ‖T‖_∞ bound for the zero tolerance.
        let mut anorm = T::zero();
        for i in 0..n {
            let mut s = Scalar::abs(a[i]);
            if i > 0 {
                s += Scalar::abs(b[i - 1]);
            }
            if i + 1 < n {
                s += Scalar::abs(b[i]);
            }
            if s > anorm {
                anorm = s;
            }
        }
        let two = T::one() + T::one();
        let eight = two * two * two;
        let eps = <T as Scalar>::epsilon();
        let tol = eps * eight * anorm; // pivot ~ 0 threshold
        let det_tol = tol * anorm; // 2×2 determinant ~ 0 threshold

        let zero = T::zero();
        let mut n_pos = 0usize;
        let mut n_neg = 0usize;
        let mut n_zero = 0usize;

        let mut i = 0usize;
        while i < n {
            let a_i = a[i];
            let last = i + 1 >= n;
            let b_i = if last { zero } else { b[i] };

            if Scalar::abs(a_i) > tol {
                // 1×1 pivot: sign of the pivot contributes to the inertia.
                if a_i > zero {
                    n_pos += 1;
                } else {
                    n_neg += 1;
                }
                if !last {
                    // Schur complement: a[i+1] -= b_i^2 / a_i.
                    a[i + 1] -= b_i * b_i / a_i;
                }
                i += 1;
            } else if !last && Scalar::abs(b_i) > tol {
                // 2×2 pivot on rows i, i+1 (a_i ~ 0). The block
                // [[a_i, b_i], [b_i, a_{i+1}]] has determinant ~ -b_i^2 < 0.
                let a_ip = a[i + 1];
                let det2 = a_i * a_ip - b_i * b_i;
                let trace = a_i + a_ip;
                if Scalar::abs(det2) <= det_tol {
                    // One zero eigenvalue; the other has the sign of the trace.
                    n_zero += 1;
                    if Scalar::abs(trace) > tol {
                        if trace > zero {
                            n_pos += 1;
                        } else {
                            n_neg += 1;
                        }
                    } else {
                        n_zero += 1;
                    }
                } else if det2 < zero {
                    n_pos += 1;
                    n_neg += 1;
                } else if trace > zero {
                    n_pos += 2;
                } else {
                    n_neg += 2;
                }
                if i + 2 < n {
                    // Schur complement into a[i+2] through the coupling b[i+1].
                    let b_next = b[i + 1];
                    a[i + 2] -= b_next * b_next * a_i / det2;
                }
                i += 2;
            } else {
                // Isolated (near-)zero pivot with no coupling: a zero eigenvalue.
                n_zero += 1;
                i += 1;
            }
        }

        (n_pos, n_neg, n_zero)
    }
}

/// Convenience function to compute Aasen's factorization.
pub fn aasen<T: Field + Real + bytemuck::Zeroable + FromPrimitive>(
    a: MatRef<'_, T>,
) -> Result<Aasen<T>, AasenError> {
    Aasen::compute(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Full verification of a factorization for a symmetric `f64` matrix given
    /// as rows: checks the structural invariants of `L` and `T`, the exact
    /// reconstruction `P A P^T = L T L^T`, and the residual of a solve against
    /// a known right-hand side.
    fn check_factorization(rows: &[&[f64]]) {
        let n = rows.len();
        let a = Mat::from_rows(rows);
        let aasen = Aasen::compute(a.as_ref()).expect("compute should succeed");

        // --- Structural invariants of L ---
        let l = aasen.l_factor();
        for i in 0..n {
            assert!(
                approx_eq(l[(i, i)], 1.0, 1e-12),
                "L must be unit diagonal: L[{i},{i}] = {}",
                l[(i, i)]
            );
        }
        for i in 1..n {
            assert!(
                approx_eq(l[(i, 0)], 0.0, 1e-12),
                "first column of L must be e_0: L[{i},0] = {}",
                l[(i, 0)]
            );
        }
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    approx_eq(l[(i, j)], 0.0, 1e-12),
                    "L must be lower triangular: L[{i},{j}] = {}",
                    l[(i, j)]
                );
            }
            // Aasen stability: every multiplier is bounded by 1 in magnitude.
            for j in 0..i {
                assert!(
                    l[(i, j)].abs() <= 1.0 + 1e-9,
                    "Aasen multiplier bound violated: |L[{i},{j}]| = {}",
                    l[(i, j)].abs()
                );
            }
        }

        // --- T is tridiagonal & symmetric ---
        let t = aasen.t_matrix();
        for i in 0..n {
            for j in 0..n {
                if i != j && (i as isize - j as isize).abs() != 1 {
                    assert!(
                        approx_eq(t[(i, j)], 0.0, 1e-12),
                        "T must be tridiagonal: T[{i},{j}] = {}",
                        t[(i, j)]
                    );
                }
                assert!(
                    approx_eq(t[(i, j)], t[(j, i)], 1e-12),
                    "T must be symmetric at [{i},{j}]"
                );
            }
        }

        // --- Reconstruction P A P^T = L T L^T ---
        // L * T
        let mut lt = vec![vec![0.0f64; n]; n];
        for (i, lt_row) in lt.iter_mut().enumerate() {
            for (j, lt_ij) in lt_row.iter_mut().enumerate() {
                let mut s = 0.0;
                for k in 0..n {
                    s += l[(i, k)] * t[(k, j)];
                }
                *lt_ij = s;
            }
        }
        // (L T) * L^T
        let mut ltlt = vec![vec![0.0f64; n]; n];
        for (i, ltlt_row) in ltlt.iter_mut().enumerate() {
            for (j, ltlt_ij) in ltlt_row.iter_mut().enumerate() {
                let mut s = 0.0;
                for k in 0..n {
                    s += lt[i][k] * l[(j, k)];
                }
                *ltlt_ij = s;
            }
        }
        // P A P^T by replaying the pivot transpositions on a copy of A.
        let piv = aasen.pivot();
        let mut pa = vec![vec![0.0f64; n]; n];
        for (i, pa_row) in pa.iter_mut().enumerate() {
            for (j, pa_ij) in pa_row.iter_mut().enumerate() {
                *pa_ij = a[(i, j)];
            }
        }
        for k in 1..n {
            let p = piv[k];
            if p != k {
                pa.swap(k, p);
                for row in pa.iter_mut() {
                    row.swap(k, p);
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(ltlt[i][j], pa[i][j], 1e-8),
                    "reconstruction mismatch at [{i},{j}]: LTL^T = {}, PAP^T = {}",
                    ltlt[i][j],
                    pa[i][j]
                );
            }
        }

        // --- Solve residual for a known solution ---
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += a[(i, j)] * x_true[j];
            }
            b[(i, 0)] = s;
        }
        let x = aasen.solve(b.as_ref()).expect("solve should succeed");
        for i in 0..n {
            assert!(
                approx_eq(x[(i, 0)], x_true[i], 1e-6),
                "solve mismatch at {i}: x = {}, expected {}",
                x[(i, 0)],
                x_true[i]
            );
        }
    }

    #[test]
    fn test_aasen_2x2() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);

        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        let b = Mat::from_rows(&[&[6.0], &[7.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        let ax0 = a[(0, 0)] * x[(0, 0)] + a[(0, 1)] * x[(1, 0)];
        let ax1 = a[(1, 0)] * x[(0, 0)] + a[(1, 1)] * x[(1, 0)];

        assert!(approx_eq(ax0, b[(0, 0)], 1e-10), "ax0 = {ax0}");
        assert!(approx_eq(ax1, b[(1, 0)], 1e-10), "ax1 = {ax1}");
    }

    #[test]
    fn test_aasen_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 4.0]]);

        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        let b = Mat::from_rows(&[&[2.0], &[6.0], &[12.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        assert!(approx_eq(x[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(2, 0)], 3.0, 1e-10));
    }

    #[test]
    fn test_aasen_identity() {
        let a: Mat<f64> = Mat::eye(4);

        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        for i in 0..4 {
            assert!(approx_eq(x[(i, 0)], b[(i, 0)], 1e-10));
        }
    }

    #[test]
    fn test_aasen_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);

        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        let b = Mat::from_rows(&[&[1.0, 4.0], &[2.0, 5.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        for col in 0..2 {
            for i in 0..2 {
                let mut ax_i = 0.0;
                for j in 0..2 {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                assert!(
                    approx_eq(ax_i, b[(i, col)], 1e-10),
                    "Ax[{i},{col}] = {ax_i}, b = {}",
                    b[(i, col)]
                );
            }
        }
    }

    #[test]
    fn test_aasen_t_matrix() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 2.0], &[1.0, 2.0, 6.0]]);

        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        let t = aasen.t_matrix();

        for i in 0..3 {
            for j in 0..3 {
                if !(i == j || (i as i32 - j as i32).abs() == 1) {
                    assert!(
                        approx_eq(t[(i, j)], 0.0, 1e-10),
                        "T[{i},{j}] = {} should be 0",
                        t[(i, j)]
                    );
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(t[(i, j)], t[(j, i)], 1e-10),
                    "T not symmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_aasen_error_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = Aasen::compute(a.as_ref());
        assert!(matches!(result, Err(AasenError::NotSquare { .. })));
    }

    #[test]
    fn test_aasen_error_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);

        let result = Aasen::compute(a.as_ref());
        assert!(matches!(result, Err(AasenError::EmptyMatrix)));
    }

    #[test]
    fn test_aasen_1x1() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        let b = Mat::from_rows(&[&[10.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        assert!(approx_eq(x[(0, 0)], 2.0, 1e-10));
    }

    // ---- Correctness on general n >= 3 indefinite matrices (the cases the
    //      original stub silently got wrong). Each exercises reconstruction
    //      P A P^T = L T L^T *and* the solve residual. ----

    #[test]
    fn test_aasen_3x3_general() {
        // Hand-verified example: pivots rows 1,2 at the first step.
        check_factorization(&[&[1.0, 2.0, 4.0], &[2.0, 3.0, 5.0], &[4.0, 5.0, 6.0]]);
    }

    #[test]
    fn test_aasen_3x3_zero_diagonal_pivot() {
        // Zero diagonal forces a genuine pivot swap at step 0.
        check_factorization(&[&[0.0, 2.0, 1.0], &[2.0, 0.0, 3.0], &[1.0, 3.0, 0.0]]);
    }

    #[test]
    fn test_aasen_3x3_spd() {
        check_factorization(&[&[4.0, 2.0, 1.0], &[2.0, 5.0, 2.0], &[1.0, 2.0, 6.0]]);
    }

    #[test]
    fn test_aasen_4x4_indefinite() {
        check_factorization(&[
            &[1.0, 3.0, -2.0, 4.0],
            &[3.0, -5.0, 1.0, 2.0],
            &[-2.0, 1.0, 6.0, -1.0],
            &[4.0, 2.0, -1.0, 3.0],
        ]);
    }

    #[test]
    fn test_aasen_4x4_repeated_eigenvalues() {
        // Block diag of [[0,1],[1,0]] twice: eigenvalues {+1,+1,-1,-1}
        // (clustered/repeated) and every diagonal entry is zero, so pivoting
        // is mandatory at several steps.
        check_factorization(&[
            &[0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0, 1.0, 0.0],
        ]);
    }

    /// Build a deterministic symmetric matrix with a sign-alternating,
    /// strictly diagonally dominant diagonal. Dominance guarantees the matrix
    /// is nonsingular (Levy–Desplanques) and the alternating signs guarantee it
    /// is genuinely indefinite, so the solve leg is always well posed while the
    /// off-diagonals still force real pivoting.
    fn dominant_indefinite(n: usize, seed: usize) -> Vec<Vec<f64>> {
        let mut rows = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..i {
                let v = (((i * seed + j * 3 + 1) % 9) as f64) - 4.0;
                rows[i][j] = v;
                rows[j][i] = v;
            }
        }
        for i in 0..n {
            let mut off = 0.0;
            for j in 0..n {
                if j != i {
                    off += rows[i][j].abs();
                }
            }
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            rows[i][i] = sign * (off + 2.0);
        }
        rows
    }

    #[test]
    fn test_aasen_5x5_indefinite() {
        let rows = dominant_indefinite(5, 7);
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        check_factorization(&refs);
    }

    #[test]
    fn test_aasen_8x8_indefinite() {
        let rows = dominant_indefinite(8, 5);
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        check_factorization(&refs);
    }

    #[test]
    fn test_aasen_solve_multiple_rhs_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 4.0], &[2.0, 3.0, 5.0], &[4.0, 5.0, 6.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        // Two right-hand sides at once.
        let b = Mat::from_rows(&[&[7.0, 1.0], &[10.0, 0.0], &[15.0, -2.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        for col in 0..2 {
            for i in 0..3 {
                let mut ax = 0.0;
                for j in 0..3 {
                    ax += a[(i, j)] * x[(j, col)];
                }
                assert!(
                    approx_eq(ax, b[(i, col)], 1e-9),
                    "Ax[{i},{col}] = {ax}, b = {}",
                    b[(i, col)]
                );
            }
        }
    }

    #[test]
    fn test_aasen_f32() {
        let a = Mat::from_rows(&[
            &[2.0f32, 1.0, 0.0],
            &[1.0, -3.0, 2.0],
            &[0.0, 2.0, 1.0],
        ]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");

        let b = Mat::from_rows(&[&[3.0f32], &[0.0], &[3.0]]);
        let x = aasen.solve(b.as_ref()).expect("solve");

        for i in 0..3 {
            let mut ax = 0.0f32;
            for j in 0..3 {
                ax += a[(i, j)] * x[(j, 0)];
            }
            assert!(approx_eq(ax as f64, b[(i, 0)] as f64, 1e-4), "Ax[{i}] = {ax}");
        }
    }

    #[test]
    fn test_aasen_singular_solve_reports_error() {
        // Rank-deficient symmetric matrix (row 2 = row 0 + row 1):
        // [[1,2,3],[2,4,6],[3,6,9]] is singular.
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("factorization still completes");
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let result = aasen.solve(b.as_ref());
        assert!(
            matches!(result, Err(AasenError::Singular { .. })),
            "singular T must be reported, got {result:?}"
        );
    }

    #[test]
    fn test_aasen_nan_propagates() {
        // A NaN in the data must propagate through the solve, not be clamped.
        let a = Mat::from_rows(&[
            &[f64::NAN, 1.0, 0.0],
            &[1.0, 2.0, 1.0],
            &[0.0, 1.0, 3.0],
        ]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        let b = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);
        // Either the solve returns a result containing NaN, or an honest error
        // — but it must never silently return a finite (fabricated) answer.
        if let Ok(x) = aasen.solve(b.as_ref()) {
            let any_nan = (0..3).any(|i| x[(i, 0)].is_nan());
            assert!(any_nan, "NaN in the matrix must propagate to the solution");
        }
    }

    #[test]
    fn test_aasen_dimension_mismatch() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 3.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let result = aasen.solve(b.as_ref());
        assert!(matches!(result, Err(AasenError::DimensionMismatch { .. })));
    }

    // ---- Inertia (Sylvester's law): inertia(A) == inertia(T). ----

    #[test]
    fn test_inertia_positive_definite() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 2.0], &[1.0, 2.0, 6.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        assert_eq!(aasen.inertia(), (3, 0, 0));
    }

    #[test]
    fn test_inertia_indefinite() {
        // A = [[1,2,4],[2,3,5],[4,5,6]] has leading minors 1, -1, 1 => two
        // sign changes => 2 negative, 1 positive eigenvalue.
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 4.0], &[2.0, 3.0, 5.0], &[4.0, 5.0, 6.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        assert_eq!(aasen.inertia(), (1, 2, 0));
    }

    #[test]
    fn test_inertia_zero_diagonal_block() {
        // [[0,1],[1,0]] has eigenvalues +1 and -1.
        let a = Mat::from_rows(&[&[0.0f64, 1.0], &[1.0, 0.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        assert_eq!(aasen.inertia(), (1, 1, 0));
    }

    #[test]
    fn test_inertia_diagonal_signs() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, -1.0]]);
        let aasen = Aasen::compute(a.as_ref()).expect("compute");
        assert_eq!(aasen.inertia(), (2, 1, 0));
    }
}
