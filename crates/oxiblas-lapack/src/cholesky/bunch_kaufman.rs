//! Bunch-Kaufman Factorization for Symmetric Indefinite Matrices.
//!
//! Computes A = P * L * D * L^T * P^T (or A = P * U * D * U^T * P^T)
//! where:
//! - P is a permutation matrix
//! - L (or U) is unit lower (or upper) triangular
//! - D is block diagonal with 1×1 and 2×2 blocks
//!
//! This is LAPACK's DSYTRF/DSYTRS algorithm.
//!
//! # Algorithm
//!
//! The Bunch-Kaufman algorithm uses partial pivoting to maintain numerical
//! stability for symmetric indefinite matrices. It chooses between:
//! - 1×1 pivots when the diagonal element is large enough
//! - 2×2 pivots when off-diagonal elements dominate
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::cholesky::BunchKaufman;
//! use oxiblas_matrix::Mat;
//!
//! // Symmetric indefinite matrix
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 0.0],
//!     &[2.0, 0.0, 3.0],  // Note: a[1,1] = 0, so standard LDLT would fail
//!     &[0.0, 3.0, 4.0],
//! ]);
//!
//! let bk = BunchKaufman::compute(a.as_ref()).unwrap();
//!
//! // Solve Ax = b
//! let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
//! let x = bk.solve(b.as_ref()).unwrap();
//! ```

use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for Bunch-Kaufman factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BunchKaufmanError {
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
}

impl core::fmt::Display for BunchKaufmanError {
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
        }
    }
}

impl std::error::Error for BunchKaufmanError {}

/// Storage mode for symmetric matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uplo {
    /// Upper triangular storage.
    Upper,
    /// Lower triangular storage.
    Lower,
}

/// Bunch-Kaufman factorization of a symmetric indefinite matrix.
///
/// For a symmetric matrix A, computes the factorization:
/// - Lower: A = P * L * D * L^T * P^T
/// - Upper: A = P * U * D * U^T * P^T
///
/// where P is a permutation, L/U is unit triangular, and D is block diagonal
/// with 1×1 and 2×2 blocks.
#[derive(Clone, Debug)]
pub struct BunchKaufman<T: Scalar> {
    /// Factored matrix (L or U and D stored compactly).
    factors: Mat<T>,
    /// Pivot indices. ipiv[k] > 0 means 1×1 pivot, ipiv[k] < 0 means 2×2 pivot.
    /// The actual pivot row is |ipiv[k]| - 1 (1-indexed like LAPACK).
    ipiv: Vec<i32>,
    /// Storage mode.
    uplo: Uplo,
    /// Matrix dimension.
    n: usize,
}

/// Bunch-Kaufman growth factor bound.
/// α = (1 + sqrt(17)) / 8 ≈ 0.6404
fn alpha<T: Field + Real>() -> T {
    let one = T::one();
    let seventeen = T::from_f64(17.0).unwrap_or(one);
    let eight = T::from_f64(8.0).unwrap_or(one);
    (one + Real::sqrt(seventeen)) / eight
}

impl<T: Field + Real + bytemuck::Zeroable + FromPrimitive> BunchKaufman<T> {
    /// Computes the Bunch-Kaufman factorization of a symmetric matrix.
    ///
    /// Uses lower triangular storage by default.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (only lower triangle is used)
    ///
    /// # Returns
    ///
    /// Bunch-Kaufman factorization or error.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, BunchKaufmanError> {
        Self::compute_with_uplo(a, Uplo::Lower)
    }

    /// Computes the Bunch-Kaufman factorization with specified storage mode.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix
    /// * `uplo` - Whether to use upper or lower triangular storage
    ///
    /// # Returns
    ///
    /// Bunch-Kaufman factorization or error.
    pub fn compute_with_uplo(a: MatRef<'_, T>, uplo: Uplo) -> Result<Self, BunchKaufmanError> {
        let n = a.nrows();

        if n == 0 {
            return Err(BunchKaufmanError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(BunchKaufmanError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        // Copy matrix to working storage
        let mut factors = Mat::zeros(n, n);
        match uplo {
            Uplo::Lower => {
                for i in 0..n {
                    for j in 0..=i {
                        factors[(i, j)] = a[(i, j)];
                    }
                }
            }
            Uplo::Upper => {
                for i in 0..n {
                    for j in i..n {
                        factors[(i, j)] = a[(i, j)];
                    }
                }
            }
        }

        let mut ipiv = vec![0i32; n];

        // Perform factorization
        match uplo {
            Uplo::Lower => Self::factor_lower(&mut factors, &mut ipiv, n)?,
            Uplo::Upper => Self::factor_upper(&mut factors, &mut ipiv, n)?,
        }

        Ok(Self {
            factors,
            ipiv,
            uplo,
            n,
        })
    }

    /// Factor using lower triangular storage (LAPACK DSYTF2 lower).
    fn factor_lower(a: &mut Mat<T>, ipiv: &mut [i32], n: usize) -> Result<(), BunchKaufmanError> {
        let alpha_val = alpha::<T>();
        let mut k = 0;

        while k < n {
            let kstep;

            // Determine pivot type
            let absakk = Scalar::abs(a[(k, k)]);

            // Find maximum off-diagonal in column k (below diagonal)
            let mut imax = k;
            let mut colmax = T::zero();
            if k + 1 < n {
                for i in (k + 1)..n {
                    let absval = Scalar::abs(a[(i, k)]);
                    if absval > colmax {
                        colmax = absval;
                        imax = i;
                    }
                }
            }

            if colmax == T::zero() && absakk == T::zero() {
                // Column is zero - singular matrix
                ipiv[k] = (k + 1) as i32;
                return Err(BunchKaufmanError::Singular { index: k });
            }

            if absakk >= alpha_val * colmax {
                // Use 1×1 pivot
                kstep = 1;
                ipiv[k] = (k + 1) as i32;
            } else {
                // Find maximum in row imax (to the left of diagonal)
                let mut rowmax = T::zero();
                for j in k..imax {
                    let absval = Scalar::abs(a[(imax, j)]);
                    if absval > rowmax {
                        rowmax = absval;
                    }
                }
                // Also check below imax
                if imax + 1 < n {
                    for i in (imax + 1)..n {
                        let absval = Scalar::abs(a[(i, imax)]);
                        if absval > rowmax {
                            rowmax = absval;
                        }
                    }
                }

                let absaimax = Scalar::abs(a[(imax, imax)]);

                if absakk >= alpha_val * colmax * (colmax / rowmax) {
                    // Use 1×1 pivot
                    kstep = 1;
                    ipiv[k] = (k + 1) as i32;
                } else if absaimax >= alpha_val * rowmax {
                    // Use 1×1 pivot at imax
                    kstep = 1;
                    // Swap rows/cols k and imax
                    Self::swap_rows_cols_lower(a, k, imax, n);
                    ipiv[k] = (imax + 1) as i32;
                } else {
                    // Use 2×2 pivot
                    kstep = 2;
                    if imax != k + 1 {
                        // Swap rows/cols k+1 and imax
                        Self::swap_rows_cols_lower(a, k + 1, imax, n);
                    }
                    // Negative pivot indicates 2×2 block
                    ipiv[k] = -((imax + 1) as i32);
                    ipiv[k + 1] = -((imax + 1) as i32);
                }
            }

            // Perform the factorization step
            if kstep == 1 {
                // 1×1 pivot
                let akk = a[(k, k)];
                if akk == T::zero() {
                    return Err(BunchKaufmanError::Singular { index: k });
                }
                let akk_inv = T::one() / akk;

                // Update column k below diagonal
                for i in (k + 1)..n {
                    a[(i, k)] = a[(i, k)] * akk_inv;
                }

                // Update remaining submatrix
                for j in (k + 1)..n {
                    let ajk = a[(j, k)];
                    for i in j..n {
                        a[(i, j)] = a[(i, j)] - ajk * a[(i, k)] * akk;
                    }
                }
            } else {
                // 2×2 pivot
                let akk = a[(k, k)];
                let akp1k = a[(k + 1, k)];
                let akp1kp1 = a[(k + 1, k + 1)];

                // D = [akk, akp1k; akp1k, akp1kp1]
                // det(D) = akk * akp1kp1 - akp1k^2
                let det = akk * akp1kp1 - akp1k * akp1k;
                if det == T::zero() {
                    return Err(BunchKaufmanError::Singular { index: k });
                }
                let det_inv = T::one() / det;

                // D^{-1} = 1/det * [akp1kp1, -akp1k; -akp1k, akk]
                let d11 = akp1kp1 * det_inv;
                let d22 = akk * det_inv;
                let d21 = -akp1k * det_inv;

                // Compute L21 = A21 * D^{-1}
                // where A21 is the submatrix below the 2×2 pivot
                for i in (k + 2)..n {
                    let wk = a[(i, k)];
                    let wkp1 = a[(i, k + 1)];
                    a[(i, k)] = wk * d11 + wkp1 * d21;
                    a[(i, k + 1)] = wk * d21 + wkp1 * d22;
                }

                // Update remaining submatrix: A22 = A22 - L21 * D * L21^T
                for j in (k + 2)..n {
                    let ljk = a[(j, k)];
                    let ljkp1 = a[(j, k + 1)];
                    // D * L21^T column j = [akk*ljk + akp1k*ljkp1, akp1k*ljk + akp1kp1*ljkp1]
                    let djk = akk * ljk + akp1k * ljkp1;
                    let djkp1 = akp1k * ljk + akp1kp1 * ljkp1;
                    for i in j..n {
                        let lik = a[(i, k)];
                        let likp1 = a[(i, k + 1)];
                        a[(i, j)] = a[(i, j)] - lik * djk - likp1 * djkp1;
                    }
                }

                // Store D^{-1} in the diagonal block (for solve)
                // Actually, store the original D values - we'll invert during solve
                // (LAPACK stores the 2×2 block as-is)
            }

            k += kstep;
        }

        Ok(())
    }

    /// Factor using upper triangular storage (LAPACK DSYTF2 upper).
    fn factor_upper(a: &mut Mat<T>, ipiv: &mut [i32], n: usize) -> Result<(), BunchKaufmanError> {
        let alpha_val = alpha::<T>();
        let mut k = n;

        while k > 0 {
            k -= 1;
            let kstep;

            let absakk = Scalar::abs(a[(k, k)]);

            // Find maximum off-diagonal in column k (above diagonal)
            let mut imax = 0;
            let mut colmax = T::zero();
            if k > 0 {
                for i in 0..k {
                    let absval = Scalar::abs(a[(i, k)]);
                    if absval > colmax {
                        colmax = absval;
                        imax = i;
                    }
                }
            }

            if colmax == T::zero() && absakk == T::zero() {
                ipiv[k] = (k + 1) as i32;
                return Err(BunchKaufmanError::Singular { index: k });
            }

            if absakk >= alpha_val * colmax {
                kstep = 1;
                ipiv[k] = (k + 1) as i32;
            } else {
                // Find maximum in row imax
                let mut rowmax = T::zero();
                for j in (imax + 1)..=k {
                    let absval = Scalar::abs(a[(imax, j)]);
                    if absval > rowmax {
                        rowmax = absval;
                    }
                }
                if imax > 0 {
                    for i in 0..imax {
                        let absval = Scalar::abs(a[(i, imax)]);
                        if absval > rowmax {
                            rowmax = absval;
                        }
                    }
                }

                let absaimax = Scalar::abs(a[(imax, imax)]);

                if absakk >= alpha_val * colmax * (colmax / rowmax) {
                    kstep = 1;
                    ipiv[k] = (k + 1) as i32;
                } else if absaimax >= alpha_val * rowmax {
                    kstep = 1;
                    Self::swap_rows_cols_upper(a, k, imax, n);
                    ipiv[k] = (imax + 1) as i32;
                } else {
                    kstep = 2;
                    if imax != k - 1 {
                        Self::swap_rows_cols_upper(a, k - 1, imax, n);
                    }
                    ipiv[k] = -((imax + 1) as i32);
                    ipiv[k - 1] = -((imax + 1) as i32);
                }
            }

            if kstep == 1 {
                let akk = a[(k, k)];
                if akk == T::zero() {
                    return Err(BunchKaufmanError::Singular { index: k });
                }
                let akk_inv = T::one() / akk;

                for i in 0..k {
                    a[(i, k)] = a[(i, k)] * akk_inv;
                }

                for j in 0..k {
                    let ajk = a[(j, k)];
                    for i in 0..=j {
                        a[(i, j)] = a[(i, j)] - ajk * a[(i, k)] * akk;
                    }
                }
            } else {
                let km1 = k - 1;
                let akm1km1 = a[(km1, km1)];
                let akkm1 = a[(km1, k)];
                let akk = a[(k, k)];

                let det = akm1km1 * akk - akkm1 * akkm1;
                if det == T::zero() {
                    return Err(BunchKaufmanError::Singular { index: k });
                }
                let det_inv = T::one() / det;

                let d11 = akk * det_inv;
                let d22 = akm1km1 * det_inv;
                let d12 = -akkm1 * det_inv;

                for i in 0..km1 {
                    let wkm1 = a[(i, km1)];
                    let wk = a[(i, k)];
                    a[(i, km1)] = wkm1 * d11 + wk * d12;
                    a[(i, k)] = wkm1 * d12 + wk * d22;
                }

                for j in 0..km1 {
                    let ljkm1 = a[(j, km1)];
                    let ljk = a[(j, k)];
                    let djkm1 = akm1km1 * ljkm1 + akkm1 * ljk;
                    let djk = akkm1 * ljkm1 + akk * ljk;
                    for i in 0..=j {
                        let likm1 = a[(i, km1)];
                        let lik = a[(i, k)];
                        a[(i, j)] = a[(i, j)] - likm1 * djkm1 - lik * djk;
                    }
                }

                k -= 1; // Extra decrement for 2×2 block
            }
        }

        Ok(())
    }

    /// Swap rows and columns i and j in lower triangular storage.
    /// This performs a symmetric permutation P * A * P^T.
    fn swap_rows_cols_lower(a: &mut Mat<T>, i: usize, j: usize, n: usize) {
        if i == j {
            return;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };

        // Swap diagonal elements a[i,i] and a[j,j]
        let tmp = a[(i, i)];
        a[(i, i)] = a[(j, j)];
        a[(j, j)] = tmp;

        // Swap row i and row j for columns k < i
        // a[i,k] <-> a[j,k]
        for k in 0..i {
            let tmp = a[(i, k)];
            a[(i, k)] = a[(j, k)];
            a[(j, k)] = tmp;
        }

        // Swap a[k,i] (column i) with a[j,k] (row j) for i < k < j
        // In lower triangular: a[k,i] is stored, a[j,k] is stored
        for k in (i + 1)..j {
            let tmp = a[(k, i)];
            a[(k, i)] = a[(j, k)];
            a[(j, k)] = tmp;
        }

        // Swap column i and column j for rows k > j
        // a[k,i] <-> a[k,j]
        for k in (j + 1)..n {
            let tmp = a[(k, i)];
            a[(k, i)] = a[(k, j)];
            a[(k, j)] = tmp;
        }
    }

    /// Swap rows and columns i and j in upper triangular storage.
    fn swap_rows_cols_upper(a: &mut Mat<T>, i: usize, j: usize, n: usize) {
        if i == j {
            return;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };

        // Swap diagonal elements
        let tmp = a[(i, i)];
        a[(i, i)] = a[(j, j)];
        a[(j, j)] = tmp;

        // Swap column i and column j for rows < i
        for k in 0..i {
            let tmp = a[(k, i)];
            a[(k, i)] = a[(k, j)];
            a[(k, j)] = tmp;
        }

        // Swap row i elements (columns i+1 to j-1) with column j elements (rows i+1 to j-1)
        for k in (i + 1)..j {
            let tmp = a[(i, k)];
            a[(i, k)] = a[(k, j)];
            a[(k, j)] = tmp;
        }

        // Swap row i and row j for columns > j
        for k in (j + 1)..n {
            let tmp = a[(i, k)];
            a[(i, k)] = a[(j, k)];
            a[(j, k)] = tmp;
        }
    }

    /// Solves Ax = b using the factorization.
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector or matrix
    ///
    /// # Returns
    ///
    /// Solution x such that Ax = b.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, BunchKaufmanError> {
        if b.nrows() != self.n {
            return Err(BunchKaufmanError::DimensionMismatch {
                expected: self.n,
                actual: b.nrows(),
            });
        }

        let nrhs = b.ncols();
        let mut x = Mat::zeros(self.n, nrhs);
        for i in 0..self.n {
            for j in 0..nrhs {
                x[(i, j)] = b[(i, j)];
            }
        }

        match self.uplo {
            Uplo::Lower => self.solve_lower(&mut x, nrhs),
            Uplo::Upper => self.solve_upper(&mut x, nrhs),
        }

        Ok(x)
    }

    /// Solve using lower triangular factorization.
    fn solve_lower(&self, x: &mut Mat<T>, nrhs: usize) {
        let n = self.n;

        // Forward substitution: apply P and solve L * z = P * b
        let mut k = 0;
        while k < n {
            if self.ipiv[k] > 0 {
                // 1×1 pivot
                let kp = (self.ipiv[k] - 1) as usize;
                if kp != k {
                    // Swap rows k and kp
                    for j in 0..nrhs {
                        let tmp = x[(k, j)];
                        x[(k, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
                // Solve: x[k+1:n] -= L[k+1:n, k] * x[k]
                for i in (k + 1)..n {
                    let lik = self.factors[(i, k)];
                    for j in 0..nrhs {
                        x[(i, j)] = x[(i, j)] - lik * x[(k, j)];
                    }
                }
                k += 1;
            } else {
                // 2×2 pivot
                let kp = (-self.ipiv[k] - 1) as usize;
                if kp != k + 1 {
                    // Swap rows k+1 and kp
                    for j in 0..nrhs {
                        let tmp = x[(k + 1, j)];
                        x[(k + 1, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
                // Solve: x[k+2:n] -= L[k+2:n, k:k+2] * x[k:k+2]
                for i in (k + 2)..n {
                    let lik = self.factors[(i, k)];
                    let likp1 = self.factors[(i, k + 1)];
                    for j in 0..nrhs {
                        x[(i, j)] = x[(i, j)] - lik * x[(k, j)] - likp1 * x[(k + 1, j)];
                    }
                }
                k += 2;
            }
        }

        // Solve D * y = z
        k = 0;
        while k < n {
            if self.ipiv[k] > 0 {
                // 1×1 block
                let dkk = self.factors[(k, k)];
                for j in 0..nrhs {
                    x[(k, j)] = x[(k, j)] / dkk;
                }
                k += 1;
            } else {
                // 2×2 block
                let dkk = self.factors[(k, k)];
                let dkp1k = self.factors[(k + 1, k)];
                let dkp1kp1 = self.factors[(k + 1, k + 1)];
                let det = dkk * dkp1kp1 - dkp1k * dkp1k;
                for j in 0..nrhs {
                    let xk = x[(k, j)];
                    let xkp1 = x[(k + 1, j)];
                    x[(k, j)] = (dkp1kp1 * xk - dkp1k * xkp1) / det;
                    x[(k + 1, j)] = (dkk * xkp1 - dkp1k * xk) / det;
                }
                k += 2;
            }
        }

        // Backward substitution: solve L^T * x = y and apply P^T
        k = n;
        while k > 0 {
            k -= 1;
            if self.ipiv[k] > 0 {
                // 1×1 pivot
                // Solve: x[k] -= L[k+1:n, k]^T * x[k+1:n]
                for i in (k + 1)..n {
                    let lik = self.factors[(i, k)];
                    for j in 0..nrhs {
                        x[(k, j)] = x[(k, j)] - lik * x[(i, j)];
                    }
                }
                let kp = (self.ipiv[k] - 1) as usize;
                if kp != k {
                    for j in 0..nrhs {
                        let tmp = x[(k, j)];
                        x[(k, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
            } else if k > 0 && self.ipiv[k - 1] < 0 {
                // 2×2 pivot (we're at the second row of the block)
                k -= 1;
                // Solve: x[k:k+2] -= L[k+2:n, k:k+2]^T * x[k+2:n]
                for i in (k + 2)..n {
                    let lik = self.factors[(i, k)];
                    let likp1 = self.factors[(i, k + 1)];
                    for j in 0..nrhs {
                        x[(k, j)] = x[(k, j)] - lik * x[(i, j)];
                        x[(k + 1, j)] = x[(k + 1, j)] - likp1 * x[(i, j)];
                    }
                }
                let kp = (-self.ipiv[k] - 1) as usize;
                if kp != k + 1 {
                    for j in 0..nrhs {
                        let tmp = x[(k + 1, j)];
                        x[(k + 1, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
            }
        }
    }

    /// Solve using upper triangular factorization.
    fn solve_upper(&self, x: &mut Mat<T>, nrhs: usize) {
        let n = self.n;

        // Backward substitution: apply P and solve U * z = P * b
        let mut k = n;
        while k > 0 {
            k -= 1;
            if self.ipiv[k] > 0 {
                // 1×1 pivot
                let kp = (self.ipiv[k] - 1) as usize;
                if kp != k {
                    for j in 0..nrhs {
                        let tmp = x[(k, j)];
                        x[(k, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
                // Solve: x[0:k] -= U[0:k, k] * x[k]
                for i in 0..k {
                    let uik = self.factors[(i, k)];
                    for j in 0..nrhs {
                        x[(i, j)] = x[(i, j)] - uik * x[(k, j)];
                    }
                }
            } else if k > 0 && self.ipiv[k - 1] < 0 {
                // 2×2 pivot
                k -= 1;
                let kp = (-self.ipiv[k] - 1) as usize;
                if kp != k {
                    for j in 0..nrhs {
                        let tmp = x[(k, j)];
                        x[(k, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
                // Solve: x[0:k] -= U[0:k, k:k+2] * x[k:k+2]
                for i in 0..k {
                    let uik = self.factors[(i, k)];
                    let uikp1 = self.factors[(i, k + 1)];
                    for j in 0..nrhs {
                        x[(i, j)] = x[(i, j)] - uik * x[(k, j)] - uikp1 * x[(k + 1, j)];
                    }
                }
            }
        }

        // Solve D * y = z
        k = n;
        while k > 0 {
            k -= 1;
            if self.ipiv[k] > 0 {
                let dkk = self.factors[(k, k)];
                for j in 0..nrhs {
                    x[(k, j)] = x[(k, j)] / dkk;
                }
            } else if k > 0 && self.ipiv[k - 1] < 0 {
                k -= 1;
                let dkk = self.factors[(k, k)];
                let dkkp1 = self.factors[(k, k + 1)];
                let dkp1kp1 = self.factors[(k + 1, k + 1)];
                let det = dkk * dkp1kp1 - dkkp1 * dkkp1;
                for j in 0..nrhs {
                    let xk = x[(k, j)];
                    let xkp1 = x[(k + 1, j)];
                    x[(k, j)] = (dkp1kp1 * xk - dkkp1 * xkp1) / det;
                    x[(k + 1, j)] = (dkk * xkp1 - dkkp1 * xk) / det;
                }
            }
        }

        // Forward substitution: solve U^T * x = y and apply P^T
        k = 0;
        while k < n {
            if self.ipiv[k] > 0 {
                // 1×1 pivot
                for i in 0..k {
                    let uik = self.factors[(i, k)];
                    for j in 0..nrhs {
                        x[(k, j)] = x[(k, j)] - uik * x[(i, j)];
                    }
                }
                let kp = (self.ipiv[k] - 1) as usize;
                if kp != k {
                    for j in 0..nrhs {
                        let tmp = x[(k, j)];
                        x[(k, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
                k += 1;
            } else {
                // 2×2 pivot
                for i in 0..k {
                    let uik = self.factors[(i, k)];
                    let uikp1 = self.factors[(i, k + 1)];
                    for j in 0..nrhs {
                        x[(k, j)] = x[(k, j)] - uik * x[(i, j)];
                        x[(k + 1, j)] = x[(k + 1, j)] - uikp1 * x[(i, j)];
                    }
                }
                let kp = (-self.ipiv[k] - 1) as usize;
                if kp != k + 1 {
                    for j in 0..nrhs {
                        let tmp = x[(k + 1, j)];
                        x[(k + 1, j)] = x[(kp, j)];
                        x[(kp, j)] = tmp;
                    }
                }
                k += 2;
            }
        }
    }

    /// Returns the pivot indices.
    #[must_use]
    pub fn ipiv(&self) -> &[i32] {
        &self.ipiv
    }

    /// Returns the matrix dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the storage mode.
    #[must_use]
    pub fn uplo(&self) -> Uplo {
        self.uplo
    }

    /// Computes the inertia of the matrix (number of positive, negative, zero eigenvalues).
    ///
    /// # Returns
    ///
    /// A tuple (n_positive, n_negative, n_zero).
    #[must_use]
    pub fn inertia(&self) -> (usize, usize, usize) {
        let mut n_pos = 0;
        let mut n_neg = 0;
        let mut n_zero = 0;

        let mut k = 0;
        while k < self.n {
            if self.ipiv[k] > 0 {
                // 1×1 block
                let d = self.factors[(k, k)];
                if d.real() > T::Real::zero() {
                    n_pos += 1;
                } else if d.real() < T::Real::zero() {
                    n_neg += 1;
                } else {
                    n_zero += 1;
                }
                k += 1;
            } else {
                // 2×2 block - compute eigenvalues
                let (d11, d21, d22) = match self.uplo {
                    Uplo::Lower => {
                        let d11 = self.factors[(k, k)];
                        let d21 = self.factors[(k + 1, k)];
                        let d22 = self.factors[(k + 1, k + 1)];
                        (d11, d21, d22)
                    }
                    Uplo::Upper => {
                        let d11 = self.factors[(k, k)];
                        let d12 = self.factors[(k, k + 1)];
                        let d22 = self.factors[(k + 1, k + 1)];
                        (d11, d12, d22)
                    }
                };

                // Eigenvalues of 2×2 symmetric: (trace ± sqrt(trace² - 4*det)) / 2
                let trace = d11 + d22;
                let det = d11 * d22 - d21 * d21;
                let discriminant = trace * trace - T::from_f64(4.0).unwrap_or_else(T::zero) * det;

                if discriminant.real() >= T::Real::zero() {
                    let sqrt_disc = Real::sqrt(discriminant);
                    let two = T::from_f64(2.0).unwrap_or_else(T::zero);
                    let lambda1 = (trace + sqrt_disc) / two;
                    let lambda2 = (trace - sqrt_disc) / two;

                    if lambda1.real() > T::Real::zero() {
                        n_pos += 1;
                    } else if lambda1.real() < T::Real::zero() {
                        n_neg += 1;
                    } else {
                        n_zero += 1;
                    }

                    if lambda2.real() > T::Real::zero() {
                        n_pos += 1;
                    } else if lambda2.real() < T::Real::zero() {
                        n_neg += 1;
                    } else {
                        n_zero += 1;
                    }
                } else {
                    // Complex eigenvalues - shouldn't happen for real symmetric
                    n_zero += 2;
                }

                k += 2;
            }
        }

        (n_pos, n_neg, n_zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_bunch_kaufman_positive_definite() {
        // Positive definite matrix - should work like Cholesky
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 2.0], &[1.0, 2.0, 6.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!(
                approx_eq(ax_i, b[(i, 0)], 1e-10),
                "Ax[{}] = {}, b = {}",
                i,
                ax_i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_bunch_kaufman_indefinite() {
        // Indefinite matrix where standard LDLT might fail
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 0.0], &[2.0, 0.0, 3.0], &[0.0, 3.0, 4.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!(
                approx_eq(ax_i, b[(i, 0)], 1e-10),
                "Ax[{}] = {}, b = {}",
                i,
                ax_i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_bunch_kaufman_2x2() {
        let a = Mat::from_rows(&[&[0.0f64, 1.0], &[1.0, 0.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[1.0], &[2.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        // A = [[0, 1], [1, 0]], solution should be x = [2, 1]
        assert!(approx_eq(x[(0, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 1.0, 1e-10));
    }

    #[test]
    fn test_bunch_kaufman_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, -3.0, 0.0], &[0.0, 0.0, 4.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[2.0], &[-6.0], &[12.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        assert!(approx_eq(x[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(x[(1, 0)], 2.0, 1e-10));
        assert!(approx_eq(x[(2, 0)], 3.0, 1e-10));
    }

    #[test]
    fn test_bunch_kaufman_multiple_rhs() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[1.0, 4.0], &[2.0, 5.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        // Verify each column
        for col in 0..2 {
            for i in 0..2 {
                let mut ax_i = 0.0;
                for j in 0..2 {
                    ax_i += a[(i, j)] * x[(j, col)];
                }
                assert!(
                    approx_eq(ax_i, b[(i, col)], 1e-10),
                    "Ax[{},{}] = {}, b = {}",
                    i,
                    col,
                    ax_i,
                    b[(i, col)]
                );
            }
        }
    }

    #[test]
    fn test_bunch_kaufman_inertia_positive_definite() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0], &[2.0, 5.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();
        let (pos, neg, zero) = bk.inertia();

        assert_eq!(pos, 2);
        assert_eq!(neg, 0);
        assert_eq!(zero, 0);
    }

    #[test]
    fn test_bunch_kaufman_inertia_indefinite() {
        // Matrix with one positive and one negative eigenvalue
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[2.0, 1.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();
        let (pos, neg, zero) = bk.inertia();

        // Eigenvalues are 3 and -1
        assert_eq!(pos, 1);
        assert_eq!(neg, 1);
        assert_eq!(zero, 0);
    }

    #[test]
    fn test_bunch_kaufman_upper() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 2.0], &[1.0, 2.0, 6.0]]);

        let bk = BunchKaufman::compute_with_uplo(a.as_ref(), Uplo::Upper).unwrap();

        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!(
                approx_eq(ax_i, b[(i, 0)], 1e-10),
                "Ax[{}] = {}, b = {}",
                i,
                ax_i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_bunch_kaufman_f32() {
        let a = Mat::from_rows(&[&[4.0f32, 2.0, 1.0], &[2.0, 5.0, 2.0], &[1.0, 2.0, 6.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[1.0f32], &[2.0], &[3.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        // Verify Ax = b (with looser tolerance for f32)
        for i in 0..3 {
            let mut ax_i = 0.0f32;
            for j in 0..3 {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!(
                (ax_i - b[(i, 0)]).abs() < 1e-5,
                "Ax[{}] = {}, b = {}",
                i,
                ax_i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_bunch_kaufman_identity() {
        let a: Mat<f64> = Mat::eye(4);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
        let x = bk.solve(b.as_ref()).unwrap();

        for i in 0..4 {
            assert!(approx_eq(x[(i, 0)], b[(i, 0)], 1e-10));
        }
    }

    #[test]
    fn test_bunch_kaufman_large() {
        // Larger matrix to test the algorithm more thoroughly
        let n = 8;
        let mut a = Mat::zeros(n, n);

        // Create a well-conditioned symmetric indefinite matrix
        // Use a diagonally dominant structure with mix of positive/negative
        for i in 0..n {
            // Alternate positive/negative diagonal with larger magnitude
            a[(i, i)] = if i % 2 == 0 { 10.0 } else { -10.0 };
            for j in (i + 1)..n {
                // Small off-diagonal elements
                let val = 0.1 / ((i + j + 1) as f64);
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();

        // Create RHS
        let mut b = Mat::zeros(n, 1);
        for i in 0..n {
            b[(i, 0)] = (i as f64) + 1.0;
        }

        let x = bk.solve(b.as_ref()).unwrap();

        // Verify Ax = b
        for i in 0..n {
            let mut ax_i = 0.0;
            for j in 0..n {
                ax_i += a[(i, j)] * x[(j, 0)];
            }
            assert!(
                approx_eq(ax_i, b[(i, 0)], 1e-8),
                "Ax[{}] = {}, b = {}",
                i,
                ax_i,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_bunch_kaufman_negative_definite() {
        let a = Mat::from_rows(&[&[-4.0f64, -2.0], &[-2.0, -5.0]]);

        let bk = BunchKaufman::compute(a.as_ref()).unwrap();
        let (pos, neg, zero) = bk.inertia();

        assert_eq!(pos, 0);
        assert_eq!(neg, 2);
        assert_eq!(zero, 0);
    }
}
