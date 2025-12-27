//! Cholesky decomposition for symmetric positive definite band matrices.
//!
//! For SPD band matrices with kd sub/super-diagonals, the Cholesky factorization
//! produces a lower triangular band matrix L with the same bandwidth.
//!
//! # Band Storage Format (Lower Triangular)
//!
//! For an n×n SPD matrix A with bandwidth kd, only the lower triangular band
//! is stored in a (kd+1) × n array AB (column-major):
//!
//! ```text
//! AB[i-j, j] = A[i, j]  for max(0, j) ≤ i ≤ min(n-1, j+kd)
//! ```
//!
//! Example for 4×4 SPD matrix with kd=1 (tridiagonal):
//! ```text
//! A = [d0  a0   0   0]     AB = [d0 d1 d2 d3]  (row 0: main diagonal)
//!     [a0  d1  a1   0]          [a0 a1 a2  *]  (row 1: sub-diagonal)
//!     [ 0  a1  d2  a2]
//!     [ 0   0  a2  d3]
//! ```

use num_traits::{FromPrimitive, One};
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error returned when band Cholesky decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandCholeskyError {
    /// The matrix is not positive definite.
    NotPositiveDefinite {
        /// The row/column index where failure was detected.
        index: usize,
    },
    /// Invalid band dimensions.
    InvalidDimensions {
        /// Matrix size.
        n: usize,
        /// Bandwidth (number of sub-diagonals).
        kd: usize,
    },
    /// Band storage array has wrong length.
    InvalidStorageLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Dimension mismatch in solve operation.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for BandCholeskyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BandCholeskyError::NotPositiveDefinite { index } => {
                write!(
                    f,
                    "Band matrix is not positive definite (detected at index {index})"
                )
            }
            BandCholeskyError::InvalidDimensions { n, kd } => {
                write!(f, "Invalid band dimensions: n={n}, kd={kd}")
            }
            BandCholeskyError::InvalidStorageLength { expected, actual } => {
                write!(
                    f,
                    "Invalid band storage length: expected {expected}, got {actual}"
                )
            }
            BandCholeskyError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for BandCholeskyError {}

/// Cholesky decomposition for symmetric positive definite band matrices.
///
/// Computes A = LL^T where L is a lower triangular band matrix with
/// the same bandwidth as A.
///
/// The storage format follows LAPACK's DPBTRF convention.
#[derive(Clone, Debug)]
pub struct BandCholesky<T: Scalar> {
    /// The L factor in band storage.
    /// Size: (kd + 1) × n
    ab: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Bandwidth (number of sub-diagonals).
    kd: usize,
    /// Leading dimension of band storage.
    ldab: usize,
}

impl<T: Field + Real> BandCholesky<T> {
    /// Computes the Cholesky decomposition of a symmetric positive definite band matrix.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension (n×n matrix)
    /// * `kd` - Number of sub-diagonals (bandwidth - 1)
    /// * `ab` - Lower triangular band storage array (column-major)
    ///          Size must be (kd + 1) * n
    ///          AB[i-j + j*ldab] = A\[i, j\] for j ≤ i ≤ min(n-1, j+kd)
    ///
    /// # Returns
    ///
    /// The Cholesky factor L in band storage format.
    ///
    /// # Errors
    ///
    /// Returns `BandCholeskyError::InvalidDimensions` if kd >= n.
    /// Returns `BandCholeskyError::InvalidStorageLength` if ab has wrong length.
    /// Returns `BandCholeskyError::NotPositiveDefinite` if the matrix is not SPD.
    pub fn compute(n: usize, kd: usize, ab: &[T]) -> Result<Self, BandCholeskyError> {
        // Validate dimensions
        if n == 0 {
            return Ok(BandCholesky {
                ab: Vec::new(),
                n: 0,
                kd,
                ldab: kd + 1,
            });
        }

        if kd >= n {
            return Err(BandCholeskyError::InvalidDimensions { n, kd });
        }

        let ldab = kd + 1;
        let expected_len = ldab * n;

        if ab.len() != expected_len {
            return Err(BandCholeskyError::InvalidStorageLength {
                expected: expected_len,
                actual: ab.len(),
            });
        }

        // Copy to work array
        let mut ab_work = ab.to_vec();

        // Band Cholesky factorization
        // For each column j:
        //   L[j,j] = sqrt(A[j,j] - sum_{k=max(0,j-kd)}^{j-1} L[j,k]^2)
        //   L[i,j] = (A[i,j] - sum_{k=max(0,j-kd)}^{j-1} L[i,k]*L[j,k]) / L[j,j]
        //            for i = j+1, ..., min(n-1, j+kd)

        for j in 0..n {
            // Compute the diagonal element
            let mut sum = T::zero();
            let k_start = j.saturating_sub(kd);

            for k in k_start..j {
                // L[j,k] is at ab[(j-k) + k*ldab] in lower band storage
                let l_jk = ab_work[(j - k) + k * ldab];
                sum = sum + l_jk * l_jk;
            }

            // A[j,j] is at ab[0 + j*ldab]
            let diag = ab_work[j * ldab] - sum;

            // Check for positive definiteness
            let tol = <T as Scalar>::epsilon()
                * <T as FromPrimitive>::from_usize(n).unwrap_or(<T as One>::one());
            if diag <= tol {
                return Err(BandCholeskyError::NotPositiveDefinite { index: j });
            }

            ab_work[j * ldab] = Real::sqrt(diag);
            let l_jj = ab_work[j * ldab];

            // Compute off-diagonal elements in column j
            let i_end = (j + kd).min(n - 1);

            for i in (j + 1)..=i_end {
                let mut sum_off = T::zero();

                for k in k_start..j {
                    // L[i,k] and L[j,k]
                    // L[i,k] is at ab[(i-k) + k*ldab] if i-k <= kd, else 0
                    // L[j,k] is at ab[(j-k) + k*ldab] if j-k <= kd, else 0
                    let i_row = i - k;
                    let j_row = j - k;

                    if i_row <= kd && j_row <= kd {
                        let l_ik = ab_work[i_row + k * ldab];
                        let l_jk = ab_work[j_row + k * ldab];
                        sum_off = sum_off + l_ik * l_jk;
                    }
                }

                // A[i,j] is at ab[(i-j) + j*ldab]
                let a_ij = ab_work[(i - j) + j * ldab];
                ab_work[(i - j) + j * ldab] = (a_ij - sum_off) / l_jj;
            }
        }

        Ok(BandCholesky {
            ab: ab_work,
            n,
            kd,
            ldab,
        })
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns the bandwidth (number of sub-diagonals).
    #[inline]
    pub fn kd(&self) -> usize {
        self.kd
    }

    /// Returns the L factor in band storage format.
    pub fn ab(&self) -> &[T] {
        &self.ab
    }

    /// Computes the determinant of the original matrix.
    ///
    /// For A = LL^T, det(A) = det(L)^2 = (∏ L\[i,i\])^2
    pub fn determinant(&self) -> T {
        if self.n == 0 {
            return T::one();
        }

        let mut det_l = T::one();
        for j in 0..self.n {
            det_l = det_l * self.ab[j * self.ldab]; // L[j,j]
        }

        det_l * det_l
    }

    /// Returns the log-determinant of the original matrix.
    ///
    /// log(det(A)) = 2 * sum(log(L\[i,i\]))
    pub fn log_determinant(&self) -> T {
        if self.n == 0 {
            return T::zero();
        }

        let mut log_det = T::zero();
        let two = T::one() + T::one();
        for j in 0..self.n {
            log_det = log_det + Real::ln(self.ab[j * self.ldab]);
        }

        two * log_det
    }

    /// Solves the system Ax = b for SPD band matrix A.
    ///
    /// Given A = LL^T, solves:
    /// 1. Forward substitution: Ly = b
    /// 2. Back substitution: L^T x = y
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side vector (length n)
    ///
    /// # Errors
    ///
    /// Returns `BandCholeskyError::DimensionMismatch` if b has wrong length.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, BandCholeskyError> {
        if b.len() != self.n {
            return Err(BandCholeskyError::DimensionMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }

        if self.n == 0 {
            return Ok(Vec::new());
        }

        let mut x = b.to_vec();

        // Forward substitution: Ly = b
        // L is lower triangular band with bandwidth kd
        for j in 0..self.n {
            x[j] = x[j] / self.ab[j * self.ldab]; // x[j] /= L[j,j]

            let i_end = (j + self.kd).min(self.n - 1);
            for i in (j + 1)..=i_end {
                // L[i,j] is at ab[(i-j) + j*ldab]
                let l_ij = self.ab[(i - j) + j * self.ldab];
                x[i] = x[i] - l_ij * x[j];
            }
        }

        // Back substitution: L^T x = y
        // L^T is upper triangular band
        for j in (0..self.n).rev() {
            x[j] = x[j] / self.ab[j * self.ldab]; // x[j] /= L[j,j]

            let i_start = j.saturating_sub(self.kd);
            for i in i_start..j {
                // L[j,i] = L^T[i,j] is at ab[(j-i) + i*ldab]
                let l_ji = self.ab[(j - i) + i * self.ldab];
                x[i] = x[i] - l_ji * x[j];
            }
        }

        Ok(x)
    }

    /// Solves the system Ax = B for multiple right-hand sides.
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × nrhs in row-major order)
    /// * `nrhs` - Number of right-hand sides
    ///
    /// # Errors
    ///
    /// Returns `BandCholeskyError::DimensionMismatch` if b has wrong length.
    pub fn solve_multiple(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, BandCholeskyError> {
        if b.len() != self.n * nrhs {
            return Err(BandCholeskyError::DimensionMismatch {
                expected: self.n * nrhs,
                actual: b.len(),
            });
        }

        if self.n == 0 || nrhs == 0 {
            return Ok(Vec::new());
        }

        let mut x = b.to_vec();
        let ldb = nrhs;

        // Forward substitution: Ly = b
        for j in 0..self.n {
            let l_jj = self.ab[j * self.ldab];
            for col in 0..nrhs {
                x[j * ldb + col] = x[j * ldb + col] / l_jj;
            }

            let i_end = (j + self.kd).min(self.n - 1);
            for i in (j + 1)..=i_end {
                let l_ij = self.ab[(i - j) + j * self.ldab];
                for col in 0..nrhs {
                    x[i * ldb + col] = x[i * ldb + col] - l_ij * x[j * ldb + col];
                }
            }
        }

        // Back substitution: L^T x = y
        for j in (0..self.n).rev() {
            let l_jj = self.ab[j * self.ldab];
            for col in 0..nrhs {
                x[j * ldb + col] = x[j * ldb + col] / l_jj;
            }

            let i_start = j.saturating_sub(self.kd);
            for i in i_start..j {
                let l_ji = self.ab[(j - i) + i * self.ldab];
                for col in 0..nrhs {
                    x[i * ldb + col] = x[i * ldb + col] - l_ji * x[j * ldb + col];
                }
            }
        }

        Ok(x)
    }
}

/// Creates lower triangular band storage from a dense SPD matrix.
///
/// # Arguments
///
/// * `a` - Dense matrix as row-major array (n × n)
/// * `n` - Matrix dimension
/// * `kd` - Number of sub-diagonals (bandwidth - 1)
///
/// # Returns
///
/// Lower triangular band storage array of size (kd + 1) * n
pub fn dense_to_band_lower<T: Field + Real>(a: &[T], n: usize, kd: usize) -> Vec<T> {
    let ldab = kd + 1;
    let mut ab = vec![T::zero(); ldab * n];

    for j in 0..n {
        // Elements in column j: A[i,j] for j ≤ i ≤ min(n-1, j+kd)
        let i_end = (j + kd).min(n - 1);

        for i in j..=i_end {
            let row_in_band = i - j;
            ab[row_in_band + j * ldab] = a[i * n + j];
        }
    }

    ab
}

/// Extracts a dense matrix from lower triangular band storage.
///
/// # Arguments
///
/// * `ab` - Lower triangular band storage array
/// * `n` - Matrix dimension
/// * `kd` - Number of sub-diagonals
///
/// # Returns
///
/// Dense symmetric matrix as row-major array (n × n)
pub fn band_lower_to_dense<T: Field + Real>(ab: &[T], n: usize, kd: usize) -> Vec<T> {
    let ldab = kd + 1;
    let mut a = vec![T::zero(); n * n];

    for j in 0..n {
        let i_end = (j + kd).min(n - 1);

        for i in j..=i_end {
            let row_in_band = i - j;
            let val = ab[row_in_band + j * ldab];
            a[i * n + j] = val;
            // Symmetric: A[j,i] = A[i,j]
            if i != j {
                a[j * n + i] = val;
            }
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_to_band_lower_tridiagonal() {
        // Tridiagonal SPD matrix (kd=1)
        // [4 -1  0  0]
        // [-1 4 -1  0]
        // [0 -1  4 -1]
        // [0  0 -1  4]
        let n = 4;
        let kd = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);

        let ldab = kd + 1;
        assert_eq!(ab.len(), ldab * n);

        // Check main diagonal (row 0)
        assert!((ab[0] - 4.0).abs() < 1e-10);
        assert!((ab[ldab] - 4.0).abs() < 1e-10);
        assert!((ab[2 * ldab] - 4.0).abs() < 1e-10);
        assert!((ab[3 * ldab] - 4.0).abs() < 1e-10);

        // Check sub-diagonal (row 1)
        assert!((ab[1] - (-1.0)).abs() < 1e-10);
        assert!((ab[1 + ldab] - (-1.0)).abs() < 1e-10);
        assert!((ab[1 + 2 * ldab] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_band_lower_to_dense() {
        let n = 4;
        let kd = 1;
        #[rustfmt::skip]
        let a_orig: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band_lower(&a_orig, n, kd);
        let a_back = band_lower_to_dense(&ab, n, kd);

        for i in 0..n * n {
            assert!(
                (a_orig[i] - a_back[i]).abs() < 1e-10,
                "Mismatch at index {i}: {} vs {}",
                a_orig[i],
                a_back[i]
            );
        }
    }

    #[test]
    fn test_band_cholesky_tridiagonal() {
        // SPD tridiagonal matrix
        // [4 -1  0  0]
        // [-1 4 -1  0]
        // [0 -1  4 -1]
        // [0  0 -1  4]
        let n = 4;
        let kd = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);
        let chol = BandCholesky::compute(n, kd, &ab).expect("Should be SPD");

        // Solve Ax = b where b = [3, 2, 2, 3]
        // Expected solution: x = [1, 1, 1, 1]
        let b = vec![3.0, 2.0, 2.0, 3.0];
        let x = chol.solve(&b).expect("Should solve");

        for i in 0..n {
            assert!(
                (x[i] - 1.0).abs() < 1e-10,
                "x[{i}] = {}, expected 1.0",
                x[i]
            );
        }
    }

    #[test]
    fn test_band_cholesky_determinant() {
        // 2x2 SPD matrix with kd=1
        // [4 2]
        // [2 5]
        // det = 20 - 4 = 16
        let n = 2;
        let kd = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, 2.0,
            2.0, 5.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);
        let chol = BandCholesky::compute(n, kd, &ab).expect("Should be SPD");

        let det = chol.determinant();
        assert!((det - 16.0).abs() < 1e-10, "det = {det}, expected 16");
    }

    #[test]
    fn test_band_cholesky_pentadiagonal() {
        // SPD pentadiagonal matrix (kd=2)
        let n = 5;
        let kd = 2;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            10.0, -1.0, -2.0,  0.0,  0.0,
            -1.0, 10.0, -1.0, -2.0,  0.0,
            -2.0, -1.0, 10.0, -1.0, -2.0,
             0.0, -2.0, -1.0, 10.0, -1.0,
             0.0,  0.0, -2.0, -1.0, 10.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);
        let chol = BandCholesky::compute(n, kd, &ab).expect("Should be SPD");

        // Create RHS such that x = [1, 1, 1, 1, 1]
        let b = vec![7.0, 6.0, 4.0, 6.0, 7.0];
        let x = chol.solve(&b).expect("Should solve");

        // Verify Ax ≈ b
        for i in 0..n {
            let mut ax_i = 0.0;
            for j in 0..n {
                ax_i += a[i * n + j] * x[j];
            }
            assert!(
                (ax_i - b[i]).abs() < 1e-9,
                "Ax[{i}] = {ax_i}, expected {}",
                b[i]
            );
        }
    }

    #[test]
    fn test_band_cholesky_not_spd() {
        // Not positive definite (diagonal too small)
        let n = 3;
        let kd = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            1.0, -2.0, 0.0,
            -2.0, 1.0, -2.0,
            0.0, -2.0, 1.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);
        let result = BandCholesky::<f64>::compute(n, kd, &ab);

        assert!(result.is_err());
        match result {
            Err(BandCholeskyError::NotPositiveDefinite { index: _ }) => {}
            _ => panic!("Expected NotPositiveDefinite error"),
        }
    }

    #[test]
    fn test_band_cholesky_solve_multiple() {
        let n = 4;
        let kd = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);
        let chol = BandCholesky::compute(n, kd, &ab).expect("Should be SPD");

        // Two RHS
        let nrhs = 2;
        #[rustfmt::skip]
        let b = vec![
            3.0, 1.0,  // row 0: b1=3, b2=1
            2.0, 2.0,  // row 1
            2.0, 2.0,  // row 2
            3.0, 1.0,  // row 3
        ];

        let x = chol.solve_multiple(&b, nrhs).expect("Should solve");

        // Verify each RHS
        for rhs in 0..nrhs {
            for i in 0..n {
                let mut ax_i = 0.0;
                for j in 0..n {
                    ax_i += a[i * n + j] * x[j * nrhs + rhs];
                }
                let b_i = b[i * nrhs + rhs];
                assert!(
                    (ax_i - b_i).abs() < 1e-9,
                    "RHS {rhs}: Ax[{i}] = {ax_i}, expected {b_i}"
                );
            }
        }
    }

    #[test]
    fn test_band_cholesky_empty() {
        let result = BandCholesky::<f64>::compute(0, 0, &[]);
        assert!(result.is_ok());
        let chol = result.unwrap();
        assert_eq!(chol.size(), 0);
    }

    #[test]
    fn test_band_cholesky_f32() {
        let n = 3;
        let kd = 1;
        #[rustfmt::skip]
        let a: Vec<f32> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band_lower(&a, n, kd);
        let chol = BandCholesky::compute(n, kd, &ab).expect("Should be SPD");

        let b = vec![3.0f32, 2.0, 3.0];
        let x = chol.solve(&b).expect("Should solve");

        // Verify Ax ≈ b
        for i in 0..n {
            let mut ax_i = 0.0f32;
            for j in 0..n {
                ax_i += a[i * n + j] * x[j];
            }
            assert!(
                (ax_i - b[i]).abs() < 1e-5,
                "Ax[{i}] = {ax_i}, expected {}",
                b[i]
            );
        }
    }
}
