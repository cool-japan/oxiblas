//! LU decomposition for band matrices.
//!
//! Band matrices have non-zero elements only in a diagonal band around the main diagonal.
//! A band matrix with `kl` sub-diagonals and `ku` super-diagonals has bandwidth `kl + ku + 1`.
//!
//! The band storage format stores the diagonals as rows:
//! - Rows 0..kl: extra storage for fill-in during pivoting
//! - Rows kl..(kl+ku): super-diagonals
//! - Row kl+ku: main diagonal
//! - Rows (kl+ku+1)..(2*kl+ku+1): sub-diagonals
//!
//! For an n×n matrix A with kl sub-diagonals and ku super-diagonals,
//! the band storage uses a (2*kl + ku + 1) × n array.
//!
//! The element A\[i,j\] (with max(0, j-ku) ≤ i ≤ min(n-1, j+kl)) is stored at:
//! band[kl + ku + i - j, j]

use num_traits::{FromPrimitive, One};
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Helper to get absolute value using Scalar trait
#[inline]
fn scalar_abs<T: Scalar>(x: T) -> T::Real {
    Scalar::abs(x)
}

/// Helper to get epsilon using Scalar trait
#[inline]
fn scalar_epsilon<T: Scalar>() -> T::Real {
    <T as Scalar>::epsilon()
}

/// Error returned when band LU decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandLuError {
    /// The matrix is singular (has a zero or near-zero pivot).
    Singular {
        /// The row/column index where the singularity was detected.
        index: usize,
    },
    /// Invalid band dimensions.
    InvalidDimensions {
        /// Matrix size.
        n: usize,
        /// Lower bandwidth.
        kl: usize,
        /// Upper bandwidth.
        ku: usize,
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

impl core::fmt::Display for BandLuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BandLuError::Singular { index } => {
                write!(f, "Band matrix is singular at index {index}")
            }
            BandLuError::InvalidDimensions { n, kl, ku } => {
                write!(f, "Invalid band dimensions: n={n}, kl={kl}, ku={ku}")
            }
            BandLuError::InvalidStorageLength { expected, actual } => {
                write!(
                    f,
                    "Invalid band storage length: expected {expected}, got {actual}"
                )
            }
            BandLuError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for BandLuError {}

/// LU decomposition for band matrices with partial pivoting.
///
/// Stores the factorization PA = LU where:
/// - P is a permutation matrix (stored as pivot indices)
/// - L is lower triangular band with unit diagonal and bandwidth kl
/// - U is upper triangular band with bandwidth ku + kl (due to fill-in)
///
/// The storage format follows LAPACK's DGBTRF convention.
#[derive(Clone, Debug)]
pub struct BandLu<T: Scalar> {
    /// Combined L and U factors in band storage.
    /// Size: (2*kl + ku + 1) × n
    ab: Vec<T>,
    /// Matrix dimension.
    n: usize,
    /// Number of sub-diagonals (lower bandwidth).
    kl: usize,
    /// Number of super-diagonals (upper bandwidth).
    ku: usize,
    /// Leading dimension of band storage array.
    ldab: usize,
    /// Pivot indices: row i was swapped with row pivot[i].
    pivot: Vec<usize>,
}

impl<T: Field + Real> BandLu<T> {
    /// Computes the LU decomposition of a band matrix.
    ///
    /// Uses partial pivoting (row permutations) for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension (n×n matrix)
    /// * `kl` - Number of sub-diagonals (lower bandwidth)
    /// * `ku` - Number of super-diagonals (upper bandwidth)
    /// * `ab` - Band storage array in LAPACK format (row-major)
    ///          Size must be (2*kl + ku + 1) * n
    ///          The matrix is stored with the j-th column in column j,
    ///          and diagonal i is in row kl + ku + i.
    ///
    /// # Storage Format
    ///
    /// For a band matrix with kl=1, ku=1 (tridiagonal):
    /// ```text
    /// Original matrix:
    /// [d0  u0   0   0]
    /// [l0  d1  u1   0]
    /// [ 0  l1  d2  u2]
    /// [ 0   0  l2  d3]
    ///
    /// Band storage (ldab = 2*1+1+1 = 4, for fill-in space at top):
    /// Row 0:  *   *   *   *   (fill-in space)
    /// Row 1: u0  u1  u2   *   (super-diagonal)
    /// Row 2: d0  d1  d2  d3   (main diagonal)
    /// Row 3: l0  l1  l2   *   (sub-diagonal)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `BandLuError::InvalidDimensions` if kl or ku >= n.
    /// Returns `BandLuError::InvalidStorageLength` if ab has wrong length.
    /// Returns `BandLuError::Singular` if the matrix is singular.
    pub fn compute(n: usize, kl: usize, ku: usize, ab: &[T]) -> Result<Self, BandLuError> {
        // Validate dimensions
        if n == 0 {
            return Ok(BandLu {
                ab: Vec::new(),
                n: 0,
                kl,
                ku,
                ldab: 2 * kl + ku + 1,
                pivot: Vec::new(),
            });
        }

        if kl >= n || ku >= n {
            return Err(BandLuError::InvalidDimensions { n, kl, ku });
        }

        let ldab = 2 * kl + ku + 1;
        let expected_len = ldab * n;

        if ab.len() != expected_len {
            return Err(BandLuError::InvalidStorageLength {
                expected: expected_len,
                actual: ab.len(),
            });
        }

        // Copy to work array
        let mut ab_work = ab.to_vec();
        let mut pivot = vec![0usize; n];

        // Perform band LU factorization with partial pivoting
        // Algorithm based on LAPACK's DGBTRF

        // ju tracks the maximum column affected by fill-in
        let mut ju = 0usize;

        for j in 0..n {
            // Find pivot in column j
            // Search from row j to min(j + kl, n-1)
            let km = kl.min(n - 1 - j);
            let mut pivot_row = 0; // relative to j
            let mut pivot_val = scalar_abs(ab_work[band_idx(ldab, kl, ku, j, j)]);

            for i in 1..=km {
                let val = scalar_abs(ab_work[band_idx(ldab, kl, ku, j + i, j)]);
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }

            pivot[j] = j + pivot_row;

            // Check for singularity
            let tol = scalar_epsilon::<T>()
                * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
            if pivot_val <= tol {
                return Err(BandLuError::Singular { index: j });
            }

            // Swap rows if needed
            if pivot_row != 0 {
                // Update ju to track fill-in extent
                ju = ju.max((j + ku + pivot_row).min(n - 1));

                // Swap elements in columns max(j - ku - kl, 0) to min(j + ku + kl, n-1)
                let jfirst = j.saturating_sub(ku + kl);
                let jlast = (j + ku + kl).min(n - 1);

                for jj in jfirst..=jlast {
                    // Only swap if both rows have elements in this column
                    let row1 = j;
                    let row2 = j + pivot_row;

                    // Check if row1 has element in column jj
                    let row1_has = jj.saturating_sub(ku) <= row1 && row1 <= jj + kl;
                    // Check if row2 has element in column jj
                    let row2_has = jj.saturating_sub(ku) <= row2 && row2 <= jj + kl;

                    if row1_has && row2_has {
                        let idx1 = band_idx(ldab, kl, ku, row1, jj);
                        let idx2 = band_idx(ldab, kl, ku, row2, jj);
                        ab_work.swap(idx1, idx2);
                    } else if row2_has && !row1_has {
                        // row2 has element but row1 is outside band - need fill-in
                        let idx2 = band_idx(ldab, kl, ku, row2, jj);
                        // Fill-in goes to extended band storage
                        if jj >= j && jj <= j + ku + kl {
                            let idx1 = band_idx_extended(ldab, kl, ku, row1, jj);
                            ab_work.swap(idx1, idx2);
                        }
                    }
                }
            }

            // Compute multipliers
            let pivot_inv = T::one() / ab_work[band_idx(ldab, kl, ku, j, j)];

            for i in 1..=km {
                let idx = band_idx(ldab, kl, ku, j + i, j);
                ab_work[idx] = ab_work[idx] * pivot_inv;
            }

            // Update the trailing matrix
            // U row j affects columns j+1 to min(j + ku + kl, n-1)
            let ju_local = (j + ku).min(n - 1);

            for jj in (j + 1)..=ju.max(ju_local) {
                // Check if U[j, jj] exists
                if jj > j + ku + kl {
                    break;
                }

                // Get U[j, jj]
                let u_jjj = if jj <= j + ku {
                    ab_work[band_idx(ldab, kl, ku, j, jj)]
                } else {
                    // Fill-in element
                    ab_work[band_idx_extended(ldab, kl, ku, j, jj)]
                };

                if u_jjj == T::zero() {
                    continue;
                }

                // Update A[j+1:j+km, jj] -= L[j+1:j+km, j] * U[j, jj]
                for i in 1..=km {
                    // Check if A[j+i, jj] exists in band
                    let row = j + i;
                    let col = jj;

                    // Distance from main diagonal
                    let diag_dist = col as isize - row as isize;

                    // Check if within band (including fill-in region)
                    if diag_dist >= -(kl as isize) && diag_dist <= (ku as isize + kl as isize) {
                        let l_elem = ab_work[band_idx(ldab, kl, ku, row, j)];
                        let idx = if diag_dist <= ku as isize {
                            band_idx(ldab, kl, ku, row, col)
                        } else {
                            band_idx_extended(ldab, kl, ku, row, col)
                        };
                        ab_work[idx] = ab_work[idx] - l_elem * u_jjj;
                    }
                }
            }
        }

        Ok(BandLu {
            ab: ab_work,
            n,
            kl,
            ku,
            ldab,
            pivot,
        })
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns the number of sub-diagonals (lower bandwidth).
    #[inline]
    pub fn kl(&self) -> usize {
        self.kl
    }

    /// Returns the number of super-diagonals (upper bandwidth).
    #[inline]
    pub fn ku(&self) -> usize {
        self.ku
    }

    /// Returns the pivot indices.
    pub fn pivot(&self) -> &[usize] {
        &self.pivot
    }

    /// Returns the factored band matrix storage.
    pub fn ab(&self) -> &[T] {
        &self.ab
    }

    /// Solves the system Ax = b.
    ///
    /// Given the LU factorization PA = LU, solves:
    /// 1. Apply permutation: Pb
    /// 2. Forward substitution: Ly = Pb
    /// 3. Back substitution: Ux = y
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side vector (length n)
    ///
    /// # Errors
    ///
    /// Returns `BandLuError::DimensionMismatch` if b has wrong length.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, BandLuError> {
        if b.len() != self.n {
            return Err(BandLuError::DimensionMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }

        if self.n == 0 {
            return Ok(Vec::new());
        }

        let mut x = b.to_vec();

        // Apply row permutations (forward)
        for k in 0..self.n {
            let pk = self.pivot[k];
            if k != pk {
                x.swap(k, pk);
            }
        }

        // Forward substitution: Ly = Pb
        // L has unit diagonal and bandwidth kl
        for j in 0..self.n {
            let km = self.kl.min(self.n - 1 - j);
            for i in 1..=km {
                let l_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j + i, j)];
                x[j + i] = x[j + i] - l_elem * x[j];
            }
        }

        // Back substitution: Ux = y
        // U has bandwidth ku + kl (due to fill-in during pivoting)
        for j in (0..self.n).rev() {
            // Divide by diagonal
            let diag = self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)];
            x[j] = x[j] / diag;

            // Update x[i] for i < j where U[i, j] != 0
            let kmax = self.ku + self.kl;
            for i in j.saturating_sub(kmax)..j {
                // Distance from diagonal
                let diag_dist = j as isize - i as isize;

                // Get U[i, j]
                let u_elem = if diag_dist <= self.ku as isize {
                    self.ab[band_idx(self.ldab, self.kl, self.ku, i, j)]
                } else {
                    // Fill-in element
                    self.ab[band_idx_extended(self.ldab, self.kl, self.ku, i, j)]
                };

                x[i] = x[i] - u_elem * x[j];
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
    /// Returns `BandLuError::DimensionMismatch` if b has wrong length.
    pub fn solve_multiple(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, BandLuError> {
        if b.len() != self.n * nrhs {
            return Err(BandLuError::DimensionMismatch {
                expected: self.n * nrhs,
                actual: b.len(),
            });
        }

        if self.n == 0 || nrhs == 0 {
            return Ok(Vec::new());
        }

        let mut x = b.to_vec();
        let ldb = nrhs;

        // Apply row permutations (forward)
        for k in 0..self.n {
            let pk = self.pivot[k];
            if k != pk {
                // Swap rows k and pk
                for col in 0..nrhs {
                    x.swap(k * ldb + col, pk * ldb + col);
                }
            }
        }

        // Forward substitution: Ly = Pb
        for j in 0..self.n {
            let km = self.kl.min(self.n - 1 - j);
            for i in 1..=km {
                let l_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j + i, j)];
                for col in 0..nrhs {
                    x[(j + i) * ldb + col] = x[(j + i) * ldb + col] - l_elem * x[j * ldb + col];
                }
            }
        }

        // Back substitution: Ux = y
        for j in (0..self.n).rev() {
            let diag = self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)];
            for col in 0..nrhs {
                x[j * ldb + col] = x[j * ldb + col] / diag;
            }

            let kmax = self.ku + self.kl;
            for i in j.saturating_sub(kmax)..j {
                let diag_dist = j as isize - i as isize;
                let u_elem = if diag_dist <= self.ku as isize {
                    self.ab[band_idx(self.ldab, self.kl, self.ku, i, j)]
                } else {
                    self.ab[band_idx_extended(self.ldab, self.kl, self.ku, i, j)]
                };

                for col in 0..nrhs {
                    x[i * ldb + col] = x[i * ldb + col] - u_elem * x[j * ldb + col];
                }
            }
        }

        Ok(x)
    }

    /// Solves the transposed system A^T x = b.
    ///
    /// Given the LU factorization PA = LU, solves A^T x = b:
    /// 1. Forward substitution with U^T
    /// 2. Back substitution with L^T
    /// 3. Apply inverse permutation
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side vector (length n)
    ///
    /// # Errors
    ///
    /// Returns `BandLuError::DimensionMismatch` if b has wrong length.
    pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, BandLuError> {
        if b.len() != self.n {
            return Err(BandLuError::DimensionMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }

        if self.n == 0 {
            return Ok(Vec::new());
        }

        let mut x = b.to_vec();

        // Forward substitution with U^T: U^T y = b
        // U^T has bandwidth ku + kl (due to fill-in)
        for j in 0..self.n {
            // Divide by diagonal (U[j,j])
            let diag = self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)];
            x[j] = x[j] / diag;

            // Update x[i] for i > j where U[j, i] != 0 (which is U^T[i, j])
            let kmax = self.ku + self.kl;
            for i in (j + 1)..=((j + kmax).min(self.n - 1)) {
                // Get U[j, i] (transpose of what we're solving)
                let diag_dist = i as isize - j as isize;
                let u_elem = if diag_dist <= self.ku as isize {
                    self.ab[band_idx(self.ldab, self.kl, self.ku, j, i)]
                } else {
                    self.ab[band_idx_extended(self.ldab, self.kl, self.ku, j, i)]
                };

                x[i] = x[i] - u_elem * x[j];
            }
        }

        // Back substitution with L^T: L^T z = y
        // L has unit diagonal and bandwidth kl
        for j in (0..self.n).rev() {
            let km = self.kl.min(j);
            for i in (j.saturating_sub(km))..j {
                // L[j, i] = self.ab[...] but L is lower triangular, so we need L[j, i]
                // which is stored where L[j, i] for j > i
                let l_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j, i)];
                x[i] = x[i] - l_elem * x[j];
            }
        }

        // Apply inverse permutation (backward)
        for k in (0..self.n).rev() {
            let pk = self.pivot[k];
            if k != pk {
                x.swap(k, pk);
            }
        }

        Ok(x)
    }

    /// Estimates the reciprocal condition number of the matrix (LAPACK DGBCON).
    ///
    /// Computes rcond = 1 / (||A||_1 * ||A^{-1}||_1) where ||A^{-1}||_1 is
    /// estimated using Hager's algorithm.
    ///
    /// # Arguments
    ///
    /// * `anorm_1` - The 1-norm of the original matrix A (before factorization).
    ///               Compute this using `band_norm_1` before calling `compute`.
    ///
    /// # Returns
    ///
    /// Reciprocal condition number. A value close to 1 indicates a well-conditioned
    /// matrix, while a value close to 0 or machine epsilon indicates ill-conditioning.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::lu::{BandLu, dense_to_band, band_norm_1};
    ///
    /// // Tridiagonal matrix
    /// let a = vec![2.0f64, -1.0, 0.0, 0.0,
    ///             -1.0, 2.0, -1.0, 0.0,
    ///              0.0, -1.0, 2.0, -1.0,
    ///              0.0, 0.0, -1.0, 2.0];
    /// let n = 4;
    /// let kl = 1;
    /// let ku = 1;
    ///
    /// let ab = dense_to_band(&a, n, kl, ku);
    /// let anorm = band_norm_1(&ab, n, kl, ku);
    ///
    /// let lu = BandLu::compute(n, kl, ku, &ab).unwrap();
    /// let rcond = lu.rcond(anorm);
    /// assert!(rcond > 0.0 && rcond <= 1.0);
    /// ```
    pub fn rcond(&self, anorm_1: T) -> T {
        if self.n == 0 || anorm_1 == T::zero() {
            return T::zero();
        }

        // Estimate ||A^{-1}||_1 using Hager's algorithm
        let ainv_norm_1 = self.estimate_inv_norm_1();

        if ainv_norm_1 == T::zero() {
            return T::one();
        }

        T::one() / (anorm_1 * ainv_norm_1)
    }

    /// Estimates ||A^{-1}||_1 using Hager's algorithm.
    fn estimate_inv_norm_1(&self) -> T {
        let n = self.n;
        if n == 0 {
            return T::zero();
        }

        // Initialize with x = (1/n, 1/n, ..., 1/n)
        let one_over_n = T::one() / T::from_usize(n).unwrap_or(T::one());
        let mut x = vec![one_over_n; n];

        // Maximum iterations for Hager's algorithm
        const MAX_ITER: usize = 5;

        let mut gamma = T::zero();

        for _iter in 0..MAX_ITER {
            // Solve A * w = x
            let w = match self.solve(&x) {
                Ok(w) => w,
                Err(_) => return T::from_f64(1e30).unwrap_or(T::one() / <T as Scalar>::epsilon()),
            };

            // Compute ||w||_1
            let mut gamma_new = T::zero();
            for &wi in &w {
                gamma_new = gamma_new + Scalar::abs(wi);
            }

            // Check for convergence
            if gamma_new <= gamma {
                return gamma;
            }
            gamma = gamma_new;

            // Set xi = sign(w)
            for i in 0..n {
                x[i] = if w[i] >= T::zero() {
                    T::one()
                } else {
                    -T::one()
                };
            }

            // Solve A^T * z = xi
            let z = match self.solve_transpose(&x) {
                Ok(z) => z,
                Err(_) => return gamma,
            };

            // Find j = argmax |z_j|
            let mut j_max = 0;
            let mut z_max = Scalar::abs(z[0]);
            for j in 1..n {
                let z_abs = Scalar::abs(z[j]);
                if z_abs > z_max {
                    z_max = z_abs;
                    j_max = j;
                }
            }

            // Check if z_max <= z^T * xi (Hager's termination criterion)
            let mut z_dot_xi = T::zero();
            for i in 0..n {
                z_dot_xi = z_dot_xi + z[i] * x[i];
            }

            if z_max <= z_dot_xi {
                return gamma;
            }

            // Set x = e_{j_max} for next iteration
            for i in 0..n {
                x[i] = T::zero();
            }
            x[j_max] = T::one();
        }

        gamma
    }

    /// Returns the condition number (ratio of max to min diagonal of U).
    ///
    /// This is a simple upper bound on the condition number based on the
    /// diagonal elements of U after LU factorization. For more accurate
    /// estimation, use `rcond` with the original matrix norm.
    pub fn condition_number_estimate(&self) -> T {
        if self.n == 0 {
            return T::one();
        }

        let mut max_diag = T::zero();
        let mut min_diag = T::from_f64(1e30).unwrap_or(T::one() / <T as Scalar>::epsilon());

        for j in 0..self.n {
            let diag = Scalar::abs(self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)]);
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag && diag > T::zero() {
                min_diag = diag;
            }
        }

        if min_diag > T::zero() {
            max_diag / min_diag
        } else {
            T::from_f64(1e30).unwrap_or(T::one() / <T as Scalar>::epsilon())
        }
    }
}

/// Computes the 1-norm of a band matrix (maximum column sum).
///
/// # Arguments
///
/// * `ab` - Band storage array
/// * `n` - Matrix dimension
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
///
/// # Returns
///
/// The 1-norm of the matrix.
pub fn band_norm_1<T: Field + Real>(ab: &[T], n: usize, kl: usize, ku: usize) -> T {
    let ldab = 2 * kl + ku + 1;
    let mut max_col_sum = T::zero();

    for j in 0..n {
        let mut col_sum = T::zero();
        let i_start = j.saturating_sub(ku);
        let i_end = (j + kl).min(n - 1);

        for i in i_start..=i_end {
            let row_in_band = kl + ku + i - j;
            col_sum = col_sum + Scalar::abs(ab[row_in_band + j * ldab]);
        }

        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }

    max_col_sum
}

/// Computes the infinity-norm of a band matrix (maximum row sum).
///
/// # Arguments
///
/// * `ab` - Band storage array
/// * `n` - Matrix dimension
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
///
/// # Returns
///
/// The infinity-norm of the matrix.
pub fn band_norm_inf<T: Field + Real>(ab: &[T], n: usize, kl: usize, ku: usize) -> T {
    let ldab = 2 * kl + ku + 1;
    let mut row_sums = vec![T::zero(); n];

    for j in 0..n {
        let i_start = j.saturating_sub(ku);
        let i_end = (j + kl).min(n - 1);

        for i in i_start..=i_end {
            let row_in_band = kl + ku + i - j;
            row_sums[i] = row_sums[i] + Scalar::abs(ab[row_in_band + j * ldab]);
        }
    }

    let mut max_row_sum = T::zero();
    for &sum in &row_sums {
        if sum > max_row_sum {
            max_row_sum = sum;
        }
    }

    max_row_sum
}

/// Computes the index in band storage for element (i, j).
///
/// For a band matrix with kl sub-diagonals and ku super-diagonals,
/// stored in a (2*kl + ku + 1) × n array, element A\[i,j\] is at index:
/// band[kl + ku + i - j, j]
///
/// This function returns the flat index in column-major order.
/// Index = row_in_band + j * ldab
#[inline]
fn band_idx(ldab: usize, kl: usize, ku: usize, i: usize, j: usize) -> usize {
    let row_in_band = kl + ku + i - j;
    row_in_band + j * ldab
}

/// Index for extended band storage (fill-in elements).
#[inline]
fn band_idx_extended(ldab: usize, kl: usize, ku: usize, i: usize, j: usize) -> usize {
    // Fill-in elements go in the top kl rows
    let row_in_band = kl + ku + i - j;
    row_in_band + j * ldab
}

/// Creates band storage from a dense matrix.
///
/// # Arguments
///
/// * `a` - Dense matrix as row-major array (n × n)
/// * `n` - Matrix dimension
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
///
/// # Returns
///
/// Band storage array of size (2*kl + ku + 1) * n
pub fn dense_to_band<T: Field + Real>(a: &[T], n: usize, kl: usize, ku: usize) -> Vec<T> {
    let ldab = 2 * kl + ku + 1;
    let mut ab = vec![T::zero(); ldab * n];

    for j in 0..n {
        // Elements in column j
        let i_start = j.saturating_sub(ku);
        let i_end = (j + kl).min(n - 1);

        for i in i_start..=i_end {
            let row_in_band = kl + ku + i - j;
            ab[row_in_band + j * ldab] = a[i * n + j];
        }
    }

    ab
}

/// Extracts a dense matrix from band storage.
///
/// # Arguments
///
/// * `ab` - Band storage array
/// * `n` - Matrix dimension
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
///
/// # Returns
///
/// Dense matrix as row-major array (n × n)
pub fn band_to_dense<T: Field + Real>(ab: &[T], n: usize, kl: usize, ku: usize) -> Vec<T> {
    let ldab = 2 * kl + ku + 1;
    let mut a = vec![T::zero(); n * n];

    for j in 0..n {
        let i_start = j.saturating_sub(ku);
        let i_end = (j + kl).min(n - 1);

        for i in i_start..=i_end {
            let row_in_band = kl + ku + i - j;
            a[i * n + j] = ab[row_in_band + j * ldab];
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_to_band_tridiagonal() {
        // Tridiagonal matrix (kl=1, ku=1)
        // [2 -1  0  0]
        // [-1 2 -1  0]
        // [0 -1  2 -1]
        // [0  0 -1  2]
        let n = 4;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            2.0, -1.0, 0.0, 0.0,
            -1.0, 2.0, -1.0, 0.0,
            0.0, -1.0, 2.0, -1.0,
            0.0, 0.0, -1.0, 2.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);

        // ldab = 2*1 + 1 + 1 = 4
        // Band storage format (column-major):
        // A[i,j] stored at ab[kl+ku+i-j + j*ldab]
        // For kl=1, ku=1: row_in_band = 2 + i - j
        //
        // Column j=0: A[0,0] at row 2
        // Column j=1: A[0,1] at row 1, A[1,1] at row 2
        // Column j=2: A[1,2] at row 1, A[2,2] at row 2
        // Column j=3: A[2,3] at row 1, A[3,3] at row 2
        // Sub-diagonals:
        // Column j=0: A[1,0] at row 3
        // Column j=1: A[2,1] at row 3
        // Column j=2: A[3,2] at row 3
        let ldab = 4;
        assert_eq!(ab.len(), ldab * n);

        // Check main diagonal: A[i,i] at row_in_band = 2
        assert!((ab[2] - 2.0).abs() < 1e-10); // A[0,0]
        assert!((ab[2 + ldab] - 2.0).abs() < 1e-10); // A[1,1]
        assert!((ab[2 + 2 * ldab] - 2.0).abs() < 1e-10); // A[2,2]
        assert!((ab[2 + 3 * ldab] - 2.0).abs() < 1e-10); // A[3,3]

        // Check super-diagonal: A[i,i+1] at row_in_band = 1
        // Note: A[i,j] is in column j of band storage
        assert!((ab[1 + ldab] - (-1.0)).abs() < 1e-10); // A[0,1] in column 1
        assert!((ab[1 + 2 * ldab] - (-1.0)).abs() < 1e-10); // A[1,2] in column 2
        assert!((ab[1 + 3 * ldab] - (-1.0)).abs() < 1e-10); // A[2,3] in column 3

        // Check sub-diagonal: A[i+1,i] at row_in_band = 3
        // Note: A[i,j] is in column j of band storage
        assert!((ab[3] - (-1.0)).abs() < 1e-10); // A[1,0] in column 0
        assert!((ab[3 + ldab] - (-1.0)).abs() < 1e-10); // A[2,1] in column 1
        assert!((ab[3 + 2 * ldab] - (-1.0)).abs() < 1e-10); // A[3,2] in column 2
    }

    #[test]
    fn test_band_to_dense() {
        let n = 4;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a_orig: Vec<f64> = vec![
            2.0, -1.0, 0.0, 0.0,
            -1.0, 2.0, -1.0, 0.0,
            0.0, -1.0, 2.0, -1.0,
            0.0, 0.0, -1.0, 2.0,
        ];

        let ab = dense_to_band(&a_orig, n, kl, ku);
        let a_back = band_to_dense(&ab, n, kl, ku);

        for i in 0..n * n {
            assert!(
                (a_orig[i] - a_back[i]).abs() < 1e-10,
                "Mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_band_lu_tridiagonal() {
        // Tridiagonal matrix (kl=1, ku=1)
        // [4 -1  0  0]
        // [-1 4 -1  0]
        // [0 -1  4 -1]
        // [0  0 -1  4]
        let n = 4;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");

        // Solve Ax = b where b = [3, 2, 2, 3]
        // Expected solution: x = [1, 1, 1, 1]
        // Verify: 4-1+0+0=3, -1+4-1+0=2, 0-1+4-1=2, 0+0-1+4=3 ✓
        let b = vec![3.0, 2.0, 2.0, 3.0];
        let x = lu.solve(&b).expect("Should solve");

        for i in 0..n {
            assert!(
                (x[i] - 1.0).abs() < 1e-10,
                "x[{i}] = {}, expected 1.0",
                x[i]
            );
        }
    }

    #[test]
    fn test_band_lu_pentadiagonal() {
        // Pentadiagonal matrix (kl=2, ku=2)
        // [10 -1 -2  0  0]
        // [-1 10 -1 -2  0]
        // [-2 -1 10 -1 -2]
        // [ 0 -2 -1 10 -1]
        // [ 0  0 -2 -1 10]
        let n = 5;
        let kl = 2;
        let ku = 2;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            10.0, -1.0, -2.0,  0.0,  0.0,
            -1.0, 10.0, -1.0, -2.0,  0.0,
            -2.0, -1.0, 10.0, -1.0, -2.0,
             0.0, -2.0, -1.0, 10.0, -1.0,
             0.0,  0.0, -2.0, -1.0, 10.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");

        // Create a test RHS
        let b = vec![7.0, 6.0, 4.0, 6.0, 7.0];
        let x = lu.solve(&b).expect("Should solve");

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
    fn test_band_lu_solve_multiple() {
        let n = 4;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");

        // Two RHS vectors
        let nrhs = 2;
        #[rustfmt::skip]
        let b = vec![
            3.0, 1.0,  // row 0: b1=3, b2=1
            2.0, 2.0,  // row 1: b1=2, b2=2
            2.0, 2.0,  // row 2: b1=2, b2=2
            3.0, 1.0,  // row 3: b1=3, b2=1
        ];

        let x = lu.solve_multiple(&b, nrhs).expect("Should solve");

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
    fn test_band_lu_singular() {
        // Singular tridiagonal matrix (second row is linearly dependent)
        let n = 3;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            1.0, -1.0, 0.0,
            -1.0, 1.0, 0.0,  // Row 2 = -Row 1
            0.0, 0.0, 1.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let result = BandLu::<f64>::compute(n, kl, ku, &ab);

        assert!(result.is_err());
        match result {
            Err(BandLuError::Singular { index: _ }) => {}
            _ => panic!("Expected Singular error"),
        }
    }

    #[test]
    fn test_band_lu_asymmetric_bandwidth() {
        // Matrix with kl=1, ku=2
        // [10 -1 -2  0]
        // [-1 10 -1 -2]
        // [ 0 -1 10 -1]
        // [ 0  0 -1 10]
        let n = 4;
        let kl = 1;
        let ku = 2;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            10.0, -1.0, -2.0,  0.0,
            -1.0, 10.0, -1.0, -2.0,
             0.0, -1.0, 10.0, -1.0,
             0.0,  0.0, -1.0, 10.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");

        let b = vec![7.0, 6.0, 8.0, 9.0];
        let x = lu.solve(&b).expect("Should solve");

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
    fn test_band_lu_empty() {
        let result = BandLu::<f64>::compute(0, 0, 0, &[]);
        assert!(result.is_ok());
        let lu = result.unwrap();
        assert_eq!(lu.size(), 0);
    }

    #[test]
    fn test_band_lu_f32() {
        let n = 3;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f32> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");

        let b = vec![3.0f32, 2.0, 3.0];
        let x = lu.solve(&b).expect("Should solve");

        // Verify Ax ≈ b with f32 tolerance
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

    #[test]
    fn test_band_norm_1() {
        let n = 3;
        let kl = 1;
        let ku = 1;
        // Matrix: [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]
        // Column sums: |4|+|-1| = 5, |-1|+|4|+|-1| = 6, |-1|+|4| = 5
        // Max column sum = 6
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let norm = band_norm_1(&ab, n, kl, ku);

        assert!(
            (norm - 6.0).abs() < 1e-10,
            "norm_1 = {}, expected 6.0",
            norm
        );
    }

    #[test]
    fn test_band_norm_inf() {
        let n = 3;
        let kl = 1;
        let ku = 1;
        // Matrix: [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]
        // Row sums: |4|+|-1| = 5, |-1|+|4|+|-1| = 6, |-1|+|4| = 5
        // Max row sum = 6
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let norm = band_norm_inf(&ab, n, kl, ku);

        assert!(
            (norm - 6.0).abs() < 1e-10,
            "norm_inf = {}, expected 6.0",
            norm
        );
    }

    #[test]
    fn test_band_rcond() {
        let n = 4;
        let kl = 1;
        let ku = 1;
        // Well-conditioned tridiagonal matrix
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let anorm = band_norm_1(&ab, n, kl, ku);

        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");
        let rcond = lu.rcond(anorm);

        // rcond should be positive and at most 1
        assert!(rcond > 0.0, "rcond = {}, should be > 0", rcond);
        assert!(rcond <= 1.0, "rcond = {}, should be <= 1", rcond);

        // For a well-conditioned matrix, rcond should be reasonably large
        assert!(
            rcond > 0.1,
            "rcond = {}, matrix seems ill-conditioned",
            rcond
        );
    }

    #[test]
    fn test_band_condition_number_estimate() {
        let n = 3;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");
        let cond = lu.condition_number_estimate();

        // Condition number should be >= 1
        assert!(cond >= 1.0, "cond = {}, should be >= 1", cond);
    }

    #[test]
    fn test_band_solve_transpose() {
        let n = 3;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("Should not be singular");

        let b = vec![1.0, 2.0, 3.0];
        let x = lu.solve_transpose(&b).expect("Should solve transpose");

        // Verify A^T * x = b
        for i in 0..n {
            let mut atx_i = 0.0;
            for j in 0..n {
                atx_i += a[j * n + i] * x[j]; // A^T[i,j] = A[j,i]
            }
            assert!(
                (atx_i - b[i]).abs() < 1e-10,
                "A^T*x[{i}] = {atx_i}, expected {}",
                b[i]
            );
        }
    }
}
