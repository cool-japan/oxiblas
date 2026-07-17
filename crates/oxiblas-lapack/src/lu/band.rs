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

        // Copy to work array.
        let mut ab_work = ab.to_vec();

        // Zero the fill-in region.
        //
        // As partial pivoting interchanges rows, the factor U acquires up to `kl`
        // extra super-diagonals beyond the original `ku` (LAPACK calls the total
        // KV = KU + KL). Those extra super-diagonals occupy band rows `0..kl` of
        // the storage, which the *input* matrix never uses (`dense_to_band` leaves
        // them zero). LAPACK's DGBTF2 zeroes this region defensively before the
        // sweep so that any stale data a caller left in the fill rows cannot leak
        // into the factorization; we mirror that to match reference behaviour.
        for j in 0..n {
            for r in 0..kl {
                ab_work[r + j * ldab] = T::zero();
            }
        }

        let mut pivot = vec![0usize; n];

        // `ju` is the index of the right-most column reached so far by fill-in.
        // It is monotonically non-decreasing and bounds every trailing update.
        let mut ju = 0usize;

        // Gaussian elimination with partial pivoting, one column at a time
        // (LAPACK DGBTF2, the unblocked band factorization).
        for j in 0..n {
            // Number of sub-diagonal candidates below the diagonal of column j.
            let km = kl.min(n - 1 - j);

            // Find the pivot: the row in j..=j+km whose entry has largest magnitude.
            // `jp_rel` is the pivot row's offset relative to j (0 == the diagonal).
            let mut jp_rel = 0usize;
            let mut pivot_val = scalar_abs(ab_work[band_idx(ldab, kl, ku, j, j)]);
            for i in 1..=km {
                let val = scalar_abs(ab_work[band_idx(ldab, kl, ku, j + i, j)]);
                if val > pivot_val {
                    pivot_val = val;
                    jp_rel = i;
                }
            }

            // Record the absolute pivot row index (0-based) for `solve`.
            pivot[j] = j + jp_rel;

            // Singularity test. LAPACK reports an *exactly* zero pivot; we keep the
            // slightly stronger near-zero guard used across this crate so a solve
            // against a numerically singular band matrix fails loudly instead of
            // dividing by an (almost) zero pivot. A NaN pivot fails the `> tol`
            // comparison and is therefore reported/propagated, never clamped.
            let tol = scalar_epsilon::<T>()
                * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
            if pivot_val <= tol {
                return Err(BandLuError::Singular { index: j });
            }

            // Extend the fill-in frontier. Pulling up row `j+jp_rel` reaches
            // `ku + jp_rel` super-diagonals to the right of column j.
            ju = ju.max((j + ku + jp_rel).min(n - 1));

            // Apply the interchange ONLY to columns j..=ju (the current column and
            // the trailing sub-matrix). We deliberately do NOT permute the already
            // computed multipliers stored in columns < j: L's column k holds the
            // multipliers relative to the row order in effect immediately after
            // step k, and the outstanding permutation is replayed, interleaved,
            // at solve time. Retroactively swapping earlier L columns (the previous
            // implementation's bug) corrupts the factorization the moment any real
            // row swap occurs. This is the exact convention of LAPACK DGBTF2/DGBTRS.
            if jp_rel != 0 {
                for c in j..=ju {
                    let idx_j = band_idx(ldab, kl, ku, j, c);
                    let idx_p = band_idx(ldab, kl, ku, j + jp_rel, c);
                    ab_work.swap(idx_j, idx_p);
                }
            }

            if km > 0 {
                // Scale column j's sub-diagonal entries to form L's multipliers.
                let pivot_inv = T::one() / ab_work[band_idx(ldab, kl, ku, j, j)];
                for i in 1..=km {
                    let idx = band_idx(ldab, kl, ku, j + i, j);
                    ab_work[idx] = ab_work[idx] * pivot_inv;
                }

                // Rank-1 update of the trailing sub-matrix inside the band:
                // A[j+1..=j+km, c] -= L[j+1..=j+km, j] * U[j, c] for c in j+1..=ju.
                for c in (j + 1)..=ju {
                    let u_jc = ab_work[band_idx(ldab, kl, ku, j, c)];
                    if u_jc != T::zero() {
                        for i in 1..=km {
                            let l_ij = ab_work[band_idx(ldab, kl, ku, j + i, j)];
                            let idx = band_idx(ldab, kl, ku, j + i, c);
                            ab_work[idx] = ab_work[idx] - l_ij * u_jc;
                        }
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

        // Forward sweep: solve L y = P b (LAPACK DGBTRS, TRANS = 'N').
        //
        // The permutation is *interleaved* with the elimination, not applied up
        // front: the stored L factors as P(0) L(0) P(1) L(1) ... where each L(j) is
        // the rank-one multiplier column produced at step j and each P(j) is the
        // step-j interchange. Applying every interchange first and then doing a
        // single L sweep (the previous, buggy approach) reorders the unknowns
        // inconsistently with how the multipliers were recorded, giving silently
        // wrong results the moment a real row swap occurs. We therefore
        // swap-then-eliminate at each step, exactly mirroring `compute`.
        //
        // `LNOTI` in DGBTRS: the whole forward stage is skipped when kl == 0, since
        // an upper-triangular band matrix never pivots (pivot[j] == j for all j).
        if self.kl > 0 {
            for j in 0..self.n {
                let lm = self.kl.min(self.n - 1 - j);
                let p = self.pivot[j];
                if p != j {
                    x.swap(j, p);
                }
                let xj = x[j];
                for i in 1..=lm {
                    let l_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j + i, j)];
                    x[j + i] = x[j + i] - l_elem * xj;
                }
            }
        }

        // Back substitution: solve U x = y. U is upper triangular with kl+ku
        // super-diagonals (the pivoting fill-in), so column j reaches back to
        // row j-(kl+ku).
        let kmax = self.ku + self.kl;
        for j in (0..self.n).rev() {
            let diag = self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)];
            x[j] = x[j] / diag;
            let xj = x[j];
            for i in j.saturating_sub(kmax)..j {
                let u_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, i, j)];
                x[i] = x[i] - u_elem * xj;
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

        // Forward sweep: solve L Y = P B, interchanges interleaved with the
        // elimination (see `solve` for why an up-front permutation is incorrect).
        if self.kl > 0 {
            for j in 0..self.n {
                let lm = self.kl.min(self.n - 1 - j);
                let p = self.pivot[j];
                if p != j {
                    for col in 0..nrhs {
                        x.swap(j * ldb + col, p * ldb + col);
                    }
                }
                for i in 1..=lm {
                    let l_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j + i, j)];
                    for col in 0..nrhs {
                        x[(j + i) * ldb + col] =
                            x[(j + i) * ldb + col] - l_elem * x[j * ldb + col];
                    }
                }
            }
        }

        // Back substitution: solve U X = Y, one band-limited column sweep per RHS.
        let kmax = self.ku + self.kl;
        for j in (0..self.n).rev() {
            let diag = self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)];
            for col in 0..nrhs {
                x[j * ldb + col] = x[j * ldb + col] / diag;
            }
            for i in j.saturating_sub(kmax)..j {
                let u_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, i, j)];
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

        // Solve A^T x = b. Since P A = L U we have A^T = U^T L^T P, hence
        // x = P^T L^-T U^-T b (LAPACK DGBTRS, TRANS = 'T').

        // Step 1: forward solve U^T w = b. U^T is lower triangular with kl+ku
        // sub-diagonals; row j of U^T (column j of U) reaches down to row j+kl+ku.
        let kmax = self.ku + self.kl;
        for j in 0..self.n {
            let diag = self.ab[band_idx(self.ldab, self.kl, self.ku, j, j)];
            x[j] = x[j] / diag;
            let xj = x[j];
            let i_end = (j + kmax).min(self.n - 1);
            for i in (j + 1)..=i_end {
                // U^T[i, j] == U[j, i].
                let u_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j, i)];
                x[i] = x[i] - u_elem * xj;
            }
        }

        // Step 2: back solve L^T z = w with the interchanges interleaved in
        // reverse. At each step we first apply L(j)^T (a rank-one update folding
        // column j's multipliers back into x[j]) and THEN undo interchange P(j).
        // This mirrors, in reverse, the forward sweep of `solve`; deferring all
        // interchanges to a single pass at the end (the previous approach) is
        // inconsistent with the deferred-permutation storage and silently wrong
        // under pivoting. Skipped entirely when kl == 0 (no sub-diagonals, so L is
        // the identity and no interchanges were recorded).
        if self.kl > 0 {
            for j in (0..self.n).rev() {
                let lm = self.kl.min(self.n - 1 - j);
                let mut acc = x[j];
                for i in 1..=lm {
                    let l_elem = self.ab[band_idx(self.ldab, self.kl, self.ku, j + i, j)];
                    acc = acc - l_elem * x[j + i];
                }
                x[j] = acc;
                let p = self.pivot[j];
                if p != j {
                    x.swap(j, p);
                }
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

    /// Reconstructs the dense matrix `A` from a factored `BandLu` as `A = P L U`.
    ///
    /// The banded LAPACK factorization stores `L = P(0) L(0) P(1) L(1) ...`, i.e.
    /// the interchanges are *deferred* rather than applied to the earlier L columns,
    /// so `A = [P(0)(I + l_0 e_0^T)] [P(1)(I + l_1 e_1^T)] ... U`. We replay that
    /// product right-to-left: starting from dense `U`, then for `j = n-2 .. 0` apply
    /// the rank-one `L(j)` (a row-add) followed by the interchange `P(j)` (a
    /// row-swap). This is the exact algebraic inverse of the elimination and equals
    /// the original `A` only when `compute` stored the multipliers WITHOUT
    /// retroactively permuting earlier columns — precisely the invariant guarded here.
    fn reconstruct_plu(lu: &BandLu<f64>) -> Vec<f64> {
        let n = lu.size();
        let kl = lu.kl();
        let ku = lu.ku();
        let ldab = 2 * kl + ku + 1;
        let ab = lu.ab();
        let pivot = lu.pivot();

        // Dense U (row-major): U[i,j] for i <= j and j-i <= kl+ku.
        let mut x = vec![0.0f64; n * n];
        for j in 0..n {
            let i_start = j.saturating_sub(kl + ku);
            for i in i_start..=j {
                x[i * n + j] = ab[band_idx(ldab, kl, ku, i, j)];
            }
        }

        // Apply F_j = P_j (I + l_j e_j^T) for j = n-2 down to 0.
        for j in (0..n.saturating_sub(1)).rev() {
            let km = kl.min(n - 1 - j);
            // (I + l_j e_j^T): row (j+i) += L[j+i, j] * row j.
            for i in 1..=km {
                let l = ab[band_idx(ldab, kl, ku, j + i, j)];
                for c in 0..n {
                    x[(j + i) * n + c] += l * x[j * n + c];
                }
            }
            // P_j: swap rows j and pivot[j].
            let p = pivot[j];
            if p != j {
                for c in 0..n {
                    x.swap(j * n + c, p * n + c);
                }
            }
        }
        x
    }

    #[test]
    fn test_band_lu_pivot_swap_tridiagonal() {
        // Tridiagonal matrix DELIBERATELY built so partial pivoting MUST swap:
        // in column 0 the sub-diagonal |3| dominates the diagonal |1|, and every
        // subsequent column likewise has a dominant sub-diagonal, forcing a genuine
        // interchange at essentially every step (pivot ends up [1, 2, 3, 4, 4]).
        //   [1 2 0 0 0]
        //   [3 1 2 0 0]
        //   [0 4 1 2 0]
        //   [0 0 5 1 2]
        //   [0 0 0 6 1]
        // A random banded matrix usually would NOT need a swap, so this hand-picked
        // matrix is what makes the regression actually exercise the pivot path.
        let n = 5;
        let kl = 1;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            1.0, 2.0, 0.0, 0.0, 0.0,
            3.0, 1.0, 2.0, 0.0, 0.0,
            0.0, 4.0, 1.0, 2.0, 0.0,
            0.0, 0.0, 5.0, 1.0, 2.0,
            0.0, 0.0, 0.0, 6.0, 1.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("nonsingular");

        // The whole point: at least one genuine row interchange must have happened,
        // otherwise this test would not exercise the previously-buggy code path.
        assert!(
            lu.pivot().iter().enumerate().any(|(j, &p)| p != j),
            "expected a real pivot swap, got pivot = {:?}",
            lu.pivot()
        );

        // (1) A = P L U reconstruction must match the original matrix.
        let a_rec = reconstruct_plu(&lu);
        for i in 0..n * n {
            assert!(
                (a_rec[i] - a[i]).abs() < 1e-12,
                "reconstruction mismatch at {i}: got {}, expected {}",
                a_rec[i],
                a[i]
            );
        }

        // (2) solve() must return the exact solution for this pivoting matrix.
        let x_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i * n + j] * x_true[j];
            }
        }
        let x = lu.solve(&b).expect("solve");
        for i in 0..n {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-10,
                "solve x[{i}] = {}, expected {}",
                x[i],
                x_true[i]
            );
        }

        // (3) solve_transpose() must also be correct under pivoting.
        let mut bt = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                bt[i] += a[j * n + i] * x_true[j]; // (A^T x_true)_i = sum_j A[j,i] x_j
            }
        }
        let xt = lu.solve_transpose(&bt).expect("solve_transpose");
        for i in 0..n {
            assert!(
                (xt[i] - x_true[i]).abs() < 1e-10,
                "solve_transpose x[{i}] = {}, expected {}",
                xt[i],
                x_true[i]
            );
        }

        // (4) solve_multiple() must agree with solve() column-by-column.
        let nrhs = 2;
        let mut bm = vec![0.0f64; n * nrhs];
        for i in 0..n {
            bm[i * nrhs] = b[i];
            bm[i * nrhs + 1] = 2.0 * b[i];
        }
        let xm = lu.solve_multiple(&bm, nrhs).expect("solve_multiple");
        for i in 0..n {
            assert!(
                (xm[i * nrhs] - x_true[i]).abs() < 1e-10,
                "solve_multiple rhs0 x[{i}] = {}, expected {}",
                xm[i * nrhs],
                x_true[i]
            );
            assert!(
                (xm[i * nrhs + 1] - 2.0 * x_true[i]).abs() < 1e-10,
                "solve_multiple rhs1 x[{i}] = {}, expected {}",
                xm[i * nrhs + 1],
                2.0 * x_true[i]
            );
        }
    }

    #[test]
    fn test_band_lu_pivot_swap_wide_band() {
        // kl=2, ku=1. In column 0 the candidate magnitudes are |1|, |3|, |6|, so the
        // pivot lands TWO rows below the diagonal (jp_rel = 2 = kl): this exercises
        // the maximum-distance interchange AND the resulting super-diagonal fill-in
        // in U (kv = kl+ku = 3), the widest path through the band factorization.
        //   [1 2 0 0 0]
        //   [3 1 2 0 0]
        //   [6 4 1 2 0]
        //   [0 7 5 1 2]
        //   [0 0 8 6 1]
        let n = 5;
        let kl = 2;
        let ku = 1;
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            1.0, 2.0, 0.0, 0.0, 0.0,
            3.0, 1.0, 2.0, 0.0, 0.0,
            6.0, 4.0, 1.0, 2.0, 0.0,
            0.0, 7.0, 5.0, 1.0, 2.0,
            0.0, 0.0, 8.0, 6.0, 1.0,
        ];

        let ab = dense_to_band(&a, n, kl, ku);
        let lu = BandLu::compute(n, kl, ku, &ab).expect("nonsingular");

        assert_eq!(lu.pivot()[0], 2, "column 0 must pivot two rows down");
        assert!(
            lu.pivot().iter().enumerate().any(|(j, &p)| p != j),
            "expected a real pivot swap, got pivot = {:?}",
            lu.pivot()
        );

        // A = P L U reconstruction over the wider band (with fill-in) must match.
        let a_rec = reconstruct_plu(&lu);
        for i in 0..n * n {
            assert!(
                (a_rec[i] - a[i]).abs() < 1e-12,
                "reconstruction mismatch at {i}: got {}, expected {}",
                a_rec[i],
                a[i]
            );
        }

        // solve() with a non-trivial (mixed-sign, fractional) exact solution.
        let x_true = [2.0, -1.0, 3.0, 0.5, -4.0];
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i * n + j] * x_true[j];
            }
        }
        let x = lu.solve(&b).expect("solve");
        for i in 0..n {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-10,
                "solve x[{i}] = {}, expected {}",
                x[i],
                x_true[i]
            );
        }

        // Transpose consistency under the wide-band pivoting as well.
        let mut bt = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                bt[i] += a[j * n + i] * x_true[j];
            }
        }
        let xt = lu.solve_transpose(&bt).expect("solve_transpose");
        for i in 0..n {
            assert!(
                (xt[i] - x_true[i]).abs() < 1e-10,
                "solve_transpose x[{i}] = {}, expected {}",
                xt[i],
                x_true[i]
            );
        }
    }
}
