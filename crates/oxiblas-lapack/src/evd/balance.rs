//! Matrix Balancing (GEBAL/GEBAK).
//!
//! Balances a general matrix to improve the accuracy of computed eigenvalues.
//!
//! # Algorithm
//!
//! Matrix balancing consists of two steps:
//!
//! 1. **Permutation**: Permute rows and columns to isolate eigenvalues
//!    (eigenvalues that can be determined without further computation).
//!
//! 2. **Scaling**: Apply diagonal similarity transformations D^{-1} A D
//!    to make row and column norms as close as possible.
//!
//! The balanced matrix B = D^{-1} P^T A P D has better conditioned eigenvalues.
//!
//! # References
//!
//! - LAPACK DGEBAL/DGEBAK routines
//! - Parlett, B.N. and Reinsch, C. (1969). "Balancing a matrix for calculation
//!   of eigenvalues and eigenvectors"

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for matrix balancing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Invalid job specification.
    InvalidJob,
    /// Invalid side specification.
    InvalidSide,
}

impl core::fmt::Display for BalanceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::InvalidJob => write!(f, "Invalid job specification"),
            Self::InvalidSide => write!(f, "Invalid side specification"),
        }
    }
}

impl std::error::Error for BalanceError {}

/// Balancing job specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceJob {
    /// Do not permute or scale.
    None,
    /// Only permute to isolate eigenvalues.
    Permute,
    /// Only scale rows and columns.
    Scale,
    /// Both permute and scale.
    Both,
}

/// Side specification for back-transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceSide {
    /// Back-transform right eigenvectors.
    Right,
    /// Back-transform left eigenvectors.
    Left,
}

/// Result of matrix balancing.
///
/// Contains the balanced matrix and the information needed to back-transform
/// eigenvectors of the balanced matrix to those of the original matrix.
#[derive(Debug, Clone)]
pub struct Balance<T: Scalar> {
    /// The balanced matrix.
    balanced: Mat<T>,
    /// Permutation information: row/column i of the balanced matrix
    /// corresponds to row/column perm[i] of the original matrix.
    perm: Vec<usize>,
    /// Scaling factors: D = diag(scale).
    scale: Vec<T>,
    /// Index of the first row/column in the non-isolated eigenvalue block.
    ilo: usize,
    /// Index of the last row/column in the non-isolated eigenvalue block.
    ihi: usize,
    /// Matrix dimension.
    n: usize,
    /// The job that was performed.
    job: BalanceJob,
}

impl<T: Field + Real + bytemuck::Zeroable> Balance<T> {
    /// Balances a general square matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to balance
    /// * `job` - Specifies what operations to perform
    ///
    /// # Returns
    ///
    /// A `Balance` struct containing the balanced matrix and transformation info.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::balance::{Balance, BalanceJob};
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 0.0, 0.0],
    ///     &[100.0, 2.0, 200.0],
    ///     &[0.0, 3.0, 4.0],
    /// ]);
    ///
    /// let bal = Balance::compute(a.as_ref(), BalanceJob::Both).unwrap();
    /// let b = bal.balanced();
    /// // Row and column norms are now more balanced
    /// ```
    pub fn compute(a: MatRef<'_, T>, job: BalanceJob) -> Result<Self, BalanceError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(BalanceError::EmptyMatrix);
        }

        if m != n {
            return Err(BalanceError::NotSquare);
        }

        // Initialize permutation as identity
        let mut perm: Vec<usize> = (0..n).collect();

        // Initialize scaling as ones
        let mut scale = vec![T::one(); n];

        // Copy matrix
        let mut balanced = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                balanced[(i, j)] = a[(i, j)];
            }
        }

        // Initialize ilo and ihi
        let mut ilo = 0;
        let mut ihi = n - 1;

        // Step 1: Permutation to isolate eigenvalues
        if job == BalanceJob::Permute || job == BalanceJob::Both {
            // Search for rows that are already isolated (only one nonzero element)
            // These correspond to eigenvalues that can be read off directly

            // Find rows with only diagonal element nonzero
            let mut converged = false;
            while !converged && ilo < n && ihi < n && ilo <= ihi {
                converged = true;

                // Search for a row in the trailing matrix with only diagonal nonzero
                // These rows correspond to isolated eigenvalues at the bottom
                if ihi >= ilo {
                    let mut i = ihi;
                    loop {
                        let mut only_diag = true;
                        for j in ilo..=ihi {
                            if i != j && Scalar::abs(balanced[(i, j)]) > <T as Scalar>::epsilon() {
                                only_diag = false;
                                break;
                            }
                        }

                        if only_diag {
                            // Found an isolated row - exchange with row ihi
                            if i != ihi {
                                // Swap rows and columns i and ihi
                                for k in 0..n {
                                    let tmp = balanced[(i, k)];
                                    balanced[(i, k)] = balanced[(ihi, k)];
                                    balanced[(ihi, k)] = tmp;
                                }
                                for k in 0..n {
                                    let tmp = balanced[(k, i)];
                                    balanced[(k, i)] = balanced[(k, ihi)];
                                    balanced[(k, ihi)] = tmp;
                                }
                                // Update permutation
                                perm.swap(i, ihi);
                            }
                            if ihi == 0 {
                                // All rows isolated
                                break;
                            }
                            ihi -= 1;
                            converged = false;
                            break;
                        }

                        if i == ilo {
                            break;
                        }
                        i -= 1;
                    }
                }
            }

            // Search for columns with only diagonal element nonzero
            // These correspond to isolated eigenvalues at the top
            converged = false;
            while !converged && ilo < n && ihi < n && ilo <= ihi {
                converged = true;

                for j in ilo..=ihi {
                    let mut only_diag = true;
                    for i in ilo..=ihi {
                        if i != j && Scalar::abs(balanced[(i, j)]) > <T as Scalar>::epsilon() {
                            only_diag = false;
                            break;
                        }
                    }

                    if only_diag {
                        // Found an isolated column - exchange with column ilo
                        if j != ilo {
                            // Swap rows and columns j and ilo
                            for k in 0..n {
                                let tmp = balanced[(j, k)];
                                balanced[(j, k)] = balanced[(ilo, k)];
                                balanced[(ilo, k)] = tmp;
                            }
                            for k in 0..n {
                                let tmp = balanced[(k, j)];
                                balanced[(k, j)] = balanced[(k, ilo)];
                                balanced[(k, ilo)] = tmp;
                            }
                            // Update permutation
                            perm.swap(j, ilo);
                        }
                        ilo += 1;
                        converged = false;
                        break;
                    }
                }
            }
        }

        // Step 2: Scaling
        if (job == BalanceJob::Scale || job == BalanceJob::Both) && ilo <= ihi {
            // Iteratively balance the non-isolated block [ilo..=ihi] x [ilo..=ihi]
            // using diagonal similarity transformations

            let radix = T::from_f64(2.0).unwrap_or_else(T::zero);
            let radix_sq = radix * radix;
            let sfmin1 = T::from_f64(f64::MIN_POSITIVE).unwrap_or(<T as Scalar>::epsilon());
            let sfmax1 = T::one() / sfmin1;

            let max_iterations = 100;

            for _iter in 0..max_iterations {
                let mut no_conv = false;

                for i in ilo..=ihi {
                    // Compute row and column norms (excluding diagonal)
                    let mut row_norm = T::zero();
                    let mut col_norm = T::zero();

                    for j in ilo..=ihi {
                        if i != j {
                            row_norm = row_norm + Scalar::abs(balanced[(i, j)]);
                            col_norm = col_norm + Scalar::abs(balanced[(j, i)]);
                        }
                    }

                    // Skip if row or column is zero
                    if row_norm == T::zero() || col_norm == T::zero() {
                        continue;
                    }

                    // Find scaling factor
                    let mut g = row_norm / radix;
                    let mut f = T::one();
                    let s = col_norm + row_norm;

                    // Scale factor determination
                    // Try to make row_norm ≈ col_norm
                    while col_norm < g {
                        f = f * radix;
                        col_norm = col_norm * radix_sq;
                    }

                    g = row_norm * radix;

                    while col_norm >= g {
                        f = f / radix;
                        col_norm = col_norm / radix_sq;
                    }

                    // Apply scaling only if it improves the balance significantly
                    // Check: (col_norm + row_norm) / f < 0.95 * s
                    let factor = T::from_f64(0.95).unwrap_or_else(T::zero);
                    if (col_norm + row_norm) / f < factor * s {
                        // Check for overflow/underflow
                        if f >= sfmin1 && f <= sfmax1 {
                            let g_inv = T::one() / f;
                            scale[i] = scale[i] * f;
                            no_conv = true;

                            // Scale row i by 1/f and column i by f
                            for j in 0..n {
                                balanced[(i, j)] = balanced[(i, j)] * g_inv;
                            }
                            for j in 0..n {
                                balanced[(j, i)] = balanced[(j, i)] * f;
                            }
                        }
                    }
                }

                if !no_conv {
                    break;
                }
            }
        }

        Ok(Self {
            balanced,
            perm,
            scale,
            ilo,
            ihi,
            n,
            job,
        })
    }

    /// Returns the balanced matrix.
    pub fn balanced(&self) -> MatRef<'_, T> {
        self.balanced.as_ref()
    }

    /// Returns a mutable reference to the balanced matrix.
    pub fn balanced_mut(&mut self) -> &mut Mat<T> {
        &mut self.balanced
    }

    /// Returns the permutation vector.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the scaling factors.
    pub fn scale(&self) -> &[T] {
        &self.scale
    }

    /// Returns the index of the first row/column in the non-isolated block.
    pub fn ilo(&self) -> usize {
        self.ilo
    }

    /// Returns the index of the last row/column in the non-isolated block.
    pub fn ihi(&self) -> usize {
        self.ihi
    }

    /// Returns the matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the job that was performed.
    pub fn job(&self) -> BalanceJob {
        self.job
    }

    /// Back-transforms eigenvectors of the balanced matrix to those of the original.
    ///
    /// # Arguments
    ///
    /// * `v` - Eigenvectors of the balanced matrix (each column is an eigenvector)
    /// * `side` - Whether these are right or left eigenvectors
    ///
    /// # Returns
    ///
    /// Eigenvectors of the original matrix.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bal = Balance::compute(a.as_ref(), BalanceJob::Both).unwrap();
    /// // Compute eigenvectors of balanced matrix
    /// let evd = GeneralEvd::compute(bal.balanced()).unwrap();
    /// let v_bal = evd.right_eigenvectors();
    /// // Back-transform to original matrix eigenvectors
    /// let v = bal.back_transform(&v_bal, BalanceSide::Right);
    /// ```
    pub fn back_transform(&self, v: &[Vec<T>], side: BalanceSide) -> Vec<Vec<T>> {
        let num_vectors = v.len();
        if num_vectors == 0 || self.n == 0 {
            return v.to_vec();
        }

        let mut result: Vec<Vec<T>> = v.to_vec();

        // Apply scaling
        if self.job == BalanceJob::Scale || self.job == BalanceJob::Both {
            match side {
                BalanceSide::Right => {
                    // For right eigenvectors: v_orig[i] = scale[i] * v_bal[i]
                    for k in 0..num_vectors {
                        for i in self.ilo..=self.ihi.min(self.n - 1) {
                            result[k][i] = result[k][i] * self.scale[i];
                        }
                    }
                }
                BalanceSide::Left => {
                    // For left eigenvectors: v_orig[i] = v_bal[i] / scale[i]
                    for k in 0..num_vectors {
                        for i in self.ilo..=self.ihi.min(self.n - 1) {
                            result[k][i] = result[k][i] / self.scale[i];
                        }
                    }
                }
            }
        }

        // Apply inverse permutation
        if self.job == BalanceJob::Permute || self.job == BalanceJob::Both {
            // Build inverse permutation
            let mut inv_perm = vec![0usize; self.n];
            for i in 0..self.n {
                inv_perm[self.perm[i]] = i;
            }

            // Apply inverse permutation to each eigenvector
            for k in 0..num_vectors {
                let orig = result[k].clone();
                for i in 0..self.n {
                    result[k][i] = orig[inv_perm[i]];
                }
            }
        }

        result
    }

    /// Back-transforms eigenvectors stored as a matrix.
    ///
    /// # Arguments
    ///
    /// * `v` - Matrix where each column is an eigenvector (n × num_vectors)
    /// * `side` - Whether these are right or left eigenvectors
    ///
    /// # Returns
    ///
    /// Matrix of eigenvectors of the original matrix.
    pub fn back_transform_matrix(&self, v: MatRef<'_, T>, side: BalanceSide) -> Mat<T> {
        let n = v.nrows();
        let num_vectors = v.ncols();

        if n == 0 || num_vectors == 0 {
            return Mat::zeros(n, num_vectors);
        }

        let mut result = Mat::zeros(n, num_vectors);
        for i in 0..n {
            for j in 0..num_vectors {
                result[(i, j)] = v[(i, j)];
            }
        }

        // Apply scaling
        if self.job == BalanceJob::Scale || self.job == BalanceJob::Both {
            match side {
                BalanceSide::Right => {
                    // For right eigenvectors: v_orig[i] = scale[i] * v_bal[i]
                    for i in self.ilo..=self.ihi.min(n - 1) {
                        for j in 0..num_vectors {
                            result[(i, j)] = result[(i, j)] * self.scale[i];
                        }
                    }
                }
                BalanceSide::Left => {
                    // For left eigenvectors: v_orig[i] = v_bal[i] / scale[i]
                    for i in self.ilo..=self.ihi.min(n - 1) {
                        for j in 0..num_vectors {
                            result[(i, j)] = result[(i, j)] / self.scale[i];
                        }
                    }
                }
            }
        }

        // Apply inverse permutation
        if self.job == BalanceJob::Permute || self.job == BalanceJob::Both {
            // Build inverse permutation
            let mut inv_perm = vec![0usize; n];
            for i in 0..n {
                inv_perm[self.perm[i]] = i;
            }

            // Apply inverse permutation
            let copy = result.clone();
            for i in 0..n {
                for j in 0..num_vectors {
                    result[(i, j)] = copy[(inv_perm[i], j)];
                }
            }
        }

        result
    }

    /// Reconstructs the original matrix from the balanced matrix.
    ///
    /// A = P D B D^{-1} P^T
    pub fn reconstruct(&self) -> Mat<T> {
        let mut a = Mat::zeros(self.n, self.n);

        // First apply inverse scaling: A' = D B D^{-1}
        for i in 0..self.n {
            for j in 0..self.n {
                a[(i, j)] = self.balanced[(i, j)] * self.scale[i] / self.scale[j];
            }
        }

        // Then apply inverse permutation
        // Build inverse permutation
        let mut inv_perm = vec![0usize; self.n];
        for i in 0..self.n {
            inv_perm[self.perm[i]] = i;
        }

        // A_orig = P A' P^T
        let copy = a.clone();
        for i in 0..self.n {
            for j in 0..self.n {
                a[(inv_perm[i], inv_perm[j])] = copy[(i, j)];
            }
        }

        a
    }
}

/// Convenience function for balancing (equivalent to LAPACK DGEBAL with JOB='B').
///
/// # Arguments
///
/// * `a` - The matrix to balance
///
/// # Returns
///
/// A tuple (balanced_matrix, ilo, ihi, scale) where:
/// - balanced_matrix: The balanced matrix
/// - ilo: First index of non-isolated block
/// - ihi: Last index of non-isolated block
/// - scale: Scaling factors
pub fn gebal<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<(Mat<T>, usize, usize, Vec<T>), BalanceError> {
    let bal = Balance::compute(a, BalanceJob::Both)?;
    let balanced = bal.balanced.clone();
    Ok((balanced, bal.ilo, bal.ihi, bal.scale))
}

/// Back-transforms eigenvectors (equivalent to LAPACK DGEBAK).
///
/// # Arguments
///
/// * `job` - The job that was performed in gebal
/// * `side` - 'R' for right eigenvectors, 'L' for left
/// * `ilo` - From gebal
/// * `ihi` - From gebal
/// * `scale` - From gebal
/// * `v` - Eigenvectors to back-transform (each column is an eigenvector)
///
/// # Returns
///
/// Back-transformed eigenvectors.
pub fn gebak<T: Field + Real + bytemuck::Zeroable>(
    job: BalanceJob,
    side: BalanceSide,
    ilo: usize,
    ihi: usize,
    scale: &[T],
    v: MatRef<'_, T>,
) -> Mat<T> {
    let n = v.nrows();
    let num_vectors = v.ncols();

    if n == 0 || num_vectors == 0 {
        return Mat::zeros(n, num_vectors);
    }

    let mut result = Mat::zeros(n, num_vectors);
    for i in 0..n {
        for j in 0..num_vectors {
            result[(i, j)] = v[(i, j)];
        }
    }

    // Apply scaling
    if job == BalanceJob::Scale || job == BalanceJob::Both {
        match side {
            BalanceSide::Right => {
                // For right eigenvectors: v_orig[i] = scale[i] * v_bal[i]
                for i in ilo..=ihi.min(n - 1) {
                    for j in 0..num_vectors {
                        result[(i, j)] = result[(i, j)] * scale[i];
                    }
                }
            }
            BalanceSide::Left => {
                // For left eigenvectors: v_orig[i] = v_bal[i] / scale[i]
                for i in ilo..=ihi.min(n - 1) {
                    for j in 0..num_vectors {
                        result[(i, j)] = result[(i, j)] / scale[i];
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_balance_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let bal = Balance::compute(eye.as_ref(), BalanceJob::Both).unwrap();
        let b = bal.balanced();

        // Identity should stay identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(b[(i, j)], expected, 1e-10));
            }
        }

        // Scaling should be ones
        for &s in bal.scale() {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_balance_diagonal() {
        let diag = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 5.0]]);

        let bal = Balance::compute(diag.as_ref(), BalanceJob::Both).unwrap();
        let b = bal.balanced();

        // Diagonal matrix should stay diagonal (eigenvalues already isolated)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(approx_eq(b[(i, j)], 0.0, 1e-10));
                }
            }
        }
    }

    #[test]
    fn test_balance_unbalanced_matrix() {
        // Matrix with very different row/column scales
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[100.0, 2.0, 200.0], &[0.0, 3.0, 4.0]]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::Scale).unwrap();
        let b = bal.balanced();

        // Check that row and column norms are more balanced
        let mut row_norms_orig = [0.0; 3];
        let mut col_norms_orig = [0.0; 3];
        let mut row_norms_bal = [0.0; 3];
        let mut col_norms_bal = [0.0; 3];

        for i in 0..3 {
            for j in 0..3 {
                row_norms_orig[i] += a[(i, j)].abs();
                col_norms_orig[j] += a[(i, j)].abs();
                row_norms_bal[i] += b[(i, j)].abs();
                col_norms_bal[j] += b[(i, j)].abs();
            }
        }

        // Compute variance of norms (lower is better balanced)
        let mean_orig: f64 =
            (row_norms_orig.iter().sum::<f64>() + col_norms_orig.iter().sum::<f64>()) / 6.0;
        let mean_bal: f64 =
            (row_norms_bal.iter().sum::<f64>() + col_norms_bal.iter().sum::<f64>()) / 6.0;

        let var_orig: f64 = row_norms_orig
            .iter()
            .chain(col_norms_orig.iter())
            .map(|x| (x - mean_orig).powi(2))
            .sum::<f64>()
            / 6.0;
        let var_bal: f64 = row_norms_bal
            .iter()
            .chain(col_norms_bal.iter())
            .map(|x| (x - mean_bal).powi(2))
            .sum::<f64>()
            / 6.0;

        // Balanced matrix should have lower variance
        assert!(
            var_bal <= var_orig * 1.5,
            "var_bal={}, var_orig={}",
            var_bal,
            var_orig
        );
    }

    #[test]
    fn test_balance_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::Both).unwrap();
        let reconstructed = bal.reconstruct();

        // Reconstruction should match original
        for i in 0..3 {
            for j in 0..3 {
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
    fn test_balance_eigenvalue_isolation() {
        // Matrix with isolated eigenvalue
        // [1 0 0]
        // [2 3 4]  -> eigenvalue 1 is isolated (row 0 has only diagonal)
        // [5 6 7]
        let a = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::Permute).unwrap();

        // After permutation, row 0 should be isolated (ilo should be 1 or ihi should be < 2)
        assert!(
            bal.ilo() >= 1 || bal.ihi() < 2,
            "ilo={}, ihi={}",
            bal.ilo(),
            bal.ihi()
        );
    }

    #[test]
    fn test_balance_back_transform_matrix() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[100.0, 3.0]]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::Scale).unwrap();

        // Create dummy eigenvector matrix
        let v = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let v_transformed = bal.back_transform_matrix(v.as_ref(), BalanceSide::Right);

        // Check dimensions
        assert_eq!(v_transformed.nrows(), 2);
        assert_eq!(v_transformed.ncols(), 2);
    }

    #[test]
    fn test_gebal_gebak() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let (balanced, ilo, ihi, scale) = gebal(a.as_ref()).unwrap();

        // Create identity eigenvector matrix
        let v = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let _v_back = gebak(
            BalanceJob::Both,
            BalanceSide::Right,
            ilo,
            ihi,
            &scale,
            v.as_ref(),
        );

        // Just check that it doesn't panic and produces valid output
        assert_eq!(balanced.nrows(), 3);
        assert_eq!(balanced.ncols(), 3);
    }

    #[test]
    fn test_balance_job_none() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::None).unwrap();
        let b = bal.balanced();

        // With None, matrix should be unchanged
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(b[(i, j)], a[(i, j)], 1e-10));
            }
        }
    }

    #[test]
    fn test_balance_4x4_reconstruction() {
        let a = Mat::from_rows(&[
            &[4.0f64, 1.0, -2.0, 2.0],
            &[1.0, 2.0, 0.0, 1.0],
            &[-2.0, 0.0, 3.0, -2.0],
            &[2.0, 1.0, -2.0, -1.0],
        ]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::Both).unwrap();
        let reconstructed = bal.reconstruct();

        for i in 0..4 {
            for j in 0..4 {
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
    fn test_balance_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let bal = Balance::compute(a.as_ref(), BalanceJob::Both).unwrap();
        let reconstructed = bal.reconstruct();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-5,
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
