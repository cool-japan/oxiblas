//! QR decomposition with column pivoting.
//!
//! Computes A·P = Q·R where:
//! - Q is orthogonal (Q^T·Q = I)
//! - R is upper triangular
//! - P is a permutation matrix (column permutation)
//!
//! Column pivoting provides a rank-revealing decomposition - the diagonal
//! elements of R are ordered by magnitude, making it useful for detecting
//! numerical rank.

use num_traits::{FromPrimitive, One};
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for QR decomposition with column pivoting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrPivotError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Dimension mismatch in solve operation.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for QrPivotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for QrPivotError {}

/// QR decomposition with column pivoting.
///
/// For an m×n matrix A, computes A·P = Q·R where:
/// - Q is m×m orthogonal
/// - R is m×n upper triangular
/// - P is an n×n permutation matrix
///
/// The column pivoting strategy selects at each step the column with
/// the largest norm in the remaining submatrix, ensuring that
/// |R\[k,k\]| >= |R\[k,j\]| for all j > k.
///
/// This makes the decomposition rank-revealing: if the matrix has
/// numerical rank r, then R\[r,r\] should be close to zero.
#[derive(Debug, Clone)]
pub struct QrPivot<T: Scalar> {
    /// QR factors (compact storage)
    qr: Mat<T>,
    /// Householder scalars
    tau: Vec<T>,
    /// Column permutation (col_perm[i] = original column index at position i)
    col_perm: Vec<usize>,
    /// Number of rows
    m: usize,
    /// Number of columns
    n: usize,
    /// Numerical rank (columns with |R\[k,k\]| > tolerance)
    rank: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> QrPivot<T> {
    /// Computes the QR decomposition with column pivoting.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::QrPivot;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    ///     &[7.0, 8.0, 9.0],  // Rank 2 (third row is linear combination)
    /// ]);
    ///
    /// let qr = QrPivot::compute(a.as_ref()).unwrap();
    /// // The decomposition reveals the rank
    /// let r = qr.r();
    /// // R[2,2] should be close to zero for this rank-deficient matrix
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, QrPivotError> {
        Self::compute_with_tol(a, None)
    }

    /// Computes the QR decomposition with column pivoting and custom tolerance.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to decompose
    /// * `tol` - Optional tolerance for determining numerical rank
    pub fn compute_with_tol(a: MatRef<'_, T>, tol: Option<T>) -> Result<Self, QrPivotError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(QrPivotError::EmptyMatrix);
        }

        // Copy A to working matrix
        let mut qr = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                qr[(i, j)] = a[(i, j)];
            }
        }

        let k = m.min(n);
        let mut tau = vec![T::zero(); k];
        let mut col_perm: Vec<usize> = (0..n).collect();

        // Initialize column norms squared
        let mut col_norms = vec![T::zero(); n];
        for j in 0..n {
            for i in 0..m {
                col_norms[j] = col_norms[j] + qr[(i, j)] * qr[(i, j)];
            }
        }

        // Base tolerance for rank determination (will be scaled by first pivot)
        let eps = <T as Scalar>::epsilon();
        let scale = <T as FromPrimitive>::from_usize(m.max(n)).unwrap_or(<T as One>::one());
        let base_tol = eps * scale;
        let user_tol = tol;

        let mut rank = k;
        let mut first_diag_abs = T::zero();

        // Apply Householder reflections with column pivoting
        for j in 0..k {
            // Find column with maximum norm among j..n
            let mut max_norm = col_norms[j];
            let mut max_col = j;
            for col in (j + 1)..n {
                if col_norms[col] > max_norm {
                    max_norm = col_norms[col];
                    max_col = col;
                }
            }

            // Swap columns if needed
            if max_col != j {
                // Swap in qr matrix
                for i in 0..m {
                    let tmp = qr[(i, j)];
                    qr[(i, j)] = qr[(i, max_col)];
                    qr[(i, max_col)] = tmp;
                }
                // Swap in permutation
                col_perm.swap(j, max_col);
                // Swap norms
                col_norms.swap(j, max_col);
            }

            // Compute Householder vector for column j
            let (tau_j, beta) = householder_vector(&mut qr, j, m);
            tau[j] = tau_j;

            // Update the diagonal element
            qr[(j, j)] = beta;

            // Store the first (largest) diagonal absolute value for relative tolerance
            let diag_abs = Scalar::abs(beta);
            if j == 0 {
                first_diag_abs = diag_abs;
            }

            // Check for numerical rank deficiency using relative tolerance
            // Use max(m,n) * eps * |R[0,0]| as default tolerance (LAPACK-style)
            let tolerance = user_tol.unwrap_or(base_tol * first_diag_abs);
            if diag_abs <= tolerance && j > 0 {
                rank = j;
                // Fill remaining tau with zeros
                for idx in j..k {
                    tau[idx] = T::zero();
                }
                break;
            }

            // Apply Householder reflection to trailing submatrix
            if j < n - 1 {
                apply_householder_left(&mut qr, j, m, n, tau_j);

                // Update column norms for columns j+1..n
                for col in (j + 1)..n {
                    // Subtract contribution from row j
                    let elem = qr[(j, col)];
                    col_norms[col] = col_norms[col] - elem * elem;
                    // Prevent negative due to rounding
                    if col_norms[col] < T::zero() {
                        col_norms[col] = T::zero();
                    }
                }
            }
        }

        Ok(Self {
            qr,
            tau,
            col_perm,
            m,
            n,
            rank,
        })
    }

    /// Returns the number of rows in the original matrix.
    pub fn nrows(&self) -> usize {
        self.m
    }

    /// Returns the number of columns in the original matrix.
    pub fn ncols(&self) -> usize {
        self.n
    }

    /// Returns the numerical rank.
    ///
    /// This is the number of columns processed before a near-zero pivot was found.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the column permutation.
    ///
    /// `col_perm[i]` gives the original column index that now appears at position i.
    pub fn column_permutation(&self) -> &[usize] {
        &self.col_perm
    }

    /// Extracts the R matrix (upper triangular).
    pub fn r(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut r = Mat::zeros(self.m, self.n);

        for j in 0..self.n {
            for i in 0..=j.min(self.m - 1) {
                r[(i, j)] = self.qr[(i, j)];
            }
        }

        // Zero out below diagonal
        for j in 0..k {
            for i in (j + 1)..self.m {
                r[(i, j)] = T::zero();
            }
        }

        r
    }

    /// Extracts the thin R matrix (k×n where k = min(m, n)).
    pub fn r_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut r = Mat::zeros(k, self.n);

        for j in 0..self.n {
            for i in 0..=j.min(k - 1) {
                r[(i, j)] = self.qr[(i, j)];
            }
        }

        r
    }

    /// Computes and returns the Q matrix (orthogonal).
    pub fn q(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        // Start with identity matrix
        let mut q = Mat::zeros(self.m, self.m);
        for i in 0..self.m {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order
        for j in (0..k).rev() {
            apply_householder_to_q(&mut q, &self.qr, j, self.m, self.tau[j]);
        }

        q
    }

    /// Computes and returns the thin Q matrix (m×k where k = min(m, n)).
    pub fn q_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        // Start with the first k columns of identity
        let mut q = Mat::zeros(self.m, k);
        for i in 0..k {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order
        for j in (0..k).rev() {
            apply_householder_to_q_thin(&mut q, &self.qr, j, self.m, k, self.tau[j]);
        }

        q
    }

    /// Constructs the permutation matrix P.
    ///
    /// P is such that A·P = Q·R, so A = Q·R·P^T.
    pub fn permutation_matrix(&self) -> Mat<T> {
        let mut p = Mat::zeros(self.n, self.n);
        for (i, &orig_col) in self.col_perm.iter().enumerate() {
            p[(orig_col, i)] = T::one();
        }
        p
    }

    /// Returns the absolute values of the R diagonal (pivot magnitudes).
    ///
    /// These are ordered in decreasing magnitude due to column pivoting.
    pub fn r_diagonal_abs(&self) -> Vec<T> {
        let k = self.m.min(self.n);
        (0..k).map(|i| Scalar::abs(self.qr[(i, i)])).collect()
    }

    /// Solves the least squares problem: min ||A·x - b||_2
    ///
    /// Returns x that minimizes the residual norm.
    pub fn solve_least_squares(&self, b: MatRef<'_, T>) -> Result<Mat<T>, QrPivotError> {
        if b.nrows() != self.m {
            return Err(QrPivotError::DimensionMismatch {
                expected: self.m,
                actual: b.nrows(),
            });
        }

        let nrhs = b.ncols();
        let k = self.m.min(self.n);

        // Copy b to working matrix
        let mut work = Mat::zeros(self.m, nrhs);
        for j in 0..nrhs {
            for i in 0..self.m {
                work[(i, j)] = b[(i, j)];
            }
        }

        // Apply Q^T to b: Q^T * b
        for j in 0..k {
            apply_householder_to_rhs(&mut work, &self.qr, j, self.m, nrhs, self.tau[j]);
        }

        // Back substitution: solve R * y = Q^T * b (only first k rows)
        let mut y = Mat::zeros(self.n, nrhs);

        let r_rank = self.rank;
        for col in 0..nrhs {
            for i in (0..r_rank).rev() {
                let mut sum = work[(i, col)];
                for j in (i + 1)..self.n.min(r_rank) {
                    sum = sum - self.qr[(i, j)] * y[(j, col)];
                }
                let diag = self.qr[(i, i)];
                if Scalar::abs(diag)
                    > <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one())
                {
                    y[(i, col)] = sum / diag;
                }
            }
        }

        // Apply inverse permutation: x = P * y
        let mut x = Mat::zeros(self.n, nrhs);
        for j in 0..nrhs {
            for (i, &orig_col) in self.col_perm.iter().enumerate() {
                x[(orig_col, j)] = y[(i, j)];
            }
        }

        Ok(x)
    }
}

/// Computes the Householder vector for column j.
fn householder_vector<T: Field + Real>(qr: &mut Mat<T>, j: usize, m: usize) -> (T, T) {
    // Compute the norm of the column below the diagonal
    let mut norm_sq = T::zero();
    for i in j..m {
        norm_sq = norm_sq + qr[(i, j)] * qr[(i, j)];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    // Compute beta = -sign(x[j]) * ||x||
    let x_j = qr[(j, j)];
    let beta = if x_j >= T::zero() { -norm } else { norm };

    // Compute tau = (beta - x[j]) / beta
    let tau = (beta - x_j) / beta;

    // Scale the Householder vector
    let scale = T::one() / (x_j - beta);
    for i in (j + 1)..m {
        qr[(i, j)] = qr[(i, j)] * scale;
    }

    (tau, beta)
}

/// Applies Householder reflection to trailing submatrix.
fn apply_householder_left<T: Field + Real>(qr: &mut Mat<T>, j: usize, m: usize, n: usize, tau: T) {
    if tau == T::zero() {
        return;
    }

    for k in (j + 1)..n {
        let mut w = qr[(j, k)];
        for i in (j + 1)..m {
            w = w + qr[(i, j)] * qr[(i, k)];
        }

        let tw = tau * w;
        qr[(j, k)] = qr[(j, k)] - tw;
        for i in (j + 1)..m {
            qr[(i, k)] = qr[(i, k)] - tw * qr[(i, j)];
        }
    }
}

/// Applies Householder reflection to Q matrix.
fn apply_householder_to_q<T: Field + Real>(
    q: &mut Mat<T>,
    qr: &Mat<T>,
    j: usize,
    m: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }

    for k in 0..m {
        let mut w = q[(j, k)];
        for i in (j + 1)..m {
            w = w + qr[(i, j)] * q[(i, k)];
        }

        let tw = tau * w;
        q[(j, k)] = q[(j, k)] - tw;
        for i in (j + 1)..m {
            q[(i, k)] = q[(i, k)] - tw * qr[(i, j)];
        }
    }
}

/// Applies Householder reflection to thin Q matrix.
fn apply_householder_to_q_thin<T: Field + Real>(
    q: &mut Mat<T>,
    qr: &Mat<T>,
    j: usize,
    m: usize,
    ncols: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }

    for k in 0..ncols {
        let mut w = q[(j, k)];
        for i in (j + 1)..m {
            w = w + qr[(i, j)] * q[(i, k)];
        }

        let tw = tau * w;
        q[(j, k)] = q[(j, k)] - tw;
        for i in (j + 1)..m {
            q[(i, k)] = q[(i, k)] - tw * qr[(i, j)];
        }
    }
}

/// Applies Householder reflection to RHS for solving.
fn apply_householder_to_rhs<T: Field + Real>(
    x: &mut Mat<T>,
    qr: &Mat<T>,
    j: usize,
    m: usize,
    nrhs: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }

    for k in 0..nrhs {
        let mut w = x[(j, k)];
        for i in (j + 1)..m {
            w = w + qr[(i, j)] * x[(i, k)];
        }

        let tw = tau * w;
        x[(j, k)] = x[(j, k)] - tw;
        for i in (j + 1)..m {
            x[(i, k)] = x[(i, k)] - tw * qr[(i, j)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_qr_pivot_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();
        let p = qr.permutation_matrix();

        // Verify Q is orthogonal
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-10));
            }
        }

        // Verify R is upper triangular
        assert!(approx_eq(r[(1, 0)], 0.0, 1e-10));
        assert!(approx_eq(r[(2, 0)], 0.0, 1e-10));
        assert!(approx_eq(r[(2, 1)], 0.0, 1e-10));

        // Verify A * P = Q * R
        // First compute Q * R
        let mut qr_prod: Mat<f64> = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    qr_prod[(i, j)] = qr_prod[(i, j)] + q[(i, k)] * r[(k, j)];
                }
            }
        }

        // Compute A * P
        let mut ap: Mat<f64> = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    ap[(i, j)] = ap[(i, j)] + a[(i, k)] * p[(k, j)];
                }
            }
        }

        // Check A * P = Q * R
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(ap[(i, j)], qr_prod[(i, j)], 1e-10),
                    "AP[{},{}] = {}, QR = {}",
                    i,
                    j,
                    ap[(i, j)],
                    qr_prod[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_pivot_rank_deficient() {
        // Rank 2 matrix (third row is sum of first two)
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[5.0, 7.0, 9.0], // = row1 + row2
        ]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();

        // The last diagonal of R should be very small
        let r_diag = qr.r_diagonal_abs();
        assert!(
            r_diag[2] < 1e-10,
            "R[2,2] = {} should be near zero",
            r_diag[2]
        );

        // Rank should be detected as 2
        assert_eq!(qr.rank(), 2, "Rank should be 2, got {}", qr.rank());
    }

    #[test]
    fn test_qr_pivot_ordering() {
        // Columns have different norms
        let a = Mat::from_rows(&[&[1.0f64, 10.0, 5.0], &[1.0, 10.0, 5.0], &[1.0, 10.0, 5.0]]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();

        // Column 1 (norm = sqrt(300)) should be selected first
        // Permutation should start with column 1
        let perm = qr.column_permutation();
        assert_eq!(
            perm[0], 1,
            "Column 1 should be first, got column {}",
            perm[0]
        );
    }

    #[test]
    fn test_qr_pivot_least_squares() {
        // Overdetermined system
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[2.0], &[2.5]]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();
        let x = qr.solve_least_squares(b.as_ref()).unwrap();

        // Verify Ax is close to b in least squares sense
        let mut ax = [0.0; 3];
        for i in 0..3 {
            for j in 0..2 {
                ax[i] += a[(i, j)] * x[(j, 0)];
            }
        }

        let mut residual = 0.0;
        for i in 0..3 {
            residual += (ax[i] - b[(i, 0)]).powi(2);
        }
        residual = residual.sqrt();
        assert!(residual < 0.5);
    }

    #[test]
    fn test_qr_pivot_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();
        let p = qr.permutation_matrix();

        assert_eq!(q.nrows(), 4);
        assert_eq!(q.ncols(), 4);
        assert_eq!(r.nrows(), 4);
        assert_eq!(r.ncols(), 2);

        // Verify A * P = Q * R
        let mut qr_prod: Mat<f64> = Mat::zeros(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                for k in 0..4 {
                    qr_prod[(i, j)] = qr_prod[(i, j)] + q[(i, k)] * r[(k, j)];
                }
            }
        }

        let mut ap: Mat<f64> = Mat::zeros(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                for k in 0..2 {
                    ap[(i, j)] = ap[(i, j)] + a[(i, k)] * p[(k, j)];
                }
            }
        }

        for i in 0..4 {
            for j in 0..2 {
                assert!(approx_eq(ap[(i, j)], qr_prod[(i, j)], 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_pivot_identity() {
        let eye: Mat<f64> = Mat::eye(3);

        let qr = QrPivot::compute(eye.as_ref()).unwrap();
        let r = qr.r();

        // Diagonal of R should have magnitude 1
        for i in 0..3 {
            assert!(r[(i, i)].abs() > 0.99);
        }
    }

    #[test]
    fn test_qr_pivot_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);

        let result = QrPivot::compute(a.as_ref());
        assert!(matches!(result, Err(QrPivotError::EmptyMatrix)));
    }

    #[test]
    fn test_qr_pivot_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();
        let p = qr.permutation_matrix();

        // Verify A * P = Q * R
        for i in 0..2 {
            for j in 0..2 {
                let mut qr_val: f32 = 0.0;
                for k in 0..2 {
                    qr_val += q[(i, k)] * r[(k, j)];
                }
                let mut ap_val: f32 = 0.0;
                for k in 0..2 {
                    ap_val += a[(i, k)] * p[(k, j)];
                }
                assert!((ap_val - qr_val).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_qr_pivot_thin() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = QrPivot::compute(a.as_ref()).unwrap();
        let q_thin = qr.q_thin();
        let r_thin = qr.r_thin();

        assert_eq!(q_thin.nrows(), 3);
        assert_eq!(q_thin.ncols(), 2);
        assert_eq!(r_thin.nrows(), 2);
        assert_eq!(r_thin.ncols(), 2);

        // Q_thin should have orthonormal columns
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q_thin[(k, i)] * q_thin[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-10));
            }
        }
    }
}
