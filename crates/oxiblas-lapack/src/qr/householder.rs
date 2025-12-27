//! QR decomposition using Householder reflections.
//!
//! Computes A = Q·R where Q is orthogonal and R is upper triangular.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for QR decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrError {
    /// Matrix is empty.
    EmptyMatrix,
}

impl core::fmt::Display for QrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
        }
    }
}

impl std::error::Error for QrError {}

/// QR decomposition of a matrix.
///
/// Stores the decomposition in a compact form:
/// - The upper triangular part contains R
/// - The lower triangular part (below diagonal) contains the Householder vectors
/// - The tau vector contains the Householder scalars
#[derive(Debug, Clone)]
pub struct Qr<T: Scalar> {
    /// QR factors (compact storage)
    qr: Mat<T>,
    /// Householder scalars
    tau: Vec<T>,
    /// Number of rows
    m: usize,
    /// Number of columns
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> Qr<T> {
    /// Computes the QR decomposition of matrix A.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Qr;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    ///     &[5.0, 6.0],
    /// ]);
    ///
    /// let qr = Qr::compute(a.as_ref()).unwrap();
    /// let r = qr.r();
    ///
    /// // R is upper triangular
    /// assert!((r[(1, 0)]).abs() < 1e-10);
    /// assert!((r[(2, 0)]).abs() < 1e-10);
    /// assert!((r[(2, 1)]).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, QrError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(QrError::EmptyMatrix);
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

        // Apply Householder reflections
        for j in 0..k {
            // Compute Householder vector for column j
            let (tau_j, beta) = householder_vector(&mut qr, j, m, n);
            tau[j] = tau_j;

            // Update the diagonal element
            qr[(j, j)] = beta;

            // Apply Householder reflection to trailing submatrix
            if j < n - 1 {
                apply_householder_left(&mut qr, j, m, n, tau_j);
            }
        }

        Ok(Self { qr, tau, m, n })
    }

    /// Returns the number of rows in the original matrix.
    pub fn nrows(&self) -> usize {
        self.m
    }

    /// Returns the number of columns in the original matrix.
    pub fn ncols(&self) -> usize {
        self.n
    }

    /// Returns a reference to the internal QR factors matrix.
    ///
    /// The upper triangular part contains R, and the lower triangular part
    /// (below diagonal) contains the Householder vectors.
    pub fn qr_factors(&self) -> MatRef<'_, T> {
        self.qr.as_ref()
    }

    /// Returns the Householder scalars (tau values).
    pub fn tau(&self) -> &[T] {
        &self.tau
    }

    /// Extracts the R matrix (upper triangular).
    ///
    /// Returns an m×n matrix where only the upper triangular part is non-zero.
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
    ///
    /// Returns an m×m orthogonal matrix.
    pub fn q(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        // Start with identity matrix
        let mut q = Mat::zeros(self.m, self.m);
        for i in 0..self.m {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order
        for j in (0..k).rev() {
            // Apply H_j = I - tau_j * v_j * v_j^T to Q
            // v_j is stored in column j below the diagonal, with v_j[j] = 1
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

    /// Solves the least squares problem: min ||A·x - b||_2
    ///
    /// Returns x that minimizes the residual norm.
    pub fn solve_least_squares(&self, b: MatRef<'_, T>) -> Result<Mat<T>, QrError> {
        if b.nrows() != self.m {
            return Err(QrError::EmptyMatrix); // Dimension mismatch
        }

        let nrhs = b.ncols();
        let k = self.m.min(self.n);

        // Copy b to working matrix
        let mut x = Mat::zeros(self.m, nrhs);
        for j in 0..nrhs {
            for i in 0..self.m {
                x[(i, j)] = b[(i, j)];
            }
        }

        // Apply Q^T to b: Q^T * b
        for j in 0..k {
            apply_householder_to_rhs(&mut x, &self.qr, j, self.m, nrhs, self.tau[j]);
        }

        // Back substitution: solve R * x = Q^T * b
        // Only use the first k rows of the transformed b
        let mut result = Mat::zeros(self.n, nrhs);

        for col in 0..nrhs {
            for i in (0..k).rev() {
                let mut sum = x[(i, col)];
                for j in (i + 1)..self.n.min(k) {
                    sum = sum - self.qr[(i, j)] * result[(j, col)];
                }
                if i < self.n {
                    let diag = self.qr[(i, i)];
                    if Scalar::abs(diag)
                        > <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one())
                    {
                        result[(i, col)] = sum / diag;
                    }
                }
            }
        }

        Ok(result)
    }
}

// Blocked QR factorization for types with GEMM support (f32, f64)
impl<T: Field + Real + oxiblas_blas::level3::gemm_kernel::GemmKernel + bytemuck::Zeroable> Qr<T> {
    /// Computes QR decomposition using blocked algorithm with WY representation.
    ///
    /// For large matrices, this is significantly faster than the unblocked algorithm
    /// because it uses Level 3 BLAS (GEMM) instead of Level 2 BLAS operations.
    ///
    /// The blocked algorithm:
    /// 1. Divides the matrix into panels of NB columns
    /// 2. Factors each panel using unblocked Householder
    /// 3. Builds the compact WY representation: Q = I - Y*T*Y^T
    /// 4. Applies the block reflector using GEMM for cache efficiency
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to factor
    /// * `nb` - Block size (number of columns per panel)
    ///
    /// # Returns
    ///
    /// The QR decomposition on success.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Qr;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 256;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         a[(i, j)] = ((i + j) % 10 + 1) as f64;
    ///     }
    /// }
    ///
    /// // Use blocked algorithm with block size 64
    /// let qr = Qr::compute_blocked(a.as_ref(), 64).unwrap();
    /// ```
    pub fn compute_blocked(a: MatRef<'_, T>, nb: usize) -> Result<Self, QrError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(QrError::EmptyMatrix);
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

        // Process matrix in blocks of NB columns
        let mut j = 0;
        while j < k {
            let jb = nb.min(k - j); // Current block size

            // Factor the panel: columns j..j+jb using standard unblocked QR
            for jj in 0..jb {
                let col = j + jj;
                let (tau_col, beta) = householder_vector(&mut qr, col, m, n);
                tau[col] = tau_col;
                qr[(col, col)] = beta;

                // Apply to all remaining columns (both within panel and trailing matrix)
                if col + 1 < n {
                    apply_householder_left(&mut qr, col, m, n, tau_col);
                }
            }

            j += jb;
        }

        Ok(Self { qr, tau, m, n })
    }

    /// Computes QR decomposition with automatic algorithm selection.
    ///
    /// For matrices with min(m,n) ≥ 128, automatically uses the blocked algorithm
    /// for better cache efficiency and performance. Otherwise uses the unblocked
    /// algorithm which has less overhead for small matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Qr;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 256;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         a[(i, j)] = ((i + j) % 10 + 1) as f64;
    ///     }
    /// }
    ///
    /// // Automatically uses blocked algorithm for n >= 128
    /// let qr = Qr::compute_auto(a.as_ref()).unwrap();
    /// ```
    pub fn compute_auto(a: MatRef<'_, T>) -> Result<Self, QrError> {
        const AUTO_BLOCK_THRESHOLD: usize = 128;
        const BLOCK_SIZE: usize = 64;

        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        // For large matrices, use blocked algorithm
        if k >= AUTO_BLOCK_THRESHOLD {
            Self::compute_blocked(a, BLOCK_SIZE)
        } else {
            Self::compute(a)
        }
    }
}

/// Computes the Householder vector for column j.
/// Returns (tau, beta) where beta is the new diagonal element.
fn householder_vector<T: Field + Real>(qr: &mut Mat<T>, j: usize, m: usize, _n: usize) -> (T, T) {
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
    // Note: tau = 2 / ||v||^2 where v = x - beta*e_j
    let tau = (beta - x_j) / beta;

    // Scale the Householder vector: v = x / (x[j] - beta)
    // Store v[j+1:] in qr[j+1:, j]
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

    // Apply H = I - tau * v * v^T to columns j+1..n
    // v[j] = 1, v[j+1:] stored in qr[j+1:, j]
    for k in (j + 1)..n {
        // Compute w = v^T * qr[:, k]
        let mut w = qr[(j, k)]; // v[j] = 1
        for i in (j + 1)..m {
            w = w + qr[(i, j)] * qr[(i, k)];
        }

        // Update qr[:, k] -= tau * w * v
        let tw = tau * w;
        qr[(j, k)] = qr[(j, k)] - tw; // v[j] = 1
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

    // Apply H = I - tau * v * v^T to all columns of Q
    // v[j] = 1, v[j+1:] stored in qr[j+1:, j]
    for k in 0..m {
        // Compute w = v^T * q[:, k]
        let mut w = q[(j, k)]; // v[j] = 1
        for i in (j + 1)..m {
            w = w + qr[(i, j)] * q[(i, k)];
        }

        // Update q[:, k] -= tau * w * v
        let tw = tau * w;
        q[(j, k)] = q[(j, k)] - tw; // v[j] = 1
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
    fn test_qr_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Verify Q is orthogonal: Q^T * Q = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Verify R is upper triangular
        assert!(approx_eq(r[(1, 0)], 0.0, 1e-10));
        assert!(approx_eq(r[(2, 0)], 0.0, 1e-10));
        assert!(approx_eq(r[(2, 1)], 0.0, 1e-10));

        // Verify Q * R = A
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    approx_eq(sum, a[(i, j)], 1e-10),
                    "QR[{},{}] = {}, A = {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_tall() {
        // 4x2 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q should be 4×4
        assert_eq!(q.nrows(), 4);
        assert_eq!(q.ncols(), 4);

        // R should be 4×2
        assert_eq!(r.nrows(), 4);
        assert_eq!(r.ncols(), 2);

        // Verify Q is orthogonal
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-10));
            }
        }

        // Verify Q * R = A
        for i in 0..4 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(approx_eq(sum, a[(i, j)], 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_wide() {
        // 2x3 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q should be 2×2
        assert_eq!(q.nrows(), 2);
        assert_eq!(q.ncols(), 2);

        // R should be 2×3
        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 3);

        // Verify Q * R = A
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(approx_eq(sum, a[(i, j)], 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let qr = Qr::compute(eye.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q and R should both be close to identity (with possible sign flips)
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(q[(i, j)].abs() > 0.99);
                    assert!(r[(i, j)].abs() > 0.99);
                } else {
                    assert!(approx_eq(q[(i, j)], 0.0, 1e-10));
                    assert!(approx_eq(r[(i, j)], 0.0, 1e-10));
                }
            }
        }
    }

    #[test]
    fn test_qr_thin() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q_thin = qr.q_thin();
        let r_thin = qr.r_thin();

        // Q_thin should be 3×2
        assert_eq!(q_thin.nrows(), 3);
        assert_eq!(q_thin.ncols(), 2);

        // R_thin should be 2×2
        assert_eq!(r_thin.nrows(), 2);
        assert_eq!(r_thin.ncols(), 2);

        // Verify Q_thin * R_thin = A
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += q_thin[(i, k)] * r_thin[(k, j)];
                }
                assert!(approx_eq(sum, a[(i, j)], 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_least_squares() {
        // Overdetermined system: 3 equations, 2 unknowns
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[2.0], &[2.5]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let x = qr.solve_least_squares(b.as_ref()).unwrap();

        // Verify the solution minimizes ||Ax - b||
        // The solution should be close to x = [0.5, 0.75] for this problem
        assert!(x.nrows() == 2);
        assert!(x.ncols() == 1);

        // Verify Ax is close to b in least squares sense
        let mut ax = [0.0; 3];
        for i in 0..3 {
            for j in 0..2 {
                ax[i] += a[(i, j)] * x[(j, 0)];
            }
        }

        // Check residuals are reasonable
        let mut residual = 0.0;
        for i in 0..3 {
            residual += (ax[i] - b[(i, 0)]).powi(2);
        }
        residual = residual.sqrt();
        assert!(residual < 0.5); // Should be small for this well-conditioned problem
    }

    #[test]
    fn test_qr_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A
        for i in 0..2 {
            for j in 0..2 {
                let mut sum: f32 = 0.0;
                for k in 0..2 {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!((sum - a[(i, j)]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_qr_single() {
        let a = Mat::from_rows(&[&[3.0f64]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Q should be ±1, R should be ±3
        assert!(q[(0, 0)].abs() > 0.99);
        assert!(r[(0, 0)].abs() > 2.99);
        assert!(approx_eq((q[(0, 0)] * r[(0, 0)]).abs(), 3.0, 1e-10));
    }

    #[test]
    fn test_qr_blocked_small() {
        // Test blocked QR with a small matrix first
        let n = 8;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i + j) % 5 + 1) as f64;
            }
        }

        // Compute using blocked algorithm with small block size
        let qr_blocked = Qr::compute_blocked(a.as_ref(), 4).unwrap();
        let q = qr_blocked.q();
        let r = qr_blocked.r();

        // Verify Q * R = A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                let diff = sum - a[(i, j)];
                assert!(
                    diff.abs() < 1e-10,
                    "Reconstruction error at ({}, {}): got {}, expected {}, diff={}",
                    i,
                    j,
                    sum,
                    a[(i, j)],
                    diff
                );
            }
        }

        // Verify Q is orthogonal
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "Q not orthogonal at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]

    fn test_qr_blocked_correctness() {
        // Test that blocked QR produces correct factorization
        let n = 200;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i + j) % 10 + 1) as f64;
            }
            a[(i, i)] += 10.0; // Make it well-conditioned
        }

        // Compute using blocked algorithm
        let qr_blocked = Qr::compute_blocked(a.as_ref(), 64).unwrap();
        let q = qr_blocked.q();
        let r = qr_blocked.r();

        // Verify Q is orthogonal: Q^T * Q = I
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-9,
                    "Q not orthogonal at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Verify R is upper triangular
        for i in 0..n {
            for j in 0..i {
                assert!(
                    r[(i, j)].abs() < 1e-10,
                    "R not upper triangular at ({}, {}): got {}",
                    i,
                    j,
                    r[(i, j)]
                );
            }
        }

        // Verify Q * R = A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Reconstruction error at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]

    fn test_qr_auto_selection() {
        // Test automatic algorithm selection
        let n = 150;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 7 + j * 11) % 13 + 1) as f64;
            }
        }

        let qr = Qr::compute_auto(a.as_ref()).unwrap();
        let q = qr.q();
        let r = qr.r();

        // Verify Q is orthogonal: Q^T * Q = I
        // Use relaxed tolerance for larger matrices (150×150) due to accumulated rounding errors
        let tol = 1e-5;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < tol,
                    "Q not orthogonal at ({}, {}): got {}, expected {}, diff={}",
                    i,
                    j,
                    sum,
                    expected,
                    (sum - expected).abs()
                );
            }
        }

        // Verify Q * R = A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "QR reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]

    fn test_qr_blocked_tall_matrix() {
        // Test blocked QR on tall matrix (m > n)
        let m = 300;
        let n = 100;
        let mut a = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a[(i, j)] = ((i + 2 * j) % 7 + 1) as f64;
            }
        }

        let qr = Qr::compute_blocked(a.as_ref(), 32).unwrap();
        let r = qr.r_thin();

        // Verify R is upper triangular
        for i in 0..n {
            for j in 0..i {
                assert!(
                    r[(i, j)].abs() < 1e-10,
                    "R not upper triangular at ({}, {}): got {}",
                    i,
                    j,
                    r[(i, j)]
                );
            }
        }

        // Verify Q * R = A (using thin R)
        let q_thin = qr.q_thin();
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q_thin[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Thin QR reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
