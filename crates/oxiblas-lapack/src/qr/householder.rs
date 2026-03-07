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
    /// let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
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
        use oxiblas_blas::level3::gemm::gemm;

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

            // Step 1: Factor the panel columns j..j+jb using unblocked Householder.
            // Apply each reflector only within the panel (not the trailing matrix).
            for jj in 0..jb {
                let col = j + jj;
                let (tau_col, beta) = householder_vector(&mut qr, col, m, n);
                tau[col] = tau_col;
                qr[(col, col)] = beta;

                // Apply this reflector only to remaining columns within the panel
                if tau_col != T::zero() {
                    for panel_col in (col + 1)..(j + jb).min(n) {
                        // Compute w = v^T * qr[:, panel_col]
                        let mut w = qr[(col, panel_col)]; // v[col] = 1
                        for i in (col + 1)..m {
                            w = w + qr[(i, col)] * qr[(i, panel_col)];
                        }
                        // Update qr[:, panel_col] -= tau * w * v
                        let tw = tau_col * w;
                        qr[(col, panel_col)] = qr[(col, panel_col)] - tw;
                        for i in (col + 1)..m {
                            qr[(i, panel_col)] = qr[(i, panel_col)] - tw * qr[(i, col)];
                        }
                    }
                }
            }

            // Step 2: If there is a trailing submatrix, build the T matrix and
            // apply the block reflector using GEMM (Level 3 BLAS).
            let trailing_start = j + jb;
            if trailing_start < n {
                let panel_rows = m - j; // Rows from j to m
                let trailing_cols = n - trailing_start;

                // Build T matrix (upper triangular, jb x jb).
                // T is defined by: T(i,i) = tau(i), and for i < j:
                //   T(i,j) = -tau(j) * V(:,i)^T * V(:,j)  then premultiply by T(0..i, 0..i)
                // This is the LAPACK DLARFT algorithm (forward, column-wise).
                let mut t_mat = Mat::zeros(jb, jb);
                for jj in 0..jb {
                    let col = j + jj;
                    t_mat[(jj, jj)] = tau[col];

                    if tau[col] != T::zero() && jj > 0 {
                        // Compute z = -tau[col] * V(:, 0..jj)^T * V(:, jj)
                        // V(:,jj) has v[col]=1 implicit, and v[col+1..m] stored in qr[col+1..m, col]
                        // V(:,ii) for ii<jj has v[j+ii]=1 implicit, v[j+ii+1..m] in qr[j+ii+1..m, j+ii]
                        // But we need to work in the "panel coordinate" where V starts at row j.
                        for ii in 0..jj {
                            let col_ii = j + ii;
                            // Compute V(:,ii)^T * V(:,jj) where both vectors start at row j
                            // V(:,ii) has 1 at row j+ii, zeros above, and stored values below
                            // V(:,jj) has 1 at row j+jj, zeros above, and stored values below
                            let mut dot = T::zero();

                            // Both vectors are zero above their respective pivots.
                            // V(row, ii) is nonzero for row >= j+ii (1 at j+ii, stored below)
                            // V(row, jj) is nonzero for row >= j+jj (1 at j+jj, stored below)
                            // Since ii < jj, we have j+ii < j+jj.
                            // The overlap starts at row j+jj.

                            // At row j+jj: V(j+jj, ii) = qr[(j+jj, col_ii)], V(j+jj, jj) = 1
                            dot = dot + qr[(col, col_ii)]; // V(j+jj, ii) * 1

                            // For rows below j+jj:
                            for i in (col + 1)..m {
                                dot = dot + qr[(i, col_ii)] * qr[(i, col)];
                            }

                            t_mat[(ii, jj)] = -tau[col] * dot;
                        }

                        // Apply T(0..jj, 0..jj) to T(0..jj, jj):
                        // T(0..jj, jj) = T(0..jj, 0..jj) * T(0..jj, jj)
                        // Since T is upper triangular, solve from top.
                        // Actually this is a matrix-vector multiply: z = T_sub * z
                        // where T_sub is jj x jj upper triangular.
                        let mut temp = vec![T::zero(); jj];
                        for ii in 0..jj {
                            temp[ii] = t_mat[(ii, jj)];
                        }
                        for ii in 0..jj {
                            let mut s = T::zero();
                            for kk in ii..jj {
                                s = s + t_mat[(ii, kk)] * temp[kk];
                            }
                            t_mat[(ii, jj)] = s;
                        }
                    }
                }

                // Step 3: Apply the block reflector I - V * T * V^T to trailing matrix.
                //
                // The trailing submatrix is qr[j..m, trailing_start..n].
                // V is panel_rows x jb, stored in qr[j..m, j..j+jb] with unit diagonal.
                //
                // We compute:
                //   W = V^T * A_trail    (jb x trailing_cols)
                //   W = T * W            (jb x trailing_cols)
                //   A_trail -= V * W     (panel_rows x trailing_cols)
                //
                // All three steps use GEMM for Level 3 performance.

                // Extract V matrix explicitly (panel_rows x jb) with unit lower triangular
                let mut v_mat = Mat::zeros(panel_rows, jb);
                for jj in 0..jb {
                    // V has unit diagonal at position (jj, jj) in panel coords
                    v_mat[(jj, jj)] = T::one();
                    // Below diagonal of V: stored in qr
                    for i in (jj + 1)..panel_rows {
                        v_mat[(i, jj)] = qr[(j + i, j + jj)];
                    }
                }

                // Extract trailing submatrix
                let mut a_trail = Mat::zeros(panel_rows, trailing_cols);
                for jj in 0..trailing_cols {
                    for i in 0..panel_rows {
                        a_trail[(i, jj)] = qr[(j + i, trailing_start + jj)];
                    }
                }

                // W = V^T * A_trail  (jb x trailing_cols)
                // Compute V^T explicitly since gemm requires MatRef (not TransposeRef)
                let mut v_t = Mat::zeros(jb, panel_rows);
                for ii in 0..jb {
                    for i in 0..panel_rows {
                        v_t[(ii, i)] = v_mat[(i, ii)];
                    }
                }
                let mut w_mat = Mat::zeros(jb, trailing_cols);
                gemm(
                    T::one(),
                    v_t.as_ref(),
                    a_trail.as_ref(),
                    T::zero(),
                    w_mat.as_mut(),
                );

                // W = T^T * W  (jb x trailing_cols)
                // We apply the TRANSPOSE of the block reflector: (I - V*T*V^T)^T = I - V*T^T*V^T
                // because the forward WY form I - V*T*V^T = H(0)*H(1)*...*H(jb-1),
                // and we need H(jb-1)*...*H(1)*H(0) (i.e., H(0) applied first).
                // Since each H(i) is symmetric, (H(0)*...*H(jb-1))^T = H(jb-1)*...*H(0).
                // Compute T^T explicitly since T is small (jb x jb).
                let mut t_trans = Mat::zeros(jb, jb);
                for ii in 0..jb {
                    for kk in 0..jb {
                        t_trans[(ii, kk)] = t_mat[(kk, ii)];
                    }
                }
                let mut tw_mat = Mat::zeros(jb, trailing_cols);
                gemm(
                    T::one(),
                    t_trans.as_ref(),
                    w_mat.as_ref(),
                    T::zero(),
                    tw_mat.as_mut(),
                );

                // A_trail -= V * (T * W)
                gemm(
                    -T::one(),
                    v_mat.as_ref(),
                    tw_mat.as_ref(),
                    T::one(),
                    a_trail.as_mut(),
                );

                // Write back the updated trailing submatrix
                for jj in 0..trailing_cols {
                    for i in 0..panel_rows {
                        qr[(j + i, trailing_start + jj)] = a_trail[(i, jj)];
                    }
                }
            }

            j += jb;
        }

        Ok(Self { qr, tau, m, n })
    }

    /// Recursion threshold for compute_recursive: panels at or below this width
    /// use the blocked algorithm (which itself falls back to unblocked for small panels).
    const RECURSIVE_THRESHOLD: usize = 48;

    /// Computes QR decomposition using a recursive cache-oblivious algorithm.
    ///
    /// This divide-and-conquer approach automatically adapts to the cache hierarchy
    /// by recursively splitting the column space. At each level:
    ///
    /// 1. Split the columns into a left half (n1 cols) and right half (n2 cols)
    /// 2. Recursively factor the left panel to get its Householder vectors and tau
    /// 3. Build the compact WY representation (I - V*T*V^T) for the left panel
    /// 4. Apply the block reflector Q_left^T to the right panel using GEMM
    /// 5. Recursively factor the trailing submatrix of the right panel
    ///
    /// For panels narrower than the recursion threshold (48), falls back to the
    /// blocked algorithm which is more efficient at that scale.
    ///
    /// # Arguments
    ///
    /// * `a` - The m x n matrix to factor
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Qr;
    /// use oxiblas_matrix::Mat;
    ///
    /// let n = 200;
    /// let mut a = Mat::zeros(n, n);
    /// for i in 0..n {
    ///     for j in 0..n {
    ///         a[(i, j)] = ((i * 7 + j * 11) % 13 + 1) as f64;
    ///     }
    ///     a[(i, i)] += 50.0;
    /// }
    ///
    /// let qr = Qr::compute_recursive(a.as_ref()).expect("recursive QR should succeed");
    /// let q = qr.q();
    /// let r = qr.r();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `QrError::EmptyMatrix` if the matrix has zero rows or columns.
    pub fn compute_recursive(a: MatRef<'_, T>) -> Result<Self, QrError> {
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

        // Launch the recursive factorization starting at column 0
        Self::recursive_qr_factor(&mut qr, &mut tau, m, n, 0, k)?;

        Ok(Self { qr, tau, m, n })
    }

    /// Recursive QR factorization on columns col_start..col_start+width.
    ///
    /// Factors the submatrix qr[col_start..m, col_start..col_start+width] in place,
    /// storing Householder vectors below the diagonal and R on/above the diagonal.
    /// The tau values for columns col_start..col_start+width are filled in.
    ///
    /// After factoring, the Householder reflectors are also applied to any trailing
    /// columns col_start+width..n so that the full matrix remains consistent.
    ///
    /// # Arguments
    ///
    /// * `qr` - The full m x n working matrix (modified in place)
    /// * `tau` - Householder scalar array (length min(m,n))
    /// * `m` - Number of rows of the full matrix
    /// * `n` - Number of columns of the full matrix
    /// * `col_start` - Starting column for this recursive call
    /// * `width` - Number of columns to factor in this call
    fn recursive_qr_factor(
        qr: &mut Mat<T>,
        tau: &mut [T],
        m: usize,
        n: usize,
        col_start: usize,
        width: usize,
    ) -> Result<(), QrError> {
        if width == 0 {
            return Ok(());
        }

        // Base case: use blocked (or unblocked for very small) factorization
        if width <= Self::RECURSIVE_THRESHOLD {
            // Factor the panel columns col_start..col_start+width using
            // the panel-level Householder, applying reflectors only within the panel.
            Self::factor_panel_unblocked(qr, tau, m, col_start, width);

            // Apply the panel's reflectors to the trailing columns
            let trailing_start = col_start + width;
            if trailing_start < n {
                Self::apply_block_reflector_to_trailing(qr, tau, m, n, col_start, width)?;
            }
            return Ok(());
        }

        // Recursive case: split the width in half
        let n1 = width / 2;
        let n2 = width - n1;

        // Step 1: Recursively factor the left half.
        // This factors columns col_start..col_start+n1 and applies the resulting
        // reflectors to all trailing columns (col_start+n1..n), including the right half.
        Self::recursive_qr_factor(qr, tau, m, n, col_start, n1)?;

        // Step 2: Recursively factor the right half.
        // The trailing submatrix for the right half starts at row col_start+n1.
        // Columns col_start+n1..col_start+width need factoring, and reflectors
        // must be applied to columns col_start+width..n.
        Self::recursive_qr_factor(qr, tau, m, n, col_start + n1, n2)?;

        Ok(())
    }

    /// Factors a panel of columns using unblocked Householder reflections.
    ///
    /// Applies each reflector only within the panel (not to trailing columns).
    /// The trailing column update is done separately via block reflector application.
    fn factor_panel_unblocked(
        qr: &mut Mat<T>,
        tau: &mut [T],
        m: usize,
        col_start: usize,
        width: usize,
    ) {
        let panel_end = col_start + width;

        for jj in 0..width {
            let col = col_start + jj;

            // Compute Householder vector for this column
            let (tau_col, beta) = householder_vector(qr, col, m, panel_end);
            tau[col] = tau_col;
            qr[(col, col)] = beta;

            // Apply this reflector to remaining columns within the panel
            if tau_col != T::zero() {
                for panel_col in (col + 1)..panel_end {
                    // Compute w = v^T * qr[:, panel_col]
                    let mut w = qr[(col, panel_col)]; // v[col] = 1
                    for i in (col + 1)..m {
                        w = w + qr[(i, col)] * qr[(i, panel_col)];
                    }
                    // Update qr[:, panel_col] -= tau * w * v
                    let tw = tau_col * w;
                    qr[(col, panel_col)] = qr[(col, panel_col)] - tw;
                    for i in (col + 1)..m {
                        qr[(i, panel_col)] = qr[(i, panel_col)] - tw * qr[(i, col)];
                    }
                }
            }
        }
    }

    /// Builds the T matrix for the compact WY representation and applies
    /// the block reflector (I - V*T^T*V^T) to the trailing columns.
    ///
    /// This uses GEMM (Level 3 BLAS) for cache-efficient application.
    fn apply_block_reflector_to_trailing(
        qr: &mut Mat<T>,
        tau: &[T],
        m: usize,
        n: usize,
        col_start: usize,
        width: usize,
    ) -> Result<(), QrError> {
        use oxiblas_blas::level3::gemm::gemm;

        let trailing_start = col_start + width;
        if trailing_start >= n {
            return Ok(());
        }

        let panel_rows = m - col_start; // Rows from col_start to m
        let trailing_cols = n - trailing_start;

        // Build T matrix (upper triangular, width x width) using DLARFT-style algorithm.
        // T(i,i) = tau(col_start + i)
        // For i < j: T(i,j) = -tau(col_start + j) * V(:,i)^T * V(:,j), then T(0..j, j) = T(0..j, 0..j) * T(0..j, j)
        let mut t_mat = Mat::zeros(width, width);
        for jj in 0..width {
            let col = col_start + jj;
            t_mat[(jj, jj)] = tau[col];

            if tau[col] != T::zero() && jj > 0 {
                for ii in 0..jj {
                    let col_ii = col_start + ii;
                    // Compute V(:,ii)^T * V(:,jj)
                    // Both vectors are zero above their respective pivots.
                    // V(row, ii) is nonzero for row >= col_start+ii (1 at col_start+ii, stored below)
                    // V(row, jj) is nonzero for row >= col_start+jj (1 at col_start+jj, stored below)
                    // Since ii < jj, the overlap starts at row col_start+jj = col.
                    let mut dot = T::zero();

                    // At row col: V(col, ii) = qr[(col, col_ii)], V(col, jj) = 1
                    dot = dot + qr[(col, col_ii)]; // V(col, ii) * 1

                    // For rows below col:
                    for i in (col + 1)..m {
                        dot = dot + qr[(i, col_ii)] * qr[(i, col)];
                    }

                    t_mat[(ii, jj)] = -tau[col] * dot;
                }

                // Apply T(0..jj, 0..jj) to T(0..jj, jj): z = T_sub * z
                let mut temp = vec![T::zero(); jj];
                for ii in 0..jj {
                    temp[ii] = t_mat[(ii, jj)];
                }
                for ii in 0..jj {
                    let mut s = T::zero();
                    for kk in ii..jj {
                        s = s + t_mat[(ii, kk)] * temp[kk];
                    }
                    t_mat[(ii, jj)] = s;
                }
            }
        }

        // Extract V matrix explicitly (panel_rows x width) with unit lower triangular
        let mut v_mat = Mat::zeros(panel_rows, width);
        for jj in 0..width {
            v_mat[(jj, jj)] = T::one();
            for i in (jj + 1)..panel_rows {
                v_mat[(i, jj)] = qr[(col_start + i, col_start + jj)];
            }
        }

        // Extract trailing submatrix (panel_rows x trailing_cols)
        let mut a_trail = Mat::zeros(panel_rows, trailing_cols);
        for jj in 0..trailing_cols {
            for i in 0..panel_rows {
                a_trail[(i, jj)] = qr[(col_start + i, trailing_start + jj)];
            }
        }

        // W = V^T * A_trail  (width x trailing_cols)
        let mut v_t = Mat::zeros(width, panel_rows);
        for ii in 0..width {
            for i in 0..panel_rows {
                v_t[(ii, i)] = v_mat[(i, ii)];
            }
        }
        let mut w_mat = Mat::zeros(width, trailing_cols);
        gemm(
            T::one(),
            v_t.as_ref(),
            a_trail.as_ref(),
            T::zero(),
            w_mat.as_mut(),
        );

        // W = T^T * W  (width x trailing_cols)
        // We need the transpose of T because the forward WY form I - V*T*V^T = H(0)*H(1)*...*H(jb-1),
        // and we need the product applied as H(0) first (from left: H(jb-1)*...*H(0)*A).
        let mut t_trans = Mat::zeros(width, width);
        for ii in 0..width {
            for kk in 0..width {
                t_trans[(ii, kk)] = t_mat[(kk, ii)];
            }
        }
        let mut tw_mat = Mat::zeros(width, trailing_cols);
        gemm(
            T::one(),
            t_trans.as_ref(),
            w_mat.as_ref(),
            T::zero(),
            tw_mat.as_mut(),
        );

        // A_trail -= V * (T^T * W)
        gemm(
            -T::one(),
            v_mat.as_ref(),
            tw_mat.as_ref(),
            T::one(),
            a_trail.as_mut(),
        );

        // Write back the updated trailing submatrix
        for jj in 0..trailing_cols {
            for i in 0..panel_rows {
                qr[(col_start + i, trailing_start + jj)] = a_trail[(i, jj)];
            }
        }

        Ok(())
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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
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

        let qr = Qr::compute(eye.as_ref()).expect("QR should succeed");
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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
        let x = qr
            .solve_least_squares(b.as_ref())
            .expect("least squares should succeed");

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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
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

        let qr = Qr::compute(a.as_ref()).expect("QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Q should be ±1, R should be ±3
        assert!(q[(0, 0)].abs() > 0.99);
        assert!(r[(0, 0)].abs() > 2.99);
        assert!(approx_eq((q[(0, 0)] * r[(0, 0)]).abs(), 3.0, 1e-10));
    }

    #[test]
    fn test_qr_blocked_vs_unblocked_4x4() {
        // Verify blocked and unblocked produce equivalent factorizations
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
        ]);

        let qr_blocked = Qr::compute_blocked(a.as_ref(), 2).expect("blocked QR should succeed");
        let q_b = qr_blocked.q();
        let r_b = qr_blocked.r();

        // Verify Q * R = A for blocked
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += q_b[(i, k)] * r_b[(k, j)];
                }
                let diff = sum - a[(i, j)];
                assert!(
                    diff.abs() < 1e-10,
                    "Blocked reconstruction error at ({}, {}): got {}, expected {}, diff={}",
                    i,
                    j,
                    sum,
                    a[(i, j)],
                    diff
                );
            }
        }

        // Verify Q is orthogonal
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += q_b[(k, i)] * q_b[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "Q not orthogonal at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Verify R is upper triangular
        for i in 0..4 {
            for j in 0..i {
                assert!(
                    r_b[(i, j)].abs() < 1e-10,
                    "R not upper triangular at ({}, {}): got {}",
                    i,
                    j,
                    r_b[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_blocked_various_block_sizes() {
        // Test blocked QR with different block sizes on a 12x12 matrix
        let n = 12;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 3 + j * 7 + 1) % 11) as f64 + 1.0;
            }
            a[(i, i)] += 20.0; // Make well-conditioned
        }

        for nb in [1, 2, 3, 4, 6, 12] {
            let qr = Qr::compute_blocked(a.as_ref(), nb).expect("blocked QR should succeed");
            let q = qr.q();
            let r = qr.r();

            // Verify Q * R = A
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += q[(i, k)] * r[(k, j)];
                    }
                    assert!(
                        (sum - a[(i, j)]).abs() < 1e-9,
                        "nb={}: reconstruction error at ({}, {}): diff={}",
                        nb,
                        i,
                        j,
                        sum - a[(i, j)]
                    );
                }
            }

            // Verify Q^T * Q = I
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += q[(k, i)] * q[(k, j)];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1e-9,
                        "nb={}: Q not orthogonal at ({}, {})",
                        nb,
                        i,
                        j
                    );
                }
            }
        }
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
        let qr_blocked = Qr::compute_blocked(a.as_ref(), 4).expect("blocked QR should succeed");
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
        let qr_blocked = Qr::compute_blocked(a.as_ref(), 64).expect("blocked QR should succeed");
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

        let qr = Qr::compute_auto(a.as_ref()).expect("auto QR should succeed");
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

        let qr = Qr::compute_blocked(a.as_ref(), 32).expect("blocked QR should succeed");
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

    #[test]
    fn test_qr_blocked_wide_matrix() {
        // Test blocked QR on wide matrix (m < n)
        let m = 50;
        let n = 120;
        let mut a = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a[(i, j)] = ((i * 5 + j * 3 + 2) % 11) as f64 + 0.5;
            }
        }

        let qr = Qr::compute_blocked(a.as_ref(), 16).expect("blocked QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q^T * Q = I
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-9,
                    "Q not orthogonal at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - expected).abs()
                );
            }
        }

        // Verify R is upper triangular
        let k = m.min(n);
        for i in 0..m {
            for j in 0..i.min(k) {
                assert!(
                    r[(i, j)].abs() < 1e-10,
                    "R not upper triangular at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Verify Q * R = A
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Wide reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_blocked_f32() {
        // Test blocked QR with f32 precision
        let n = 32;
        let mut a: Mat<f32> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 3 + j * 5 + 1) % 9 + 1) as f32;
            }
            a[(i, i)] += 10.0;
        }

        let qr = Qr::compute_blocked(a.as_ref(), 8).expect("f32 blocked QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A with f32 tolerance
        for i in 0..n {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-3,
                    "f32 blocked reconstruction error at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - a[(i, j)]).abs()
                );
            }
        }

        // Verify Q^T * Q = I with f32 tolerance
        for i in 0..n {
            for j in 0..n {
                let mut sum: f32 = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected: f32 = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-3,
                    "f32 Q not orthogonal at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_blocked_identity_matrix() {
        // The identity should decompose trivially
        let n = 16;
        let mut eye = Mat::zeros(n, n);
        for i in 0..n {
            eye[(i, i)] = 1.0f64;
        }

        let qr = Qr::compute_blocked(eye.as_ref(), 4).expect("identity blocked QR should succeed");
        let q = qr.q();
        let r = qr.r();

        for i in 0..n {
            for j in 0..n {
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                // Q and R should each be +/-I (with possible sign flips on diagonal)
                assert!(
                    (q[(i, j)].abs() - expected.abs()).abs() < 1e-10
                        || (i == j && q[(i, j)].abs() > 0.99),
                    "Identity Q error at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Q * R should reconstruct identity
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "Identity reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_blocked_block_size_1() {
        // Block size 1 should be equivalent to unblocked
        let n = 20;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i + j + 1) % 7) as f64 + 1.0;
            }
            a[(i, i)] += 15.0;
        }

        let qr_unblocked = Qr::compute(a.as_ref()).expect("unblocked QR should succeed");
        let qr_blocked =
            Qr::compute_blocked(a.as_ref(), 1).expect("nb=1 blocked QR should succeed");

        // R matrices should be numerically identical
        let r_u = qr_unblocked.r();
        let r_b = qr_blocked.r();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (r_u[(i, j)] - r_b[(i, j)]).abs() < 1e-10,
                    "R mismatch at ({}, {}): unblocked={}, blocked={}",
                    i,
                    j,
                    r_u[(i, j)],
                    r_b[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_blocked_block_size_exceeds_n() {
        // Block size larger than matrix dimension -- single panel, no trailing update
        let n = 8;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = (i * n + j + 1) as f64;
            }
        }

        let qr = Qr::compute_blocked(a.as_ref(), 64).expect("large nb should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-9,
                    "Large nb reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_auto_small_uses_unblocked() {
        // For small matrices (< 128), auto should use unblocked
        let n = 64;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 3 + j * 7) % 17 + 1) as f64;
            }
            a[(i, i)] += 20.0;
        }

        let qr = Qr::compute_auto(a.as_ref()).expect("auto QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Verify reconstruction
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-8,
                    "Auto small reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_blocked_well_conditioned() {
        // Test with a well-conditioned matrix (diagonally dominant)
        let n = 100;
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = if i == j {
                    100.0
                } else {
                    1.0 / ((i as f64 - j as f64).abs() + 1.0)
                };
            }
        }

        let qr = Qr::compute_blocked(a.as_ref(), 32).expect("well-conditioned QR should succeed");
        let q = qr.q();
        let r = qr.r();

        // Tight orthogonality check
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "Well-conditioned Q orthogonality error at ({}, {}): diff={}",
                    i,
                    j,
                    (sum - expected).abs()
                );
            }
        }

        // Tight reconstruction check
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[(i, k)] * r[(k, j)];
                }
                assert!(
                    (sum - a[(i, j)]).abs() < 1e-9,
                    "Well-conditioned reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
