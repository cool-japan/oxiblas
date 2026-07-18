//! Sparse QR decomposition for overdetermined and square sparse systems.
//!
//! This is a genuine *sparse* multifrontal-style Householder QR (the
//! George & Heath / Davis `cs_qr` algorithm). It never densifies the matrix:
//! the only dense scratch it uses is a single length-`m` column workspace,
//! so it factorizes large sparse matrices in `O(nnz(R) + nnz(V))` work and
//! memory rather than `O(m * n)`.
//!
//! Factorization: `A * P = Q * R`, where
//! - `P` is a fill-reducing COLAMD column permutation (`col_perm`),
//! - `Q` is orthogonal, stored implicitly as a sequence of *sparse* Householder
//!   reflections `H_k = I - beta_k * v_k * v_k^T` (the columns `v_k` are kept in
//!   CSC form in `v_factor`, the scalars `beta_k` in `beta`),
//! - `R` is upper triangular (stored as [`CscMatrix`]).
//!
//! The sparsity of `Q` and `R` is discovered symbolically from the *column
//! elimination tree* of `A` (the elimination tree of `A^T A`, computed without
//! ever forming `A^T A`) together with a row permutation that gives every
//! Householder vector a well-defined leading row. Each column's Householder
//! reflections are applied only along the relevant path of that tree, so the
//! working set touches nothing outside the nonzero structure.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oxiblas_sparse::linalg::{SparseQr, SparseQrConfig};
//!
//! let qr = SparseQr::compute(&a).unwrap();
//! let x = qr.solve_least_squares(&b).unwrap();
//! ```

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use crate::linalg::ordering::colamd;
use oxiblas_core::scalar::Scalar;

/// Sentinel used in place of a signed `-1` for `usize` index arrays.
const NONE: usize = usize::MAX;

/// Error type for sparse QR decomposition.
#[derive(Debug, Clone)]
pub enum SparseQrError {
    /// Matrix is singular (zero diagonal in R).
    SingularMatrix,
    /// RHS vector has incompatible length.
    IncompatibleDimensions {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension provided.
        got: usize,
    },
    /// Matrix data is structurally invalid.
    InvalidMatrix(String),
    /// Numerical failure during factorization.
    NumericalFailure(String),
    /// Matrix is rank deficient.
    RankDeficient {
        /// Detected rank.
        rank: usize,
        /// Number of columns.
        n: usize,
    },
}

impl std::fmt::Display for SparseQrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingularMatrix => write!(f, "Matrix is singular: zero diagonal in R"),
            Self::IncompatibleDimensions { expected, got } => {
                write!(f, "Incompatible dimensions: expected {expected}, got {got}")
            }
            Self::InvalidMatrix(msg) => write!(f, "Invalid matrix: {msg}"),
            Self::NumericalFailure(msg) => write!(f, "Numerical failure: {msg}"),
            Self::RankDeficient { rank, n } => {
                write!(f, "Matrix is rank deficient: rank {rank} of {n}")
            }
        }
    }
}

impl std::error::Error for SparseQrError {}

/// Configuration for sparse QR factorization.
#[derive(Debug, Clone)]
pub struct SparseQrConfig {
    /// Whether to use COLAMD column ordering (default: `true`).
    pub column_ordering: bool,
    /// Threshold for numerical zero when storing R off-diagonals (default: `1e-14`).
    /// The diagonal of R is always kept regardless of this value.
    pub drop_tol: f64,
    /// Minimum acceptable diagonal pivot in R (default: `1e-12`).
    /// Entries smaller than this trigger rank-deficiency detection.
    pub min_diagonal: f64,
}

impl Default for SparseQrConfig {
    fn default() -> Self {
        Self {
            column_ordering: true,
            drop_tol: 1e-14,
            min_diagonal: 1e-12,
        }
    }
}

/// Sparse QR factorization result: `A * P = Q * R`.
///
/// - `P` is the COLAMD column permutation (`col_perm`),
/// - `Q` is orthogonal, stored implicitly as a sequence of *sparse* Householder
///   reflections,
/// - `R` is upper triangular, stored in CSC format (`r_factor`).
///
/// Supports solving overdetermined (`m > n`) least squares problems and
/// square (`m == n`) full-rank exact solves.
#[derive(Debug)]
pub struct SparseQr<T: Scalar> {
    /// Upper triangular R factor in CSC format (n × n).
    pub r_factor: CscMatrix<T>,
    /// Column permutation: `col_perm[i]` is the original column index for
    /// the i-th column in the permuted system.
    pub col_perm: Vec<usize>,
    /// Estimated rank of the matrix.
    pub rank: usize,
    /// Diagonal entries of R (length n), used for rank and condition estimation.
    pub r_diag: Vec<T>,
    config: SparseQrConfig,
    m: usize,
    n: usize,
    /// Householder vectors, stored column-by-column in CSC format
    /// (`m2` × n, where `m2 >= m` includes fictitious rows). Column `k` holds
    /// the sparse reflection vector `v_k` with leading entry at (permuted) row `k`.
    v_factor: CscMatrix<T>,
    /// Householder scalars `beta_k` such that `H_k = I - beta_k * v_k * v_k^T`.
    beta: Vec<T>,
    /// Row permutation: `pinv[i]` is the permuted row index of original row `i`.
    pinv: Vec<usize>,
    /// Number of rows in the permuted (Householder) row space (`m2 >= m`).
    m2: usize,
}

impl SparseQr<f64> {
    /// Compute the sparse QR factorization of matrix `a` (m × n) with default config.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::RankDeficient`] if the matrix has rank < n.
    /// Returns [`SparseQrError::InvalidMatrix`] if the matrix is empty.
    pub fn compute(a: &CsrMatrix<f64>) -> Result<Self, SparseQrError> {
        Self::compute_with_config(a, SparseQrConfig::default())
    }

    /// Compute the sparse QR factorization of matrix `a` (m × n) with custom config.
    ///
    /// The factorization proceeds in three sparse phases and never allocates a
    /// dense `m × n` working array:
    ///
    /// 1. **Ordering** — a fill-reducing COLAMD column permutation is applied.
    /// 2. **Symbolic analysis** — the column elimination tree (the elimination
    ///    tree of `A^T A`) and a row permutation (`cs_vcount`) fix the nonzero
    ///    structure of `V` and `R`.
    /// 3. **Numeric factorization** — sparse Householder reflections are applied
    ///    along the elimination-tree paths, using only an `O(m)` dense column
    ///    workspace.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::RankDeficient`] if the matrix has rank < n.
    /// Returns [`SparseQrError::InvalidMatrix`] if the matrix is empty.
    pub fn compute_with_config(
        a: &CsrMatrix<f64>,
        config: SparseQrConfig,
    ) -> Result<Self, SparseQrError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SparseQrError::InvalidMatrix(
                "Matrix must have at least one row and one column".to_string(),
            ));
        }

        // Convert CSR -> CSC (COLAMD and the sparse QR work column-wise).
        let a_csc = csr_to_csc_f64(a);

        // Phase 1: fill-reducing column ordering.
        let col_perm: Vec<usize> = if config.column_ordering {
            colamd(&a_csc)
        } else {
            (0..n).collect()
        };

        // Build the column-permuted matrix C = A(:, col_perm) once, in CSC form.
        // Everything downstream (etree, vcount, numeric) operates on C's columns.
        let c = permute_columns_csc(&a_csc, &col_perm);

        // Phase 2a: column elimination tree (etree of C^T C, never forming C^T C).
        let parent = column_etree(&c);

        // Phase 2b: row permutation, leftmost columns, and m2 (fictitious rows).
        let (pinv, leftmost, m2) = vcount(&c, &parent);

        // Phase 3: numeric sparse Householder QR.
        let SparseFactors {
            v_factor,
            beta,
            r_factor,
            r_diag,
        } = sparse_householder_qr(&c, &parent, &pinv, &leftmost, m2, &config);

        // Rank detection from the R diagonal.
        let mut rank = 0usize;
        for &d in &r_diag {
            if d.abs() > config.min_diagonal {
                rank += 1;
            }
        }
        if rank < n {
            return Err(SparseQrError::RankDeficient { rank, n });
        }

        Ok(Self {
            r_factor,
            col_perm,
            rank,
            r_diag,
            config,
            m,
            n,
            v_factor,
            beta,
            pinv,
            m2,
        })
    }

    /// Solve the least squares problem: minimize `||A*x - b||` for overdetermined systems.
    ///
    /// Procedure:
    /// 1. Permute `b` into the Householder row space: `x[pinv[i]] = b[i]`.
    /// 2. Apply all Householder reflections in order (`x = Q^T * b`).
    /// 3. Back-substitute with R on the leading `n` entries: `x_perm = R^{-1} * x[0..n]`.
    /// 4. Apply the inverse column permutation: `out[col_perm[i]] = x_perm[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::IncompatibleDimensions`] if `b.len() != m`.
    /// Returns [`SparseQrError::SingularMatrix`] if a zero diagonal is encountered.
    pub fn solve_least_squares(&self, b: &[f64]) -> Result<Vec<f64>, SparseQrError> {
        if b.len() != self.m {
            return Err(SparseQrError::IncompatibleDimensions {
                expected: self.m,
                got: b.len(),
            });
        }

        // Step 1: scatter b into the permuted (m2) row space; fictitious rows stay 0.
        let mut x = vec![0.0f64; self.m2];
        for (i, &bi) in b.iter().enumerate() {
            x[self.pinv[i]] = bi;
        }

        // Step 2: apply the sparse Householder reflections H_0, H_1, ..., H_{n-1}
        // in order, which computes Q^T * b in place.
        for k in 0..self.n {
            happly_csc(&self.v_factor, k, self.beta[k], &mut x);
        }

        // Step 3: back-substitute R (upper triangular, CSC, diagonal stored last).
        let col_ptrs = self.r_factor.col_ptrs();
        let row_indices = self.r_factor.row_indices();
        let values = self.r_factor.values();
        for j in (0..self.n).rev() {
            let start = col_ptrs[j];
            let end = col_ptrs[j + 1];
            if start == end {
                return Err(SparseQrError::SingularMatrix);
            }
            // The diagonal R[j, j] is the last (largest-row) entry in column j.
            let diag_idx = end - 1;
            let rjj = values[diag_idx];
            if rjj.abs() < self.config.min_diagonal {
                return Err(SparseQrError::SingularMatrix);
            }
            x[j] /= rjj;
            for idx in start..diag_idx {
                let row = row_indices[idx];
                x[row] -= values[idx] * x[j];
            }
        }

        // Step 4: undo the column permutation.
        let mut out = vec![0.0f64; self.n];
        for i in 0..self.n {
            out[self.col_perm[i]] = x[i];
        }
        Ok(out)
    }

    /// Solve exactly: `A * x = b` when `m == n` and full rank.
    ///
    /// # Errors
    ///
    /// Returns [`SparseQrError::IncompatibleDimensions`] if `m != n`.
    /// Returns errors from [`Self::solve_least_squares`].
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SparseQrError> {
        if self.m != self.n {
            return Err(SparseQrError::IncompatibleDimensions {
                expected: self.n,
                got: self.m,
            });
        }
        self.solve_least_squares(b)
    }

    /// Return the estimated rank of the matrix.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Estimate the condition number as `max(|R_diag|) / min(|R_diag|)`.
    ///
    /// Returns `f64::INFINITY` if any diagonal entry is zero or the diag is empty.
    pub fn condition_number_estimate(&self) -> f64 {
        if self.r_diag.is_empty() {
            return f64::INFINITY;
        }
        let max_d = self.r_diag.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
        let min_d = self
            .r_diag
            .iter()
            .map(|d| d.abs())
            .fold(f64::INFINITY, f64::min);
        if min_d == 0.0 {
            f64::INFINITY
        } else {
            max_d / min_d
        }
    }
}

/// Bundle of the sparse factors produced by [`sparse_householder_qr`].
struct SparseFactors {
    v_factor: CscMatrix<f64>,
    beta: Vec<f64>,
    r_factor: CscMatrix<f64>,
    r_diag: Vec<f64>,
}

/// Numeric sparse Householder QR of the column-permuted matrix `c`.
///
/// This is a faithful, allocation-conscious port of the George & Heath /
/// Davis `cs_qr` kernel. The Householder vectors `V` and the upper-triangular
/// factor `R` are built incrementally in CSC form; the only dense scratch is
/// the length-`m2` workspace `x`, so no dense `m × n` array is ever allocated.
fn sparse_householder_qr(
    c: &CscMatrix<f64>,
    parent: &[usize],
    pinv: &[usize],
    leftmost: &[usize],
    m2: usize,
    config: &SparseQrConfig,
) -> SparseFactors {
    let n = c.ncols();
    let col_ptrs = c.col_ptrs();
    let row_indices = c.row_indices();
    let values = c.values();

    // Dense scratch (O(m2)) and a stack of etree nodes on the current path.
    let mut x = vec![0.0f64; m2];
    let mut w = vec![-1i64; m2]; // node/row marker; -1 means "unmarked"
    let mut stack = vec![0usize; n];

    // Householder vectors V (CSC, m2 × n) built incrementally.
    let mut vp = vec![0usize; n + 1];
    let mut vi: Vec<usize> = Vec::new();
    let mut vx: Vec<f64> = Vec::new();
    let mut beta = vec![0.0f64; n];

    // Upper triangular R (CSC, n × n) built incrementally.
    let mut rp = vec![0usize; n + 1];
    let mut ri: Vec<usize> = Vec::new();
    let mut rx: Vec<f64> = Vec::new();
    let mut r_diag = Vec::with_capacity(n);

    for k in 0..n {
        rp[k] = ri.len();
        vp[k] = vi.len();
        let p1 = vi.len();

        // The Householder vector for column k always contains its leading row k.
        w[k] = k as i64;
        vi.push(k);

        let mut top = n; // stack grows downward from n; path lives in stack[top..n]

        // Scatter C(:, k) into x and discover the pattern of R(:, k) via the
        // reach of {leftmost[i]} up the column elimination tree.
        for p in col_ptrs[k]..col_ptrs[k + 1] {
            let row = row_indices[p];

            // Traverse the etree from leftmost[row] up to (the already-marked) k.
            let mut i = leftmost[row];
            let mut len = 0usize;
            while i != NONE && w[i] != k as i64 {
                stack[len] = i;
                len += 1;
                w[i] = k as i64;
                i = parent[i];
            }
            // Push the traversed path (reversed) onto the descending stack.
            while len > 0 {
                len -= 1;
                top -= 1;
                stack[top] = stack[len];
            }

            // Scatter the numeric value into the permuted row.
            let pr = pinv[row];
            x[pr] = values[p];
            // Rows strictly below the diagonal join the pattern of V(:, k).
            if pr > k && w[pr] < k as i64 {
                vi.push(pr);
                w[pr] = k as i64;
            }
        }

        // Apply the previously-computed Householder reflections along the path
        // (ascending order), forming R(:, k) above the diagonal.
        for sp in top..n {
            let i = stack[sp];
            happly_raw(&vp, &vi, &vx, i, beta[i], &mut x);
            ri.push(i);
            rx.push(x[i]);
            x[i] = 0.0;
            // Symbolic fill: propagate the pattern of child V(:, i) into V(:, k).
            if parent[i] == k {
                let src_start = vp[i];
                let src_end = vp[i + 1];
                for pp in src_start..src_end {
                    let row = vi[pp];
                    if w[row] < k as i64 {
                        w[row] = k as i64;
                        vi.push(row);
                    }
                }
            }
        }

        // Gather V(:, k) values out of the dense workspace, clearing x as we go.
        let vend = vi.len();
        for p in p1..vend {
            vx.push(x[vi[p]]);
            x[vi[p]] = 0.0;
        }

        // Compute the Householder reflection for column k; R[k, k] = norm.
        let mut b = 0.0f64;
        let rkk = cs_house(&mut vx[p1..vend], &mut b);
        beta[k] = b;
        r_diag.push(rkk);
        ri.push(k);
        rx.push(rkk);
    }
    rp[n] = ri.len();
    vp[n] = vi.len();

    // Assemble R with sorted columns (diagonal becomes the last entry) and
    // drop_tol filtering of off-diagonals.
    let (r_col_ptrs, r_rows, r_vals) = finalize_upper_csc(&rp, &ri, &rx, n, config.drop_tol);

    // Safety: constructed to satisfy the CSC invariants (see finalize/build).
    let r_factor = unsafe { CscMatrix::new_unchecked(n, n, r_col_ptrs, r_rows, r_vals) };
    let v_factor = unsafe { CscMatrix::new_unchecked(m2, n, vp, vi, vx) };

    SparseFactors {
        v_factor,
        beta,
        r_factor,
        r_diag,
    }
}

/// Apply the sparse Householder reflection stored in column `i` of a raw CSC
/// triple `(vp, vi, vx)` to the dense vector `x`: `x <- (I - beta v v^T) x`.
fn happly_raw(vp: &[usize], vi: &[usize], vx: &[f64], i: usize, beta: f64, x: &mut [f64]) {
    let start = vp[i];
    let end = vp[i + 1];
    let mut tau = 0.0f64;
    for p in start..end {
        tau += vx[p] * x[vi[p]];
    }
    tau *= beta;
    if tau == 0.0 {
        return;
    }
    for p in start..end {
        x[vi[p]] -= vx[p] * tau;
    }
}

/// Apply the sparse Householder reflection stored in column `i` of a CSC matrix
/// `v` to the dense vector `x`: `x <- (I - beta v v^T) x`.
fn happly_csc(v: &CscMatrix<f64>, i: usize, beta: f64, x: &mut [f64]) {
    let start = v.col_ptrs()[i];
    let end = v.col_ptrs()[i + 1];
    let rows = v.row_indices();
    let vals = v.values();
    let mut tau = 0.0f64;
    for p in start..end {
        tau += vals[p] * x[rows[p]];
    }
    tau *= beta;
    if tau == 0.0 {
        return;
    }
    for p in start..end {
        x[rows[p]] -= vals[p] * tau;
    }
}

/// Compute a Householder reflection for the vector held in `v`.
///
/// On entry `v[0]` is the diagonal entry and `v[1..]` the sub-diagonal tail.
/// On return `v` holds the Householder vector and `*beta` the scalar such that
/// `(I - beta v v^T)` maps the input to `s * e_1` (up to sign). Returns the
/// non-negative norm `s`, which becomes the R diagonal.
fn cs_house(v: &mut [f64], beta: &mut f64) -> f64 {
    if v.is_empty() {
        *beta = 0.0;
        return 0.0;
    }
    let mut sigma = 0.0f64;
    for &vi in v.iter().skip(1) {
        sigma += vi * vi;
    }
    let x0 = v[0];
    let s;
    if sigma == 0.0 {
        s = x0.abs();
        *beta = if x0 <= 0.0 { 2.0 } else { 0.0 };
        v[0] = 1.0;
    } else {
        s = (x0 * x0 + sigma).sqrt();
        v[0] = if x0 <= 0.0 { x0 - s } else { -sigma / (x0 + s) };
        *beta = -1.0 / (s * v[0]);
    }
    s
}

/// Compute the column elimination tree of `c`, i.e. the elimination tree of
/// `C^T C`, without ever forming `C^T C` (Davis's `cs_etree` with `ata = 1`).
///
/// Returns `parent`, where `parent[j]` is the parent column node of `j` or
/// [`NONE`] if `j` is a root of the forest.
fn column_etree(c: &CscMatrix<f64>) -> Vec<usize> {
    let m = c.nrows();
    let n = c.ncols();
    let col_ptrs = c.col_ptrs();
    let row_indices = c.row_indices();

    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    let mut prev = vec![NONE; m];

    for k in 0..n {
        for p in col_ptrs[k]..col_ptrs[k + 1] {
            let row = row_indices[p];
            // For the A^T A tree, start from the previous column touching `row`.
            let mut i = prev[row];
            while i != NONE && i < k {
                let inext = ancestor[i];
                ancestor[i] = k;
                if inext == NONE {
                    parent[i] = k;
                }
                i = inext;
            }
            prev[row] = k;
        }
    }
    parent
}

/// Compute the row permutation `pinv`, the `leftmost` column of each row, and
/// the number of Householder rows `m2` (Davis's `cs_vcount`).
///
/// `pinv[i]` is the permuted row index of original row `i`. Fictitious rows are
/// added (increasing `m2` beyond `m`) so that every column has a Householder
/// vector with a distinct leading row.
fn vcount(c: &CscMatrix<f64>, parent: &[usize]) -> (Vec<usize>, Vec<usize>, usize) {
    let m = c.nrows();
    let n = c.ncols();
    let col_ptrs = c.col_ptrs();
    let row_indices = c.row_indices();

    // pinv may address fictitious rows in [m, m2); m2 <= m + n, so size m + n.
    let mut pinv = vec![NONE; m + n];
    let mut leftmost = vec![NONE; m];

    // Bucket queues keyed by leftmost column.
    let mut next = vec![NONE; m];
    let mut head = vec![NONE; n];
    let mut tail = vec![NONE; n];
    let mut nque = vec![0usize; n];

    // leftmost[i] = min { k : C[i, k] != 0 }.
    for k in (0..n).rev() {
        for p in col_ptrs[k]..col_ptrs[k + 1] {
            leftmost[row_indices[p]] = k;
        }
    }

    // Insert rows into their leftmost-column queue (reverse order keeps queues
    // in ascending row order at the head).
    for i in (0..m).rev() {
        let k = leftmost[i];
        if k == NONE {
            continue; // empty row
        }
        if nque[k] == 0 {
            tail[k] = i;
        }
        nque[k] += 1;
        next[i] = head[k];
        head[k] = i;
    }

    let mut m2 = m;
    for k in 0..n {
        let mut i = head[k]; // pop the head row of queue k
        if i == NONE {
            i = m2; // no real row available: add a fictitious one
            m2 += 1;
        }
        pinv[i] = k; // row i is the leading row of V(:, k)
        if nque[k] <= 1 {
            // Queue k is now empty (or was fictitious): nothing to move up.
            continue;
        }
        nque[k] -= 1; // remaining rows below the diagonal of V(:, k)
        let pa = parent[k];
        if pa != NONE {
            // Splice the remaining rows onto the parent's queue.
            if nque[pa] == 0 {
                tail[pa] = tail[k];
            }
            next[tail[k]] = head[pa];
            head[pa] = next[i];
            nque[pa] += nque[k];
        }
    }

    // Any real row not yet placed gets the next free permuted index.
    let mut free = n;
    for i in 0..m {
        if pinv[i] == NONE {
            pinv[i] = free;
            free += 1;
        }
    }

    pinv.truncate(m);
    (pinv, leftmost, m2)
}

/// Build C = A(:, q) as a CSC matrix, where `q[k]` is the original column for
/// permuted column `k`.
fn permute_columns_csc(a: &CscMatrix<f64>, q: &[usize]) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let ap = a.col_ptrs();
    let ai = a.row_indices();
    let ax = a.values();

    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_indices = Vec::with_capacity(ai.len());
    let mut values = Vec::with_capacity(ax.len());

    for k in 0..n {
        let old_j = q[k];
        for p in ap[old_j]..ap[old_j + 1] {
            row_indices.push(ai[p]);
            values.push(ax[p]);
        }
        col_ptrs[k + 1] = values.len();
    }

    // Safety: col_ptrs, row_indices, values are consistent by construction.
    unsafe { CscMatrix::new_unchecked(m, n, col_ptrs, row_indices, values) }
}

/// Finalize the raw upper-triangular CSC triple into sorted columns.
///
/// Each column is sorted by ascending row index (so the diagonal `j`, the
/// largest row index present, ends up last) and off-diagonal entries with
/// magnitude `<= drop_tol` are dropped. The diagonal is always kept.
fn finalize_upper_csc(
    rp: &[usize],
    ri: &[usize],
    rx: &[f64],
    n: usize,
    drop_tol: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut col_ptrs = vec![0usize; n + 1];
    let mut rows = Vec::with_capacity(ri.len());
    let mut vals = Vec::with_capacity(rx.len());

    let mut col: Vec<(usize, f64)> = Vec::new();
    for j in 0..n {
        col.clear();
        for p in rp[j]..rp[j + 1] {
            let row = ri[p];
            let val = rx[p];
            if row == j || val.abs() > drop_tol {
                col.push((row, val));
            }
        }
        col.sort_unstable_by_key(|&(r, _)| r);
        for &(r, v) in &col {
            rows.push(r);
            vals.push(v);
        }
        col_ptrs[j + 1] = vals.len();
    }

    (col_ptrs, rows, vals)
}

/// Convert a `CsrMatrix<f64>` to `CscMatrix<f64>` using a two-pass algorithm.
fn csr_to_csc_f64(a: &CsrMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();

    // First pass: count non-zeros per column.
    let mut col_counts = vec![0usize; n];
    for i in 0..m {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        for idx in start..end {
            col_counts[a.col_indices()[idx]] += 1;
        }
    }

    // Build col_ptrs from counts.
    let mut col_ptrs = vec![0usize; n + 1];
    for j in 0..n {
        col_ptrs[j + 1] = col_ptrs[j] + col_counts[j];
    }
    let nnz = col_ptrs[n];

    // Second pass: fill row_indices and values.
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    let mut fill = vec![0usize; n];

    for i in 0..m {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        for idx in start..end {
            let j = a.col_indices()[idx];
            let pos = col_ptrs[j] + fill[j];
            row_indices[pos] = i;
            values[pos] = a.values()[idx];
            fill[j] += 1;
        }
    }

    // Safety: col_ptrs, row_indices, values are consistent by construction.
    unsafe { CscMatrix::new_unchecked(m, n, col_ptrs, row_indices, values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `CsrMatrix<f64>` from (row, col, val) triplets.
    /// Duplicate entries are not supported; each (row, col) must be unique.
    fn build_csr(m: usize, n: usize, triplets: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut row_counts = vec![0usize; m];
        for &(r, _, _) in triplets {
            row_counts[r] += 1;
        }
        let mut row_ptrs = vec![0usize; m + 1];
        for i in 0..m {
            row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
        }
        let nnz = triplets.len();
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut fill = vec![0usize; m];
        for &(r, c, v) in triplets {
            let pos = row_ptrs[r] + fill[r];
            col_indices[pos] = c;
            values[pos] = v;
            fill[r] += 1;
        }
        CsrMatrix::new(m, n, row_ptrs, col_indices, values).expect("valid CSR matrix")
    }

    /// Compute A*x via CSR SpMV.
    fn spmv_csr(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let m = a.nrows();
        let mut y = vec![0.0f64; m];
        for i in 0..m {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                y[i] += a.values()[idx] * x[a.col_indices()[idx]];
            }
        }
        y
    }

    /// L2 norm of a vector.
    fn norm2(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_sparse_qr_3x3_identity() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let a = build_csr(3, 3, &triplets);
        let qr = SparseQr::compute(&a).expect("QR of identity should succeed");

        let b = vec![3.0, 5.0, 7.0];
        let x = qr.solve(&b).expect("solve of identity should succeed");

        assert_eq!(x.len(), 3);
        for i in 0..3 {
            assert!(
                (x[i] - b[i]).abs() < 1e-10,
                "identity solve: x[{i}]={} should equal b[{i}]={}",
                x[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_tridiagonal() {
        // 5x5 tridiagonal: main diag = 4, off-diags = -1 (1D discrete Laplacian).
        let mut triplets = Vec::new();
        for i in 0..5usize {
            triplets.push((i, i, 4.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i < 4 {
                triplets.push((i, i + 1, -1.0));
            }
        }
        let a = build_csr(5, 5, &triplets);
        let qr = SparseQr::compute(&a).expect("QR of tridiagonal should succeed");

        let b = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let x = qr.solve(&b).expect("tridiagonal solve should succeed");

        let ax = spmv_csr(&a, &x);
        let residual: f64 = (0..5).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(
            residual < 1e-9,
            "tridiagonal residual {residual} exceeds tolerance"
        );
    }

    #[test]
    fn test_sparse_qr_overdetermined() {
        // 10x5 overdetermined system with b exactly in the column space of A.
        // A[i][j] = 1 / (1 + |i - j|) for i in 0..10, j in 0..5 (entries > 1e-3).
        let mut triplets = Vec::new();
        for i in 0..10usize {
            for j in 0..5usize {
                let v = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                if v > 1e-3 {
                    triplets.push((i, j, v));
                }
            }
        }
        let a = build_csr(10, 5, &triplets);

        let x_true = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let b = spmv_csr(&a, &x_true);

        let qr = SparseQr::compute(&a).expect("overdetermined QR should succeed");
        let x = qr
            .solve_least_squares(&b)
            .expect("overdetermined solve should succeed");

        let ax = spmv_csr(&a, &x);
        let residual = norm2(&(0..10).map(|i| ax[i] - b[i]).collect::<Vec<_>>());
        assert!(
            residual < 1e-8,
            "overdetermined residual {residual} exceeds tolerance"
        );
    }

    #[test]
    fn test_sparse_qr_rank() {
        // Rank-deficient 3x3 matrix: column 2 = column 0 + column 1.
        // A = [1  0  1]
        //     [0  1  1]
        //     [1  1  2]
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 1.0),
            (1, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ];
        let a = build_csr(3, 3, &triplets);

        let config = SparseQrConfig {
            column_ordering: true,
            drop_tol: 1e-14,
            min_diagonal: 1e-10,
        };
        let result = SparseQr::compute_with_config(&a, config);
        assert!(
            matches!(result, Err(SparseQrError::RankDeficient { .. })),
            "expected RankDeficient error, got: {result:?}"
        );
        if let Err(SparseQrError::RankDeficient { rank, n }) = result {
            assert!(rank < n, "rank {rank} should be less than n {n}");
            assert_eq!(n, 3);
        }
    }

    #[test]
    fn test_sparse_qr_condition_number() {
        // Well-conditioned 3x3: diagonally dominant.
        let well = build_csr(
            3,
            3,
            &[
                (0, 0, 10.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 10.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 10.0),
            ],
        );
        let qr_well = SparseQr::compute(&well).expect("well-conditioned QR should succeed");
        let cond_well = qr_well.condition_number_estimate();

        // Ill-conditioned 3x3: rows nearly linearly dependent.
        let ill = build_csr(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 1.0 + 1e-7),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
                (2, 2, 1.0 + 2e-7),
            ],
        );
        let qr_ill = SparseQr::compute(&ill).expect("ill-conditioned QR should succeed");
        let cond_ill = qr_ill.condition_number_estimate();

        assert!(
            cond_well >= 1.0,
            "condition number must be >= 1, got {cond_well}"
        );
        assert!(
            cond_ill > cond_well,
            "ill-conditioned matrix should have larger condition number: \
             cond_ill={cond_ill}, cond_well={cond_well}"
        );
    }

    #[test]
    fn test_sparse_qr_exact_solve() {
        // A = [2 1 0; 1 3 1; 0 1 2], b = [5; 10; 5].
        // Verify that QR solve gives A*x == b to floating-point precision.
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
        ];
        let a = build_csr(3, 3, &triplets);
        let b = vec![5.0, 10.0, 5.0];

        let qr = SparseQr::compute(&a).expect("exact solve QR should succeed");
        let x = qr.solve(&b).expect("exact solve should succeed");

        let ax = spmv_csr(&a, &x);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-9,
                "exact solve: A*x[{i}]={} should equal b[{i}]={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_r_is_upper_triangular() {
        // R must be structurally upper triangular in every column.
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 2.0),
            (3, 0, 1.0),
            (3, 2, 2.0),
        ];
        let a = build_csr(4, 3, &triplets);
        let qr = SparseQr::compute(&a).expect("QR should succeed");
        let r = &qr.r_factor;
        for j in 0..r.ncols() {
            let start = r.col_ptrs()[j];
            let end = r.col_ptrs()[j + 1];
            for idx in start..end {
                assert!(
                    r.row_indices()[idx] <= j,
                    "R not upper triangular at column {j}"
                );
            }
            // The diagonal is stored last.
            if start < end {
                assert_eq!(r.row_indices()[end - 1], j, "diagonal must be stored last");
            }
        }
    }

    #[test]
    fn test_sparse_qr_large_banded() {
        // A genuinely sparse 400x400 pentadiagonal SPD-ish system. A dense
        // m x n working array would be 160000 entries; the sparse QR keeps the
        // working set proportional to the (banded) nonzero structure.
        let dim = 400usize;
        let mut triplets = Vec::new();
        for i in 0..dim {
            triplets.push((i, i, 6.0));
            if i >= 1 {
                triplets.push((i, i - 1, -1.0));
            }
            if i + 1 < dim {
                triplets.push((i, i + 1, -1.0));
            }
            if i >= 2 {
                triplets.push((i, i - 2, -0.5));
            }
            if i + 2 < dim {
                triplets.push((i, i + 2, -0.5));
            }
        }
        let a = build_csr(dim, dim, &triplets);

        let x_true: Vec<f64> = (0..dim).map(|i| ((i % 7) as f64) - 3.0).collect();
        let b = spmv_csr(&a, &x_true);

        let qr = SparseQr::compute(&a).expect("large banded QR should succeed");
        let x = qr.solve(&b).expect("large banded solve should succeed");

        // R must remain sparse (banded), not the dense dim*(dim+1)/2 upper block.
        assert!(
            qr.r_factor.values().len() < dim * dim / 4,
            "R should stay sparse, got {} nonzeros",
            qr.r_factor.values().len()
        );

        let ax = spmv_csr(&a, &x);
        let residual = norm2(&(0..dim).map(|i| ax[i] - b[i]).collect::<Vec<_>>());
        assert!(
            residual < 1e-7,
            "large banded residual {residual} exceeds tolerance"
        );
    }

    #[test]
    fn test_sparse_qr_no_column_ordering() {
        // Same system solved with the natural ordering must also be correct.
        let triplets = vec![
            (0, 0, 3.0),
            (0, 2, 1.0),
            (1, 1, 4.0),
            (2, 0, 1.0),
            (2, 2, 5.0),
        ];
        let a = build_csr(3, 3, &triplets);
        let config = SparseQrConfig {
            column_ordering: false,
            ..Default::default()
        };
        let qr = SparseQr::compute_with_config(&a, config).expect("QR should succeed");
        assert_eq!(qr.col_perm, vec![0, 1, 2]);

        let b = vec![7.0, 8.0, 11.0];
        let x = qr.solve(&b).expect("solve should succeed");
        let ax = spmv_csr(&a, &x);
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-9, "natural-order solve failed");
        }
    }
}
