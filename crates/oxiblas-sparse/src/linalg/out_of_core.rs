//! Out-of-core sparse matrix factorization.
//!
//! This module provides factorization algorithms for sparse matrices that are
//! larger than available RAM. Blocks of the matrix are written to temporary
//! disk files and processed block by block using a right-looking algorithm.
//!
//! # Overview
//!
//! - [`OutOfCoreLu`]: Block LU factorization via right-looking algorithm
//! - [`OutOfCoreCholesky`]: Block Cholesky factorization for SPD matrices
//! - [`OutOfCoreConfig`]: Shared configuration (block size, memory budget, temp dir)
//! - [`OutOfCoreSolver`]: Common trait for out-of-core solvers
//!
//! # Example
//!
//! ```no_run
//! use oxiblas_sparse::linalg::out_of_core::{OutOfCoreLu, OutOfCoreConfig};
//! use oxiblas_sparse::CsrMatrix;
//!
//! let cfg = OutOfCoreConfig::default();
//! let mut lu = OutOfCoreLu::<f64>::new(cfg);
//! // build a CsrMatrix<f64> `mat`, then:
//! // lu.factorize_csr(&mat).unwrap();
//! // let sol = lu.solve(&rhs).unwrap();
//! ```

use std::io::{BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Monotonic counter used to generate unique file name prefixes.
static INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Return a prefix string that is unique per-process and per-solver-instance.
fn unique_prefix(tag: &str) -> String {
    let id = INSTANCE_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let pid = std::process::id();
    format!("oxiblas_{tag}_{pid}_{id}")
}

use crate::csr::CsrMatrix;
use oxiblas_core::scalar::Scalar;

// ────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during out-of-core factorization.
#[derive(Debug)]
pub enum OutOfCoreError {
    /// An I/O error occurred while reading or writing a block file.
    Io(std::io::Error),
    /// The factorization failed (e.g. singular pivot).
    Factorization(String),
    /// The input matrix is invalid.
    InvalidMatrix(String),
    /// A block-size related error.
    BlockSizeError(String),
}

impl core::fmt::Display for OutOfCoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Factorization(s) => write!(f, "Factorization error: {s}"),
            Self::InvalidMatrix(s) => write!(f, "Invalid matrix: {s}"),
            Self::BlockSizeError(s) => write!(f, "Block size error: {s}"),
        }
    }
}

impl std::error::Error for OutOfCoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OutOfCoreError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for out-of-core solvers.
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Number of rows per disk block (default 256).
    pub block_size: usize,
    /// Maximum in-core memory budget in megabytes (default 512).
    pub max_memory_mb: usize,
    /// Directory in which temporary block files are written.
    pub temp_dir: PathBuf,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            max_memory_mb: 512,
            temp_dir: std::env::temp_dir(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Trait
// ────────────────────────────────────────────────────────────────────────────

/// Common interface for out-of-core direct solvers.
pub trait OutOfCoreSolver {
    /// The scalar type (e.g. `f64`).
    type Scalar: Copy + Scalar + oxiblas_core::Scalar;

    /// Factorize the given CSR matrix.
    fn factorize(&mut self, csr: &CsrMatrix<Self::Scalar>) -> Result<(), OutOfCoreError>;

    /// Solve A x = rhs using the stored factorization.
    fn solve(&self, rhs: &[Self::Scalar], sol: &mut [Self::Scalar]) -> Result<(), OutOfCoreError>;

    /// Approximate in-core memory footprint in bytes.
    fn memory_usage_bytes(&self) -> usize;
}

// ────────────────────────────────────────────────────────────────────────────
// Low-level block I/O helpers
// ────────────────────────────────────────────────────────────────────────────

/// Write a slice of f64 values to a file as raw little-endian bytes.
fn write_dense_block(path: &Path, data: &[f64]) -> Result<(), OutOfCoreError> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &v in data {
        writer.write_all(&v.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a file of raw little-endian f64 bytes into a Vec.
fn read_dense_block(path: &Path) -> Result<Vec<f64>, OutOfCoreError> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    if buf.len() % 8 != 0 {
        return Err(OutOfCoreError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "block file size is not a multiple of 8 bytes",
        )));
    }
    let mut out = Vec::with_capacity(buf.len() / 8);
    for chunk in buf.chunks_exact(8) {
        let arr: [u8; 8] = chunk.try_into().map_err(|_| {
            OutOfCoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chunk conversion failed",
            ))
        })?;
        out.push(f64::from_le_bytes(arr));
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────────
// TempBlockFile – owns a single block file and deletes it on drop
// ────────────────────────────────────────────────────────────────────────────

/// A temporary dense block stored on disk.
struct TempBlockFile {
    path: PathBuf,
}

impl TempBlockFile {
    fn new(path: PathBuf, _nrows: usize, _ncols: usize) -> Self {
        Self { path }
    }
}

impl Drop for TempBlockFile {
    fn drop(&mut self) {
        // Best-effort removal; ignore errors at drop time.
        let _ = std::fs::remove_file(&self.path);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dense LU helpers (no external LAPACK – pure Rust)
// ────────────────────────────────────────────────────────────────────────────

/// In-place dense LU with partial pivoting on a square `n×n` row-major matrix.
/// Returns the pivot permutation `piv` where `piv[k]` is the row swapped with `k`.
/// Returns an error if a zero pivot is encountered.
fn dense_lu_inplace(a: &mut [f64], n: usize) -> Result<Vec<usize>, OutOfCoreError> {
    debug_assert_eq!(a.len(), n * n);
    let mut piv = vec![0usize; n];

    for k in 0..n {
        // Find pivot row.
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        piv[k] = max_row;

        if max_val == 0.0 {
            return Err(OutOfCoreError::Factorization(format!(
                "zero pivot at column {k}"
            )));
        }

        // Swap rows k and max_row.
        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
        }

        let a_kk = a[k * n + k];
        // Compute multipliers and update trailing submatrix.
        for i in (k + 1)..n {
            let m = a[i * n + k] / a_kk;
            a[i * n + k] = m;
            for j in (k + 1)..n {
                let delta = m * a[k * n + j];
                a[i * n + j] -= delta;
            }
        }
    }
    Ok(piv)
}

/// Forward substitution L y = b where L is unit lower triangular (n×n row-major).
fn forward_sub(l: &[f64], b: &[f64], y: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        y[i] = s; // diagonal of L is 1
    }
}

/// Backward substitution U x = y where U is upper triangular (n×n row-major).
fn backward_sub(u: &[f64], y: &[f64], x: &mut [f64], n: usize) {
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= u[i * n + j] * x[j];
        }
        x[i] = s / u[i * n + i];
    }
}

/// Apply pivot permutation to vector in-place.
fn apply_pivots(v: &mut [f64], piv: &[usize]) {
    for (k, &p) in piv.iter().enumerate() {
        v.swap(k, p);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dense Cholesky helpers
// ────────────────────────────────────────────────────────────────────────────

/// In-place Cholesky factorization A = L L^T on an n×n row-major SPD matrix.
/// Only the lower triangle of `a` is used; on exit, `a` holds L in its lower triangle.
fn dense_cholesky_inplace(a: &mut [f64], n: usize) -> Result<(), OutOfCoreError> {
    debug_assert_eq!(a.len(), n * n);
    for j in 0..n {
        // Compute diagonal element.
        let mut diag = a[j * n + j];
        for k in 0..j {
            let l_jk = a[j * n + k];
            diag -= l_jk * l_jk;
        }
        if diag <= 0.0 {
            return Err(OutOfCoreError::Factorization(format!(
                "matrix is not positive definite at column {j}"
            )));
        }
        let l_jj = diag.sqrt();
        a[j * n + j] = l_jj;

        // Compute column j of L below diagonal.
        for i in (j + 1)..n {
            let mut val = a[i * n + j];
            for k in 0..j {
                val -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = val / l_jj;
        }
    }
    Ok(())
}

/// Forward substitution L y = b (n×n lower triangular, row-major).
fn chol_forward_sub(l: &[f64], b: &[f64], y: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        y[i] = s / l[i * n + i];
    }
}

/// Backward substitution L^T x = y (n×n lower triangular stored, row-major).
fn chol_backward_sub(l: &[f64], y: &[f64], x: &mut [f64], n: usize) {
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = s / l[i * n + i];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dense GEMM helper  C -= A * B  (row-major, all dimensions explicit)
// ────────────────────────────────────────────────────────────────────────────

/// `c[m×p] -= a[m×k] * b[k×p]` (row-major).
fn dgemm_sub(c: &mut [f64], a: &[f64], b: &[f64], m: usize, k: usize, p: usize) {
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..p {
                c[i * p + j] -= a_il * b[l * p + j];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Utility: extract a dense rectangular sub-block from CSR
// ────────────────────────────────────────────────────────────────────────────

/// Extract rows `[row_start, row_end)` and columns `[col_start, col_end)` of a
/// CSR matrix into a dense row-major buffer.
fn extract_dense_block(
    csr: &CsrMatrix<f64>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> Vec<f64> {
    let nrows = row_end - row_start;
    let ncols = col_end - col_start;
    let mut buf = vec![0.0f64; nrows * ncols];
    for i in row_start..row_end {
        for (col, val) in csr.row_iter(i) {
            if col >= col_start && col < col_end {
                buf[(i - row_start) * ncols + (col - col_start)] = *val;
            }
        }
    }
    buf
}

// ────────────────────────────────────────────────────────────────────────────
// OutOfCoreLu
// ────────────────────────────────────────────────────────────────────────────

/// Out-of-core LU factorization for `f64` sparse matrices.
///
/// Uses a right-looking block LU algorithm:
/// 1. For each block-column `k`:
///    a. Factorize the diagonal block with partial-pivot dense LU.
///    b. Apply row swaps to off-diagonal blocks in the same block-row.
///    c. Compute the multiplier blocks below the diagonal block (TRSM).
///    d. Update the trailing sub-matrix (GEMM).
///    e. Write all modified blocks back to disk.
///
/// The factored blocks are stored in temporary files in `config.temp_dir`.
pub struct OutOfCoreLu<T> {
    config: OutOfCoreConfig,
    /// Unique prefix for temp file names (avoids collisions across concurrent instances).
    prefix: String,
    /// Matrix dimension.
    n: usize,
    /// One file per block-row × block-column tile (lower-triangular / diagonal).
    block_files: Vec<TempBlockFile>,
    /// Number of block-columns (== number of blocks along one dimension).
    nb: usize,
    /// Block sizes (last block may be smaller).
    block_sizes: Vec<usize>,
    /// Per-diagonal-block pivot permutations (length n, assembled from block pivots).
    pivots: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T> OutOfCoreLu<T> {
    /// Create a new (unfactorized) `OutOfCoreLu` with the given configuration.
    pub fn new(config: OutOfCoreConfig) -> Self {
        Self {
            config,
            prefix: unique_prefix("lu"),
            n: 0,
            block_files: Vec::new(),
            nb: 0,
            block_sizes: Vec::new(),
            pivots: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl OutOfCoreLu<f64> {
    // ── public API ──────────────────────────────────────────────────────────

    /// Factorize a square CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square, if a zero pivot is
    /// encountered, or if an I/O error occurs while writing block files.
    pub fn factorize_csr(&mut self, csr: &CsrMatrix<f64>) -> Result<(), OutOfCoreError> {
        let n = csr.nrows();
        if n != csr.ncols() {
            return Err(OutOfCoreError::InvalidMatrix(format!(
                "matrix must be square, got {}×{}",
                csr.nrows(),
                csr.ncols()
            )));
        }
        if n == 0 {
            return Err(OutOfCoreError::InvalidMatrix(
                "matrix dimension is zero".into(),
            ));
        }

        let bs = self.config.block_size.max(1);
        let nb = n.div_ceil(bs);
        let mut block_sizes = Vec::with_capacity(nb);
        for b in 0..nb {
            let start = b * bs;
            let end = (start + bs).min(n);
            block_sizes.push(end - start);
        }

        self.n = n;
        self.nb = nb;
        self.block_sizes = block_sizes.clone();
        self.pivots = (0..n).collect();

        // ── Write initial tiles to disk ──────────────────────────────────
        // We store all nb×nb tiles (upper and lower triangular both needed
        // for the right-looking update).  Tile (i,j) = rows of block i,
        // columns of block j.
        self.block_files.clear();
        let mut col_starts = Vec::with_capacity(nb);
        let mut row_starts = Vec::with_capacity(nb);
        {
            let mut acc = 0usize;
            for &sz in &block_sizes {
                row_starts.push(acc);
                col_starts.push(acc);
                acc += sz;
            }
        }

        for i in 0..nb {
            for j in 0..nb {
                let rs = row_starts[i];
                let re = rs + block_sizes[i];
                let cs = col_starts[j];
                let ce = cs + block_sizes[j];
                let data = extract_dense_block(csr, rs, re, cs, ce);
                let path = self
                    .config
                    .temp_dir
                    .join(format!("{}_block_{}_{}.bin", self.prefix, i, j));
                write_dense_block(&path, &data)?;
                self.block_files
                    .push(TempBlockFile::new(path, block_sizes[i], block_sizes[j]));
            }
        }

        // ── Right-looking block LU ───────────────────────────────────────
        for k in 0..nb {
            let bs_k = block_sizes[k];

            // Load diagonal block A[k,k].
            let mut diag = self.read_block(k, k)?;

            // Factorize diagonal block.
            let local_piv = dense_lu_inplace(&mut diag, bs_k)?;

            // Store global pivots.
            let k_start = row_starts[k];
            for (local_row, &swap_row) in local_piv.iter().enumerate() {
                let global_row = k_start + local_row;
                let global_swap = k_start + swap_row;
                self.pivots.swap(global_row, global_swap);
            }

            // Apply row swaps to blocks A[k, j] for j > k.
            for j in (k + 1)..nb {
                let mut right = self.read_block(k, j)?;
                let bs_j = block_sizes[j];
                for (local_row, &swap_row) in local_piv.iter().enumerate() {
                    if swap_row != local_row {
                        for col in 0..bs_j {
                            right.swap(local_row * bs_j + col, swap_row * bs_j + col);
                        }
                    }
                }
                self.write_block(k, j, &right)?;
            }

            // Write back factored diagonal block.
            self.write_block(k, k, &diag)?;

            // Compute multipliers: L[i,k] = A[i,k] * U[k,k]^{-1}  for i > k
            for i in (k + 1)..nb {
                let bs_i = block_sizes[i];
                let mut left = self.read_block(i, k)?;
                // Solve left * U[k,k] = A[i,k]  →  left = A[i,k] * U^{-1}
                // Equivalently, for each row r of left: solve U^T l_r^T = a_r^T
                // but since we want row operations: trsm right-side with upper U.
                // We use column-wise back-sub on each row.
                let u = &diag; // upper part
                for r in 0..bs_i {
                    for c in 0..bs_k {
                        let mut val = left[r * bs_k + c];
                        for p in 0..c {
                            val -= left[r * bs_k + p] * u[p * bs_k + c];
                        }
                        left[r * bs_k + c] = val / u[c * bs_k + c];
                    }
                }
                self.write_block(i, k, &left)?;

                // Update trailing: A[i,j] -= L[i,k] * U[k,j]  for j > k
                for j in (k + 1)..nb {
                    let bs_j = block_sizes[j];
                    let mut trail = self.read_block(i, j)?;
                    let upper = self.read_block(k, j)?;
                    dgemm_sub(&mut trail, &left, &upper, bs_i, bs_k, bs_j);
                    self.write_block(i, j, &trail)?;
                }
            }
        }

        Ok(())
    }

    /// Solve A x = rhs using the stored factorization.
    ///
    /// # Errors
    ///
    /// Returns an error if the factorization has not been performed, if
    /// `rhs.len() != n`, or on I/O failure.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, OutOfCoreError> {
        if self.n == 0 {
            return Err(OutOfCoreError::Factorization(
                "factorize_csr must be called before solve".into(),
            ));
        }
        if rhs.len() != self.n {
            return Err(OutOfCoreError::InvalidMatrix(format!(
                "rhs length {} does not match matrix dimension {}",
                rhs.len(),
                self.n
            )));
        }

        let nb = self.nb;
        let block_sizes = &self.block_sizes;
        let mut row_starts = Vec::with_capacity(nb);
        {
            let mut acc = 0usize;
            for &sz in block_sizes {
                row_starts.push(acc);
                acc += sz;
            }
        }

        // Apply pivot permutation to rhs.
        let mut b = rhs.to_vec();
        apply_pivots(&mut b, &self.pivots);

        // Forward substitution block-by-block (unit lower triangular L).
        let mut y = vec![0.0f64; self.n];
        for i in 0..nb {
            let bs_i = block_sizes[i];
            let rs_i = row_starts[i];
            // Start with b[i..i+bs_i].
            let mut rhs_blk: Vec<f64> = b[rs_i..rs_i + bs_i].to_vec();
            // Subtract L[i,j] * y[j] for j < i.
            for j in 0..i {
                let bs_j = block_sizes[j];
                let rs_j = row_starts[j];
                let l_ij = self.read_block(i, j)?;
                for r in 0..bs_i {
                    for c in 0..bs_j {
                        rhs_blk[r] -= l_ij[r * bs_j + c] * y[rs_j + c];
                    }
                }
            }
            // Diagonal block L[i,i] is unit lower triangular.
            let diag = self.read_block(i, i)?;
            let mut y_blk = vec![0.0f64; bs_i];
            forward_sub(&diag, &rhs_blk, &mut y_blk, bs_i);
            y[rs_i..rs_i + bs_i].copy_from_slice(&y_blk);
        }

        // Backward substitution block-by-block (upper triangular U).
        let mut x = vec![0.0f64; self.n];
        for i in (0..nb).rev() {
            let bs_i = block_sizes[i];
            let rs_i = row_starts[i];
            let mut rhs_blk: Vec<f64> = y[rs_i..rs_i + bs_i].to_vec();
            // Subtract U[i,j] * x[j] for j > i.
            for j in (i + 1)..nb {
                let bs_j = block_sizes[j];
                let rs_j = row_starts[j];
                let u_ij = self.read_block(i, j)?;
                for r in 0..bs_i {
                    for c in 0..bs_j {
                        rhs_blk[r] -= u_ij[r * bs_j + c] * x[rs_j + c];
                    }
                }
            }
            // Diagonal block U[i,i] is upper triangular.
            let diag = self.read_block(i, i)?;
            let mut x_blk = vec![0.0f64; bs_i];
            backward_sub(&diag, &rhs_blk, &mut x_blk, bs_i);
            x[rs_i..rs_i + bs_i].copy_from_slice(&x_blk);
        }

        Ok(x)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Index into `block_files` for tile (i, j).
    #[inline]
    fn tile_index(&self, i: usize, j: usize) -> usize {
        i * self.nb + j
    }

    fn read_block(&self, i: usize, j: usize) -> Result<Vec<f64>, OutOfCoreError> {
        let idx = self.tile_index(i, j);
        read_dense_block(&self.block_files[idx].path)
    }

    fn write_block(&self, i: usize, j: usize, data: &[f64]) -> Result<(), OutOfCoreError> {
        let idx = self.tile_index(i, j);
        write_dense_block(&self.block_files[idx].path, data)
    }
}

impl OutOfCoreSolver for OutOfCoreLu<f64> {
    type Scalar = f64;

    fn factorize(&mut self, csr: &CsrMatrix<f64>) -> Result<(), OutOfCoreError> {
        self.factorize_csr(csr)
    }

    fn solve(&self, rhs: &[f64], sol: &mut [f64]) -> Result<(), OutOfCoreError> {
        let x = OutOfCoreLu::solve(self, rhs)?;
        if sol.len() != x.len() {
            return Err(OutOfCoreError::InvalidMatrix(format!(
                "sol buffer length {} does not match solution length {}",
                sol.len(),
                x.len()
            )));
        }
        sol.copy_from_slice(&x);
        Ok(())
    }

    fn memory_usage_bytes(&self) -> usize {
        // In-core: pivot vector + block_sizes + block_file metadata.
        self.pivots.len() * std::mem::size_of::<usize>()
            + self.block_sizes.len() * std::mem::size_of::<usize>()
            + self.block_files.len() * std::mem::size_of::<TempBlockFile>()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// OutOfCoreCholesky
// ────────────────────────────────────────────────────────────────────────────

/// Out-of-core Cholesky factorization for symmetric positive definite `f64` matrices.
///
/// Uses a blocked right-looking algorithm:
/// 1. For each block-column `k`:
///    a. Factorize the diagonal block (dense Cholesky).
///    b. Solve off-diagonal blocks below the diagonal: L\[i,k\] = A\[i,k\] * L\[k,k\]^{-T} (TRSM).
///    c. Update the trailing sub-matrix: A\[i,j\] -= L\[i,k\] * L\[j,k\]^T (SYRK / GEMM).
///    d. Write modified blocks back to disk.
pub struct OutOfCoreCholesky<T> {
    config: OutOfCoreConfig,
    /// Unique prefix for temp file names.
    prefix: String,
    n: usize,
    nb: usize,
    block_sizes: Vec<usize>,
    /// Lower-triangular blocks only; tile (i, j) with i >= j.
    block_files: Vec<Option<TempBlockFile>>,
    _phantom: PhantomData<T>,
}

impl<T> OutOfCoreCholesky<T> {
    /// Create a new (unfactorized) `OutOfCoreCholesky` with the given configuration.
    pub fn new(config: OutOfCoreConfig) -> Self {
        Self {
            config,
            prefix: unique_prefix("chol"),
            n: 0,
            nb: 0,
            block_sizes: Vec::new(),
            block_files: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl OutOfCoreCholesky<f64> {
    // ── public API ──────────────────────────────────────────────────────────

    /// Factorize a symmetric positive definite CSR matrix.
    ///
    /// Only the lower triangle of the input matrix is referenced.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square, if it is not positive
    /// definite, or on I/O failure.
    pub fn factorize_csr(&mut self, csr: &CsrMatrix<f64>) -> Result<(), OutOfCoreError> {
        let n = csr.nrows();
        if n != csr.ncols() {
            return Err(OutOfCoreError::InvalidMatrix(format!(
                "matrix must be square, got {}×{}",
                csr.nrows(),
                csr.ncols()
            )));
        }
        if n == 0 {
            return Err(OutOfCoreError::InvalidMatrix(
                "matrix dimension is zero".into(),
            ));
        }

        let bs = self.config.block_size.max(1);
        let nb = n.div_ceil(bs);
        let mut block_sizes = Vec::with_capacity(nb);
        let mut row_starts = Vec::with_capacity(nb);
        {
            let mut acc = 0usize;
            for b in 0..nb {
                let start = b * bs;
                let end = (start + bs).min(n);
                row_starts.push(acc);
                block_sizes.push(end - start);
                acc += end - start;
            }
        }

        self.n = n;
        self.nb = nb;
        self.block_sizes = block_sizes.clone();

        // Allocate nb×nb option slots; only lower-triangular (i >= j) used.
        self.block_files = (0..nb * nb).map(|_| None).collect();

        // Write initial lower-triangular tiles.
        for i in 0..nb {
            for j in 0..=i {
                let rs = row_starts[i];
                let re = rs + block_sizes[i];
                let cs = row_starts[j];
                let ce = cs + block_sizes[j];
                let data = extract_dense_block(csr, rs, re, cs, ce);
                let path = self
                    .config
                    .temp_dir
                    .join(format!("{}_block_{}_{}.bin", self.prefix, i, j));
                write_dense_block(&path, &data)?;
                self.block_files[i * nb + j] =
                    Some(TempBlockFile::new(path, block_sizes[i], block_sizes[j]));
            }
        }

        // Blocked right-looking Cholesky.
        for k in 0..nb {
            let bs_k = block_sizes[k];

            // Load and factorize diagonal block.
            let mut diag = self.read_block(k, k)?;
            dense_cholesky_inplace(&mut diag, bs_k)?;
            self.write_block(k, k, &diag)?;

            // Compute L[i,k] = A[i,k] * L[k,k]^{-T}  for i > k.
            for i in (k + 1)..nb {
                let bs_i = block_sizes[i];
                let mut panel = self.read_block(i, k)?;
                // Solve panel * L_kk^T = A_ik  ↔  for each row r, solve L_kk^T col = a_r^T
                // i.e. forward substitution on each row with L^T (lower triangular by columns).
                for r in 0..bs_i {
                    for c in 0..bs_k {
                        let mut val = panel[r * bs_k + c];
                        for p in 0..c {
                            val -= panel[r * bs_k + p] * diag[c * bs_k + p];
                        }
                        panel[r * bs_k + c] = val / diag[c * bs_k + c];
                    }
                }
                self.write_block(i, k, &panel)?;

                // Update A[i,j] -= L[i,k] * L[j,k]^T  for k < j <= i.
                for j in (k + 1)..=i {
                    let bs_j = block_sizes[j];
                    let mut trail = self.read_block(i, j)?;
                    let l_jk = self.read_block(j, k)?;
                    // C -= A * B^T  where A = panel (bs_i × bs_k), B = l_jk (bs_j × bs_k).
                    for r in 0..bs_i {
                        for s in 0..bs_j {
                            let mut sum = 0.0f64;
                            for p in 0..bs_k {
                                sum += panel[r * bs_k + p] * l_jk[s * bs_k + p];
                            }
                            trail[r * bs_j + s] -= sum;
                        }
                    }
                    self.write_block(i, j, &trail)?;
                }
            }
        }

        Ok(())
    }

    /// Solve A x = rhs using the stored Cholesky factorization (L L^T x = b).
    ///
    /// # Errors
    ///
    /// Returns an error if `factorize_csr` has not been called, if
    /// `rhs.len() != n`, or on I/O failure.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, OutOfCoreError> {
        if self.n == 0 {
            return Err(OutOfCoreError::Factorization(
                "factorize_csr must be called before solve".into(),
            ));
        }
        if rhs.len() != self.n {
            return Err(OutOfCoreError::InvalidMatrix(format!(
                "rhs length {} does not match matrix dimension {}",
                rhs.len(),
                self.n
            )));
        }

        let nb = self.nb;
        let block_sizes = &self.block_sizes;
        let mut row_starts = Vec::with_capacity(nb);
        {
            let mut acc = 0usize;
            for &sz in block_sizes {
                row_starts.push(acc);
                acc += sz;
            }
        }

        // Forward substitution: L y = b.
        let mut y = vec![0.0f64; self.n];
        for i in 0..nb {
            let bs_i = block_sizes[i];
            let rs_i = row_starts[i];
            let mut rhs_blk: Vec<f64> = rhs[rs_i..rs_i + bs_i].to_vec();
            for j in 0..i {
                let bs_j = block_sizes[j];
                let rs_j = row_starts[j];
                let l_ij = self.read_block(i, j)?;
                for r in 0..bs_i {
                    for c in 0..bs_j {
                        rhs_blk[r] -= l_ij[r * bs_j + c] * y[rs_j + c];
                    }
                }
            }
            let diag = self.read_block(i, i)?;
            let mut y_blk = vec![0.0f64; bs_i];
            chol_forward_sub(&diag, &rhs_blk, &mut y_blk, bs_i);
            y[rs_i..rs_i + bs_i].copy_from_slice(&y_blk);
        }

        // Backward substitution: L^T x = y.
        let mut x = vec![0.0f64; self.n];
        for i in (0..nb).rev() {
            let bs_i = block_sizes[i];
            let rs_i = row_starts[i];
            let mut rhs_blk: Vec<f64> = y[rs_i..rs_i + bs_i].to_vec();
            // L^T[i, j] = L[j, i]  →  read block (j, i) for j > i.
            for j in (i + 1)..nb {
                let bs_j = block_sizes[j];
                let rs_j = row_starts[j];
                // L[j, i] has shape bs_j × bs_i.
                let l_ji = self.read_block(j, i)?;
                // Contribution: L^T[i, j] * x[j]  = L[j,i]^T * x[j]
                for r in 0..bs_i {
                    for c in 0..bs_j {
                        rhs_blk[r] -= l_ji[c * bs_i + r] * x[rs_j + c];
                    }
                }
            }
            let diag = self.read_block(i, i)?;
            let mut x_blk = vec![0.0f64; bs_i];
            chol_backward_sub(&diag, &rhs_blk, &mut x_blk, bs_i);
            x[rs_i..rs_i + bs_i].copy_from_slice(&x_blk);
        }

        Ok(x)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn read_block(&self, i: usize, j: usize) -> Result<Vec<f64>, OutOfCoreError> {
        let idx = i * self.nb + j;
        match &self.block_files[idx] {
            Some(f) => read_dense_block(&f.path),
            None => Err(OutOfCoreError::Factorization(format!(
                "block ({i},{j}) not initialized"
            ))),
        }
    }

    fn write_block(&self, i: usize, j: usize, data: &[f64]) -> Result<(), OutOfCoreError> {
        let idx = i * self.nb + j;
        match &self.block_files[idx] {
            Some(f) => write_dense_block(&f.path, data),
            None => Err(OutOfCoreError::Factorization(format!(
                "block ({i},{j}) not initialized"
            ))),
        }
    }
}

impl OutOfCoreSolver for OutOfCoreCholesky<f64> {
    type Scalar = f64;

    fn factorize(&mut self, csr: &CsrMatrix<f64>) -> Result<(), OutOfCoreError> {
        self.factorize_csr(csr)
    }

    fn solve(&self, rhs: &[f64], sol: &mut [f64]) -> Result<(), OutOfCoreError> {
        let x = OutOfCoreCholesky::solve(self, rhs)?;
        if sol.len() != x.len() {
            return Err(OutOfCoreError::InvalidMatrix(format!(
                "sol buffer length {} does not match solution length {}",
                sol.len(),
                x.len()
            )));
        }
        sol.copy_from_slice(&x);
        Ok(())
    }

    fn memory_usage_bytes(&self) -> usize {
        self.block_sizes.len() * std::mem::size_of::<usize>()
            + self.block_files.len() * std::mem::size_of::<Option<TempBlockFile>>()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 10×10 tridiagonal CSR matrix with diagonal 4 and off-diagonals -1.
    fn tridiagonal_csr(n: usize) -> CsrMatrix<f64> {
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptrs.push(0);
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0f64);
            }
            col_indices.push(i);
            values.push(4.0f64);
            if i + 1 < n {
                col_indices.push(i + 1);
                values.push(-1.0f64);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).expect("valid tridiagonal construction")
    }

    fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_out_of_core_config_default() {
        let cfg = OutOfCoreConfig::default();
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.max_memory_mb, 512);
        // temp_dir must be a valid path.
        assert!(cfg.temp_dir.to_str().is_some());
    }

    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_out_of_core_lu_small() {
        let n = 10usize;
        let mat = tridiagonal_csr(n);

        // Use a small block size to exercise multi-block code paths.
        let cfg = OutOfCoreConfig {
            block_size: 3,
            max_memory_mb: 64,
            temp_dir: std::env::temp_dir(),
        };
        let mut lu = OutOfCoreLu::<f64>::new(cfg);
        lu.factorize_csr(&mat).expect("factorize succeeded");

        // rhs = all-ones.
        let rhs: Vec<f64> = vec![1.0; n];
        let sol = lu.solve(&rhs).expect("solve succeeded");
        assert_eq!(sol.len(), n);

        // Verify A x ≈ rhs by re-multiplying.
        let mut ax = vec![0.0f64; n];
        for i in 0..n {
            for (c, v) in mat.row_iter(i) {
                ax[i] += v * sol[c];
            }
        }
        let err = max_abs_error(&ax, &rhs);
        assert!(
            err < 1e-10,
            "residual norm too large: {err:.3e}; ax={ax:?}, rhs={rhs:?}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_out_of_core_cholesky_small() {
        let n = 10usize;
        let mat = tridiagonal_csr(n); // symmetric, diagonally dominant → SPD

        let cfg = OutOfCoreConfig {
            block_size: 3,
            max_memory_mb: 64,
            temp_dir: std::env::temp_dir(),
        };
        let mut chol = OutOfCoreCholesky::<f64>::new(cfg);
        chol.factorize_csr(&mat)
            .expect("cholesky factorize succeeded");

        let rhs: Vec<f64> = vec![1.0; n];
        let sol = chol.solve(&rhs).expect("cholesky solve succeeded");
        assert_eq!(sol.len(), n);

        // Verify A x ≈ rhs.
        let mut ax = vec![0.0f64; n];
        for i in 0..n {
            for (c, v) in mat.row_iter(i) {
                ax[i] += v * sol[c];
            }
        }
        let err = max_abs_error(&ax, &rhs);
        assert!(
            err < 1e-10,
            "residual norm too large: {err:.3e}; ax={ax:?}, rhs={rhs:?}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_temp_files_cleaned_up() {
        let n = 4usize;
        let mat = tridiagonal_csr(n);

        let tmp = std::env::temp_dir();
        let cfg = OutOfCoreConfig {
            block_size: 2,
            max_memory_mb: 64,
            temp_dir: tmp.clone(),
        };

        // Collect paths before drop.
        let paths: Vec<PathBuf> = {
            let mut lu = OutOfCoreLu::<f64>::new(cfg.clone());
            lu.factorize_csr(&mat).expect("factorize succeeded");
            lu.block_files.iter().map(|f| f.path.clone()).collect()
        }; // lu is dropped here → TempBlockFile::drop fires

        for p in &paths {
            assert!(
                !p.exists(),
                "temp file was not removed on drop: {}",
                p.display()
            );
        }

        // Also test Cholesky cleanup.
        let chol_paths: Vec<PathBuf> = {
            let mut chol = OutOfCoreCholesky::<f64>::new(cfg);
            chol.factorize_csr(&mat).expect("factorize succeeded");
            chol.block_files
                .iter()
                .flatten()
                .map(|f| f.path.clone())
                .collect()
        };

        for p in &chol_paths {
            assert!(
                !p.exists(),
                "cholesky temp file was not removed on drop: {}",
                p.display()
            );
        }
    }
}
