//! Workspace size query functions (LAPACK-style lwork queries).
//!
//! This module provides workspace size queries for LAPACK operations,
//! allowing users to pre-allocate workspace buffers for repeated operations.
//!
//! In LAPACK, workspace queries are performed by calling routines with `lwork = -1`,
//! which returns the optimal workspace size without performing the computation.
//! This module provides equivalent functionality through dedicated query functions.
//!
//! # Benefits of Pre-allocation
//!
//! - **Performance**: Amortize allocation costs when calling routines multiple times
//! - **Memory Control**: Use custom allocators or stack-allocated buffers
//! - **Predictability**: Know memory requirements before computation
//!
//! # Usage Pattern
//!
//! ```ignore
//! use oxiblas_lapack::workspace::{WorkspaceQuery, qr_workspace};
//!
//! // Query workspace size
//! let ws = qr_workspace(100, 50);
//! println!("Optimal workspace: {} elements", ws.optimal);
//! println!("Minimum workspace: {} elements", ws.minimum);
//!
//! // Pre-allocate workspace (for repeated operations)
//! let mut work: Vec<f64> = vec![0.0; ws.optimal];
//!
//! // Use the workspace in computations (when workspace-accepting APIs are available)
//! ```
//!
//! # Workspace Size Structures
//!
//! Most query functions return a `WorkspaceQuery` struct containing:
//! - `optimal`: The optimal workspace size for best performance
//! - `minimum`: The minimum workspace size required for correctness
//!
//! Some decompositions require multiple workspace arrays (real and integer),
//! which are returned in specialized structures.

use std::cmp::max;

/// Result of a workspace size query.
///
/// Contains both optimal and minimum workspace sizes.
/// Using the optimal size typically provides better performance through
/// better cache utilization and blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceQuery {
    /// Optimal workspace size (elements, not bytes).
    /// Using this size provides best performance.
    pub optimal: usize,
    /// Minimum workspace size required for correctness.
    /// Using less than this will cause computation to fail.
    pub minimum: usize,
}

impl WorkspaceQuery {
    /// Creates a new workspace query result.
    #[inline]
    pub const fn new(optimal: usize, minimum: usize) -> Self {
        Self { optimal, minimum }
    }

    /// Creates a workspace query where optimal equals minimum.
    #[inline]
    pub const fn fixed(size: usize) -> Self {
        Self {
            optimal: size,
            minimum: size,
        }
    }

    /// Returns the optimal workspace size.
    #[inline]
    pub const fn optimal(&self) -> usize {
        self.optimal
    }

    /// Returns the minimum workspace size.
    #[inline]
    pub const fn minimum(&self) -> usize {
        self.minimum
    }

    /// Calculates workspace size in bytes for a given element type.
    #[inline]
    pub const fn optimal_bytes<T>(&self) -> usize {
        self.optimal * std::mem::size_of::<T>()
    }

    /// Calculates minimum workspace size in bytes for a given element type.
    #[inline]
    pub const fn minimum_bytes<T>(&self) -> usize {
        self.minimum * std::mem::size_of::<T>()
    }
}

/// Workspace query for operations requiring separate real and integer workspaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceQueryWithInt {
    /// Real workspace requirements.
    pub real_work: WorkspaceQuery,
    /// Integer workspace requirements.
    pub int_work: WorkspaceQuery,
}

impl WorkspaceQueryWithInt {
    /// Creates a new workspace query with both real and integer workspace.
    #[inline]
    pub const fn new(real_work: WorkspaceQuery, int_work: WorkspaceQuery) -> Self {
        Self {
            real_work,
            int_work,
        }
    }
}

/// Workspace query for SVD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SvdWorkspaceQuery {
    /// Main workspace requirements.
    pub work: WorkspaceQuery,
    /// Integer workspace requirements (for divide-and-conquer).
    pub iwork: Option<usize>,
}

impl SvdWorkspaceQuery {
    /// Creates a new SVD workspace query.
    #[inline]
    pub const fn new(work: WorkspaceQuery, iwork: Option<usize>) -> Self {
        Self { work, iwork }
    }
}

/// Workspace query for eigenvalue operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvdWorkspaceQuery {
    /// Main workspace requirements.
    pub work: WorkspaceQuery,
    /// Integer workspace requirements (for divide-and-conquer).
    pub iwork: Option<usize>,
    /// Real workspace for complex eigenvalue problems.
    pub rwork: Option<usize>,
}

impl EvdWorkspaceQuery {
    /// Creates a new eigenvalue workspace query.
    #[inline]
    pub const fn new(work: WorkspaceQuery, iwork: Option<usize>, rwork: Option<usize>) -> Self {
        Self { work, iwork, rwork }
    }
}

// ============================================================================
// LU Factorization Workspace Queries
// ============================================================================

/// Query workspace size for LU factorization (GETRF).
///
/// LU factorization with partial pivoting requires minimal workspace
/// as the factorization is performed in-place.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Workspace query result (typically very small for in-place LU).
#[inline]
pub fn lu_workspace(m: usize, n: usize) -> WorkspaceQuery {
    // LU factorization is in-place, but we need space for pivot indices
    // The pivot array is stored separately, so work is minimal
    let k = m.min(n);
    // Only need a small work array for blocking (nb columns)
    let nb = optimal_block_size_lu(m, n);
    let optimal = nb * k;
    let minimum = 1; // Can work with minimal extra space
    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for LU solve (GETRS).
///
/// # Arguments
///
/// * `n` - Matrix dimension (n×n system)
/// * `nrhs` - Number of right-hand sides
#[inline]
pub fn lu_solve_workspace(n: usize, nrhs: usize) -> WorkspaceQuery {
    // Solve is in-place with triangular operations
    // May use small blocking workspace
    let nb = optimal_block_size_trsm(n, nrhs);
    let optimal = nb * nrhs;
    let minimum = 1;
    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for band LU factorization (GBTRF).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
#[inline]
pub fn band_lu_workspace(_n: usize, kl: usize, ku: usize) -> WorkspaceQuery {
    // Band LU needs extra super-diagonals for fill-in during pivoting
    // AB should be (2*kl + ku + 1) × n, extra kl for fill-in
    let bandwidth = 2 * kl + ku + 1;
    let optimal = bandwidth;
    let minimum = bandwidth;
    WorkspaceQuery::new(optimal, minimum)
}

// ============================================================================
// Cholesky Factorization Workspace Queries
// ============================================================================

/// Query workspace size for Cholesky factorization (POTRF).
///
/// Cholesky factorization is performed in-place and requires minimal workspace.
///
/// # Arguments
///
/// * `n` - Matrix dimension (n×n positive definite matrix)
#[inline]
pub fn cholesky_workspace(n: usize) -> WorkspaceQuery {
    // Cholesky is in-place, blocking uses nb columns
    let nb = optimal_block_size_cholesky(n);
    let optimal = nb * n;
    let minimum = 1;
    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for Cholesky solve (POTRS).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `nrhs` - Number of right-hand sides
#[inline]
pub fn cholesky_solve_workspace(n: usize, nrhs: usize) -> WorkspaceQuery {
    // Solve uses triangular solves, minimal workspace
    let nb = optimal_block_size_trsm(n, nrhs);
    let optimal = nb * nrhs;
    let minimum = 1;
    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for LDL^T factorization.
///
/// # Arguments
///
/// * `n` - Matrix dimension
#[inline]
pub fn ldlt_workspace(n: usize) -> WorkspaceQuery {
    // LDLT needs workspace for block operations
    let nb = optimal_block_size_cholesky(n);
    let optimal = n * nb;
    let minimum = n;
    WorkspaceQuery::new(optimal, minimum)
}

// ============================================================================
// QR Factorization Workspace Queries
// ============================================================================

/// Query workspace size for QR factorization (GEQRF).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Workspace query result. The tau array (Householder scalars) is stored separately.
#[inline]
pub fn qr_workspace(m: usize, n: usize) -> WorkspaceQuery {
    let _k = m.min(n);
    let nb = optimal_block_size_qr(m, n);

    // Optimal: blocked algorithm uses nb×n workspace for T matrix
    // Plus nb columns for intermediate results
    let optimal = nb * n + nb * m;

    // Minimum: column-by-column Householder requires n workspace
    let minimum = n.max(1);

    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for generating Q from QR (ORGQR/UNGQR).
///
/// # Arguments
///
/// * `m` - Number of rows of Q
/// * `n` - Number of columns of Q
/// * `k` - Number of Householder reflections (columns used from QR)
#[inline]
pub fn orgqr_workspace(m: usize, n: usize, _k: usize) -> WorkspaceQuery {
    let nb = optimal_block_size_qr(m, n);

    // Optimal: blocked algorithm
    let optimal = n * nb;

    // Minimum: unblocked algorithm
    let minimum = n.max(1);

    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for multiplying by Q from QR (ORMQR/UNMQR).
///
/// # Arguments
///
/// * `side` - 'L' for Q*C or Q'*C, 'R' for C*Q or C*Q'
/// * `m` - Number of rows of C
/// * `n` - Number of columns of C
/// * `k` - Number of Householder reflections
#[inline]
pub fn ormqr_workspace(side: char, m: usize, n: usize, _k: usize) -> WorkspaceQuery {
    let nw = if side == 'L' || side == 'l' { n } else { m };
    let nb = optimal_block_size_qr(m, n);

    // Optimal: blocked algorithm
    let optimal = nw * nb;

    // Minimum: unblocked
    let minimum = nw.max(1);

    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for QR with column pivoting (GEQP3).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
#[inline]
pub fn qr_pivot_workspace(m: usize, n: usize) -> WorkspaceQuery {
    let nb = optimal_block_size_qr(m, n);

    // Needs extra space for column norms
    let optimal = 2 * n + (n + 1) * nb;
    let minimum = 3 * n + 1;

    WorkspaceQuery::new(optimal, minimum)
}

// ============================================================================
// SVD Workspace Queries
// ============================================================================

/// Query workspace size for SVD (GESVD) using Jacobi algorithm.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `compute_u` - Whether to compute left singular vectors
/// * `compute_vt` - Whether to compute right singular vectors
#[inline]
pub fn svd_workspace(m: usize, n: usize, compute_u: bool, compute_vt: bool) -> SvdWorkspaceQuery {
    let k = m.min(n);

    // Jacobi SVD workspace requirements
    let mut optimal = k * k; // For B = A^T * A or A * A^T

    // Additional for singular vectors
    if compute_u {
        optimal += m * k;
    }
    if compute_vt {
        optimal += n * k;
    }

    // Column norms workspace
    optimal += 2 * n;

    let minimum = max(1, 3 * k + max(m, n));

    SvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None)
}

/// Query workspace size for divide-and-conquer SVD (GESDD).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `job` - 'A' for all, 'S' for thin, 'O' for overwrite, 'N' for no vectors
#[inline]
pub fn svd_dc_workspace(m: usize, n: usize, job: char) -> SvdWorkspaceQuery {
    let mn = m.min(n);
    let mx = m.max(n);

    // Divide-and-conquer has larger workspace requirements
    let (optimal, minimum) = match job {
        'N' | 'n' => {
            // No singular vectors
            let opt = 3 * mn + max(mx, 7 * mn);
            let min = 3 * mn + max(mx, 6 * mn);
            (opt, min)
        }
        'O' | 'o' => {
            // Overwrite A
            let opt = 3 * mn * mn + max(mx, 5 * mn * mn + 4 * mn);
            let min = 3 * mn + max(mx, 5 * mn * mn + 4 * mn);
            (opt, min)
        }
        'S' | 's' => {
            // Thin SVD
            let opt = 4 * mn * mn + max(mx, 5 * mn * mn + 4 * mn);
            let min = 3 * mn + max(mx, 5 * mn * mn + 4 * mn);
            (opt, min)
        }
        _ => {
            // Full SVD ('A')
            let opt = 4 * mn * mn + max(mx, 5 * mn * mn + 4 * mn);
            let min = 3 * mn + max(mx, 5 * mn * mn + 4 * mn);
            (opt, min)
        }
    };

    // Integer workspace for D&C
    let iwork = 8 * mn;

    SvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), Some(iwork))
}

/// Query workspace size for bidiagonal reduction (GEBRD).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
#[inline]
pub fn bidiag_workspace(m: usize, n: usize) -> WorkspaceQuery {
    let _k = m.min(n);
    let nb = optimal_block_size_bidiag(m, n);

    // Optimal: blocked algorithm
    let optimal = (m + n) * nb;

    // Minimum: unblocked
    let minimum = max(m, n);

    WorkspaceQuery::new(optimal, minimum)
}

// ============================================================================
// Eigenvalue Decomposition Workspace Queries
// ============================================================================

/// Query workspace size for symmetric eigenvalue decomposition (SYEV/HEEV).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_vectors` - Whether to compute eigenvectors
#[inline]
pub fn symmetric_evd_workspace(n: usize, compute_vectors: bool) -> EvdWorkspaceQuery {
    let nb = optimal_block_size_evd(n);

    let (optimal, minimum) = if compute_vectors {
        // Need more workspace for eigenvectors
        let opt = (nb + 2) * n;
        let min = 3 * n - 1;
        (opt, min)
    } else {
        // Only eigenvalues
        let opt = (nb + 1) * n;
        let min = max(1, 3 * n - 1);
        (opt, min)
    };

    EvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None, None)
}

/// Query workspace size for symmetric divide-and-conquer EVD (SYEVD/HEEVD).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_vectors` - Whether to compute eigenvectors
#[inline]
pub fn symmetric_evd_dc_workspace(n: usize, compute_vectors: bool) -> EvdWorkspaceQuery {
    let (work_opt, work_min, iwork) = if compute_vectors {
        // D&C with eigenvectors
        let opt = 1 + 6 * n + 2 * n * n;
        let min = 1 + 6 * n + 2 * n * n;
        let iw = 3 + 5 * n;
        (opt, min, Some(iw))
    } else {
        // D&C without eigenvectors
        let opt = 2 * n + 1;
        let min = 2 * n + 1;
        (opt, min, Some(1))
    };

    EvdWorkspaceQuery::new(WorkspaceQuery::new(work_opt, work_min), iwork, None)
}

/// Query workspace size for Hermitian eigenvalue decomposition.
///
/// For complex Hermitian matrices, additional real workspace is needed.
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_vectors` - Whether to compute eigenvectors
#[inline]
pub fn hermitian_evd_workspace(n: usize, compute_vectors: bool) -> EvdWorkspaceQuery {
    let nb = optimal_block_size_evd(n);

    let (optimal, minimum) = if compute_vectors {
        let opt = (nb + 1) * n;
        let min = 2 * n - 1;
        (opt, min)
    } else {
        let opt = nb * n;
        let min = max(1, 2 * n - 1);
        (opt, min)
    };

    // Complex Hermitian needs real workspace for eigenvalues
    let rwork = 3 * n - 2;

    EvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None, Some(rwork))
}

/// Query workspace size for general eigenvalue decomposition (GEEV).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_left` - Whether to compute left eigenvectors
/// * `compute_right` - Whether to compute right eigenvectors
#[inline]
pub fn general_evd_workspace(
    n: usize,
    compute_left: bool,
    compute_right: bool,
) -> EvdWorkspaceQuery {
    let optimal: usize;
    let minimum: usize;

    if compute_left || compute_right {
        optimal = (2 + optimal_block_size_evd(n)) * n;
        minimum = 4 * n;
    } else {
        optimal = (2 + optimal_block_size_evd(n)) * n;
        minimum = 3 * n;
    }

    EvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None, None)
}

/// Query workspace size for Schur decomposition (GEES).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_vectors` - Whether to compute Schur vectors
#[inline]
pub fn schur_workspace(n: usize, compute_vectors: bool) -> EvdWorkspaceQuery {
    let nb = optimal_block_size_hessenberg(n);

    let (optimal, minimum) = if compute_vectors {
        let opt = n * (1 + nb);
        let min = max(1, 3 * n);
        (opt, min)
    } else {
        let opt = n * (1 + nb);
        let min = max(1, 2 * n);
        (opt, min)
    };

    EvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None, None)
}

/// Query workspace size for Hessenberg reduction (GEHRD).
///
/// # Arguments
///
/// * `n` - Matrix dimension
#[inline]
pub fn hessenberg_workspace(n: usize) -> WorkspaceQuery {
    let nb = optimal_block_size_hessenberg(n);

    let optimal = n * nb;
    let minimum = max(1, n);

    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for generalized eigenvalue problem (GGEV).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_left` - Whether to compute left eigenvectors
/// * `compute_right` - Whether to compute right eigenvectors
#[inline]
pub fn generalized_evd_workspace(
    n: usize,
    _compute_left: bool,
    _compute_right: bool,
) -> EvdWorkspaceQuery {
    let optimal = max(1, 2 * n + max(6 * n, n * n));
    let minimum = max(1, 8 * n);

    EvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None, None)
}

/// Query workspace size for generalized Schur (QZ) decomposition (GGES).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `compute_left` - Whether to compute left Schur vectors
/// * `compute_right` - Whether to compute right Schur vectors
#[inline]
pub fn qz_workspace(n: usize, _compute_left: bool, _compute_right: bool) -> EvdWorkspaceQuery {
    let optimal = max(1, 8 * n + 16);
    let minimum = max(1, 8 * n);

    EvdWorkspaceQuery::new(WorkspaceQuery::new(optimal, minimum), None, None)
}

// ============================================================================
// Solve Workspace Queries
// ============================================================================

/// Query workspace size for least squares solve (GELS).
///
/// # Arguments
///
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `nrhs` - Number of right-hand sides
#[inline]
pub fn least_squares_workspace(m: usize, n: usize, nrhs: usize) -> WorkspaceQuery {
    let k = m.min(n);
    let nb = optimal_block_size_qr(m, n);

    // Uses QR factorization for overdetermined, LQ for underdetermined
    let optimal = k + max(k, nrhs) * nb;
    let minimum = k + max(1, max(m, max(n, nrhs)));

    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for triangular solve (TRSM).
///
/// # Arguments
///
/// * `n` - Matrix dimension
/// * `nrhs` - Number of right-hand sides
#[inline]
pub fn triangular_solve_workspace(n: usize, nrhs: usize) -> WorkspaceQuery {
    // TRSM is typically in-place, minimal workspace
    let nb = optimal_block_size_trsm(n, nrhs);
    let optimal = nb * nrhs;
    let minimum = 1;

    WorkspaceQuery::new(optimal, minimum)
}

/// Query workspace size for tridiagonal solve (GTSV).
///
/// # Arguments
///
/// * `n` - System size
/// * `nrhs` - Number of right-hand sides
#[inline]
pub fn tridiagonal_solve_workspace(n: usize, _nrhs: usize) -> WorkspaceQuery {
    // Tridiagonal solvers need minimal extra workspace
    // Extra storage for the factorization's fill-in
    let optimal = 2 * n;
    let minimum = n;

    WorkspaceQuery::new(optimal, minimum)
}

// ============================================================================
// Optimal Block Size Estimation
// ============================================================================

/// Get optimal block size for LU factorization.
///
/// Block sizes are tuned for typical cache hierarchies.
#[inline]
fn optimal_block_size_lu(m: usize, n: usize) -> usize {
    let k = m.min(n);
    if k < 64 {
        k
    } else if k < 256 {
        32
    } else if k < 1024 {
        64
    } else {
        128
    }
}

/// Get optimal block size for Cholesky factorization.
#[inline]
fn optimal_block_size_cholesky(n: usize) -> usize {
    if n < 64 {
        n
    } else if n < 256 {
        32
    } else if n < 1024 {
        64
    } else {
        128
    }
}

/// Get optimal block size for QR factorization.
#[inline]
fn optimal_block_size_qr(m: usize, n: usize) -> usize {
    let k = m.min(n);
    if k < 32 {
        k.max(1)
    } else if k < 128 {
        32
    } else if k < 512 {
        48
    } else {
        64
    }
}

/// Get optimal block size for bidiagonal reduction.
#[inline]
fn optimal_block_size_bidiag(m: usize, n: usize) -> usize {
    let k = m.min(n);
    if k < 32 {
        k.max(1)
    } else if k < 256 {
        32
    } else {
        48
    }
}

/// Get optimal block size for eigenvalue decomposition.
#[inline]
fn optimal_block_size_evd(n: usize) -> usize {
    if n < 32 {
        n.max(1)
    } else if n < 256 {
        32
    } else if n < 1024 {
        48
    } else {
        64
    }
}

/// Get optimal block size for Hessenberg reduction.
#[inline]
fn optimal_block_size_hessenberg(n: usize) -> usize {
    if n < 32 {
        n.max(1)
    } else if n < 256 {
        32
    } else {
        48
    }
}

/// Get optimal block size for triangular solve.
#[inline]
fn optimal_block_size_trsm(n: usize, nrhs: usize) -> usize {
    let k = n.min(nrhs);
    if k < 32 {
        k.max(1)
    } else if k < 128 {
        32
    } else {
        64
    }
}

// ============================================================================
// Workspace Buffer Trait
// ============================================================================

/// Trait for types that can be used as workspace buffers.
pub trait Workspace {
    /// The element type stored in the workspace.
    type Element;

    /// Returns a mutable slice to the workspace data.
    fn as_mut_slice(&mut self) -> &mut [Self::Element];

    /// Returns the length of the workspace.
    fn len(&self) -> usize;

    /// Returns whether the workspace is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Workspace for Vec<T> {
    type Element = T;

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T> Workspace for [T] {
    type Element = T;

    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T, const N: usize> Workspace for [T; N] {
    type Element = T;

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }

    fn len(&self) -> usize {
        N
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_query_new() {
        let ws = WorkspaceQuery::new(100, 50);
        assert_eq!(ws.optimal(), 100);
        assert_eq!(ws.minimum(), 50);
    }

    #[test]
    fn test_workspace_query_fixed() {
        let ws = WorkspaceQuery::fixed(64);
        assert_eq!(ws.optimal(), 64);
        assert_eq!(ws.minimum(), 64);
    }

    #[test]
    fn test_workspace_query_bytes() {
        let ws = WorkspaceQuery::new(100, 50);
        assert_eq!(ws.optimal_bytes::<f64>(), 800);
        assert_eq!(ws.minimum_bytes::<f64>(), 400);
        assert_eq!(ws.optimal_bytes::<f32>(), 400);
    }

    #[test]
    fn test_lu_workspace() {
        let ws = lu_workspace(100, 100);
        assert!(ws.optimal >= ws.minimum);
        assert!(ws.minimum >= 1);
    }

    #[test]
    fn test_qr_workspace() {
        let ws = qr_workspace(100, 50);
        assert!(ws.optimal >= ws.minimum);
        assert!(ws.minimum >= 1);

        // Larger matrices should have larger workspace
        let ws_large = qr_workspace(1000, 500);
        assert!(ws_large.optimal > ws.optimal);
    }

    #[test]
    fn test_svd_workspace() {
        let ws = svd_workspace(100, 50, true, true);
        assert!(ws.work.optimal >= ws.work.minimum);
        assert!(ws.iwork.is_none()); // Jacobi doesn't need iwork
    }

    #[test]
    fn test_svd_dc_workspace() {
        let ws = svd_dc_workspace(100, 50, 'A');
        assert!(ws.work.optimal >= ws.work.minimum);
        assert!(ws.iwork.is_some()); // D&C needs iwork
        assert!(ws.iwork.unwrap() > 0);
    }

    #[test]
    fn test_symmetric_evd_workspace() {
        let ws_novecs = symmetric_evd_workspace(100, false);
        let ws_vecs = symmetric_evd_workspace(100, true);

        // With vectors needs more workspace
        assert!(ws_vecs.work.optimal >= ws_novecs.work.optimal);
    }

    #[test]
    fn test_symmetric_evd_dc_workspace() {
        let ws = symmetric_evd_dc_workspace(100, true);
        assert!(ws.work.optimal >= ws.work.minimum);
        assert!(ws.iwork.is_some());
    }

    #[test]
    fn test_general_evd_workspace() {
        let ws = general_evd_workspace(100, true, true);
        assert!(ws.work.optimal >= ws.work.minimum);
    }

    #[test]
    fn test_schur_workspace() {
        let ws = schur_workspace(100, true);
        assert!(ws.work.optimal >= ws.work.minimum);
    }

    #[test]
    fn test_hessenberg_workspace() {
        let ws = hessenberg_workspace(100);
        assert!(ws.optimal >= ws.minimum);
    }

    #[test]
    fn test_cholesky_workspace() {
        let ws = cholesky_workspace(100);
        assert!(ws.optimal >= ws.minimum);
    }

    #[test]
    fn test_least_squares_workspace() {
        let ws = least_squares_workspace(100, 50, 10);
        assert!(ws.optimal >= ws.minimum);
    }

    #[test]
    fn test_block_sizes() {
        // Small matrices should have smaller block sizes
        assert!(optimal_block_size_lu(10, 10) <= 32);
        assert!(optimal_block_size_qr(10, 10) <= 32);

        // Large matrices should use larger blocks
        assert!(optimal_block_size_lu(2000, 2000) >= 64);
        assert!(optimal_block_size_qr(2000, 2000) >= 48);
    }

    #[test]
    fn test_workspace_trait() {
        let mut vec_work: Vec<f64> = vec![0.0; 100];
        assert_eq!(vec_work.len(), 100);
        assert!(!vec_work.is_empty());

        let slice = vec_work.as_mut_slice();
        slice[0] = 1.0;
        assert_eq!(vec_work[0], 1.0);
    }

    #[test]
    fn test_workspace_trait_array() {
        let mut arr_work: [f64; 64] = [0.0; 64];
        assert_eq!(Workspace::len(&arr_work), 64);

        let slice = arr_work.as_mut_slice();
        slice[0] = 2.0;
        assert_eq!(arr_work[0], 2.0);
    }

    #[test]
    fn test_small_matrix_workspace() {
        // Edge cases with small matrices
        let ws = qr_workspace(1, 1);
        assert!(ws.minimum >= 1);

        let ws = svd_workspace(1, 1, true, true);
        assert!(ws.work.minimum >= 1);

        let ws = symmetric_evd_workspace(1, true);
        assert!(ws.work.minimum >= 1);
    }

    #[test]
    fn test_qr_pivot_workspace() {
        let ws = qr_pivot_workspace(100, 50);
        assert!(ws.optimal >= ws.minimum);
        assert!(ws.minimum > 3 * 50);
    }

    #[test]
    fn test_orgqr_workspace() {
        let ws = orgqr_workspace(100, 50, 50);
        assert!(ws.optimal >= ws.minimum);
    }

    #[test]
    fn test_ormqr_workspace() {
        let ws_left = ormqr_workspace('L', 100, 50, 50);
        let ws_right = ormqr_workspace('R', 100, 50, 50);

        assert!(ws_left.optimal >= ws_left.minimum);
        assert!(ws_right.optimal >= ws_right.minimum);
    }

    #[test]
    fn test_generalized_evd_workspace() {
        let ws = generalized_evd_workspace(100, true, true);
        assert!(ws.work.optimal >= ws.work.minimum);
    }

    #[test]
    fn test_qz_workspace() {
        let ws = qz_workspace(100, true, true);
        assert!(ws.work.optimal >= ws.work.minimum);
    }

    #[test]
    fn test_band_lu_workspace() {
        let ws = band_lu_workspace(100, 3, 3);
        assert_eq!(ws.optimal, 2 * 3 + 3 + 1); // 2*kl + ku + 1
    }

    #[test]
    fn test_bidiag_workspace() {
        let ws = bidiag_workspace(100, 50);
        assert!(ws.optimal >= ws.minimum);
    }

    #[test]
    fn test_tridiagonal_solve_workspace() {
        let ws = tridiagonal_solve_workspace(100, 1);
        assert!(ws.optimal >= ws.minimum);
    }

    #[test]
    fn test_hermitian_evd_workspace() {
        let ws = hermitian_evd_workspace(100, true);
        assert!(ws.work.optimal >= ws.work.minimum);
        assert!(ws.rwork.is_some()); // Complex needs rwork
    }
}
