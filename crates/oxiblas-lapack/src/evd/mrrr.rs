//! MRRR Algorithm (Multiple Relatively Robust Representations).
//!
//! This module implements the MRRR algorithm for computing eigenvalues and
//! eigenvectors of symmetric tridiagonal matrices. Unlike naive inverse
//! iteration, MRRR computes each eigenvector in `O(n)` work from a *twisted
//! factorization* of a relatively robust representation (RRR), so the full set
//! of eigenvectors costs `O(n^2)` and requires **no** global Gram–Schmidt
//! reorthogonalization for eigenvalues that are relatively well separated.
//!
//! # Algorithm overview
//!
//! The implementation follows Dhillon's thesis and the structure of LAPACK's
//! `dstemr`/`dlarrv`/`dlar1v`:
//!
//! 1. **Splitting.** The tridiagonal is split at negligible off-diagonal
//!    entries into unreduced blocks. Each block has simple (distinct)
//!    eigenvalues, and eigenvectors of different blocks are orthogonal by
//!    disjoint support. This is what handles exact/multiple eigenvalues (which
//!    can only arise from a decoupled block) correctly.
//! 2. **Base representation.** For each block a base RRR `L D L^T = T - sigma0 I`
//!    is formed with `sigma0` just below the spectrum, so the factorization is
//!    positive definite and hence relatively robust.
//! 3. **Representation tree.** Eigenvalues are refined relative to the current
//!    RRR (bisection on the LDL^T Sturm/negcount). Eigenvalues whose *relative
//!    gap* w.r.t. the current RRR is large are "singletons": their eigenvector
//!    is read off a twisted factorization directly. A group of eigenvalues with
//!    small relative gaps forms a cluster; the algorithm shifts to a child RRR
//!    `L' D' L'^T = L D L^T - tau I` placed just outside the cluster (which
//!    makes the cluster's shifted eigenvalues small and their *relative* gaps
//!    large) and recurses.
//! 4. **Twisted eigenvector (`dlar1v`).** For a shift `mu` close to an
//!    eigenvalue of the RRR, a differential stationary qd sweep (top-down) and a
//!    differential progressive qd sweep (bottom-up) are combined. The twist
//!    index `r = argmin_k |gamma_k|` gives the pivot of smallest magnitude; the
//!    eigenvector solves the twisted triangular system with `z_r = 1`. A few
//!    Rayleigh-quotient corrections polish the eigenpair.
//!
//! The only place a small *local* Gram–Schmidt is used is the last-resort branch
//! for a residual cluster that cannot be separated further (eigenvalues equal to
//! working accuracy). For an unreduced block this branch is effectively never
//! taken; when it is, orthonormalizing the tiny cluster block is exact because
//! the eigenvalues coincide to working precision (any orthonormal basis of the
//! invariant subspace is a valid set of eigenvectors). This is *not* the global
//! `O(n^3)` reorthogonalization that a naive inverse-iteration solver needs.
//!
//! # Key features
//!
//! - `O(n^2)` complexity for computing all eigenvectors of a block.
//! - High accuracy for clustered eigenvalues via child representations.
//! - No global reorthogonalization on the twisted-factorization path.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::MrrrEvd;
//!
//! let diagonal = vec![2.0, 3.0, 4.0, 5.0];
//! let off_diagonal = vec![1.0, 1.0, 1.0];
//!
//! let evd = MrrrEvd::compute(&diagonal, &off_diagonal).unwrap();
//! let eigenvalues = evd.eigenvalues();
//! let eigenvectors = evd.eigenvectors();
//! ```
//!
//! # References
//!
//! - I. S. Dhillon, "A New O(n²) Algorithm for the Symmetric Tridiagonal
//!   Eigenvalue/Eigenvector Problem", Ph.D. thesis, UC Berkeley, 1997.
//! - I. S. Dhillon and B. N. Parlett, "Multiple representations to compute
//!   orthogonal eigenvectors of symmetric tridiagonal matrices", Linear Algebra
//!   and its Applications, 2004.
//! - I. S. Dhillon, B. N. Parlett and C. Vömel, "The design and implementation
//!   of the MRRR algorithm", ACM TOMS, 2006.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

/// Maximum iterations for bisection refinement.
const MAX_BISECTION_ITER: usize = 100;

/// Maximum number of Rayleigh-quotient corrections per eigenvector.
const MAX_RQI_ITER: usize = 8;

/// Maximum depth of the representation tree (bounds recursion on hard clusters).
const MAX_TREE_DEPTH: usize = 64;

/// Maximum number of shift attempts when constructing a child RRR.
const MAX_SHIFT_ATTEMPTS: usize = 16;

/// Minimum relative gap for an eigenvalue to be treated as a singleton
/// (isolated) w.r.t. the current representation. Matches LAPACK's `MINRGP`.
const MIN_RELATIVE_GAP: f64 = 1.0e-3;

/// Error type for MRRR algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MrrrError {
    /// Empty input.
    EmptyInput,
    /// Dimension mismatch between diagonal and off-diagonal.
    DimensionMismatch,
    /// LDL factorization failed (matrix became indefinite).
    LdlFactorizationFailed,
    /// Eigenvalue computation did not converge.
    EigenvalueNotConverged,
    /// Eigenvector computation failed.
    EigenvectorComputationFailed,
    /// Invalid index range.
    InvalidIndexRange,
}

impl core::fmt::Display for MrrrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Empty input"),
            Self::DimensionMismatch => write!(f, "Off-diagonal must have length n-1"),
            Self::LdlFactorizationFailed => write!(f, "LDL factorization failed"),
            Self::EigenvalueNotConverged => write!(f, "Eigenvalue computation did not converge"),
            Self::EigenvectorComputationFailed => write!(f, "Eigenvector computation failed"),
            Self::InvalidIndexRange => write!(f, "Invalid index range"),
        }
    }
}

impl std::error::Error for MrrrError {}

/// Returns `mag` with the sign of `sign_of` (branch-based `copysign`).
#[inline]
fn copysign_mag<T: Real>(mag: T, sign_of: T) -> T {
    if sign_of >= T::zero() { mag } else { -mag }
}

/// Scale-dependent numerical parameters shared across the representation tree.
#[derive(Debug, Clone, Copy)]
struct MrrrParams<T> {
    /// Spectral diameter (upper Gershgorin bound minus lower).
    spdiam: T,
    /// Pivot floor used to avoid division by (near-)zero pivots.
    pivmin: T,
    /// Minimum relative gap for the singleton criterion.
    minrgp: T,
    /// Rayleigh-quotient convergence tolerance.
    rqtol: T,
    /// Threshold below which a cluster is treated as numerically degenerate.
    degtol: T,
    /// Machine epsilon.
    eps: T,
}

/// A relatively robust representation `L D L^T` of a shifted tridiagonal.
///
/// `L` is unit lower bidiagonal with sub-diagonal `l`, `D = diag(d)`.
/// The auxiliary products `ld[i] = l[i] * d[i]` and `lld[i] = l[i]^2 * d[i]`
/// are cached because every qd sweep needs them.
#[derive(Debug, Clone)]
struct Rrr<T> {
    /// Diagonal of `D` (length `n`).
    d: Vec<T>,
    /// Sub-diagonal of `L` (length `n-1`).
    l: Vec<T>,
    /// `l[i] * d[i]` (length `n-1`).
    ld: Vec<T>,
    /// `l[i] * l[i] * d[i]` (length `n-1`).
    lld: Vec<T>,
}

impl<T: Field + Real> Rrr<T> {
    /// Factor `T - sigma*I = L D L^T` directly from the tridiagonal entries.
    fn from_tridiagonal(diagonal: &[T], off_diagonal: &[T], sigma: T, pivmin: T) -> Self {
        let n = diagonal.len();
        let mut d = vec![T::zero(); n];
        let mut l = vec![T::zero(); n.saturating_sub(1)];
        let mut ld = vec![T::zero(); n.saturating_sub(1)];
        let mut lld = vec![T::zero(); n.saturating_sub(1)];

        let mut d0 = diagonal[0] - sigma;
        if Scalar::abs(d0) < pivmin {
            d0 = copysign_mag(pivmin, d0);
        }
        d[0] = d0;

        for i in 0..n.saturating_sub(1) {
            let li = off_diagonal[i] / d[i];
            let ldi = li * d[i];
            let lldi = li * ldi;
            l[i] = li;
            ld[i] = ldi;
            lld[i] = lldi;

            let mut dn = (diagonal[i + 1] - sigma) - lldi;
            if Scalar::abs(dn) < pivmin {
                dn = copysign_mag(pivmin, dn);
            }
            d[i + 1] = dn;
        }

        Rrr { d, l, ld, lld }
    }

    /// Form the child representation `self - tau*I = L+ D+ L+^T` via the
    /// differential stationary qd transform (dstqds).
    fn factor_child(&self, tau: T, pivmin: T) -> Self {
        let n = self.d.len();
        let mut d = vec![T::zero(); n];
        let mut l = vec![T::zero(); n.saturating_sub(1)];
        let mut ld = vec![T::zero(); n.saturating_sub(1)];
        let mut lld = vec![T::zero(); n.saturating_sub(1)];

        let mut s = -tau;
        for i in 0..n.saturating_sub(1) {
            let mut dplus = self.d[i] + s;
            if Scalar::abs(dplus) < pivmin {
                dplus = copysign_mag(pivmin, dplus);
            }
            let lplus = self.ld[i] / dplus;
            d[i] = dplus;
            l[i] = lplus;
            ld[i] = lplus * dplus;
            lld[i] = lplus * ld[i];
            s = self.lld[i] * (s / dplus) - tau;
            if !s.is_finite() {
                s = -tau;
            }
        }
        let mut dlast = self.d[n - 1] + s;
        if Scalar::abs(dlast) < pivmin {
            dlast = copysign_mag(pivmin, dlast);
        }
        d[n - 1] = dlast;

        Rrr { d, l, ld, lld }
    }

    /// Number of eigenvalues of `self` strictly less than `mu` (negcount via the
    /// LDL^T Sturm sequence). Tiny pivots are floored to `-pivmin`, which counts
    /// a pivot passing through zero as negative and keeps the count monotone.
    fn neg_count(&self, mu: T, pivmin: T) -> usize {
        let n = self.d.len();
        let mut neg = 0usize;
        let mut s = -mu;
        for i in 0..n - 1 {
            let mut dplus = self.d[i] + s;
            if Scalar::abs(dplus) < pivmin {
                dplus = -pivmin;
            }
            if dplus < T::zero() {
                neg += 1;
            }
            s = self.lld[i] * (s / dplus) - mu;
            if !s.is_finite() {
                s = -mu;
            }
        }
        let mut dlast = self.d[n - 1] + s;
        if Scalar::abs(dlast) < pivmin {
            dlast = -pivmin;
        }
        if dlast < T::zero() {
            neg += 1;
        }
        neg
    }

    /// Compute an (unnormalized) eigenvector from the twisted factorization at
    /// shift `mu` (the LAPACK `dlar1v` kernel).
    ///
    /// Returns `(z, ztz, mingma)` where `z` is the unnormalized eigenvector,
    /// `ztz = z^T z`, and `mingma = gamma_r` is the twist pivot of smallest
    /// magnitude. The Rayleigh-quotient correction is `mingma / ztz`.
    fn twisted_eigenvector(&self, mu: T, pivmin: T) -> (Vec<T>, T, T) {
        let n = self.d.len();

        // Top-down differential stationary qd: LDL^T - mu I = L+ D+ L+^T.
        // s[i] is the auxiliary offset with dplus[i] = d[i] + s[i].
        let mut s = vec![T::zero(); n];
        let mut lplus = vec![T::zero(); n.saturating_sub(1)];
        let mut dplus = vec![T::zero(); n];
        s[0] = -mu;
        for i in 0..n {
            let mut dp = self.d[i] + s[i];
            if Scalar::abs(dp) < pivmin {
                dp = copysign_mag(pivmin, dp);
            }
            dplus[i] = dp;
            if i < n - 1 {
                lplus[i] = self.ld[i] / dp;
                let mut snext = self.lld[i] * (s[i] / dp) - mu;
                if !snext.is_finite() {
                    snext = -mu;
                }
                s[i + 1] = snext;
            }
        }

        // Bottom-up differential progressive qd: LDL^T - mu I = U- D- U-^T.
        // p[i] is the auxiliary offset with dminus[i] = lld[i-1] + p[i].
        let mut p = vec![T::zero(); n];
        let mut uminus = vec![T::zero(); n.saturating_sub(1)];
        p[n - 1] = self.d[n - 1] - mu;
        for i in (0..n - 1).rev() {
            let mut dminus = self.lld[i] + p[i + 1];
            if Scalar::abs(dminus) < pivmin {
                dminus = copysign_mag(pivmin, dminus);
            }
            let tmp = self.d[i] / dminus;
            uminus[i] = self.l[i] * tmp;
            let mut pi = p[i + 1] * tmp - mu;
            if !pi.is_finite() {
                pi = self.d[i] - mu;
            }
            p[i] = pi;
        }

        // Twist pivots gamma_r = s[r] + p[r] + mu; pick the smallest magnitude.
        let mut r = 0usize;
        let mut best = Scalar::abs(s[0] + p[0] + mu);
        for i in 1..n {
            let g = Scalar::abs(s[i] + p[i] + mu);
            if g < best {
                best = g;
                r = i;
            }
        }
        let mingma = s[r] + p[r] + mu;

        // Solve the twisted triangular system: z_r = 1, sweep outward.
        let mut z = vec![T::zero(); n];
        z[r] = T::one();
        for i in (0..r).rev() {
            let mut zi = -lplus[i] * z[i + 1];
            if !zi.is_finite() {
                zi = T::zero();
            }
            z[i] = zi;
        }
        for i in r..n - 1 {
            let mut zi = -uminus[i] * z[i];
            if !zi.is_finite() {
                zi = T::zero();
            }
            z[i + 1] = zi;
        }

        let ztz: T = z.iter().map(|&x| x * x).sum();
        (z, ztz, mingma)
    }
}

/// A pending node of the representation tree.
struct WorkItem<T> {
    /// The representation `L D L^T = T_block - sigma I` for this node.
    rrr: Rrr<T>,
    /// Accumulated shift: `rrr` represents `T_block - sigma I`.
    sigma: T,
    /// First block-local eigenvalue index owned by this node.
    first: usize,
    /// Last block-local eigenvalue index owned by this node.
    last: usize,
    /// Depth in the representation tree.
    depth: usize,
}

/// MRRR eigenvalue decomposition result.
#[derive(Debug, Clone)]
pub struct MrrrEvd<T: Scalar> {
    /// Computed eigenvalues (sorted in ascending order).
    eigenvalues: Vec<T>,
    /// Eigenvectors (columns correspond to eigenvalues).
    eigenvectors: Option<Mat<T>>,
    /// Original matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> MrrrEvd<T> {
    /// Compute all eigenvalues and eigenvectors using MRRR.
    ///
    /// # Arguments
    ///
    /// * `diagonal` - Main diagonal elements (length n)
    /// * `off_diagonal` - Off-diagonal elements (length n-1)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::MrrrEvd;
    ///
    /// let diag = vec![2.0, 3.0, 4.0];
    /// let off_diag = vec![1.0, 1.0];
    ///
    /// let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
    /// assert_eq!(evd.eigenvalues().len(), 3);
    /// ```
    pub fn compute(diagonal: &[T], off_diagonal: &[T]) -> Result<Self, MrrrError> {
        Self::compute_range(diagonal, off_diagonal, 0, diagonal.len().saturating_sub(1))
    }

    /// Compute eigenvalues only (no eigenvectors).
    pub fn eigenvalues_only(diagonal: &[T], off_diagonal: &[T]) -> Result<Self, MrrrError> {
        let n = diagonal.len();

        if n == 0 {
            return Err(MrrrError::EmptyInput);
        }

        if off_diagonal.len() != n.saturating_sub(1) {
            return Err(MrrrError::DimensionMismatch);
        }

        if n == 1 {
            return Ok(Self {
                eigenvalues: vec![diagonal[0]],
                eigenvectors: None,
                n,
            });
        }

        let eigenvalues = compute_all_eigenvalues(diagonal, off_diagonal)?;

        Ok(Self {
            eigenvalues,
            eigenvectors: None,
            n,
        })
    }

    /// Compute eigenvalues and eigenvectors in a specified index range.
    ///
    /// The full decomposition is computed (so orthogonality across clustered
    /// eigenvalues is preserved) and then the requested range is extracted.
    ///
    /// # Arguments
    ///
    /// * `diagonal` - Main diagonal elements
    /// * `off_diagonal` - Off-diagonal elements
    /// * `il` - First eigenvalue index (0-indexed)
    /// * `iu` - Last eigenvalue index (0-indexed, inclusive)
    pub fn compute_range(
        diagonal: &[T],
        off_diagonal: &[T],
        il: usize,
        iu: usize,
    ) -> Result<Self, MrrrError> {
        let n = diagonal.len();

        if n == 0 {
            return Err(MrrrError::EmptyInput);
        }

        if off_diagonal.len() != n.saturating_sub(1) {
            return Err(MrrrError::DimensionMismatch);
        }

        if il > iu || iu >= n {
            return Err(MrrrError::InvalidIndexRange);
        }

        if n == 1 {
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues: vec![diagonal[0]],
                eigenvectors: Some(eigenvectors),
                n,
            });
        }

        let (all_eigenvalues, all_eigenvectors) = mrrr_eigenvectors(diagonal, off_diagonal)?;

        let eigenvalues: Vec<T> = all_eigenvalues[il..=iu].to_vec();
        let num_eigs = eigenvalues.len();

        let mut eigenvectors = Mat::zeros(n, num_eigs);
        for (col, src) in (il..=iu).enumerate() {
            for row in 0..n {
                eigenvectors[(row, col)] = all_eigenvectors[(row, src)];
            }
        }

        Ok(Self {
            eigenvalues,
            eigenvectors: Some(eigenvectors),
            n,
        })
    }

    /// Returns the computed eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix (columns correspond to eigenvalues).
    pub fn eigenvectors(&self) -> Option<&Mat<T>> {
        self.eigenvectors.as_ref()
    }

    /// Returns the original matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the number of computed eigenvalues.
    pub fn num_eigenvalues(&self) -> usize {
        self.eigenvalues.len()
    }
}

/// Compute all eigenvalues and eigenvectors of a symmetric tridiagonal matrix
/// with the MRRR algorithm.
///
/// The matrix is split at negligible off-diagonal entries; each unreduced block
/// is solved by [`block_mrrr`]. Eigenpairs from all blocks are merged and sorted
/// ascending. Eigenvectors from different blocks are orthogonal by construction
/// (disjoint support).
fn mrrr_eigenvectors<T: Field + Real + bytemuck::Zeroable>(
    diagonal: &[T],
    off_diagonal: &[T],
) -> Result<(Vec<T>, Mat<T>), MrrrError> {
    let n = diagonal.len();

    if n == 0 {
        return Err(MrrrError::EmptyInput);
    }

    if n == 1 {
        let mut vecs = Mat::zeros(1, 1);
        vecs[(0, 0)] = T::one();
        return Ok((vec![diagonal[0]], vecs));
    }

    let (glow, ghigh) = gershgorin_bounds(diagonal, off_diagonal);
    let spdiam = ghigh - glow;
    let eps = <T as Scalar>::epsilon();
    // An off-diagonal is negligible (the matrix decouples there) when it is at
    // the level of roundoff relative to the spectral diameter.
    let split_tol = eps * spdiam + <T as Scalar>::min_positive();

    // Partition [0, n) into unreduced blocks at negligible off-diagonals.
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for i in 0..n - 1 {
        if Scalar::abs(off_diagonal[i]) <= split_tol {
            blocks.push((start, i));
            start = i + 1;
        }
    }
    blocks.push((start, n - 1));

    // Solve each block; build full-length eigenvectors (zero outside the block).
    let mut pairs: Vec<(T, Vec<T>)> = Vec::with_capacity(n);
    for (bs, be) in blocks {
        let nb = be - bs + 1;
        let dblock = &diagonal[bs..=be];
        let oblock: &[T] = if nb > 1 { &off_diagonal[bs..be] } else { &[] };

        let (block_eigs, block_vecs) = block_mrrr(dblock, oblock)?;

        for (col, &lam) in block_eigs.iter().enumerate() {
            let mut full = vec![T::zero(); n];
            for row in 0..nb {
                full[bs + row] = block_vecs[(row, col)];
            }
            pairs.push((lam, full));
        }
    }

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Mat::zeros(n, n);
    for (col, (lam, full)) in pairs.into_iter().enumerate() {
        eigenvalues.push(lam);
        for row in 0..n {
            eigenvectors[(row, col)] = full[row];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Solve one *unreduced* symmetric tridiagonal block with MRRR.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are ascending and
/// column `j` of the matrix is the eigenvector for eigenvalue `j`.
fn block_mrrr<T: Field + Real + bytemuck::Zeroable>(
    diagonal: &[T],
    off_diagonal: &[T],
) -> Result<(Vec<T>, Mat<T>), MrrrError> {
    let n = diagonal.len();

    if n == 0 {
        return Ok((Vec::new(), Mat::zeros(0, 0)));
    }
    if n == 1 {
        let mut vecs = Mat::zeros(1, 1);
        vecs[(0, 0)] = T::one();
        return Ok((vec![diagonal[0]], vecs));
    }

    let eps = <T as Scalar>::epsilon();
    let two = T::one() + T::one();
    let (glow, ghigh) = gershgorin_bounds(diagonal, off_diagonal);
    let spdiam = ghigh - glow;
    let pivmin = spdiam * eps * eps + <T as Scalar>::min_positive();
    let params = MrrrParams {
        spdiam,
        pivmin,
        minrgp: T::from_f64(MIN_RELATIVE_GAP).unwrap_or_else(T::zero),
        rqtol: two * eps,
        degtol: spdiam * eps * T::from_f64(n as f64).unwrap_or_else(T::one)
            + <T as Scalar>::min_positive(),
        eps,
    };

    // Initial eigenvalue approximations (absolute, ascending) via bisection.
    let lambda = compute_all_eigenvalues(diagonal, off_diagonal)?;

    // Positive-definite base representation T - sigma0 I.
    let (base, sigma0) = build_base_rrr(diagonal, off_diagonal, glow, spdiam, pivmin);

    // w[i] holds the eigenvalue relative to the *current* representation.
    let mut w: Vec<T> = lambda.iter().map(|&l| l - sigma0).collect();
    let mut werr: Vec<T> = vec![eps * spdiam + <T as Scalar>::min_positive(); n];
    let mut eig_abs = lambda;
    let mut vecs = Mat::zeros(n, n);

    let mut stack: Vec<WorkItem<T>> = Vec::new();
    stack.push(WorkItem {
        rrr: base,
        sigma: sigma0,
        first: 0,
        last: n - 1,
        depth: 0,
    });

    while let Some(item) = stack.pop() {
        let WorkItem {
            rrr,
            sigma,
            first,
            last,
            depth,
        } = item;

        // Refine all eigenvalues owned by this node relative to its RRR.
        for i in first..=last {
            let (mu, err) = refine_relative(&rrr, i, w[i], werr[i], &params);
            w[i] = mu;
            werr[i] = err;
        }

        // Split [first, last] into singletons and clusters by relative gap.
        let mut j = first;
        while j <= last {
            let mut k = j;
            while k < last {
                let gap = w[k + 1] - w[k];
                let scale =
                    Scalar::abs(w[k]).max(Scalar::abs(w[k + 1])) + params.spdiam * params.eps;
                if gap < params.minrgp * scale {
                    k += 1;
                } else {
                    break;
                }
            }

            if k == j {
                // Singleton: eigenvector straight from the twisted factorization.
                let (z, mu) = compute_singleton_eigenvector(&rrr, w[j], werr[j], &params);
                w[j] = mu;
                eig_abs[j] = sigma + mu;
                for row in 0..n {
                    vecs[(row, j)] = z[row];
                }
            } else {
                let spread = w[k] - w[j];
                let child = if depth >= MAX_TREE_DEPTH || spread <= params.degtol {
                    None
                } else {
                    find_child_rrr(&rrr, &w, &werr, j, k, &params)
                };

                match child {
                    Some((child_rrr, tau)) => {
                        for wm in &mut w[j..=k] {
                            *wm = *wm - tau;
                        }
                        stack.push(WorkItem {
                            rrr: child_rrr,
                            sigma: sigma + tau,
                            first: j,
                            last: k,
                            depth: depth + 1,
                        });
                    }
                    None => {
                        // Last resort: eigenvalues coincide to working accuracy
                        // (or the tree is exhausted). Compute each vector and
                        // orthonormalize this small block locally.
                        for m in j..=k {
                            let (z, mu) =
                                compute_singleton_eigenvector(&rrr, w[m], werr[m], &params);
                            w[m] = mu;
                            eig_abs[m] = sigma + mu;
                            for row in 0..n {
                                vecs[(row, m)] = z[row];
                            }
                        }
                        local_orthonormalize(&mut vecs, j, k, n);
                    }
                }
            }

            j = k + 1;
        }
    }

    Ok((eig_abs, vecs))
}

/// Build a positive-definite base representation `T - sigma0 I = L D L^T`.
///
/// `sigma0` starts at the (lower) Gershgorin bound, which is below the spectrum,
/// so the factorization is positive definite (all pivots positive) and therefore
/// relatively robust. If rounding leaves a non-positive pivot, the shift is
/// pushed further below the spectrum.
fn build_base_rrr<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    glow: T,
    spdiam: T,
    pivmin: T,
) -> (Rrr<T>, T) {
    let mut sigma0 = glow;
    for attempt in 0..40usize {
        let rrr = Rrr::from_tridiagonal(diagonal, off_diagonal, sigma0, pivmin);
        if rrr.d.iter().all(|&x| x > T::zero()) {
            return (rrr, sigma0);
        }
        let step = spdiam * T::from_f64(0.05 * (attempt as f64 + 1.0)).unwrap_or_else(T::one)
            + <T as Scalar>::min_positive();
        sigma0 = sigma0 - step;
    }
    let rrr = Rrr::from_tridiagonal(diagonal, off_diagonal, sigma0, pivmin);
    (rrr, sigma0)
}

/// Refine the `i`-th eigenvalue (block-local index) relative to `rrr` by
/// bisection on the LDL^T negcount. Returns `(eigenvalue, half-width)`.
fn refine_relative<T: Field + Real>(
    rrr: &Rrr<T>,
    i: usize,
    mu0: T,
    werr0: T,
    params: &MrrrParams<T>,
) -> (T, T) {
    let two = T::one() + T::one();
    let floor = params.spdiam * params.eps + <T as Scalar>::min_positive();

    let mut lo = mu0 - werr0 - floor;
    let mut hi = mu0 + werr0 + floor;

    // Expand the bracket until it straddles eigenvalue i: count(lo) <= i < count(hi).
    let mut ex = werr0 + floor;
    let mut tries = 0usize;
    while rrr.neg_count(lo, params.pivmin) > i && tries < 80 {
        lo = lo - ex;
        ex = ex * two;
        tries += 1;
    }
    let mut ex = werr0 + floor;
    tries = 0;
    while rrr.neg_count(hi, params.pivmin) <= i && tries < 80 {
        hi = hi + ex;
        ex = ex * two;
        tries += 1;
    }

    for _ in 0..MAX_BISECTION_ITER {
        let mid = (lo + hi) / two;
        let width = hi - lo;
        if width
            <= two * params.eps * (Scalar::abs(mid) + params.spdiam) + <T as Scalar>::min_positive()
        {
            break;
        }
        if rrr.neg_count(mid, params.pivmin) <= i {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let mu = (lo + hi) / two;
    let err = ((hi - lo) / two).max(params.spdiam * params.eps);
    (mu, err)
}

/// Compute a single eigenvector for a singleton eigenvalue via the twisted
/// factorization, polishing the eigenvalue with Rayleigh-quotient corrections.
/// Returns the normalized eigenvector and the refined eigenvalue (relative to
/// the representation).
fn compute_singleton_eigenvector<T: Field + Real>(
    rrr: &Rrr<T>,
    mu0: T,
    werr: T,
    params: &MrrrParams<T>,
) -> (Vec<T>, T) {
    let floor = params.spdiam * params.eps + <T as Scalar>::min_positive();
    let lo = mu0 - werr - floor;
    let hi = mu0 + werr + floor;

    let mut mu = mu0;
    for _ in 0..MAX_RQI_ITER {
        let (_z, ztz, mingma) = rrr.twisted_eigenvector(mu, params.pivmin);
        if ztz <= T::zero() {
            break;
        }
        let rqcorr = mingma / ztz;
        if !rqcorr.is_finite() {
            break;
        }
        if Scalar::abs(rqcorr)
            <= params.rqtol * (Scalar::abs(mu) + params.spdiam) + <T as Scalar>::min_positive()
        {
            break;
        }
        let mu_new = mu + rqcorr;
        if mu_new > lo && mu_new < hi {
            mu = mu_new;
        } else {
            break;
        }
    }

    let (mut z, _ztz, _mingma) = rrr.twisted_eigenvector(mu, params.pivmin);
    let norm = vector_norm(&z);
    if norm > <T as Scalar>::min_positive() {
        for x in &mut z {
            *x = *x / norm;
        }
    } else {
        // Degenerate safety net: return a valid unit vector.
        for x in &mut z {
            *x = T::zero();
        }
        if !z.is_empty() {
            z[0] = T::one();
        }
    }

    (z, mu)
}

/// Construct a child representation for the cluster `first..=last` by shifting
/// `rrr` to just outside the cluster (dlarrf). Returns `(child, tau)` where
/// `child = rrr - tau*I`, or `None` if no robust shift was found.
fn find_child_rrr<T: Field + Real>(
    rrr: &Rrr<T>,
    w: &[T],
    werr: &[T],
    first: usize,
    last: usize,
    params: &MrrrParams<T>,
) -> Option<(Rrr<T>, T)> {
    let spread = w[last] - w[first];
    let mag = Scalar::abs(w[first]).max(Scalar::abs(w[last]));
    let mut off = spread + werr[first].max(werr[last]) + params.eps * mag + params.pivmin;

    for _ in 0..MAX_SHIFT_ATTEMPTS {
        // Shift below the cluster (cluster eigenvalues become small positive).
        let tau_left = w[first] - off;
        if let Some(child) = try_factor_child(rrr, tau_left, params) {
            return Some((child, tau_left));
        }
        // Shift above the cluster (cluster eigenvalues become small negative).
        let tau_right = w[last] + off;
        if let Some(child) = try_factor_child(rrr, tau_right, params) {
            return Some((child, tau_right));
        }
        off = off / (T::one() + T::one());
        if off <= params.pivmin {
            break;
        }
    }
    None
}

/// Try to factor `rrr - tau*I` and accept it only if the child is a robust
/// representation (finite, no near-zero pivots, no dangerous element growth).
fn try_factor_child<T: Field + Real>(
    rrr: &Rrr<T>,
    tau: T,
    params: &MrrrParams<T>,
) -> Option<Rrr<T>> {
    let child = rrr.factor_child(tau, params.pivmin);
    let mut maxd = T::zero();
    for &d in &child.d {
        let ad = Scalar::abs(d);
        if !d.is_finite() || ad <= params.pivmin {
            return None;
        }
        maxd = maxd.max(ad);
    }
    let growth_limit = params.spdiam / params.eps;
    if maxd < growth_limit {
        Some(child)
    } else {
        None
    }
}

/// Modified Gram–Schmidt orthonormalization of eigenvector columns `first..=last`
/// only. Used as a last resort for a residual cluster whose eigenvalues coincide
/// to working accuracy (any orthonormal basis of the invariant subspace is a
/// valid set of eigenvectors). This touches at most a few columns and is never
/// the global `O(n^3)` reorthogonalization of a naive solver.
fn local_orthonormalize<T: Field + Real>(vecs: &mut Mat<T>, first: usize, last: usize, n: usize) {
    for col in first..=last {
        for prev in first..col {
            let mut dot = T::zero();
            for row in 0..n {
                dot = dot + vecs[(row, col)] * vecs[(row, prev)];
            }
            for row in 0..n {
                vecs[(row, col)] = vecs[(row, col)] - dot * vecs[(row, prev)];
            }
        }
        let mut norm_sq = T::zero();
        for row in 0..n {
            norm_sq = norm_sq + vecs[(row, col)] * vecs[(row, col)];
        }
        let norm = Real::sqrt(norm_sq);
        if norm > <T as Scalar>::min_positive() {
            for row in 0..n {
                vecs[(row, col)] = vecs[(row, col)] / norm;
            }
        }
    }
}

/// Compute all eigenvalues of a symmetric tridiagonal by bisection with the
/// classic tridiagonal Sturm count. Returns eigenvalues in ascending order.
fn compute_all_eigenvalues<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
) -> Result<Vec<T>, MrrrError> {
    let n = diagonal.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    let (glow, ghigh) = gershgorin_bounds(diagonal, off_diagonal);

    let eps = <T as Scalar>::epsilon();
    let two = T::one() + T::one();

    let mut eigenvalues = Vec::with_capacity(n);

    for target_index in 0..n {
        let mut lo = glow;
        let mut hi = ghigh;

        for _iter in 0..MAX_BISECTION_ITER {
            let tol = eps * (Scalar::abs(lo) + Scalar::abs(hi) + T::one());
            if hi - lo <= tol {
                break;
            }

            let mid = (lo + hi) / two;
            let count = sturm_count(diagonal, off_diagonal, mid);

            if count <= target_index {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        eigenvalues.push((lo + hi) / two);
    }

    Ok(eigenvalues)
}

/// Compute Gershgorin bounds for eigenvalues.
fn gershgorin_bounds<T: Field + Real>(diagonal: &[T], off_diagonal: &[T]) -> (T, T) {
    let n = diagonal.len();

    if n == 0 {
        return (T::zero(), T::zero());
    }

    if n == 1 {
        return (diagonal[0], diagonal[0]);
    }

    let mut min = diagonal[0] - Scalar::abs(off_diagonal[0]);
    let mut max = diagonal[0] + Scalar::abs(off_diagonal[0]);

    for i in 1..(n - 1) {
        let radius = Scalar::abs(off_diagonal[i - 1]) + Scalar::abs(off_diagonal[i]);
        let low = diagonal[i] - radius;
        let high = diagonal[i] + radius;
        if low < min {
            min = low;
        }
        if high > max {
            max = high;
        }
    }

    let last_low = diagonal[n - 1] - Scalar::abs(off_diagonal[n - 2]);
    let last_high = diagonal[n - 1] + Scalar::abs(off_diagonal[n - 2]);
    if last_low < min {
        min = last_low;
    }
    if last_high > max {
        max = last_high;
    }

    let margin =
        (max - min) * T::from_f64(0.01).unwrap_or_else(T::zero) + <T as Scalar>::min_positive();
    (min - margin, max + margin)
}

/// Sturm count: number of eigenvalues less than or equal to `x` for the
/// tridiagonal matrix `(diagonal, off_diagonal)`.
fn sturm_count<T: Field + Real>(diagonal: &[T], off_diagonal: &[T], x: T) -> usize {
    let n = diagonal.len();
    if n == 0 {
        return 0;
    }

    let eps = <T as Scalar>::epsilon();
    let mut count = 0;

    let mut d = diagonal[0] - x;
    if d < T::zero() {
        count += 1;
    } else if d < eps && d > -eps {
        d = -eps;
        count += 1;
    }

    for i in 1..n {
        let e_sq = off_diagonal[i - 1] * off_diagonal[i - 1];

        if Scalar::abs(d) < eps {
            d = copysign_mag(eps, d);
        }

        d = (diagonal[i] - x) - e_sq / d;

        if d < T::zero() {
            count += 1;
        } else if d < eps && d > -eps {
            d = -eps;
            count += 1;
        }
    }

    count
}

/// Compute the 2-norm of a vector.
fn vector_norm<T: Field + Real>(v: &[T]) -> T {
    let sum: T = v.iter().map(|&x| x * x).sum();
    Real::sqrt(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evd::SymmetricEvd;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Build a dense symmetric tridiagonal matrix from its diagonal and
    /// off-diagonal, so it can be fed to the reference `SymmetricEvd`.
    fn dense_tridiagonal(diag: &[f64], off: &[f64]) -> Mat<f64> {
        let n = diag.len();
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = diag[i];
        }
        for i in 0..n.saturating_sub(1) {
            a[(i, i + 1)] = off[i];
            a[(i + 1, i)] = off[i];
        }
        a
    }

    /// Maximum residual ||T v_j - lambda_j v_j||_inf over all eigenpairs.
    fn max_residual(diag: &[f64], off: &[f64], evd: &MrrrEvd<f64>) -> f64 {
        let n = diag.len();
        let eigs = evd.eigenvalues();
        let vecs = match evd.eigenvectors() {
            Some(v) => v,
            None => return f64::INFINITY,
        };
        let mut worst = 0.0f64;
        for (j, &lambda) in eigs.iter().enumerate() {
            for i in 0..n {
                let mut tv = diag[i] * vecs[(i, j)];
                if i > 0 {
                    tv += off[i - 1] * vecs[(i - 1, j)];
                }
                if i + 1 < n {
                    tv += off[i] * vecs[(i + 1, j)];
                }
                worst = worst.max((tv - lambda * vecs[(i, j)]).abs());
            }
        }
        worst
    }

    /// Maximum off-diagonal of V^T V and maximum deviation of the diagonal from 1.
    fn max_orthogonality_error(evd: &MrrrEvd<f64>) -> f64 {
        let vecs = match evd.eigenvectors() {
            Some(v) => v,
            None => return f64::INFINITY,
        };
        let n = vecs.nrows();
        let m = vecs.ncols();
        let mut worst = 0.0f64;
        for a in 0..m {
            for b in a..m {
                let mut dot = 0.0;
                for i in 0..n {
                    dot += vecs[(i, a)] * vecs[(i, b)];
                }
                let target = if a == b { 1.0 } else { 0.0 };
                worst = worst.max((dot - target).abs());
            }
        }
        worst
    }

    /// Compare MRRR eigenvalues to the crate's reference symmetric EVD.
    fn max_eigenvalue_error_vs_reference(diag: &[f64], off: &[f64], evd: &MrrrEvd<f64>) -> f64 {
        let a = dense_tridiagonal(diag, off);
        let reference = SymmetricEvd::compute(a.as_ref()).unwrap();
        let mut ref_eigs: Vec<f64> = reference.eigenvalues().to_vec();
        ref_eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut mrrr_eigs: Vec<f64> = evd.eigenvalues().to_vec();
        mrrr_eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut worst = 0.0f64;
        for (a, b) in ref_eigs.iter().zip(mrrr_eigs.iter()) {
            worst = worst.max((a - b).abs());
        }
        worst
    }

    #[test]
    fn test_mrrr_2x2() {
        // Matrix [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 2);
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_mrrr_diagonal() {
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![0.0, 0.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 3);
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));

        // Diagonal matrix: eigenvectors are the standard basis (up to sign).
        assert!(max_orthogonality_error(&evd) < 1e-12);
        assert!(max_residual(&diag, &off_diag, &evd) < 1e-12);
    }

    #[test]
    fn test_mrrr_eigenvalues_only() {
        let diag = vec![4.0, 3.0, 2.0, 1.0];
        let off_diag = vec![1.0, 2.0, 1.0];

        let evd = MrrrEvd::eigenvalues_only(&diag, &off_diag).unwrap();
        assert_eq!(evd.eigenvalues().len(), 4);
        assert!(evd.eigenvectors().is_none());
    }

    #[test]
    fn test_mrrr_eigenvectors_orthogonal() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        assert!(max_orthogonality_error(&evd) < 1e-10);
    }

    #[test]
    fn test_mrrr_eigenvectors_normalized() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let vecs = evd.eigenvectors().unwrap();

        for j in 0..2 {
            let mut norm = 0.0;
            for i in 0..2 {
                norm += vecs[(i, j)] * vecs[(i, j)];
            }
            assert!(approx_eq(norm, 1.0, 1e-10), "norm[{}] = {}", j, norm);
        }
    }

    #[test]
    fn test_mrrr_single_element() {
        let diag = vec![5.0];
        let off_diag: Vec<f64> = vec![];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        assert_eq!(evd.eigenvalues().len(), 1);
        assert!(approx_eq(evd.eigenvalues()[0], 5.0, 1e-10));
    }

    #[test]
    fn test_mrrr_range() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let off_diag = vec![0.0, 0.0, 0.0, 0.0];

        let evd = MrrrEvd::compute_range(&diag, &off_diag, 1, 3).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 3);
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 4.0, 1e-10));

        // Requested three eigenvectors, each of length 5.
        let vecs = evd.eigenvectors().unwrap();
        assert_eq!(vecs.nrows(), 5);
        assert_eq!(vecs.ncols(), 3);
    }

    #[test]
    fn test_mrrr_eigenvalue_equation() {
        let diag = vec![4.0, 3.0, 2.0, 1.0];
        let off_diag = vec![1.0, 2.0, 1.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        assert!(max_residual(&diag, &off_diag, &evd) < 1e-8);
        assert!(max_orthogonality_error(&evd) < 1e-8);
        assert!(max_eigenvalue_error_vs_reference(&diag, &off_diag, &evd) < 1e-8);
    }

    #[test]
    fn test_mrrr_negative_eigenvalues() {
        let diag = vec![-2.0, -2.0];
        let off_diag = vec![1.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], -3.0, 1e-10));
        assert!(approx_eq(eigs[1], -1.0, 1e-10));
        assert!(max_residual(&diag, &off_diag, &evd) < 1e-8);
    }

    #[test]
    fn test_mrrr_clustered_eigenvalues() {
        // Tight but distinct cluster: forces a child representation.
        let diag = vec![2.0, 2.0, 2.0];
        let off_diag = vec![1e-6, 1e-6];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        for &e in eigs {
            assert!(approx_eq(e, 2.0, 1e-5), "eigenvalue = {}", e);
        }
        // The key MRRR property: orthogonal eigenvectors for a cluster.
        assert!(
            max_orthogonality_error(&evd) < 1e-8,
            "orthogonality error = {}",
            max_orthogonality_error(&evd)
        );
        assert!(max_residual(&diag, &off_diag, &evd) < 1e-6);
    }

    #[test]
    fn test_mrrr_repeated_eigenvalue_via_split() {
        // off-diagonal exactly zero -> decoupled 1x1 blocks -> repeated eigenvalue.
        let diag = vec![5.0, 5.0, 5.0];
        let off_diag = vec![0.0, 0.0];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();
        assert_eq!(eigs.len(), 3);
        for &e in eigs {
            assert!(approx_eq(e, 5.0, 1e-12));
        }
        // Eigenvectors must still be orthonormal despite the triple eigenvalue.
        assert!(max_orthogonality_error(&evd) < 1e-12);
        assert!(max_residual(&diag, &off_diag, &evd) < 1e-12);
    }

    #[test]
    fn test_mrrr_larger_matrix() {
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let off_diag: Vec<f64> = vec![0.0; n - 1];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), n);
        for (i, &e) in eigs.iter().enumerate() {
            assert!(approx_eq(e, (i + 1) as f64, 1e-10));
        }
        assert!(max_orthogonality_error(&evd) < 1e-12);
    }

    #[test]
    fn test_mrrr_laplacian() {
        // 1D Laplacian: diag = 2, off = -1. Known eigenvalues 2 - 2 cos(k*pi/(n+1)).
        let n = 12;
        let diag = vec![2.0; n];
        let off = vec![-1.0; n - 1];

        let evd = MrrrEvd::compute(&diag, &off).unwrap();
        let eigs = evd.eigenvalues();
        assert_eq!(eigs.len(), n);

        for (k, &lam) in eigs.iter().enumerate() {
            let expected =
                2.0 - 2.0 * ((k as f64 + 1.0) * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!(
                approx_eq(lam, expected, 1e-9),
                "k={}, got {}, expected {}",
                k,
                lam,
                expected
            );
        }
        assert!(max_residual(&diag, &off, &evd) < 1e-9);
        assert!(max_orthogonality_error(&evd) < 1e-9);
        assert!(max_eigenvalue_error_vs_reference(&diag, &off, &evd) < 1e-9);
    }

    #[test]
    fn test_mrrr_general_dense_reference() {
        // A non-trivial tridiagonal with varied entries, checked against the
        // reference dense symmetric EVD plus residual/orthogonality.
        let diag = vec![1.0, -3.0, 4.0, 0.5, 2.5, -1.0, 6.0, 3.0];
        let off = vec![0.7, 1.3, -0.9, 2.1, 0.4, -1.7, 0.6];

        let evd = MrrrEvd::compute(&diag, &off).unwrap();
        assert_eq!(evd.eigenvalues().len(), diag.len());
        assert!(
            max_residual(&diag, &off, &evd) < 1e-8,
            "residual = {}",
            max_residual(&diag, &off, &evd)
        );
        assert!(
            max_orthogonality_error(&evd) < 1e-8,
            "orthogonality = {}",
            max_orthogonality_error(&evd)
        );
        assert!(
            max_eigenvalue_error_vs_reference(&diag, &off, &evd) < 1e-8,
            "eig error = {}",
            max_eigenvalue_error_vs_reference(&diag, &off, &evd)
        );
    }

    #[test]
    fn test_mrrr_two_clusters() {
        // Two separated tight clusters: exercises multiple child representations
        // and cross-cluster orthogonality.
        let diag = vec![1.0, 1.0, 1.0, 8.0, 8.0, 8.0];
        let off = vec![1e-5, 1e-5, 3.0, 1e-5, 1e-5];

        let evd = MrrrEvd::compute(&diag, &off).unwrap();
        assert_eq!(evd.eigenvalues().len(), 6);
        assert!(
            max_orthogonality_error(&evd) < 1e-7,
            "orthogonality = {}",
            max_orthogonality_error(&evd)
        );
        assert!(max_residual(&diag, &off, &evd) < 1e-6);
        assert!(max_eigenvalue_error_vs_reference(&diag, &off, &evd) < 1e-6);
    }

    #[test]
    fn test_mrrr_split_blocks() {
        // A negligible off-diagonal in the middle decouples the matrix.
        let diag = vec![2.0, 1.0, 5.0, 4.0];
        let off = vec![0.5, 0.0, 0.5];

        let evd = MrrrEvd::compute(&diag, &off).unwrap();
        assert_eq!(evd.eigenvalues().len(), 4);
        assert!(max_residual(&diag, &off, &evd) < 1e-9);
        assert!(max_orthogonality_error(&evd) < 1e-9);
        assert!(max_eigenvalue_error_vs_reference(&diag, &off, &evd) < 1e-9);
    }

    #[test]
    fn test_mrrr_large_pseudo_random() {
        // A larger tridiagonal with deterministic pseudo-random entries. This
        // exercises the representation tree at scale and validates the O(n^2)
        // twisted-factorization path against the reference dense EVD.
        let n = 40usize;
        // Simple LCG for reproducible entries in [-1, 1].
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let diag: Vec<f64> = (0..n).map(|_| 3.0 * next()).collect();
        let off: Vec<f64> = (0..n - 1).map(|_| 1.0 + 0.5 * next()).collect();

        let evd = MrrrEvd::compute(&diag, &off).unwrap();
        assert_eq!(evd.eigenvalues().len(), n);
        assert!(
            max_residual(&diag, &off, &evd) < 1e-8,
            "residual = {}",
            max_residual(&diag, &off, &evd)
        );
        assert!(
            max_orthogonality_error(&evd) < 1e-8,
            "orthogonality = {}",
            max_orthogonality_error(&evd)
        );
        assert!(
            max_eigenvalue_error_vs_reference(&diag, &off, &evd) < 1e-8,
            "eig error = {}",
            max_eigenvalue_error_vs_reference(&diag, &off, &evd)
        );
    }

    #[test]
    fn test_mrrr_f32() {
        let diag = vec![2.0f32, 2.0];
        let off_diag = vec![1.0f32];

        let evd = MrrrEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);

        // f32 orthogonality / residual.
        let vecs = evd.eigenvectors().unwrap();
        let mut dot = 0.0f32;
        for i in 0..2 {
            dot += vecs[(i, 0)] * vecs[(i, 1)];
        }
        assert!(dot.abs() < 1e-4, "dot = {}", dot);
    }

    #[test]
    fn test_rrr_factorization_reconstructs() {
        // The base RRR factors T - sigma I; verify L D L^T reproduces T - sigma I.
        let diag = vec![4.0, 3.0, 2.0];
        let off = vec![1.0, 1.0];
        let eps = f64::EPSILON;
        let (glow, ghigh) = gershgorin_bounds(&diag, &off);
        let spdiam = ghigh - glow;
        let pivmin = spdiam * eps * eps + f64::MIN_POSITIVE;
        let (rrr, sigma) = build_base_rrr(&diag, &off, glow, spdiam, pivmin);

        // Reconstruct A = L D L^T and compare with T - sigma I.
        let n = 3;
        for i in 0..n {
            // Diagonal of L D L^T at (i,i): d[i] + l[i-1]^2 d[i-1].
            let mut aii = rrr.d[i];
            if i > 0 {
                aii += rrr.l[i - 1] * rrr.l[i - 1] * rrr.d[i - 1];
            }
            assert!((aii - (diag[i] - sigma)).abs() < 1e-10);
        }
        for i in 0..n - 1 {
            // Off-diagonal (i, i+1) of L D L^T is l[i] d[i] = ld[i].
            assert!((rrr.ld[i] - off[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_child_representation_shift() {
        let diag = vec![4.0, 3.0, 2.0];
        let off = vec![1.0, 1.0];
        let eps = f64::EPSILON;
        let (glow, ghigh) = gershgorin_bounds(&diag, &off);
        let spdiam = ghigh - glow;
        let pivmin = spdiam * eps * eps + f64::MIN_POSITIVE;
        let (rrr, _sigma) = build_base_rrr(&diag, &off, glow, spdiam, pivmin);

        let tau = 0.5;
        let child = rrr.factor_child(tau, pivmin);

        // neg_count of child at mu equals neg_count of parent at mu + tau.
        for &probe in &[-1.0, 0.0, 0.3, 1.0, 2.5] {
            let cc = child.neg_count(probe, pivmin);
            let pc = rrr.neg_count(probe + tau, pivmin);
            assert_eq!(cc, pc, "probe = {}", probe);
        }
    }

    #[test]
    fn test_error_empty_input() {
        let empty: Vec<f64> = vec![];
        assert!(matches!(
            MrrrEvd::compute(&empty, &empty),
            Err(MrrrError::EmptyInput)
        ));
    }

    #[test]
    fn test_error_dimension_mismatch() {
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![1.0]; // Should be length 2.
        assert!(matches!(
            MrrrEvd::compute(&diag, &off_diag),
            Err(MrrrError::DimensionMismatch)
        ));
    }

    #[test]
    fn test_error_invalid_range() {
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![1.0, 1.0];
        assert!(matches!(
            MrrrEvd::compute_range(&diag, &off_diag, 5, 6),
            Err(MrrrError::InvalidIndexRange)
        ));
    }
}
