//! Symmetric Eigenvalue Decomposition using Divide-and-Conquer.
//!
//! The pipeline mirrors LAPACK `DSYEVD`:
//!
//! 1. blocked Householder reduction to tridiagonal form on the lower
//!    triangle (`DSYTRD`; reflectors kept, `Q` not formed),
//! 2. Cuppen's divide and conquer on the tridiagonal (`DSTEDC`): the matrix is
//!    torn into two halves by a rank-one modification, the halves are solved
//!    recursively and merged by
//!    - deflating negligible components of the coupling vector `z` and
//!      (near-)equal eigenvalues (Gu-Eisenstat / `DLAED2`-style deflation),
//!    - solving the secular equation
//!      `f(lambda) = 1 + rho * sum_i z_i^2 / (d_i - lambda) = 0`
//!      with a safeguarded rational-interpolation root finder (`DLAED4`-style
//!      "middle way"),
//!    - forming the eigenvectors from the *stabilised* rank-one vector
//!      recomputed via the Löwner formula (`DLAED3`) and combining them with
//!      the block-diagonal sub-block eigenvectors by GEMM,
//! 3. back-transformation `V = Q·Z` with compact-WY blocks of reflectors
//!    (`DORMTR`).
//!
//! The eigenvectors are numerically orthogonal. For `f32`/`f64` every
//! `O(n³)` stage runs through the `oxiblas_blas` GEMM; other scalar types use
//! the same algorithm with plain loops.

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::dc_merge::{DcError, stedc};
use super::sym_kernels::{
    MatMul, Pipeline, lower_from_upper, scale_into_safe_range, with_gemm_backend,
};
use super::sym_tridiag::{apply_q, reduce};

/// Error type for divide-and-conquer symmetric eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricEvdDcError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed: a root did not converge within the
    /// iteration limit, or was not finite.
    SecularEquationFailed,
}

impl core::fmt::Display for SymmetricEvdDcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for SymmetricEvdDcError {}

impl From<DcError> for SymmetricEvdDcError {
    fn from(e: DcError) -> Self {
        match e {
            DcError::NotConverged => Self::NotConverged,
            DcError::SecularEquationFailed => Self::SecularEquationFailed,
        }
    }
}

/// Symmetric eigenvalue decomposition using divide-and-conquer.
///
/// Computes A = V·D·V^T where V contains eigenvectors and D is diagonal.
/// As in LAPACK, divide-and-conquer overtakes the QR algorithm
/// ([`SymmetricEvd`](super::SymmetricEvd)) on large matrices because almost
/// all of its eigenvector work is matrix multiplication.
///
/// [`compute`](Self::compute) runs on the calling thread. With the `parallel`
/// feature, `compute_with_par` runs the GEMMs of the tridiagonal reduction,
/// the merges and the back-transformation with a given [`Par`] and returns
/// bitwise the same result.
#[derive(Debug, Clone)]
pub struct SymmetricEvdDc<T: Scalar> {
    /// Eigenvalues (sorted in ascending order).
    eigenvalues: Vec<T>,
    /// Eigenvectors (columns of V).
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

/// Sub-problem size at or below which the base-case QR solver is used.
///
/// This mirrors LAPACK's `SMLSIZ`.  For blocks larger than this the genuine
/// divide-and-conquer merge is used; for smaller blocks the (already correct and
/// numerically robust) tridiagonal QR iteration is cheaper.  The merge is exact
/// at every size, so this threshold only trades performance, never correctness.
const SMLSIZ: usize = 25;

impl<T: Field + Real + bytemuck::Zeroable> SymmetricEvdDc<T> {
    /// Computes the eigendecomposition of a symmetric matrix using divide-and-conquer.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (only upper triangle is used)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::SymmetricEvdDc;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[2.0f64, 1.0],
    ///     &[1.0, 2.0],
    /// ]);
    ///
    /// let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
    /// let eigs = evd.eigenvalues();
    ///
    /// // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    /// assert!((eigs[0] - 1.0).abs() < 1e-10);
    /// assert!((eigs[1] - 3.0).abs() < 1e-10);
    /// ```
    ///
    /// # Errors
    ///
    /// [`SymmetricEvdDcError::EmptyMatrix`] or
    /// [`SymmetricEvdDcError::NotSquare`] for an empty or non-square `a`;
    /// [`SymmetricEvdDcError::NotConverged`] if the tridiagonal has
    /// non-finite entries or the QR iteration on a leaf block does not
    /// converge; [`SymmetricEvdDcError::SecularEquationFailed`] if a secular
    /// equation of a merge does not converge within its iteration limit.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SymmetricEvdDcError> {
        Self::compute_impl(a, Par::Seq)
    }

    /// Same as [`Self::compute`], with the GEMMs of the tridiagonal
    /// reduction, of the divide-and-conquer merges and of the
    /// back-transformation run with `par`.
    ///
    /// The result is bitwise identical to [`Self::compute`] for every `Par`:
    /// the parallel GEMM only partitions its output, and every other stage
    /// runs on the calling thread.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::compute`].
    #[cfg(feature = "parallel")]
    pub fn compute_with_par(a: MatRef<'_, T>, par: Par) -> Result<Self, SymmetricEvdDcError> {
        Self::compute_impl(a, par)
    }

    /// [`Self::compute`] with the GEMMs run with `par`.
    fn compute_impl(a: MatRef<'_, T>, par: Par) -> Result<Self, SymmetricEvdDcError> {
        let n = a.nrows();

        if n == 0 {
            return Err(SymmetricEvdDcError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(SymmetricEvdDcError::NotSquare);
        }

        // Handle trivial case
        if n == 1 {
            let eigenvalues = vec![a[(0, 0)]];
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues,
                eigenvectors,
                n,
            });
        }

        // The upper triangle of `a` becomes the lower triangle of the work
        // matrix, which the reduction overwrites with its reflectors.
        let mut work = lower_from_upper(a);
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Mat::zeros(n, n);
        with_gemm_backend(
            &mut work,
            &mut eigenvalues,
            &mut eigenvectors,
            DcPipeline {
                smlsiz: SMLSIZ,
                par,
            },
        )?;

        Ok(Self {
            eigenvalues,
            eigenvectors,
            n,
        })
    }

    /// Returns the eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix V.
    ///
    /// Column i contains the eigenvector corresponding to eigenvalue i.
    pub fn eigenvectors(&self) -> MatRef<'_, T> {
        self.eigenvectors.as_ref()
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = V * D * V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a = Mat::zeros(n, n);

        // A = V * D * V^T = sum_i lambda_i * v_i * v_i^T
        for k in 0..n {
            let lambda = self.eigenvalues[k];
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] =
                        a[(i, j)] + lambda * self.eigenvectors[(i, k)] * self.eigenvectors[(j, k)];
                }
            }
        }

        a
    }
}

/// Tridiagonal reduction, divide and conquer, back-transformation.
#[derive(Debug, Clone, Copy)]
struct DcPipeline {
    /// Leaf size of the divide and conquer.
    smlsiz: usize,
    /// Parallelism of the GEMMs.
    par: Par,
}

impl Pipeline for DcPipeline {
    type Output = Result<(), SymmetricEvdDcError>;

    fn run<T, G>(self, work: &mut Mat<T>, values: &mut Vec<T>, vectors: &mut Mat<T>) -> Self::Output
    where
        T: Field + Real + bytemuck::Zeroable,
        G: MatMul<T>,
    {
        let sigma = scale_into_safe_range(work);
        let tri = reduce::<T, G>(work, self.par);
        let mut d = tri.d;
        let mut e = tri.e;
        // Eigenvectors of the tridiagonal, in ascending eigenvalue order ...
        stedc::<T, G>(&mut d, &mut e, vectors, self.smlsiz, self.par)?;
        // ... mapped back to those of A: V = Q·Z.
        apply_q::<T, G>(work, &tri.tau, vectors, self.par);
        if sigma != T::one() {
            for v in d.iter_mut() {
                *v /= sigma;
            }
        }
        *values = d;
        Ok(())
    }
}

/// Eigen-decomposition of `tridiag(off_diag, diag, off_diag)` by divide and
/// conquer with leaves of at most `smlsiz` rows (`1` forces a merge at every
/// level): `(eigenvalues ascending, eigenvectors)`.
#[cfg(test)]
fn tridiag_dc<T: Field + Real + bytemuck::Zeroable>(
    diag: Vec<T>,
    off_diag: Vec<T>,
    smlsiz: usize,
) -> Result<(Vec<T>, Mat<T>), SymmetricEvdDcError> {
    let n = diag.len();
    let mut d = diag;
    let mut e = off_diag;
    let mut z = Mat::zeros(n, n);
    stedc::<T, super::sym_kernels::NaiveMatMul>(&mut d, &mut e, &mut z, smlsiz, Par::Seq)?;
    Ok((d, z))
}

/// QR iteration on `tridiag(off_diag, diag, off_diag)` accumulating into
/// `v`: returns the eigenvalues in ascending order (columns of `v` sorted
/// alike).
#[cfg(test)]
fn qr_algorithm<T: Field + Real>(
    diag: Vec<T>,
    off_diag: Vec<T>,
    v: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T>, SymmetricEvdDcError> {
    use super::sym_kernels::{argsort, permute_columns};
    use super::sym_steqr::{Deflation, RotTarget, RotationLog, steqr};

    let mut d = diag;
    let mut e = off_diag;
    let ld = v.row_stride();
    let mut log = RotationLog::new();
    let target = RotTarget {
        data: v.raw_data_mut(),
        off: 0,
        ld,
        nrows: n,
        identity: false,
    };
    steqr(
        &mut d,
        &mut e,
        Some(target),
        100,
        Deflation::Relaxed,
        &mut log,
    )
    .map_err(|_| SymmetricEvdDcError::NotConverged)?;
    let mut perm = Vec::new();
    argsort(&d, &mut perm);
    let sorted: Vec<T> = perm.iter().map(|&p| d[p]).collect();
    let (mut saved, mut visited) = (Vec::new(), Vec::new());
    permute_columns(v.raw_data_mut(), 0, ld, n, &perm, &mut saved, &mut visited);
    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// The constant `c` of the `c·n·ε` tolerances below.
    const DC_TOL_C: f64 = 16.0;

    /// Deterministic split-mix PRNG producing values in [-1, 1] (no external deps).
    struct Prng {
        state: u64,
    }

    impl Prng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f64(&mut self) -> f64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) * 2.0 - 1.0
        }
    }

    /// Builds a dense symmetric matrix with pseudo-random entries.
    fn random_symmetric(n: usize, seed: u64) -> Mat<f64> {
        let mut prng = Prng::new(seed);
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = prng.next_f64();
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }
        a
    }

    /// Builds A = H * diag(eigs) * H where H = I - 2 w w^T is a Householder
    /// reflector.  The eigenvalues of A are exactly `eigs` (with any repeats).
    fn symmetric_with_spectrum(eigs: &[f64], seed: u64) -> Mat<f64> {
        let n = eigs.len();
        let mut prng = Prng::new(seed);
        let mut w = vec![0.0f64; n];
        let mut norm_sq = 0.0;
        for wi in w.iter_mut() {
            *wi = prng.next_f64();
            norm_sq += *wi * *wi;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for wi in w.iter_mut() {
                *wi /= norm;
            }
        }
        // H = I - 2 w w^T ; A = H D H.
        let mut h: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let delta = if i == j { 1.0 } else { 0.0 };
                h[(i, j)] = delta - 2.0 * w[i] * w[j];
            }
        }
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += h[(i, k)] * eigs[k] * h[(j, k)];
                }
                a[(i, j)] = acc;
            }
        }
        a
    }

    /// Maximum |V^T V - I| over all entries.
    fn orthogonality_error(v: MatRef<'_, f64>, n: usize) -> f64 {
        let mut err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                err = err.max((dot - expected).abs());
            }
        }
        err
    }

    /// Maximum residual |A v_i - lambda_i v_i| over all eigenpairs.
    fn residual_error(a: &Mat<f64>, evd: &SymmetricEvdDc<f64>, n: usize) -> f64 {
        let v = evd.eigenvectors();
        let eigs = evd.eigenvalues();
        let mut err = 0.0f64;
        for col in 0..n {
            let lambda = eigs[col];
            for row in 0..n {
                let mut av = 0.0;
                for k in 0..n {
                    av += a[(row, k)] * v[(k, col)];
                }
                err = err.max((av - lambda * v[(row, col)]).abs());
            }
        }
        err
    }

    #[test]
    fn test_evd_dc_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_3x3() {
        // Symmetric 3x3 matrix
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();

        // Reconstruct and verify
        let reconstructed = evd.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
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
    fn test_evd_dc_diagonal() {
        // Diagonal matrix - eigenvalues are the diagonal elements
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues sorted: 1, 2, 3
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let evd = SymmetricEvdDc::compute(eye.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // All eigenvalues should be 1
        for &e in eigs {
            assert!(approx_eq(e, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_evd_dc_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 1);
        assert!(approx_eq(eigs[0], 5.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_negative_eigenvalues() {
        // Matrix with negative eigenvalues
        let a = Mat::from_rows(&[&[-2.0f64, 1.0], &[1.0, -2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues: -3 and -1
        assert!(approx_eq(eigs[0], -3.0, 1e-10));
        assert!(approx_eq(eigs[1], -1.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[1.0, 2.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_evd_dc_repeated_eigenvalues() {
        // Matrix with repeated eigenvalue (3 appears twice)
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, 1.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_evd_dc_orthogonality() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();

        // Verify V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-8),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_evd_dc_larger_matrix() {
        // Test with a simple diagonal 10x10 matrix
        let n = 10;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        // Create a simple diagonal matrix with distinct eigenvalues
        for i in 0..n {
            a[(i, i)] = (i + 1) as f64;
        }

        let evd = SymmetricEvdDc::compute(a.as_ref()).unwrap();

        // Eigenvalues should be 1, 2, 3, ..., 10
        let eigs = evd.eigenvalues();
        for i in 0..n {
            assert!(
                approx_eq(eigs[i], (i + 1) as f64, 1e-10),
                "eigenvalue {} = {}, expected {}",
                i,
                eigs[i],
                i + 1
            );
        }

        // Verify reconstruction
        let reconstructed = evd.reconstruct();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }

        // Verify orthogonality
        let v = evd.eigenvectors();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-8),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_evd_dc_vs_qr() {
        // Compare results with standard QR-based EVD
        use super::super::symmetric::SymmetricEvd;

        let a = Mat::from_rows(&[
            &[4.0f64, 2.0, 1.0, 0.5],
            &[2.0, 5.0, 3.0, 1.0],
            &[1.0, 3.0, 6.0, 2.0],
            &[0.5, 1.0, 2.0, 4.0],
        ]);

        let evd_qr = SymmetricEvd::compute(a.as_ref()).unwrap();
        let evd_dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();

        let eigs_qr = evd_qr.eigenvalues();
        let eigs_dc = evd_dc.eigenvalues();

        for i in 0..4 {
            assert!(
                approx_eq(eigs_qr[i], eigs_dc[i], 1e-8),
                "eigenvalue {} mismatch: QR={}, DC={}",
                i,
                eigs_qr[i],
                eigs_dc[i]
            );
        }
    }

    /// Compares divide-and-conquer eigenvalues against the QR reference and checks
    /// eigenvector orthogonality and residuals across a range of sizes, including
    /// sizes well above `SMLSIZ` that exercise the real D&C merge.
    #[test]
    fn test_evd_dc_reference_across_sizes() {
        use super::super::symmetric::SymmetricEvd;

        for (idx, &n) in [3usize, 5, 10, 50, 150, 300].iter().enumerate() {
            let a = random_symmetric(n, 0x1234_5678 + idx as u64 * 97);

            let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();
            let qr = SymmetricEvd::compute(a.as_ref()).unwrap();

            let dc_eigs = dc.eigenvalues();
            let qr_eigs = qr.eigenvalues();

            // c·n·ε, scaled by the spectral radius where it applies. The
            // largest errors measured (Apple M3) are below 1.0·n·ε (·scale)
            // for every size; c = 16 leaves an order of magnitude for other
            // targets and rounding orders.
            let scale = qr_eigs.iter().fold(1.0f64, |acc, &e| acc.max(e.abs()));
            let n_eps = DC_TOL_C * n as f64 * f64::EPSILON;
            let eig_tol = n_eps * scale;

            for i in 0..n {
                assert!(
                    (dc_eigs[i] - qr_eigs[i]).abs() <= eig_tol,
                    "n={}: eigenvalue {} mismatch DC={} QR={} (tol {})",
                    n,
                    i,
                    dc_eigs[i],
                    qr_eigs[i],
                    eig_tol
                );
            }

            let ortho = orthogonality_error(dc.eigenvectors(), n);
            assert!(
                ortho <= n_eps,
                "n={}: orthogonality error {} too large",
                n,
                ortho
            );

            let resid = residual_error(&a, &dc, n);
            assert!(
                resid <= n_eps * scale,
                "n={}: residual {} too large",
                n,
                resid
            );
        }
    }

    /// Directly exercises the divide-and-conquer merge at *every* level (smlsiz=1)
    /// for small tridiagonal matrices, comparing against the QR base solver.
    #[test]
    fn test_tridiag_dc_forced_merge_small() {
        for &n in &[2usize, 3, 5, 10] {
            let mut prng = Prng::new(0xABCD_00FF + n as u64);
            let diag: Vec<f64> = (0..n).map(|_| prng.next_f64() * 4.0).collect();
            let off: Vec<f64> = (0..n - 1).map(|_| prng.next_f64() * 2.0 + 0.5).collect();

            // Reference: QR base solver on the same tridiagonal.
            let mut z_ref = Mat::eye(n);
            let ref_eigs = qr_algorithm(diag.clone(), off.clone(), &mut z_ref, n).unwrap();

            // Forced divide-and-conquer (merge at every level).
            let (dc_eigs, dc_vec) = tridiag_dc(diag.clone(), off.clone(), 1).unwrap();

            for i in 0..n {
                assert!(
                    (dc_eigs[i] - ref_eigs[i]).abs() <= 1e-9 * (1.0 + ref_eigs[i].abs()),
                    "n={}: forced-merge eigenvalue {} mismatch DC={} REF={}",
                    n,
                    i,
                    dc_eigs[i],
                    ref_eigs[i]
                );
            }

            let ortho = orthogonality_error(dc_vec.as_ref(), n);
            assert!(
                ortho <= 1e-10 * (n as f64),
                "n={}: forced-merge orthogonality error {} too large",
                n,
                ortho
            );
        }
    }

    /// Exercises type-2 deflation in the merge path: a matrix built with many
    /// exactly-repeated eigenvalues but non-trivial (dense) coupling.
    #[test]
    fn test_evd_dc_deflation_repeated_spectrum() {
        // 40 eigenvalues with several repeats; SMLSIZ=25 forces one D&C merge.
        let mut eigs = vec![
            1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, -3.0, -3.0, -3.0, -3.0, -3.0, 7.0, 7.0, 0.0, 0.0,
            0.0, 5.5, 5.5, 9.0,
        ];
        // Pad up to 40 with a few more repeats and distinct values.
        while eigs.len() < 40 {
            let v = eigs.len() as f64 * 0.31 - 6.0;
            eigs.push(v);
            eigs.push(v); // ensure repeats appear across the split boundary
        }
        eigs.truncate(40);
        let n = eigs.len();

        let a = symmetric_with_spectrum(&eigs, 0x5151_2727);

        let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let dc_eigs = dc.eigenvalues();

        let mut sorted = eigs.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

        // c·n·ε as in `test_evd_dc_reference_across_sizes` (measured below
        // 0.4·n·ε here); `a` is formed in floating point, so its spectrum is
        // the exact one only to about n·ε·scale.
        let scale = sorted.iter().fold(1.0f64, |acc, &e| acc.max(e.abs()));
        let n_eps = DC_TOL_C * n as f64 * f64::EPSILON;
        for i in 0..n {
            assert!(
                (dc_eigs[i] - sorted[i]).abs() <= n_eps * scale,
                "eigenvalue {} mismatch DC={} expected={}",
                i,
                dc_eigs[i],
                sorted[i]
            );
        }

        let ortho = orthogonality_error(dc.eigenvectors(), n);
        assert!(
            ortho <= n_eps,
            "deflation orthogonality error {} too large",
            ortho
        );

        let resid = residual_error(&a, &dc, n);
        assert!(
            resid <= n_eps * scale,
            "deflation residual {} too large",
            resid
        );
    }

    /// The generic (plain-loop) pipeline used for scalar types without a GEMM
    /// kernel agrees with the GEMM pipeline, including the blocked paths.
    #[test]
    fn test_generic_pipeline_matches_gemm_pipeline() {
        use super::super::sym_kernels::{BlasMatMul, NaiveMatMul, lower_from_upper};

        for (idx, &n) in [2usize, 7, 60, 180].iter().enumerate() {
            let a = random_symmetric(n, 0x6E6E + idx as u64);
            let run = |naive: bool| {
                let mut work = lower_from_upper(a.as_ref());
                let mut vals = Vec::new();
                let mut vecs = Mat::zeros(n, n);
                let p = DcPipeline {
                    smlsiz: SMLSIZ,
                    par: Par::Seq,
                };
                let r = if naive {
                    p.run::<f64, NaiveMatMul>(&mut work, &mut vals, &mut vecs)
                } else {
                    p.run::<f64, BlasMatMul>(&mut work, &mut vals, &mut vecs)
                };
                r.unwrap();
                (vals, vecs)
            };
            let (v_blas, z_blas) = run(false);
            let (v_gen, z_gen) = run(true);
            let scale = v_blas.iter().fold(1.0f64, |m, &x| m.max(x.abs()));
            for i in 0..n {
                assert!(
                    (v_blas[i] - v_gen[i]).abs() <= 1e-12 * scale,
                    "n={}: eigenvalue {} differs: {} vs {}",
                    n,
                    i,
                    v_blas[i],
                    v_gen[i]
                );
                let dot: f64 = (0..n).map(|r| z_blas[(r, i)] * z_gen[(r, i)]).sum();
                assert!(
                    (dot.abs() - 1.0).abs() <= 1e-10,
                    "n={}: eigenvector {} differs (|dot| = {})",
                    n,
                    i,
                    dot
                );
            }
            assert!(orthogonality_error(z_gen.as_ref(), n) <= 1e-12 * n as f64);
        }
    }

    /// f32 divide-and-conquer at a size that exercises the merge.
    #[test]
    fn test_evd_dc_f32_larger() {
        let n = 60usize;
        let mut prng = Prng::new(0x2468_ACE0);
        let mut a: Mat<f32> = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = (prng.next_f64() as f32) * 2.0;
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }

        let dc = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let eigs = dc.eigenvalues();

        // Ascending order.
        for i in 0..n - 1 {
            assert!(eigs[i] <= eigs[i + 1] + 1e-3);
        }

        // Orthogonality (looser for f32).
        let v = dc.eigenvectors();
        let mut ortho = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f32;
                for k in 0..n {
                    dot += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                ortho = ortho.max((dot - expected).abs());
            }
        }
        assert!(
            ortho <= 1e-3 * n as f32,
            "f32 orthogonality {} too large",
            ortho
        );
    }
}
