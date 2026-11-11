//! SVD using divide-and-conquer algorithm.
//!
//! This algorithm is more efficient for large matrices than the Jacobi method.
//! It first reduces the matrix to bidiagonal form, then applies divide-and-conquer
//! to compute the SVD of the bidiagonal matrix (the LAPACK `dgesdd` scheme).
//!
//! Pipeline (`m ≥ n`; a wide matrix is handled through its transpose):
//!
//! 0. Non-finite input is rejected; a matrix whose largest entry lies outside
//!    `[√safmin/ε, ε/√safmin]` is scaled into that range by a power of two
//!    (exactly, as `dgesdd` does with `dlascl`) and the singular values are
//!    unscaled at the end.
//! 1. Blocked Householder bidiagonalisation `A = Q·B·Pᵀ` (`dgebrd`/`dlabrd`
//!    panels, GEMM trailing updates).
//! 2. Divide-and-conquer SVD of the bidiagonal `B` itself (`dbdsdc`; see
//!    [`crate::svd::bidiag_dc`]): Gu–Eisenstat merges whose singular vectors
//!    are orthogonal without re-orthogonalisation, the `dbdsqr` QR iteration
//!    for blocks of at most 25 rows.
//! 3. `U = Q·diag(U_B, I)` and `V = P·V_B`, with the reflectors applied as
//!    compact-WY blocks (three GEMMs per block) instead of forming `Q`, `P`.
//!
//! The result is backward stable: the residual `‖A − UΣVᵀ‖` and the loss of
//! orthogonality of `U` and `V` are of order `n·ε` (relative to `‖A‖`), and
//! every singular value has an absolute error of order `n·ε·‖A‖`.
//!
//! For `f32`/`f64` every Level-3 step runs on the optimised `oxiblas_blas`
//! GEMM, sequentially for [`SvdDc::compute`] and with the requested [`Par`]
//! mode for `SvdDc::compute_with_par` (feature `parallel`; the parallel
//! GEMM is bitwise identical to the sequential one); other real scalar types
//! (e.g. `f16`) run the same algorithm on a portable column-oriented backend.
//!
//! Complexity: O(mn²) for the bidiagonalization and back-transformation, and
//! O(n³) (GEMM merges) for the bidiagonal SVD, typically much less once
//! deflation sets in.

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::bidiag_dc::{BidiagDcError, bidiag_svd_into};
use super::blocked_bidiag::{ReflectorLayout, apply_reflectors_left, gebrd_tall};
use super::dc_kernels::{
    BlasOps, DenseOps, PortableOps, fabs, mul_pow2, pow2_exponent, same_mut, scale_pow2,
    transpose_into,
};

/// Error type for divide-and-conquer SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdDcError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
}

impl core::fmt::Display for SvdDcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "SVD algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for SvdDcError {}

/// Singular Value Decomposition using divide-and-conquer algorithm.
///
/// A = U·Σ·V^T where Σ contains singular values on the diagonal.
///
/// [`compute`](Self::compute) runs on the calling thread. With the `parallel`
/// feature, `compute_with_par` runs the GEMMs of the bidiagonal reduction,
/// the divide-and-conquer merges and the back-transformation with a given
/// [`Par`] and returns bitwise the same result.
#[derive(Debug, Clone)]
pub struct SvdDc<T: Scalar> {
    /// Left singular vectors (m×m orthogonal matrix).
    u: Mat<T>,
    /// Singular values (sorted in descending order).
    sigma: Vec<T>,
    /// Right singular vectors (n×n orthogonal matrix, stored as V^T).
    vt: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> SvdDc<T> {
    /// Computes the full SVD of matrix A using divide-and-conquer algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::SvdDc;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = SvdDc::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// // Singular values of diagonal matrix are absolute values of diagonal
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    ///
    /// # Errors
    ///
    /// - [`SvdDcError::EmptyMatrix`] if `a` has no rows or no columns.
    /// - [`SvdDcError::NotConverged`] if `a` has a NaN or infinite entry (no
    ///   finite SVD exists; nothing is computed), or if the bidiagonal QR
    ///   iteration exhausts its iteration budget.
    /// - [`SvdDcError::SecularEquationFailed`] if a merge of the divide and
    ///   conquer produces a non-finite singular value or vector.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SvdDcError> {
        Self::compute_impl(a, Par::Seq)
    }

    /// Same as [`Self::compute`], with the GEMMs of the bidiagonal reduction,
    /// of the divide-and-conquer merges and of the back-transformation run
    /// with `par`.
    ///
    /// The result is bitwise identical to [`Self::compute`] for every `Par`:
    /// the parallel GEMM only partitions its output, and every other stage
    /// runs on the calling thread. Scalar types without an optimised GEMM
    /// (e.g. `f16`) ignore `par`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::compute`].
    #[cfg(feature = "parallel")]
    pub fn compute_with_par(a: MatRef<'_, T>, par: Par) -> Result<Self, SvdDcError> {
        Self::compute_impl(a, par)
    }

    /// [`Self::compute`] with the GEMMs run with `par`.
    fn compute_impl(a: MatRef<'_, T>, par: Par) -> Result<Self, SvdDcError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SvdDcError::EmptyMatrix);
        }

        // Largest entry; a NaN or an infinity has no finite SVD.
        let mut anrm = T::zero();
        for j in 0..n {
            for i in 0..m {
                let x = a[(i, j)];
                if !x.is_finite() {
                    return Err(SvdDcError::NotConverged);
                }
                anrm = anrm.max(fabs(x));
            }
        }

        // Handle 1x1 case
        if m == 1 && n == 1 {
            let val = a[(0, 0)];
            let sigma = vec![Scalar::abs(val)];
            let mut u = Mat::zeros(1, 1);
            let mut vt = Mat::zeros(1, 1);
            u[(0, 0)] = if val >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            vt[(0, 0)] = T::one();
            return Ok(Self { u, sigma, vt, m, n });
        }

        // Scale into the safe range (exact power of two, `dgesdd`).
        let ks = safe_range_exponent(anrm);

        // For wide matrices (m < n), compute the SVD of the (tall) transpose:
        // Aᵀ = U'·Σ·V'ᵀ  ⇒  A = V'·Σ·U'ᵀ, so U = V' and Vᵀ = U'ᵀ.
        if m < n {
            let mut at = Mat::zeros(n, m);
            transpose_into(a, at.as_mut());
            scale_pow2(at.raw_data_mut(), ks);
            let (u_t, mut sigma, v_t) = svd_tall(at, par)?;
            scale_pow2(&mut sigma, -ks);
            let mut vt = Mat::zeros(n, n);
            transpose_into(u_t.as_ref(), vt.as_mut());
            return Ok(Self {
                u: v_t,
                sigma,
                vt,
                m,
                n,
            });
        }

        let mut work = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                work[(i, j)] = a[(i, j)];
            }
        }
        scale_pow2(work.raw_data_mut(), ks);
        let (u, mut sigma, v) = svd_tall(work, par)?;
        scale_pow2(&mut sigma, -ks);
        let mut vt = Mat::zeros(n, n);
        transpose_into(v.as_ref(), vt.as_mut());
        Ok(Self { u, sigma, vt, m, n })
    }

    /// Splits the decomposition into `(U, σ, Vᵀ)`.
    #[cfg(feature = "parallel")]
    pub(crate) fn into_parts(self) -> (Mat<T>, Vec<T>, Mat<T>) {
        (self.u, self.sigma, self.vt)
    }

    /// Returns the singular values (sorted in descending order).
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the left singular vectors U (m×m orthogonal matrix).
    pub fn u(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }

    /// Returns V^T (n×n orthogonal matrix).
    pub fn vt(&self) -> MatRef<'_, T> {
        self.vt.as_ref()
    }

    /// Returns the thin U matrix (m×k where k = min(m,n)).
    pub fn u_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut u_thin = Mat::zeros(self.m, k);
        for i in 0..self.m {
            for j in 0..k {
                u_thin[(i, j)] = self.u[(i, j)];
            }
        }
        u_thin
    }

    /// Returns the thin V^T matrix (k×n where k = min(m,n)).
    pub fn vt_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut vt_thin = Mat::zeros(k, self.n);
        for i in 0..k {
            for j in 0..self.n {
                vt_thin[(i, j)] = self.vt[(i, j)];
            }
        }
        vt_thin
    }

    /// Computes the rank of the matrix given a tolerance.
    pub fn rank(&self, tol: T) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }

    /// Computes the 2-norm (largest singular value).
    pub fn norm2(&self) -> T {
        if self.sigma.is_empty() {
            T::zero()
        } else {
            self.sigma[0]
        }
    }

    /// Computes the condition number (ratio of largest to smallest singular value).
    pub fn condition_number(&self) -> T {
        if self.sigma.is_empty() {
            T::zero()
        } else {
            let max_sv = self.sigma[0];
            let min_sv = self.sigma[self.sigma.len() - 1];
            if min_sv > T::zero() {
                max_sv / min_sv
            } else {
                <T as Scalar>::max_value()
            }
        }
    }

    /// Reconstructs the original matrix: A = U·Σ·V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let mut a = Mat::zeros(self.m, self.n);
        let k = self.m.min(self.n);

        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }

    /// Computes the pseudoinverse using SVD.
    pub fn pseudoinverse(&self, tol: T) -> Mat<T> {
        let mut pinv = Mat::zeros(self.n, self.m);
        let k = self.m.min(self.n);

        for i in 0..self.n {
            for j in 0..self.m {
                let mut sum = T::zero();
                for l in 0..k {
                    if self.sigma[l] > tol {
                        sum = sum + self.vt[(l, i)] * (T::one() / self.sigma[l]) * self.u[(j, l)];
                    }
                }
                pinv[(i, j)] = sum;
            }
        }

        pinv
    }
}

/// Error mapping from the shared bidiagonal kernel.
fn map_bidiag_error(err: BidiagDcError) -> SvdDcError {
    match err {
        BidiagDcError::NotConverged => SvdDcError::NotConverged,
        BidiagDcError::SecularEquationFailed => SvdDcError::SecularEquationFailed,
    }
}

/// The power-of-two exponent `k` that brings `anrm·2^k` into
/// `[smlnum, bignum]` (`smlnum = √safmin/ε`, `bignum = 1/smlnum`, as in
/// `dgesdd`), or `0` when `anrm` already lies there (or is zero).
///
/// For formats too narrow for that range to exist (`smlnum ≥ bignum`, e.g.
/// `f16`) the matrix is scaled to unit max-norm instead.
fn safe_range_exponent<T: Real>(anrm: T) -> i32 {
    if !(anrm > T::zero() && anrm.is_finite()) {
        return 0;
    }
    let smlnum = <T as Real>::sqrt(<T as Scalar>::min_positive()) / <T as Scalar>::epsilon();
    let bignum = T::one() / smlnum;
    if smlnum >= bignum {
        return pow2_exponent(anrm);
    }
    let to_i32 = |x: T| num_traits::ToPrimitive::to_i32(&x).unwrap_or(0);
    if anrm < smlnum {
        let mut k = to_i32(<T as Real>::ceil(num_traits::Float::log2(smlnum / anrm)));
        while mul_pow2(anrm, k) < smlnum {
            k += 1;
        }
        k
    } else if anrm > bignum {
        let mut k = -to_i32(<T as Real>::ceil(num_traits::Float::log2(anrm / bignum)));
        while mul_pow2(anrm, k) > bignum {
            k -= 1;
        }
        k
    } else {
        0
    }
}

/// Full SVD of the tall-or-square `work` (`m ≥ n`), consumed as workspace.
///
/// Returns `(U, σ, V)` with `U` `m×m` and `V` `n×n` (by columns). `f64` and
/// `f32` run the GEMM-backed pipeline with every GEMM in mode `par`; every
/// other real type runs the same algorithm on the portable column-oriented
/// backend.
fn svd_tall<T: Field + Real + bytemuck::Zeroable>(
    mut work: Mat<T>,
    par: Par,
) -> Result<(Mat<T>, Vec<T>, Mat<T>), SvdDcError> {
    let m = work.nrows();
    let n = work.ncols();
    let mut u = Mat::zeros(m, m);
    let mut sigma = vec![T::zero(); n];
    let mut v = Mat::zeros(n, n);
    if let Some(res) = typed_svd_tall::<T, f64>(par, &mut work, &mut u, &mut sigma, &mut v) {
        res?;
    } else if let Some(res) = typed_svd_tall::<T, f32>(par, &mut work, &mut u, &mut sigma, &mut v) {
        res?;
    } else {
        svd_tall_into::<T, PortableOps>(PortableOps, &mut work, &mut u, &mut sigma, &mut v)?;
    }
    Ok((u, sigma, v))
}

/// Runs the GEMM-backed pipeline when `T` is the BLAS scalar `G`.
fn typed_svd_tall<T, G>(
    par: Par,
    work: &mut Mat<T>,
    u: &mut Mat<T>,
    sigma: &mut Vec<T>,
    v: &mut Mat<T>,
) -> Option<Result<(), SvdDcError>>
where
    T: Field + Real + bytemuck::Zeroable,
    G: Field + Real + bytemuck::Zeroable + oxiblas_blas::level3::gemm_kernel::GemmKernel,
{
    let work = same_mut::<Mat<T>, Mat<G>>(work)?;
    let u = same_mut::<Mat<T>, Mat<G>>(u)?;
    let sigma = same_mut::<Vec<T>, Vec<G>>(sigma)?;
    let v = same_mut::<Mat<T>, Mat<G>>(v)?;
    Some(svd_tall_into::<G, BlasOps>(
        BlasOps { par },
        work,
        u,
        sigma,
        v,
    ))
}

/// The divide-and-conquer SVD pipeline for `m ≥ n` (see the module docs),
/// writing `U` (`m×m`), `σ` and `V` (`n×n`, by columns) into the buffers.
fn svd_tall_into<T, B>(
    ops: B,
    work: &mut Mat<T>,
    u: &mut Mat<T>,
    sigma: &mut [T],
    v: &mut Mat<T>,
) -> Result<(), SvdDcError>
where
    T: Field + Real + bytemuck::Zeroable,
    B: DenseOps<T>,
{
    let m = work.nrows();
    let n = work.ncols();
    let factors = gebrd_tall::<T, B>(ops, work);

    let mut u_b = Mat::zeros(n, n);
    bidiag_svd_into::<T, B>(ops, &factors.d, &factors.e, &mut u_b, sigma, v)
        .map_err(map_bidiag_error)?;

    // U = Q · [U_B 0; 0 I].
    for j in 0..n {
        for i in 0..n {
            u[(i, j)] = u_b[(i, j)];
        }
    }
    for i in n..m {
        u[(i, i)] = T::one();
    }
    apply_reflectors_left::<T, B>(ops, work, &factors.tauq, ReflectorLayout::Columns, u);
    // V = P · V_B.
    apply_reflectors_left::<T, B>(ops, work, &factors.taup, ReflectorLayout::Rows, v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_svd_dc_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 4.0, 1e-10));
        assert!(approx_eq(sigma[1], 3.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_2x2() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
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
    fn test_svd_dc_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8));
            }
        }
    }

    #[test]
    fn test_svd_dc_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..2 {
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
    fn test_svd_dc_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let svd = SvdDc::compute(eye.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for &s in sigma {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_svd_dc_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 1);
        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_norm() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.norm2(), 4.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_condition() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.condition_number(), 2.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-12),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    /// `Q·B·Pᵀ = A` for the reduction used by the pipeline (square and tall).
    #[test]
    fn test_bidiagonalize() {
        for a in [
            Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]),
            Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]),
        ] {
            let (m, n) = (a.nrows(), a.ncols());
            let mut work = a.clone();
            let f = gebrd_tall::<f64, BlasOps>(BlasOps::SEQ, &mut work);
            let mut q = Mat::<f64>::zeros(m, m);
            for i in 0..m {
                q[(i, i)] = 1.0;
            }
            apply_reflectors_left::<f64, BlasOps>(
                BlasOps::SEQ,
                &work,
                &f.tauq,
                ReflectorLayout::Columns,
                &mut q,
            );
            let mut p = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                p[(i, i)] = 1.0;
            }
            apply_reflectors_left::<f64, BlasOps>(
                BlasOps::SEQ,
                &work,
                &f.taup,
                ReflectorLayout::Rows,
                &mut p,
            );

            let mut b = Mat::<f64>::zeros(m, n);
            for i in 0..n {
                b[(i, i)] = f.d[i];
                if i + 1 < n {
                    b[(i, i + 1)] = f.e[i];
                }
            }
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for r in 0..m {
                        for t in 0..n {
                            s += q[(i, r)] * b[(r, t)] * p[(j, t)];
                        }
                    }
                    assert!(
                        approx_eq(s, a[(i, j)], 1e-12),
                        "QBP^T[{i},{j}] = {s}, A = {}",
                        a[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_svd_dc_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-4,
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    /// Deterministic pseudo-random values in `[-1, 1)` (no external rng).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn lcg_matrix(m: usize, n: usize, seed: u64) -> Mat<f64> {
        let mut state = seed;
        let mut a = Mat::<f64>::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a[(i, j)] = lcg(&mut state);
            }
        }
        a
    }

    /// Accuracy metrics of an SVD of `a`.
    struct Accuracy {
        /// `‖A − UΣVᵀ‖_F / ‖A‖_F`.
        resid: f64,
        /// `max |UᵀU − I|` over the full `m×m` U.
        orth_u: f64,
        /// `max |VᵀV − I|` over the full `n×n` V.
        orth_v: f64,
    }

    fn accuracy(a: &Mat<f64>, svd: &SvdDc<f64>) -> Accuracy {
        let (m, n) = (a.nrows(), a.ncols());
        let k = m.min(n);
        let u = svd.u();
        let vt = svd.vt();
        let s = svd.singular_values();
        let mut us = Mat::<f64>::zeros(m, k);
        for j in 0..k {
            for i in 0..m {
                us[(i, j)] = u[(i, j)] * s[j];
            }
        }
        let mut rec = Mat::<f64>::zeros(m, n);
        oxiblas_blas::level3::gemm::gemm(
            1.0,
            us.as_ref(),
            vt.submatrix(0, 0, k, n),
            0.0,
            rec.as_mut(),
        );
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for j in 0..n {
            for i in 0..m {
                num += (rec[(i, j)] - a[(i, j)]).powi(2);
                den += a[(i, j)].powi(2);
            }
        }
        let gram_err = |x: MatRef<'_, f64>, trans_first: bool| {
            let dim = if trans_first { x.ncols() } else { x.nrows() };
            let mut g = Mat::<f64>::zeros(dim, dim);
            let (ta, tb) = if trans_first {
                (
                    oxiblas_blas::level3::Trans::Trans,
                    oxiblas_blas::level3::Trans::NoTrans,
                )
            } else {
                (
                    oxiblas_blas::level3::Trans::NoTrans,
                    oxiblas_blas::level3::Trans::Trans,
                )
            };
            oxiblas_blas::level3::gemm::gemm_transposed(ta, tb, 1.0, x, x, 0.0, g.as_mut());
            let mut worst = 0.0f64;
            for i in 0..dim {
                for j in 0..dim {
                    let expect = if i == j { 1.0 } else { 0.0 };
                    worst = worst.max((g[(i, j)] - expect).abs());
                }
            }
            worst
        };
        Accuracy {
            resid: (num / den.max(f64::MIN_POSITIVE)).sqrt(),
            orth_u: gram_err(u, true),
            orth_v: gram_err(vt, false),
        }
    }

    #[test]
    fn test_svd_dc_vs_reference_across_sizes() {
        use crate::svd::{QrSvd, Svd};

        // Sizes straddle SMLSIZ (25): 5, 25 use the bidiagonal QR iteration;
        // 26, 50, 100, 200 exercise the divide-and-conquer merges with
        // deflation.
        for &n in &[5usize, 25, 26, 50, 100, 200] {
            let a = lcg_matrix(n, n, 0x2545_f491_4f6c_dd1d ^ (n as u64));

            let dc = SvdDc::compute(a.as_ref()).unwrap();
            // High-accuracy reference: one-sided Jacobi SVD (used for all sizes).
            let reference = Svd::compute(a.as_ref()).unwrap();

            let s_dc = dc.singular_values();
            let s_ref = reference.singular_values();
            let smax = s_dc[0].abs().max(1.0);

            // Singular values agree tightly with the Jacobi reference SVD.
            for k in 0..n {
                assert!(
                    (s_dc[k] - s_ref[k]).abs() < 1e-12 * smax,
                    "n={n}: sigma[{k}] dc={} jacobi={}",
                    s_dc[k],
                    s_ref[k]
                );
            }

            // Independent cross-check against the QR-based reference SVD
            // (Golub–Kahan–Reinsch), at the same tolerance as the Jacobi
            // check (the two agree to about 2e-15·σ_max at these sizes).
            if n <= 50 {
                let qr_reference = QrSvd::compute(a.as_ref()).unwrap();
                let s_qr = qr_reference.singular_values();
                for k in 0..n {
                    assert!(
                        (s_dc[k] - s_qr[k]).abs() < 1e-12 * smax,
                        "n={n}: sigma[{k}] dc={} qr={}",
                        s_dc[k],
                        s_qr[k]
                    );
                }
            }

            // UᵀU = I and VᵀV = I to working precision.
            let acc = accuracy(&a, &dc);
            assert!(acc.orth_u < 1e-12, "n={n}: UtU error {}", acc.orth_u);
            assert!(acc.orth_v < 1e-12, "n={n}: VtV error {}", acc.orth_v);

            // Reconstruction ||A - U Σ Vᵀ|| is tight.
            let rec = dc.reconstruct();
            let mut rec_err = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    rec_err = rec_err.max((rec[(i, j)] - a[(i, j)]).abs());
                }
            }
            assert!(
                rec_err < 1e-12 * smax,
                "n={n}: reconstruction error {rec_err}"
            );
        }
    }

    /// Checks the SVD of `a` (`m × n`) against reference singular values
    /// `s_ref` (descending): σ to `1e-12·σ_max`, the shapes of the full U and
    /// V, the reconstruction and the orthogonality of U and V.
    fn check_against_reference(what: &str, a: &Mat<f64>, s_ref: &[f64]) {
        let (m, n) = (a.nrows(), a.ncols());
        let k = m.min(n);
        let dc = SvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(dc.singular_values().len(), k);
        assert_eq!((dc.u().nrows(), dc.u().ncols()), (m, m));
        assert_eq!((dc.vt().nrows(), dc.vt().ncols()), (n, n));
        let s_dc = dc.singular_values();
        for i in 0..k {
            assert!(
                (s_dc[i] - s_ref[i]).abs() < 1e-12 * s_ref[0],
                "{what}: sigma[{i}] dc={} reference={}",
                s_dc[i],
                s_ref[i]
            );
        }
        let acc = accuracy(a, &dc);
        assert!(acc.resid < 1e-13, "{what}: residual {}", acc.resid);
        assert!(acc.orth_u < 1e-12, "{what}: UtU error {}", acc.orth_u);
        assert!(acc.orth_v < 1e-12, "{what}: VtV error {}", acc.orth_v);
    }

    /// Random `m × n` matrix checked against the one-sided Jacobi `Svd`.
    ///
    /// The Jacobi reference dominates the run time in the debug profile
    /// (minutes for `300 × 31` or `500 × 500`), so the tests that use it for
    /// large shapes are release-only; their debug counterparts use
    /// [`check_prescribed`] at the same shapes.
    fn check_random_vs_jacobi(m: usize, n: usize, seed: u64) {
        use crate::svd::Svd;
        let a = lcg_matrix(m, n, seed);
        let jac = Svd::compute(a.as_ref()).unwrap();
        check_against_reference(&format!("{m}x{n} random"), &a, jac.singular_values());
    }

    /// `m × n` matrix with prescribed singular values spread over three
    /// decades (irregularly spaced, descending), checked against them.
    fn check_prescribed(m: usize, n: usize, seed: u64) {
        let k = m.min(n);
        let mut state = seed;
        let mut s: Vec<f64> = (0..k)
            .map(|i| {
                let t = if k > 1 {
                    i as f64 / (k - 1) as f64
                } else {
                    0.0
                };
                10.0 * 10f64.powf(-3.0 * t) * (1.0 + 0.05 * lcg(&mut state))
            })
            .collect();
        s.sort_by(|x, y| y.total_cmp(x));
        let a = matrix_with_singular_values(m, n, &s, seed);
        check_against_reference(&format!("{m}x{n} prescribed"), &a, &s);
    }

    /// n = 300: blocked bidiagonalisation and several levels of GEMM merges.
    #[test]
    fn test_svd_dc_accuracy_n300() {
        check_prescribed(300, 300, 0x5eed_0000 ^ 300);
    }

    /// n = 500 (leading dimension 512, the former 4 KiB-stride slow case).
    #[test]
    fn test_svd_dc_accuracy_n500() {
        check_prescribed(500, 500, 0x5eed_0000 ^ 500);
    }

    /// [`test_svd_dc_accuracy_n300`] on a random matrix, against Jacobi.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "slow in debug (one-sided Jacobi reference); run with --release"
    )]
    fn test_svd_dc_accuracy_n300_vs_jacobi() {
        check_random_vs_jacobi(300, 300, 0x5eed_0000 ^ 300);
    }

    /// [`test_svd_dc_accuracy_n500`] on a random matrix, against Jacobi.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "slow in debug (one-sided Jacobi reference); run with --release"
    )]
    fn test_svd_dc_accuracy_n500_vs_jacobi() {
        check_random_vs_jacobi(500, 500, 0x5eed_0000 ^ 500);
    }

    /// Tall and wide shapes (blocked and unblocked reductions, full U / V).
    const TALL_AND_WIDE_SHAPES: [(usize, usize); 5] =
        [(260, 150), (150, 260), (61, 7), (7, 61), (300, 31)];

    /// Tall and wide shapes: the small ones (cheap even in debug) as random
    /// matrices against Jacobi, the others with prescribed singular values.
    #[test]
    fn test_svd_dc_tall_and_wide_shapes() {
        for &(m, n) in &TALL_AND_WIDE_SHAPES {
            let seed = 0xabcd ^ (m * 1000 + n) as u64;
            if m * n <= 1_000 {
                check_random_vs_jacobi(m, n, seed);
            } else {
                check_prescribed(m, n, seed);
            }
        }
    }

    /// [`test_svd_dc_tall_and_wide_shapes`] with every shape a random matrix
    /// checked against Jacobi.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "slow in debug (one-sided Jacobi reference); run with --release"
    )]
    fn test_svd_dc_tall_and_wide_shapes_vs_jacobi() {
        for &(m, n) in &TALL_AND_WIDE_SHAPES {
            check_random_vs_jacobi(m, n, 0xabcd ^ (m * 1000 + n) as u64);
        }
    }

    /// `A = H·diag(σ)·Gᵀ` (`m × n`) where `H` and `G` are products of three
    /// random Householder reflections, so the singular values of `A` are
    /// exactly `σ` (descending) up to rounding.
    fn matrix_with_singular_values(m: usize, n: usize, sigma: &[f64], seed: u64) -> Mat<f64> {
        let mut a = Mat::<f64>::zeros(m, n);
        for (i, &s) in sigma.iter().enumerate() {
            a[(i, i)] = s;
        }
        let mut state = seed;
        let mut unit = |len: usize| {
            let mut v: Vec<f64> = (0..len).map(|_| lcg(&mut state)).collect();
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            v
        };
        for _ in 0..3 {
            // a ← (I − 2vvᵀ)·a
            let v = unit(m);
            for j in 0..n {
                let dotp: f64 = (0..m).map(|i| v[i] * a[(i, j)]).sum();
                for i in 0..m {
                    a[(i, j)] -= 2.0 * dotp * v[i];
                }
            }
            // a ← a·(I − 2wwᵀ)
            let w = unit(n);
            for i in 0..m {
                let dotp: f64 = (0..n).map(|j| a[(i, j)] * w[j]).sum();
                for j in 0..n {
                    a[(i, j)] -= 2.0 * dotp * w[j];
                }
            }
        }
        a
    }

    /// f32 through the GEMM-backed pipeline (blocked sizes included).
    #[test]
    fn test_svd_dc_f32_large() {
        for &(m, n) in &[(200usize, 200usize), (230, 140), (90, 170)] {
            let a64 = lcg_matrix(m, n, 0xf32 ^ (m + n) as u64);
            let mut a = Mat::<f32>::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    a[(i, j)] = a64[(i, j)] as f32;
                }
            }
            let dc = SvdDc::compute(a.as_ref()).unwrap();
            let k = m.min(n);
            let (u, vt, s) = (dc.u(), dc.vt(), dc.singular_values());
            let mut worst_u = 0.0f32;
            for i in 0..m {
                for j in 0..m {
                    let mut g = 0.0f32;
                    for r in 0..m {
                        g += u[(r, i)] * u[(r, j)];
                    }
                    worst_u = worst_u.max((g - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            let mut worst_v = 0.0f32;
            for i in 0..n {
                for j in 0..n {
                    let mut g = 0.0f32;
                    for r in 0..n {
                        g += vt[(i, r)] * vt[(j, r)];
                    }
                    worst_v = worst_v.max((g - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            let mut worst_rec = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut x = 0.0f32;
                    for l in 0..k {
                        x += u[(i, l)] * s[l] * vt[(l, j)];
                    }
                    worst_rec = worst_rec.max((x - a[(i, j)]).abs());
                }
            }
            // c·n·ε_f32 with c = 1 (observed: about 1e-6 and 2.5e-7·σ₁); the
            // former BᵀB solver reached 3e-4·σ₁ on graded f32 spectra.
            let tol = m.max(n) as f32 * f32::EPSILON;
            assert!(worst_u < tol, "{m}x{n} f32 UtU {worst_u}");
            assert!(worst_v < tol, "{m}x{n} f32 VtV {worst_v}");
            assert!(worst_rec < tol * s[0], "{m}x{n} f32 recon {worst_rec}");
        }
    }

    /// Zero and rank-one inputs: (almost) every column of U comes from the
    /// orthonormal completion, which must still give orthogonal U and V.
    ///
    /// The exactly rank-one matrix also drives the trailing "noise of noise"
    /// of the reduction into subnormal numbers, which once turned `larfg`'s
    /// norm (and so the whole bidiagonal) into NaN.
    #[test]
    fn test_svd_dc_zero_and_rank_one() {
        let z = Mat::<f64>::zeros(60, 45);
        let dc = SvdDc::compute(z.as_ref()).unwrap();
        assert!(dc.singular_values().iter().all(|&s| s == 0.0));
        let acc = accuracy(&Mat::<f64>::zeros(60, 45), &dc);
        assert!(
            acc.orth_u < 1e-12 && acc.orth_v < 1e-12,
            "zero matrix: UtU {} VtV {}",
            acc.orth_u,
            acc.orth_v
        );

        let n = 150;
        let mut a = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                a[(i, j)] = ((i % 7) as f64 - 3.0) * ((j % 5) as f64 + 1.0);
            }
        }
        let dc = SvdDc::compute(a.as_ref()).unwrap();
        let s = dc.singular_values();
        assert!(s[0] > 1.0);
        assert!(s[1] < 1e-12 * s[0], "rank one: sigma[1] = {}", s[1]);
        let acc = accuracy(&a, &dc);
        assert!(acc.resid < 1e-13, "rank one residual {}", acc.resid);
        assert!(acc.orth_u < 1e-12, "rank one UtU {}", acc.orth_u);
        assert!(acc.orth_v < 1e-12, "rank one VtV {}", acc.orth_v);
    }

    /// The portable backend (used for scalars without a BLAS kernel, e.g.
    /// `f16`) runs the same pipeline: same σ, orthogonal U and V.
    #[test]
    fn test_svd_dc_portable_backend() {
        let (m, n) = (150usize, 140usize);
        let a = lcg_matrix(m, n, 0x9047);
        let reference = SvdDc::compute(a.as_ref()).unwrap();

        let mut work = a.clone();
        let mut u = Mat::<f64>::zeros(m, m);
        let mut sigma = vec![0.0f64; n];
        let mut v = Mat::<f64>::zeros(n, n);
        svd_tall_into::<f64, PortableOps>(PortableOps, &mut work, &mut u, &mut sigma, &mut v)
            .unwrap();
        let s_ref = reference.singular_values();
        for k in 0..n {
            assert!(
                (sigma[k] - s_ref[k]).abs() < 1e-12 * s_ref[0],
                "sigma[{k}]: portable {} blas {}",
                sigma[k],
                s_ref[k]
            );
        }
        let mut vt = Mat::<f64>::zeros(n, n);
        transpose_into(v.as_ref(), vt.as_mut());
        let portable = SvdDc { u, sigma, vt, m, n };
        let acc = accuracy(&a, &portable);
        assert!(acc.resid < 1e-13, "portable residual {}", acc.resid);
        assert!(acc.orth_u < 1e-12, "portable UtU {}", acc.orth_u);
        assert!(acc.orth_v < 1e-12, "portable VtV {}", acc.orth_v);
    }

    /// Rank-deficient input: the null-space part of U is completed with an
    /// orthonormal basis and the reconstruction stays exact.
    #[test]
    fn test_svd_dc_rank_deficient() {
        for &n in &[40usize, 200] {
            // a[i, j] = i + j·n + 1 has rank 2.
            let mut a = Mat::<f64>::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    a[(i, j)] = (i + j * n + 1) as f64;
                }
            }
            let dc = SvdDc::compute(a.as_ref()).unwrap();
            let s = dc.singular_values();
            assert!(s[1] > 1e-6 * s[0]);
            for (k, &sk) in s.iter().enumerate().skip(2) {
                assert!(sk < 1e-10 * s[0], "n={n}: sigma[{k}] = {sk}");
            }
            let acc = accuracy(&a, &dc);
            assert!(acc.resid < 1e-13, "n={n}: residual {}", acc.resid);
            assert!(acc.orth_u < 1e-12, "n={n}: UtU error {}", acc.orth_u);
            assert!(acc.orth_v < 1e-12, "n={n}: VtV error {}", acc.orth_v);
        }
    }
}
