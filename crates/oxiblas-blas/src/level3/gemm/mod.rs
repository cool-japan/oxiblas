//! General Matrix Multiplication (GEMM).
//!
//! Computes C = alpha * op(A) * op(B) + beta * C
//!
//! This implementation uses a BLIS-style blocked algorithm with
//! SIMD-optimized micro-kernels for high performance.
//!
//! ## Transpose support
//!
//! The plain [`gemm`] entry point computes `C = alpha*A*B + beta*C` (no
//! transpose). The [`gemm_transposed`] family adds `trans_a`/`trans_b`
//! parameters so callers can compute `op(A)*op(B)` where `op(X)` is `X`,
//! `X^T`, or `X^H` (conjugate transpose) without allocating a transposed copy.
//!
//! Transpose is threaded through the **packing layer**: the micro-kernels only
//! ever see packed panels, so a transposed operand is simply *read with swapped
//! (row, col) indices* (and conjugated for `ConjTrans`) while it is copied into
//! the packing buffer. This is genuinely zero-copy — no materialized transpose
//! of the source is ever produced for the real (`f32`/`f64`) path.
//!
//! The complex wrappers [`gemm_transposed_c64`] / [`gemm_transposed_c32`] route
//! through the 3M complex kernel (which already splits operands into real/imag
//! buffers). For a transposed complex operand they materialize `op(A)`/`op(B)`
//! once; this is an honest fallback, not a zero-copy path, and is documented as
//! such on those functions.
//!
//! ## Parallelization
//!
//! When the `parallel` feature is enabled and `Par::Rayon` is used,
//! GEMM operations are parallelized over a 2D grid of blocks of C.
//! This provides good work distribution without requiring synchronization.

use crate::level3::complex_gemm::{gemm3m_c32, gemm3m_c64};
use crate::level3::gemm_kernel::{GemmKernel, MicroKernelShape};
use crate::level3::gemm_packing::{pack_a_optimized, pack_b_optimized};
use crate::level3::gemm_small::{SMALL_THRESHOLD, gemm_small};
use crate::level3::trsm::Trans;
use num_complex::{Complex32, Complex64};
use oxiblas_core::memory::{AlignedVec, StackReq};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::ParThreshold;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Smallest number of multiply-adds for one task of the parallel GEMM.
#[cfg(feature = "parallel")]
const PAR_TASK_MIN_WORK: usize = 1 << 25;

/// Smallest `m * n * k` below [`SMALL_THRESHOLD`] that uses the blocked path.
const SMALL_BLOCKED_MIN: usize = 12 * 12 * 12;

/// Blocking parameters for GEMM.
///
/// These parameters control the cache-aware blocking of the GEMM algorithm.
#[derive(Debug, Clone, Copy)]
pub struct GemmBlocking {
    /// Block size for columns of B (NC).
    pub nc: usize,
    /// Block size for the K dimension (KC).
    pub kc: usize,
    /// Block size for rows of A (MC).
    pub mc: usize,
}

impl Default for GemmBlocking {
    fn default() -> Self {
        Self {
            nc: 2048, // L3 cache blocking
            kc: 128,  // L2 cache blocking
            mc: 512,  // L1 cache blocking
        }
    }
}

impl GemmBlocking {
    /// Creates blocking parameters optimized for the given micro-kernel shape.
    ///
    /// Parameters are tuned based on element size and cache hierarchy:
    /// - f64: mc=576, kc=384, nc=2048 (optimized for large L2 caches)
    /// - f32: mc=576, kc=768, nc=2048 (larger KC for smaller elements)
    ///
    /// Apple Silicon cache hierarchy (M1/M2/M3):
    /// - L1D: 128-192 KB per performance core
    /// - L2: 4-16 MB per cluster
    /// - Cache line: 128 bytes (important for alignment)
    ///
    /// These defaults are optimized for modern CPUs with large L2 caches.
    /// For size-adaptive blocking that considers matrix dimensions,
    /// use [`GemmBlocking::auto_tuned`] or [`GemmBlocking::asymmetric`].
    #[must_use]
    pub const fn for_kernel<T: Field>(shape: &MicroKernelShape) -> Self {
        let mr = shape.mr;
        let nr = shape.nr;

        // Optimal parameters depend on element size
        // f64 = 8 bytes, f32 = 4 bytes
        let elem_size = std::mem::size_of::<T>();

        // Optimized for modern CPUs with large L2 caches (4+ MB):
        // - Larger KC reduces K-loop iterations → less packing overhead
        // - B micro-panel (KC×NR) should fit in L1
        // - pack_a (MC×KC) should fit in L2 (~70% utilization)
        //
        // For f64 (8×6 micro-kernel, 128-byte cache lines on Apple Silicon):
        //   - KC=448: B micro-panel = 448×6×8 = 21 KB (fits in 128 KB L1)
        //   - MC=576: pack_a = 576×448×8 = 2.0 MB (fits in 4+ MB L2)
        //   - NC=2046: pack_b = 448×2046×8 = 7.3 MB (fits in L3/SLC)
        //   - 17% fewer K-loop iterations vs KC=384
        //
        // For f32 (8×8 micro-kernel):
        //   - KC=896: B micro-panel = 896×8×4 = 28 KB (fits in L1)
        //   - MC=576: pack_a = 576×896×4 = 2.0 MB (fits in L2)
        //   - NC=2048: pack_b = 896×2048×4 = 7.3 MB (fits in L3)
        let (mc, kc) = if elem_size >= 8 {
            // f64: optimized for Apple Silicon's 4MB L2
            (576, 448)
        } else {
            // f32: larger KC optimized for Apple Silicon's 4MB L2
            (576, 896)
        };

        Self {
            nc: (2048 / nr) * nr,
            kc,
            mc: (mc / mr) * mr,
        }
    }

    /// Creates asymmetric blocking parameters optimized for rectangular matrices.
    ///
    /// For highly rectangular matrices (tall-thin or short-wide), this method
    /// adjusts blocking parameters to better utilize cache and reduce overhead.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in A and C
    /// * `k` - Inner dimension (columns of A, rows of B)
    /// * `n` - Number of columns in B and C
    /// * `shape` - Micro-kernel shape
    ///
    /// # Blocking Strategy
    ///
    /// - **Tall-thin (m >> k, m >> n)**: Larger MC, smaller KC/NC
    /// - **Short-wide (n >> m, n >> k)**: Larger NC, smaller MC/KC
    /// - **Inner-product (k >> m, k >> n)**: Larger KC, smaller MC/NC
    /// - **Balanced**: Standard blocking
    #[must_use]
    pub fn asymmetric<T: Field>(m: usize, k: usize, n: usize, shape: &MicroKernelShape) -> Self {
        let mr = shape.mr;
        let nr = shape.nr;
        let elem_size = std::mem::size_of::<T>();

        // Compute aspect ratios
        let max_dim = m.max(k).max(n);
        let min_dim = m.min(k).min(n);

        // If dimensions are roughly balanced (within 4x), use standard blocking
        if max_dim < 4 * min_dim {
            return Self::for_kernel::<T>(shape);
        }

        // Base blocking parameters (matching for_kernel)
        let (base_mc, base_kc) = if elem_size >= 8 {
            (512, 256)
        } else {
            (512, 512)
        };
        let base_nc = 2048;

        // Tall-thin matrix: A is tall (m >> k) and C is tall (m >> n)
        // Prioritize MC to process more rows per iteration
        if m > 4 * k && m > 4 * n {
            // Increase MC, decrease KC and NC
            let mc = (base_mc * 2).min(m).max(mr);
            let kc = (base_kc / 2).max(32);
            let nc = (base_nc / 2).max(nr);

            return Self {
                mc: (mc / mr) * mr,
                kc,
                nc: (nc / nr) * nr,
            };
        }

        // Short-wide matrix: C is wide (n >> m)
        // Prioritize NC to process more columns per iteration
        if n > 4 * m && n > 4 * k {
            // Increase NC, decrease MC
            let mc = (base_mc / 2).max(mr);
            let kc = base_kc;
            let nc = (base_nc * 2).min(n).max(nr);

            return Self {
                mc: (mc / mr) * mr,
                kc,
                nc: (nc / nr) * nr,
            };
        }

        // Inner-product dominated: k >> m, k >> n
        // Prioritize KC to reduce packing overhead for the K dimension
        if k > 4 * m && k > 4 * n {
            // Increase KC significantly to amortize packing cost
            let mc = (base_mc / 2).max(mr);
            let kc = (base_kc * 4).min(k);
            let nc = (base_nc / 2).max(nr);

            return Self {
                mc: (mc / mr) * mr,
                kc,
                nc: (nc / nr) * nr,
            };
        }

        // Panel-panel: m and n are both large, k is small
        // This is already handled well by standard blocking
        if m > 4 * k && n > 4 * k {
            // Reduce KC since K is small
            let mc = base_mc;
            let kc = base_kc.min(k);
            let nc = base_nc;

            return Self {
                mc: (mc / mr) * mr,
                kc,
                nc: (nc / nr) * nr,
            };
        }

        // Default to standard blocking
        Self::for_kernel::<T>(shape)
    }

    /// Creates custom blocking parameters with alignment to micro-kernel shape.
    ///
    /// Block sizes are rounded down to a multiple of the micro-kernel shape
    /// (`mr` for `mc`, `nr` for `nc`) and then **clamped to a valid minimum**:
    /// `mc >= mr`, `nc >= nr`, and `kc >= 1`. Without this clamp, a caller
    /// passing `mc < mr`, `nc < nr`, or `kc == 0` would round down to a zero
    /// block size, and the resulting `step_by(0)` in the blocked GEMM loops
    /// would panic deep inside the kernel. Clamping (rather than returning an
    /// error) matches the silent-minimum convention used by
    /// [`GemmBlocking::asymmetric`].
    #[must_use]
    pub const fn custom(mc: usize, kc: usize, nc: usize, shape: &MicroKernelShape) -> Self {
        let mr = shape.mr;
        let nr = shape.nr;

        // Round down to the micro-kernel multiple, then clamp up to at least one
        // micro-tile so no dimension can collapse to zero.
        let mc_aligned = (mc / mr) * mr;
        let mc_final = if mc_aligned < mr { mr } else { mc_aligned };

        let nc_aligned = (nc / nr) * nr;
        let nc_final = if nc_aligned < nr { nr } else { nc_aligned };

        let kc_final = if kc == 0 { 1 } else { kc };

        Self {
            nc: nc_final,
            kc: kc_final,
            mc: mc_final,
        }
    }

    /// Returns the scratch space requirement for packing.
    ///
    /// Note: Matrix dimensions are accepted for API consistency but the current
    /// implementation uses blocking parameters for a conservative estimate.
    #[must_use]
    pub const fn pack_req<T: Field>(&self, _m: usize, _n: usize, _k: usize) -> StackReq {
        let pack_a_size = self.mc * self.kc;
        let pack_b_size = self.kc * self.nc;

        StackReq::new_for::<T>(pack_a_size + pack_b_size)
    }

    /// Creates blocking parameters that are auto-tuned for the specific matrix dimensions.
    ///
    /// This method uses runtime cache detection to compute optimal blocking parameters.
    /// It considers:
    /// - CPU cache sizes (L1, L2, L3)
    /// - Matrix dimensions (m, k, n)
    /// - Element size (f32 vs f64)
    /// - Micro-kernel shape (MR × NR)
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in A and C
    /// * `k` - Inner dimension (columns of A, rows of B)
    /// * `n` - Number of columns in B and C
    /// * `shape` - Micro-kernel shape
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_blas::level3::{GemmBlocking, GemmKernel};
    ///
    /// let shape = f64::micro_kernel_shape();
    /// let blocking = GemmBlocking::auto_tuned::<f64>(1024, 1024, 1024, &shape);
    /// println!("Auto-tuned: MC={}, KC={}, NC={}", blocking.mc, blocking.kc, blocking.nc);
    /// ```
    #[must_use]
    pub fn auto_tuned<T: Field>(m: usize, k: usize, n: usize, shape: &MicroKernelShape) -> Self {
        let elem_size = std::mem::size_of::<T>();
        let tuned = crate::level3::autotune::compute_blocking_adaptive(m, k, n, elem_size, shape);

        Self {
            mc: tuned.mc,
            kc: tuned.kc,
            nc: tuned.nc,
        }
    }
}

/// GEMM operation: C = alpha * A * B + beta * C
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A * B
/// * `a` - Left matrix (m × k)
/// * `b` - Right matrix (k × n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m × n)
///
/// # Panics
///
/// Panics if matrix dimensions are incompatible.
pub fn gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) {
    gemm_with_par(alpha, a, b, beta, c, Par::Seq);
}

/// GEMM with parallelization control.
///
/// Automatically selects between standard and auto-tuned blocking based on matrix size.
/// For matrices larger than 512x512x512, auto-tuning is used for better performance.
pub fn gemm_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let shape = T::micro_kernel_shape();

    // Use auto-tuning for larger matrices
    const AUTO_TUNE_THRESHOLD: usize = 512;
    let blocking =
        if m >= AUTO_TUNE_THRESHOLD || k >= AUTO_TUNE_THRESHOLD || n >= AUTO_TUNE_THRESHOLD {
            GemmBlocking::auto_tuned::<T>(m, k, n, &shape)
        } else {
            GemmBlocking::for_kernel::<T>(&shape)
        };

    gemm_with_blocking(alpha, a, b, beta, c, par, &blocking);
}

/// GEMM with asymmetric blocking optimized for rectangular matrices.
///
/// This variant automatically selects blocking parameters based on the
/// aspect ratio of the input matrices, which can improve performance
/// for highly rectangular matrices (tall-thin, short-wide, or inner-product dominated).
///
/// For balanced (roughly square) matrices, this falls back to standard blocking.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::gemm::gemm_asymmetric;
/// use oxiblas_matrix::Mat;
///
/// // Tall-thin matrix multiplication: (1000 x 10) * (10 x 100)
/// let a: Mat<f64> = Mat::filled(1000, 10, 1.0);
/// let b: Mat<f64> = Mat::filled(10, 100, 2.0);
/// let mut c: Mat<f64> = Mat::zeros(1000, 100);
///
/// gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
///
/// // Each element = 10 * 1.0 * 2.0 = 20.0
/// assert!((c[(0, 0)] - 20.0).abs() < 1e-10);
/// ```
pub fn gemm_asymmetric<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) {
    gemm_asymmetric_with_par(alpha, a, b, beta, c, Par::Seq);
}

/// GEMM with asymmetric blocking and parallelization control.
pub fn gemm_asymmetric_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let shape = T::micro_kernel_shape();
    let blocking = GemmBlocking::asymmetric::<T>(m, k, n, &shape);
    gemm_with_blocking(alpha, a, b, beta, c, par, &blocking);
}

/// GEMM with auto-tuned blocking based on runtime cache detection.
///
/// This variant uses runtime detection of CPU cache sizes to compute
/// optimal blocking parameters (MC, KC, NC). It provides the best
/// performance on machines with known cache hierarchies.
///
/// # Cache-Aware Blocking
///
/// The algorithm computes:
/// - KC: B micro-panel (KC × NR) should fit in L1
/// - MC: A macro-panel (MC × KC) should fit in L2
/// - NC: B macro-panel (KC × NC) should fit in L3
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::gemm::gemm_auto;
/// use oxiblas_matrix::Mat;
///
/// let a: Mat<f64> = Mat::filled(1024, 512, 1.0);
/// let b: Mat<f64> = Mat::filled(512, 768, 2.0);
/// let mut c: Mat<f64> = Mat::zeros(1024, 768);
///
/// gemm_auto(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
///
/// // Each element = 512 * 1.0 * 2.0 = 1024.0
/// assert!((c[(0, 0)] - 1024.0).abs() < 1e-8);
/// ```
pub fn gemm_auto<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) {
    gemm_auto_with_par(alpha, a, b, beta, c, Par::Seq);
}

/// GEMM with auto-tuned blocking and parallelization control.
pub fn gemm_auto_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let shape = T::micro_kernel_shape();
    let blocking = GemmBlocking::auto_tuned::<T>(m, k, n, &shape);
    gemm_with_blocking(alpha, a, b, beta, c, par, &blocking);
}

/// GEMM with custom blocking parameters (for benchmarking/tuning).
///
/// This is the no-transpose entry point; it delegates to
/// [`gemm_transposed_with_blocking`] with `Trans::NoTrans` for both operands,
/// so its behavior for existing callers is unchanged.
pub fn gemm_with_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
    blocking: &GemmBlocking,
) {
    gemm_transposed_with_blocking(
        Trans::NoTrans,
        Trans::NoTrans,
        alpha,
        a,
        b,
        beta,
        c,
        par,
        blocking,
    );
}

/// Transpose-aware GEMM: `C = alpha * op(A) * op(B) + beta * C`.
///
/// `op(X)` is `X` for `Trans::NoTrans`, `X^T` for `Trans::Trans`, and `X^H`
/// (conjugate transpose) for `Trans::ConjTrans`. For the real (`f32`/`f64`)
/// element types covered by [`GemmKernel`], `ConjTrans` is identical to `Trans`
/// because conjugation is the identity on reals; the parameter is still
/// accepted so the API is uniform with the complex wrappers
/// ([`gemm_transposed_c64`] / [`gemm_transposed_c32`]).
///
/// Transposed operands are **not** copied: they are read with swapped indices
/// directly while packing (see the module docs). The `NoTrans`/`NoTrans` case
/// takes exactly the same code path as [`gemm`].
///
/// # Panics
///
/// Panics if the (post-transpose) operand dimensions are incompatible, i.e. if
/// `op(A).ncols != op(B).nrows`, `C.nrows != op(A).nrows`, or
/// `C.ncols != op(B).ncols`.
pub fn gemm_transposed<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
) {
    gemm_transposed_with_par(trans_a, trans_b, alpha, a, b, beta, c, Par::Seq);
}

/// Transpose-aware GEMM with parallelization control.
///
/// Selects blocking parameters from the *logical* (post-transpose) dimensions,
/// mirroring [`gemm_with_par`].
pub fn gemm_transposed_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
) {
    // Logical dimensions of op(A) (m x k) and op(B) (k x n).
    let (m, k) = op_dims(&a, trans_a);
    let (_, n) = op_dims(&b, trans_b);

    let shape = T::micro_kernel_shape();

    const AUTO_TUNE_THRESHOLD: usize = 512;
    let blocking =
        if m >= AUTO_TUNE_THRESHOLD || k >= AUTO_TUNE_THRESHOLD || n >= AUTO_TUNE_THRESHOLD {
            GemmBlocking::auto_tuned::<T>(m, k, n, &shape)
        } else {
            GemmBlocking::for_kernel::<T>(&shape)
        };

    gemm_transposed_with_blocking(trans_a, trans_b, alpha, a, b, beta, c, par, &blocking);
}

/// Transpose-aware GEMM core with explicit blocking parameters.
///
/// All the public GEMM entry points funnel through here. The `NoTrans`/`NoTrans`
/// path is byte-for-byte the previous implementation (same small-matrix fast
/// path, same packers, same blocked kernel); the transpose flags only change how
/// operands are *read* during packing.
#[allow(clippy::too_many_arguments)]
pub fn gemm_transposed_with_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    par: Par,
    blocking: &GemmBlocking,
) {
    // Logical (post-transpose) dimensions: op(A) is m x k, op(B) is k x n.
    let (m, k) = op_dims(&a, trans_a);
    let (kb, n) = op_dims(&b, trans_b);

    // Dimension checks on the logical operands.
    assert_eq!(k, kb, "op(A).ncols must equal op(B).nrows");
    assert_eq!(c.nrows(), m, "C.nrows must equal op(A).nrows");
    assert_eq!(c.ncols(), n, "C.ncols must equal op(B).ncols");

    // Handle trivial cases.
    if m == 0 || n == 0 {
        return;
    }

    if k == 0 {
        // C = beta * C
        scale_matrix(&mut c, beta);
        return;
    }

    let shape = T::micro_kernel_shape();

    // Small matrix fast path; transposed cases use the transpose-aware naive kernel.
    // Problems that fill a micro-tile are faster blocked from SMALL_BLOCKED_MIN; alpha == 0 skips A and B there.
    let blocked =
        m >= shape.mr && n >= shape.nr && m * n * k >= SMALL_BLOCKED_MIN && alpha != T::zero();
    if m * n * k <= SMALL_THRESHOLD && !blocked {
        if trans_a == Trans::NoTrans && trans_b == Trans::NoTrans {
            gemm_small(alpha, &a, &b, beta, &mut c);
        } else {
            gemm_small_trans(trans_a, trans_b, alpha, &a, &b, beta, &mut c);
        }
        return;
    }

    // Use blocked GEMM.
    gemm_blocked(
        trans_a, trans_b, alpha, &a, &b, beta, &mut c, blocking, &shape, par,
    );
}

/// Complex transpose-aware GEMM for `Complex64`: `C = alpha * op(A) * op(B) + beta * C`.
///
/// `op(X)` is `X` (`NoTrans`), `X^T` (`Trans`), or `X^H` (`ConjTrans`).
///
/// # Zero-copy note
///
/// This routes through the 3M complex kernel [`gemm3m_c64`], which already
/// splits its operands into real/imag buffers. For a transposed or
/// conjugate-transposed operand this wrapper materializes `op(A)` / `op(B)` once
/// (an honest copy, **not** a zero-copy path — unlike the real
/// [`gemm_transposed`]). The `NoTrans`/`NoTrans` case allocates nothing beyond
/// what `gemm3m_c64` itself uses.
pub fn gemm_transposed_c64(
    trans_a: Trans,
    trans_b: Trans,
    alpha: Complex64,
    a: MatRef<'_, Complex64>,
    b: MatRef<'_, Complex64>,
    beta: Complex64,
    c: MatMut<'_, Complex64>,
) {
    match (trans_a, trans_b) {
        (Trans::NoTrans, Trans::NoTrans) => gemm3m_c64(alpha, a, b, beta, c),
        _ => {
            let op_a = materialize_op(&a, trans_a);
            let op_b = materialize_op(&b, trans_b);
            gemm3m_c64(alpha, op_a.as_ref(), op_b.as_ref(), beta, c);
        }
    }
}

/// Complex transpose-aware GEMM for `Complex32`.
///
/// See [`gemm_transposed_c64`] for the zero-copy caveat (transposed operands are
/// materialized once before the 3M kernel runs).
pub fn gemm_transposed_c32(
    trans_a: Trans,
    trans_b: Trans,
    alpha: Complex32,
    a: MatRef<'_, Complex32>,
    b: MatRef<'_, Complex32>,
    beta: Complex32,
    c: MatMut<'_, Complex32>,
) {
    match (trans_a, trans_b) {
        (Trans::NoTrans, Trans::NoTrans) => gemm3m_c32(alpha, a, b, beta, c),
        _ => {
            let op_a = materialize_op(&a, trans_a);
            let op_b = materialize_op(&b, trans_b);
            gemm3m_c32(alpha, op_a.as_ref(), op_b.as_ref(), beta, c);
        }
    }
}

/// Returns the dimensions `(rows, cols)` of `op(mat)` for the given transpose.
///
/// `Trans`/`ConjTrans` swap rows and cols; `NoTrans` leaves them as-is.
#[inline]
fn op_dims<T: Scalar>(mat: &MatRef<'_, T>, trans: Trans) -> (usize, usize) {
    match trans {
        Trans::NoTrans => (mat.nrows(), mat.ncols()),
        Trans::Trans | Trans::ConjTrans => (mat.ncols(), mat.nrows()),
    }
}

/// Reads the element `op(mat)[row, col]` for the given transpose mode.
///
/// `NoTrans` reads `mat[row, col]`; `Trans`/`ConjTrans` read `mat[col, row]`
/// (and conjugate for `ConjTrans`). `row`/`col` are indices into `op(mat)`, so
/// the caller must ensure `row < op(mat).nrows` and `col < op(mat).ncols`.
#[inline]
fn read_op<T: Scalar>(mat: &MatRef<'_, T>, trans: Trans, row: usize, col: usize) -> T {
    match trans {
        // SAFETY: `row < op(mat).nrows == mat.nrows` and
        // `col < op(mat).ncols == mat.ncols` by the caller's contract.
        Trans::NoTrans => unsafe { *mat.ptr_at(row, col) },
        // SAFETY: for a transposed read the source indices are swapped, so
        // `col < op(mat).ncols == mat.nrows` and `row < op(mat).nrows == mat.ncols`.
        Trans::Trans => unsafe { *mat.ptr_at(col, row) },
        Trans::ConjTrans => unsafe { *mat.ptr_at(col, row) }.conj(),
    }
}

/// Materializes `op(src)` (transpose/conjugate applied) into a fresh owned matrix.
///
/// Used only by the complex transpose-aware wrappers, which route through the
/// 3M complex kernel that already copies operands internally — so this extra
/// copy is not on any zero-copy hot path. The real-typed [`gemm_transposed`]
/// path never calls this: it reads transposed operands in place while packing.
fn materialize_op<T: Scalar + bytemuck::Zeroable>(src: &MatRef<'_, T>, trans: Trans) -> Mat<T> {
    let (rows, cols) = op_dims(src, trans);
    let mut out: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            out[(i, j)] = read_op(src, trans, i, j);
        }
    }
    out
}

/// Transpose-aware naive GEMM for the small-matrix fast path.
///
/// Computes `C = alpha * op(A) * op(B) + beta * C` with an explicit triple loop
/// that reads each operand through [`read_op`]. Only used for the transposed
/// small-matrix case; the `NoTrans`/`NoTrans` case keeps the optimized
/// `gemm_small` kernel.
fn gemm_small_trans<T: Field>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) {
    let m = c.nrows();
    let n = c.ncols();
    let k = op_dims(a, trans_a).1;

    for j in 0..n {
        for i in 0..m {
            let mut acc = T::zero();
            for p in 0..k {
                acc = acc + read_op(a, trans_a, i, p) * read_op(b, trans_b, p, j);
            }
            let scaled = alpha * acc;
            let val = if beta == T::zero() {
                scaled
            } else {
                scaled + beta * c[(i, j)]
            };
            c.set(i, j, val);
        }
    }
}

/// Blocked GEMM implementation with parallelization support.
#[allow(clippy::too_many_arguments)]
fn gemm_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    shape: &MicroKernelShape,
    par: Par,
) {
    // Suppress unused warning when parallel feature is disabled
    let _ = &par;

    #[cfg(feature = "parallel")]
    {
        // Work estimate uses the logical (post-transpose) dimensions.
        let (m, k) = op_dims(a, trans_a);
        let (_, n) = op_dims(b, trans_b);

        // Check if we should use parallelization
        let threshold = ParThreshold::new(64 * 64 * 64, 32 * 32);
        let total_work = m * n * k;

        if threshold.should_parallelize(total_work, par) && total_work >= 2 * PAR_TASK_MIN_WORK {
            gemm_blocked_parallel(trans_a, trans_b, alpha, a, b, beta, c, blocking, shape, par);
            return;
        }
    }

    // Sequential fallback
    gemm_blocked_sequential(trans_a, trans_b, alpha, a, b, beta, c, blocking, shape);
}

/// Packs a panel of `op(A)` into the A packing buffer, transpose-aware.
///
/// For `NoTrans` this is exactly the existing SIMD-friendly `pack_a_optimized`
/// (unchanged fast path). For `Trans`/`ConjTrans` it reads `op(A)[r, c]` with
/// swapped source indices (and conjugation), producing the identical packed
/// layout — the micro-kernel cannot tell the difference. Genuinely zero-copy:
/// no transposed copy of `A` is ever materialized.
///
/// `row_start`/`col_start` and `nrows`/`ncols` are all in `op(A)` coordinates.
#[inline]
#[allow(clippy::too_many_arguments)]
fn pack_a_trans_dispatch<T: Field>(
    a: &MatRef<'_, T>,
    trans_a: Trans,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    mr: usize,
) {
    if trans_a == Trans::NoTrans {
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
        return;
    }

    let dst = pack.as_mut_ptr();
    let mut idx = 0usize;
    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);
        for p in 0..ncols {
            for ii in 0..mr {
                let val = if ii < ib {
                    read_op(a, trans_a, row_start + i + ii, col_start + p)
                } else {
                    T::zero()
                };
                // SAFETY: `idx` walks `[0, ceil(nrows/mr)*mr*ncols)`, which is
                // <= the buffer size the caller allocated for this MR panel.
                unsafe {
                    *dst.add(idx) = val;
                }
                idx += 1;
            }
        }
    }
}

/// Packs a panel of `op(B)` into the B packing buffer, transpose-aware.
///
/// Mirrors [`pack_a_trans_dispatch`]: `NoTrans` uses the existing
/// `pack_b_optimized`; transposed modes read `op(B)[r, c]` with swapped source
/// indices (and conjugation) into the identical packed layout, with no copy.
///
/// `row_start`/`col_start` and `nrows`/`ncols` are all in `op(B)` coordinates.
#[inline]
#[allow(clippy::too_many_arguments)]
fn pack_b_trans_dispatch<T: Field>(
    b: &MatRef<'_, T>,
    trans_b: Trans,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    if trans_b == Trans::NoTrans {
        pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
        return;
    }

    let dst = pack.as_mut_ptr();
    let mut idx = 0usize;
    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);
        for p in 0..nrows {
            for jj in 0..nr {
                let val = if jj < jb {
                    read_op(b, trans_b, row_start + p, col_start + j + jj)
                } else {
                    T::zero()
                };
                // SAFETY: `idx` walks `[0, ceil(ncols/nr)*nr*nrows)`, which is
                // <= the buffer size the caller allocated for this NR panel.
                unsafe {
                    *dst.add(idx) = val;
                }
                idx += 1;
            }
        }
    }
}

/// Sequential blocked GEMM implementation.
#[allow(clippy::too_many_arguments)]
fn gemm_blocked_sequential<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    shape: &MicroKernelShape,
) {
    // Logical (post-transpose) dimensions.
    let (m, k) = op_dims(a, trans_a);
    let (_, n) = op_dims(b, trans_b);

    let nc = blocking.nc.min(n);
    let kc = blocking.kc.min(k);
    let mc = blocking.mc.min(m);

    let mr = shape.mr;
    let nr = shape.nr;

    // Allocate packing buffers with padding for micro-kernel alignment
    let padded_mc = mc.div_ceil(mr) * mr;
    let padded_nc = nc.div_ceil(nr) * nr;
    let mut pack_a: AlignedVec<T> = AlignedVec::zeros(padded_mc * kc);
    let mut pack_b: AlignedVec<T> = AlignedVec::zeros(kc * padded_nc);

    // Track if this is the first k-iteration (need to apply beta)
    let mut first_k = true;

    // Loop over k in blocks of kc
    for p in (0..k).step_by(kc) {
        let pb = kc.min(k - p);

        // Loop over n in blocks of nc
        for j in (0..n).step_by(nc) {
            let jb = nc.min(n - j);

            // Pack op(B) block: op(B)[p:p+pb, j:j+jb] -> pack_b (transpose-aware)
            pack_b_trans_dispatch(b, trans_b, p, j, pb, jb, &mut pack_b, shape.nr);

            // Loop over m in blocks of mc
            for i in (0..m).step_by(mc) {
                let ib = mc.min(m - i);

                // Pack op(A) block: op(A)[i:i+ib, p:p+pb] -> pack_a (transpose-aware)
                pack_a_trans_dispatch(a, trans_a, i, p, ib, pb, &mut pack_a, shape.mr);

                // Compute C[i:i+ib, j:j+jb] += alpha * pack_a * pack_b
                let effective_beta = if first_k { beta } else { T::one() };

                // Get mutable submatrix of C
                let c_sub = c.rb_mut().submatrix(i, j, ib, jb);

                macro_panel_multiply(
                    alpha,
                    &pack_a,
                    ib,
                    pb,
                    &pack_b,
                    pb,
                    jb,
                    effective_beta,
                    c_sub,
                    shape,
                );
            }
        }

        first_k = false;
    }
}

/// Block sizes `(rows, columns)` of the parallel task grid over C.
///
/// Columns are split into chunks of at most `nc`, then further (NR-aligned)
/// until there is about one task per thread (at most one per
/// [`PAR_TASK_MIN_WORK`] multiply-adds); rows are then split into chunks of at
/// most `mc` (MR-aligned) to make up the rest. `m`, `n`, `k`, `mc`, `nc` > 0.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn parallel_grid(
    m: usize,
    n: usize,
    k: usize,
    mc: usize,
    nc: usize,
    shape: &MicroKernelShape,
    threads: usize,
) -> (usize, usize) {
    let (mr, nr) = (shape.mr, shape.nr);
    let tasks = threads.min(m * n * k / PAR_TASK_MIN_WORK).max(1);
    let n_target = n.div_ceil(nc).max(tasks.min(n.div_ceil(nr)));
    let jw = n.div_ceil(n_target).next_multiple_of(nr).min(nc);
    let m_target = tasks.div_ceil(n.div_ceil(jw)).min(m.div_ceil(mr));
    let iw = m.div_ceil(m_target).next_multiple_of(mr).min(mc);
    (iw, jw)
}

/// Parallel blocked GEMM implementation.
///
/// Parallelizes over a 2D grid of C blocks (rows and columns), so that a
/// problem with fewer NC column blocks than threads still uses every thread.
/// Each task packs its own A and B blocks and writes a disjoint block of C.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn gemm_blocked_parallel<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    shape: &MicroKernelShape,
    par: Par,
) {
    // Logical (post-transpose) dimensions.
    let (m, k) = op_dims(a, trans_a);
    let (_, n) = op_dims(b, trans_b);

    let nc = blocking.nc.min(n);
    let kc = blocking.kc.min(k);
    let mc = blocking.mc.min(m);

    let mr = shape.mr;
    let nr = shape.nr;

    let (iw, jw) = parallel_grid(m, n, k, mc, nc, shape, par.num_threads());
    let m_chunks = m.div_ceil(iw);
    let n_chunks = n.div_ceil(jw);

    let padded_iw = iw.div_ceil(mr) * mr;
    let padded_jw = jw.div_ceil(nr) * nr;

    // Get raw pointer to C for thread-safe access
    let c_ptr = c.as_ptr() as usize;
    let c_row_stride = c.row_stride();

    let mut first_k = true;

    // Loop over k in blocks of kc
    for p in (0..k).step_by(kc) {
        let pb = kc.min(k - p);
        let effective_beta = if first_k { beta } else { T::one() };

        (0..m_chunks * n_chunks).into_par_iter().for_each(|t| {
            let i = (t % m_chunks) * iw;
            let j = (t / m_chunks) * jw;
            let ib = iw.min(m - i);
            let jb = jw.min(n - j);

            // Thread-local packing buffers
            let mut pack_a: AlignedVec<T> = AlignedVec::zeros(padded_iw * kc);
            let mut pack_b: AlignedVec<T> = AlignedVec::zeros(kc * padded_jw);

            pack_b_trans_dispatch(b, trans_b, p, j, pb, jb, &mut pack_b, nr);
            pack_a_trans_dispatch(a, trans_a, i, p, ib, pb, &mut pack_a, mr);

            // SAFETY: tasks cover disjoint (i, j) blocks of C inside its bounds,
            // and C outlives the parallel loop.
            let c_sub = unsafe {
                let ptr = c_ptr as *mut T;
                MatMut::new(ptr.add(i + j * c_row_stride), ib, jb, c_row_stride)
            };

            macro_panel_multiply(
                alpha,
                &pack_a,
                ib,
                pb,
                &pack_b,
                pb,
                jb,
                effective_beta,
                c_sub,
                shape,
            );
        });

        first_k = false;
    }
}

/// Software prefetch hint for reading.
#[inline(always)]
#[allow(dead_code)]
unsafe fn prefetch_read_panel<T>(ptr: *const T, len: usize) {
    // Prefetch in cache-line sized chunks (128 bytes for Apple Silicon, 64 for x86)
    #[cfg(target_arch = "aarch64")]
    const CACHE_LINE: usize = 128;
    #[cfg(not(target_arch = "aarch64"))]
    const CACHE_LINE: usize = 64;

    let byte_ptr = ptr.cast::<u8>();
    let bytes = len * std::mem::size_of::<T>();
    let lines = bytes.div_ceil(CACHE_LINE);

    for i in 0..lines.min(8) {
        // Limit to 8 prefetch ops
        #[cfg(target_arch = "aarch64")]
        {
            core::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) byte_ptr.add(i * CACHE_LINE),
                options(nostack, preserves_flags)
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
            core::arch::x86_64::_mm_prefetch(
                byte_ptr.add(i * CACHE_LINE) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
}

/// Multiplies packed panels using micro-kernels.
///
/// This function includes software prefetching to hide memory latency
/// by prefetching the next micro-panel of A while computing the current one.
fn macro_panel_multiply<T: Field + GemmKernel>(
    alpha: T,
    pack_a: &AlignedVec<T>,
    m: usize,
    k: usize,
    pack_b: &AlignedVec<T>,
    _kb: usize,
    n: usize,
    beta: T,
    mut c: MatMut<'_, T>,
    shape: &MicroKernelShape,
) {
    let mr = shape.mr;
    let nr = shape.nr;

    // Number of MR×NR blocks
    let m_blocks = m.div_ceil(mr);
    let n_blocks = n.div_ceil(nr);

    // Size of one A micro-panel
    let a_panel_size = mr * k;

    for jb in 0..n_blocks {
        let j = jb * nr;
        let jn = nr.min(n - j);

        // Pointer to B panel for this column block
        let b_ptr: *const T = &raw const pack_b[jb * nr * k];

        for ib in 0..m_blocks {
            let i = ib * mr;
            let im = mr.min(m - i);

            // Pointers to packed data for this micro-kernel
            let a_ptr: *const T = &raw const pack_a[ib * a_panel_size];

            // Prefetch next A micro-panel while computing current
            if ib + 1 < m_blocks {
                unsafe {
                    let next_a_ptr: *const T = &raw const pack_a[(ib + 1) * a_panel_size];
                    prefetch_read_panel(next_a_ptr, a_panel_size.min(mr * 64));
                }
            }

            // Get C submatrix for this micro-kernel
            let c_row_stride = c.row_stride();

            // Call the micro-kernel
            // For partial blocks at edges, we use a temporary buffer
            if im == mr && jn == nr {
                // Full block - can write directly to C
                unsafe {
                    let c_ptr = c.as_ptr().cast_mut();
                    let c_ptr = c_ptr.add(i + j * c_row_stride);

                    T::micro_kernel(k, alpha, a_ptr, b_ptr, beta, c_ptr, c_row_stride);
                }
            } else {
                // Partial block - use temporary buffer
                let mut temp = [T::zero(); 32 * 32]; // Max MR * NR

                // Copy existing C values if beta != 0
                if beta != T::zero() {
                    for jj in 0..jn {
                        for ii in 0..im {
                            temp[ii + jj * mr] = c[(i + ii, j + jj)];
                        }
                    }
                }

                unsafe {
                    T::micro_kernel(k, alpha, a_ptr, b_ptr, beta, temp.as_mut_ptr(), mr);
                }

                // Copy back to C
                for jj in 0..jn {
                    for ii in 0..im {
                        c.set(i + ii, j + jj, temp[ii + jj * mr]);
                    }
                }
            }
        }
    }
}

/// Scales a matrix by a scalar.
fn scale_matrix<T: Field>(c: &mut MatMut<'_, T>, beta: T) {
    if beta == T::zero() {
        c.fill_zero();
    } else if beta != T::one() {
        c.scale(beta);
    }
}

#[cfg(test)]
mod tests;
