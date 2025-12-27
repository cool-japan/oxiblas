//! General Matrix Multiplication (GEMM).
//!
//! Computes C = alpha * A * B + beta * C
//!
//! This implementation uses a BLIS-style blocked algorithm with
//! SIMD-optimized micro-kernels for high performance.
//!
//! ## Parallelization
//!
//! When the `parallel` feature is enabled and `Par::Rayon` is used,
//! GEMM operations are parallelized over the outer loop (columns of C).
//! This provides good work distribution without requiring synchronization.

use crate::level3::gemm_kernel::{GemmKernel, MicroKernelShape};
use crate::level3::gemm_packing::{pack_a_optimized, pack_b_optimized};
use crate::level3::gemm_small::{SMALL_THRESHOLD, gemm_small};
use oxiblas_core::memory::{AlignedVec, StackReq};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::ParThreshold;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    #[must_use]
    pub const fn custom(mc: usize, kc: usize, nc: usize, shape: &MicroKernelShape) -> Self {
        let mr = shape.mr;
        let nr = shape.nr;

        Self {
            nc: (nc / nr) * nr,
            kc,
            mc: (mc / mr) * mr,
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
pub fn gemm_with_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    par: Par,
    blocking: &GemmBlocking,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // Dimension checks
    assert_eq!(a.ncols(), b.nrows(), "A.ncols must equal B.nrows");
    assert_eq!(c.nrows(), m, "C.nrows must equal A.nrows");
    assert_eq!(c.ncols(), n, "C.ncols must equal B.ncols");

    // Handle trivial cases
    if m == 0 || n == 0 {
        return;
    }

    if k == 0 {
        // C = beta * C
        scale_matrix(&mut c, beta);
        return;
    }

    // Get micro-kernel shape for this type
    let shape = T::micro_kernel_shape();

    // Small matrix fast path using specialized kernels
    if m * n * k <= SMALL_THRESHOLD {
        gemm_small(alpha, &a, &b, beta, &mut c);
        return;
    }

    // Use blocked GEMM
    gemm_blocked(alpha, &a, &b, beta, &mut c, blocking, &shape, par);
}

/// Blocked GEMM implementation with parallelization support.
fn gemm_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
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
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        // Check if we should use parallelization
        let threshold = ParThreshold::new(64 * 64 * 64, 32 * 32);
        let total_work = m * n * k;

        if threshold.should_parallelize(total_work, par) {
            gemm_blocked_parallel(alpha, a, b, beta, c, blocking, shape, par);
            return;
        }
    }

    // Sequential fallback
    gemm_blocked_sequential(alpha, a, b, beta, c, blocking, shape);
}

/// Sequential blocked GEMM implementation.
fn gemm_blocked_sequential<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    shape: &MicroKernelShape,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

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

            // Pack B block: B[p:p+pb, j:j+jb] -> pack_b (optimized with 4-way unrolling)
            pack_b_optimized(b, p, j, pb, jb, &mut pack_b, shape.nr);

            // Loop over m in blocks of mc
            for i in (0..m).step_by(mc) {
                let ib = mc.min(m - i);

                // Pack A block: A[i:i+ib, p:p+pb] -> pack_a (optimized with 4-way unrolling)
                pack_a_optimized(a, i, p, ib, pb, &mut pack_a, shape.mr);

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

/// Parallel blocked GEMM implementation.
///
/// Parallelizes over columns of C (the n dimension), which provides
/// good work distribution without requiring synchronization for writes.
#[cfg(feature = "parallel")]
fn gemm_blocked_parallel<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &GemmBlocking,
    shape: &MicroKernelShape,
    par: Par,
) {
    use oxiblas_core::parallel::partition_work;
    use std::sync::atomic::{AtomicBool, Ordering};

    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let nc = blocking.nc.min(n);
    let kc = blocking.kc.min(k);
    let mc = blocking.mc.min(m);

    let mr = shape.mr;
    let nr = shape.nr;

    // Number of column blocks
    let n_blocks = (n + nc - 1) / nc;
    let num_threads = par.num_threads().min(n_blocks);

    // Partition column blocks among threads
    let work_ranges = partition_work(n_blocks, num_threads);

    // Track if this is the first k-iteration (need to apply beta)
    let first_k = AtomicBool::new(true);

    // Loop over k in blocks of kc
    for p in (0..k).step_by(kc) {
        let pb = kc.min(k - p);
        let is_first_k = first_k.load(Ordering::SeqCst);

        // Get raw pointer to C for thread-safe access
        let c_ptr = c.as_ptr() as usize;
        let c_row_stride = c.row_stride();

        // Each thread processes a set of column blocks
        work_ranges.par_iter().for_each(|range| {
            // Thread-local packing buffers
            let padded_mc = ((mc + mr - 1) / mr) * mr;
            let padded_nc = ((nc + nr - 1) / nr) * nr;
            let mut pack_a: AlignedVec<T> = AlignedVec::zeros(padded_mc * kc);
            let mut pack_b: AlignedVec<T> = AlignedVec::zeros(kc * padded_nc);

            // Process column blocks assigned to this thread
            for block_idx in range.start..range.end {
                let j = block_idx * nc;
                let jb = nc.min(n - j);

                // Pack B block (optimized with 4-way unrolling and prefetching)
                pack_b_optimized(b, p, j, pb, jb, &mut pack_b, nr);

                // Loop over m in blocks of mc
                for i in (0..m).step_by(mc) {
                    let ib = mc.min(m - i);

                    // Pack A block (optimized with 4-way unrolling and prefetching)
                    pack_a_optimized(a, i, p, ib, pb, &mut pack_a, mr);

                    // Compute C[i:i+ib, j:j+jb] += alpha * pack_a * pack_b
                    let effective_beta = if is_first_k { beta } else { T::one() };

                    // Create a view into C for this submatrix
                    // SAFETY: Each thread writes to non-overlapping columns of C
                    let c_sub = unsafe {
                        let ptr = c_ptr as *mut T;
                        let offset = i + j * c_row_stride;
                        let submat_ptr = ptr.add(offset);
                        MatMut::new(submat_ptr, ib, jb, c_row_stride)
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
                }
            }
        });

        first_k.store(false, Ordering::SeqCst);
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
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_gemm_identity() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::eye(2);
        let mut c: Mat<f64> = Mat::zeros(2, 2);

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_small() {
        // A = [1 2 3]    B = [1 4]
        //     [4 5 6]        [2 5]
        //                    [3 6]
        // A * B = [1*1+2*2+3*3  1*4+2*5+3*6] = [14 32]
        //         [4*1+5*2+6*3  4*4+5*5+6*6]   [32 77]

        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]]);
        let mut c: Mat<f64> = Mat::zeros(2, 2);

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        assert!((c[(0, 0)] - 14.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 32.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 32.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 77.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        let a: Mat<f64> = Mat::filled(2, 2, 1.0);
        let b: Mat<f64> = Mat::filled(2, 2, 2.0);
        let mut c: Mat<f64> = Mat::filled(2, 2, 10.0);

        // C = 2 * A * B + 3 * C
        // A * B = [4 4; 4 4] (each element is 1*2 + 1*2)
        // Result = 2 * [4 4; 4 4] + 3 * [10 10; 10 10] = [8 8; 8 8] + [30 30; 30 30] = [38 38; 38 38]
        gemm(2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (c[(i, j)] - 38.0).abs() < 1e-10,
                    "c[{},{}] = {}",
                    i,
                    j,
                    c[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_gemm_larger() {
        let n = 64;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Each element should be n (sum of n ones)
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - n as f64).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    n
                );
            }
        }
    }

    #[test]
    fn test_gemm_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[5.0f32, 6.0], &[7.0, 8.0]]);
        let mut c: Mat<f32> = Mat::zeros(2, 2);

        // A * B = [1*5+2*7  1*6+2*8] = [19 22]
        //         [3*5+4*7  3*6+4*8]   [43 50]

        gemm(1.0f32, a.as_ref(), b.as_ref(), 0.0f32, c.as_mut());

        assert!((c[(0, 0)] - 19.0).abs() < 1e-5);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-5);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-5);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_parallel() {
        // Test with a larger matrix to trigger parallel execution
        let n = 256;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        #[cfg(feature = "parallel")]
        {
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
        }
        #[cfg(not(feature = "parallel"))]
        {
            gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        }

        // Each element should be n (sum of n ones)
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - n as f64).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    n
                );
            }
        }
    }

    #[test]
    fn test_asymmetric_blocking_parameters() {
        use crate::level3::gemm_kernel::MicroKernelShape;

        let shape = MicroKernelShape { mr: 8, nr: 6 };

        // Test balanced matrix (should use standard blocking)
        let blocking = GemmBlocking::asymmetric::<f64>(100, 100, 100, &shape);
        assert!(blocking.mc > 0);
        assert!(blocking.kc > 0);
        assert!(blocking.nc > 0);

        // Test tall-thin matrix (m >> k, m >> n)
        let blocking_tall = GemmBlocking::asymmetric::<f64>(1000, 10, 20, &shape);
        // Should have larger MC for tall matrices
        assert!(blocking_tall.mc >= 8); // At least mr

        // Test short-wide matrix (n >> m, n >> k)
        let blocking_wide = GemmBlocking::asymmetric::<f64>(20, 10, 1000, &shape);
        // Should have larger NC for wide matrices
        assert!(blocking_wide.nc >= 6); // At least nr

        // Test inner-product dominated (k >> m, k >> n)
        let blocking_inner = GemmBlocking::asymmetric::<f64>(20, 1000, 20, &shape);
        // Should have larger KC for inner-product
        assert!(blocking_inner.kc >= 128);
    }

    #[test]
    fn test_gemm_asymmetric_tall_thin() {
        // Tall-thin: A is 500x10, B is 10x50
        let m = 500;
        let k = 10;
        let n = 50;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Each element should be k * 1.0 * 2.0 = 20.0
        let expected = k as f64 * 2.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gemm_asymmetric_short_wide() {
        // Short-wide: A is 20x10, B is 10x500
        let m = 20;
        let k = 10;
        let n = 500;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        let expected = k as f64 * 2.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gemm_asymmetric_inner_product() {
        // Inner-product dominated: A is 20x500, B is 500x20
        let m = 20;
        let k = 500;
        let n = 20;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        let expected = k as f64 * 2.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gemm_asymmetric_panel_panel() {
        // Panel-panel: A is 200x10, B is 10x200 (m and n large, k small)
        let m = 200;
        let k = 10;
        let n = 200;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        let expected = k as f64 * 2.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gemm_asymmetric_with_alpha_beta() {
        let m = 100;
        let k = 10;
        let n = 200;

        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::filled(m, n, 5.0);

        // C = 2 * A * B + 3 * C
        // A * B each element = 10 * 2 = 20
        // Result = 2 * 20 + 3 * 5 = 40 + 15 = 55
        gemm_asymmetric(2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());

        let expected = 2.0 * (k as f64 * 2.0) + 3.0 * 5.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }
}
