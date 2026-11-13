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
//! When the `parallel` feature is enabled and `Par::Rayon` (or
//! `Par::RayonWith`) is used, large GEMMs split every NC block of C into a
//! two-dimensional (M × N) grid of tiles aligned to the micro-kernel grid.
//! Each KC step packs the shared B panel in parallel, then computes all tiles
//! in parallel (one task per tile, about three tiles per thread so work
//! stealing can balance uneven tiles and heterogeneous cores). Every tile
//! owns a disjoint part of C, so no synchronization on C is needed, and the
//! result is bitwise identical to the sequential path. `Par::RayonWith(n)`
//! sizes the tile grid and the parallel gate for `n` threads, but the tiles run
//! on the ambient rayon pool (run the call inside `rayon::ThreadPool::install`
//! to bound the thread count).

use crate::level3::autotune::{
    CacheInfo, KC_BLOCK_UNIT, balanced_block, capped_nc, compute_blocking_adaptive_with,
    get_cache_info,
};
use crate::level3::complex_gemm::{gemm3m_c32, gemm3m_c64};
use crate::level3::gemm_kernel::{GemmKernel, MicroKernel, MicroKernelShape};
use crate::level3::gemm_packing::{PackBuffer, PackSlot, pack_a_optimized, pack_b_optimized};
use crate::level3::gemm_plan::{BCapScope, DEFAULT_POLICY, GemmPlan, GemmRoute, RouteInputs};
use crate::level3::gemm_small::{gemm_small_trans_copy, gemm_small_with};
use crate::level3::gemm_thin::gemm_thin;
use crate::level3::trsm::Trans;
use num_complex::{Complex32, Complex64};
use oxiblas_core::memory::{AlignedVec, StackReq};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

#[cfg(feature = "parallel")]
use crate::level3::gemm_parallel::try_gemm_blocked_parallel;
#[cfg(feature = "parallel")]
use oxiblas_core::parallel::ParThreshold;

/// Work gate of the parallel blocked GEMM, in multiply-adds (`m · n · k`).
///
/// Below it the fork/joins of the 2D driver (shared B pack, then tiles) and
/// the slowest worker cost more than the parallel speedup. The per-thread
/// floor makes wider pools wait for more work: on an Apple M3 (4 P + 4 E
/// cores) the 2D driver overtakes the sequential one from about 112³ (f64) /
/// 128³ (f32) on 4 threads, but only from about 128³ (f64) / 160³ (f32) on 8
/// threads (f32 at 144³ on 8 threads ran at 0.87× of sequential). With
/// `128³` and `80³` per thread the gate opens at 128³ on up to 4 threads and
/// at 160³ on 8.
#[cfg(feature = "parallel")]
pub(super) const PARALLEL_GATE: ParThreshold = ParThreshold::new(128 * 128 * 128, 80 * 80 * 80);

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
    /// Parameters are tuned based on element size and cache hierarchy (`mc` and
    /// `nc` are then rounded down to multiples of the micro-kernel's `MR` and
    /// `NR`, e.g. `nc = (2048 / NR) * NR`):
    /// - f64 (any element of 8 bytes or more): mc=576, kc=448, nc=2048
    ///   (optimized for large L2 caches)
    /// - f32: mc=576, kc=896, nc=2048 (larger KC for smaller elements)
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
    ///
    /// The per-shape values are targets that are then fitted to `m`, `k`, `n`:
    /// a dimension that fits its target is one block of exactly that size, and
    /// a larger one is split into the fewest near-equal MR / NR / 8-aligned
    /// blocks (never a full block plus a sliver).
    #[must_use]
    pub fn asymmetric<T: Field>(m: usize, k: usize, n: usize, shape: &MicroKernelShape) -> Self {
        let elem_size = std::mem::size_of::<T>();

        // Compute aspect ratios
        let max_dim = m.max(k).max(n);
        let min_dim = m.min(k).min(n);

        // Base blocking targets for the rectangular shapes
        let (base_mc, base_kc) = if elem_size >= 8 {
            (512, 256)
        } else {
            (512, 512)
        };
        let base_nc = 2048;

        let targets = if max_dim < 4 * min_dim {
            // Roughly balanced (within 4x): standard blocking
            Self::for_kernel::<T>(shape)
        } else if m > 4 * k && m > 4 * n {
            // Tall-thin matrix: A is tall (m >> k) and C is tall (m >> n)
            // Prioritize MC to process more rows per iteration:
            // increase MC, decrease KC and NC
            Self {
                mc: base_mc * 2,
                kc: (base_kc / 2).max(32),
                nc: base_nc / 2,
            }
        } else if n > 4 * m && n > 4 * k {
            // Short-wide matrix: C is wide (n >> m)
            // Prioritize NC to process more columns per iteration:
            // increase NC, decrease MC
            Self {
                mc: base_mc / 2,
                kc: base_kc,
                nc: base_nc * 2,
            }
        } else if k > 4 * m && k > 4 * n {
            // Inner-product dominated: k >> m, k >> n
            // Increase KC significantly to amortize packing cost
            Self {
                mc: base_mc / 2,
                kc: base_kc * 4,
                nc: base_nc / 2,
            }
        } else if m > 4 * k && n > 4 * k {
            // Panel-panel: m and n are both large, k is small
            // (a k below the KC target becomes a single KC block)
            Self {
                mc: base_mc,
                kc: base_kc,
                nc: base_nc,
            }
        } else {
            // Default to standard blocking
            Self::for_kernel::<T>(shape)
        };

        targets.fit_to_dims(m, k, n, shape)
    }

    /// Fits these block sizes, read as *targets*, to the dimensions of one
    /// GEMM with [`balanced_block`] (units: MR for MC, [`KC_BLOCK_UNIT`] for
    /// KC, NR for NC).
    ///
    /// A dimension that fits its target becomes a single block of exactly that
    /// size; a larger one is split into the fewest near-equal blocks. Rounding
    /// a clamped block down to a multiple of MR/NR instead (e.g. `n = 2050`
    /// with `NC = 2046` gave blocks `[2046, 4]`) adds whole extra passes: every
    /// NC block re-packs all of A. Idempotent, so applying it twice is free.
    fn fit_to_dims(&self, m: usize, k: usize, n: usize, shape: &MicroKernelShape) -> Self {
        Self {
            mc: balanced_block(m, self.mc, shape.mr),
            kc: balanced_block(k, self.kc, KC_BLOCK_UNIT),
            nc: balanced_block(n, self.nc, shape.nr),
        }
    }

    /// Creates custom blocking parameters with alignment to micro-kernel shape.
    ///
    /// `mc`, `kc` and `nc` are block-size *targets*, not exact block sizes.
    /// `mc` and `nc` are rounded down to a multiple of the micro-kernel shape
    /// (`mr` for `mc`, `nr` for `nc`), and every target is then **clamped to a
    /// valid minimum**: `mc >= mr`, `nc >= nr`, and `kc >= 1`. The clamp keeps a
    /// request such as `mc < mr`, `nc < nr`, or `kc == 0` from collapsing to a
    /// zero block size (a `step_by(0)` in the blocked GEMM loops); out-of-range
    /// targets are clamped rather than rejected.
    ///
    /// The GEMM core then fits the targets to each call's dimensions with
    /// `balanced_block` (units: `mr` for MC, 8 for KC, `nr` for NC; see
    /// [`gemm_with_blocking`]). A dimension that fits within its target is
    /// processed as one block of exactly that size and is never split; a larger
    /// one is split into the fewest near-equal blocks.
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
/// Auto-tuning is used for better performance as soon as any dimension (`m`, `k`
/// or `n`) is at least 512; smaller problems use the standard blocking.
pub fn gemm_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
) {
    gemm_transposed_with_par(Trans::NoTrans, Trans::NoTrans, alpha, a, b, beta, c, par);
}

/// Dimension threshold from which [`gemm_with_par`] and
/// [`gemm_transposed_with_par`] use [`GemmBlocking::auto_tuned`].
const AUTO_TUNE_THRESHOLD: usize = 512;

/// The default blocking of an `m × k` by `k × n` product whose packed panels
/// use the micro-kernel shape `shape` (the blocking [`gemm_with_par`] uses).
///
/// KC always comes from the canonical micro-kernel of `T`
/// (`T::micro_kernel_shape()`): the KC partition decides the rounding of
/// every result element, so a plan whose kernel has another shape must keep
/// it. MC and NC are derived for `shape` from that KC. With the canonical
/// shape this is [`GemmBlocking::auto_tuned`] from [`AUTO_TUNE_THRESHOLD`] on
/// and [`GemmBlocking::for_kernel`] below, exactly as in 0.2.3.
///
/// The triangle updates (`tri_gemm`) use it with the canonical shape, so
/// their accumulation order stays the one of GEMM.
pub(super) fn default_blocking<T: GemmKernel>(
    m: usize,
    k: usize,
    n: usize,
    shape: &MicroKernelShape,
) -> GemmBlocking {
    default_blocking_in::<T>(get_cache_info, m, k, n, shape)
}

/// [`default_blocking`] for the cache description `cache` returns (asked
/// for only from [`AUTO_TUNE_THRESHOLD`] on).
///
/// This is where the canonical shape `T::micro_kernel_shape()` is chosen as
/// the source of KC; [`default_blocking`] only supplies the host's caches,
/// so the tests reach the choice with synthetic ones.
fn default_blocking_in<T: GemmKernel>(
    cache: impl FnOnce() -> CacheInfo,
    m: usize,
    k: usize,
    n: usize,
    shape: &MicroKernelShape,
) -> GemmBlocking {
    default_blocking_with::<T>(cache, m, k, n, shape, &T::micro_kernel_shape())
}

/// [`default_blocking_in`] with the canonical shape `canonical` given
/// explicitly.
fn default_blocking_with<T: Field>(
    cache: impl FnOnce() -> CacheInfo,
    m: usize,
    k: usize,
    n: usize,
    shape: &MicroKernelShape,
    canonical: &MicroKernelShape,
) -> GemmBlocking {
    if m >= AUTO_TUNE_THRESHOLD || k >= AUTO_TUNE_THRESHOLD || n >= AUTO_TUNE_THRESHOLD {
        let tuned = compute_blocking_adaptive_with(
            cache,
            m,
            k,
            n,
            core::mem::size_of::<T>(),
            shape,
            canonical,
        );
        GemmBlocking {
            mc: tuned.mc,
            kc: tuned.kc,
            nc: tuned.nc,
        }
    } else {
        // The KC of `for_kernel` depends on the element size only.
        GemmBlocking::for_kernel::<T>(shape)
    }
}

/// Where the block sizes of one GEMM call come from, and with it whether the
/// policy's B-block cap may change them.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BlockingSource {
    /// The default blocking of the plan ([`default_blocking`] for the plan's
    /// micro-kernel). Cap-eligible.
    Default,
    /// Targets computed by this crate's own heuristics
    /// ([`gemm_auto_with_par`], [`gemm_asymmetric_with_par`], the Strassen
    /// leaves, the dispatch of the runtime-tuned GEMM): the caller did not
    /// choose block sizes, so they are cap-eligible.
    Heuristic(GemmBlocking),
    /// Targets chosen by the caller ([`gemm_with_blocking`],
    /// [`gemm_transposed_with_blocking`]). Never capped, whatever the policy
    /// says.
    Explicit(GemmBlocking),
}

/// The block sizes of one blocked GEMM call, fitted to its dimensions, with
/// the B-block cap decided once.
///
/// MC and KC are the same for both drivers; the cap changes NC only, so it
/// never changes a result bit.
#[derive(Debug, Clone, Copy)]
pub(super) struct CallBlocking {
    /// The targets fitted to the dimensions with the plan's kernel shape.
    fitted: GemmBlocking,
    /// `fitted` with the B-block cap applied to NC (`fitted` when `cap` is
    /// [`BCapScope::Off`]).
    capped: GemmBlocking,
    /// Where the cap applies: the policy's scope, or `Off` for explicit
    /// blocking.
    cap: BCapScope,
}

impl CallBlocking {
    /// The blocks of the sequential driver: capped unless the scope is `Off`.
    pub(super) fn sequential(&self) -> &GemmBlocking {
        match self.cap {
            BCapScope::Off => &self.fitted,
            BCapScope::Sequential | BCapScope::All => &self.capped,
        }
    }

    /// The blocks of the 2D parallel driver: capped only for
    /// [`BCapScope::All`].
    #[cfg(feature = "parallel")]
    pub(super) fn parallel(&self) -> &GemmBlocking {
        match self.cap {
            BCapScope::Off | BCapScope::Sequential => &self.fitted,
            BCapScope::All => &self.capped,
        }
    }
}

/// The fitted blocking of an `m × k` by `k × n` call of `plan` whose block
/// sizes come from `source`.
///
/// This is the single place that decides the block sizes and the B-block cap
/// of a blocked call: targets from `source` (the default blocking for the
/// plan's kernel, or the caller's / heuristic targets), fitted with the
/// plan's kernel shape; the cap of `plan`'s policy unless the source is
/// [`BlockingSource::Explicit`]. Every driver of the call (and any backend
/// that must reproduce its KC partition) takes its block sizes from here.
pub(super) fn call_blocking<T: Field + GemmKernel>(
    source: BlockingSource,
    plan: &GemmPlan<T>,
    m: usize,
    k: usize,
    n: usize,
) -> CallBlocking {
    let shape = &plan.kernel.shape;
    let policy = &plan.ctx.policy;
    let (targets, cap) = match source {
        BlockingSource::Default => (default_blocking::<T>(m, k, n, shape), policy.b_cap),
        BlockingSource::Heuristic(targets) => (targets, policy.b_cap),
        BlockingSource::Explicit(targets) => (targets, BCapScope::Off),
    };
    let fitted = targets.fit_to_dims(m, k, n, shape);
    let capped = match cap {
        BCapScope::Off => fitted,
        BCapScope::Sequential | BCapScope::All => GemmBlocking {
            nc: capped_nc(
                m,
                k,
                n,
                fitted.mc,
                fitted.kc,
                fitted.nc,
                shape.nr,
                core::mem::size_of::<T>(),
                &policy.b_caps,
            ),
            ..fitted
        },
    };
    CallBlocking {
        fitted,
        capped,
        cap,
    }
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
    gemm_with_heuristic_blocking(alpha, a, b, beta, c, par, &blocking);
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
    gemm_with_heuristic_blocking(alpha, a, b, beta, c, par, &blocking);
}

/// `C = alpha * A * B + beta * C` with block-size targets that this crate's
/// own heuristics chose ([`BlockingSource::Heuristic`]).
///
/// The targets are fitted exactly as in [`gemm_with_blocking`], but, unlike
/// caller-chosen blocking, the policy's B-block cap may lower NC (which
/// never changes a result bit).
pub(crate) fn gemm_with_heuristic_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
    targets: &GemmBlocking,
) {
    let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, a.nrows(), b.ncols(), a.ncols());
    gemm_planned(
        Trans::NoTrans,
        Trans::NoTrans,
        alpha,
        a,
        b,
        beta,
        c,
        par,
        BlockingSource::Heuristic(*targets),
        &plan,
    );
}

/// GEMM with custom blocking parameters (for benchmarking/tuning).
///
/// This is the no-transpose entry point; it delegates to
/// [`gemm_transposed_with_blocking`] with `Trans::NoTrans` for both operands,
/// so its behavior for existing callers is unchanged.
///
/// The `mc`/`kc`/`nc` values are **targets**, not exact block sizes: each is
/// fitted to the problem so that a dimension no larger than its target is
/// processed as one block of exactly that size, and a larger one is split
/// into the fewest near-equal blocks (multiples of MR for MC, of NR for NC and
/// of 8 for KC). For example `kc = 448` with `k = 500` runs as `[256, 244]`
/// rather than `[448, 52]`.
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
    // One plan for the whole call: the micro-kernel it carries shapes the
    // blocking (MC / NC; KC is the canonical one) and is passed down to the
    // packers and the macro-kernel.
    let (m, k) = op_dims(&a, trans_a);
    let n = op_dims(&b, trans_b).1;
    let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
    gemm_planned(
        trans_a,
        trans_b,
        alpha,
        a,
        b,
        beta,
        c,
        par,
        BlockingSource::Default,
        &plan,
    );
}

/// Transpose-aware GEMM core with explicit blocking parameters.
///
/// All the public GEMM entry points funnel through the same core as this
/// function: each call builds one plan (micro-kernel, SIMD tier and routing
/// policy) and every route of the call uses it. The `NoTrans`/`NoTrans`
/// path uses the same small-matrix fast path, packers and blocked kernel as
/// [`gemm`]; the transpose flags only change how operands are *read* during
/// packing.
///
/// As in [`gemm_with_blocking`], the blocking values are targets that are
/// fitted to the logical dimensions (no dimension that fits a target is
/// split), and no other block-size rule changes them. When `alpha == 0`
/// neither A nor B is read and C becomes `beta * C` (C is not read when
/// `beta == 0`), matching reference BLAS.
#[allow(clippy::too_many_arguments)]
pub fn gemm_transposed_with_blocking<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
    blocking: &GemmBlocking,
) {
    let (m, k) = op_dims(&a, trans_a);
    let n = op_dims(&b, trans_b).1;
    let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k);
    gemm_planned(
        trans_a,
        trans_b,
        alpha,
        a,
        b,
        beta,
        c,
        par,
        BlockingSource::Explicit(*blocking),
        &plan,
    );
}

/// The single internal GEMM funnel: `C = alpha * op(A) * op(B) + beta * C`
/// computed as `plan` says.
///
/// `blocking` says where the block sizes of the blocked route come from
/// ([`BlockingSource`]): the default blocking of the plan's micro-kernel, a
/// heuristic of this crate, or the caller (never capped). Nothing here reads
/// global state other than through `plan` (and the cache description behind
/// the default blocking), so tests and A/B runs can compare plans (policies,
/// forced routes, micro-kernels, pack scopes) on one binary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_planned<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
    blocking: BlockingSource,
    plan: &GemmPlan<T>,
) {
    gemm_core(trans_a, trans_b, alpha, a, b, beta, c, par, blocking, plan);
}

/// The GEMM core behind [`gemm_planned`].
///
/// The order of decisions is fixed: the dimension checks, then the quick
/// returns (`m == 0`, `n == 0`, `k == 0`, `alpha == 0`; none of them reads A
/// or B, and C is not read when `beta == 0`), then the route of the plan.
/// Only the blocked route uses `blocking` ([`call_blocking`] fits it there).
#[allow(clippy::too_many_arguments)]
fn gemm_core<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    par: Par,
    blocking: BlockingSource,
    plan: &GemmPlan<T>,
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

    if alpha == T::zero() {
        // Reference BLAS does not read A or B when alpha == 0: C = beta * C
        // (C is not read either when beta == 0). Multiplying through would
        // turn NaN/Inf in the ignored operands into NaN (0 * NaN = NaN).
        scale_matrix(&mut c, beta);
        return;
    }

    // Everything below reads A and B.

    // (Wave 6: the SME2 hook goes here, ahead of the route; it takes its block
    // sizes from `call_blocking`, so its KC partition is the NEON path's.)

    let inputs = RouteInputs::new(trans_a, trans_b, &a, &b, m, n, k, plan.ctx.tier());
    let route = plan.route(&inputs);
    // m == 1 or n == 1 as one GEMV. `gemm_thin` declines (leaving C as it
    // is) only for operands the router never sends it, which the blocked
    // route below then computes.
    if route == GemmRoute::Thin
        && gemm_thin(plan.ctx, trans_a, trans_b, alpha, &a, &b, beta, &mut c, par)
    {
        return;
    }
    match route {
        // NoTrans/NoTrans without packing (SIMD small kernels where the tier
        // has them).
        GemmRoute::Small => gemm_small_with(plan.ctx, alpha, &a, &b, beta, &mut c),
        // Transposed or strided small problems copied to column-major for
        // the SIMD small kernels (the loops if no copy can be made).
        GemmRoute::SmallTransCopy => {
            if !gemm_small_trans_copy(plan.ctx, trans_a, trans_b, alpha, &a, &b, beta, &mut c) {
                gemm_small_trans(trans_a, trans_b, alpha, &a, &b, beta, &mut c);
            }
        }
        // Transposed small problems: the transpose-aware loops.
        GemmRoute::SmallTrans => {
            gemm_small_trans(trans_a, trans_b, alpha, &a, &b, beta, &mut c);
        }
        // `Thin` reaches this arm only when `gemm_thin` declined above (it
        // returned `false` and left C untouched): the blocked driver then
        // computes the product like any other.
        GemmRoute::Thin | GemmRoute::Blocked => {
            // The blocking targets fitted to this problem: every caller
            // (including hand-built `for_kernel` / `custom` blockings) gets
            // balanced blocks instead of a full block plus a sliver.
            let blocks = call_blocking(blocking, plan, m, k, n);
            gemm_blocked(
                trans_a, trans_b, alpha, &a, &b, beta, &mut c, &blocks, plan, par,
            );
        }
    }
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
pub(super) fn op_dims<T: Scalar>(mat: &MatRef<'_, T>, trans: Trans) -> (usize, usize) {
    match trans {
        Trans::NoTrans => (mat.nrows(), mat.ncols()),
        Trans::Trans | Trans::ConjTrans => (mat.ncols(), mat.nrows()),
    }
}

/// Reads the element `op(mat)[row, col]` for the given transpose mode.
///
/// `NoTrans` reads `mat[row, col]`; `Trans`/`ConjTrans` read `mat[col, row]`
/// (and conjugate for `ConjTrans`). `row`/`col` are indices into `op(mat)`, so
/// the caller must ensure `row < op(mat).nrows` and `col < op(mat).ncols`
/// (checked in debug builds by `MatRef::get_unchecked`).
#[inline]
pub(super) fn read_op<T: Scalar>(mat: &MatRef<'_, T>, trans: Trans, row: usize, col: usize) -> T {
    match trans {
        // SAFETY: `get_unchecked(row, col)` needs `row < mat.nrows()` and
        // `col < mat.ncols()`; the caller guarantees `row < op(mat).nrows ==
        // mat.nrows()` and `col < op(mat).ncols == mat.ncols()` for `NoTrans`.
        Trans::NoTrans => unsafe { *mat.get_unchecked(row, col) },
        // SAFETY: the transposed read swaps the indices: `get_unchecked(col,
        // row)` needs `col < mat.nrows()` and `row < mat.ncols()`, which are
        // the caller's `col < op(mat).ncols == mat.nrows()` and
        // `row < op(mat).nrows == mat.ncols()`.
        Trans::Trans => unsafe { *mat.get_unchecked(col, row) },
        // SAFETY: the same swapped indices and caller bounds as the `Trans`
        // arm.
        Trans::ConjTrans => unsafe { *mat.get_unchecked(col, row) }.conj(),
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

/// Rows of C from which [`gemm_small_trans`] uses the axpy order for
/// `op(A) = A` (real types only).
///
/// The axpy order pays a fixed cost per `(j, p)` pair and then updates `m`
/// elements with vector instructions; the dot order costs per multiply-add.
/// On an Apple M3 (`NoTrans`/`Trans`, `m * n * k <= 32³`) the axpy order took
/// 1.2-1.6x the dot order's time at `m = 1, 3, 5`, and 0.30-0.93x of it for
/// `m >= 8` (f64 and f32; square, wide and tall shapes), except for a tie
/// (+1.7%) at `8 x 64 x 64`.
const SMALL_AXPY_MIN_ROWS: usize = 8;

/// Transpose-aware naive GEMM for the small-matrix fast path.
///
/// Computes `C = alpha * op(A) * op(B) + beta * C` with explicit loops that
/// read B (and A, when transposed) through [`read_op`]. Only used for the
/// transposed small-matrix case; the `NoTrans`/`NoTrans` case keeps the
/// optimized `gemm_small` kernel. The loop order follows the operand whose
/// elements are adjacent in memory:
///
/// - real `op(A) = A` in column-major storage with at least
///   [`SMALL_AXPY_MIN_ROWS`] rows: `C = beta * C`, then
///   `C[:, j] += A[:, p] * (alpha * op(B)[p, j])` over unit-stride columns of
///   A and C (the reference-BLAS order, which vectorizes).
/// - otherwise every `C[i, j]` becomes `alpha * dot + beta * C[i, j]`, where
///   the dot product runs along a row of `op(A)` (a column of the stored A
///   when A is transposed).
///
/// C is not read when `beta == 0`, and neither A nor B when `alpha == 0`.
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

    if alpha == T::zero() {
        // A and B are not read when alpha == 0 (reference BLAS semantics).
        scale_matrix(c, beta);
        return;
    }

    if T::is_real() && trans_a == Trans::NoTrans && a.col_stride() == 1 && m >= SMALL_AXPY_MIN_ROWS
    {
        // C = beta * C first (C is not read when beta == 0).
        scale_matrix(c, beta);
        for j in 0..n {
            for p in 0..k {
                let t = alpha * read_op(b, trans_b, p, j);
                if let (Some(a_col), Some(c_col)) = (a.col_as_slice(p), c.col_as_slice_mut(j)) {
                    for (ci, &ai) in c_col.iter_mut().zip(a_col) {
                        *ci += ai * t;
                    }
                    continue;
                }
                // Not reached (`p < k`, `j < n`, unit row strides); kept so the
                // update never depends on that.
                for i in 0..m {
                    let v = c[(i, j)] + a[(i, p)] * t;
                    c.set(i, j, v);
                }
            }
        }
        return;
    }

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
///
/// `blocking` is fitted to the problem, with the B-block cap decided per
/// driver ([`CallBlocking`]); the cap only changes NC, so not the result.
/// Both drivers run the plan's micro-kernel, and the 2D parallel driver uses
/// the plan's pack scope.
#[allow(clippy::too_many_arguments)]
fn gemm_blocked<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    blocking: &CallBlocking,
    plan: &GemmPlan<T>,
    par: Par,
) {
    let kernel = &plan.kernel;

    #[cfg(feature = "parallel")]
    {
        // Below 128^3 multiply-adds (or 80^3 per thread) the fork/join and
        // shared-packing overhead outweighs the parallel speedup; see
        // `PARALLEL_GATE`.
        let (m, k) = op_dims(a, trans_a);
        let n = op_dims(b, trans_b).1;
        let total_work = m * n * k;

        // The 2D driver declines (returns false) when C is too small to form
        // at least two tiles. The pack scope needs the thread count, which
        // for `Par::Rayon` builds rayon's global pool; it is asked for only
        // once the gate has passed, so a call the gate keeps sequential
        // never starts the pool (as in 0.2.3).
        if PARALLEL_GATE.should_parallelize(total_work, par)
            && try_gemm_blocked_parallel(
                trans_a,
                trans_b,
                alpha,
                a,
                b,
                beta,
                c,
                blocking.parallel(),
                kernel,
                plan.scope(par.num_threads()),
                par,
            )
        {
            return;
        }
    }
    // Without `parallel` the only `Par` is `Par::Seq` (one thread, no pool)
    // and the 2D driver does not exist; the scope is decided all the same.
    #[cfg(not(feature = "parallel"))]
    let _ = plan.scope(par.num_threads());

    // Sequential driver.
    gemm_blocked_sequential(
        trans_a,
        trans_b,
        alpha,
        a,
        b,
        beta,
        c,
        blocking.sequential(),
        kernel,
    );
}

/// Returns whether `op(X)` for an operand in plain column-major storage is the
/// stored matrix transposed without conjugation (`Trans`, or `ConjTrans` on a
/// real type, where conjugation is the identity).
#[inline]
fn is_plain_transpose<T: Field>(trans: Trans) -> bool {
    match trans {
        Trans::NoTrans => false,
        Trans::Trans => true,
        Trans::ConjTrans => T::is_real(),
    }
}

/// Packs a panel of `op(A)` into the A packing buffer, transpose-aware.
///
/// For operands in plain column-major storage (`col_stride == 1`):
/// - `NoTrans` uses [`pack_a_optimized`], which copies `MR` contiguous
///   elements per column.
/// - `Trans` (and `ConjTrans` on real types) uses [`pack_b_optimized`] on the
///   stored matrix: the `MR`-row micro-panels of `op(A) = A^T` are the
///   `MR`-column micro-panels of `A`, in exactly the layout `pack_b_optimized`
///   writes (`MR` elements of one stored row per step, zero-padded).
///
/// Every other operand — `ConjTrans` on complex types, and views whose rows are
/// not adjacent in memory (e.g. a [`MatRef::transpose`] view) — is read
/// element-wise through [`read_op`], which honours both strides (and
/// conjugates for `ConjTrans`). All paths produce the identical packed layout,
/// so the micro-kernel cannot tell the difference. Genuinely zero-copy: no
/// transposed copy of `A` is ever materialized.
///
/// `row_start`/`col_start` and `nrows`/`ncols` are all in `op(A)` coordinates.
///
/// # Panics
///
/// Panics if `pack` holds fewer than `⌈nrows / mr⌉ · mr · ncols` elements.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn pack_a_trans_dispatch<T: Field>(
    a: &MatRef<'_, T>,
    trans_a: Trans,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    mr: usize,
) {
    // Checked in every build: the element-wise fallback below writes through
    // a raw pointer and relies on it.
    assert!(
        pack.len() >= nrows.div_ceil(mr) * mr * ncols,
        "A packing buffer too small"
    );
    if a.col_stride() == 1 {
        if trans_a == Trans::NoTrans {
            pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
            return;
        }
        if is_plain_transpose::<T>(trans_a) {
            // op(A)[row_start + i, col_start + p] = A[col_start + p, row_start + i].
            pack_b_optimized(a, col_start, row_start, ncols, nrows, pack, mr);
            return;
        }
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
                // SAFETY: `idx` walks `[0, ceil(nrows/mr)*mr*ncols)` (one write
                // per `(i, p, ii)`), and the `assert!` at the top of this
                // function guarantees `pack.len() >= ceil(nrows/mr)*mr*ncols`,
                // so `dst.add(idx)` stays inside `pack`, which is exclusively
                // borrowed and does not overlap `a`.
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
/// Mirrors [`pack_a_trans_dispatch`]: for operands with `col_stride == 1`,
/// `NoTrans` uses [`pack_b_optimized`] and `Trans` (and `ConjTrans` on real
/// types) uses [`pack_a_optimized`] on the stored matrix (the `NR`-column
/// micro-panels of `op(B) = B^T` are the `NR`-row micro-panels of `B`). Every
/// other operand (complex `ConjTrans`, strided views) is read through
/// [`read_op`] into the identical packed layout, with no copy.
///
/// `row_start`/`col_start` and `nrows`/`ncols` are all in `op(B)` coordinates.
///
/// # Panics
///
/// Panics if `pack` holds fewer than `⌈ncols / nr⌉ · nr · nrows` elements.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn pack_b_trans_dispatch<T: Field>(
    b: &MatRef<'_, T>,
    trans_b: Trans,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    // Checked in every build: the element-wise fallback below writes through
    // a raw pointer and relies on it.
    assert!(
        pack.len() >= ncols.div_ceil(nr) * nr * nrows,
        "B packing buffer too small"
    );
    if b.col_stride() == 1 {
        if trans_b == Trans::NoTrans {
            pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
            return;
        }
        if is_plain_transpose::<T>(trans_b) {
            // op(B)[row_start + p, col_start + j] = B[col_start + j, row_start + p].
            pack_a_optimized(b, col_start, row_start, ncols, nrows, pack, nr);
            return;
        }
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
                // SAFETY: `idx` walks `[0, ceil(ncols/nr)*nr*nrows)` (one write
                // per `(j, p, jj)`), and the `assert!` at the top of this
                // function guarantees `pack.len() >= ceil(ncols/nr)*nr*nrows`,
                // so `dst.add(idx)` stays inside `pack`, which is exclusively
                // borrowed and does not overlap `b`.
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
    kernel: &MicroKernel<T>,
) {
    // Logical (post-transpose) dimensions.
    let (m, k) = op_dims(a, trans_a);
    let (_, n) = op_dims(b, trans_b);

    let nc = blocking.nc.min(n);
    let kc = blocking.kc.min(k);
    let mc = blocking.mc.min(m);

    let MicroKernelShape { mr, nr } = kernel.shape;

    // Packing buffers (padded to whole micro-panels), on loan from this
    // thread's cache: back-to-back calls reuse them instead of allocating and
    // zero-filling them every time (policy: `PACK_CACHE_MAX_BYTES`).
    let padded_mc = mc.div_ceil(mr) * mr;
    let padded_nc = nc.div_ceil(nr) * nr;
    let mut pack_a = PackBuffer::<T>::take(PackSlot::A, padded_mc * kc);
    let mut pack_b = PackBuffer::<T>::take(PackSlot::B, kc * padded_nc);

    // Track if this is the first k-iteration (need to apply beta)
    let mut first_k = true;

    // Loop over k in blocks of kc
    for p in (0..k).step_by(kc) {
        let pb = kc.min(k - p);

        // Loop over n in blocks of nc
        for j in (0..n).step_by(nc) {
            let jb = nc.min(n - j);

            // Pack op(B) block: op(B)[p:p+pb, j:j+jb] -> pack_b (transpose-aware)
            pack_b_trans_dispatch(b, trans_b, p, j, pb, jb, &mut pack_b, nr);

            // Loop over m in blocks of mc
            for i in (0..m).step_by(mc) {
                let ib = mc.min(m - i);

                // Pack op(A) block: op(A)[i:i+ib, p:p+pb] -> pack_a (transpose-aware)
                pack_a_trans_dispatch(a, trans_a, i, p, ib, pb, &mut pack_a, mr);

                // Compute C[i:i+ib, j:j+jb] += alpha * pack_a * pack_b
                let effective_beta = if first_k { beta } else { T::one() };

                // Get mutable submatrix of C
                let c_sub = c.rb_mut().submatrix(i, j, ib, jb);

                macro_panel_multiply_with(
                    alpha,
                    &pack_a,
                    ib,
                    pb,
                    &pack_b,
                    jb,
                    effective_beta,
                    c_sub,
                    kernel,
                );
            }
        }

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

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
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

    // No prefetch instruction is emitted on other architectures.
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (ptr, len);
    }
}

/// Largest `MR × NR` of the micro-kernels in this crate (AVX-512 f32 is
/// 16 × 16); edge tiles of larger kernels use a heap scratch tile.
const MAX_EDGE_TILE: usize = 16 * 16;

/// Multiplies packed panels using micro-kernels.
///
/// Computes `C = alpha * A~ * B~ + beta * C` for an `m x n` block of C, where
/// `pack_a` holds `ceil(m / MR)` packed `MR x k` micro-panels and `pack_b`
/// holds `ceil(n / NR)` packed `k x NR` micro-panels (both zero-padded). C is
/// not read when `beta == 0`.
///
/// `kernel` is the micro-kernel of the call's plan, and the panels must have
/// been packed with its shape. This is the single seam between the drivers
/// and the micro-kernels: the sequential blocked loop and every tile of the
/// parallel driver call it with the same kernel, so no driver re-dispatches.
///
/// Full `MR × NR` tiles are computed in place in C. Partial (edge) tiles go
/// through one `MR × NR` scratch tile per call (leading dimension `MR`): C is
/// copied in when `beta != 0`, the kernel runs on the scratch tile with the
/// same `alpha`/`beta` as a full tile (so edge and interior tiles round alike),
/// and only the live `im × jn` corner is copied back. Entries of the scratch
/// tile outside that corner may hold values of an earlier edge tile; the kernel
/// may read them (when `beta != 0`), but its results there are discarded.
///
/// On aarch64 and x86_64 it prefetches (the first cache lines of) the next
/// micro-panel of A while computing the current one; other targets get no
/// prefetch.
///
/// # Panics
///
/// Panics if the packed panels are shorter than the block needs.
#[allow(clippy::too_many_arguments)]
pub(super) fn macro_panel_multiply_with<T: Field + GemmKernel>(
    alpha: T,
    pack_a: &[T],
    m: usize,
    k: usize,
    pack_b: &[T],
    n: usize,
    beta: T,
    mut c: MatMut<'_, T>,
    kernel: &MicroKernel<T>,
) {
    let MicroKernelShape { mr, nr } = kernel.shape;
    let micro = kernel.kernel;

    // Number of MR×NR blocks
    let m_blocks = m.div_ceil(mr);
    let n_blocks = n.div_ceil(nr);

    // Size of one A micro-panel
    let a_panel_size = mr * k;

    // Every micro-panel the kernel reads must lie inside the packed buffers.
    assert!(
        pack_a.len() >= m_blocks * a_panel_size && pack_b.len() >= n_blocks * nr * k,
        "packed panels shorter than the {m} x {n} x {k} block"
    );

    let ldc = c.row_stride();
    let c_base = c.as_mut_ptr();

    // Scratch tile for the edge tiles, created on first use.
    let tile_len = mr * nr;
    let mut edge_stack: Option<[T; MAX_EDGE_TILE]> = None;
    let mut edge_heap: Vec<T> = Vec::new();
    let read_c = beta != T::zero();

    for jb in 0..n_blocks {
        let j = jb * nr;
        let jn = nr.min(n - j);

        // B micro-panel for this column block (inside `pack_b` by the assert).
        let b_ptr = pack_b.as_ptr().wrapping_add(jb * nr * k);

        for ib in 0..m_blocks {
            let i = ib * mr;
            let im = mr.min(m - i);

            // A micro-panel for this row block (inside `pack_a` by the assert).
            let a_ptr = pack_a.as_ptr().wrapping_add(ib * a_panel_size);

            // Prefetch next A micro-panel while computing current
            if ib + 1 < m_blocks {
                // SAFETY: by the assert above the next micro-panel lies inside
                // `pack_a`; prefetching is only a cache hint and never
                // dereferences memory architecturally.
                unsafe {
                    prefetch_read_panel(a_ptr.add(a_panel_size), a_panel_size.min(mr * 64));
                }
            }

            if im == mr && jn == nr {
                // Full block - can write directly to C
                // SAFETY: the tile `C[i..i+MR, j..j+NR]` lies inside the `m x n`
                // view `c` (full block), whose exclusive borrow we hold, at
                // offset `i + j * ldc` with leading dimension `ldc`; the packed
                // panels hold `MR * k` / `k * NR` elements at `a_ptr` / `b_ptr`
                // (assert above), and `kernel` has the shape they were packed
                // with.
                unsafe {
                    micro(k, alpha, a_ptr, b_ptr, beta, c_base.add(i + j * ldc), ldc);
                }
            } else {
                // Partial block - use the scratch tile
                let temp: &mut [T] = if tile_len <= MAX_EDGE_TILE {
                    &mut edge_stack.get_or_insert([T::zero(); MAX_EDGE_TILE])[..tile_len]
                } else {
                    if edge_heap.len() < tile_len {
                        edge_heap = vec![T::zero(); tile_len];
                    }
                    &mut edge_heap[..tile_len]
                };

                // SAFETY: `C[i..i+im, j..j+jn]` lies inside the `m x n` view `c`
                // (exclusively borrowed), so every `c_tile.add(ii + jj * ldc)`
                // with `ii < im`, `jj < jn` is in bounds; `temp` holds
                // `MR * NR` initialized elements with leading dimension `mr`,
                // which is what the kernel reads and writes; the packed panels
                // are as for the full block (zero-padded past `im` / `jn`).
                unsafe {
                    let c_tile = c_base.add(i + j * ldc);
                    if read_c {
                        for jj in 0..jn {
                            for ii in 0..im {
                                temp[ii + jj * mr] = *c_tile.add(ii + jj * ldc);
                            }
                        }
                    }
                    micro(k, alpha, a_ptr, b_ptr, beta, temp.as_mut_ptr(), mr);
                    for jj in 0..jn {
                        for ii in 0..im {
                            *c_tile.add(ii + jj * ldc) = temp[ii + jj * mr];
                        }
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
#[path = "gemm_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "gemm_reuse_tests.rs"]
mod reuse_tests;

#[cfg(test)]
#[path = "gemm_blocking_tests.rs"]
mod blocking_tests;
