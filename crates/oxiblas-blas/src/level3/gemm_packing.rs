//! Optimized packing strategies for GEMM.
//!
//! This module provides enhanced packing functions with:
//! - Cache-line aware packing
//! - Software prefetching hints
//! - Optimized loop unrolling
//! - Streaming-friendly memory access patterns
//!
//! ## Packing Layout
//!
//! For A panel (MR × KC blocks):
//! ```text
//! A_packed = [ A00 A01 ... A0(KC-1) ]  <- MR elements per column
//!            [ A10 A11 ... A1(KC-1) ]
//!            [ ... ]
//! ```
//!
//! For B panel (KC × NR blocks):
//! ```text
//! B_packed = [ B00 B01 ... B0(NR-1) ]  <- NR elements per row
//!            [ B10 B11 ... B1(NR-1) ]
//!            [ ... ]
//! ```

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Prefetch distance in cache lines.
///
/// Packing has streaming memory access patterns, so we can use a longer
/// prefetch distance to hide memory latency. Apple Silicon's 128-byte
/// cache lines and better memory bandwidth support this.
#[cfg(target_arch = "aarch64")]
const PREFETCH_DISTANCE: usize = 6; // 6 cache lines = 768 bytes

#[cfg(not(target_arch = "aarch64"))]
const PREFETCH_DISTANCE: usize = 4; // 4 cache lines = 256 bytes

/// Packs a panel of A with optimized memory access patterns.
///
/// This version uses 4-way unrolling and prefetching for better performance.
///
/// # Arguments
///
/// * `a` - Source matrix
/// * `row_start` - Starting row in A
/// * `col_start` - Starting column in A
/// * `nrows` - Number of rows to pack
/// * `ncols` - Number of columns to pack
/// * `pack` - Destination packed buffer
/// * `mr` - Micro-kernel row block size
#[inline]
pub fn pack_a_optimized<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    mr: usize,
) {
    let row_stride = a.row_stride();
    let base_ptr = a.as_ptr();
    let dst = pack.as_mut_ptr();

    let mut idx = 0;

    // Pack in blocks of MR rows
    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            // Full block - use optimized 4-way unrolled path
            let src_base = unsafe { base_ptr.add(row_start + i) };

            // Process 4 columns at a time
            let mut p = 0;
            while p + 4 <= ncols {
                // Prefetch next cache lines
                if p + PREFETCH_DISTANCE < ncols {
                    unsafe {
                        let prefetch_ptr =
                            src_base.add((col_start + p + PREFETCH_DISTANCE) * row_stride);
                        prefetch_read(prefetch_ptr);
                    }
                }

                unsafe {
                    let src0 = src_base.add((col_start + p) * row_stride);
                    let src1 = src_base.add((col_start + p + 1) * row_stride);
                    let src2 = src_base.add((col_start + p + 2) * row_stride);
                    let src3 = src_base.add((col_start + p + 3) * row_stride);
                    let dst_ptr = dst.add(idx);

                    // Copy MR elements for each of the 4 columns
                    std::ptr::copy_nonoverlapping(src0, dst_ptr, mr);
                    std::ptr::copy_nonoverlapping(src1, dst_ptr.add(mr), mr);
                    std::ptr::copy_nonoverlapping(src2, dst_ptr.add(2 * mr), mr);
                    std::ptr::copy_nonoverlapping(src3, dst_ptr.add(3 * mr), mr);
                }
                idx += 4 * mr;
                p += 4;
            }

            // Handle remaining columns
            while p < ncols {
                unsafe {
                    let src = src_base.add((col_start + p) * row_stride);
                    std::ptr::copy_nonoverlapping(src, dst.add(idx), mr);
                }
                idx += mr;
                p += 1;
            }
        } else {
            // Partial block - scalar path with zero padding
            for p in 0..ncols {
                for ii in 0..ib {
                    unsafe {
                        *dst.add(idx) =
                            *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    }
                    idx += 1;
                }
                // Pad with zeros
                for _ in ib..mr {
                    unsafe {
                        *dst.add(idx) = T::zero();
                    }
                    idx += 1;
                }
            }
        }
    }
}

/// Packs a panel of B with optimized memory access patterns.
///
/// This version uses 4-way unrolling and cache-aware access for better performance.
///
/// # Arguments
///
/// * `b` - Source matrix
/// * `row_start` - Starting row in B
/// * `col_start` - Starting column in B
/// * `nrows` - Number of rows to pack
/// * `ncols` - Number of columns to pack
/// * `pack` - Destination packed buffer
/// * `nr` - Micro-kernel column block size
#[inline]
pub fn pack_b_optimized<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    let row_stride = b.row_stride();
    let base_ptr = b.as_ptr();
    let dst = pack.as_mut_ptr();

    let mut idx = 0;

    // Pack in blocks of NR columns
    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);

        if jb == nr {
            // Full block - optimized path with 4-way row unrolling
            let col_base = col_start + j;

            // Process 4 rows at a time
            let mut p = 0;
            while p + 4 <= nrows {
                // Prefetch future rows
                if p + PREFETCH_DISTANCE < nrows {
                    unsafe {
                        let prefetch_ptr =
                            base_ptr.add(row_start + p + PREFETCH_DISTANCE + col_base * row_stride);
                        prefetch_read(prefetch_ptr);
                    }
                }

                unsafe {
                    // For each of 4 rows, gather NR elements from consecutive columns
                    for row_offset in 0..4 {
                        let row_base =
                            base_ptr.add(row_start + p + row_offset + col_base * row_stride);
                        let dst_row = dst.add(idx + row_offset * nr);

                        // Gather NR elements from strided columns
                        for jj in 0..nr {
                            *dst_row.add(jj) = *row_base.add(jj * row_stride);
                        }
                    }
                }
                idx += 4 * nr;
                p += 4;
            }

            // Handle remaining rows
            while p < nrows {
                unsafe {
                    let row_base = base_ptr.add(row_start + p + col_base * row_stride);
                    for jj in 0..nr {
                        *dst.add(idx + jj) = *row_base.add(jj * row_stride);
                    }
                }
                idx += nr;
                p += 1;
            }
        } else {
            // Partial block - scalar path with zero padding
            for p in 0..nrows {
                unsafe {
                    let row_base = base_ptr.add(row_start + p + (col_start + j) * row_stride);
                    for jj in 0..jb {
                        *dst.add(idx + jj) = *row_base.add(jj * row_stride);
                    }
                    for jj in jb..nr {
                        *dst.add(idx + jj) = T::zero();
                    }
                }
                idx += nr;
            }
        }
    }
}

/// Packs an A panel, taking a flat-copy fast path when the source is a single
/// fully-contiguous micro-panel.
///
/// This crate stores matrices in column-major order: element `(i, j)` lives at
/// `ptr + i + j * row_stride`, so consecutive rows within a column are always
/// one element apart and `row_stride` is the leading dimension. A matrix is
/// therefore *fully contiguous* (no padding between columns) exactly when
/// `row_stride == nrows` — not `row_stride == ncols`, which the previous check
/// used and which is wrong for any non-square matrix.
///
/// [`pack_a_optimized`] re-tiles the panel into `mr`-row sub-panels, so its
/// packed order coincides with plain column-major order **only** when the whole
/// panel is a single `mr`-row block (`nrows == mr`). For a taller panel a flat
/// `memcpy` would emit the wrong layout, so the fast path additionally requires
/// `nrows == mr`; every other shape delegates to the general path.
#[inline]
pub fn pack_a_contiguous<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    mr: usize,
) {
    // Column-major contiguity: the leading dimension equals the row count only
    // when columns sit back-to-back with no inter-column padding.
    let row_stride = a.row_stride();
    let is_contiguous = row_stride == a.nrows();

    if is_contiguous
        && row_start == 0
        && col_start == 0
        && nrows == a.nrows()
        && ncols == a.ncols()
        && nrows == mr
    {
        // The whole matrix is contiguous AND a single micro-row-block, so the
        // packed layout is exactly column-major and one bulk copy reproduces it.
        let src = a.as_ptr();
        let dst = pack.as_mut_ptr();
        // SAFETY: `row_stride == nrows` means the `nrows * ncols` elements are
        // stored contiguously with no gaps, `src` points at the first of them
        // (row_start == col_start == 0), and `dst` is a distinct AlignedVec the
        // caller sized for the panel; the ranges do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, nrows * ncols);
        }
    } else {
        // General path: correct for any stride, padding, or multi-block panel.
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Packs a B panel using non-temporal (streaming) stores when the CPU supports
/// them, falling back to ordinary cache-allocating stores otherwise.
///
/// The packed B buffer is written once here and then read exactly once by the
/// micro-kernel, which makes it a textbook candidate for non-temporal stores:
/// routing these writes through the cache would evict the packed-A panel and the
/// live C tile that the kernel actually reuses. On AVX hardware we assemble each
/// 32-byte lane and emit it with `_mm256_stream_si256` (`VMOVNTDQ`), bypassing
/// the cache hierarchy; an `sfence` before returning makes those non-temporal
/// writes visible to the subsequent (ordinary) reads.
///
/// The store is issued through the integer lane intrinsic so it is agnostic to
/// `T`'s element type — only the raw bytes matter. The produced layout is
/// byte-for-byte identical to [`pack_b_optimized`] on every path: streaming only
/// changes *how* the bytes reach memory, never their values or their order.
///
/// (256-bit non-temporal stores, rather than 512-bit, are used deliberately: the
/// AVX-512 `_mm512_stream_*` intrinsics only stabilized in Rust 1.89, above this
/// workspace's MSRV, whereas `_mm256_stream_si256` has been stable since 1.27.)
#[inline]
pub fn pack_b_streaming<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // `_mm256_stream_si256` needs a 32-byte-aligned destination and moves a
        // full 32-byte lane at a time. `AlignedVec`'s base is aligned to
        // `DEFAULT_ALIGN` (>= 64 on x86_64), but we still verify at runtime so the
        // SAFETY contract holds for any buffer. Requiring `size_of::<T>()` to
        // divide 32 guarantees every lane boundary lands on a 32-byte-aligned
        // element offset.
        let elem_size = core::mem::size_of::<T>();
        if is_x86_feature_detected!("avx")
            && elem_size != 0
            && 32 % elem_size == 0
            && (pack.as_ptr() as usize) % 32 == 0
        {
            // SAFETY: AVX is present, the destination base is 32-byte aligned, and
            // the element size divides the 32-byte lane, so every streaming-store
            // target is 32-byte aligned and in bounds.
            unsafe {
                pack_b_streaming_avx(b, row_start, col_start, nrows, ncols, pack, nr);
            }
            return;
        }
    }

    // Portable fallback: ordinary temporal stores, identical output.
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

/// AVX (256-bit) non-temporal streaming backend for [`pack_b_streaming`].
///
/// Traverses `B` in the exact linear order [`pack_b_optimized`] writes, buffers
/// a full 32-byte lane, then emits it with `_mm256_stream_si256`. Any trailing
/// partial lane uses ordinary stores; a final `sfence` orders all non-temporal
/// writes before the buffer is read back.
///
/// # Safety
///
/// The caller must guarantee that AVX is available, that `pack`'s base pointer is
/// 32-byte aligned, and that `size_of::<T>()` divides 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pack_b_streaming_avx<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    use std::arch::x86_64::*;

    // Elements per 32-byte non-temporal lane. `size_of::<T>()` divides 32 (caller
    // contract), so this is exact and `1 <= lane <= 32`.
    let lane = 32 / core::mem::size_of::<T>();

    let row_stride = b.row_stride();
    let base_ptr = b.as_ptr();
    let dst = pack.as_mut_ptr();

    // Staging holds one full lane. Sized to the widest possible lane (32, for a
    // hypothetical 1-byte element) so it is always large enough regardless of T;
    // only the first `lane` elements are ever touched per flush.
    let mut staging = [T::zero(); 32];
    let mut buf_len = 0usize; // elements currently staged
    let mut written = 0usize; // elements already streamed to `dst`

    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);
        let col_base = col_start + j;
        for p in 0..nrows {
            let row_base = base_ptr.add(row_start + p + col_base * row_stride);
            for jj in 0..nr {
                // Real element inside the block, zero padding beyond `jb`.
                staging[buf_len] = if jj < jb {
                    *row_base.add(jj * row_stride)
                } else {
                    T::zero()
                };
                buf_len += 1;

                if buf_len == lane {
                    // `written` is a multiple of `lane`, so `written * size_of`
                    // is a multiple of 32 and the store target stays 32-aligned.
                    let lane_vec = _mm256_loadu_si256(staging.as_ptr().cast::<__m256i>());
                    _mm256_stream_si256(dst.add(written).cast::<__m256i>(), lane_vec);
                    written += lane;
                    buf_len = 0;
                }
            }
        }
    }

    // Trailing partial lane (fewer than `lane` elements left): ordinary stores.
    for k in 0..buf_len {
        *dst.add(written + k) = staging[k];
    }

    // Order the non-temporal stores before the packed buffer is read back.
    _mm_sfence();
}

/// Prefetch data for reading.
///
/// Uses architecture-specific prefetch instructions when available.
/// On unsupported platforms, this is a no-op.
#[inline(always)]
#[allow(unused_variables)]
unsafe fn prefetch_read<T>(ptr: *const T) {
    // Use intrinsics when available, otherwise no-op
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    // Note: aarch64 prefetch intrinsics are unstable, so we skip them for now.
    // The compiler and hardware prefetchers generally handle prefetching well
    // on modern ARM processors.
}

/// Packing configuration for different matrix shapes.
#[derive(Debug, Clone, Copy)]
pub struct PackingConfig {
    /// Use streaming stores (for large matrices).
    pub use_streaming: bool,
    /// Prefetch distance in elements.
    pub prefetch_distance: usize,
    /// Whether to use 4-way unrolling.
    pub use_unrolling: bool,
}

impl Default for PackingConfig {
    fn default() -> Self {
        Self {
            use_streaming: false,
            prefetch_distance: PREFETCH_DISTANCE,
            use_unrolling: true,
        }
    }
}

impl PackingConfig {
    /// Create config optimized for large matrices.
    #[must_use]
    pub const fn for_large_matrix() -> Self {
        Self {
            use_streaming: true,
            prefetch_distance: 8,
            use_unrolling: true,
        }
    }

    /// Create config optimized for small matrices.
    #[must_use]
    pub const fn for_small_matrix() -> Self {
        Self {
            use_streaming: false,
            prefetch_distance: 2,
            use_unrolling: false,
        }
    }
}

// =============================================================================
// SIMD-Optimized Packing Functions
// =============================================================================

/// SIMD-optimized `pack_a` for f64.
///
/// This crate's column-major storage keeps every column contiguous — element
/// `(i, j)` is at `ptr + i + j * row_stride`, so successive rows are one element
/// apart. Packing an `mr`-row block therefore copies `mr` *contiguous* elements
/// per column, which the AVX2 fast path vectorizes for **any** `row_stride`. No
/// special "row_stride == 1" layout is required (and indeed `row_stride == 1`
/// never holds for a genuine multi-row column-major matrix); we only need `mr`
/// wide enough for a vector and AVX2 present at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pack_a_simd_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    // A full AVX2 register holds 4 f64; below that the scalar path is faster.
    if mr >= 4 && is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 confirmed available at runtime.
        unsafe {
            pack_a_simd_contiguous_f64(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        // Fall back to optimized scalar path.
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Pack contiguous column data using SIMD for f64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_a_simd_contiguous_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    use std::arch::x86_64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            // Full block - use AVX2 for copies
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using AVX2 (4 doubles at a time)
                let mut j = 0;
                while j + 4 <= mr {
                    let v = _mm256_loadu_pd(src.add(j));
                    _mm256_storeu_pd(dst_ptr.add(j), v);
                    j += 4;
                }

                // Handle remaining elements
                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            // Partial block - scalar with zero padding
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// SIMD-optimized `pack_a` for f32.
///
/// See [`pack_a_simd_f64`] for why any `row_stride` is supported: columns are
/// always contiguous in this crate's column-major storage, so the AVX2 path
/// copies `mr` contiguous elements per column. A full AVX2 register holds 8 f32.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pack_a_simd_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    if mr >= 8 && is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 confirmed available at runtime.
        unsafe {
            pack_a_simd_contiguous_f32(a, row_start, col_start, nrows, ncols, pack, mr);
        }
        return;
    }
    pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
}

/// Pack contiguous column data using SIMD for f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_a_simd_contiguous_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    use std::arch::x86_64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using AVX2 (8 floats at a time)
                let mut j = 0;
                while j + 8 <= mr {
                    let v = _mm256_loadu_ps(src.add(j));
                    _mm256_storeu_ps(dst_ptr.add(j), v);
                    j += 8;
                }

                // Handle remaining elements
                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// SIMD-optimized pack_b for f64 using 8-way unrolling with prefetch.
///
/// This version is optimized for common NR values (4, 6, 8).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn pack_b_simd_f64(
    b: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    nr: usize,
) {
    let row_stride = b.row_stride();
    let base_ptr = b.as_ptr();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);

        if jb == nr {
            let col_base = col_start + j;

            // 8-way unrolled loop for rows
            let mut p = 0;
            while p + 8 <= nrows {
                unsafe {
                    // Prefetch ahead
                    if p + 16 < nrows {
                        let prefetch_ptr = base_ptr.add(row_start + p + 16 + col_base * row_stride);
                        prefetch_read(prefetch_ptr);
                    }

                    // Process 8 rows
                    for row_off in 0..8 {
                        let row_ptr = base_ptr.add(row_start + p + row_off + col_base * row_stride);
                        let dst_row = dst.add(idx + row_off * nr);

                        // Gather NR elements (strided access)
                        for jj in 0..nr {
                            *dst_row.add(jj) = *row_ptr.add(jj * row_stride);
                        }
                    }
                }
                idx += 8 * nr;
                p += 8;
            }

            // Handle remaining rows (4-way unroll)
            while p + 4 <= nrows {
                unsafe {
                    for row_off in 0..4 {
                        let row_ptr = base_ptr.add(row_start + p + row_off + col_base * row_stride);
                        let dst_row = dst.add(idx + row_off * nr);
                        for jj in 0..nr {
                            *dst_row.add(jj) = *row_ptr.add(jj * row_stride);
                        }
                    }
                }
                idx += 4 * nr;
                p += 4;
            }

            // Handle remaining rows
            while p < nrows {
                unsafe {
                    let row_ptr = base_ptr.add(row_start + p + col_base * row_stride);
                    for jj in 0..nr {
                        *dst.add(idx + jj) = *row_ptr.add(jj * row_stride);
                    }
                }
                idx += nr;
                p += 1;
            }
        } else {
            // Partial block with zero padding
            for p in 0..nrows {
                unsafe {
                    let row_ptr = base_ptr.add(row_start + p + (col_start + j) * row_stride);
                    for jj in 0..jb {
                        *dst.add(idx + jj) = *row_ptr.add(jj * row_stride);
                    }
                    for jj in jb..nr {
                        *dst.add(idx + jj) = 0.0;
                    }
                }
                idx += nr;
            }
        }
    }
}

/// SIMD-optimized pack_b for f32 using 8-way unrolling with prefetch.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn pack_b_simd_f32(
    b: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    nr: usize,
) {
    let row_stride = b.row_stride();
    let base_ptr = b.as_ptr();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);

        if jb == nr {
            let col_base = col_start + j;

            // 8-way unrolled loop for rows
            let mut p = 0;
            while p + 8 <= nrows {
                unsafe {
                    if p + 16 < nrows {
                        let prefetch_ptr = base_ptr.add(row_start + p + 16 + col_base * row_stride);
                        prefetch_read(prefetch_ptr);
                    }

                    for row_off in 0..8 {
                        let row_ptr = base_ptr.add(row_start + p + row_off + col_base * row_stride);
                        let dst_row = dst.add(idx + row_off * nr);
                        for jj in 0..nr {
                            *dst_row.add(jj) = *row_ptr.add(jj * row_stride);
                        }
                    }
                }
                idx += 8 * nr;
                p += 8;
            }

            while p + 4 <= nrows {
                unsafe {
                    for row_off in 0..4 {
                        let row_ptr = base_ptr.add(row_start + p + row_off + col_base * row_stride);
                        let dst_row = dst.add(idx + row_off * nr);
                        for jj in 0..nr {
                            *dst_row.add(jj) = *row_ptr.add(jj * row_stride);
                        }
                    }
                }
                idx += 4 * nr;
                p += 4;
            }

            while p < nrows {
                unsafe {
                    let row_ptr = base_ptr.add(row_start + p + col_base * row_stride);
                    for jj in 0..nr {
                        *dst.add(idx + jj) = *row_ptr.add(jj * row_stride);
                    }
                }
                idx += nr;
                p += 1;
            }
        } else {
            for p in 0..nrows {
                unsafe {
                    let row_ptr = base_ptr.add(row_start + p + (col_start + j) * row_stride);
                    for jj in 0..jb {
                        *dst.add(idx + jj) = *row_ptr.add(jj * row_stride);
                    }
                    for jj in jb..nr {
                        *dst.add(idx + jj) = 0.0;
                    }
                }
                idx += nr;
            }
        }
    }
}

// =============================================================================
// NEON-Optimized Packing for ARM (aarch64)
// =============================================================================

/// SIMD-optimized `pack_a` for f64 on ARM NEON.
///
/// Columns are always contiguous in this crate's column-major storage (element
/// `(i, j)` is at `ptr + i + j * row_stride`), so the NEON path copies `mr`
/// contiguous elements per column for **any** `row_stride`. A NEON 128-bit
/// register holds 2 f64.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_a_simd_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    if mr >= 2 {
        // SAFETY: NEON is part of the aarch64 baseline, always available here.
        unsafe {
            pack_a_neon_contiguous_f64(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Pack contiguous column data using NEON for f64.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pack_a_neon_contiguous_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    use std::arch::aarch64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using NEON (2 doubles at a time)
                let mut j = 0;
                while j + 2 <= mr {
                    let v = vld1q_f64(src.add(j));
                    vst1q_f64(dst_ptr.add(j), v);
                    j += 2;
                }

                // Handle remaining
                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// SIMD-optimized `pack_a` for f32 on ARM NEON.
///
/// See [`pack_a_simd_f64`] (aarch64) for why any `row_stride` is supported:
/// columns are always contiguous, so the NEON path copies `mr` contiguous
/// elements per column. A NEON 128-bit register holds 4 f32.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_a_simd_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    if mr >= 4 {
        // SAFETY: NEON is part of the aarch64 baseline, always available here.
        unsafe {
            pack_a_neon_contiguous_f32(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Pack contiguous column data using NEON for f32.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pack_a_neon_contiguous_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    use std::arch::aarch64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using NEON (4 floats at a time)
                let mut j = 0;
                while j + 4 <= mr {
                    let v = vld1q_f32(src.add(j));
                    vst1q_f32(dst_ptr.add(j), v);
                    j += 4;
                }

                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// SIMD-optimized pack_b for f64 on ARM (uses 8-way unrolling).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_b_simd_f64(
    b: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    nr: usize,
) {
    // Use the generic optimized version with 8-way unrolling
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

/// SIMD-optimized pack_b for f32 on ARM.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_b_simd_f32(
    b: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    nr: usize,
) {
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_pack_a_optimized() {
        let a: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
        ]);

        let mr = 2;
        let mut pack: AlignedVec<f64> = AlignedVec::zeros(16);

        pack_a_optimized(&a.as_ref(), 0, 0, 4, 4, &mut pack, mr);

        // Check first block (rows 0-1, all columns)
        // Should be: [1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16]
        assert!((pack[0] - 1.0).abs() < 1e-10);
        assert!((pack[1] - 5.0).abs() < 1e-10);
        assert!((pack[2] - 2.0).abs() < 1e-10);
        assert!((pack[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_pack_b_optimized() {
        let b: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
        ]);

        let nr = 2;
        let mut pack: AlignedVec<f64> = AlignedVec::zeros(16);

        pack_b_optimized(&b.as_ref(), 0, 0, 4, 4, &mut pack, nr);

        // Check that packing produces correct layout
        // First NR columns, then next NR columns
        assert!((pack[0] - 1.0).abs() < 1e-10);
        assert!((pack[1] - 2.0).abs() < 1e-10);
        assert!((pack[2] - 5.0).abs() < 1e-10);
        assert!((pack[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_pack_a_partial_block() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let mr = 4; // Larger than nrows, will need padding
        let mut pack: AlignedVec<f64> = AlignedVec::zeros(12);

        pack_a_optimized(&a.as_ref(), 0, 0, 3, 3, &mut pack, mr);

        // First column: [1, 4, 7, 0] (padded)
        assert!((pack[0] - 1.0).abs() < 1e-10);
        assert!((pack[1] - 4.0).abs() < 1e-10);
        assert!((pack[2] - 7.0).abs() < 1e-10);
        assert!((pack[3] - 0.0).abs() < 1e-10); // padding
    }

    #[test]
    fn test_packing_config() {
        let config = PackingConfig::default();
        assert!(!config.use_streaming);
        assert!(config.use_unrolling);

        let large_config = PackingConfig::for_large_matrix();
        assert!(large_config.use_streaming);
        assert_eq!(large_config.prefetch_distance, 8);

        let small_config = PackingConfig::for_small_matrix();
        assert!(!small_config.use_streaming);
        assert!(!small_config.use_unrolling);
    }

    #[test]
    fn test_pack_b_simd_f64() {
        let b: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
            &[33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
            &[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
            &[49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            &[57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        ]);

        let nr = 4;
        let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(64);
        let mut pack_simd: AlignedVec<f64> = AlignedVec::zeros(64);

        pack_b_optimized(&b.as_ref(), 0, 0, 8, 8, &mut pack_opt, nr);
        pack_b_simd_f64(&b.as_ref(), 0, 0, 8, 8, &mut pack_simd, nr);

        // SIMD version should produce identical results
        for i in 0..64 {
            assert!(
                (pack_opt[i] - pack_simd[i]).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                pack_opt[i],
                pack_simd[i]
            );
        }
    }

    #[test]
    fn test_pack_b_simd_f32() {
        let b: Mat<f32> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
            &[33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
            &[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
            &[49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            &[57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        ]);

        let nr = 4;
        let mut pack_opt: AlignedVec<f32> = AlignedVec::zeros(64);
        let mut pack_simd: AlignedVec<f32> = AlignedVec::zeros(64);

        pack_b_optimized(&b.as_ref(), 0, 0, 8, 8, &mut pack_opt, nr);
        pack_b_simd_f32(&b.as_ref(), 0, 0, 8, 8, &mut pack_simd, nr);

        for i in 0..64 {
            assert!(
                (pack_opt[i] - pack_simd[i]).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                pack_opt[i],
                pack_simd[i]
            );
        }
    }

    #[test]
    fn test_pack_b_simd_large() {
        // Test with a larger matrix to exercise the 8-way unrolling
        let size = 32;
        let data: Vec<Vec<f64>> = (0..size)
            .map(|i| (0..size).map(|j| (i * size + j) as f64).collect())
            .collect();
        let b: Mat<f64> = Mat::from_rows(&data.iter().map(|r| r.as_slice()).collect::<Vec<_>>());

        let nr = 8;
        let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(size * size);
        let mut pack_simd: AlignedVec<f64> = AlignedVec::zeros(size * size);

        pack_b_optimized(&b.as_ref(), 0, 0, size, size, &mut pack_opt, nr);
        pack_b_simd_f64(&b.as_ref(), 0, 0, size, size, &mut pack_simd, nr);

        for i in 0..(size * size) {
            assert!(
                (pack_opt[i] - pack_simd[i]).abs() < 1e-10,
                "Large matrix mismatch at {}: {} vs {}",
                i,
                pack_opt[i],
                pack_simd[i]
            );
        }
    }

    // ------------------------------------------------------------------
    // Regression tests for the packing correctness fixes.
    // ------------------------------------------------------------------

    /// Naive column-major reference for `pack_a`: re-tiles the panel into
    /// `mr`-row blocks with zero padding for the trailing partial block, exactly
    /// matching the layout contract of [`pack_a_optimized`].
    fn pack_a_reference<T: Field>(
        a: &MatRef<'_, T>,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
        mr: usize,
    ) -> Vec<T> {
        let mut out = Vec::new();
        for i in (0..nrows).step_by(mr) {
            let ib = mr.min(nrows - i);
            for p in 0..ncols {
                for ii in 0..mr {
                    if ii < ib {
                        out.push(a[(row_start + i + ii, col_start + p)]);
                    } else {
                        out.push(T::zero());
                    }
                }
            }
        }
        out
    }

    /// Naive reference for `pack_b`: `nr`-column blocks, row-major within a
    /// block, zero padding for the trailing partial block.
    fn pack_b_reference<T: Field>(
        b: &MatRef<'_, T>,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
        nr: usize,
    ) -> Vec<T> {
        let mut out = Vec::new();
        for j in (0..ncols).step_by(nr) {
            let jb = nr.min(ncols - j);
            for p in 0..nrows {
                for jj in 0..nr {
                    if jj < jb {
                        out.push(b[(row_start + p, col_start + j + jj)]);
                    } else {
                        out.push(T::zero());
                    }
                }
            }
        }
        out
    }

    /// Row-major sequential test matrix `[[1,2,..],[..]]` of the given shape.
    fn seq_mat_f64(nrows: usize, ncols: usize) -> Mat<f64> {
        let rows: Vec<Vec<f64>> = (0..nrows)
            .map(|i| (0..ncols).map(|j| (i * ncols + j + 1) as f64).collect())
            .collect();
        Mat::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>())
    }

    fn seq_mat_f32(nrows: usize, ncols: usize) -> Mat<f32> {
        let rows: Vec<Vec<f32>> = (0..nrows)
            .map(|i| (0..ncols).map(|j| (i * ncols + j + 1) as f32).collect())
            .collect();
        Mat::from_rows(&rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>())
    }

    #[test]
    fn test_pack_a_contiguous_matches_reference() {
        // For every shape/`mr` the fast path and the fallback must both agree
        // with the naive reference AND with the general `pack_a_optimized` path.
        //
        // The (8x8, mr=4) case is the direct regression for the original bug:
        // the old flat-copy fast path fired for the whole contiguous matrix
        // regardless of `mr`, emitting plain column-major order instead of the
        // 4-row-tiled layout `pack_a_optimized` produces. The (8x4, mr=8) case
        // covers the wrong `row_stride == ncols` contiguity test on a non-square
        // matrix.
        let cases = [
            (8usize, 8usize, 8usize), // fast-path eligible on x86_64 (rs==nrows==mr)
            (8, 8, 4),                // contiguous but multi-block -> must delegate
            (8, 4, 8),                // non-square: old `== ncols` check was wrong
            (5, 7, 3),                // padded (non-contiguous) -> fallback
            (3, 3, 4),                // partial block with zero padding
        ];

        for (nrows, ncols, mr) in cases {
            let a = seq_mat_f64(nrows, ncols);
            let a_ref = a.as_ref();
            let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
            let total = reference.len();

            let mut pack_contig: AlignedVec<f64> = AlignedVec::zeros(total);
            let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(total);
            pack_a_contiguous(&a_ref, 0, 0, nrows, ncols, &mut pack_contig, mr);
            pack_a_optimized(&a_ref, 0, 0, nrows, ncols, &mut pack_opt, mr);

            for k in 0..total {
                assert!(
                    (pack_contig[k] - reference[k]).abs() < 1e-12,
                    "pack_a_contiguous vs reference mismatch at {k} for {nrows}x{ncols} mr={mr}: {} vs {}",
                    pack_contig[k],
                    reference[k]
                );
                assert!(
                    (pack_contig[k] - pack_opt[k]).abs() < 1e-12,
                    "pack_a_contiguous vs pack_a_optimized mismatch at {k} for {nrows}x{ncols} mr={mr}",
                );
            }
        }
    }

    #[test]
    fn test_pack_a_contiguous_fast_path_layout() {
        // A genuinely contiguous, single-micro-panel matrix (row_stride == nrows,
        // nrows == mr) that takes the flat-copy fast path. 16 rows is a multiple
        // of the f64 cache-line element count under both 64-byte (8) and 128-byte
        // (16) padding, so the matrix is contiguous on every supported target. We
        // assert the precondition so the fast path is actually exercised, then
        // check it reproduces the reference layout.
        let nrows = 16;
        let ncols = 6;
        let a = seq_mat_f64(nrows, ncols);
        let a_ref = a.as_ref();
        assert_eq!(
            a_ref.row_stride(),
            a_ref.nrows(),
            "expected a contiguous {nrows}-row matrix on this target"
        );
        let mr = a_ref.nrows();
        let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
        let mut pack: AlignedVec<f64> = AlignedVec::zeros(reference.len());
        pack_a_contiguous(&a_ref, 0, 0, nrows, ncols, &mut pack, mr);
        for k in 0..reference.len() {
            assert!(
                (pack[k] - reference[k]).abs() < 1e-12,
                "fast-path mismatch at {k}"
            );
        }
    }

    #[test]
    fn test_pack_b_streaming_matches_optimized_f64() {
        // Shapes cover full blocks, zero-padded partial blocks, and total element
        // counts that are and are not multiples of the 4-element (32-byte) AVX
        // lane, exercising both the streaming lanes and the ordinary tail on AVX
        // hardware. Elsewhere this validates the identical fallback.
        let cases = [
            (8usize, 8usize, 4usize),
            (3, 4, 4),  // total 12 -> three full lanes, no tail
            (3, 6, 4),  // partial second block (jb=2, zero-padded), total 24
            (5, 5, 6),  // nr > ncols -> single partial block, total 30 (tail 2)
            (16, 12, 8),
        ];
        for (nrows, ncols, nr) in cases {
            let b = seq_mat_f64(nrows, ncols);
            let b_ref = b.as_ref();
            let reference = pack_b_reference(&b_ref, 0, 0, nrows, ncols, nr);
            let total = reference.len();

            let mut pack_stream: AlignedVec<f64> = AlignedVec::zeros(total);
            let mut pack_opt: AlignedVec<f64> = AlignedVec::zeros(total);
            pack_b_streaming(&b_ref, 0, 0, nrows, ncols, &mut pack_stream, nr);
            pack_b_optimized(&b_ref, 0, 0, nrows, ncols, &mut pack_opt, nr);

            for k in 0..total {
                assert!(
                    (pack_stream[k] - reference[k]).abs() < 1e-12,
                    "streaming vs reference mismatch at {k} for {nrows}x{ncols} nr={nr}: {} vs {}",
                    pack_stream[k],
                    reference[k]
                );
                assert!(
                    (pack_stream[k] - pack_opt[k]).abs() < 1e-12,
                    "streaming vs optimized mismatch at {k} for {nrows}x{ncols} nr={nr}",
                );
            }
        }
    }

    #[test]
    fn test_pack_b_streaming_matches_optimized_f32() {
        // f32 uses an 8-element (32-byte) lane; (5,3,2) yields total 20 = two
        // full lanes plus a 4-element tail.
        let cases = [(4usize, 8usize, 4usize), (7, 5, 4), (5, 3, 2), (20, 16, 8)];
        for (nrows, ncols, nr) in cases {
            let b = seq_mat_f32(nrows, ncols);
            let b_ref = b.as_ref();
            let reference = pack_b_reference(&b_ref, 0, 0, nrows, ncols, nr);
            let total = reference.len();

            let mut pack_stream: AlignedVec<f32> = AlignedVec::zeros(total);
            pack_b_streaming(&b_ref, 0, 0, nrows, ncols, &mut pack_stream, nr);

            for k in 0..total {
                assert!(
                    (pack_stream[k] - reference[k]).abs() < 1e-5,
                    "f32 streaming mismatch at {k} for {nrows}x{ncols} nr={nr}",
                );
            }
        }
    }

    #[test]
    fn test_pack_b_streaming_subregion() {
        // Pack an interior region (row_start/col_start != 0) of a larger matrix,
        // exercising the streaming gather with a non-zero origin and a padded
        // leading dimension.
        let b = seq_mat_f64(10, 10);
        let b_ref = b.as_ref();
        let (row_start, col_start, nrows, ncols, nr) = (2usize, 3usize, 6usize, 5usize, 4usize);
        let reference = pack_b_reference(&b_ref, row_start, col_start, nrows, ncols, nr);
        let total = reference.len();

        let mut pack_stream: AlignedVec<f64> = AlignedVec::zeros(total);
        pack_b_streaming(&b_ref, row_start, col_start, nrows, ncols, &mut pack_stream, nr);

        for k in 0..total {
            assert!(
                (pack_stream[k] - reference[k]).abs() < 1e-12,
                "subregion streaming mismatch at {k}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_a_simd_f64_matches_reference() {
        // With the corrected guard the AVX2 path fires for real column-major
        // storage (any row_stride, including the padded strides these matrices
        // have), so on AVX2 hardware this exercises `pack_a_simd_contiguous_f64`
        // and checks it against the scalar reference.
        let cases = [(8usize, 8usize, 4usize), (16, 5, 8), (10, 7, 4)];
        for (nrows, ncols, mr) in cases {
            let a = seq_mat_f64(nrows, ncols);
            let a_ref = a.as_ref();
            let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
            let mut pack: AlignedVec<f64> = AlignedVec::zeros(reference.len());
            pack_a_simd_f64(&a_ref, 0, 0, nrows, ncols, &mut pack, mr);
            for k in 0..reference.len() {
                assert!(
                    (pack[k] - reference[k]).abs() < 1e-12,
                    "pack_a_simd_f64 mismatch at {k} for {nrows}x{ncols} mr={mr}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_a_simd_f32_matches_reference() {
        let cases = [(16usize, 8usize, 8usize), (20, 5, 8)];
        for (nrows, ncols, mr) in cases {
            let a = seq_mat_f32(nrows, ncols);
            let a_ref = a.as_ref();
            let reference = pack_a_reference(&a_ref, 0, 0, nrows, ncols, mr);
            let mut pack: AlignedVec<f32> = AlignedVec::zeros(reference.len());
            pack_a_simd_f32(&a_ref, 0, 0, nrows, ncols, &mut pack, mr);
            for k in 0..reference.len() {
                assert!(
                    (pack[k] - reference[k]).abs() < 1e-5,
                    "pack_a_simd_f32 mismatch at {k} for {nrows}x{ncols} mr={mr}"
                );
            }
        }
    }
}
