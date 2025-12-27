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

use oxiblas_core::memory::{AlignedVec, CACHE_LINE_SIZE};
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
    let elem_size = std::mem::size_of::<T>();

    // Calculate cache line elements
    let _elems_per_cache_line = CACHE_LINE_SIZE / elem_size;

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

/// Packs A panel with contiguous row storage (for row-major matrices).
///
/// When the source matrix has contiguous rows, we can use more efficient
/// copy operations.
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
    // Check if rows are contiguous (row_stride == number of columns)
    let row_stride = a.row_stride();
    let is_contiguous = row_stride == a.ncols();

    if is_contiguous && row_start == 0 && col_start == 0 && nrows == a.nrows() && ncols == a.ncols()
    {
        // Special case: entire matrix is contiguous
        // Use block copy for maximum efficiency
        let src = a.as_ptr();
        let dst = pack.as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, nrows * ncols);
        }
    } else {
        // Fall back to optimized packing
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Packs B panel with streaming stores for write-through behavior.
///
/// Uses non-temporal stores when available to avoid cache pollution
/// when the packed buffer won't be immediately reused.
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
    // For now, use optimized packing
    // TODO: Add streaming stores for AVX-512 when stabilized
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
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

/// SIMD-optimized pack_a for f64 when data is column-major (row_stride == 1).
///
/// Uses SIMD loads/stores when packing contiguous column data.
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
    let row_stride = a.row_stride();

    // If row_stride == 1, data is contiguous per column - use SIMD
    if row_stride == 1 && mr >= 4 && is_x86_feature_detected!("avx2") {
        // SAFETY: We just checked that AVX2 is available
        unsafe {
            pack_a_simd_contiguous_f64(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        // Fall back to optimized scalar path
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

/// SIMD-optimized pack_a for f32 when data is column-major.
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
    let row_stride = a.row_stride();

    if row_stride == 1 && mr >= 8 {
        // Safety: function is only called on x86_64 with AVX2
        if is_x86_feature_detected!("avx2") {
            unsafe {
                pack_a_simd_contiguous_f32(a, row_start, col_start, nrows, ncols, pack, mr);
            }
            return;
        }
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

/// SIMD-optimized pack_a for f64 on ARM NEON.
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
    let row_stride = a.row_stride();

    if row_stride == 1 && mr >= 2 {
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

/// SIMD-optimized pack_a for f32 on ARM NEON.
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
    let row_stride = a.row_stride();

    if row_stride == 1 && mr >= 4 {
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
}
