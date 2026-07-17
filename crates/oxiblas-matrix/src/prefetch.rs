//! Prefetch utilities for improved cache performance.
//!
//! This module provides matrix-shaped prefetch helpers ([`prefetch_column`],
//! [`prefetch_block`], [`MatrixPrefetcher`]) built on top of the primitive
//! prefetch intrinsics and cache-line-size constant defined in
//! `oxiblas-core` ([`oxiblas_core::memory`]).
//!
//! The primitives ([`PrefetchLocality`], [`prefetch_read`],
//! [`prefetch_write`], [`CACHE_LINE_SIZE`], [`prefetch_range_read`],
//! [`prefetch_range_write`]) are re-exported/delegated here rather than
//! reimplemented, so that this crate can never disagree with
//! `oxiblas-core` about what a cache line is: `CACHE_LINE_SIZE` is
//! architecture-dependent in `oxiblas-core` (128 bytes on Apple
//! Silicon/aarch64, 64 bytes on x86_64 and elsewhere), not a hardcoded 64.
//!
//! # Cache Hierarchy
//!
//! Modern CPUs have multiple cache levels:
//! - L1 (fastest, smallest, ~32KB per core)
//! - L2 (fast, medium, ~256KB-1MB per core)
//! - L3 (slower, shared, ~8-32MB)
//!
//! # Usage
//!
//! Prefetching is most effective when:
//! - Processing large matrices that don't fit in cache
//! - Access patterns are predictable (sequential or strided)
//! - There's enough distance between prefetch and use
//!
//! # Status
//!
//! The matrix-shaped helpers in this module ([`prefetch_column`],
//! [`prefetch_block`], [`MatrixPrefetcher`]) are available for downstream
//! users of `oxiblas-matrix`, but they are **not** currently invoked by this
//! crate's own kernels. The GEMM/packing kernels in `oxiblas-blas` call
//! `oxiblas-core`'s prefetch primitives directly instead of going through
//! this module.
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_matrix::prefetch::{prefetch_read, PrefetchLocality, PREFETCH_DISTANCE_LINES};
//!
//! // Prefetch data for upcoming reads
//! for i in (0..n).step_by(64 / size_of::<f64>()) {
//!     prefetch_read(&data[i + PREFETCH_DISTANCE_LINES], PrefetchLocality::Medium);
//! }
//! ```

// The primitive prefetch intrinsics and the cache-line-size constant live in
// oxiblas-core. Re-export them here instead of duplicating the unsafe,
// architecture-specific code: duplicating it risks the two crates silently
// drifting apart on cache-line size (oxiblas-core already accounts for
// aarch64's 128-byte lines; a hardcoded local `64` would be wrong there).
pub use oxiblas_core::memory::{CACHE_LINE_SIZE, PrefetchLocality, prefetch_read, prefetch_write};

/// Suggested prefetch distance in cache lines for sequential access.
///
/// This is the number of cache lines ahead to prefetch. The optimal value
/// depends on memory latency and processing speed.
pub const PREFETCH_DISTANCE_LINES: usize = 8;

/// Suggested prefetch distance in bytes for sequential access.
pub const PREFETCH_DISTANCE_BYTES: usize = PREFETCH_DISTANCE_LINES * CACHE_LINE_SIZE;

/// Prefetch a range of memory for reading.
///
/// Prefetches cache lines covering the range `[ptr, ptr + len)`.
/// Useful for preparing a contiguous block of data.
///
/// Delegates to `oxiblas_core::memory::prefetch_read_range`.
#[inline]
pub fn prefetch_range_read<T>(ptr: *const T, len: usize, locality: PrefetchLocality) {
    oxiblas_core::memory::prefetch_read_range(ptr, len, locality);
}

/// Prefetch a range of memory for writing.
///
/// Delegates to `oxiblas_core::memory::prefetch_write_range`.
#[inline]
pub fn prefetch_range_write<T>(ptr: *mut T, len: usize, locality: PrefetchLocality) {
    oxiblas_core::memory::prefetch_write_range(ptr, len, locality);
}

/// Row indices that must be individually prefetched for a strided column
/// access where consecutive rows do **not** share a cache line (i.e.
/// `row_stride * size_of::<T>() > CACHE_LINE_SIZE`).
///
/// Every such row lands on a distinct cache line, so there is no way to
/// "skip" rows without leaving cache lines unprefetched -- unlike the
/// contiguous case, one prefetch instruction cannot cover several rows at
/// once here. This is split out as its own function (rather than inlined
/// into [`prefetch_column`]) so the coverage math can be regression-tested
/// directly: a previous version of this loop computed
/// `row = i * (CACHE_LINE_SIZE / elem_size)` while only iterating `i` up to
/// `ceil(nrows * elem_size / CACHE_LINE_SIZE)`, which for `f64`
/// (`elem_size == 8`, `CACHE_LINE_SIZE == 64`) only ever visited about 1/8
/// of the rows in the column.
#[inline]
fn strided_prefetch_row_indices(nrows: usize) -> impl Iterator<Item = usize> {
    0..nrows
}

/// Prefetch a column of a matrix for reading.
///
/// For column-major storage, this prefetches contiguous memory.
/// For row-major or strided access, this prefetches with the given stride.
// Takes a raw pointer for strided address computation but only issues prefetch
// hints (no actual dereference); scoped allow replaces the former crate-wide one.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[inline]
pub fn prefetch_column<T>(
    ptr: *const T,
    nrows: usize,
    row_stride: usize,
    locality: PrefetchLocality,
) {
    let elem_size = core::mem::size_of::<T>();

    // If contiguous (row_stride == 1), or the stride is small enough that
    // consecutive rows still fall within (or before) one cache line, the
    // whole column can be prefetched as a single contiguous range.
    if row_stride == 1 || (row_stride * elem_size) <= CACHE_LINE_SIZE {
        prefetch_range_read(ptr, nrows, locality);
    } else {
        // Strided access wider than a cache line: every row needs its own
        // prefetch (see `strided_prefetch_row_indices`).
        for row in strided_prefetch_row_indices(nrows) {
            let addr = unsafe { ptr.add(row * row_stride) };
            prefetch_read(addr, locality);
        }
    }
}

/// Prefetch a block of a matrix for reading.
///
/// Prefetches a rectangular block starting at `ptr` with dimensions
/// `block_rows × block_cols`.
// Raw pointer used only for strided address computation feeding prefetch hints.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[inline]
pub fn prefetch_block<T>(
    ptr: *const T,
    block_rows: usize,
    block_cols: usize,
    row_stride: usize,
    locality: PrefetchLocality,
) {
    for j in 0..block_cols {
        let col_ptr = unsafe { ptr.add(j * row_stride) };
        prefetch_column(col_ptr, block_rows, 1, locality);
    }
}

/// Prefetch hint for matrix operations.
///
/// This struct provides a convenient interface for prefetching during
/// matrix operations with predictable access patterns.
pub struct MatrixPrefetcher<T> {
    /// Base pointer.
    ptr: *const T,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Row stride.
    row_stride: usize,
    /// Current prefetch column.
    current_col: usize,
    /// Prefetch distance in columns.
    distance: usize,
    /// Locality hint.
    locality: PrefetchLocality,
}

impl<T> MatrixPrefetcher<T> {
    /// Creates a new matrix prefetcher.
    ///
    /// # Parameters
    /// - `ptr`: Pointer to matrix data
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns
    /// - `row_stride`: Stride between rows (leading dimension)
    /// - `distance`: Number of columns to prefetch ahead
    /// - `locality`: Cache locality hint
    // Raw pointer used only for strided address computation feeding prefetch hints.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn new(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        distance: usize,
        locality: PrefetchLocality,
    ) -> Self {
        let prefetcher = MatrixPrefetcher {
            ptr,
            nrows,
            ncols,
            row_stride,
            current_col: 0,
            distance,
            locality,
        };

        // Prefetch initial columns
        for j in 0..distance.min(ncols) {
            let col_ptr = unsafe { ptr.add(j * row_stride) };
            prefetch_column(col_ptr, nrows, 1, locality);
        }

        prefetcher
    }

    /// Advance to the next column and prefetch ahead.
    ///
    /// Call this as you process each column to keep data prefetched.
    #[inline]
    pub fn advance(&mut self) {
        self.current_col += 1;

        let prefetch_col = self.current_col + self.distance;
        if prefetch_col < self.ncols {
            let col_ptr = unsafe { self.ptr.add(prefetch_col * self.row_stride) };
            prefetch_column(col_ptr, self.nrows, 1, self.locality);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_locality() {
        assert_ne!(PrefetchLocality::High, PrefetchLocality::Low);
        assert_eq!(PrefetchLocality::Medium, PrefetchLocality::Medium);
    }

    // Prefetch tests use inline assembly which miri doesn't support
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_read_safety() {
        // Prefetching should not crash even with unusual inputs
        let data = [1.0f64; 1024];

        prefetch_read(data.as_ptr(), PrefetchLocality::High);
        prefetch_read(data.as_ptr().wrapping_add(100), PrefetchLocality::Medium);
        prefetch_read(data.as_ptr().wrapping_add(500), PrefetchLocality::Low);
        prefetch_read(
            data.as_ptr().wrapping_add(900),
            PrefetchLocality::NonTemporal,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_write_safety() {
        let mut data = [1.0f64; 1024];

        prefetch_write(data.as_mut_ptr(), PrefetchLocality::High);
        prefetch_write(
            data.as_mut_ptr().wrapping_add(100),
            PrefetchLocality::Medium,
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_range() {
        let data = vec![1.0f64; 4096];

        // Should not crash
        prefetch_range_read(data.as_ptr(), data.len(), PrefetchLocality::Medium);
        prefetch_range_read(data.as_ptr(), 0, PrefetchLocality::High); // Empty range
        prefetch_range_read(data.as_ptr(), 1, PrefetchLocality::Low); // Single element
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_column() {
        let data = vec![1.0f64; 1000];

        // Contiguous column
        prefetch_column(data.as_ptr(), 100, 1, PrefetchLocality::High);

        // Strided column (simulating row-major access)
        prefetch_column(data.as_ptr(), 10, 100, PrefetchLocality::Medium);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_block() {
        let data = vec![1.0f64; 10000];

        // Prefetch a 64x64 block with stride 100
        prefetch_block(data.as_ptr(), 64, 64, 100, PrefetchLocality::High);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_matrix_prefetcher() {
        let data = vec![1.0f64; 10000];

        let mut prefetcher = MatrixPrefetcher::new(
            data.as_ptr(),
            100, // nrows
            100, // ncols
            100, // row_stride
            8,   // distance
            PrefetchLocality::Medium,
        );

        // Simulate processing columns
        for _ in 0..100 {
            prefetcher.advance();
        }
    }

    #[test]
    fn test_cache_constants() {
        // CACHE_LINE_SIZE now comes from oxiblas-core and is
        // architecture-dependent (128 on aarch64, 64 elsewhere) -- assert
        // the invariants that must hold on any target rather than a
        // hardcoded value that would be wrong on aarch64.
        assert!(CACHE_LINE_SIZE.is_power_of_two());
        assert_eq!(CACHE_LINE_SIZE, oxiblas_core::memory::CACHE_LINE_SIZE);
        const { assert!(PREFETCH_DISTANCE_LINES > 0) };
        assert_eq!(
            PREFETCH_DISTANCE_BYTES,
            PREFETCH_DISTANCE_LINES * CACHE_LINE_SIZE
        );
    }

    /// Regression test for the strided-column coverage bug: the old loop
    /// computed `row = i * (CACHE_LINE_SIZE / elem_size)` while only
    /// iterating `i` up to `ceil(nrows * elem_size / CACHE_LINE_SIZE)`. For
    /// `f64` (`elem_size == 8`, `CACHE_LINE_SIZE == 64`) that stepped by 8
    /// rows at a time but only ran for `nrows / 8` iterations, so it only
    /// ever touched about 1/8 of the rows in the column (e.g. rows
    /// `0, 8, 16, ...` up to `nrows / 8 * 8`, leaving the vast majority of
    /// rows -- and the vast majority of cache lines -- never prefetched).
    ///
    /// The fixed strided path must visit every row exactly once, since each
    /// row lies on its own cache line when the stride exceeds a cache line.
    #[test]
    fn test_strided_column_prefetch_covers_all_rows() {
        for &nrows in &[0usize, 1, 7, 8, 9, 64, 100, 137, 1000] {
            let rows: Vec<usize> = strided_prefetch_row_indices(nrows).collect();
            assert_eq!(
                rows.len(),
                nrows,
                "strided prefetch must visit every row, not a fraction of them (nrows = {nrows})"
            );
            assert_eq!(
                rows,
                (0..nrows).collect::<Vec<_>>(),
                "strided prefetch must visit rows 0..nrows in order (nrows = {nrows})"
            );
        }
    }

    /// End-to-end version of the same regression: drive `prefetch_column`'s
    /// strided branch (`row_stride * size_of::<T>() > CACHE_LINE_SIZE`)
    /// directly with a large matrix so it exercises the real function, not
    /// just the extracted row-index helper. This should not crash and
    /// (indirectly, via `test_strided_column_prefetch_covers_all_rows`) is
    /// backed by the same coverage math.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_prefetch_column_strided_large() {
        // f64 is 8 bytes; a stride of CACHE_LINE_SIZE elements guarantees
        // `row_stride * elem_size > CACHE_LINE_SIZE`, landing us in the
        // strided branch for every row.
        let row_stride = CACHE_LINE_SIZE + 1;
        let nrows = 200;
        let data = vec![1.0f64; nrows * row_stride];

        prefetch_column(data.as_ptr(), nrows, row_stride, PrefetchLocality::Medium);
    }
}
