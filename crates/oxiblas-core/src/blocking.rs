//! Cache-oblivious blocking utilities.
//!
//! This module provides utilities for implementing cache-oblivious algorithms,
//! which achieve near-optimal cache performance without requiring knowledge of
//! cache sizes.
//!
//! # Cache-Oblivious Algorithms
//!
//! Cache-oblivious algorithms use recursive divide-and-conquer strategies that
//! naturally adapt to all levels of the memory hierarchy. The key insight is that
//! by recursively splitting the problem until the subproblems fit in cache, we get
//! optimal cache behavior without needing to know the cache size.
//!
//! # Block Size Calculation
//!
//! For cache-aware algorithms, this module also provides utilities to calculate
//! optimal block sizes based on:
//! - Available cache sizes (L1, L2, L3)
//! - SIMD register widths
//! - Memory layout (row-major, column-major)

use crate::tuning::{L1_CACHE_SIZE, L2_CACHE_SIZE};
use core::mem::size_of;

/// Base case threshold for recursive algorithms.
///
/// When the problem size drops below this threshold, we switch to
/// a direct (non-recursive) implementation.
pub const BASE_CASE_THRESHOLD: usize = 64;

/// Minimum block size for tiled algorithms.
pub const MIN_BLOCK_SIZE: usize = 16;

/// Maximum block size for tiled algorithms.
pub const MAX_BLOCK_SIZE: usize = 512;

/// Calculates the optimal block size for GEMM-like operations.
///
/// The block size is chosen to maximize data reuse in the L2 cache.
/// For GEMM with blocks of size M×K and K×N, we want:
/// `2 * M * K + K * N ≈ L2_CACHE_SIZE`
///
/// # Arguments
/// * `m` - Number of rows in the result
/// * `n` - Number of columns in the result
/// * `k` - Inner dimension
///
/// # Returns
/// A tuple `(block_m, block_n, block_k)` of optimal block sizes.
pub fn gemm_block_sizes<T>(m: usize, n: usize, k: usize) -> (usize, usize, usize) {
    let elem_size = size_of::<T>();

    // Target: fit 2 input panels + 1 output panel in L2
    // A panel: block_m × block_k
    // B panel: block_k × block_n
    // C panel: block_m × block_n
    let target_bytes = L2_CACHE_SIZE / 2;

    // Start with a balanced block size
    let max_block = ((target_bytes / elem_size / 3) as f64).sqrt() as usize;
    let mut block = max_block.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE);

    // Align to SIMD-friendly boundaries
    block = (block / 8) * 8;
    if block < MIN_BLOCK_SIZE {
        block = MIN_BLOCK_SIZE;
    }

    // Adjust for actual dimensions
    let block_m = block.min(m);
    let block_n = block.min(n);
    let block_k = block.min(k);

    (block_m, block_n, block_k)
}

/// Calculates the optimal block size for triangular solves (TRSM).
///
/// For TRSM, we need to balance between:
/// - Keeping the triangular block in L1 cache
/// - Processing multiple right-hand side columns
pub fn trsm_block_size<T>(n: usize, nrhs: usize) -> usize {
    let elem_size = size_of::<T>();

    // Target: fit triangular block in L1
    // Triangular block: n² / 2 elements
    let max_block = ((2 * L1_CACHE_SIZE / elem_size) as f64).sqrt() as usize;
    let block = max_block.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE / 2);

    // Align and adjust
    let block = (block / 8) * 8;
    block.min(n).min(nrhs).max(MIN_BLOCK_SIZE)
}

/// Calculates the optimal panel width for factorizations (LU, Cholesky, QR).
///
/// The panel width determines how many columns are processed together
/// before updating the trailing submatrix.
pub fn factorization_panel_width<T>(n: usize) -> usize {
    let elem_size = size_of::<T>();

    // For factorization, we want the panel to fit in L2 cache
    // Panel size: n × panel_width
    let max_panel = L2_CACHE_SIZE / (elem_size * n.max(1));
    let panel = max_panel.clamp(16, 128);

    // Align to SIMD boundaries
    ((panel / 4) * 4).min(n).max(16)
}

/// Recursive block range for cache-oblivious algorithms.
///
/// This structure represents a range that can be recursively split
/// for divide-and-conquer algorithms.
#[derive(Debug, Clone, Copy)]
pub struct BlockRange {
    /// Start index (inclusive)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
}

impl BlockRange {
    /// Creates a new block range.
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        BlockRange { start, end }
    }

    /// Creates a range from 0 to n.
    #[inline]
    pub const fn from_len(n: usize) -> Self {
        BlockRange { start: 0, end: n }
    }

    /// Returns the length of this range.
    #[inline]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true if this range is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Returns true if this range is a base case (should not be split further).
    #[inline]
    pub fn is_base_case(&self, threshold: usize) -> bool {
        self.len() <= threshold
    }

    /// Splits this range in half.
    ///
    /// Returns `(left_half, right_half)`.
    #[inline]
    pub fn split(&self) -> (Self, Self) {
        let mid = self.start + self.len() / 2;
        (
            BlockRange::new(self.start, mid),
            BlockRange::new(mid, self.end),
        )
    }

    /// Splits at a specific point.
    #[inline]
    pub fn split_at(&self, point: usize) -> (Self, Self) {
        let split = (self.start + point).min(self.end);
        (
            BlockRange::new(self.start, split),
            BlockRange::new(split, self.end),
        )
    }
}

/// Task for cache-oblivious recursive algorithm.
///
/// This represents a subproblem in a recursive decomposition.
#[derive(Debug, Clone, Copy)]
pub struct RecursiveTask {
    /// Row range
    pub rows: BlockRange,
    /// Column range
    pub cols: BlockRange,
}

impl RecursiveTask {
    /// Creates a new recursive task.
    #[inline]
    pub const fn new(rows: BlockRange, cols: BlockRange) -> Self {
        RecursiveTask { rows, cols }
    }

    /// Creates a task for an m×n matrix.
    #[inline]
    pub const fn from_dims(m: usize, n: usize) -> Self {
        RecursiveTask {
            rows: BlockRange::from_len(m),
            cols: BlockRange::from_len(n),
        }
    }

    /// Returns the number of elements in this task.
    #[inline]
    pub fn size(&self) -> usize {
        self.rows.len() * self.cols.len()
    }

    /// Returns true if this is a base case.
    #[inline]
    pub fn is_base_case(&self, threshold: usize) -> bool {
        self.rows.len() <= threshold && self.cols.len() <= threshold
    }

    /// Splits along the larger dimension.
    ///
    /// Returns two subtasks by splitting the larger dimension in half.
    pub fn split(&self) -> (Self, Self) {
        if self.rows.len() >= self.cols.len() {
            // Split rows
            let (r1, r2) = self.rows.split();
            (
                RecursiveTask::new(r1, self.cols),
                RecursiveTask::new(r2, self.cols),
            )
        } else {
            // Split columns
            let (c1, c2) = self.cols.split();
            (
                RecursiveTask::new(self.rows, c1),
                RecursiveTask::new(self.rows, c2),
            )
        }
    }

    /// Quadrant decomposition for 2D recursive algorithms.
    ///
    /// Returns `(top_left, top_right, bottom_left, bottom_right)`.
    pub fn quadrants(&self) -> (Self, Self, Self, Self) {
        let (r1, r2) = self.rows.split();
        let (c1, c2) = self.cols.split();

        (
            RecursiveTask::new(r1, c1), // top-left
            RecursiveTask::new(r1, c2), // top-right
            RecursiveTask::new(r2, c1), // bottom-left
            RecursiveTask::new(r2, c2), // bottom-right
        )
    }
}

/// Visitor pattern for cache-oblivious matrix traversal.
///
/// Implement this trait to process matrix blocks in a cache-efficient order.
pub trait BlockVisitor {
    /// The error type for visit operations.
    type Error;

    /// Visits a matrix block.
    ///
    /// # Arguments
    /// * `row_start`, `row_end` - Row range (exclusive end)
    /// * `col_start`, `col_end` - Column range (exclusive end)
    fn visit_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<(), Self::Error>;
}

/// Performs a cache-oblivious traversal of a matrix.
///
/// This recursively divides the matrix into quadrants until reaching
/// the base case threshold, then visits each block.
pub fn cache_oblivious_traverse<V: BlockVisitor>(
    visitor: &mut V,
    task: RecursiveTask,
    threshold: usize,
) -> Result<(), V::Error> {
    if task.is_base_case(threshold) {
        // Base case: visit this block directly
        visitor.visit_block(
            task.rows.start,
            task.rows.end,
            task.cols.start,
            task.cols.end,
        )
    } else {
        // Recursive case: split and process
        let (t1, t2) = task.split();
        cache_oblivious_traverse(visitor, t1, threshold)?;
        cache_oblivious_traverse(visitor, t2, threshold)
    }
}

/// Morton (Z-order) curve index calculation.
///
/// Morton ordering provides good cache locality for 2D data by
/// interleaving the bits of x and y coordinates.
#[inline]
pub fn morton_index(x: u32, y: u32) -> u64 {
    fn expand_bits(v: u32) -> u64 {
        let mut v = v as u64;
        v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
        v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        v = (v | (v << 1)) & 0x5555_5555_5555_5555;
        v
    }
    expand_bits(x) | (expand_bits(y) << 1)
}

/// Inverse Morton index: extracts (x, y) from a Morton index.
#[inline]
pub fn morton_decode(z: u64) -> (u32, u32) {
    fn compact_bits(mut v: u64) -> u32 {
        v &= 0x5555_5555_5555_5555;
        v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
        v = (v | (v >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
        v = (v | (v >> 4)) & 0x00FF_00FF_00FF_00FF;
        v = (v | (v >> 8)) & 0x0000_FFFF_0000_FFFF;
        v = (v | (v >> 16)) & 0x0000_0000_FFFF_FFFF;
        v as u32
    }
    (compact_bits(z), compact_bits(z >> 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_block_sizes() {
        let (bm, bn, bk) = gemm_block_sizes::<f64>(1024, 1024, 1024);

        // Block sizes should be reasonable
        assert!(bm >= MIN_BLOCK_SIZE);
        assert!(bn >= MIN_BLOCK_SIZE);
        assert!(bk >= MIN_BLOCK_SIZE);
        assert!(bm <= MAX_BLOCK_SIZE);
        assert!(bn <= MAX_BLOCK_SIZE);
        assert!(bk <= MAX_BLOCK_SIZE);

        // Should be divisible by 8
        assert_eq!(bm % 8, 0);
    }

    #[test]
    fn test_block_range() {
        let range = BlockRange::new(0, 100);
        assert_eq!(range.len(), 100);

        let (left, right) = range.split();
        assert_eq!(left.start, 0);
        assert_eq!(left.end, 50);
        assert_eq!(right.start, 50);
        assert_eq!(right.end, 100);

        assert!(BlockRange::new(0, 32).is_base_case(64));
        assert!(!BlockRange::new(0, 100).is_base_case(64));
    }

    #[test]
    fn test_recursive_task() {
        let task = RecursiveTask::from_dims(100, 200);
        assert_eq!(task.size(), 20000);

        // Should split along columns (larger dimension)
        let (t1, t2) = task.split();
        assert_eq!(t1.cols.len(), 100);
        assert_eq!(t2.cols.len(), 100);
        assert_eq!(t1.rows.len(), 100);
        assert_eq!(t2.rows.len(), 100);
    }

    #[test]
    fn test_quadrants() {
        let task = RecursiveTask::from_dims(100, 100);
        let (tl, _tr, _bl, br) = task.quadrants();

        assert_eq!(tl.rows.start, 0);
        assert_eq!(tl.rows.end, 50);
        assert_eq!(tl.cols.start, 0);
        assert_eq!(tl.cols.end, 50);

        assert_eq!(br.rows.start, 50);
        assert_eq!(br.rows.end, 100);
        assert_eq!(br.cols.start, 50);
        assert_eq!(br.cols.end, 100);
    }

    #[test]
    fn test_morton_index() {
        // Morton index interleaves bits
        assert_eq!(morton_index(0, 0), 0);
        assert_eq!(morton_index(1, 0), 1);
        assert_eq!(morton_index(0, 1), 2);
        assert_eq!(morton_index(1, 1), 3);
        assert_eq!(morton_index(2, 0), 4);

        // Roundtrip test
        for x in 0..100 {
            for y in 0..100 {
                let z = morton_index(x, y);
                let (dx, dy) = morton_decode(z);
                assert_eq!((dx, dy), (x, y));
            }
        }
    }

    struct CountingVisitor {
        count: usize,
        total_elements: usize,
    }

    impl BlockVisitor for CountingVisitor {
        type Error = ();

        fn visit_block(
            &mut self,
            row_start: usize,
            row_end: usize,
            col_start: usize,
            col_end: usize,
        ) -> Result<(), ()> {
            self.count += 1;
            self.total_elements += (row_end - row_start) * (col_end - col_start);
            Ok(())
        }
    }

    #[test]
    fn test_cache_oblivious_traverse() {
        let task = RecursiveTask::from_dims(128, 128);
        let mut visitor = CountingVisitor {
            count: 0,
            total_elements: 0,
        };

        cache_oblivious_traverse(&mut visitor, task, 32).unwrap();

        // Should visit multiple blocks
        assert!(visitor.count > 1);
        // Should cover all elements
        assert_eq!(visitor.total_elements, 128 * 128);
    }
}
