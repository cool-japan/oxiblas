//! Helper functions for BLAS Level 2 operations.

/// Compute index for upper triangular packed storage.
pub(crate) fn upper_packed_index(i: usize, j: usize) -> usize {
    // Upper triangular: column j contains j+1 elements
    j * (j + 1) / 2 + i
}

/// Compute index for lower triangular packed storage.
pub(crate) fn lower_packed_index(i: usize, j: usize, n: usize) -> usize {
    // Lower triangular: column j has n-j elements
    j * n - j * (j - 1) / 2 + (i - j)
}
