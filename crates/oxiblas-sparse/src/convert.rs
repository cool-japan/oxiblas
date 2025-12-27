//! Format conversion utilities for sparse matrices.
//!
//! Provides efficient conversions between:
//! - CSR ↔ CSC
//! - COO → CSR/CSC
//! - CSR ↔ DIA
//! - CSR ↔ ELL
//! - CSR ↔ BSR
//! - CSR ↔ BSC
//! - CSR ↔ HYB
//! - CSR ↔ SELL
//! - Sparse ↔ Dense
//!
//! # Automatic Format Selection
//!
//! Use [`analyze_sparsity_pattern`] to determine the optimal format for a matrix.

use crate::bsc::BscMatrix;
use crate::bsr::BsrMatrix;
use crate::coo::CooMatrix;
use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use crate::dia::DiaMatrix;
use crate::ell::EllMatrix;
use crate::hyb::{HybMatrix, HybWidthStrategy};
use crate::sell::{SellMatrix, SliceSize};
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Converts a CSR matrix to CSC format.
///
/// Time complexity: O(nnz)
/// Space complexity: O(nnz) for the output
pub fn csr_to_csc<T: Scalar + Clone>(csr: &CsrMatrix<T>) -> CscMatrix<T> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let nnz = csr.nnz();

    if nnz == 0 {
        return CscMatrix::zeros(nrows, ncols);
    }

    // Count entries per column
    let mut col_counts = vec![0usize; ncols];
    for &col in csr.col_indices() {
        col_counts[col] += 1;
    }

    // Build column pointers
    let mut col_ptrs = vec![0usize; ncols + 1];
    for i in 0..ncols {
        col_ptrs[i + 1] = col_ptrs[i] + col_counts[i];
    }

    // Fill in values and row indices
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut write_pos = col_ptrs.clone();

    for row in 0..nrows {
        let start = csr.row_ptrs()[row];
        let end = csr.row_ptrs()[row + 1];

        for i in start..end {
            let col = csr.col_indices()[i];
            let pos = write_pos[col];

            row_indices[pos] = row;
            values[pos] = csr.values()[i].clone();

            write_pos[col] += 1;
        }
    }

    // SAFETY: We've constructed valid CSC data
    unsafe { CscMatrix::new_unchecked(nrows, ncols, col_ptrs, row_indices, values) }
}

/// Converts a CSC matrix to CSR format.
///
/// Time complexity: O(nnz)
/// Space complexity: O(nnz) for the output
pub fn csc_to_csr<T: Scalar + Clone>(csc: &CscMatrix<T>) -> CsrMatrix<T> {
    let nrows = csc.nrows();
    let ncols = csc.ncols();
    let nnz = csc.nnz();

    if nnz == 0 {
        return CsrMatrix::zeros(nrows, ncols);
    }

    // Count entries per row
    let mut row_counts = vec![0usize; nrows];
    for &row in csc.row_indices() {
        row_counts[row] += 1;
    }

    // Build row pointers
    let mut row_ptrs = vec![0usize; nrows + 1];
    for i in 0..nrows {
        row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
    }

    // Fill in values and column indices
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut write_pos = row_ptrs.clone();

    for col in 0..ncols {
        let start = csc.col_ptrs()[col];
        let end = csc.col_ptrs()[col + 1];

        for i in start..end {
            let row = csc.row_indices()[i];
            let pos = write_pos[row];

            col_indices[pos] = col;
            values[pos] = csc.values()[i].clone();

            write_pos[row] += 1;
        }
    }

    // SAFETY: We've constructed valid CSR data
    unsafe { CsrMatrix::new_unchecked(nrows, ncols, row_ptrs, col_indices, values) }
}

/// Converts a COO matrix to CSR format, summing duplicate entries.
///
/// Time complexity: O(nnz log nnz) due to sorting
/// Space complexity: O(nnz) for the output
pub fn coo_to_csr<T: Scalar<Real = T> + Clone + Field + Real>(coo: &CooMatrix<T>) -> CsrMatrix<T> {
    let nrows = coo.nrows();
    let ncols = coo.ncols();

    if coo.is_empty() {
        return CsrMatrix::zeros(nrows, ncols);
    }

    // Sort entries by (row, col)
    let mut indices: Vec<usize> = (0..coo.len()).collect();
    indices.sort_by_key(|&i| (coo.row_indices()[i], coo.col_indices()[i]));

    // Build CSR data, summing duplicates
    let mut row_ptrs = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::with_capacity(coo.len());
    let mut values: Vec<T> = Vec::with_capacity(coo.len());

    row_ptrs.push(0);
    let mut current_row = 0;

    for &idx in &indices {
        let row = coo.row_indices()[idx];
        let col = coo.col_indices()[idx];
        let val = coo.values()[idx].clone();

        // Fill in empty rows
        while current_row < row {
            row_ptrs.push(values.len());
            current_row += 1;
        }

        // Check for duplicate
        if !values.is_empty() && col_indices.last() == Some(&col) && current_row == row {
            // Same position as last entry, accumulate
            let last = values.len() - 1;
            values[last] = values[last].clone() + val;
        } else {
            // Skip zeros after accumulation
            if !values.is_empty() {
                let last = values.len() - 1;
                if Scalar::abs(values[last].clone()) <= <T as Scalar>::epsilon() {
                    values.pop();
                    col_indices.pop();
                }
            }
            // New entry
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                col_indices.push(col);
                values.push(val);
            }
        }
    }

    // Clean up last entry if it became zero
    if !values.is_empty() {
        let last = values.len() - 1;
        if Scalar::abs(values[last].clone()) <= <T as Scalar>::epsilon() {
            values.pop();
            col_indices.pop();
        }
    }

    // Fill remaining row pointers
    while current_row < nrows {
        row_ptrs.push(values.len());
        current_row += 1;
    }
    row_ptrs.push(values.len());

    // SAFETY: We've constructed valid CSR data
    unsafe { CsrMatrix::new_unchecked(nrows, ncols, row_ptrs, col_indices, values) }
}

/// Converts a COO matrix to CSC format, summing duplicate entries.
///
/// Time complexity: O(nnz log nnz) due to sorting
/// Space complexity: O(nnz) for the output
pub fn coo_to_csc<T: Scalar<Real = T> + Clone + Field + Real>(coo: &CooMatrix<T>) -> CscMatrix<T> {
    let nrows = coo.nrows();
    let ncols = coo.ncols();

    if coo.is_empty() {
        return CscMatrix::zeros(nrows, ncols);
    }

    // Sort entries by (col, row)
    let mut indices: Vec<usize> = (0..coo.len()).collect();
    indices.sort_by_key(|&i| (coo.col_indices()[i], coo.row_indices()[i]));

    // Build CSC data, summing duplicates
    let mut col_ptrs = Vec::with_capacity(ncols + 1);
    let mut row_indices = Vec::with_capacity(coo.len());
    let mut values: Vec<T> = Vec::with_capacity(coo.len());

    col_ptrs.push(0);
    let mut current_col = 0;

    for &idx in &indices {
        let row = coo.row_indices()[idx];
        let col = coo.col_indices()[idx];
        let val = coo.values()[idx].clone();

        // Fill in empty columns
        while current_col < col {
            col_ptrs.push(values.len());
            current_col += 1;
        }

        // Check for duplicate
        if !values.is_empty() && row_indices.last() == Some(&row) && current_col == col {
            // Same position as last entry, accumulate
            let last = values.len() - 1;
            values[last] = values[last].clone() + val;
        } else {
            // Skip zeros after accumulation
            if !values.is_empty() {
                let last = values.len() - 1;
                if Scalar::abs(values[last].clone()) <= <T as Scalar>::epsilon() {
                    values.pop();
                    row_indices.pop();
                }
            }
            // New entry
            if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                row_indices.push(row);
                values.push(val);
            }
        }
    }

    // Clean up last entry if it became zero
    if !values.is_empty() {
        let last = values.len() - 1;
        if Scalar::abs(values[last].clone()) <= <T as Scalar>::epsilon() {
            values.pop();
            row_indices.pop();
        }
    }

    // Fill remaining column pointers
    while current_col < ncols {
        col_ptrs.push(values.len());
        current_col += 1;
    }
    col_ptrs.push(values.len());

    // SAFETY: We've constructed valid CSC data
    unsafe { CscMatrix::new_unchecked(nrows, ncols, col_ptrs, row_indices, values) }
}

/// Converts a CSR matrix to COO format.
pub fn csr_to_coo<T: Scalar + Clone>(csr: &CsrMatrix<T>) -> CooMatrix<T> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let nnz = csr.nnz();

    let mut row_indices = Vec::with_capacity(nnz);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for row in 0..nrows {
        let start = csr.row_ptrs()[row];
        let end = csr.row_ptrs()[row + 1];

        for i in start..end {
            row_indices.push(row);
            col_indices.push(csr.col_indices()[i]);
            values.push(csr.values()[i].clone());
        }
    }

    // SAFETY: Valid COO data derived from valid CSR
    unsafe { CooMatrix::new_unchecked(nrows, ncols, row_indices, col_indices, values) }
}

/// Converts a CSC matrix to COO format.
pub fn csc_to_coo<T: Scalar + Clone>(csc: &CscMatrix<T>) -> CooMatrix<T> {
    let nrows = csc.nrows();
    let ncols = csc.ncols();
    let nnz = csc.nnz();

    let mut row_indices = Vec::with_capacity(nnz);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for col in 0..ncols {
        let start = csc.col_ptrs()[col];
        let end = csc.col_ptrs()[col + 1];

        for i in start..end {
            row_indices.push(csc.row_indices()[i]);
            col_indices.push(col);
            values.push(csc.values()[i].clone());
        }
    }

    // SAFETY: Valid COO data derived from valid CSC
    unsafe { CooMatrix::new_unchecked(nrows, ncols, row_indices, col_indices, values) }
}

// ============================================================================
// DIA Conversions
// ============================================================================

/// Converts a CSR matrix to DIA format.
///
/// # Arguments
///
/// * `csr` - Source CSR matrix
/// * `offsets` - Optional list of diagonal offsets to extract. If None, all non-empty diagonals are extracted.
///
/// Time complexity: O(nnz)
pub fn csr_to_dia<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
    offsets: Option<Vec<isize>>,
) -> DiaMatrix<T> {
    let (nrows, ncols) = csr.shape();
    let eps = <T as Scalar>::epsilon();

    // Find all non-empty diagonals if not specified
    let offsets = offsets.unwrap_or_else(|| {
        let mut found = std::collections::HashSet::new();
        for (row, col, val) in csr.iter() {
            if Scalar::abs(val.clone()) > eps {
                found.insert(col as isize - row as isize);
            }
        }
        let mut offsets: Vec<_> = found.into_iter().collect();
        offsets.sort();
        offsets
    });

    if offsets.is_empty() {
        return DiaMatrix::zeros(nrows, ncols);
    }

    let diag_len = nrows.min(ncols);
    let mut data = Vec::with_capacity(offsets.len());

    for &offset in &offsets {
        let mut diag = vec![T::zero(); diag_len];

        // Fill diagonal from CSR
        // Element A[row, col] where col = row + offset goes to data index (row + offset)
        // This matches DiaMatrix::data_index which uses (row as isize + offset) as usize
        for (row, col, val) in csr.iter() {
            let expected_col = (row as isize + offset) as usize;
            if col == expected_col && row < nrows && col < ncols {
                // data_index = row + offset (accounting for padding)
                let idx = (row as isize + offset) as usize;
                if idx < diag_len {
                    diag[idx] = val.clone();
                }
            }
        }

        data.push(diag);
    }

    // Safety: we constructed valid DIA data
    unsafe { DiaMatrix::new_unchecked(nrows, ncols, offsets, data) }
}

/// Converts a DIA matrix to CSR format.
///
/// Time complexity: O(nrows * ndiag)
pub fn dia_to_csr<T: Scalar + Clone + Field>(dia: &DiaMatrix<T>) -> CsrMatrix<T> {
    dia.to_csr()
}

// ============================================================================
// ELL Conversions
// ============================================================================

/// Converts a CSR matrix to ELL format.
///
/// # Arguments
///
/// * `csr` - Source CSR matrix
/// * `max_width` - Optional maximum width (if None, uses actual max non-zeros per row)
///
/// Time complexity: O(nnz)
pub fn csr_to_ell<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
    max_width: Option<usize>,
) -> Result<EllMatrix<T>, crate::ell::EllError> {
    EllMatrix::from_csr(csr, max_width)
}

/// Converts an ELL matrix to CSR format.
///
/// Time complexity: O(nrows * width)
pub fn ell_to_csr<T: Scalar + Clone + Field>(ell: &EllMatrix<T>) -> CsrMatrix<T> {
    ell.to_csr()
}

// ============================================================================
// BSR Conversions
// ============================================================================

/// Converts a CSR matrix to BSR format.
///
/// # Arguments
///
/// * `csr` - Source CSR matrix
/// * `block_rows` - Block row size
/// * `block_cols` - Block column size
///
/// Time complexity: O(nnz)
pub fn csr_to_bsr<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
    block_rows: usize,
    block_cols: usize,
) -> BsrMatrix<T> {
    BsrMatrix::from_csr(csr, block_rows, block_cols)
}

/// Converts a BSR matrix to CSR format.
///
/// Time complexity: O(nblocks * block_size)
pub fn bsr_to_csr<T: Scalar + Clone + Field>(bsr: &BsrMatrix<T>) -> CsrMatrix<T> {
    bsr.to_csr()
}

// ============================================================================
// Cross-format conversions
// ============================================================================

/// Converts a DIA matrix to ELL format.
pub fn dia_to_ell<T: Scalar + Clone + Field>(
    dia: &DiaMatrix<T>,
    max_width: Option<usize>,
) -> Result<EllMatrix<T>, crate::ell::EllError> {
    let csr = dia.to_csr();
    EllMatrix::from_csr(&csr, max_width)
}

/// Converts an ELL matrix to DIA format.
pub fn ell_to_dia<T: Scalar + Clone + Field>(
    ell: &EllMatrix<T>,
    offsets: Option<Vec<isize>>,
) -> DiaMatrix<T> {
    let csr = ell.to_csr();
    csr_to_dia(&csr, offsets)
}

/// Converts a DIA matrix to BSR format.
pub fn dia_to_bsr<T: Scalar + Clone + Field>(
    dia: &DiaMatrix<T>,
    block_rows: usize,
    block_cols: usize,
) -> BsrMatrix<T> {
    let csr = dia.to_csr();
    BsrMatrix::from_csr(&csr, block_rows, block_cols)
}

/// Converts a BSR matrix to DIA format.
pub fn bsr_to_dia<T: Scalar + Clone + Field>(
    bsr: &BsrMatrix<T>,
    offsets: Option<Vec<isize>>,
) -> DiaMatrix<T> {
    let csr = bsr.to_csr();
    csr_to_dia(&csr, offsets)
}

/// Converts an ELL matrix to BSR format.
pub fn ell_to_bsr<T: Scalar + Clone + Field>(
    ell: &EllMatrix<T>,
    block_rows: usize,
    block_cols: usize,
) -> BsrMatrix<T> {
    let csr = ell.to_csr();
    BsrMatrix::from_csr(&csr, block_rows, block_cols)
}

/// Converts a BSR matrix to ELL format.
pub fn bsr_to_ell<T: Scalar + Clone + Field>(
    bsr: &BsrMatrix<T>,
    max_width: Option<usize>,
) -> Result<EllMatrix<T>, crate::ell::EllError> {
    let csr = bsr.to_csr();
    EllMatrix::from_csr(&csr, max_width)
}

// ============================================================================
// BSC Conversions
// ============================================================================

/// Converts a CSR matrix to BSC format.
///
/// # Arguments
///
/// * `csr` - Source CSR matrix
/// * `block_rows` - Block row size
/// * `block_cols` - Block column size
pub fn csr_to_bsc<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
    block_rows: usize,
    block_cols: usize,
) -> BscMatrix<T> {
    let bsr = BsrMatrix::from_csr(csr, block_rows, block_cols);
    BscMatrix::from_bsr(&bsr)
}

/// Converts a BSC matrix to CSR format.
pub fn bsc_to_csr<T: Scalar + Clone + Field>(bsc: &BscMatrix<T>) -> CsrMatrix<T> {
    let bsr = bsc.to_bsr();
    bsr.to_csr()
}

/// Converts a BSC matrix to BSR format.
pub fn bsc_to_bsr<T: Scalar + Clone + Field>(bsc: &BscMatrix<T>) -> BsrMatrix<T> {
    bsc.to_bsr()
}

/// Converts a BSR matrix to BSC format.
pub fn bsr_to_bsc<T: Scalar + Clone + Field>(bsr: &BsrMatrix<T>) -> BscMatrix<T> {
    BscMatrix::from_bsr(bsr)
}

// ============================================================================
// HYB Conversions
// ============================================================================

/// Converts a CSR matrix to HYB format.
///
/// # Arguments
///
/// * `csr` - Source CSR matrix
/// * `strategy` - Strategy for determining ELL width
pub fn csr_to_hyb<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
    strategy: HybWidthStrategy,
) -> HybMatrix<T> {
    HybMatrix::from_csr(csr, strategy)
}

/// Converts a HYB matrix to CSR format.
pub fn hyb_to_csr<T: Scalar + Clone + Field>(hyb: &HybMatrix<T>) -> CsrMatrix<T> {
    hyb.to_csr()
}

/// Converts an ELL matrix to HYB format (no COO overflow).
pub fn ell_to_hyb<T: Scalar + Clone + Field>(ell: &EllMatrix<T>) -> HybMatrix<T> {
    HybMatrix::from_ell(ell)
}

/// Converts a HYB matrix to ELL format.
pub fn hyb_to_ell<T: Scalar + Clone + Field>(hyb: &HybMatrix<T>) -> EllMatrix<T> {
    hyb.to_ell()
}

// ============================================================================
// SELL Conversions
// ============================================================================

/// Converts a CSR matrix to SELL (Sliced ELLPACK) format.
///
/// # Arguments
///
/// * `csr` - Source CSR matrix
/// * `slice_size` - Size of each slice (typically 32 or 64 for GPU)
pub fn csr_to_sell<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
    slice_size: SliceSize,
) -> SellMatrix<T> {
    SellMatrix::from_csr(csr, slice_size)
}

/// Converts a SELL matrix to CSR format.
pub fn sell_to_csr<T: Scalar + Clone + Field>(sell: &SellMatrix<T>) -> CsrMatrix<T> {
    sell.to_csr()
}

// ============================================================================
// Format Detection and Analysis
// ============================================================================

/// Recommended sparse matrix format based on sparsity analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedFormat {
    /// CSR: General purpose, good for row-wise operations.
    Csr,
    /// CSC: Good for column-wise operations and direct solvers.
    Csc,
    /// DIA: Optimal for banded/diagonal matrices.
    Dia,
    /// ELL: Good for matrices with uniform row lengths.
    Ell,
    /// HYB: Good for matrices with mostly uniform rows but some outliers.
    Hyb,
    /// SELL: Good for GPU computation with variable row lengths.
    Sell,
    /// BSR: Good for block-structured matrices.
    Bsr,
    /// BSC: Good for column-oriented block-structured matrices.
    Bsc,
}

/// Analysis of a sparse matrix's sparsity pattern.
#[derive(Debug, Clone)]
pub struct SparsityAnalysis {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Number of non-zeros.
    pub nnz: usize,
    /// Density (nnz / (nrows * ncols)).
    pub density: f64,
    /// Maximum row length.
    pub max_row_length: usize,
    /// Minimum row length.
    pub min_row_length: usize,
    /// Average row length.
    pub avg_row_length: f64,
    /// Standard deviation of row lengths.
    pub row_length_stddev: f64,
    /// Number of distinct diagonals with entries.
    pub num_diagonals: usize,
    /// True if matrix appears to have block structure.
    pub has_block_structure: bool,
    /// Detected block size (if any).
    pub detected_block_size: Option<(usize, usize)>,
    /// Recommended format for this matrix.
    pub recommended_format: RecommendedFormat,
}

/// Analyzes the sparsity pattern of a CSR matrix and recommends a format.
///
/// # Returns
///
/// A `SparsityAnalysis` containing statistics and a recommended format.
pub fn analyze_sparsity_pattern<T: Scalar + Clone + Field>(csr: &CsrMatrix<T>) -> SparsityAnalysis {
    let (nrows, ncols) = csr.shape();
    let nnz = csr.nnz();
    let eps = <T as Scalar>::epsilon();

    if nrows == 0 || ncols == 0 {
        return SparsityAnalysis {
            nrows,
            ncols,
            nnz,
            density: 0.0,
            max_row_length: 0,
            min_row_length: 0,
            avg_row_length: 0.0,
            row_length_stddev: 0.0,
            num_diagonals: 0,
            has_block_structure: false,
            detected_block_size: None,
            recommended_format: RecommendedFormat::Csr,
        };
    }

    // Compute row lengths
    let mut row_lengths = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let mut count = 0;
        for (_, val) in csr.row_iter(row) {
            if Scalar::abs(val.clone()) > eps {
                count += 1;
            }
        }
        row_lengths.push(count);
    }

    let max_row_length = row_lengths.iter().max().copied().unwrap_or(0);
    let min_row_length = row_lengths.iter().min().copied().unwrap_or(0);
    let avg_row_length = if nrows > 0 {
        row_lengths.iter().sum::<usize>() as f64 / nrows as f64
    } else {
        0.0
    };

    // Compute standard deviation
    let variance: f64 = row_lengths
        .iter()
        .map(|&x| {
            let diff = x as f64 - avg_row_length;
            diff * diff
        })
        .sum::<f64>()
        / nrows.max(1) as f64;
    let row_length_stddev = variance.sqrt();

    // Count distinct diagonals
    let mut diagonals = std::collections::HashSet::new();
    for (row, col, val) in csr.iter() {
        if Scalar::abs(val.clone()) > eps {
            diagonals.insert(col as isize - row as isize);
        }
    }
    let num_diagonals = diagonals.len();

    // Check for block structure (simple heuristic)
    let (has_block_structure, detected_block_size) = detect_block_structure(csr);

    let density = if nrows * ncols > 0 {
        nnz as f64 / (nrows * ncols) as f64
    } else {
        0.0
    };

    // Determine recommended format
    let recommended_format = determine_recommended_format(
        nrows,
        ncols,
        nnz,
        max_row_length,
        min_row_length,
        row_length_stddev,
        num_diagonals,
        has_block_structure,
    );

    SparsityAnalysis {
        nrows,
        ncols,
        nnz,
        density,
        max_row_length,
        min_row_length,
        avg_row_length,
        row_length_stddev,
        num_diagonals,
        has_block_structure,
        detected_block_size,
        recommended_format,
    }
}

/// Detects if a matrix has block structure.
fn detect_block_structure<T: Scalar + Clone + Field>(
    csr: &CsrMatrix<T>,
) -> (bool, Option<(usize, usize)>) {
    let (nrows, ncols) = csr.shape();
    let eps = <T as Scalar>::epsilon();

    if nrows < 4 || ncols < 4 {
        return (false, None);
    }

    // Try common block sizes
    for block_size in [2, 3, 4, 6, 8] {
        if nrows % block_size != 0 || ncols % block_size != 0 {
            continue;
        }

        let _num_block_rows = nrows / block_size;
        let _num_block_cols = ncols / block_size;

        // Check if entries align with blocks
        let block_aligned = true;
        let mut blocks_found = std::collections::HashSet::new();

        for (row, col, val) in csr.iter() {
            if Scalar::abs(val.clone()) > eps {
                let block_row = row / block_size;
                let block_col = col / block_size;
                blocks_found.insert((block_row, block_col));
            }
        }

        // Verify that within each block, we have dense or near-dense entries
        let mut dense_blocks = 0;
        for &(br, bc) in &blocks_found {
            let mut count = 0;
            for i in 0..block_size {
                for j in 0..block_size {
                    let row = br * block_size + i;
                    let col = bc * block_size + j;
                    if let Some(val) = csr.get(row, col) {
                        if Scalar::abs(val.clone()) > eps {
                            count += 1;
                        }
                    }
                }
            }
            // Consider block dense if > 50% full
            if count * 2 >= block_size * block_size {
                dense_blocks += 1;
            }
        }

        // Consider it block-structured if > 70% of found blocks are dense
        if !blocks_found.is_empty() && dense_blocks * 10 >= blocks_found.len() * 7 {
            return (true, Some((block_size, block_size)));
        }
        if !block_aligned {
            // Just to avoid warnings, this is always true
            continue;
        }
    }

    (false, None)
}

/// Determines the recommended format based on matrix characteristics.
fn determine_recommended_format(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    max_row_length: usize,
    min_row_length: usize,
    row_length_stddev: f64,
    num_diagonals: usize,
    has_block_structure: bool,
) -> RecommendedFormat {
    // Empty or very small matrix
    if nnz == 0 || nrows <= 10 || ncols <= 10 {
        return RecommendedFormat::Csr;
    }

    let avg_row_length = nnz as f64 / nrows.max(1) as f64;

    // Block structure
    if has_block_structure {
        return RecommendedFormat::Bsr;
    }

    // Diagonal/banded structure
    // If number of diagonals is small relative to matrix size
    if num_diagonals <= 10 && num_diagonals * 2 <= nrows.max(1) {
        return RecommendedFormat::Dia;
    }

    // Uniform row lengths (low variance)
    let coefficient_of_variation = row_length_stddev / avg_row_length.max(1.0);

    if coefficient_of_variation < 0.3 {
        // Very uniform - ELL is efficient
        return RecommendedFormat::Ell;
    }

    if coefficient_of_variation < 0.8 {
        // Moderately uniform but with some variation - HYB is good
        return RecommendedFormat::Hyb;
    }

    // High variance in row lengths
    if max_row_length > min_row_length * 10 {
        // Very irregular - SELL handles this well for GPU
        return RecommendedFormat::Sell;
    }

    // Default to CSR
    RecommendedFormat::Csr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsr::DenseBlock;

    #[test]
    fn test_csr_to_csc() {
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let csc = csr_to_csc(&csr);

        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.get(0, 0), Some(&1.0));
        assert_eq!(csc.get(0, 2), Some(&2.0));
        assert_eq!(csc.get(1, 1), Some(&3.0));
        assert_eq!(csc.get(2, 0), Some(&4.0));
        assert_eq!(csc.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_csc_to_csr() {
        // [1 0 4]
        // [0 3 0]
        // [2 0 5]
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        let csr = csc_to_csr(&csc);

        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(0, 2), Some(&4.0));
        assert_eq!(csr.get(1, 1), Some(&3.0));
        assert_eq!(csr.get(2, 0), Some(&2.0));
        assert_eq!(csr.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_coo_to_csr() {
        let row_indices = vec![0, 1, 2, 0, 2];
        let col_indices = vec![0, 1, 0, 2, 2];
        let values = vec![1.0f64, 3.0, 4.0, 2.0, 5.0];

        let coo = CooMatrix::new(3, 3, row_indices, col_indices, values).unwrap();
        let csr = coo_to_csr(&coo);

        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(0, 2), Some(&2.0));
        assert_eq!(csr.get(1, 1), Some(&3.0));
        assert_eq!(csr.get(2, 0), Some(&4.0));
        assert_eq!(csr.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_coo_to_csr_duplicates() {
        // Duplicate entries at (0,0)
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 0, 1];
        let values = vec![1.0f64, 2.0, 3.0];

        let coo = CooMatrix::new(2, 2, row_indices, col_indices, values).unwrap();
        let csr = coo_to_csr(&coo);

        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.get(0, 0), Some(&3.0)); // 1 + 2
        assert_eq!(csr.get(1, 1), Some(&3.0));
    }

    #[test]
    fn test_coo_to_csc() {
        let row_indices = vec![0, 1, 2, 0, 2];
        let col_indices = vec![0, 1, 0, 2, 2];
        let values = vec![1.0f64, 3.0, 4.0, 2.0, 5.0];

        let coo = CooMatrix::new(3, 3, row_indices, col_indices, values).unwrap();
        let csc = coo_to_csc(&coo);

        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.get(0, 0), Some(&1.0));
        assert_eq!(csc.get(0, 2), Some(&2.0));
        assert_eq!(csc.get(1, 1), Some(&3.0));
        assert_eq!(csc.get(2, 0), Some(&4.0));
        assert_eq!(csc.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_roundtrip_csr_csc_csr() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr1 = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let csc = csr_to_csc(&csr1);
        let csr2 = csc_to_csr(&csc);

        assert_eq!(csr1.nnz(), csr2.nnz());
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(csr1.get(row, col), csr2.get(row, col));
            }
        }
    }

    #[test]
    fn test_csr_to_coo() {
        let values = vec![1.0f64, 2.0, 3.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let coo = csr_to_coo(&csr);

        assert_eq!(coo.len(), 3);
        let entries: Vec<_> = coo.iter().map(|(r, c, v)| (r, c, *v)).collect();
        assert_eq!(entries, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    }

    #[test]
    fn test_empty_matrix_conversion() {
        let csr: CsrMatrix<f64> = CsrMatrix::zeros(5, 3);
        let csc = csr_to_csc(&csr);

        assert_eq!(csc.nrows(), 5);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 0);
    }

    // ========================================================================
    // DIA conversion tests
    // ========================================================================

    #[test]
    fn test_csr_to_dia_tridiagonal() {
        // Tridiagonal matrix:
        // [4 1 0]
        // [2 5 1]
        // [0 3 6]
        let values = vec![4.0f64, 1.0, 2.0, 5.0, 1.0, 3.0, 6.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let dia = csr_to_dia(&csr, None);

        assert_eq!(dia.ndiag(), 3);
        assert_eq!(dia.get(0, 0), Some(&4.0));
        assert_eq!(dia.get(0, 1), Some(&1.0));
        assert_eq!(dia.get(1, 0), Some(&2.0));
        assert_eq!(dia.get(1, 1), Some(&5.0));
        assert_eq!(dia.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_dia_to_csr() {
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        let csr = dia_to_csr(&dia);

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.get(0, 0), Some(&4.0));
        assert_eq!(csr.get(1, 0), Some(&2.0));
    }

    #[test]
    fn test_csr_dia_roundtrip() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 1, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr1 = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let dia = csr_to_dia(&csr1, None);
        let csr2 = dia_to_csr(&dia);

        for row in 0..3 {
            for col in 0..3 {
                let v1 = csr1.get(row, col).cloned().unwrap_or(0.0);
                let v2 = csr2.get(row, col).cloned().unwrap_or(0.0);
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    // ========================================================================
    // ELL conversion tests
    // ========================================================================

    #[test]
    fn test_csr_to_ell() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 1, 2, 0, 3];
        let row_ptrs = vec![0, 2, 4, 6];

        let csr = CsrMatrix::new(3, 4, row_ptrs, col_indices, values).unwrap();
        let ell = csr_to_ell(&csr, None).unwrap();

        assert_eq!(ell.width(), 2);
        assert_eq!(ell.get(0, 0), Some(&1.0));
        assert_eq!(ell.get(1, 2), Some(&4.0));
    }

    #[test]
    fn test_ell_to_csr() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let indices = vec![vec![0, 1], vec![1, 2]];

        let ell = EllMatrix::new(2, 3, 2, data, indices).unwrap();
        let csr = ell_to_csr(&ell);

        assert_eq!(csr.nrows(), 2);
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(1, 2), Some(&4.0));
    }

    #[test]
    fn test_csr_ell_roundtrip() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 1, 2];
        let row_ptrs = vec![0, 2, 4];

        let csr1 = CsrMatrix::new(2, 3, row_ptrs, col_indices, values).unwrap();
        let ell = csr_to_ell(&csr1, None).unwrap();
        let csr2 = ell_to_csr(&ell);

        for row in 0..2 {
            for col in 0..3 {
                let v1 = csr1.get(row, col).cloned().unwrap_or(0.0);
                let v2 = csr2.get(row, col).cloned().unwrap_or(0.0);
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    // ========================================================================
    // BSR conversion tests
    // ========================================================================

    #[test]
    fn test_csr_to_bsr() {
        // 4x4 matrix with 2x2 block structure
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col_indices = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let row_ptrs = vec![0, 2, 4, 6, 8];

        let csr = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let bsr = csr_to_bsr(&csr, 2, 2);

        assert_eq!(bsr.nblocks(), 2);
        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(3, 3), Some(8.0));
    }

    #[test]
    fn test_bsr_to_csr() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let csr = bsr_to_csr(&bsr);

        assert_eq!(csr.nrows(), 4);
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_csr_bsr_roundtrip() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col_indices = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let row_ptrs = vec![0, 2, 4, 6, 8];

        let csr1 = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let bsr = csr_to_bsr(&csr1, 2, 2);
        let csr2 = bsr_to_csr(&bsr);

        for row in 0..4 {
            for col in 0..4 {
                let v1 = csr1.get(row, col).cloned().unwrap_or(0.0);
                let v2 = csr2.get(row, col).cloned().unwrap_or(0.0);
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    // ========================================================================
    // Cross-format conversion tests
    // ========================================================================

    #[test]
    fn test_dia_to_ell() {
        let offsets = vec![0];
        let data = vec![vec![1.0, 2.0, 3.0]];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        let ell = dia_to_ell(&dia, None).unwrap();

        assert_eq!(ell.width(), 1);
        assert_eq!(ell.get(0, 0), Some(&1.0));
        assert_eq!(ell.get(1, 1), Some(&2.0));
    }

    #[test]
    fn test_dia_to_bsr() {
        let offsets = vec![0];
        let data = vec![vec![1.0, 2.0, 3.0, 4.0]];

        let dia = DiaMatrix::new(4, 4, offsets, data).unwrap();
        let bsr = dia_to_bsr(&dia, 2, 2);

        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(1, 1), Some(2.0));
    }

    #[test]
    fn test_ell_to_bsr() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let indices = vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]];

        let ell = EllMatrix::new(4, 4, 2, data, indices).unwrap();
        let bsr = ell_to_bsr(&ell, 2, 2);

        assert_eq!(bsr.nrows(), 4);
        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(3, 3), Some(8.0));
    }
}
