//! Property-based tests for oxiblas-matrix using quickcheck.
//!
//! These tests verify algebraic properties and invariants of matrix operations.

use num_complex::Complex64;
use oxiblas_matrix::{
    Mat,
    banded::BandedMat,
    ops,
    packed::{PackedMat, TriangularKind},
    symmetric::SymmetricMat,
    triangular::{DiagonalKind, TriangularMat},
};
use quickcheck_macros::quickcheck;

// ============================================================================
// Helpers for generating test data
// ============================================================================

/// Clamp dimension to reasonable size for tests.
/// Ensure at least 1 to avoid zero-size matrices.
fn clamp_dim(n: usize) -> usize {
    if n == 0 { 1 } else { n.min(64) }
}

/// Clamp bandwidth to valid range.
fn clamp_band(k: usize, n: usize) -> usize {
    k.min(n.saturating_sub(1))
}

// ============================================================================
// Mat<T> property tests
// ============================================================================

#[quickcheck]
fn prop_mat_zeros_all_zeros(rows: u8, cols: u8) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<f64> = Mat::zeros(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            if m[(i, j)] != 0.0 {
                return false;
            }
        }
    }
    true
}

#[quickcheck]
fn prop_mat_filled_all_same(rows: u8, cols: u8, val: f64) -> bool {
    if !val.is_finite() {
        return true; // Skip NaN/Inf
    }

    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<f64> = Mat::filled(rows, cols, val);

    for i in 0..rows {
        for j in 0..cols {
            if m[(i, j)] != val {
                return false;
            }
        }
    }
    true
}

#[quickcheck]
fn prop_mat_identity_correct(n: u8) -> bool {
    let n = clamp_dim(n as usize);
    let m: Mat<f64> = Mat::eye(n);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if m[(i, j)] != expected {
                return false;
            }
        }
    }
    true
}

#[quickcheck]
fn prop_mat_dimensions_correct(rows: u8, cols: u8) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<f64> = Mat::zeros(rows, cols);
    m.nrows() == rows && m.ncols() == cols
}

#[quickcheck]
fn prop_mat_transpose_involutory(rows: u8, cols: u8, seed: u64) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    // Create a matrix with deterministic values
    let mut m: Mat<f64> = Mat::zeros(rows, cols);
    let mut rng_state = seed;
    for i in 0..rows {
        for j in 0..cols {
            // Simple LCG for deterministic values
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            m[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    // Transpose twice should give original
    let t1 = m.as_ref().transpose();

    // Check dimensions
    if t1.nrows() != cols || t1.ncols() != rows {
        return false;
    }

    // Check elements: t1[i,j] == m[j,i]
    for i in 0..cols {
        for j in 0..rows {
            if (t1[(i, j)] - m[(j, i)]).abs() > 1e-14 {
                return false;
            }
        }
    }
    true
}

#[quickcheck]
fn prop_mat_diagonal_extraction_correct(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    let mut m: Mat<f64> = Mat::zeros(n, n);
    let mut rng_state = seed;
    for i in 0..n {
        for j in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            m[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    let diag = m.as_ref().diagonal();

    for i in 0..n {
        if (diag[i] - m[(i, i)]).abs() > 1e-14 {
            return false;
        }
    }
    true
}

#[quickcheck]
fn prop_mat_clone_equals_original(rows: u8, cols: u8, seed: u64) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let mut m: Mat<f64> = Mat::zeros(rows, cols);
    let mut rng_state = seed;
    for i in 0..rows {
        for j in 0..cols {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            m[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    let m2 = m.clone();

    for i in 0..rows {
        for j in 0..cols {
            if (m[(i, j)] - m2[(i, j)]).abs() > 1e-14 {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// PackedMat property tests
// ============================================================================

#[quickcheck]
fn prop_packed_upper_index_bounds(n: u8) -> bool {
    let n = clamp_dim(n as usize);
    let packed: PackedMat<f64> = PackedMat::zeros(n, TriangularKind::Upper);

    // Check that storage length is correct: n*(n+1)/2
    let expected_len = n * (n + 1) / 2;
    if packed.as_slice().len() != expected_len {
        return false;
    }

    // Check that all valid indices work
    for i in 0..n {
        for j in i..n {
            if packed.get(i, j).is_none() {
                return false;
            }
        }
    }

    // Check that invalid indices (below diagonal) return None
    for i in 1..n {
        for j in 0..i {
            if packed.get(i, j).is_some() {
                return false;
            }
        }
    }

    true
}

#[quickcheck]
fn prop_packed_lower_index_bounds(n: u8) -> bool {
    let n = clamp_dim(n as usize);
    let packed: PackedMat<f64> = PackedMat::zeros(n, TriangularKind::Lower);

    // Check that storage length is correct
    let expected_len = n * (n + 1) / 2;
    if packed.as_slice().len() != expected_len {
        return false;
    }

    // Check that all valid indices (on/below diagonal) work
    for i in 0..n {
        for j in 0..=i {
            if packed.get(i, j).is_none() {
                return false;
            }
        }
    }

    // Check that invalid indices (above diagonal) return None
    for i in 0..n {
        for j in (i + 1)..n {
            if packed.get(i, j).is_some() {
                return false;
            }
        }
    }

    true
}

#[quickcheck]
fn prop_packed_set_get_roundtrip(n: u8, row: u8, col: u8, val: f64) -> bool {
    if !val.is_finite() {
        return true;
    }

    let n = clamp_dim(n as usize);
    let row = (row as usize) % n;
    let col = (col as usize) % n;

    // Upper triangular: only set if row <= col
    let mut packed_upper: PackedMat<f64> = PackedMat::zeros(n, TriangularKind::Upper);
    if row <= col {
        packed_upper.set(row, col, val);
        if let Some(v) = packed_upper.get(row, col) {
            if (*v - val).abs() > 1e-14 {
                return false;
            }
        } else {
            return false;
        }
    }

    // Lower triangular: only set if row >= col
    let mut packed_lower: PackedMat<f64> = PackedMat::zeros(n, TriangularKind::Lower);
    if row >= col {
        packed_lower.set(row, col, val);
        if let Some(v) = packed_lower.get(row, col) {
            if (*v - val).abs() > 1e-14 {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

#[quickcheck]
fn prop_packed_from_dense_preserves_triangle(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    // Create a dense matrix
    let mut dense: Mat<f64> = Mat::zeros(n, n);
    let mut rng_state = seed;
    for i in 0..n {
        for j in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            dense[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    // Convert to packed upper
    let view = dense.as_ref();
    let packed_upper = PackedMat::from_dense(&view, TriangularKind::Upper);

    // Verify upper triangle is preserved
    for i in 0..n {
        for j in i..n {
            if let Some(v) = packed_upper.get(i, j) {
                let diff: f64 = *v - dense[(i, j)];
                if diff.abs() > 1e-14 {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    // Convert to packed lower
    let packed_lower = PackedMat::from_dense(&view, TriangularKind::Lower);

    // Verify lower triangle is preserved
    for i in 0..n {
        for j in 0..=i {
            if let Some(v) = packed_lower.get(i, j) {
                let diff: f64 = *v - dense[(i, j)];
                if diff.abs() > 1e-14 {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    true
}

// ============================================================================
// TriangularMat property tests
// ============================================================================

#[quickcheck]
fn prop_triangular_unit_diagonal_is_one(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    let mut upper: TriangularMat<f64> =
        TriangularMat::zeros(n, TriangularKind::Upper, DiagonalKind::Unit);

    // Fill some values
    let mut rng_state = seed;
    for i in 0..n {
        for j in (i + 1)..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            upper.set(i, j, val);
        }
    }

    // For unit triangular, get(i, i) returns None since diagonal is implicit.
    // Verify by converting to dense and checking diagonal.
    let dense = upper.to_dense();
    for i in 0..n {
        if (dense[(i, i)] - 1.0).abs() > 1e-14 {
            return false;
        }
    }

    true
}

#[quickcheck]
fn prop_triangular_non_unit_diagonal_preserved(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    let mut lower: TriangularMat<f64> =
        TriangularMat::zeros(n, TriangularKind::Lower, DiagonalKind::NonUnit);

    // Set diagonal to specific values
    let mut rng_state = seed;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        lower.set(i, i, val);
    }

    // Verify diagonal is preserved
    rng_state = seed;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let expected = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        if let Some(v) = lower.get(i, i) {
            if (*v - expected).abs() > 1e-14 {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

// ============================================================================
// SymmetricMat property tests
// ============================================================================

#[quickcheck]
fn prop_symmetric_is_symmetric(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    let mut sym: SymmetricMat<f64> = SymmetricMat::zeros(n, TriangularKind::Lower);

    // Fill with some values
    let mut rng_state = seed;
    for i in 0..n {
        for j in 0..=i {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            sym.set(i, j, val);
        }
    }

    // Verify symmetry: A[i,j] == A[j,i]
    for i in 0..n {
        for j in 0..n {
            let v1 = sym.get(i, j).copied().unwrap_or(0.0);
            let v2 = sym.get(j, i).copied().unwrap_or(0.0);
            if (v1 - v2).abs() > 1e-14 {
                return false;
            }
        }
    }

    true
}

#[quickcheck]
fn prop_symmetric_from_dense_correct(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    // Create a symmetric dense matrix
    let mut dense: Mat<f64> = Mat::zeros(n, n);
    let mut rng_state = seed;
    for i in 0..n {
        for j in 0..=i {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            dense[(i, j)] = val;
            dense[(j, i)] = val; // Make symmetric
        }
    }

    let view = dense.as_ref();
    let sym = SymmetricMat::from_dense(&view, TriangularKind::Lower);

    // Verify all elements match
    for i in 0..n {
        for j in 0..n {
            let sym_val = sym.get(i, j).copied().unwrap_or(0.0);
            if (sym_val - dense[(i, j)]).abs() > 1e-14 {
                return false;
            }
        }
    }

    true
}

// ============================================================================
// BandedMat property tests
// ============================================================================

#[quickcheck]
fn prop_banded_within_band_accessible(n: u8, kl: u8, ku: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);
    let kl = clamp_band(kl as usize, n);
    let ku = clamp_band(ku as usize, n);

    let mut banded: BandedMat<f64> = BandedMat::zeros(n, n, kl, ku);

    // Fill all elements within band
    let mut rng_state = seed;
    for i in 0..n {
        let j_start = i.saturating_sub(kl);
        let j_end = (i + ku + 1).min(n);

        for j in j_start..j_end {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            banded.set(i, j, val);
        }
    }

    // Verify all within-band elements are accessible and correct
    rng_state = seed;
    for i in 0..n {
        let j_start = i.saturating_sub(kl);
        let j_end = (i + ku + 1).min(n);

        for j in j_start..j_end {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let expected = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;

            if let Some(v) = banded.get(i, j) {
                if (*v - expected).abs() > 1e-14 {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    true
}

#[quickcheck]
fn prop_banded_outside_band_zero(n: u8, kl: u8, ku: u8) -> bool {
    let n = clamp_dim(n as usize);
    let kl = clamp_band(kl as usize, n);
    let ku = clamp_band(ku as usize, n);

    // Ensure there are elements outside the band
    if kl + ku + 1 >= n {
        return true; // All elements are in band, nothing to test
    }

    let banded: BandedMat<f64> = BandedMat::zeros(n, n, kl, ku);

    // Check elements outside band return None
    for i in 0..n {
        for j in 0..n {
            let in_band = (j >= i.saturating_sub(kl)) && (j <= i + ku);

            let result = banded.get(i, j);

            if in_band {
                // Should be accessible
                if result.is_none() {
                    return false;
                }
            } else {
                // Should be None (outside band)
                if result.is_some() {
                    return false;
                }
            }
        }
    }

    true
}

#[quickcheck]
fn prop_banded_storage_size_correct(n: u8, kl: u8, ku: u8) -> bool {
    let n = clamp_dim(n as usize);
    let kl = clamp_band(kl as usize, n);
    let ku = clamp_band(ku as usize, n);

    let banded: BandedMat<f64> = BandedMat::zeros(n, n, kl, ku);

    // Storage size should be (kl + ku + 1) * n
    let expected_len = (kl + ku + 1) * n;
    banded.as_slice().len() == expected_len
}

// ============================================================================
// Matrix operations property tests
// ============================================================================

#[quickcheck]
fn prop_trace_of_identity(n: u8) -> bool {
    let n = clamp_dim(n as usize);
    let eye: Mat<f64> = Mat::eye(n);

    let view = eye.as_ref();
    let trace = ops::trace(&view);
    (trace - n as f64).abs() < 1e-14
}

#[quickcheck]
fn prop_trace_is_sum_of_diagonal(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    let mut m: Mat<f64> = Mat::zeros(n, n);
    let mut rng_state = seed;
    let mut expected_trace = 0.0;

    for i in 0..n {
        for j in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            m[(i, j)] = val;
            if i == j {
                expected_trace += val;
            }
        }
    }

    let view = m.as_ref();
    let trace = ops::trace(&view);
    (trace - expected_trace).abs() < 1e-10
}

#[quickcheck]
fn prop_frobenius_norm_non_negative(rows: u8, cols: u8, seed: u64) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let mut m: Mat<f64> = Mat::zeros(rows, cols);
    let mut rng_state = seed;

    for i in 0..rows {
        for j in 0..cols {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            m[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    let view = m.as_ref();
    let norm_sq: f64 = ops::frobenius_norm_squared(&view);
    norm_sq >= 0.0
}

#[quickcheck]
fn prop_frobenius_norm_zero_for_zero_matrix(rows: u8, cols: u8) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<f64> = Mat::zeros(rows, cols);
    let view = m.as_ref();
    let norm_sq: f64 = ops::frobenius_norm_squared(&view);

    norm_sq.abs() < 1e-14
}

#[quickcheck]
fn prop_permute_rows_is_bijection(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    // Create permutation
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rng_state = seed;
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state as usize) % (i + 1);
        perm.swap(i, j);
    }

    // Create matrix with unique row sums
    let mut m: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = (i * n + j) as f64;
        }
    }

    // Calculate row sums before permutation
    let mut sums_before: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += m[(i, j)];
        }
        sums_before.push(sum);
    }

    // Permute rows - returns new matrix
    let view = m.as_ref();
    let result = ops::permute_rows(&view, &perm);

    // Calculate row sums after permutation
    let mut sums_after: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += result[(i, j)];
        }
        sums_after.push(sum);
    }

    // Verify permutation was applied correctly
    for i in 0..n {
        if (sums_after[i] - sums_before[perm[i]]).abs() > 1e-10 {
            return false;
        }
    }

    true
}

#[quickcheck]
fn prop_extract_and_set_diagonal_roundtrip(n: u8, seed: u64) -> bool {
    let n = clamp_dim(n as usize);

    // Create a matrix
    let mut m: Mat<f64> = Mat::zeros(n, n);
    let mut rng_state = seed;

    for i in 0..n {
        for j in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            m[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    // Extract diagonal
    let view = m.as_ref();
    let diag = ops::extract_diagonal(&view);

    // Create new matrix and set diagonal
    let mut m2: Mat<f64> = Mat::zeros(n, n);
    let mut view2 = m2.as_mut();
    ops::set_diagonal(&mut view2, &diag);

    // Verify diagonals match
    for i in 0..n {
        if (m2[(i, i)] - m[(i, i)]).abs() > 1e-14 {
            return false;
        }
    }

    true
}

// ============================================================================
// Layout and memory property tests
// ============================================================================

#[quickcheck]
fn prop_column_major_layout(rows: u8, cols: u8) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<f64> = Mat::zeros(rows, cols);
    let view = m.as_ref();

    // For column-major layout, the row_stride (leading dimension) should be >= nrows
    // It may be larger due to SIMD alignment padding
    view.row_stride() >= rows
}

#[quickcheck]
fn prop_mat_layout_correctness(rows: u8, cols: u8) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<f64> = Mat::zeros(rows, cols);

    // Mat's row_stride is the leading dimension (lda), which should be >= nrows
    // col_stride equals row_stride in this implementation
    m.row_stride() >= rows && m.col_stride() == m.row_stride()
}

#[quickcheck]
fn prop_submatrix_preserves_content(
    n: u8,
    seed: u64,
    offset_row: u8,
    offset_col: u8,
    sub_rows: u8,
    sub_cols: u8,
) -> bool {
    let n = clamp_dim(n as usize);

    // Create matrix
    let mut m: Mat<f64> = Mat::zeros(n, n);
    let mut rng_state = seed;

    for i in 0..n {
        for j in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            m[(i, j)] = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
    }

    // Calculate valid submatrix bounds
    let offset_row = (offset_row as usize) % n;
    let offset_col = (offset_col as usize) % n;
    let max_sub_rows = n - offset_row;
    let max_sub_cols = n - offset_col;
    let sub_rows = ((sub_rows as usize) % max_sub_rows).max(1);
    let sub_cols = ((sub_cols as usize) % max_sub_cols).max(1);

    let sub = m
        .as_ref()
        .submatrix(offset_row, offset_col, sub_rows, sub_cols);

    // Verify submatrix content matches
    for i in 0..sub_rows {
        for j in 0..sub_cols {
            if (sub[(i, j)] - m[(offset_row + i, offset_col + j)]).abs() > 1e-14 {
                return false;
            }
        }
    }

    true
}

// ============================================================================
// Complex number property tests
// ============================================================================

#[quickcheck]
fn prop_complex_mat_dimensions(rows: u8, cols: u8) -> bool {
    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);

    let m: Mat<Complex64> = Mat::filled(rows, cols, Complex64::new(0.0, 0.0));
    m.nrows() == rows && m.ncols() == cols
}

#[quickcheck]
fn prop_complex_mat_filled(rows: u8, cols: u8, re: f64, im: f64) -> bool {
    if !re.is_finite() || !im.is_finite() {
        return true;
    }

    let rows = clamp_dim(rows as usize);
    let cols = clamp_dim(cols as usize);
    let val = Complex64::new(re, im);

    let m: Mat<Complex64> = Mat::filled(rows, cols, val);

    for i in 0..rows {
        for j in 0..cols {
            if m[(i, j)] != val {
                return false;
            }
        }
    }
    true
}
