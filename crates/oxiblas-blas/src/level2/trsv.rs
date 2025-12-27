//! TRSV: Triangular solve for vectors.
//!
//! Solves A·x = b or A^T·x = b where A is triangular.
//!
//! ## Optimization
//!
//! This module uses a blocked algorithm for large systems that converts
//! TRSV into a series of smaller TRSV operations with GEMV updates.
//! The inner loops are unrolled for better instruction-level parallelism.

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Block size for blocked TRSV. Tuned for L1 cache.
const TRSV_BLOCK_SIZE: usize = 64;

/// Specifies which triangle of the matrix is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularMode {
    /// Lower triangular matrix (elements below and on diagonal).
    Lower,
    /// Upper triangular matrix (elements above and on diagonal).
    Upper,
}

/// Specifies whether the unit diagonal is assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularSide {
    /// Normal operation: solve A·x = b.
    NoTranspose,
    /// Transpose operation: solve A^T·x = b.
    Transpose,
}

/// Error type for triangular solve operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrsvError {
    /// Matrix is not square.
    NotSquare,
    /// Dimension mismatch between matrix and vector.
    DimensionMismatch,
    /// Matrix is singular (zero on diagonal).
    Singular,
}

impl core::fmt::Display for TrsvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch between matrix and vector"),
            Self::Singular => write!(f, "Matrix is singular (zero on diagonal)"),
        }
    }
}

impl std::error::Error for TrsvError {}

/// Solves the triangular system A·x = b.
///
/// # Arguments
///
/// * `a` - The triangular matrix A (only the specified triangle is used)
/// * `b` - The right-hand side vector b
/// * `mode` - Specifies whether A is lower or upper triangular
/// * `transpose` - If true, solves A^T·x = b instead of A·x = b
///
/// # Returns
///
/// The solution vector x, or an error if the operation fails.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{trsv, TriangularMode};
/// use oxiblas_matrix::Mat;
///
/// // Solve Lx = b where L is lower triangular
/// let l = Mat::from_rows(&[
///     &[2.0f64, 0.0, 0.0],
///     &[1.0, 3.0, 0.0],
///     &[2.0, 1.0, 4.0],
/// ]);
/// let b = [4.0f64, 5.0, 14.0];
///
/// let x = trsv(l.as_ref(), &b, TriangularMode::Lower, false).unwrap();
///
/// // Verify: L*x should equal b
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// ```
pub fn trsv<T: Field>(
    a: MatRef<'_, T>,
    b: &[T],
    mode: TriangularMode,
    transpose: bool,
) -> Result<Vec<T>, TrsvError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(TrsvError::NotSquare);
    }
    if n != b.len() {
        return Err(TrsvError::DimensionMismatch);
    }

    let mut x = b.to_vec();
    trsv_in_place(a, &mut x, mode, transpose)?;
    Ok(x)
}

/// Solves the triangular system A·x = b in-place.
///
/// The solution x overwrites b.
///
/// # Arguments
///
/// * `a` - The triangular matrix A
/// * `x` - On input: the vector b. On output: the solution x.
/// * `mode` - Specifies whether A is lower or upper triangular
/// * `transpose` - If true, solves A^T·x = b instead of A·x = b
pub fn trsv_in_place<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    transpose: bool,
) -> Result<(), TrsvError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(TrsvError::NotSquare);
    }
    if n != x.len() {
        return Err(TrsvError::DimensionMismatch);
    }

    if n == 0 {
        return Ok(());
    }

    // Use blocked algorithm for large systems
    if n > TRSV_BLOCK_SIZE * 2 {
        trsv_blocked(a, x, mode, transpose)
    } else {
        trsv_unblocked(a, x, mode, transpose, n)
    }
}

/// Blocked TRSV for large systems.
///
/// Divides the problem into blocks and uses GEMV for off-diagonal updates.
fn trsv_blocked<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    transpose: bool,
) -> Result<(), TrsvError> {
    let n = a.nrows();

    match (mode, transpose) {
        (TriangularMode::Lower, false) => {
            // Forward substitution with blocks
            for ib in (0..n).step_by(TRSV_BLOCK_SIZE) {
                let block_size = TRSV_BLOCK_SIZE.min(n - ib);

                // Update x[ib:ib+block_size] with contributions from previous blocks
                // x[ib:] -= A[ib:, 0:ib] * x[0:ib]
                for i in 0..block_size {
                    let row_idx = ib + i;
                    let mut sum = T::zero();

                    // Unrolled update from previous blocks
                    let chunks4 = ib / 4;
                    let remainder = ib % 4;

                    for j in 0..chunks4 {
                        let base = j * 4;
                        sum += a[(row_idx, base)] * x[base];
                        sum += a[(row_idx, base + 1)] * x[base + 1];
                        sum += a[(row_idx, base + 2)] * x[base + 2];
                        sum += a[(row_idx, base + 3)] * x[base + 3];
                    }
                    for j in 0..remainder {
                        let col_idx = chunks4 * 4 + j;
                        sum += a[(row_idx, col_idx)] * x[col_idx];
                    }

                    x[row_idx] -= sum;
                }

                // Solve the diagonal block
                trsv_unblocked_range(a, x, TriangularMode::Lower, false, ib, block_size)?;
            }
        }
        (TriangularMode::Upper, false) => {
            // Backward substitution with blocks
            let num_blocks = n.div_ceil(TRSV_BLOCK_SIZE);
            for block in (0..num_blocks).rev() {
                let ib = block * TRSV_BLOCK_SIZE;
                let block_size = TRSV_BLOCK_SIZE.min(n - ib);
                let block_end = ib + block_size;

                // Update x[ib:ib+block_size] with contributions from later blocks
                for i in 0..block_size {
                    let row_idx = ib + i;
                    let mut sum = T::zero();

                    // Unrolled update from later blocks
                    let start = block_end;
                    let len = n - start;
                    let chunks4 = len / 4;
                    let remainder = len % 4;

                    for j in 0..chunks4 {
                        let base = start + j * 4;
                        sum += a[(row_idx, base)] * x[base];
                        sum += a[(row_idx, base + 1)] * x[base + 1];
                        sum += a[(row_idx, base + 2)] * x[base + 2];
                        sum += a[(row_idx, base + 3)] * x[base + 3];
                    }
                    for j in 0..remainder {
                        let col_idx = start + chunks4 * 4 + j;
                        sum += a[(row_idx, col_idx)] * x[col_idx];
                    }

                    x[row_idx] -= sum;
                }

                // Solve the diagonal block
                trsv_unblocked_range(a, x, TriangularMode::Upper, false, ib, block_size)?;
            }
        }
        (TriangularMode::Lower, true) => {
            // L^T is upper triangular
            let num_blocks = n.div_ceil(TRSV_BLOCK_SIZE);
            for block in (0..num_blocks).rev() {
                let ib = block * TRSV_BLOCK_SIZE;
                let block_size = TRSV_BLOCK_SIZE.min(n - ib);
                let block_end = ib + block_size;

                // Update from later blocks
                for i in 0..block_size {
                    let row_idx = ib + i;
                    let mut sum = T::zero();

                    let start = block_end;
                    let len = n - start;
                    let chunks4 = len / 4;
                    let remainder = len % 4;

                    for j in 0..chunks4 {
                        let base = start + j * 4;
                        sum += a[(base, row_idx)] * x[base];
                        sum += a[(base + 1, row_idx)] * x[base + 1];
                        sum += a[(base + 2, row_idx)] * x[base + 2];
                        sum += a[(base + 3, row_idx)] * x[base + 3];
                    }
                    for j in 0..remainder {
                        let col_idx = start + chunks4 * 4 + j;
                        sum += a[(col_idx, row_idx)] * x[col_idx];
                    }

                    x[row_idx] -= sum;
                }

                // Solve the diagonal block
                trsv_unblocked_range(a, x, TriangularMode::Lower, true, ib, block_size)?;
            }
        }
        (TriangularMode::Upper, true) => {
            // U^T is lower triangular
            for ib in (0..n).step_by(TRSV_BLOCK_SIZE) {
                let block_size = TRSV_BLOCK_SIZE.min(n - ib);

                // Update from previous blocks
                for i in 0..block_size {
                    let row_idx = ib + i;
                    let mut sum = T::zero();

                    let chunks4 = ib / 4;
                    let remainder = ib % 4;

                    for j in 0..chunks4 {
                        let base = j * 4;
                        sum += a[(base, row_idx)] * x[base];
                        sum += a[(base + 1, row_idx)] * x[base + 1];
                        sum += a[(base + 2, row_idx)] * x[base + 2];
                        sum += a[(base + 3, row_idx)] * x[base + 3];
                    }
                    for j in 0..remainder {
                        let col_idx = chunks4 * 4 + j;
                        sum += a[(col_idx, row_idx)] * x[col_idx];
                    }

                    x[row_idx] -= sum;
                }

                // Solve the diagonal block
                trsv_unblocked_range(a, x, TriangularMode::Upper, true, ib, block_size)?;
            }
        }
    }

    Ok(())
}

/// Unblocked TRSV for the full matrix (small systems).
fn trsv_unblocked<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    transpose: bool,
    n: usize,
) -> Result<(), TrsvError> {
    trsv_unblocked_range(a, x, mode, transpose, 0, n)
}

/// Unblocked TRSV for a subrange of the matrix.
///
/// Solves for x[start:start+size] assuming the triangular block is at A[start:start+size, start:start+size].
fn trsv_unblocked_range<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    transpose: bool,
    start: usize,
    size: usize,
) -> Result<(), TrsvError> {
    match (mode, transpose) {
        (TriangularMode::Lower, false) => {
            // Forward substitution with 4-way unrolling
            for i in 0..size {
                let row_idx = start + i;
                let diag = a[(row_idx, row_idx)];
                if diag == T::zero() {
                    return Err(TrsvError::Singular);
                }

                let mut sum = x[row_idx];

                // Unrolled inner loop
                let chunks4 = i / 4;
                let remainder = i % 4;

                for j in 0..chunks4 {
                    let base = start + j * 4;
                    sum -= a[(row_idx, base)] * x[base];
                    sum -= a[(row_idx, base + 1)] * x[base + 1];
                    sum -= a[(row_idx, base + 2)] * x[base + 2];
                    sum -= a[(row_idx, base + 3)] * x[base + 3];
                }
                for j in 0..remainder {
                    let col_idx = start + chunks4 * 4 + j;
                    sum -= a[(row_idx, col_idx)] * x[col_idx];
                }

                x[row_idx] = sum / diag;
            }
        }
        (TriangularMode::Upper, false) => {
            // Backward substitution with 4-way unrolling
            for i in (0..size).rev() {
                let row_idx = start + i;
                let diag = a[(row_idx, row_idx)];
                if diag == T::zero() {
                    return Err(TrsvError::Singular);
                }

                let mut sum = x[row_idx];

                // Unrolled inner loop
                let remaining = size - i - 1;
                let chunks4 = remaining / 4;
                let remainder = remaining % 4;

                for j in 0..chunks4 {
                    let base = row_idx + 1 + j * 4;
                    sum -= a[(row_idx, base)] * x[base];
                    sum -= a[(row_idx, base + 1)] * x[base + 1];
                    sum -= a[(row_idx, base + 2)] * x[base + 2];
                    sum -= a[(row_idx, base + 3)] * x[base + 3];
                }
                for j in 0..remainder {
                    let col_idx = row_idx + 1 + chunks4 * 4 + j;
                    sum -= a[(row_idx, col_idx)] * x[col_idx];
                }

                x[row_idx] = sum / diag;
            }
        }
        (TriangularMode::Lower, true) => {
            // L^T is upper triangular, backward substitution
            for i in (0..size).rev() {
                let row_idx = start + i;
                let diag = a[(row_idx, row_idx)];
                if diag == T::zero() {
                    return Err(TrsvError::Singular);
                }

                let mut sum = x[row_idx];

                let remaining = size - i - 1;
                let chunks4 = remaining / 4;
                let remainder = remaining % 4;

                for j in 0..chunks4 {
                    let base = row_idx + 1 + j * 4;
                    sum -= a[(base, row_idx)] * x[base];
                    sum -= a[(base + 1, row_idx)] * x[base + 1];
                    sum -= a[(base + 2, row_idx)] * x[base + 2];
                    sum -= a[(base + 3, row_idx)] * x[base + 3];
                }
                for j in 0..remainder {
                    let col_idx = row_idx + 1 + chunks4 * 4 + j;
                    sum -= a[(col_idx, row_idx)] * x[col_idx];
                }

                x[row_idx] = sum / diag;
            }
        }
        (TriangularMode::Upper, true) => {
            // U^T is lower triangular, forward substitution
            for i in 0..size {
                let row_idx = start + i;
                let diag = a[(row_idx, row_idx)];
                if diag == T::zero() {
                    return Err(TrsvError::Singular);
                }

                let mut sum = x[row_idx];

                let chunks4 = i / 4;
                let remainder = i % 4;

                for j in 0..chunks4 {
                    let base = start + j * 4;
                    sum -= a[(base, row_idx)] * x[base];
                    sum -= a[(base + 1, row_idx)] * x[base + 1];
                    sum -= a[(base + 2, row_idx)] * x[base + 2];
                    sum -= a[(base + 3, row_idx)] * x[base + 3];
                }
                for j in 0..remainder {
                    let col_idx = start + chunks4 * 4 + j;
                    sum -= a[(col_idx, row_idx)] * x[col_idx];
                }

                x[row_idx] = sum / diag;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_trsv_lower() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // x = [2, 1, 2]
        // b = L*x = [2*2, 1*2+3*1, 2*2+1*1+4*2] = [4, 5, 13]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let b = [4.0, 5.0, 13.0];

        let x = trsv(l.as_ref(), &b, TriangularMode::Lower, false).unwrap();

        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 2.0).abs() < 1e-10);

        // Verify: L*x = b
        let check0 = 2.0 * x[0];
        let check1 = 1.0 * x[0] + 3.0 * x[1];
        let check2 = 2.0 * x[0] + 1.0 * x[1] + 4.0 * x[2];
        assert!((check0 - b[0]).abs() < 1e-10);
        assert!((check1 - b[1]).abs() < 1e-10);
        assert!((check2 - b[2]).abs() < 1e-10);
    }

    #[test]
    fn test_trsv_upper() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // x = [1, 2, 3]
        // b = U*x = [2+2+6, 6+3, 12] = [10, 9, 12]
        let u = Mat::from_rows(&[&[2.0f64, 1.0, 2.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 4.0]]);
        let b = [10.0, 9.0, 12.0];

        let x = trsv(u.as_ref(), &b, TriangularMode::Upper, false).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsv_lower_transpose() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // L^T = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // This is upper triangular, so we solve L^T * x = b by backward substitution
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);

        // L^T * x = b where x = [1, 2, 3]
        // b = [2*1 + 1*2 + 2*3, 3*2 + 1*3, 4*3] = [10, 9, 12]
        let b = [10.0, 9.0, 12.0];

        let x = trsv(l.as_ref(), &b, TriangularMode::Lower, true).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsv_upper_transpose() {
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // U^T = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // This is lower triangular, so we solve U^T * x = b by forward substitution
        let u = Mat::from_rows(&[&[2.0f64, 1.0, 2.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 4.0]]);

        // U^T * x = b where x = [2, 1, 2]
        // b = [2*2, 1*2 + 3*1, 2*2 + 1*1 + 4*2] = [4, 5, 13]
        let b = [4.0, 5.0, 13.0];

        let x = trsv(u.as_ref(), &b, TriangularMode::Upper, true).unwrap();

        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsv_singular() {
        let a = Mat::from_rows(&[
            &[2.0f64, 0.0, 0.0],
            &[1.0, 0.0, 0.0], // Zero on diagonal
            &[2.0, 1.0, 4.0],
        ]);
        let b = [4.0, 5.0, 14.0];

        let result = trsv(a.as_ref(), &b, TriangularMode::Lower, false);
        assert!(matches!(result, Err(TrsvError::Singular)));
    }

    #[test]
    fn test_trsv_dimension_mismatch() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = [4.0, 5.0, 14.0]; // Wrong size

        let result = trsv(a.as_ref(), &b, TriangularMode::Lower, false);
        assert!(matches!(result, Err(TrsvError::DimensionMismatch)));
    }

    #[test]
    fn test_trsv_not_square() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0]]);
        let b = [4.0, 5.0];

        let result = trsv(a.as_ref(), &b, TriangularMode::Lower, false);
        assert!(matches!(result, Err(TrsvError::NotSquare)));
    }

    #[test]
    fn test_trsv_identity() {
        // Identity matrix should return b unchanged
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = [1.0, 2.0, 3.0];

        let x_lower = trsv(eye.as_ref(), &b, TriangularMode::Lower, false).unwrap();
        let x_upper = trsv(eye.as_ref(), &b, TriangularMode::Upper, false).unwrap();

        for i in 0..3 {
            assert!((x_lower[i] - b[i]).abs() < 1e-10);
            assert!((x_upper[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_trsv_f32() {
        let l = Mat::from_rows(&[&[2.0f32, 0.0], &[1.0, 3.0]]);
        let b = [4.0f32, 5.0];

        let x = trsv(l.as_ref(), &b, TriangularMode::Lower, false).unwrap();

        // x[0] = 4/2 = 2, x[1] = (5 - 1*2)/3 = 3/3 = 1
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_trsv_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let b: [f64; 0] = [];

        let x = trsv(a.as_ref(), &b, TriangularMode::Lower, false).unwrap();
        assert!(x.is_empty());
    }

    #[test]
    fn test_trsv_large_lower() {
        // Test blocked TRSV with a large lower triangular matrix
        let n = 200; // Larger than TRSV_BLOCK_SIZE * 2 = 128

        // Create a lower triangular matrix with known solution
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0; // Diagonal
            for j in 0..i {
                a[(i, j)] = 0.1; // Below diagonal
            }
        }

        // Create a solution vector
        let x_expected: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Compute b = A * x_expected
        let mut b: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                b[i] += a[(i, j)] * x_expected[j];
            }
        }

        // Solve A * x = b
        let x = trsv(a.as_ref(), &b, TriangularMode::Lower, false).unwrap();

        // Verify solution
        for i in 0..n {
            assert!(
                (x[i] - x_expected[i]).abs() < 1e-8,
                "x[{}] = {}, expected {}",
                i,
                x[i],
                x_expected[i]
            );
        }
    }

    #[test]
    fn test_trsv_large_upper() {
        // Test blocked TRSV with a large upper triangular matrix
        let n = 200;

        // Create an upper triangular matrix
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 3.0; // Diagonal
            for j in (i + 1)..n {
                a[(i, j)] = 0.05; // Above diagonal
            }
        }

        // Create a solution vector
        let x_expected: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();

        // Compute b = A * x_expected
        let mut b: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in i..n {
                b[i] += a[(i, j)] * x_expected[j];
            }
        }

        // Solve A * x = b
        let x = trsv(a.as_ref(), &b, TriangularMode::Upper, false).unwrap();

        // Verify solution
        for i in 0..n {
            assert!(
                (x[i] - x_expected[i]).abs() < 1e-8,
                "x[{}] = {}, expected {}",
                i,
                x[i],
                x_expected[i]
            );
        }
    }

    #[test]
    fn test_trsv_large_transpose() {
        // Test blocked TRSV with transpose on large matrix
        let n = 200;

        // Create a lower triangular matrix
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            for j in 0..i {
                a[(i, j)] = 0.1;
            }
        }

        // x is known
        let x_expected: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Compute b = A^T * x_expected
        let mut b: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in i..n {
                // A^T[i,j] = A[j,i]
                b[i] += a[(j, i)] * x_expected[j];
            }
        }

        // Solve A^T * x = b
        let x = trsv(a.as_ref(), &b, TriangularMode::Lower, true).unwrap();

        // Verify solution
        for i in 0..n {
            assert!(
                (x[i] - x_expected[i]).abs() < 1e-8,
                "x[{}] = {}, expected {}",
                i,
                x[i],
                x_expected[i]
            );
        }
    }
}
