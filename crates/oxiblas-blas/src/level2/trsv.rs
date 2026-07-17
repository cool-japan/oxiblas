//! TRSV: Triangular solve for vectors.
//!
//! Solves op(A)·x = b where A is triangular and op(A) is one of:
//! A (no transpose), A^T (transpose), or A^H (conjugate transpose).
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

/// Specifies the operation applied to the triangular matrix A before solving.
///
/// This mirrors the BLAS `TRANS` parameter of reference `?TRSV`: `NoTrans`
/// solves `A·x = b`, `Trans` solves `A^T·x = b`, and `ConjTrans` solves
/// `A^H·x = b` (conjugate transpose, only meaningful for complex `T` — for
/// real types it behaves identically to `Trans`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrsvTrans {
    /// No transpose: solve A·x = b.
    NoTrans,
    /// Transpose: solve A^T·x = b.
    Trans,
    /// Conjugate transpose: solve A^H·x = b.
    ConjTrans,
}

/// Specifies whether the triangular matrix has a unit diagonal.
///
/// This mirrors the BLAS `DIAG` parameter of reference `?TRSV`. When `Unit`,
/// the diagonal elements of `A` are assumed to be `1.0` and are never read
/// (nor divided by) even if the underlying storage holds different, or even
/// non-invertible, values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrsvDiag {
    /// Non-unit diagonal (use actual diagonal values).
    NonUnit,
    /// Unit diagonal (assume diagonal is all ones; diagonal elements are not read).
    Unit,
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

/// Solves the triangular system op(A)·x = b.
///
/// # Arguments
///
/// * `a` - The triangular matrix A (only the specified triangle is used)
/// * `b` - The right-hand side vector b
/// * `mode` - Specifies whether A is lower or upper triangular
/// * `trans` - Specifies whether to solve with A, A^T, or A^H
/// * `diag` - Specifies whether A has an implicit unit diagonal
///
/// # Returns
///
/// The solution vector x, or an error if the operation fails.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{trsv, TriangularMode, TrsvDiag, TrsvTrans};
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
/// let x = trsv(l.as_ref(), &b, TriangularMode::Lower, TrsvTrans::NoTrans, TrsvDiag::NonUnit).unwrap();
///
/// // Verify: L*x should equal b
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// ```
pub fn trsv<T: Field>(
    a: MatRef<'_, T>,
    b: &[T],
    mode: TriangularMode,
    trans: TrsvTrans,
    diag: TrsvDiag,
) -> Result<Vec<T>, TrsvError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(TrsvError::NotSquare);
    }
    if n != b.len() {
        return Err(TrsvError::DimensionMismatch);
    }

    let mut x = b.to_vec();
    trsv_in_place(a, &mut x, mode, trans, diag)?;
    Ok(x)
}

/// Solves the triangular system op(A)·x = b in-place.
///
/// The solution x overwrites b.
///
/// # Arguments
///
/// * `a` - The triangular matrix A
/// * `x` - On input: the vector b. On output: the solution x.
/// * `mode` - Specifies whether A is lower or upper triangular
/// * `trans` - Specifies whether to solve with A, A^T, or A^H
/// * `diag` - Specifies whether A has an implicit unit diagonal
pub fn trsv_in_place<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    trans: TrsvTrans,
    diag: TrsvDiag,
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
        trsv_blocked(a, x, mode, trans, diag)
    } else {
        trsv_unblocked(a, x, mode, trans, diag, n)
    }
}

/// Applies complex conjugation to a matrix element when `conj` is set.
///
/// For real `T`, `T::conj` is the identity, so this is a no-op there.
#[inline]
fn maybe_conj<T: Field>(value: T, conj: bool) -> T {
    if conj { value.conj() } else { value }
}

/// Blocked TRSV for large systems.
///
/// Divides the problem into blocks and uses GEMV for off-diagonal updates.
fn trsv_blocked<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    trans: TrsvTrans,
    diag: TrsvDiag,
) -> Result<(), TrsvError> {
    let n = a.nrows();
    let transposed = trans != TrsvTrans::NoTrans;
    let conj = trans == TrsvTrans::ConjTrans;

    match (mode, transposed) {
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
                trsv_unblocked_range(
                    a,
                    x,
                    TriangularMode::Lower,
                    TrsvTrans::NoTrans,
                    diag,
                    ib,
                    block_size,
                )?;
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
                trsv_unblocked_range(
                    a,
                    x,
                    TriangularMode::Upper,
                    TrsvTrans::NoTrans,
                    diag,
                    ib,
                    block_size,
                )?;
            }
        }
        (TriangularMode::Lower, true) => {
            // L^T (or L^H) is upper triangular
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
                        sum += maybe_conj(a[(base, row_idx)], conj) * x[base];
                        sum += maybe_conj(a[(base + 1, row_idx)], conj) * x[base + 1];
                        sum += maybe_conj(a[(base + 2, row_idx)], conj) * x[base + 2];
                        sum += maybe_conj(a[(base + 3, row_idx)], conj) * x[base + 3];
                    }
                    for j in 0..remainder {
                        let col_idx = start + chunks4 * 4 + j;
                        sum += maybe_conj(a[(col_idx, row_idx)], conj) * x[col_idx];
                    }

                    x[row_idx] -= sum;
                }

                // Solve the diagonal block
                trsv_unblocked_range(a, x, TriangularMode::Lower, trans, diag, ib, block_size)?;
            }
        }
        (TriangularMode::Upper, true) => {
            // U^T (or U^H) is lower triangular
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
                        sum += maybe_conj(a[(base, row_idx)], conj) * x[base];
                        sum += maybe_conj(a[(base + 1, row_idx)], conj) * x[base + 1];
                        sum += maybe_conj(a[(base + 2, row_idx)], conj) * x[base + 2];
                        sum += maybe_conj(a[(base + 3, row_idx)], conj) * x[base + 3];
                    }
                    for j in 0..remainder {
                        let col_idx = chunks4 * 4 + j;
                        sum += maybe_conj(a[(col_idx, row_idx)], conj) * x[col_idx];
                    }

                    x[row_idx] -= sum;
                }

                // Solve the diagonal block
                trsv_unblocked_range(a, x, TriangularMode::Upper, trans, diag, ib, block_size)?;
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
    trans: TrsvTrans,
    diag: TrsvDiag,
    n: usize,
) -> Result<(), TrsvError> {
    trsv_unblocked_range(a, x, mode, trans, diag, 0, n)
}

/// Unblocked TRSV for a subrange of the matrix.
///
/// Solves for x[start:start+size] assuming the triangular block is at A[start:start+size, start:start+size].
fn trsv_unblocked_range<T: Field>(
    a: MatRef<'_, T>,
    x: &mut [T],
    mode: TriangularMode,
    trans: TrsvTrans,
    diag: TrsvDiag,
    start: usize,
    size: usize,
) -> Result<(), TrsvError> {
    let conj = trans == TrsvTrans::ConjTrans;

    match (mode, trans) {
        (TriangularMode::Lower, TrsvTrans::NoTrans) => {
            // Forward substitution with 4-way unrolling
            for i in 0..size {
                let row_idx = start + i;

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

                x[row_idx] = match diag {
                    TrsvDiag::Unit => sum,
                    TrsvDiag::NonUnit => {
                        let diag_val = a[(row_idx, row_idx)];
                        if diag_val == T::zero() {
                            return Err(TrsvError::Singular);
                        }
                        sum / diag_val
                    }
                };
            }
        }
        (TriangularMode::Upper, TrsvTrans::NoTrans) => {
            // Backward substitution with 4-way unrolling
            for i in (0..size).rev() {
                let row_idx = start + i;

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

                x[row_idx] = match diag {
                    TrsvDiag::Unit => sum,
                    TrsvDiag::NonUnit => {
                        let diag_val = a[(row_idx, row_idx)];
                        if diag_val == T::zero() {
                            return Err(TrsvError::Singular);
                        }
                        sum / diag_val
                    }
                };
            }
        }
        (TriangularMode::Lower, TrsvTrans::Trans | TrsvTrans::ConjTrans) => {
            // L^T (or L^H) is upper triangular, backward substitution
            for i in (0..size).rev() {
                let row_idx = start + i;

                let mut sum = x[row_idx];

                let remaining = size - i - 1;
                let chunks4 = remaining / 4;
                let remainder = remaining % 4;

                for j in 0..chunks4 {
                    let base = row_idx + 1 + j * 4;
                    sum -= maybe_conj(a[(base, row_idx)], conj) * x[base];
                    sum -= maybe_conj(a[(base + 1, row_idx)], conj) * x[base + 1];
                    sum -= maybe_conj(a[(base + 2, row_idx)], conj) * x[base + 2];
                    sum -= maybe_conj(a[(base + 3, row_idx)], conj) * x[base + 3];
                }
                for j in 0..remainder {
                    let col_idx = row_idx + 1 + chunks4 * 4 + j;
                    sum -= maybe_conj(a[(col_idx, row_idx)], conj) * x[col_idx];
                }

                x[row_idx] = match diag {
                    TrsvDiag::Unit => sum,
                    TrsvDiag::NonUnit => {
                        let diag_val = maybe_conj(a[(row_idx, row_idx)], conj);
                        if diag_val == T::zero() {
                            return Err(TrsvError::Singular);
                        }
                        sum / diag_val
                    }
                };
            }
        }
        (TriangularMode::Upper, TrsvTrans::Trans | TrsvTrans::ConjTrans) => {
            // U^T (or U^H) is lower triangular, forward substitution
            for i in 0..size {
                let row_idx = start + i;

                let mut sum = x[row_idx];

                let chunks4 = i / 4;
                let remainder = i % 4;

                for j in 0..chunks4 {
                    let base = start + j * 4;
                    sum -= maybe_conj(a[(base, row_idx)], conj) * x[base];
                    sum -= maybe_conj(a[(base + 1, row_idx)], conj) * x[base + 1];
                    sum -= maybe_conj(a[(base + 2, row_idx)], conj) * x[base + 2];
                    sum -= maybe_conj(a[(base + 3, row_idx)], conj) * x[base + 3];
                }
                for j in 0..remainder {
                    let col_idx = start + chunks4 * 4 + j;
                    sum -= maybe_conj(a[(col_idx, row_idx)], conj) * x[col_idx];
                }

                x[row_idx] = match diag {
                    TrsvDiag::Unit => sum,
                    TrsvDiag::NonUnit => {
                        let diag_val = maybe_conj(a[(row_idx, row_idx)], conj);
                        if diag_val == T::zero() {
                            return Err(TrsvError::Singular);
                        }
                        sum / diag_val
                    }
                };
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn approx_eq_c(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10
    }

    #[test]
    fn test_trsv_lower() {
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // x = [2, 1, 2]
        // b = L*x = [2*2, 1*2+3*1, 2*2+1*1+4*2] = [4, 5, 13]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let b = [4.0, 5.0, 13.0];

        let x = trsv(
            l.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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

        let x = trsv(
            u.as_ref(),
            &b,
            TriangularMode::Upper,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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

        let x = trsv(
            l.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::Trans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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

        let x = trsv(
            u.as_ref(),
            &b,
            TriangularMode::Upper,
            TrsvTrans::Trans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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

        let result = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        );
        assert!(matches!(result, Err(TrsvError::Singular)));
    }

    #[test]
    fn test_trsv_dimension_mismatch() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = [4.0, 5.0, 14.0]; // Wrong size

        let result = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        );
        assert!(matches!(result, Err(TrsvError::DimensionMismatch)));
    }

    #[test]
    fn test_trsv_not_square() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0]]);
        let b = [4.0, 5.0];

        let result = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        );
        assert!(matches!(result, Err(TrsvError::NotSquare)));
    }

    #[test]
    fn test_trsv_identity() {
        // Identity matrix should return b unchanged
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = [1.0, 2.0, 3.0];

        let x_lower = trsv(
            eye.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();
        let x_upper = trsv(
            eye.as_ref(),
            &b,
            TriangularMode::Upper,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

        for i in 0..3 {
            assert!((x_lower[i] - b[i]).abs() < 1e-10);
            assert!((x_upper[i] - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_trsv_f32() {
        let l = Mat::from_rows(&[&[2.0f32, 0.0], &[1.0, 3.0]]);
        let b = [4.0f32, 5.0];

        let x = trsv(
            l.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

        // x[0] = 4/2 = 2, x[1] = (5 - 1*2)/3 = 3/3 = 1
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_trsv_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let b: [f64; 0] = [];

        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();
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
        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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
        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Upper,
            TrsvTrans::NoTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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
        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::Trans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

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

    /// Regression test: TrsvDiag::Unit must ignore whatever garbage is stored
    /// on the diagonal (even a zero, which would otherwise be flagged
    /// Singular) and act as if the diagonal were exactly 1.0.
    #[test]
    fn test_trsv_unit_diagonal_ignores_garbage() {
        // L has an intended unit diagonal, but the storage holds garbage
        // values on the diagonal, including a non-invertible zero.
        // True L (unit diagonal) = [[1, 0, 0], [3, 1, 0], [2, 5, 1]]
        let l_garbage = Mat::from_rows(&[
            &[42.0f64, 0.0, 0.0], // garbage, non-1.0 diagonal
            &[3.0, -7.5, 0.0],    // garbage, non-1.0 diagonal
            &[2.0, 5.0, 0.0],     // garbage: zero diagonal (would be Singular if read)
        ]);

        // b = L * [1, 2, 3] with L's *implicit* unit diagonal:
        // row0: 1*1 = 1
        // row1: 3*1 + 1*2 = 5
        // row2: 2*1 + 5*2 + 1*3 = 15
        let b = [1.0, 5.0, 15.0];

        let x = trsv(
            l_garbage.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::Unit,
        )
        .unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1] = {}", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-10, "x[2] = {}", x[2]);

        // Same check for the transposed (backward substitution) path and the
        // upper-triangular paths, so both unrolled branches are covered.
        let u_garbage = Mat::from_rows(&[
            &[0.0f64, 3.0, 2.0], // garbage zero diagonal
            &[0.0, -1.0, 5.0],   // garbage diagonal
            &[0.0, 0.0, 999.0],  // garbage diagonal
        ]);
        // U (implicit unit diagonal) = [[1, 3, 2], [0, 1, 5], [0, 0, 1]]
        // b = U * [1, 2, 3]:
        // row0: 1 + 3*2 + 2*3 = 13
        // row1: 2 + 5*3 = 17
        // row2: 3
        let b_upper = [13.0, 17.0, 3.0];
        let x_upper = trsv(
            u_garbage.as_ref(),
            &b_upper,
            TriangularMode::Upper,
            TrsvTrans::NoTrans,
            TrsvDiag::Unit,
        )
        .unwrap();
        assert!((x_upper[0] - 1.0).abs() < 1e-10);
        assert!((x_upper[1] - 2.0).abs() < 1e-10);
        assert!((x_upper[2] - 3.0).abs() < 1e-10);

        // Transposed solve with unit diagonal must also ignore garbage.
        // L^T with implicit unit diagonal = [[1, 3, 2], [0, 1, 5], [0, 0, 1]]
        // Solve L^T * x = b where x = [1, 2, 3]:
        // row2: x2 = 3
        // row1: x1 + 5*x2 = b1 => b1 = 2 + 15 = 17 => x1 = b1 - 5*3 = 2
        // row0: x0 + 3*x1 + 2*x2 = b0 => b0 = 1 + 6 + 6 = 13 => x0 = 1
        let b_trans = [13.0, 17.0, 3.0];
        let x_trans = trsv(
            l_garbage.as_ref(),
            &b_trans,
            TriangularMode::Lower,
            TrsvTrans::Trans,
            TrsvDiag::Unit,
        )
        .unwrap();
        assert!((x_trans[0] - 1.0).abs() < 1e-10);
        assert!((x_trans[1] - 2.0).abs() < 1e-10);
        assert!((x_trans[2] - 3.0).abs() < 1e-10);
    }

    /// Regression test: large (blocked-path) unit-diagonal solve must also
    /// ignore garbage on the diagonal.
    #[test]
    fn test_trsv_unit_diagonal_ignores_garbage_blocked() {
        let n = 200; // forces the blocked algorithm (> TRSV_BLOCK_SIZE * 2)

        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            // Garbage, non-1.0, even zero-valued diagonal entries.
            a[(i, i)] = if i % 7 == 0 { 0.0 } else { (i as f64) * -3.3 };
            for j in 0..i {
                a[(i, j)] = 0.01 * (1 + (i + j) % 5) as f64;
            }
        }

        let x_expected: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.5).collect();

        // Compute b assuming an *implicit* unit diagonal (ignore a[(i,i)]).
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = x_expected[i]; // diagonal contributes 1.0 * x[i]
            for j in 0..i {
                sum += a[(i, j)] * x_expected[j];
            }
            b[i] = sum;
        }

        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::NoTrans,
            TrsvDiag::Unit,
        )
        .unwrap();

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

    /// Regression test: ConjTrans on a complex lower-triangular system.
    /// Verified by reconstructing A^H * x and comparing to the original b.
    #[test]
    fn test_trsv_complex_conjtrans() {
        // L = [[2+i, 0], [1-2i, 3+i]]
        let a = Mat::from_rows(&[&[c(2.0, 1.0), c(0.0, 0.0)], &[c(1.0, -2.0), c(3.0, 1.0)]]);

        // L^H = conj(L)^T = [[2-i, 1+2i], [0, 3-i]]
        // Pick a known solution x and derive b = L^H * x.
        let x_expected = [c(1.0, 2.0), c(-1.0, 0.5)];

        let l_h = [
            [a[(0, 0)].conj(), a[(1, 0)].conj()],
            [a[(0, 1)].conj(), a[(1, 1)].conj()],
        ];
        let b = [
            l_h[0][0] * x_expected[0] + l_h[0][1] * x_expected[1],
            l_h[1][0] * x_expected[0] + l_h[1][1] * x_expected[1],
        ];

        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Lower,
            TrsvTrans::ConjTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

        assert!(approx_eq_c(x[0], x_expected[0]), "x[0] = {}", x[0]);
        assert!(approx_eq_c(x[1], x_expected[1]), "x[1] = {}", x[1]);

        // Reconstruct A^H * x and compare to the original b.
        let reconstructed = [
            l_h[0][0] * x[0] + l_h[0][1] * x[1],
            l_h[1][0] * x[0] + l_h[1][1] * x[1],
        ];
        assert!(approx_eq_c(reconstructed[0], b[0]));
        assert!(approx_eq_c(reconstructed[1], b[1]));
    }

    /// Regression test: ConjTrans on a larger complex upper-triangular
    /// system, exercising the unrolled (4-wide) inner loop with conjugation.
    #[test]
    fn test_trsv_complex_conjtrans_upper_unrolled() {
        let n = 6;
        let mut a = Mat::<Complex64>::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = c(2.0 + i as f64, 1.0);
            for j in (i + 1)..n {
                a[(i, j)] = c(0.1 * (i + j) as f64, -0.2 * (j - i) as f64);
            }
        }

        let x_expected: Vec<Complex64> = (0..n).map(|i| c(1.0 + i as f64, -(i as f64))).collect();

        // b = A^H * x_expected, where A^H[i,j] = conj(A[j,i]).
        let mut b = vec![c(0.0, 0.0); n];
        for i in 0..n {
            let mut sum = c(0.0, 0.0);
            for j in 0..n {
                let a_h_ij = a[(j, i)].conj();
                sum += a_h_ij * x_expected[j];
            }
            b[i] = sum;
        }

        let x = trsv(
            a.as_ref(),
            &b,
            TriangularMode::Upper,
            TrsvTrans::ConjTrans,
            TrsvDiag::NonUnit,
        )
        .unwrap();

        for i in 0..n {
            assert!(
                approx_eq_c(x[i], x_expected[i]),
                "x[{}] = {}, expected {}",
                i,
                x[i],
                x_expected[i]
            );
        }

        // Reconstruct A^H * x and compare to b.
        let mut reconstructed = vec![c(0.0, 0.0); n];
        for i in 0..n {
            let mut sum = c(0.0, 0.0);
            for j in 0..n {
                let a_h_ij = a[(j, i)].conj();
                sum += a_h_ij * x[j];
            }
            reconstructed[i] = sum;
        }
        for i in 0..n {
            assert!(approx_eq_c(reconstructed[i], b[i]));
        }
    }
}
