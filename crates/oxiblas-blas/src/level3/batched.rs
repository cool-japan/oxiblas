//! Batched BLAS operations for processing multiple small matrices efficiently.
//!
//! This module provides batched variants of common BLAS operations, which are
//! critical for ML workloads such as batch normalization and multi-head attention.
//!
//! # Operations
//!
//! - [`gemm_batched`]: Batched general matrix-matrix multiplication
//! - [`gemm_strided_batched`]: Strided batched GEMM (matrices at regular offsets)
//! - [`axpy_batched`]: Batched vector addition (y\[i\] = alpha * x\[i\] + y\[i\])
//! - [`gemv_batched`]: Batched matrix-vector multiplication
//!
//! # Parallel Variants
//!
//! When the `parallel` feature is enabled, parallel variants process independent
//! batch elements concurrently using Rayon.

use crate::level1::axpy;
use crate::level2::{GemvTrans, gemv};
use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for batched BLAS operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchedError {
    /// Batch sizes of input arrays do not match.
    BatchSizeMismatch {
        /// Expected batch size.
        expected: usize,
        /// Actual batch size found.
        actual: usize,
    },
    /// Matrix dimensions are incompatible within a batch element.
    DimensionMismatch {
        /// Index of the batch element with the mismatch.
        batch_index: usize,
        /// Human-readable description of the mismatch.
        detail: &'static str,
    },
    /// The batch count is zero.
    EmptyBatch,
    /// A strided buffer is too small for the requested batch count.
    BufferTooSmall {
        /// Required buffer length.
        required: usize,
        /// Actual buffer length provided.
        actual: usize,
    },
    /// Stride value is too small to hold the matrix data without overlap.
    StrideTooSmall {
        /// Name of the parameter (e.g., "stride_a").
        param: &'static str,
        /// Minimum stride required.
        min_stride: usize,
        /// Actual stride provided.
        actual: usize,
    },
}

impl core::fmt::Display for BatchedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BatchSizeMismatch { expected, actual } => {
                write!(f, "Batch size mismatch: expected {expected}, got {actual}")
            }
            Self::DimensionMismatch {
                batch_index,
                detail,
            } => {
                write!(
                    f,
                    "Dimension mismatch at batch index {batch_index}: {detail}"
                )
            }
            Self::EmptyBatch => write!(f, "Batch count must be greater than zero"),
            Self::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "Buffer too small: need at least {required} elements, got {actual}"
                )
            }
            Self::StrideTooSmall {
                param,
                min_stride,
                actual,
            } => {
                write!(
                    f,
                    "Stride too small for {param}: minimum {min_stride}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for BatchedError {}

// ---------------------------------------------------------------------------
// Transpose enum for batched operations
// ---------------------------------------------------------------------------

/// Transpose operation selector for batched GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// No transpose: use matrix as-is.
    NoTrans,
    /// Transpose: use A^T.
    Trans,
}

// ---------------------------------------------------------------------------
// Batched GEMM (array-of-pointers style)
// ---------------------------------------------------------------------------

/// Batched GEMM: C\[i\] = alpha * op(A\[i\]) * op(B\[i\]) + beta * C\[i\]
///
/// Each batch element performs an independent GEMM. All matrices in a batch
/// may have different dimensions (though they must be pairwise compatible).
///
/// # Arguments
///
/// * `trans_a` - Transpose operation applied to each A\[i\]
/// * `trans_b` - Transpose operation applied to each B\[i\]
/// * `alpha` - Scalar multiplier for op(A) * op(B)
/// * `a_batch` - Slice of immutable matrix references (one per batch element)
/// * `b_batch` - Slice of immutable matrix references (one per batch element)
/// * `beta` - Scalar multiplier for C
/// * `c_batch` - Slice of mutable matrix references (one per batch element)
///
/// # Errors
///
/// Returns [`BatchedError`] if batch sizes are mismatched, if any batch element
/// has incompatible dimensions, or if the batch is empty.
pub fn gemm_batched<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a_batch: &[MatRef<'_, T>],
    b_batch: &[MatRef<'_, T>],
    beta: T,
    c_batch: &mut [MatMut<'_, T>],
) -> Result<(), BatchedError> {
    let batch_count = a_batch.len();

    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }
    if b_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: b_batch.len(),
        });
    }
    if c_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: c_batch.len(),
        });
    }

    // Validate dimensions for each batch element
    for i in 0..batch_count {
        validate_gemm_dims(trans_a, trans_b, &a_batch[i], &b_batch[i], &c_batch[i], i)?;
    }

    // Execute each batch element sequentially
    for i in 0..batch_count {
        execute_single_gemm(
            trans_a,
            trans_b,
            alpha,
            &a_batch[i],
            &b_batch[i],
            beta,
            &mut c_batch[i],
        );
    }

    Ok(())
}

/// Parallel batched GEMM: processes batch elements concurrently.
///
/// This is identical to [`gemm_batched`] but distributes batch elements
/// across threads using Rayon when the `parallel` feature is enabled.
///
/// # Errors
///
/// Returns [`BatchedError`] on dimension or batch-size mismatches.
#[cfg(feature = "parallel")]
pub fn gemm_batched_parallel<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a_batch: &[MatRef<'_, T>],
    b_batch: &[MatRef<'_, T>],
    beta: T,
    c_batch: &mut [MatMut<'_, T>],
) -> Result<(), BatchedError> {
    let batch_count = a_batch.len();

    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }
    if b_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: b_batch.len(),
        });
    }
    if c_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: c_batch.len(),
        });
    }

    for i in 0..batch_count {
        validate_gemm_dims(trans_a, trans_b, &a_batch[i], &b_batch[i], &c_batch[i], i)?;
    }

    // Each batch element writes to its own C matrix -- no aliasing.
    c_batch.par_iter_mut().enumerate().for_each(|(i, c_i)| {
        execute_single_gemm(trans_a, trans_b, alpha, &a_batch[i], &b_batch[i], beta, c_i);
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Strided Batched GEMM
// ---------------------------------------------------------------------------

/// Strided batched GEMM: matrices stored at regular offsets in contiguous buffers.
///
/// This is more memory-efficient than the array-of-pointers variant because
/// all batch elements reside in a single allocation per operand.
///
/// C\[i\] = alpha * op(A\[i\]) * op(B\[i\]) + beta * C\[i\]
///
/// where A\[i\] starts at `a[i * stride_a ..]`, etc.
///
/// # Arguments
///
/// * `trans_a` - Transpose operation for A matrices
/// * `trans_b` - Transpose operation for B matrices
/// * `m` - Number of rows of op(A) and C
/// * `n` - Number of columns of op(B) and C
/// * `k` - Number of columns of op(A) / rows of op(B)
/// * `alpha` - Scalar multiplier for A * B
/// * `a` - Buffer containing all A matrices
/// * `lda` - Leading dimension (row stride) of each A matrix
/// * `stride_a` - Element offset between consecutive A matrices
/// * `b` - Buffer containing all B matrices
/// * `ldb` - Leading dimension (row stride) of each B matrix
/// * `stride_b` - Element offset between consecutive B matrices
/// * `beta` - Scalar multiplier for C
/// * `c` - Buffer containing all C matrices (modified in place)
/// * `ldc` - Leading dimension (row stride) of each C matrix
/// * `stride_c` - Element offset between consecutive C matrices
/// * `batch_count` - Number of matrices in the batch
///
/// # Errors
///
/// Returns [`BatchedError`] if buffers are too small, strides are invalid,
/// or the batch count is zero.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided_batched<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    stride_a: usize,
    b: &[T],
    ldb: usize,
    stride_b: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
    stride_c: usize,
    batch_count: usize,
) -> Result<(), BatchedError> {
    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }

    // Compute physical dimensions of stored matrices
    let (a_rows, a_cols) = match trans_a {
        Transpose::NoTrans => (m, k),
        Transpose::Trans => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        Transpose::NoTrans => (k, n),
        Transpose::Trans => (n, k),
    };

    // Validate leading dimensions
    validate_leading_dim(lda, a_rows, "lda")?;
    validate_leading_dim(ldb, b_rows, "ldb")?;
    validate_leading_dim(ldc, m, "ldc")?;

    // Validate strides (must be large enough to hold one matrix)
    let min_stride_a = lda * a_cols;
    let min_stride_b = ldb * b_cols;
    let min_stride_c = ldc * n;

    if batch_count > 1 {
        if stride_a < min_stride_a {
            return Err(BatchedError::StrideTooSmall {
                param: "stride_a",
                min_stride: min_stride_a,
                actual: stride_a,
            });
        }
        if stride_b < min_stride_b {
            return Err(BatchedError::StrideTooSmall {
                param: "stride_b",
                min_stride: min_stride_b,
                actual: stride_b,
            });
        }
        if stride_c < min_stride_c {
            return Err(BatchedError::StrideTooSmall {
                param: "stride_c",
                min_stride: min_stride_c,
                actual: stride_c,
            });
        }
    }

    // Validate buffer sizes
    let required_a = if batch_count == 1 {
        min_stride_a
    } else {
        stride_a * (batch_count - 1) + min_stride_a
    };
    let required_b = if batch_count == 1 {
        min_stride_b
    } else {
        stride_b * (batch_count - 1) + min_stride_b
    };
    let required_c = if batch_count == 1 {
        min_stride_c
    } else {
        stride_c * (batch_count - 1) + min_stride_c
    };

    if a.len() < required_a {
        return Err(BatchedError::BufferTooSmall {
            required: required_a,
            actual: a.len(),
        });
    }
    if b.len() < required_b {
        return Err(BatchedError::BufferTooSmall {
            required: required_b,
            actual: b.len(),
        });
    }
    if c.len() < required_c {
        return Err(BatchedError::BufferTooSmall {
            required: required_c,
            actual: c.len(),
        });
    }

    for i in 0..batch_count {
        let a_offset = i * stride_a;
        let b_offset = i * stride_b;
        let c_offset = i * stride_c;

        // Create matrix views into the strided buffers.
        // SAFETY: We validated buffer sizes and strides above.
        let a_ref = MatRef::new(a[a_offset..].as_ptr(), a_rows, a_cols, lda);
        let b_ref = MatRef::new(b[b_offset..].as_ptr(), b_rows, b_cols, ldb);
        let mut c_mut = MatMut::new(c[c_offset..].as_mut_ptr(), m, n, ldc);

        execute_single_gemm(trans_a, trans_b, alpha, &a_ref, &b_ref, beta, &mut c_mut);
    }

    Ok(())
}

/// Parallel strided batched GEMM.
///
/// Same as [`gemm_strided_batched`] but distributes batch elements across
/// threads. Requires the `parallel` feature.
///
/// # Errors
///
/// Returns [`BatchedError`] on invalid parameters.
///
/// # Safety Notes
///
/// Each batch element writes to a non-overlapping region of `c` (guaranteed
/// by stride validation), so concurrent writes are safe.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided_batched_parallel<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    stride_a: usize,
    b: &[T],
    ldb: usize,
    stride_b: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
    stride_c: usize,
    batch_count: usize,
) -> Result<(), BatchedError> {
    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }

    let (a_rows, a_cols) = match trans_a {
        Transpose::NoTrans => (m, k),
        Transpose::Trans => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        Transpose::NoTrans => (k, n),
        Transpose::Trans => (n, k),
    };

    validate_leading_dim(lda, a_rows, "lda")?;
    validate_leading_dim(ldb, b_rows, "ldb")?;
    validate_leading_dim(ldc, m, "ldc")?;

    let min_stride_a = lda * a_cols;
    let min_stride_b = ldb * b_cols;
    let min_stride_c = ldc * n;

    if batch_count > 1 {
        if stride_a < min_stride_a {
            return Err(BatchedError::StrideTooSmall {
                param: "stride_a",
                min_stride: min_stride_a,
                actual: stride_a,
            });
        }
        if stride_b < min_stride_b {
            return Err(BatchedError::StrideTooSmall {
                param: "stride_b",
                min_stride: min_stride_b,
                actual: stride_b,
            });
        }
        if stride_c < min_stride_c {
            return Err(BatchedError::StrideTooSmall {
                param: "stride_c",
                min_stride: min_stride_c,
                actual: stride_c,
            });
        }
    }

    let required_a = if batch_count == 1 {
        min_stride_a
    } else {
        stride_a * (batch_count - 1) + min_stride_a
    };
    let required_b = if batch_count == 1 {
        min_stride_b
    } else {
        stride_b * (batch_count - 1) + min_stride_b
    };
    let required_c = if batch_count == 1 {
        min_stride_c
    } else {
        stride_c * (batch_count - 1) + min_stride_c
    };

    if a.len() < required_a {
        return Err(BatchedError::BufferTooSmall {
            required: required_a,
            actual: a.len(),
        });
    }
    if b.len() < required_b {
        return Err(BatchedError::BufferTooSmall {
            required: required_b,
            actual: b.len(),
        });
    }
    if c.len() < required_c {
        return Err(BatchedError::BufferTooSmall {
            required: required_c,
            actual: c.len(),
        });
    }

    // Use a raw pointer to allow concurrent mutable access to non-overlapping
    // regions. Each batch element writes to c[i*stride_c .. i*stride_c + ldc*n],
    // and we validated that stride_c >= ldc*n.
    let c_ptr = c.as_mut_ptr() as usize;

    (0..batch_count).into_par_iter().for_each(|i| {
        let a_offset = i * stride_a;
        let b_offset = i * stride_b;
        let c_offset = i * stride_c;

        let a_ref = MatRef::new(a[a_offset..].as_ptr(), a_rows, a_cols, lda);
        let b_ref = MatRef::new(b[b_offset..].as_ptr(), b_rows, b_cols, ldb);

        // SAFETY: Each iteration writes to a disjoint region of c.
        let mut c_mut = unsafe {
            let ptr = (c_ptr as *mut T).add(c_offset);
            MatMut::new(ptr, m, n, ldc)
        };

        execute_single_gemm(trans_a, trans_b, alpha, &a_ref, &b_ref, beta, &mut c_mut);
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Batched AXPY
// ---------------------------------------------------------------------------

/// Batched AXPY: y\[i\] = alpha * x\[i\] + y\[i\] for each batch element.
///
/// All vectors in a batch must have the same length.
///
/// # Errors
///
/// Returns [`BatchedError`] if batch sizes are mismatched, vectors have
/// incompatible lengths, or the batch is empty.
pub fn axpy_batched<T: Field>(
    alpha: T,
    x_batch: &[&[T]],
    y_batch: &mut [&mut [T]],
) -> Result<(), BatchedError> {
    let batch_count = x_batch.len();

    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }
    if y_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: y_batch.len(),
        });
    }

    for i in 0..batch_count {
        if x_batch[i].len() != y_batch[i].len() {
            return Err(BatchedError::DimensionMismatch {
                batch_index: i,
                detail: "x and y vectors must have the same length",
            });
        }
    }

    for i in 0..batch_count {
        axpy(alpha, x_batch[i], y_batch[i]);
    }

    Ok(())
}

/// Parallel batched AXPY.
///
/// # Errors
///
/// Returns [`BatchedError`] on dimension or batch-size mismatches.
#[cfg(feature = "parallel")]
pub fn axpy_batched_parallel<T: Field>(
    alpha: T,
    x_batch: &[&[T]],
    y_batch: &mut [&mut [T]],
) -> Result<(), BatchedError> {
    let batch_count = x_batch.len();

    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }
    if y_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: y_batch.len(),
        });
    }

    for i in 0..batch_count {
        if x_batch[i].len() != y_batch[i].len() {
            return Err(BatchedError::DimensionMismatch {
                batch_index: i,
                detail: "x and y vectors must have the same length",
            });
        }
    }

    y_batch.par_iter_mut().enumerate().for_each(|(i, y_i)| {
        axpy(alpha, x_batch[i], y_i);
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Batched GEMV
// ---------------------------------------------------------------------------

/// Batched GEMV: y\[i\] = alpha * op(A\[i\]) * x\[i\] + beta * y\[i\]
///
/// # Arguments
///
/// * `trans` - Whether to transpose each A\[i\]
/// * `alpha` - Scalar multiplier for A * x
/// * `a_batch` - Slice of matrix references
/// * `x_batch` - Slice of input vector references
/// * `beta` - Scalar multiplier for y
/// * `y_batch` - Slice of output vector references (modified in place)
///
/// # Errors
///
/// Returns [`BatchedError`] if batch sizes mismatch, vectors/matrices have
/// incompatible dimensions, or the batch is empty.
pub fn gemv_batched<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a_batch: &[MatRef<'_, T>],
    x_batch: &[&[T]],
    beta: T,
    y_batch: &mut [&mut [T]],
) -> Result<(), BatchedError> {
    let batch_count = a_batch.len();

    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }
    if x_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: x_batch.len(),
        });
    }
    if y_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: y_batch.len(),
        });
    }

    for i in 0..batch_count {
        validate_gemv_dims(trans, &a_batch[i], x_batch[i], y_batch[i], i)?;
    }

    for i in 0..batch_count {
        gemv(trans, alpha, a_batch[i], x_batch[i], beta, y_batch[i]);
    }

    Ok(())
}

/// Parallel batched GEMV.
///
/// # Errors
///
/// Returns [`BatchedError`] on dimension or batch-size mismatches.
#[cfg(feature = "parallel")]
pub fn gemv_batched_parallel<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a_batch: &[MatRef<'_, T>],
    x_batch: &[&[T]],
    beta: T,
    y_batch: &mut [&mut [T]],
) -> Result<(), BatchedError> {
    let batch_count = a_batch.len();

    if batch_count == 0 {
        return Err(BatchedError::EmptyBatch);
    }
    if x_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: x_batch.len(),
        });
    }
    if y_batch.len() != batch_count {
        return Err(BatchedError::BatchSizeMismatch {
            expected: batch_count,
            actual: y_batch.len(),
        });
    }

    for i in 0..batch_count {
        validate_gemv_dims(trans, &a_batch[i], x_batch[i], y_batch[i], i)?;
    }

    y_batch.par_iter_mut().enumerate().for_each(|(i, y_i)| {
        gemv(trans, alpha, a_batch[i], x_batch[i], beta, y_i);
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validates dimension compatibility for a single GEMM within a batch.
fn validate_gemm_dims<T: Field>(
    trans_a: Transpose,
    trans_b: Transpose,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &MatMut<'_, T>,
    batch_index: usize,
) -> Result<(), BatchedError> {
    let (m_a, k_a) = effective_dims(a.nrows(), a.ncols(), trans_a);
    let (k_b, n_b) = effective_dims(b.nrows(), b.ncols(), trans_b);

    if k_a != k_b {
        return Err(BatchedError::DimensionMismatch {
            batch_index,
            detail: "inner dimensions of op(A) and op(B) must match",
        });
    }
    if c.nrows() != m_a {
        return Err(BatchedError::DimensionMismatch {
            batch_index,
            detail: "C row count must equal op(A) row count",
        });
    }
    if c.ncols() != n_b {
        return Err(BatchedError::DimensionMismatch {
            batch_index,
            detail: "C column count must equal op(B) column count",
        });
    }

    Ok(())
}

/// Validates dimension compatibility for a single GEMV within a batch.
fn validate_gemv_dims<T: Field>(
    trans: GemvTrans,
    a: &MatRef<'_, T>,
    x: &[T],
    y: &[T],
    batch_index: usize,
) -> Result<(), BatchedError> {
    let (rows, cols) = match trans {
        GemvTrans::NoTrans => (a.nrows(), a.ncols()),
        GemvTrans::Trans | GemvTrans::ConjTrans => (a.ncols(), a.nrows()),
    };

    if x.len() != cols {
        return Err(BatchedError::DimensionMismatch {
            batch_index,
            detail: "x length must match columns of op(A)",
        });
    }
    if y.len() != rows {
        return Err(BatchedError::DimensionMismatch {
            batch_index,
            detail: "y length must match rows of op(A)",
        });
    }

    Ok(())
}

/// Returns (rows, cols) after applying the transpose operation.
#[inline]
fn effective_dims(nrows: usize, ncols: usize, trans: Transpose) -> (usize, usize) {
    match trans {
        Transpose::NoTrans => (nrows, ncols),
        Transpose::Trans => (ncols, nrows),
    }
}

/// Validates that a leading dimension is large enough.
fn validate_leading_dim(
    ld: usize,
    min_rows: usize,
    _param_name: &'static str,
) -> Result<(), BatchedError> {
    if ld < min_rows {
        return Err(BatchedError::StrideTooSmall {
            param: _param_name,
            min_stride: min_rows,
            actual: ld,
        });
    }
    Ok(())
}

/// Executes a single GEMM with optional transpose on A and/or B.
///
/// This creates temporary transposed copies when needed, then delegates
/// to the optimized `gemm` kernel.
fn execute_single_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    trans_a: Transpose,
    trans_b: Transpose,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) {
    match (trans_a, trans_b) {
        (Transpose::NoTrans, Transpose::NoTrans) => {
            gemm(alpha, *a, *b, beta, c.rb_mut());
        }
        (Transpose::Trans, Transpose::NoTrans) => {
            let a_t = transpose_to_mat(a);
            gemm(alpha, a_t.as_ref(), *b, beta, c.rb_mut());
        }
        (Transpose::NoTrans, Transpose::Trans) => {
            let b_t = transpose_to_mat(b);
            gemm(alpha, *a, b_t.as_ref(), beta, c.rb_mut());
        }
        (Transpose::Trans, Transpose::Trans) => {
            let a_t = transpose_to_mat(a);
            let b_t = transpose_to_mat(b);
            gemm(alpha, a_t.as_ref(), b_t.as_ref(), beta, c.rb_mut());
        }
    }
}

/// Creates a transposed copy of a matrix (columns become rows).
fn transpose_to_mat<T: Field + bytemuck::Zeroable>(src: &MatRef<'_, T>) -> Mat<T> {
    let m = src.nrows();
    let n = src.ncols();
    let mut dst = Mat::<T>::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            dst.set(j, i, src[(i, j)]);
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    // Helper: assert all elements of a matrix equal expected value within tolerance.
    fn assert_mat_approx(c: &Mat<f64>, expected: f64, tol: f64) {
        for i in 0..c.nrows() {
            for j in 0..c.ncols() {
                assert!(
                    (c[(i, j)] - expected).abs() < tol,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Basic batched GEMM (NoTrans, NoTrans)
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_no_trans() {
        let batch_size = 4;
        let m = 3;
        let k = 4;
        let n = 2;

        let a_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(m, k, 1.0)).collect();
        let b_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(k, n, 2.0)).collect();
        let mut c_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::zeros(m, n)).collect();

        let a_refs: Vec<MatRef<'_, f64>> = a_mats.iter().map(|m| m.as_ref()).collect();
        let b_refs: Vec<MatRef<'_, f64>> = b_mats.iter().map(|m| m.as_ref()).collect();
        let mut c_muts: Vec<MatMut<'_, f64>> = c_mats.iter_mut().map(|m| m.as_mut()).collect();

        let result = gemm_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &a_refs,
            &b_refs,
            0.0,
            &mut c_muts,
        );
        assert!(result.is_ok());

        // Each element: sum of k ones * 2 = k * 2 = 8
        drop(c_muts);
        for c in &c_mats {
            assert_mat_approx(c, 8.0, 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Batched GEMM with transpose on A
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_trans_a() {
        // A is k x m, op(A) = A^T is m x k
        let batch_size = 2;
        let m = 3;
        let k = 5;
        let n = 2;

        let a_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(k, m, 1.0)).collect();
        let b_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(k, n, 3.0)).collect();
        let mut c_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::zeros(m, n)).collect();

        let a_refs: Vec<MatRef<'_, f64>> = a_mats.iter().map(|m| m.as_ref()).collect();
        let b_refs: Vec<MatRef<'_, f64>> = b_mats.iter().map(|m| m.as_ref()).collect();
        let mut c_muts: Vec<MatMut<'_, f64>> = c_mats.iter_mut().map(|m| m.as_mut()).collect();

        let result = gemm_batched(
            Transpose::Trans,
            Transpose::NoTrans,
            1.0,
            &a_refs,
            &b_refs,
            0.0,
            &mut c_muts,
        );
        assert!(result.is_ok());

        // op(A) = A^T is m x k (all 1s), B is k x n (all 3s)
        // Each element = k * 1 * 3 = 15
        drop(c_muts);
        for c in &c_mats {
            assert_mat_approx(c, 15.0, 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: Batched GEMM with alpha and beta
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_alpha_beta() {
        let batch_size = 3;
        let n = 2;

        let a_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(n, n, 1.0)).collect();
        let b_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(n, n, 2.0)).collect();
        let mut c_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(n, n, 10.0)).collect();

        let a_refs: Vec<MatRef<'_, f64>> = a_mats.iter().map(|m| m.as_ref()).collect();
        let b_refs: Vec<MatRef<'_, f64>> = b_mats.iter().map(|m| m.as_ref()).collect();
        let mut c_muts: Vec<MatMut<'_, f64>> = c_mats.iter_mut().map(|m| m.as_mut()).collect();

        // C = 2 * A * B + 3 * C
        // A*B each elem = 2*1*2 = 4, then 2*4 + 3*10 = 8 + 30 = 38
        let result = gemm_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2.0,
            &a_refs,
            &b_refs,
            3.0,
            &mut c_muts,
        );
        assert!(result.is_ok());

        drop(c_muts);
        for c in &c_mats {
            assert_mat_approx(c, 38.0, 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Batched GEMM dimension mismatch error
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_dimension_mismatch() {
        let a_mat = Mat::<f64>::filled(3, 4, 1.0);
        let b_mat = Mat::<f64>::filled(5, 2, 1.0); // k mismatch: 4 != 5
        let mut c_mat = Mat::<f64>::zeros(3, 2);

        let result = gemm_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &[a_mat.as_ref()],
            &[b_mat.as_ref()],
            0.0,
            &mut [c_mat.as_mut()],
        );
        assert!(matches!(
            result,
            Err(BatchedError::DimensionMismatch { batch_index: 0, .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 5: Batch size mismatch error
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_batch_size_mismatch() {
        let a1 = Mat::<f64>::filled(2, 2, 1.0);
        let a2 = Mat::<f64>::filled(2, 2, 1.0);
        let b1 = Mat::<f64>::filled(2, 2, 1.0);
        let mut c1 = Mat::<f64>::zeros(2, 2);

        let result = gemm_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &[a1.as_ref(), a2.as_ref()],
            &[b1.as_ref()], // mismatch: 2 vs 1
            0.0,
            &mut [c1.as_mut()],
        );
        assert!(matches!(
            result,
            Err(BatchedError::BatchSizeMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 6: Empty batch error
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_empty() {
        let result = gemm_batched::<f64>(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0,
            &[],
            &[],
            0.0,
            &mut [],
        );
        assert!(matches!(result, Err(BatchedError::EmptyBatch)));
    }

    // -----------------------------------------------------------------------
    // Test 7: Strided batched GEMM
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_strided_batched_basic() {
        let m = 2;
        let n = 2;
        let k = 3;
        let batch_count = 4;
        let lda = m; // column-major row stride
        let ldb = k;
        let ldc = m;
        let stride_a = lda * k;
        let stride_b = ldb * n;
        let stride_c = ldc * n;

        // All A matrices: filled with 1.0
        let a = vec![1.0f64; stride_a * batch_count];
        // All B matrices: filled with 2.0
        let b = vec![2.0f64; stride_b * batch_count];
        // C matrices: zeros
        let mut c = vec![0.0f64; stride_c * batch_count];

        let result = gemm_strided_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            lda,
            stride_a,
            &b,
            ldb,
            stride_b,
            0.0,
            &mut c,
            ldc,
            stride_c,
            batch_count,
        );
        assert!(result.is_ok());

        // Each element of C = k * 1.0 * 2.0 = 6.0
        for batch in 0..batch_count {
            for col in 0..n {
                for row in 0..m {
                    let idx = batch * stride_c + col * ldc + row;
                    assert!(
                        (c[idx] - 6.0).abs() < 1e-10,
                        "batch={batch}, row={row}, col={col}: got {}, expected 6.0",
                        c[idx],
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: Strided batched GEMM buffer too small
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_strided_batched_buffer_too_small() {
        let m = 2;
        let n = 2;
        let k = 2;
        let lda = m;
        let ldb = k;
        let ldc = m;
        let stride_a = lda * k;
        let stride_b = ldb * n;
        let stride_c = ldc * n;

        let a = vec![1.0f64; stride_a]; // only 1 batch worth
        let b = vec![1.0f64; stride_b * 3];
        let mut c = vec![0.0f64; stride_c * 3];

        let result = gemm_strided_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            lda,
            stride_a,
            &b,
            ldb,
            stride_b,
            0.0,
            &mut c,
            ldc,
            stride_c,
            3, // but a only holds 1
        );
        assert!(matches!(result, Err(BatchedError::BufferTooSmall { .. })));
    }

    // -----------------------------------------------------------------------
    // Test 9: Batched AXPY
    // -----------------------------------------------------------------------
    #[test]
    fn test_axpy_batched_basic() {
        let batch_size = 5;
        let vec_len = 10;

        let x_data: Vec<Vec<f64>> = (0..batch_size).map(|_| vec![1.0; vec_len]).collect();
        let mut y_data: Vec<Vec<f64>> = (0..batch_size).map(|_| vec![2.0; vec_len]).collect();

        let x_refs: Vec<&[f64]> = x_data.iter().map(|v| v.as_slice()).collect();
        let mut y_refs: Vec<&mut [f64]> = y_data.iter_mut().map(|v| v.as_mut_slice()).collect();

        // y = 3 * x + y = 3*1 + 2 = 5
        let result = axpy_batched(3.0, &x_refs, &mut y_refs);
        assert!(result.is_ok());

        drop(y_refs);
        for y_vec in &y_data {
            for &val in y_vec {
                assert!((val - 5.0).abs() < 1e-10);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: Batched GEMV
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemv_batched_basic() {
        let batch_size = 3;
        let m = 4;
        let n = 3;

        let a_mats: Vec<Mat<f64>> = (0..batch_size).map(|_| Mat::filled(m, n, 1.0)).collect();
        let x_data: Vec<Vec<f64>> = (0..batch_size).map(|_| vec![2.0; n]).collect();
        let mut y_data: Vec<Vec<f64>> = (0..batch_size).map(|_| vec![0.0; m]).collect();

        let a_refs: Vec<MatRef<'_, f64>> = a_mats.iter().map(|m| m.as_ref()).collect();
        let x_refs: Vec<&[f64]> = x_data.iter().map(|v| v.as_slice()).collect();
        let mut y_refs: Vec<&mut [f64]> = y_data.iter_mut().map(|v| v.as_mut_slice()).collect();

        // y = 1.0 * A * x + 0.0 * y, each elem = n * 1.0 * 2.0 = 6.0
        let result = gemv_batched(GemvTrans::NoTrans, 1.0, &a_refs, &x_refs, 0.0, &mut y_refs);
        assert!(result.is_ok());

        drop(y_refs);
        for y_vec in &y_data {
            for &val in y_vec {
                assert!((val - 6.0).abs() < 1e-10, "got {val}, expected 6.0",);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 11: Batched GEMM with f32
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_f32() {
        let batch_size = 2;
        let n = 3;

        let a_mats: Vec<Mat<f32>> = (0..batch_size).map(|_| Mat::filled(n, n, 1.0f32)).collect();
        let b_mats: Vec<Mat<f32>> = (0..batch_size).map(|_| Mat::filled(n, n, 2.0f32)).collect();
        let mut c_mats: Vec<Mat<f32>> = (0..batch_size).map(|_| Mat::zeros(n, n)).collect();

        let a_refs: Vec<MatRef<'_, f32>> = a_mats.iter().map(|m| m.as_ref()).collect();
        let b_refs: Vec<MatRef<'_, f32>> = b_mats.iter().map(|m| m.as_ref()).collect();
        let mut c_muts: Vec<MatMut<'_, f32>> = c_mats.iter_mut().map(|m| m.as_mut()).collect();

        let result = gemm_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0f32,
            &a_refs,
            &b_refs,
            0.0f32,
            &mut c_muts,
        );
        assert!(result.is_ok());

        // Each element = n * 1.0 * 2.0 = 6.0
        drop(c_muts);
        for c in &c_mats {
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (c[(i, j)] - 6.0f32).abs() < 1e-5,
                        "c[{i},{j}] = {}, expected 6.0",
                        c[(i, j)],
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 12: Batched GEMM with both transposes
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_batched_trans_a_trans_b() {
        // A stored as k x m, B stored as n x k
        // op(A) = A^T: m x k, op(B) = B^T: k x n
        let m = 2;
        let k = 3;
        let n = 4;

        let a_mat = Mat::<f64>::filled(k, m, 1.0);
        let b_mat = Mat::<f64>::filled(n, k, 2.0);
        let mut c_mat = Mat::<f64>::zeros(m, n);

        let result = gemm_batched(
            Transpose::Trans,
            Transpose::Trans,
            1.0,
            &[a_mat.as_ref()],
            &[b_mat.as_ref()],
            0.0,
            &mut [c_mat.as_mut()],
        );
        assert!(result.is_ok());

        // Each element = k * 1 * 2 = 6.0
        assert_mat_approx(&c_mat, 6.0, 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test 13: Batched AXPY dimension mismatch
    // -----------------------------------------------------------------------
    #[test]
    fn test_axpy_batched_dim_mismatch() {
        let x = vec![1.0f64; 5];
        let mut y = vec![2.0f64; 3]; // mismatch

        let result = axpy_batched(1.0, &[x.as_slice()], &mut [y.as_mut_slice()]);
        assert!(matches!(
            result,
            Err(BatchedError::DimensionMismatch { batch_index: 0, .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 14: Strided batched GEMM with transpose
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_strided_batched_trans_a() {
        // A stored as k x m (will be transposed to m x k)
        let m = 2;
        let n = 2;
        let k = 3;
        let batch_count = 2;
        let lda = k; // physical rows = k
        let ldb = k; // physical rows = k
        let ldc = m;
        let stride_a = lda * m; // physical: k rows, m cols
        let stride_b = ldb * n;
        let stride_c = ldc * n;

        let a = vec![1.0f64; stride_a * batch_count];
        let b = vec![2.0f64; stride_b * batch_count];
        let mut c = vec![0.0f64; stride_c * batch_count];

        let result = gemm_strided_batched(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            lda,
            stride_a,
            &b,
            ldb,
            stride_b,
            0.0,
            &mut c,
            ldc,
            stride_c,
            batch_count,
        );
        assert!(result.is_ok());

        // Each element = k * 1 * 2 = 6.0
        for batch in 0..batch_count {
            for col in 0..n {
                for row in 0..m {
                    let idx = batch * stride_c + col * ldc + row;
                    assert!(
                        (c[idx] - 6.0).abs() < 1e-10,
                        "batch={batch}, row={row}, col={col}: got {}, expected 6.0",
                        c[idx],
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: Batched GEMV with transpose
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemv_batched_trans() {
        let m = 4;
        let n = 3;

        let a_mat = Mat::<f64>::filled(m, n, 1.0);
        let x = vec![2.0f64; m]; // x length = rows of A = m
        let mut y = vec![0.0f64; n]; // y length = cols of A = n

        let result = gemv_batched(
            GemvTrans::Trans,
            1.0,
            &[a_mat.as_ref()],
            &[x.as_slice()],
            0.0,
            &mut [y.as_mut_slice()],
        );
        assert!(result.is_ok());

        // y = A^T * x, each elem of y = m * 1 * 2 = 8.0
        for &val in &y {
            assert!((val - 8.0).abs() < 1e-10, "got {val}, expected 8.0",);
        }
    }

    // -----------------------------------------------------------------------
    // Test 16: Strided batched stride-too-small error
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemm_strided_batched_stride_too_small() {
        let m = 4;
        let n = 4;
        let k = 4;
        let lda = m;
        let ldb = k;
        let ldc = m;

        // stride_a is too small (less than lda * k)
        let stride_a = 1;
        let stride_b = ldb * n;
        let stride_c = ldc * n;

        let a = vec![1.0f64; 100];
        let b = vec![1.0f64; 100];
        let mut c = vec![0.0f64; 100];

        let result = gemm_strided_batched(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            lda,
            stride_a,
            &b,
            ldb,
            stride_b,
            0.0,
            &mut c,
            ldc,
            stride_c,
            2,
        );
        assert!(matches!(
            result,
            Err(BatchedError::StrideTooSmall {
                param: "stride_a",
                ..
            })
        ));
    }
}
