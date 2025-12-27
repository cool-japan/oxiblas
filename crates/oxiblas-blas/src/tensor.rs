//! Tensor contraction operations.
//!
//! Provides Einstein summation (einsum) style tensor contractions and
//! multi-dimensional array operations, extending BLAS to higher dimensions.
//!
//! # Optimization
//!
//! For larger matrices, tensor operations use optimized GEMM kernels when
//! available (f32, f64), providing significant performance improvements.

use crate::level3::gemm::gemm;
use crate::level3::gemm_kernel::GemmKernel;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

/// Error type for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// Dimension mismatch in tensor contraction.
    DimensionMismatch,
    /// Invalid index specification.
    InvalidIndices,
    /// Unsupported operation.
    Unsupported,
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch => write!(f, "Tensor dimensions do not match"),
            Self::InvalidIndices => write!(f, "Invalid tensor index specification"),
            Self::Unsupported => write!(f, "Unsupported tensor operation"),
        }
    }
}

impl std::error::Error for TensorError {}

/// Simple 3D tensor (rank-3 array) in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor3<T: Field> {
    data: Vec<T>,
    dim0: usize,
    dim1: usize,
    dim2: usize,
}

impl<T: Field> Tensor3<T> {
    /// Creates a new tensor filled with zeros.
    #[must_use]
    pub fn zeros(dim0: usize, dim1: usize, dim2: usize) -> Self {
        Self {
            data: vec![T::zero(); dim0 * dim1 * dim2],
            dim0,
            dim1,
            dim2,
        }
    }

    /// Creates a new tensor from a flat data vector.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != dim0 * dim1 * dim2`.
    #[must_use]
    pub fn from_data(data: Vec<T>, dim0: usize, dim1: usize, dim2: usize) -> Self {
        assert_eq!(data.len(), dim0 * dim1 * dim2);
        Self {
            data,
            dim0,
            dim1,
            dim2,
        }
    }

    /// Returns the dimensions of the tensor.
    #[must_use]
    pub const fn dims(&self) -> (usize, usize, usize) {
        (self.dim0, self.dim1, self.dim2)
    }

    /// Gets a reference to an element.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize) -> T {
        assert!(i < self.dim0 && j < self.dim1 && k < self.dim2);
        self.data[i * (self.dim1 * self.dim2) + j * self.dim2 + k]
    }

    /// Sets an element.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, i: usize, j: usize, k: usize, value: T) {
        assert!(i < self.dim0 && j < self.dim1 && k < self.dim2);
        self.data[i * (self.dim1 * self.dim2) + j * self.dim2 + k] = value;
    }

    /// Returns a reference to the underlying data.
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Tensor contraction: C\[i,k\] = `Σ_j` A\[i,j\] * B\[j,k\]
///
/// This is the 2D matrix multiplication extended to work with general
/// tensor contractions on specified indices.
///
/// # Arguments
///
/// * `a` - First tensor (shape: [m, n])
/// * `b` - Second tensor (shape: [n, k])
///
/// # Returns
///
/// Resulting tensor of shape [m, k].
///
/// # Errors
///
/// Returns `TensorError::DimensionMismatch` if inner dimensions don't match.
pub fn contract_2d<T: Field + GemmKernel + bytemuck::Zeroable>(
    a: &[T],
    a_dims: (usize, usize),
    b: &[T],
    b_dims: (usize, usize),
) -> Result<Vec<T>, TensorError> {
    let (m, n) = a_dims;
    let (n2, k) = b_dims;

    if n != n2 {
        return Err(TensorError::DimensionMismatch);
    }

    // Use optimized GEMM for larger matrices (threshold: 32)
    const GEMM_THRESHOLD: usize = 32;
    if m >= GEMM_THRESHOLD || n >= GEMM_THRESHOLD || k >= GEMM_THRESHOLD {
        // Row-major to column-major conversion using transpose interpretation:
        // A row-major (m×n) = A^T column-major (n×m)
        // B row-major (n×k) = B^T column-major (k×n)
        // C row-major (m×k) = C^T column-major (k×m)
        //
        // C = A * B in row-major
        // C^T = B^T * A^T in column-major (since (AB)^T = B^T A^T)
        //
        // So gemm(B^T, A^T) -> C^T gives us C in row-major!

        // Create views: B^T (k×n) and A^T (n×m)
        let b_t = MatRef::new(b.as_ptr(), k, n, k);
        let a_t = MatRef::new(a.as_ptr(), n, m, n);

        // Result: C^T (k×m)
        let mut c = vec![T::zero(); m * k];
        let c_t = MatMut::new(c.as_mut_ptr(), k, m, k);

        // Compute C^T = B^T * A^T
        gemm(T::one(), b_t, a_t, T::zero(), c_t);

        Ok(c)
    } else {
        // Naive implementation for small matrices
        let mut c = vec![T::zero(); m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = T::zero();
                for p in 0..n {
                    sum += a[i * n + p] * b[p * k + j];
                }
                c[i * k + j] = sum;
            }
        }
        Ok(c)
    }
}

/// Tensor contraction: C\[i,k,l\] = `Σ_j` A\[i,j,k\] * B\[j,l\]
///
/// Contracts the second index of a 3D tensor with the first index of a 2D tensor.
///
/// # Errors
///
/// Returns `TensorError::DimensionMismatch` if contraction dimensions don't match.
pub fn contract_3d_2d<T: Field>(
    a: &Tensor3<T>,
    b: &[T],
    b_dims: (usize, usize),
) -> Result<Tensor3<T>, TensorError> {
    let (dim0, dim1, dim2) = a.dims();
    let (n, l) = b_dims;

    if dim1 != n {
        return Err(TensorError::DimensionMismatch);
    }

    let mut c = Tensor3::zeros(dim0, dim2, l);

    // C[i,k,l] = Σ_j A[i,j,k] * B[j,l]
    for i in 0..dim0 {
        for k in 0..dim2 {
            for l_idx in 0..l {
                let mut sum = T::zero();
                for j in 0..dim1 {
                    sum += a.get(i, j, k) * b[j * l + l_idx];
                }
                c.set(i, k, l_idx, sum);
            }
        }
    }

    Ok(c)
}

/// Outer product of two vectors to create a matrix.
///
/// C\[i,j\] = a\[i\] * b\[j\]
///
/// # Example
///
/// ```
/// use oxiblas_blas::tensor::outer_product;
///
/// let a = [1.0f64, 2.0, 3.0];
/// let b = [4.0f64, 5.0];
///
/// let c = outer_product(&a, &b);
/// // c = [[4, 5], [8, 10], [12, 15]]
/// ```
#[must_use]
pub fn outer_product<T: Field>(a: &[T], b: &[T]) -> Vec<T> {
    let m = a.len();
    let n = b.len();
    let mut c = vec![T::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = a[i] * b[j];
        }
    }

    c
}

/// Batched matrix multiplication: C\[b,i,j\] = `Σ_k` A\[b,i,k\] * B\[b,k,j\]
///
/// Performs matrix multiplication independently for each batch.
///
/// # Arguments
///
/// * `a` - First tensor (shape: [batch, m, k])
/// * `b` - Second tensor (shape: [batch, k, n])
///
/// # Errors
///
/// Returns error if dimensions are incompatible.
pub fn batched_matmul<T: Field + GemmKernel + bytemuck::Zeroable>(
    a: &Tensor3<T>,
    b: &Tensor3<T>,
) -> Result<Tensor3<T>, TensorError> {
    let (batch_a, m, k) = a.dims();
    let (batch_b, k2, n) = b.dims();

    if batch_a != batch_b || k != k2 {
        return Err(TensorError::DimensionMismatch);
    }

    let mut c: Tensor3<T> = Tensor3::zeros(batch_a, m, n);

    // Use optimized GEMM for larger matrices (threshold: 16)
    const GEMM_THRESHOLD: usize = 16;
    if m >= GEMM_THRESHOLD || k >= GEMM_THRESHOLD || n >= GEMM_THRESHOLD {
        // For each batch, use optimized GEMM
        // Same transpose trick as contract_2d:
        // C = A * B in row-major becomes C^T = B^T * A^T in column-major

        for b_idx in 0..batch_a {
            let a_offset = b_idx * m * k;
            let b_offset = b_idx * k * n;
            let c_offset = b_idx * m * n;

            // Create views: B^T (n×k) and A^T (k×m)
            let b_t = MatRef::new(unsafe { b.data().as_ptr().add(b_offset) }, n, k, n);
            let a_t = MatRef::new(unsafe { a.data().as_ptr().add(a_offset) }, k, m, k);

            // Result: C^T (n×m)
            let c_t = MatMut::new(unsafe { c.data_mut().as_mut_ptr().add(c_offset) }, n, m, n);

            // Compute C^T = B^T * A^T
            gemm(T::one(), b_t, a_t, T::zero(), c_t);
        }
    } else {
        // Naive implementation for small matrices
        for b_idx in 0..batch_a {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for p in 0..k {
                        sum += a.get(b_idx, i, p) * b.get(b_idx, p, j);
                    }
                    c.set(b_idx, i, j, sum);
                }
            }
        }
    }

    Ok(c)
}

/// Einstein summation notation (einsum) for tensor contractions.
///
/// Supported operations:
/// - "ij,jk->ik": Matrix multiplication
/// - "ik,kj->ij": Alternative matrix multiplication notation
/// - "ij,j->i": Matrix-vector multiplication
/// - "i,j->ij": Outer product
/// - "ij,ik->ijk": Outer product to 3D tensor
/// - "ii->i": Diagonal extraction
/// - "ii->": Trace (sum of diagonal)
/// - "ij->ji": Transpose (2D)
/// - "ijk->ikj": Tensor transpose (swap middle and last axes)
/// - "ijk->jik": Tensor transpose (swap first two axes)
/// - "ijk->kji": Tensor transpose (reverse all axes)
/// - "ij,ij->ij": Element-wise multiplication (Hadamard product)
/// - "i,i->i": Element-wise vector multiplication
/// - "ij,ij->": Inner product (Frobenius)
/// - "i,i->": Dot product
/// - "ijk,ijk->": 3D Frobenius inner product
/// - "ijk,kl->ijl": 3D tensor-matrix contraction
/// - "ij->i": Row sums
/// - "ij->j": Column sums
/// - "ij->": Total sum
/// - "ijk->ij": Sum over last axis
/// - "ijk->jk": Sum over first axis
///
/// Note: This is a simplified einsum that supports common patterns.
pub fn einsum<T: Field + GemmKernel + bytemuck::Zeroable>(
    notation: &str,
    a: &[T],
    a_shape: &[usize],
    b: Option<(&[T], &[usize])>,
) -> Result<Vec<T>, TensorError> {
    match notation {
        "ij,jk->ik" => {
            // Matrix multiplication
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 2 || b_shape.len() != 2 {
                    return Err(TensorError::InvalidIndices);
                }
                contract_2d(
                    a,
                    (a_shape[0], a_shape[1]),
                    b_data,
                    (b_shape[0], b_shape[1]),
                )
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ij,j->i" => {
            // Matrix-vector multiplication
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 2 || b_shape.len() != 1 {
                    return Err(TensorError::InvalidIndices);
                }
                let (m, n) = (a_shape[0], a_shape[1]);
                if b_shape[0] != n {
                    return Err(TensorError::DimensionMismatch);
                }
                let mut result = vec![T::zero(); m];
                for i in 0..m {
                    let mut sum = T::zero();
                    for j in 0..n {
                        sum += a[i * n + j] * b_data[j];
                    }
                    result[i] = sum;
                }
                Ok(result)
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "i,j->ij" => {
            // Outer product
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 1 || b_shape.len() != 1 {
                    return Err(TensorError::InvalidIndices);
                }
                Ok(outer_product(a, b_data))
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ii->i" => {
            // Diagonal extraction
            if a_shape.len() != 2 {
                return Err(TensorError::InvalidIndices);
            }
            let (m, n) = (a_shape[0], a_shape[1]);
            let diag_len = m.min(n);
            let mut result = vec![T::zero(); diag_len];
            for i in 0..diag_len {
                result[i] = a[i * n + i];
            }
            Ok(result)
        }
        "ij->ji" => {
            // Transpose
            if a_shape.len() != 2 {
                return Err(TensorError::InvalidIndices);
            }
            let (m, n) = (a_shape[0], a_shape[1]);
            let mut result = vec![T::zero(); m * n];
            for i in 0..m {
                for j in 0..n {
                    result[j * m + i] = a[i * n + j];
                }
            }
            Ok(result)
        }
        "ij,ij->ij" => {
            // Element-wise multiplication (Hadamard product)
            if let Some((b_data, b_shape)) = b {
                if a_shape != b_shape {
                    return Err(TensorError::DimensionMismatch);
                }
                let result: Vec<T> = a.iter().zip(b_data.iter()).map(|(&x, &y)| x * y).collect();
                Ok(result)
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ij,ij->" => {
            // Inner product (Frobenius)
            if let Some((b_data, b_shape)) = b {
                if a_shape != b_shape {
                    return Err(TensorError::DimensionMismatch);
                }
                let mut sum = T::zero();
                for (&x, &y) in a.iter().zip(b_data.iter()) {
                    sum += x * y;
                }
                Ok(vec![sum])
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ijk->ikj" => {
            // Tensor transpose (swap middle and last axes)
            if a_shape.len() != 3 {
                return Err(TensorError::InvalidIndices);
            }
            let (d0, d1, d2) = (a_shape[0], a_shape[1], a_shape[2]);
            let mut result = vec![T::zero(); d0 * d1 * d2];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        let src_idx = i * (d1 * d2) + j * d2 + k;
                        let dst_idx = i * (d2 * d1) + k * d1 + j;
                        result[dst_idx] = a[src_idx];
                    }
                }
            }
            Ok(result)
        }
        "ij->i" => {
            // Sum along axis 1 (row sums)
            if a_shape.len() != 2 {
                return Err(TensorError::InvalidIndices);
            }
            let (m, n) = (a_shape[0], a_shape[1]);
            let mut result = vec![T::zero(); m];
            for i in 0..m {
                let mut sum = T::zero();
                for j in 0..n {
                    sum += a[i * n + j];
                }
                result[i] = sum;
            }
            Ok(result)
        }
        "ij->j" => {
            // Sum along axis 0 (column sums)
            if a_shape.len() != 2 {
                return Err(TensorError::InvalidIndices);
            }
            let (m, n) = (a_shape[0], a_shape[1]);
            let mut result = vec![T::zero(); n];
            for i in 0..m {
                for j in 0..n {
                    result[j] += a[i * n + j];
                }
            }
            Ok(result)
        }
        "ij->" => {
            // Sum all elements
            if a_shape.len() != 2 {
                return Err(TensorError::InvalidIndices);
            }
            let mut sum = T::zero();
            for &val in a {
                sum += val;
            }
            Ok(vec![sum])
        }
        "ii->" => {
            // Trace (sum of diagonal elements)
            if a_shape.len() != 2 {
                return Err(TensorError::InvalidIndices);
            }
            let (m, n) = (a_shape[0], a_shape[1]);
            let diag_len = m.min(n);
            let mut sum = T::zero();
            for i in 0..diag_len {
                sum += a[i * n + i];
            }
            Ok(vec![sum])
        }
        "ik,kj->ij" => {
            // Alternative matrix multiplication notation
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 2 || b_shape.len() != 2 {
                    return Err(TensorError::InvalidIndices);
                }
                contract_2d(
                    a,
                    (a_shape[0], a_shape[1]),
                    b_data,
                    (b_shape[0], b_shape[1]),
                )
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "i,i->" => {
            // Dot product
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 1 || b_shape.len() != 1 {
                    return Err(TensorError::InvalidIndices);
                }
                if a_shape[0] != b_shape[0] {
                    return Err(TensorError::DimensionMismatch);
                }
                let mut sum = T::zero();
                for i in 0..a_shape[0] {
                    sum += a[i] * b_data[i];
                }
                Ok(vec![sum])
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "i,i->i" => {
            // Element-wise vector multiplication
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 1 || b_shape.len() != 1 {
                    return Err(TensorError::InvalidIndices);
                }
                if a_shape[0] != b_shape[0] {
                    return Err(TensorError::DimensionMismatch);
                }
                let result: Vec<T> = a.iter().zip(b_data.iter()).map(|(&x, &y)| x * y).collect();
                Ok(result)
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ij,ik->ijk" => {
            // Outer product to 3D tensor
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 2 || b_shape.len() != 2 {
                    return Err(TensorError::InvalidIndices);
                }
                let (m, n) = (a_shape[0], a_shape[1]);
                let (m2, k) = (b_shape[0], b_shape[1]);
                if m != m2 {
                    return Err(TensorError::DimensionMismatch);
                }
                let mut result = vec![T::zero(); m * n * k];
                for i in 0..m {
                    for j in 0..n {
                        for p in 0..k {
                            let idx = i * (n * k) + j * k + p;
                            result[idx] = a[i * n + j] * b_data[i * k + p];
                        }
                    }
                }
                Ok(result)
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ijk->jik" => {
            // Tensor transpose (swap first two axes)
            if a_shape.len() != 3 {
                return Err(TensorError::InvalidIndices);
            }
            let (d0, d1, d2) = (a_shape[0], a_shape[1], a_shape[2]);
            let mut result = vec![T::zero(); d0 * d1 * d2];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        let src_idx = i * (d1 * d2) + j * d2 + k;
                        let dst_idx = j * (d0 * d2) + i * d2 + k;
                        result[dst_idx] = a[src_idx];
                    }
                }
            }
            Ok(result)
        }
        "ijk->kji" => {
            // Tensor transpose (reverse all axes)
            if a_shape.len() != 3 {
                return Err(TensorError::InvalidIndices);
            }
            let (d0, d1, d2) = (a_shape[0], a_shape[1], a_shape[2]);
            let mut result = vec![T::zero(); d0 * d1 * d2];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        let src_idx = i * (d1 * d2) + j * d2 + k;
                        let dst_idx = k * (d1 * d0) + j * d0 + i;
                        result[dst_idx] = a[src_idx];
                    }
                }
            }
            Ok(result)
        }
        "ijk,ijk->" => {
            // 3D Frobenius inner product
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 3 || b_shape.len() != 3 {
                    return Err(TensorError::InvalidIndices);
                }
                if a_shape != b_shape {
                    return Err(TensorError::DimensionMismatch);
                }
                let mut sum = T::zero();
                for (&x, &y) in a.iter().zip(b_data.iter()) {
                    sum += x * y;
                }
                Ok(vec![sum])
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ijk,kl->ijl" => {
            // 3D tensor-matrix contraction
            if let Some((b_data, b_shape)) = b {
                if a_shape.len() != 3 || b_shape.len() != 2 {
                    return Err(TensorError::InvalidIndices);
                }
                let (d0, d1, d2) = (a_shape[0], a_shape[1], a_shape[2]);
                let (k, l) = (b_shape[0], b_shape[1]);
                if d2 != k {
                    return Err(TensorError::DimensionMismatch);
                }
                let mut result = vec![T::zero(); d0 * d1 * l];
                for i in 0..d0 {
                    for j in 0..d1 {
                        for p in 0..l {
                            let mut sum = T::zero();
                            for q in 0..d2 {
                                let a_idx = i * (d1 * d2) + j * d2 + q;
                                let b_idx = q * l + p;
                                sum += a[a_idx] * b_data[b_idx];
                            }
                            let result_idx = i * (d1 * l) + j * l + p;
                            result[result_idx] = sum;
                        }
                    }
                }
                Ok(result)
            } else {
                Err(TensorError::InvalidIndices)
            }
        }
        "ijk->ij" => {
            // Sum over last axis
            if a_shape.len() != 3 {
                return Err(TensorError::InvalidIndices);
            }
            let (d0, d1, d2) = (a_shape[0], a_shape[1], a_shape[2]);
            let mut result = vec![T::zero(); d0 * d1];
            for i in 0..d0 {
                for j in 0..d1 {
                    let mut sum = T::zero();
                    for k in 0..d2 {
                        sum += a[i * (d1 * d2) + j * d2 + k];
                    }
                    result[i * d1 + j] = sum;
                }
            }
            Ok(result)
        }
        "ijk->jk" => {
            // Sum over first axis
            if a_shape.len() != 3 {
                return Err(TensorError::InvalidIndices);
            }
            let (d0, d1, d2) = (a_shape[0], a_shape[1], a_shape[2]);
            let mut result = vec![T::zero(); d1 * d2];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        result[j * d2 + k] += a[i * (d1 * d2) + j * d2 + k];
                    }
                }
            }
            Ok(result)
        }
        _ => Err(TensorError::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor3_basic() {
        let mut t = Tensor3::zeros(2, 3, 4);
        t.set(0, 1, 2, 5.0);

        assert_eq!(t.get(0, 1, 2), 5.0);
        assert_eq!(t.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_contract_2d() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let c = contract_2d(&a, (2, 2), &b, (2, 2)).unwrap();

        // [[1,2], [3,4]] * [[5,6], [7,8]] = [[19,22], [43,50]]
        assert_eq!(c[0], 19.0); // 1*5 + 2*7
        assert_eq!(c[1], 22.0); // 1*6 + 2*8
        assert_eq!(c[2], 43.0); // 3*5 + 4*7
        assert_eq!(c[3], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_outer_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];

        let c = outer_product(&a, &b);

        assert_eq!(c.len(), 6);
        assert_eq!(c[0], 4.0); // 1*4
        assert_eq!(c[1], 5.0); // 1*5
        assert_eq!(c[2], 8.0); // 2*4
        assert_eq!(c[3], 10.0); // 2*5
        assert_eq!(c[4], 12.0); // 3*4
        assert_eq!(c[5], 15.0); // 3*5
    }

    #[test]
    fn test_batched_matmul() {
        let mut a = Tensor3::zeros(2, 2, 2); // 2 batches, 2x2 matrices
        let mut b = Tensor3::zeros(2, 2, 2);

        // Batch 0: identity * identity = identity
        a.set(0, 0, 0, 1.0);
        a.set(0, 1, 1, 1.0);
        b.set(0, 0, 0, 1.0);
        b.set(0, 1, 1, 1.0);

        // Batch 1: [[1,2],[3,4]] * [[5,6],[7,8]]
        a.set(1, 0, 0, 1.0);
        a.set(1, 0, 1, 2.0);
        a.set(1, 1, 0, 3.0);
        a.set(1, 1, 1, 4.0);
        b.set(1, 0, 0, 5.0);
        b.set(1, 0, 1, 6.0);
        b.set(1, 1, 0, 7.0);
        b.set(1, 1, 1, 8.0);

        let c = batched_matmul(&a, &b).unwrap();

        // Batch 0 should be identity
        assert_eq!(c.get(0, 0, 0), 1.0);
        assert_eq!(c.get(0, 1, 1), 1.0);

        // Batch 1
        assert_eq!(c.get(1, 0, 0), 19.0); // 1*5 + 2*7
        assert_eq!(c.get(1, 0, 1), 22.0); // 1*6 + 2*8
        assert_eq!(c.get(1, 1, 0), 43.0); // 3*5 + 4*7
        assert_eq!(c.get(1, 1, 1), 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_einsum_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let c = einsum("ij,jk->ik", &a, &[2, 2], Some((&b, &[2, 2]))).unwrap();

        assert_eq!(c[0], 19.0);
        assert_eq!(c[1], 22.0);
        assert_eq!(c[2], 43.0);
        assert_eq!(c[3], 50.0);
    }

    #[test]
    fn test_einsum_outer() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        let c = einsum("i,j->ij", &a, &[2], Some((&b, &[2]))).unwrap();

        assert_eq!(c[0], 3.0);
        assert_eq!(c[1], 4.0);
        assert_eq!(c[2], 6.0);
        assert_eq!(c[3], 8.0);
    }

    #[test]
    fn test_einsum_transpose() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2

        let c = einsum("ij->ji", &a, &[2, 2], None).unwrap();

        assert_eq!(c[0], 1.0); // [0,0]
        assert_eq!(c[1], 3.0); // [0,1] -> [1,0]
        assert_eq!(c[2], 2.0); // [1,0] -> [0,1]
        assert_eq!(c[3], 4.0); // [1,1]
    }

    #[test]
    fn test_einsum_matvec() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0];

        let c = einsum("ij,j->i", &a, &[2, 2], Some((&b, &[2]))).unwrap();

        assert_eq!(c[0], 17.0); // 1*5 + 2*6
        assert_eq!(c[1], 39.0); // 3*5 + 4*6
    }

    #[test]
    fn test_einsum_diagonal() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2

        let c = einsum("ii->i", &a, &[2, 2], None).unwrap();

        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 4.0);
    }

    #[test]
    fn test_einsum_hadamard() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let c = einsum("ij,ij->ij", &a, &[2, 2], Some((&b, &[2, 2]))).unwrap();

        assert_eq!(c[0], 5.0);
        assert_eq!(c[1], 12.0);
        assert_eq!(c[2], 21.0);
        assert_eq!(c[3], 32.0);
    }

    #[test]
    fn test_einsum_frobenius() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let c = einsum("ij,ij->", &a, &[2, 2], Some((&b, &[2, 2]))).unwrap();

        assert_eq!(c[0], 10.0); // 1+2+3+4
    }

    #[test]
    fn test_einsum_tensor_transpose() {
        // 2x2x2 tensor
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let c = einsum("ijk->ikj", &a, &[2, 2, 2], None).unwrap();

        // Original: [[[1,2], [3,4]], [[5,6], [7,8]]]
        // Swapped: [[[1,3], [2,4]], [[5,7], [6,8]]]
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 3.0);
        assert_eq!(c[2], 2.0);
        assert_eq!(c[3], 4.0);
    }

    #[test]
    fn test_einsum_row_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3

        let c = einsum("ij->i", &a, &[2, 3], None).unwrap();

        assert_eq!(c[0], 6.0); // 1+2+3
        assert_eq!(c[1], 15.0); // 4+5+6
    }

    #[test]
    fn test_einsum_col_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3

        let c = einsum("ij->j", &a, &[2, 3], None).unwrap();

        assert_eq!(c[0], 5.0); // 1+4
        assert_eq!(c[1], 7.0); // 2+5
        assert_eq!(c[2], 9.0); // 3+6
    }

    #[test]
    fn test_einsum_total_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0];

        let c = einsum("ij->", &a, &[2, 2], None).unwrap();

        assert_eq!(c[0], 10.0); // 1+2+3+4
    }

    #[test]
    fn test_einsum_trace() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2], [3,4]]

        let c = einsum("ii->", &a, &[2, 2], None).unwrap();

        assert_eq!(c[0], 5.0); // 1+4 (diagonal sum)
    }

    #[test]
    fn test_einsum_trace_nonsquare() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]

        let c = einsum("ii->", &a, &[2, 3], None).unwrap();

        assert_eq!(c[0], 6.0); // 1+5 (diagonal sum, min(2,3)=2)
    }

    #[test]
    fn test_einsum_alt_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2], [3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6], [7,8]]

        let c = einsum("ik,kj->ij", &a, &[2, 2], Some((&b, &[2, 2]))).unwrap();

        // Same result as ij,jk->ik
        assert_eq!(c[0], 19.0);
        assert_eq!(c[1], 22.0);
        assert_eq!(c[2], 43.0);
        assert_eq!(c[3], 50.0);
    }

    #[test]
    fn test_einsum_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let c = einsum("i,i->", &a, &[3], Some((&b, &[3]))).unwrap();

        assert_eq!(c[0], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_einsum_elem_vec_mul() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let c = einsum("i,i->i", &a, &[3], Some((&b, &[3]))).unwrap();

        assert_eq!(c[0], 4.0); // 1*4
        assert_eq!(c[1], 10.0); // 2*5
        assert_eq!(c[2], 18.0); // 3*6
    }

    #[test]
    fn test_einsum_outer_3d() {
        // A: [[1,2], [3,4]] (2x2)
        // B: [[5,6,7], [8,9,10]] (2x3)
        // Result: C[i,j,k] = A[i,j] * B[i,k] (2x2x3)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let c = einsum("ij,ik->ijk", &a, &[2, 2], Some((&b, &[2, 3]))).unwrap();

        assert_eq!(c.len(), 12); // 2*2*3
        // C[0,0,0] = A[0,0] * B[0,0] = 1*5 = 5
        assert_eq!(c[0], 5.0);
        // C[0,0,1] = A[0,0] * B[0,1] = 1*6 = 6
        assert_eq!(c[1], 6.0);
        // C[0,0,2] = A[0,0] * B[0,2] = 1*7 = 7
        assert_eq!(c[2], 7.0);
        // C[0,1,0] = A[0,1] * B[0,0] = 2*5 = 10
        assert_eq!(c[3], 10.0);
    }

    #[test]
    fn test_einsum_tensor_transpose_jik() {
        // 2x2x2 tensor: [[[1,2], [3,4]], [[5,6], [7,8]]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let c = einsum("ijk->jik", &a, &[2, 2, 2], None).unwrap();

        // Swapped first two axes: [[[1,2], [5,6]], [[3,4], [7,8]]]
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 2.0);
        assert_eq!(c[2], 5.0);
        assert_eq!(c[3], 6.0);
        assert_eq!(c[4], 3.0);
        assert_eq!(c[5], 4.0);
    }

    #[test]
    fn test_einsum_tensor_transpose_kji() {
        // 2x2x2 tensor: [[[1,2], [3,4]], [[5,6], [7,8]]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let c = einsum("ijk->kji", &a, &[2, 2, 2], None).unwrap();

        // Reversed all axes: [[[1,5], [3,7]], [[2,6], [4,8]]]
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 5.0);
        assert_eq!(c[2], 3.0);
        assert_eq!(c[3], 7.0);
        assert_eq!(c[4], 2.0);
        assert_eq!(c[5], 6.0);
    }

    #[test]
    fn test_einsum_3d_frobenius() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let c = einsum("ijk,ijk->", &a, &[2, 2, 2], Some((&b, &[2, 2, 2]))).unwrap();

        assert_eq!(c[0], 36.0); // 1+2+3+4+5+6+7+8
    }

    #[test]
    fn test_einsum_3d_matrix_contract() {
        // A: 2x2x2 tensor [[[1,2], [3,4]], [[5,6], [7,8]]]
        // B: 2x3 matrix [[1,2,3], [4,5,6]]
        // C[i,j,l] = sum_k A[i,j,k] * B[k,l]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let c = einsum("ijk,kl->ijl", &a, &[2, 2, 2], Some((&b, &[2, 3]))).unwrap();

        assert_eq!(c.len(), 12); // 2*2*3
        // C[0,0,0] = A[0,0,0]*B[0,0] + A[0,0,1]*B[1,0] = 1*1 + 2*4 = 9
        assert_eq!(c[0], 9.0);
        // C[0,0,1] = A[0,0,0]*B[0,1] + A[0,0,1]*B[1,1] = 1*2 + 2*5 = 12
        assert_eq!(c[1], 12.0);
        // C[0,0,2] = A[0,0,0]*B[0,2] + A[0,0,1]*B[1,2] = 1*3 + 2*6 = 15
        assert_eq!(c[2], 15.0);
    }

    #[test]
    fn test_einsum_3d_sum_last() {
        // 2x2x2 tensor: [[[1,2], [3,4]], [[5,6], [7,8]]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let c = einsum("ijk->ij", &a, &[2, 2, 2], None).unwrap();

        assert_eq!(c.len(), 4); // 2*2
        assert_eq!(c[0], 3.0); // 1+2
        assert_eq!(c[1], 7.0); // 3+4
        assert_eq!(c[2], 11.0); // 5+6
        assert_eq!(c[3], 15.0); // 7+8
    }

    #[test]
    fn test_einsum_3d_sum_first() {
        // 2x2x2 tensor: [[[1,2], [3,4]], [[5,6], [7,8]]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let c = einsum("ijk->jk", &a, &[2, 2, 2], None).unwrap();

        assert_eq!(c.len(), 4); // 2*2
        assert_eq!(c[0], 6.0); // 1+5
        assert_eq!(c[1], 8.0); // 2+6
        assert_eq!(c[2], 10.0); // 3+7
        assert_eq!(c[3], 12.0); // 4+8
    }

    #[test]
    fn test_contract_2d_large_gemm_path() {
        // Test with n >= 32 to exercise the GEMM-based optimization path
        let n = 64;
        let k = 48;
        let m = 56;

        // Create matrices with known pattern
        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];

        for i in 0..m {
            for j in 0..k {
                a[i * k + j] = ((i + j) % 7) as f64;
            }
        }
        for i in 0..k {
            for j in 0..n {
                b[i * n + j] = ((i * 2 + j) % 5) as f64;
            }
        }

        let c = contract_2d(&a, (m, k), &b, (k, n)).unwrap();

        // Verify a few elements
        assert_eq!(c.len(), m * n);

        // C[0,0] = sum_j A[0,j] * B[j,0]
        let mut expected = 0.0;
        for j in 0..k {
            expected += a[j] * b[j * n];
        }
        assert!((c[0] - expected).abs() < 1e-10);

        // C[m-1, n-1] = sum_j A[m-1,j] * B[j,n-1]
        let mut expected2 = 0.0;
        for j in 0..k {
            expected2 += a[(m - 1) * k + j] * b[j * n + (n - 1)];
        }
        assert!((c[(m - 1) * n + (n - 1)] - expected2).abs() < 1e-10);
    }

    #[test]
    fn test_batched_matmul_large_gemm_path() {
        // Test with matrices >= 16 to exercise the GEMM-based optimization path
        let batch = 4;
        let m = 20;
        let k = 24;
        let n = 18;

        let mut a = Tensor3::zeros(batch, m, k);
        let mut b = Tensor3::zeros(batch, k, n);

        // Fill with pattern
        for batch_idx in 0..batch {
            for i in 0..m {
                for j in 0..k {
                    a.set(batch_idx, i, j, ((batch_idx + i + j) % 5) as f64);
                }
            }
            for i in 0..k {
                for j in 0..n {
                    b.set(batch_idx, i, j, ((batch_idx * 2 + i + j * 3) % 7) as f64);
                }
            }
        }

        let c = batched_matmul(&a, &b).unwrap();

        let (c_batch, c_m, c_n) = c.dims();
        assert_eq!(c_batch, batch);
        assert_eq!(c_m, m);
        assert_eq!(c_n, n);

        // Verify a few elements in each batch
        for batch_idx in 0..batch {
            // C[batch, 0, 0] = sum_j A[batch, 0, j] * B[batch, j, 0]
            let mut expected = 0.0;
            for j in 0..k {
                expected += a.get(batch_idx, 0, j) * b.get(batch_idx, j, 0);
            }
            assert!(
                (c.get(batch_idx, 0, 0) - expected).abs() < 1e-10,
                "Batch {} failed: got {}, expected {}",
                batch_idx,
                c.get(batch_idx, 0, 0),
                expected
            );
        }
    }

    #[test]
    fn test_einsum_matmul_large_gemm_path() {
        // Test einsum matrix multiply with sizes >= 32
        let m = 40;
        let k = 36;
        let n = 44;

        let mut a = vec![0.0f64; m * k];
        let mut b = vec![0.0f64; k * n];

        for i in 0..m {
            for j in 0..k {
                a[i * k + j] = ((i + j * 2) % 9) as f64;
            }
        }
        for i in 0..k {
            for j in 0..n {
                b[i * n + j] = ((i * 3 + j) % 11) as f64;
            }
        }

        let c = einsum("ij,jk->ik", &a, &[m, k], Some((&b, &[k, n]))).unwrap();

        assert_eq!(c.len(), m * n);

        // Verify C[0,0]
        let mut expected = 0.0;
        for j in 0..k {
            expected += a[j] * b[j * n];
        }
        assert!((c[0] - expected).abs() < 1e-10);
    }
}
