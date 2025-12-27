//! N-dimensional tensor type.
//!
//! Provides a flexible N-dimensional array (tensor) type with:
//! - Dynamic shape and strides
//! - Row-major (C-order) or column-major (Fortran-order) storage
//! - Views and slicing without copying
//! - Broadcasting support
//! - Contraction and reduction operations
//!
//! # Example
//!
//! ```
//! use oxiblas_blas::ndtensor::{NdTensor, Order};
//!
//! // Create a 3D tensor of shape [2, 3, 4]
//! let t: NdTensor<f64> = NdTensor::zeros(&[2, 3, 4]);
//! assert_eq!(t.shape(), &[2, 3, 4]);
//! assert_eq!(t.ndim(), 3);
//! assert_eq!(t.len(), 24);
//!
//! // Create from data
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let t = NdTensor::from_vec(data, &[2, 3]).unwrap();
//! assert_eq!(t[[0, 0]], 1.0);
//! assert_eq!(t[[1, 2]], 6.0);
//! ```

use core::ops::{Index, IndexMut};
use oxiblas_core::scalar::Field;

/// Storage order for N-dimensional tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Order {
    /// Row-major (C-style) ordering. Last index varies fastest.
    #[default]
    RowMajor,
    /// Column-major (Fortran-style) ordering. First index varies fastest.
    ColMajor,
}

/// Error type for tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NdTensorError {
    /// Shape doesn't match data length.
    ShapeMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
    /// Invalid axis for operation.
    InvalidAxis {
        /// The axis that was specified.
        axis: usize,
        /// Number of dimensions.
        ndim: usize,
    },
    /// Shapes are incompatible for broadcasting.
    BroadcastError {
        /// First shape.
        shape1: Vec<usize>,
        /// Second shape.
        shape2: Vec<usize>,
    },
    /// Dimension mismatch in contraction.
    ContractionError {
        /// Description of the mismatch.
        msg: String,
    },
    /// Index out of bounds.
    IndexOutOfBounds {
        /// The index that was out of bounds.
        index: Vec<usize>,
        /// The shape of the tensor.
        shape: Vec<usize>,
    },
}

impl core::fmt::Display for NdTensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {} elements, got {}",
                    expected, actual
                )
            }
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "Invalid axis {} for {}-dimensional tensor", axis, ndim)
            }
            Self::BroadcastError { shape1, shape2 } => {
                write!(f, "Cannot broadcast shapes {:?} and {:?}", shape1, shape2)
            }
            Self::ContractionError { msg } => write!(f, "Contraction error: {}", msg),
            Self::IndexOutOfBounds { index, shape } => {
                write!(f, "Index {:?} out of bounds for shape {:?}", index, shape)
            }
        }
    }
}

impl std::error::Error for NdTensorError {}

/// An N-dimensional tensor (multi-dimensional array).
///
/// Stores data in a contiguous buffer with configurable memory order.
/// Supports arbitrary number of dimensions with efficient indexing.
#[derive(Debug, Clone)]
pub struct NdTensor<T: Field> {
    /// Underlying data storage.
    data: Vec<T>,
    /// Shape of the tensor (dimensions).
    shape: Vec<usize>,
    /// Strides for each dimension.
    strides: Vec<usize>,
    /// Storage order.
    order: Order,
}

impl<T: Field> NdTensor<T> {
    /// Creates a new tensor filled with zeros.
    ///
    /// Uses row-major (C-order) storage by default.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self::zeros_with_order(shape, Order::RowMajor)
    }

    /// Creates a new tensor filled with zeros with specified order.
    #[must_use]
    pub fn zeros_with_order(shape: &[usize], order: Order) -> Self {
        let len = shape.iter().product();
        let strides = compute_strides(shape, order);
        Self {
            data: vec![T::zero(); len],
            shape: shape.to_vec(),
            strides,
            order,
        }
    }

    /// Creates a new tensor filled with a value.
    #[must_use]
    pub fn filled(shape: &[usize], value: T) -> Self {
        let len = shape.iter().product();
        let strides = compute_strides(shape, Order::RowMajor);
        Self {
            data: vec![value; len],
            shape: shape.to_vec(),
            strides,
            order: Order::RowMajor,
        }
    }

    /// Creates a new tensor filled with ones.
    #[must_use]
    pub fn ones(shape: &[usize]) -> Self {
        Self::filled(shape, T::one())
    }

    /// Creates a tensor from a flat vector.
    ///
    /// # Errors
    ///
    /// Returns error if data length doesn't match shape.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, NdTensorError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(NdTensorError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let strides = compute_strides(shape, Order::RowMajor);
        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides,
            order: Order::RowMajor,
        })
    }

    /// Creates a tensor from a slice (copies the data).
    ///
    /// # Errors
    ///
    /// Returns error if data length doesn't match shape.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self, NdTensorError> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Creates a 1D tensor (vector) from a slice.
    #[must_use]
    pub fn from_vec_1d(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
            strides: vec![1],
            order: Order::RowMajor,
        }
    }

    /// Creates an identity tensor (only for 2D).
    ///
    /// # Panics
    ///
    /// Panics if n is 0.
    #[must_use]
    pub fn eye(n: usize) -> Self {
        let mut t = Self::zeros(&[n, n]);
        for i in 0..n {
            t[[i, i]] = T::one();
        }
        t
    }

    /// Creates a diagonal tensor from a vector (only for 2D result).
    #[must_use]
    pub fn diag(values: &[T]) -> Self {
        let n = values.len();
        let mut t = Self::zeros(&[n, n]);
        for (i, &v) in values.iter().enumerate() {
            t[[i, i]] = v;
        }
        t
    }

    /// Returns the number of dimensions (rank).
    #[inline]
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the shape of the tensor.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides of the tensor.
    #[inline]
    #[must_use]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the storage order.
    #[inline]
    #[must_use]
    pub fn order(&self) -> Order {
        self.order
    }

    /// Returns the total number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the underlying data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Computes the flat index for a multi-dimensional index.
    #[inline]
    fn flat_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }

    /// Gets an element by multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> T {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Gets a mutable reference to an element.
    #[inline]
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.flat_index(indices);
        &mut self.data[idx]
    }

    /// Sets an element by multi-dimensional index.
    #[inline]
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// # Errors
    ///
    /// Returns error if the new shape doesn't have the same number of elements.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, NdTensorError> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return Err(NdTensorError::ShapeMismatch {
                expected: self.len(),
                actual: new_len,
            });
        }
        let strides = compute_strides(new_shape, self.order);
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides,
            order: self.order,
        })
    }

    /// Reshapes the tensor in-place.
    pub fn reshape_mut(&mut self, new_shape: &[usize]) -> Result<(), NdTensorError> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return Err(NdTensorError::ShapeMismatch {
                expected: self.len(),
                actual: new_len,
            });
        }
        self.shape = new_shape.to_vec();
        self.strides = compute_strides(new_shape, self.order);
        Ok(())
    }

    /// Flattens the tensor to 1D.
    #[must_use]
    pub fn flatten(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: vec![self.len()],
            strides: vec![1],
            order: self.order,
        }
    }

    /// Transposes the tensor (reverses all axes).
    #[must_use]
    pub fn transpose(&self) -> Self {
        let ndim = self.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.reverse();
        self.permute(&axes)
    }

    /// Transposes specific axes (swaps two axes).
    #[must_use]
    pub fn swap_axes(&self, axis1: usize, axis2: usize) -> Self {
        let ndim = self.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(axis1, axis2);
        self.permute(&axes)
    }

    /// Permutes the axes according to the given order.
    ///
    /// # Panics
    ///
    /// Panics if axes don't form a valid permutation.
    #[must_use]
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.ndim(), "Axes must match tensor dimensions");

        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&a| self.strides[a]).collect();

        // Reorder data according to new layout
        let mut new_data = vec![T::zero(); self.len()];
        let new_strides_contiguous = compute_strides(&new_shape, self.order);

        let mut indices = vec![0usize; self.ndim()];
        for flat_idx in 0..self.len() {
            // Compute multi-dimensional index in new layout
            let mut remainder = flat_idx;
            for d in 0..self.ndim() {
                indices[d] = remainder / new_strides_contiguous[d];
                remainder %= new_strides_contiguous[d];
            }

            // Map to original indices
            let orig_idx: usize = indices
                .iter()
                .zip(new_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();

            new_data[flat_idx] = self.data[orig_idx];
        }

        Self {
            data: new_data,
            shape: new_shape,
            strides: new_strides_contiguous,
            order: self.order,
        }
    }

    /// Sums all elements.
    #[must_use]
    pub fn sum(&self) -> T {
        self.data.iter().copied().fold(T::zero(), |acc, x| acc + x)
    }

    /// Computes the mean of all elements.
    #[must_use]
    pub fn mean(&self) -> T
    where
        T: From<f64>,
    {
        if self.is_empty() {
            return T::zero();
        }
        let sum = self.sum();
        let count = T::from(self.len() as f64);
        sum / count
    }

    /// Sums along an axis, reducing that dimension.
    ///
    /// # Errors
    ///
    /// Returns error if axis is out of bounds.
    pub fn sum_axis(&self, axis: usize) -> Result<Self, NdTensorError> {
        if axis >= self.ndim() {
            return Err(NdTensorError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let mut result = Self::zeros(&new_shape);

        // Iterate over all elements
        let mut indices = vec![0usize; self.ndim()];
        for flat_idx in 0..self.len() {
            // Compute multi-dimensional index
            let mut remainder = flat_idx;
            for d in 0..self.ndim() {
                indices[d] = remainder / self.strides[d];
                remainder %= self.strides[d];
            }

            // Compute result index (without the summed axis)
            let mut result_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &v)| v)
                .collect();
            if result_indices.is_empty() {
                result_indices.push(0);
            }

            let result_idx = result.flat_index(&result_indices);
            result.data[result_idx] += self.data[flat_idx];
        }

        Ok(result)
    }

    /// Returns the maximum element.
    #[must_use]
    pub fn max(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
    }

    /// Returns the minimum element.
    #[must_use]
    pub fn min(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        self.data
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
    }

    /// Element-wise addition.
    ///
    /// # Panics
    ///
    /// Panics if shapes don't match.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            order: self.order,
        }
    }

    /// Element-wise subtraction.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for subtraction");
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            order: self.order,
        }
    }

    /// Element-wise multiplication (Hadamard product).
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape, other.shape,
            "Shapes must match for multiplication"
        );
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            order: self.order,
        }
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(&self, scalar: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            order: self.order,
        }
    }

    /// Element-wise negation.
    #[must_use]
    pub fn neg(&self) -> Self {
        let data: Vec<T> = self.data.iter().map(|&x| T::zero() - x).collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            order: self.order,
        }
    }

    /// Dot product (inner product) with another tensor.
    ///
    /// Both tensors must have the same shape.
    #[must_use]
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(self.shape, other.shape, "Shapes must match for dot product");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Matrix multiplication for 2D tensors.
    ///
    /// # Errors
    ///
    /// Returns error if tensors are not 2D or dimensions don't match.
    pub fn matmul(&self, other: &Self) -> Result<Self, NdTensorError> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(NdTensorError::ContractionError {
                msg: "matmul requires 2D tensors".to_string(),
            });
        }

        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k1 != k2 {
            return Err(NdTensorError::ContractionError {
                msg: format!("Inner dimensions don't match: {} vs {}", k1, k2),
            });
        }

        let k = k1;
        let mut result = Self::zeros(&[m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for p in 0..k {
                    sum += self.get(&[i, p]) * other.get(&[p, j]);
                }
                result.set(&[i, j], sum);
            }
        }

        Ok(result)
    }

    /// General tensor contraction along specified axes.
    ///
    /// Contracts `self` along `axis_self` with `other` along `axis_other`.
    ///
    /// # Errors
    ///
    /// Returns error if contraction dimensions don't match.
    pub fn contract(
        &self,
        other: &Self,
        axis_self: usize,
        axis_other: usize,
    ) -> Result<Self, NdTensorError> {
        if axis_self >= self.ndim() {
            return Err(NdTensorError::InvalidAxis {
                axis: axis_self,
                ndim: self.ndim(),
            });
        }
        if axis_other >= other.ndim() {
            return Err(NdTensorError::InvalidAxis {
                axis: axis_other,
                ndim: other.ndim(),
            });
        }
        if self.shape[axis_self] != other.shape[axis_other] {
            return Err(NdTensorError::ContractionError {
                msg: format!(
                    "Contraction dimensions don't match: {} vs {}",
                    self.shape[axis_self], other.shape[axis_other]
                ),
            });
        }

        let contract_dim = self.shape[axis_self];

        // Build result shape
        let mut result_shape = Vec::new();
        for (i, &d) in self.shape.iter().enumerate() {
            if i != axis_self {
                result_shape.push(d);
            }
        }
        for (i, &d) in other.shape.iter().enumerate() {
            if i != axis_other {
                result_shape.push(d);
            }
        }
        if result_shape.is_empty() {
            result_shape.push(1);
        }

        let mut result = Self::zeros(&result_shape);

        // Perform contraction (naive implementation)
        let self_remaining: usize = self.len() / contract_dim;
        let other_remaining: usize = other.len() / contract_dim;

        for si in 0..self_remaining {
            for oi in 0..other_remaining {
                let mut sum = T::zero();
                for c in 0..contract_dim {
                    // Compute indices in self
                    let self_idx = self.index_for_contraction(si, c, axis_self);
                    let other_idx = other.index_for_contraction(oi, c, axis_other);
                    sum += self.data[self_idx] * other.data[other_idx];
                }
                let result_idx = si * other_remaining + oi;
                result.data[result_idx] = sum;
            }
        }

        Ok(result)
    }

    /// Helper for contraction: compute flat index given outer index and contraction index.
    fn index_for_contraction(&self, outer: usize, contract: usize, contract_axis: usize) -> usize {
        let mut indices = Vec::with_capacity(self.ndim());
        let mut outer_remaining = outer;

        for d in 0..self.ndim() {
            if d == contract_axis {
                indices.push(contract);
            } else {
                // Compute index for this non-contracted dimension
                let dim_size = self.shape[d];
                let stride_factor: usize = self
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i > d && *i != contract_axis)
                    .map(|(_, &s)| s)
                    .product::<usize>()
                    .max(1);
                let idx = outer_remaining / stride_factor;
                outer_remaining %= stride_factor;
                indices.push(idx % dim_size);
            }
        }

        self.flat_index(&indices)
    }

    /// Outer product with another tensor.
    ///
    /// Result has shape `self.shape + other.shape`.
    #[must_use]
    pub fn outer(&self, other: &Self) -> Self {
        let mut result_shape = self.shape.clone();
        result_shape.extend(other.shape.iter());

        let mut result = Self::zeros(&result_shape);

        for (i, &a) in self.data.iter().enumerate() {
            for (j, &b) in other.data.iter().enumerate() {
                result.data[i * other.len() + j] = a * b;
            }
        }

        result
    }

    /// Extracts the diagonal of a 2D tensor.
    ///
    /// # Errors
    ///
    /// Returns error if tensor is not 2D.
    pub fn diagonal(&self) -> Result<Self, NdTensorError> {
        if self.ndim() != 2 {
            return Err(NdTensorError::ContractionError {
                msg: "diagonal requires 2D tensor".to_string(),
            });
        }

        let n = self.shape[0].min(self.shape[1]);
        let mut result = Self::zeros(&[n]);
        for i in 0..n {
            result.data[i] = self.get(&[i, i]);
        }
        Ok(result)
    }

    /// Computes the trace of a 2D tensor.
    ///
    /// # Errors
    ///
    /// Returns error if tensor is not 2D.
    pub fn trace(&self) -> Result<T, NdTensorError> {
        if self.ndim() != 2 {
            return Err(NdTensorError::ContractionError {
                msg: "trace requires 2D tensor".to_string(),
            });
        }

        let n = self.shape[0].min(self.shape[1]);
        let mut sum = T::zero();
        for i in 0..n {
            sum += self.get(&[i, i]);
        }
        Ok(sum)
    }

    /// Computes the Frobenius norm (sqrt of sum of squares).
    #[must_use]
    pub fn norm(&self) -> T
    where
        T: From<f64>,
    {
        let sum_sq: T = self
            .data
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, x| acc + x);
        // Approximate sqrt using Newton's method
        if sum_sq == T::zero() {
            return T::zero();
        }
        let mut guess = sum_sq / T::from(2.0);
        for _ in 0..10 {
            guess = (guess + sum_sq / guess) / T::from(2.0);
        }
        guess
    }
}

/// Compute strides for a given shape and order.
fn compute_strides(shape: &[usize], order: Order) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1; shape.len()];
    match order {
        Order::RowMajor => {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Order::ColMajor => {
            for i in 1..shape.len() {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

// Index implementation for arrays and slices
impl<T: Field> Index<[usize; 1]> for NdTensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 1]) -> &T {
        &self.data[self.flat_index(&idx)]
    }
}

impl<T: Field> IndexMut<[usize; 1]> for NdTensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 1]) -> &mut T {
        let flat = self.flat_index(&idx);
        &mut self.data[flat]
    }
}

impl<T: Field> Index<[usize; 2]> for NdTensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[self.flat_index(&idx)]
    }
}

impl<T: Field> IndexMut<[usize; 2]> for NdTensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        let flat = self.flat_index(&idx);
        &mut self.data[flat]
    }
}

impl<T: Field> Index<[usize; 3]> for NdTensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 3]) -> &T {
        &self.data[self.flat_index(&idx)]
    }
}

impl<T: Field> IndexMut<[usize; 3]> for NdTensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut T {
        let flat = self.flat_index(&idx);
        &mut self.data[flat]
    }
}

impl<T: Field> Index<[usize; 4]> for NdTensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: [usize; 4]) -> &T {
        &self.data[self.flat_index(&idx)]
    }
}

impl<T: Field> IndexMut<[usize; 4]> for NdTensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; 4]) -> &mut T {
        let flat = self.flat_index(&idx);
        &mut self.data[flat]
    }
}

impl<T: Field> Index<&[usize]> for NdTensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: &[usize]) -> &T {
        &self.data[self.flat_index(idx)]
    }
}

impl<T: Field> IndexMut<&[usize]> for NdTensor<T> {
    #[inline]
    fn index_mut(&mut self, idx: &[usize]) -> &mut T {
        let flat = self.flat_index(idx);
        &mut self.data[flat]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndtensor_creation() {
        let t: NdTensor<f64> = NdTensor::zeros(&[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.len(), 24);
    }

    #[test]
    fn test_ndtensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = NdTensor::from_vec(data, &[2, 3]).unwrap();
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[0, 1]], 2.0);
        assert_eq!(t[[0, 2]], 3.0);
        assert_eq!(t[[1, 0]], 4.0);
        assert_eq!(t[[1, 2]], 6.0);
    }

    #[test]
    fn test_ndtensor_strides_row_major() {
        let t: NdTensor<f64> = NdTensor::zeros(&[2, 3, 4]);
        assert_eq!(t.strides(), &[12, 4, 1]); // Row-major
    }

    #[test]
    fn test_ndtensor_strides_col_major() {
        let t: NdTensor<f64> = NdTensor::zeros_with_order(&[2, 3, 4], Order::ColMajor);
        assert_eq!(t.strides(), &[1, 2, 6]); // Column-major
    }

    #[test]
    fn test_ndtensor_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = NdTensor::from_vec(data, &[2, 3]).unwrap();
        let t2 = t.reshape(&[3, 2]).unwrap();
        assert_eq!(t2.shape(), &[3, 2]);
        assert_eq!(t2[[0, 0]], 1.0);
        assert_eq!(t2[[2, 1]], 6.0);
    }

    #[test]
    fn test_ndtensor_transpose_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = NdTensor::from_vec(data, &[2, 3]).unwrap();
        let tt = t.transpose();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt[[0, 0]], 1.0);
        assert_eq!(tt[[0, 1]], 4.0);
        assert_eq!(tt[[2, 0]], 3.0);
        assert_eq!(tt[[2, 1]], 6.0);
    }

    #[test]
    fn test_ndtensor_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = NdTensor::from_vec(data, &[2, 3]).unwrap();
        assert_eq!(t.sum(), 21.0);
    }

    #[test]
    fn test_ndtensor_sum_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = NdTensor::from_vec(data, &[2, 3]).unwrap();

        // Sum over axis 0
        let s0 = t.sum_axis(0).unwrap();
        assert_eq!(s0.shape(), &[3]);
        assert_eq!(s0[[0]], 5.0); // 1 + 4
        assert_eq!(s0[[1]], 7.0); // 2 + 5
        assert_eq!(s0[[2]], 9.0); // 3 + 6

        // Sum over axis 1
        let s1 = t.sum_axis(1).unwrap();
        assert_eq!(s1.shape(), &[2]);
        assert_eq!(s1[[0]], 6.0); // 1 + 2 + 3
        assert_eq!(s1[[1]], 15.0); // 4 + 5 + 6
    }

    #[test]
    fn test_ndtensor_matmul() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = NdTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(c[[0, 1]], 22.0); // 1*6 + 2*8
        assert_eq!(c[[1, 0]], 43.0); // 3*5 + 4*7
        assert_eq!(c[[1, 1]], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_ndtensor_outer() {
        let a = NdTensor::from_vec_1d(vec![1.0, 2.0]);
        let b = NdTensor::from_vec_1d(vec![3.0, 4.0, 5.0]);
        let c = a.outer(&b);

        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c[[0, 0]], 3.0);
        assert_eq!(c[[0, 2]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 2]], 10.0);
    }

    #[test]
    fn test_ndtensor_dot() {
        let a = NdTensor::from_vec_1d(vec![1.0, 2.0, 3.0]);
        let b = NdTensor::from_vec_1d(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_ndtensor_diagonal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t = NdTensor::from_vec(data, &[3, 3]).unwrap();
        let d = t.diagonal().unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d[[0]], 1.0);
        assert_eq!(d[[1]], 5.0);
        assert_eq!(d[[2]], 9.0);
    }

    #[test]
    fn test_ndtensor_trace() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t = NdTensor::from_vec(data, &[3, 3]).unwrap();
        assert_eq!(t.trace().unwrap(), 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_ndtensor_eye() {
        let t: NdTensor<f64> = NdTensor::eye(3);
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[1, 1]], 1.0);
        assert_eq!(t[[2, 2]], 1.0);
        assert_eq!(t[[0, 1]], 0.0);
        assert_eq!(t[[1, 2]], 0.0);
    }

    #[test]
    fn test_ndtensor_add_sub_mul() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = NdTensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let c = a.add(&b);
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);

        let d = b.sub(&a);
        assert_eq!(d.data(), &[3.0, 3.0, 3.0]);

        let e = a.mul(&b);
        assert_eq!(e.data(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_ndtensor_scale() {
        let a = NdTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = a.scale(2.0);
        assert_eq!(b.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_ndtensor_3d() {
        let t: NdTensor<f64> = NdTensor::zeros(&[2, 3, 4]);
        let mut t = t;
        t[[0, 0, 0]] = 1.0;
        t[[1, 2, 3]] = 2.0;
        assert_eq!(t[[0, 0, 0]], 1.0);
        assert_eq!(t[[1, 2, 3]], 2.0);
        assert_eq!(t[[0, 1, 2]], 0.0);
    }

    #[test]
    fn test_ndtensor_4d() {
        let t: NdTensor<f64> = NdTensor::zeros(&[2, 3, 4, 5]);
        assert_eq!(t.len(), 120);
        assert_eq!(t.ndim(), 4);
    }
}
