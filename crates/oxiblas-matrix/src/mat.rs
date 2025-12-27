//! Owned matrix type.
//!
//! `Mat<T>` is a heap-allocated, resizable matrix with column-major storage
//! and guaranteed alignment for SIMD operations.
//!
//! # Custom Allocator Support
//!
//! The matrix type supports custom allocators through the third type parameter:
//!
//! ```ignore
//! use oxiblas_matrix::Mat;
//! use oxiblas_core::memory::{Alloc, Global};
//!
//! // Use global allocator (default)
//! let m: Mat<f64> = Mat::zeros(100, 100);
//!
//! // Use custom allocator
//! let m_custom: Mat<f64, Global> = Mat::zeros_in(100, 100, my_alloc);
//! ```

use crate::mat_mut::MatMut;
use crate::mat_ref::MatRef;
use oxiblas_core::memory::{AlignedVec, Alloc, DEFAULT_ALIGN, Global};
use oxiblas_core::scalar::Scalar;

/// An owned, heap-allocated matrix with column-major storage.
///
/// The matrix data is stored in a contiguous, aligned buffer. The storage
/// is column-major (Fortran order), meaning elements within a column are
/// contiguous in memory.
///
/// # Type Parameters
///
/// - `T`: The element type (must implement `Scalar`)
/// - `A`: The allocator type (default: `Global`)
///
/// # Memory Layout
///
/// For an `m × n` matrix, the element at row `i`, column `j` is stored at
/// index `i + j * row_stride`, where `row_stride >= m` to allow for padding.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::Mat;
///
/// // Create a 3x3 matrix of zeros
/// let mut m: Mat<f64> = Mat::zeros(3, 3);
///
/// // Set element at row 1, column 2
/// m[(1, 2)] = 5.0;
///
/// // Access as immutable view
/// let view = m.as_ref();
/// assert_eq!(view[(1, 2)], 5.0);
/// ```
pub struct Mat<T: Scalar, A: Alloc = Global> {
    /// Underlying data storage.
    data: AlignedVec<T, DEFAULT_ALIGN, A>,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Stride between consecutive rows (elements per column including padding).
    row_stride: usize,
}

impl<T: Scalar> Mat<T> {
    /// Creates a new matrix filled with zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let row_stride = Self::compute_row_stride(nrows);
        let total = row_stride * ncols;

        Mat {
            data: AlignedVec::zeros(total),
            nrows,
            ncols,
            row_stride,
        }
    }

    /// Creates a new matrix filled with a specific value.
    pub fn filled(nrows: usize, ncols: usize, value: T) -> Self {
        let row_stride = Self::compute_row_stride(nrows);
        let total = row_stride * ncols;

        Mat {
            data: AlignedVec::filled(total, value),
            nrows,
            ncols,
            row_stride,
        }
    }

    /// Creates a new matrix from a flat slice in column-major order.
    ///
    /// # Panics
    /// Panics if the slice length doesn't match `nrows * ncols`.
    pub fn from_slice(nrows: usize, ncols: usize, data: &[T]) -> Self
    where
        T: bytemuck::Zeroable,
    {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "Slice length must equal nrows * ncols"
        );

        let row_stride = Self::compute_row_stride(nrows);

        if row_stride == nrows {
            // No padding needed
            Mat {
                data: AlignedVec::from_slice(data),
                nrows,
                ncols,
                row_stride,
            }
        } else {
            // Need to copy with padding
            let total = row_stride * ncols;
            let mut mat_data = AlignedVec::zeros(total);

            for j in 0..ncols {
                let src_start = j * nrows;
                let dst_start = j * row_stride;
                for i in 0..nrows {
                    mat_data[dst_start + i] = data[src_start + i];
                }
            }

            Mat {
                data: mat_data,
                nrows,
                ncols,
                row_stride,
            }
        }
    }

    /// Creates a new matrix from a 2D array (row-major input).
    pub fn from_rows(rows: &[&[T]]) -> Self
    where
        T: bytemuck::Zeroable,
    {
        if rows.is_empty() {
            return Self::zeros(0, 0);
        }

        let nrows = rows.len();
        let ncols = rows[0].len();

        // Verify all rows have the same length
        for row in rows {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
        }

        let row_stride = Self::compute_row_stride(nrows);
        let total = row_stride * ncols;
        let mut data = AlignedVec::zeros(total);

        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[i + j * row_stride] = val;
            }
        }

        Mat {
            data,
            nrows,
            ncols,
            row_stride,
        }
    }

    /// Creates an identity matrix.
    pub fn eye(n: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = T::one();
        }
        mat
    }

    /// Creates a diagonal matrix from a slice.
    pub fn diag(diagonal: &[T]) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let n = diagonal.len();
        let mut mat = Self::zeros(n, n);
        for (i, &val) in diagonal.iter().enumerate() {
            mat[(i, i)] = val;
        }
        mat
    }

    /// Computes the row stride (elements per column) with padding for alignment.
    fn compute_row_stride(nrows: usize) -> usize {
        if nrows == 0 {
            return 0;
        }

        // Pad to ensure each column starts at an aligned address
        // Use element size to compute how many elements fit in a cache line
        let elem_size = core::mem::size_of::<T>();
        let elems_per_cacheline = DEFAULT_ALIGN / elem_size;

        // Round up to next multiple of cache line elements for SIMD alignment
        nrows.div_ceil(elems_per_cacheline) * elems_per_cacheline
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the shape as (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the row stride.
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the column stride (always 1 for column-major storage).
    #[inline]
    pub fn col_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Returns an immutable view of the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        MatRef::new(self.data.as_ptr(), self.nrows, self.ncols, self.row_stride)
    }

    /// Returns a mutable view of the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, T> {
        MatMut::new(
            self.data.as_mut_ptr(),
            self.nrows,
            self.ncols,
            self.row_stride,
        )
    }

    /// Returns the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col < self.ncols {
            Some(&self.data[row + col * self.row_stride])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.nrows && col < self.ncols {
            Some(&mut self.data[row + col * self.row_stride])
        } else {
            None
        }
    }

    /// Sets the element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        self.data[row + col * self.row_stride] = value;
    }

    /// Returns a submatrix view.
    #[inline]
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, T> {
        self.as_ref().submatrix(row_start, col_start, nrows, ncols)
    }

    /// Returns a mutable submatrix view.
    #[inline]
    pub fn submatrix_mut(
        &mut self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'_, T> {
        self.as_mut().submatrix(row_start, col_start, nrows, ncols)
    }

    /// Returns a column view.
    #[inline]
    pub fn col(&self, j: usize) -> MatRef<'_, T> {
        assert!(j < self.ncols, "Column index out of bounds");
        self.submatrix(0, j, self.nrows, 1)
    }

    /// Returns a mutable column view.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> MatMut<'_, T> {
        assert!(j < self.ncols, "Column index out of bounds");
        self.submatrix_mut(0, j, self.nrows, 1)
    }

    /// Returns a row view.
    #[inline]
    pub fn row(&self, i: usize) -> MatRef<'_, T> {
        assert!(i < self.nrows, "Row index out of bounds");
        self.submatrix(i, 0, 1, self.ncols)
    }

    /// Returns a mutable row view.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> MatMut<'_, T> {
        assert!(i < self.nrows, "Row index out of bounds");
        self.submatrix_mut(i, 0, 1, self.ncols)
    }

    /// Resizes the matrix, filling new elements with zeros.
    pub fn resize(&mut self, new_nrows: usize, new_ncols: usize)
    where
        T: bytemuck::Zeroable,
    {
        let new_row_stride = Self::compute_row_stride(new_nrows);
        let new_total = new_row_stride * new_ncols;

        let mut new_data = AlignedVec::zeros(new_total);

        // Copy existing data
        let copy_nrows = self.nrows.min(new_nrows);
        let copy_ncols = self.ncols.min(new_ncols);

        for j in 0..copy_ncols {
            for i in 0..copy_nrows {
                new_data[i + j * new_row_stride] = self.data[i + j * self.row_stride];
            }
        }

        self.data = new_data;
        self.nrows = new_nrows;
        self.ncols = new_ncols;
        self.row_stride = new_row_stride;
    }

    /// Transposes the matrix (creates a new matrix).
    pub fn transpose(&self) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let mut result = Mat::zeros(self.ncols, self.nrows);

        for j in 0..self.ncols {
            for i in 0..self.nrows {
                result[(j, i)] = self[(i, j)];
            }
        }

        result
    }

    /// Returns the raw data slice (including padding).
    #[inline]
    pub fn raw_data(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns the raw data slice mutably.
    #[inline]
    pub fn raw_data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Copies data from another matrix.
    pub fn copy_from(&mut self, other: &MatRef<'_, T>) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrix shapes must match for copy"
        );

        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self[(i, j)] = other[(i, j)];
            }
        }
    }

    /// Fills the matrix with a value.
    pub fn fill(&mut self, value: T) {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self[(i, j)] = value;
            }
        }
    }

    /// Scales the matrix by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self[(i, j)] *= alpha;
            }
        }
    }
}

// Methods that work with any allocator
impl<T: Scalar, A: Alloc> Mat<T, A> {
    /// Creates a new matrix filled with zeros using the specified allocator.
    pub fn zeros_in(nrows: usize, ncols: usize, alloc: A) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let row_stride = Self::compute_row_stride_generic(nrows);
        let total = row_stride * ncols;

        Mat {
            data: AlignedVec::zeros_in(total, alloc),
            nrows,
            ncols,
            row_stride,
        }
    }

    /// Creates a new matrix filled with a specific value using the specified allocator.
    pub fn filled_in(nrows: usize, ncols: usize, value: T, alloc: A) -> Self {
        let row_stride = Self::compute_row_stride_generic(nrows);
        let total = row_stride * ncols;

        Mat {
            data: AlignedVec::filled_in(total, value, alloc),
            nrows,
            ncols,
            row_stride,
        }
    }

    /// Returns a reference to the allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        self.data.allocator()
    }

    /// Computes the row stride (elements per column) with padding for alignment.
    fn compute_row_stride_generic(nrows: usize) -> usize {
        if nrows == 0 {
            return 0;
        }

        let elem_size = core::mem::size_of::<T>();
        let elems_per_cacheline = DEFAULT_ALIGN / elem_size;

        nrows.div_ceil(elems_per_cacheline) * elems_per_cacheline
    }
}

// Clone implementation for Mat with any allocator
impl<T: Scalar + Clone, A: Alloc> Clone for Mat<T, A> {
    fn clone(&self) -> Self {
        let alloc = self.data.allocator().clone();
        let row_stride = self.row_stride;
        let total = row_stride * self.ncols;

        let mut data = AlignedVec::with_capacity_in(total, alloc);
        for item in self.data.as_slice() {
            data.push(*item);
        }

        Mat {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
            row_stride,
        }
    }
}

impl<T: Scalar> Default for Mat<T>
where
    T: bytemuck::Zeroable,
{
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

impl<T: Scalar> core::ops::Index<(usize, usize)> for Mat<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        &self.data[row + col * self.row_stride]
    }
}

impl<T: Scalar> core::ops::IndexMut<(usize, usize)> for Mat<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        &mut self.data[row + col * self.row_stride]
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Mat {}x{} {{", self.nrows, self.ncols)?;
        for i in 0..self.nrows.min(10) {
            write!(f, "  [")?;
            for j in 0..self.ncols.min(10) {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", self[(i, j)])?;
            }
            if self.ncols > 10 {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.nrows > 10 {
            writeln!(f, "  ...")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_zeros() {
        let m: Mat<f64> = Mat::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);

        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_mat_filled() {
        let m: Mat<f64> = Mat::filled(2, 3, 5.0);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], 5.0);
            }
        }
    }

    #[test]
    fn test_mat_eye() {
        let m: Mat<f64> = Mat::eye(3);

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(m[(i, j)], 1.0);
                } else {
                    assert_eq!(m[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_mat_from_slice() {
        // Column-major: [1, 4, 2, 5, 3, 6] represents:
        // [1 2 3]
        // [4 5 6]
        let data = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m: Mat<f64> = Mat::from_slice(2, 3, &data);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn test_mat_from_rows() {
        let rows: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let m = Mat::from_rows(rows);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn test_mat_transpose() {
        let rows: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let m = Mat::from_rows(rows);
        let mt = m.transpose();

        assert_eq!(mt.shape(), (3, 2));
        assert_eq!(mt[(0, 0)], 1.0);
        assert_eq!(mt[(1, 0)], 2.0);
        assert_eq!(mt[(0, 1)], 4.0);
        assert_eq!(mt[(2, 1)], 6.0);
    }

    #[test]
    fn test_mat_indexing() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);

        m[(1, 2)] = 42.0;
        assert_eq!(m[(1, 2)], 42.0);

        m.set(0, 0, 1.0);
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(10, 10), None);
    }

    #[test]
    fn test_mat_submatrix() {
        let rows: &[&[f64]] = &[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ];
        let m = Mat::from_rows(rows);

        let sub = m.submatrix(1, 1, 2, 2);
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(sub[(0, 0)], 6.0);
        assert_eq!(sub[(0, 1)], 7.0);
        assert_eq!(sub[(1, 0)], 10.0);
        assert_eq!(sub[(1, 1)], 11.0);
    }

    #[test]
    fn test_mat_col_row() {
        let rows: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let m = Mat::from_rows(rows);

        let col1 = m.col(1);
        assert_eq!(col1.shape(), (2, 1));
        assert_eq!(col1[(0, 0)], 2.0);
        assert_eq!(col1[(1, 0)], 5.0);

        let row0 = m.row(0);
        assert_eq!(row0.shape(), (1, 3));
        assert_eq!(row0[(0, 0)], 1.0);
        assert_eq!(row0[(0, 1)], 2.0);
        assert_eq!(row0[(0, 2)], 3.0);
    }

    #[test]
    fn test_mat_alignment() {
        let m: Mat<f64> = Mat::zeros(100, 100);
        let ptr = m.as_ptr();

        // Should be aligned to at least 64 bytes
        assert_eq!(ptr as usize % 64, 0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_mat_serde() {
        let original = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Mat<f64> = serde_json::from_str(&json).unwrap();

        assert_eq!(original.shape(), deserialized.shape());
        for i in 0..original.nrows() {
            for j in 0..original.ncols() {
                assert!((original[(i, j)] - deserialized[(i, j)]).abs() < 1e-10);
            }
        }
    }
}

// =============================================================================
// Serde support
// =============================================================================

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl<T: Scalar + Serialize> Serialize for Mat<T> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            use serde::ser::SerializeStruct;

            // Extract data without padding (in column-major order)
            let mut data = Vec::with_capacity(self.nrows * self.ncols);
            for j in 0..self.ncols {
                for i in 0..self.nrows {
                    data.push(self[(i, j)]);
                }
            }

            let mut state = serializer.serialize_struct("Mat", 3)?;
            state.serialize_field("nrows", &self.nrows)?;
            state.serialize_field("ncols", &self.ncols)?;
            state.serialize_field("data", &data)?;
            state.end()
        }
    }

    impl<'de, T: Scalar + DeserializeOwned + bytemuck::Zeroable> Deserialize<'de> for Mat<T> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            use serde::de::{MapAccess, Visitor};

            struct MatVisitor<T>(std::marker::PhantomData<T>);

            impl<'de, T: Scalar + DeserializeOwned + bytemuck::Zeroable> Visitor<'de> for MatVisitor<T> {
                type Value = Mat<T>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    formatter.write_str("a matrix with nrows, ncols, and data fields")
                }

                fn visit_map<V>(self, mut map: V) -> Result<Mat<T>, V::Error>
                where
                    V: MapAccess<'de>,
                {
                    let mut nrows: Option<usize> = None;
                    let mut ncols: Option<usize> = None;
                    let mut data: Option<Vec<T>> = None;

                    while let Some(key) = map.next_key::<String>()? {
                        match key.as_str() {
                            "nrows" => {
                                if nrows.is_some() {
                                    return Err(serde::de::Error::duplicate_field("nrows"));
                                }
                                nrows = Some(map.next_value()?);
                            }
                            "ncols" => {
                                if ncols.is_some() {
                                    return Err(serde::de::Error::duplicate_field("ncols"));
                                }
                                ncols = Some(map.next_value()?);
                            }
                            "data" => {
                                if data.is_some() {
                                    return Err(serde::de::Error::duplicate_field("data"));
                                }
                                data = Some(map.next_value()?);
                            }
                            _ => {
                                let _: serde::de::IgnoredAny = map.next_value()?;
                            }
                        }
                    }

                    let nrows = nrows.ok_or_else(|| serde::de::Error::missing_field("nrows"))?;
                    let ncols = ncols.ok_or_else(|| serde::de::Error::missing_field("ncols"))?;
                    let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;

                    if data.len() != nrows * ncols {
                        return Err(serde::de::Error::custom(format!(
                            "Data length {} does not match dimensions {} x {}",
                            data.len(),
                            nrows,
                            ncols
                        )));
                    }

                    Ok(Mat::from_slice(nrows, ncols, &data))
                }
            }

            const FIELDS: &[&str] = &["nrows", "ncols", "data"];
            deserializer.deserialize_struct("Mat", FIELDS, MatVisitor(std::marker::PhantomData))
        }
    }
}
