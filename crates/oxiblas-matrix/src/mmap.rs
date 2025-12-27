//! Memory-mapped matrix storage.
//!
//! This module provides memory-mapped matrices for working with large datasets
//! that may not fit in RAM, or for sharing matrices between processes.
//!
//! # Features
//!
//! - **Large matrix support**: Work with matrices larger than available RAM
//! - **Process sharing**: Multiple processes can map the same file
//! - **Persistence**: Matrix data persists on disk
//! - **Zero-copy**: No data copying when opening existing matrices
//!
//! # File Format
//!
//! Memory-mapped matrices use a simple binary format:
//! - 8-byte magic number: "OXIBLAS\0"
//! - 8-byte version (u64, little-endian)
//! - 8-byte element type identifier
//! - 8-byte nrows (u64, little-endian)
//! - 8-byte ncols (u64, little-endian)
//! - 8-byte row_stride (u64, little-endian)
//! - Padding to 64-byte alignment
//! - Column-major matrix data
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_matrix::mmap::{MmapMat, MmapMatMut};
//! use std::path::Path;
//!
//! // Create a new memory-mapped matrix
//! let path = "/tmp/matrix.oxiblas";
//! {
//!     let mut mmat = MmapMatMut::<f64>::create(path, 1000, 1000)?;
//!     // Initialize data...
//!     mmat[(0, 0)] = 1.0;
//! }
//!
//! // Open for reading
//! let mmat = MmapMat::<f64>::open(path)?;
//! assert_eq!(mmat[(0, 0)], 1.0);
//! ```

use crate::mat_mut::MatMut;
use crate::mat_ref::MatRef;
use memmap2::{Mmap, MmapMut, MmapOptions};
use oxiblas_core::memory::DEFAULT_ALIGN;
use oxiblas_core::scalar::Scalar;
use std::fs::{File, OpenOptions};
use std::io;
use std::marker::PhantomData;
use std::path::Path;

/// Magic number for OxiBLAS matrix files.
const MAGIC: &[u8; 8] = b"OXIBLAS\0";

/// Current file format version.
const VERSION: u64 = 1;

/// Header size (padded to 64 bytes for alignment).
const HEADER_SIZE: usize = 64;

/// Type identifiers for elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum ElementType {
    /// 32-bit floating point
    F32 = 1,
    /// 64-bit floating point
    F64 = 2,
    /// 32-bit complex (2x f32)
    C32 = 3,
    /// 64-bit complex (2x f64)
    C64 = 4,
    /// 32-bit signed integer
    I32 = 5,
    /// 64-bit signed integer
    I64 = 6,
}

impl ElementType {
    /// Returns the size in bytes for this element type.
    #[inline]
    pub const fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::C32 => 8,
            Self::C64 => 16,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }

    /// Returns the element type for a given type parameter.
    fn from_type<T: Scalar>() -> Option<Self> {
        let size = core::mem::size_of::<T>();
        let name = core::any::type_name::<T>();

        if name.contains("f32") && size == 4 {
            Some(Self::F32)
        } else if name.contains("f64") && size == 8 {
            Some(Self::F64)
        } else if name.contains("Complex") && size == 8 {
            Some(Self::C32)
        } else if name.contains("Complex") && size == 16 {
            Some(Self::C64)
        } else if name.contains("i32") && size == 4 {
            Some(Self::I32)
        } else if name.contains("i64") && size == 8 {
            Some(Self::I64)
        } else {
            None
        }
    }

    fn from_u64(v: u64) -> Option<Self> {
        match v {
            1 => Some(Self::F32),
            2 => Some(Self::F64),
            3 => Some(Self::C32),
            4 => Some(Self::C64),
            5 => Some(Self::I32),
            6 => Some(Self::I64),
            _ => None,
        }
    }
}

/// Error type for memory-mapped matrix operations.
#[derive(Debug)]
pub enum MmapError {
    /// I/O error from the underlying file system.
    Io(io::Error),
    /// Invalid file format (bad magic number or version).
    InvalidFormat(String),
    /// Type mismatch between requested type and file contents.
    TypeMismatch {
        /// The expected element type based on the requested type parameter.
        expected: ElementType,
        /// The actual element type found in the file.
        found: ElementType,
    },
    /// Unsupported element type.
    UnsupportedType,
    /// Invalid dimensions.
    InvalidDimensions(String),
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            Self::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {expected:?}, found {found:?}")
            }
            Self::UnsupportedType => write!(f, "Unsupported element type"),
            Self::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {msg}"),
        }
    }
}

impl std::error::Error for MmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for MmapError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// File header for memory-mapped matrices.
#[repr(C)]
struct Header {
    magic: [u8; 8],
    version: u64,
    elem_type: u64,
    nrows: u64,
    ncols: u64,
    row_stride: u64,
    _padding: [u8; 16], // Pad to 64 bytes
}

impl Header {
    fn new<T: Scalar>(nrows: usize, ncols: usize, row_stride: usize) -> Result<Self, MmapError> {
        let elem_type = ElementType::from_type::<T>().ok_or(MmapError::UnsupportedType)?;

        Ok(Header {
            magic: *MAGIC,
            version: VERSION,
            elem_type: elem_type as u64,
            nrows: nrows as u64,
            ncols: ncols as u64,
            row_stride: row_stride as u64,
            _padding: [0; 16],
        })
    }

    fn validate<T: Scalar>(&self) -> Result<(), MmapError> {
        // Check magic
        if &self.magic != MAGIC {
            return Err(MmapError::InvalidFormat("Invalid magic number".to_string()));
        }

        // Check version
        if self.version != VERSION {
            return Err(MmapError::InvalidFormat(format!(
                "Unsupported version: {}",
                self.version
            )));
        }

        // Check element type
        let file_type = ElementType::from_u64(self.elem_type).ok_or(MmapError::InvalidFormat(
            format!("Unknown element type: {}", self.elem_type),
        ))?;

        let expected_type = ElementType::from_type::<T>().ok_or(MmapError::UnsupportedType)?;

        if file_type != expected_type {
            return Err(MmapError::TypeMismatch {
                expected: expected_type,
                found: file_type,
            });
        }

        Ok(())
    }

    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..8].copy_from_slice(&self.magic);
        bytes[8..16].copy_from_slice(&self.version.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.elem_type.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.nrows.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.ncols.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.row_stride.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, MmapError> {
        if bytes.len() < HEADER_SIZE {
            return Err(MmapError::InvalidFormat("Header too short".to_string()));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);

        Ok(Header {
            magic,
            version: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            elem_type: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            nrows: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            ncols: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            row_stride: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            _padding: [0; 16],
        })
    }
}

/// Computes the row stride with padding for alignment.
fn compute_row_stride<T>(nrows: usize) -> usize {
    if nrows == 0 {
        return 0;
    }

    let elem_size = core::mem::size_of::<T>();
    let elems_per_cacheline = DEFAULT_ALIGN / elem_size;

    nrows.div_ceil(elems_per_cacheline) * elems_per_cacheline
}

/// A read-only memory-mapped matrix.
///
/// This type maps a matrix file into memory for read-only access. Changes to
/// the underlying file by other processes may be visible.
pub struct MmapMat<T: Scalar> {
    mmap: Mmap,
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> MmapMat<T> {
    /// Opens an existing memory-mapped matrix file for reading.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or mapped
    /// - The file format is invalid
    /// - The element type doesn't match `T`
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MmapError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Read and validate header
        let header = Header::from_bytes(&mmap)?;
        header.validate::<T>()?;

        Ok(MmapMat {
            mmap,
            nrows: header.nrows as usize,
            ncols: header.ncols as usize,
            row_stride: header.row_stride as usize,
            _phantom: PhantomData,
        })
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

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        unsafe { self.mmap.as_ptr().add(HEADER_SIZE).cast() }
    }

    /// Returns an immutable view of the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        MatRef::new(self.as_ptr(), self.nrows, self.ncols, self.row_stride)
    }

    /// Returns the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col < self.ncols {
            Some(unsafe { &*self.as_ptr().add(row + col * self.row_stride) })
        } else {
            None
        }
    }

    /// Advise the kernel about intended access pattern.
    ///
    /// This can improve performance for sequential or random access patterns.
    #[cfg(unix)]
    pub fn advise_sequential(&self) -> Result<(), MmapError> {
        self.mmap.advise(memmap2::Advice::Sequential)?;
        Ok(())
    }

    /// Advise the kernel that this region will be needed soon.
    #[cfg(unix)]
    pub fn advise_willneed(&self) -> Result<(), MmapError> {
        self.mmap.advise(memmap2::Advice::WillNeed)?;
        Ok(())
    }
}

impl<T: Scalar> core::ops::Index<(usize, usize)> for MmapMat<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe { &*self.as_ptr().add(row + col * self.row_stride) }
    }
}

/// A mutable memory-mapped matrix.
///
/// This type maps a matrix file into memory for read-write access. Changes
/// are automatically synced to the file.
pub struct MmapMatMut<T: Scalar> {
    mmap: MmapMut,
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> MmapMatMut<T> {
    /// Creates a new memory-mapped matrix file.
    ///
    /// The file is created with the specified dimensions and initialized to zeros.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created
    /// - The element type is not supported
    pub fn create<P: AsRef<Path>>(path: P, nrows: usize, ncols: usize) -> Result<Self, MmapError> {
        let row_stride = compute_row_stride::<T>(nrows);
        let data_size = row_stride * ncols * core::mem::size_of::<T>();
        let total_size = HEADER_SIZE + data_size;

        // Create and size the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(total_size as u64)?;

        // Map the file
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Write header
        let header = Header::new::<T>(nrows, ncols, row_stride)?;
        mmap[0..HEADER_SIZE].copy_from_slice(&header.to_bytes());

        // Zero-initialize data (file should already be zero, but be safe)
        mmap[HEADER_SIZE..].fill(0);

        Ok(MmapMatMut {
            mmap,
            nrows,
            ncols,
            row_stride,
            _phantom: PhantomData,
        })
    }

    /// Opens an existing memory-mapped matrix file for reading and writing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened or mapped
    /// - The file format is invalid
    /// - The element type doesn't match `T`
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, MmapError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Read and validate header
        let header = Header::from_bytes(&mmap)?;
        header.validate::<T>()?;

        Ok(MmapMatMut {
            mmap,
            nrows: header.nrows as usize,
            ncols: header.ncols as usize,
            row_stride: header.row_stride as usize,
            _phantom: PhantomData,
        })
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

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        unsafe { self.mmap.as_ptr().add(HEADER_SIZE).cast() }
    }

    /// Returns a mutable pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.mmap.as_mut_ptr().add(HEADER_SIZE).cast() }
    }

    /// Returns an immutable view of the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        MatRef::new(self.as_ptr(), self.nrows, self.ncols, self.row_stride)
    }

    /// Returns a mutable view of the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, T> {
        MatMut::new(self.as_mut_ptr(), self.nrows, self.ncols, self.row_stride)
    }

    /// Returns the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col < self.ncols {
            Some(unsafe { &*self.as_ptr().add(row + col * self.row_stride) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.nrows && col < self.ncols {
            Some(unsafe { &mut *self.as_mut_ptr().add(row + col * self.row_stride) })
        } else {
            None
        }
    }

    /// Sets the element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe {
            *self.as_mut_ptr().add(row + col * self.row_stride) = value;
        }
    }

    /// Flushes changes to disk synchronously.
    ///
    /// This ensures all modifications are written to the underlying file.
    pub fn flush(&self) -> Result<(), MmapError> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Flushes changes to disk asynchronously.
    ///
    /// This initiates a write but may return before completion.
    pub fn flush_async(&self) -> Result<(), MmapError> {
        self.mmap.flush_async()?;
        Ok(())
    }

    /// Fills the matrix with a value.
    pub fn fill(&mut self, value: T) {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self.set(i, j, value);
            }
        }
    }

    /// Copies data from a regular matrix.
    pub fn copy_from(&mut self, src: &MatRef<'_, T>) {
        assert_eq!(
            self.shape(),
            src.shape(),
            "Matrix shapes must match for copy"
        );

        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self.set(i, j, src[(i, j)]);
            }
        }
    }

    /// Advise the kernel about intended access pattern.
    #[cfg(unix)]
    pub fn advise_sequential(&self) -> Result<(), MmapError> {
        self.mmap.advise(memmap2::Advice::Sequential)?;
        Ok(())
    }

    /// Advise the kernel that this region will be needed soon.
    #[cfg(unix)]
    pub fn advise_willneed(&self) -> Result<(), MmapError> {
        self.mmap.advise(memmap2::Advice::WillNeed)?;
        Ok(())
    }
}

impl<T: Scalar> core::ops::Index<(usize, usize)> for MmapMatMut<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe { &*self.as_ptr().add(row + col * self.row_stride) }
    }
}

impl<T: Scalar> core::ops::IndexMut<(usize, usize)> for MmapMatMut<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe { &mut *self.as_mut_ptr().add(row + col * self.row_stride) }
    }
}

/// Builder for creating memory-mapped matrices from existing data.
pub struct MmapBuilder<T: Scalar> {
    nrows: usize,
    ncols: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> MmapBuilder<T> {
    /// Creates a new builder with the specified dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        MmapBuilder {
            nrows,
            ncols,
            _phantom: PhantomData,
        }
    }

    /// Creates a new memory-mapped matrix file from a Mat.
    pub fn from_mat<P: AsRef<Path>>(
        self,
        path: P,
        mat: &MatRef<'_, T>,
    ) -> Result<MmapMatMut<T>, MmapError> {
        if mat.shape() != (self.nrows, self.ncols) {
            return Err(MmapError::InvalidDimensions(format!(
                "Builder dimensions ({}, {}) don't match matrix ({}, {})",
                self.nrows,
                self.ncols,
                mat.nrows(),
                mat.ncols()
            )));
        }

        let mut mmat = MmapMatMut::create(path, self.nrows, self.ncols)?;
        mmat.copy_from(mat);
        mmat.flush()?;
        Ok(mmat)
    }

    /// Creates a new memory-mapped matrix file from a flat slice (column-major).
    pub fn from_slice<P: AsRef<Path>>(
        self,
        path: P,
        data: &[T],
    ) -> Result<MmapMatMut<T>, MmapError> {
        let expected_len = self.nrows * self.ncols;
        if data.len() != expected_len {
            return Err(MmapError::InvalidDimensions(format!(
                "Slice length {} doesn't match dimensions {} x {} = {}",
                data.len(),
                self.nrows,
                self.ncols,
                expected_len
            )));
        }

        let mut mmat = MmapMatMut::create(path, self.nrows, self.ncols)?;

        // Copy data column by column
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                mmat.set(i, j, data[i + j * self.nrows]);
            }
        }

        mmat.flush()?;
        Ok(mmat)
    }
}

/// Utility function to write a Mat directly to a memory-mapped file.
pub fn write_mat<T: Scalar, P: AsRef<Path>>(path: P, mat: &MatRef<'_, T>) -> Result<(), MmapError> {
    let mut mmat = MmapMatMut::create(path, mat.nrows(), mat.ncols())?;
    mmat.copy_from(mat);
    mmat.flush()?;
    Ok(())
}

/// Utility function to read dimensions from a memory-mapped matrix file without mapping the data.
pub fn read_dimensions<P: AsRef<Path>>(path: P) -> Result<(usize, usize), MmapError> {
    let mut file = File::open(path)?;
    let mut header_bytes = [0u8; HEADER_SIZE];

    use std::io::Read;
    file.read_exact(&mut header_bytes)?;

    let header = Header::from_bytes(&header_bytes)?;

    // Basic validation (just magic and version)
    if &header.magic != MAGIC {
        return Err(MmapError::InvalidFormat("Invalid magic number".to_string()));
    }
    if header.version != VERSION {
        return Err(MmapError::InvalidFormat(format!(
            "Unsupported version: {}",
            header.version
        )));
    }

    Ok((header.nrows as usize, header.ncols as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_create_and_open() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_basic.oxiblas");

        // Create a matrix
        {
            let mut mmat = MmapMatMut::<f64>::create(&path, 10, 10).unwrap();
            for i in 0..10 {
                for j in 0..10 {
                    mmat[(i, j)] = (i * 10 + j) as f64;
                }
            }
            mmat.flush().unwrap();
        }

        // Open and verify
        {
            let mmat = MmapMat::<f64>::open(&path).unwrap();
            assert_eq!(mmat.shape(), (10, 10));
            for i in 0..10 {
                for j in 0..10 {
                    assert_eq!(mmat[(i, j)], (i * 10 + j) as f64);
                }
            }
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_views() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_views.oxiblas");

        let mut mmat = MmapMatMut::<f64>::create(&path, 5, 5).unwrap();

        // Initialize through view
        {
            let mut view = mmat.as_mut();
            for i in 0..5 {
                view[(i, i)] = 1.0;
            }
        }

        // Verify through immutable view
        {
            let view = mmat.as_ref();
            for i in 0..5 {
                for j in 0..5 {
                    if i == j {
                        assert_eq!(view[(i, j)], 1.0);
                    } else {
                        assert_eq!(view[(i, j)], 0.0);
                    }
                }
            }
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_f32() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_f32.oxiblas");

        {
            let mut mmat = MmapMatMut::<f32>::create(&path, 3, 3).unwrap();
            mmat[(0, 0)] = 1.0f32;
            mmat[(1, 1)] = 2.0f32;
            mmat[(2, 2)] = 3.0f32;
            mmat.flush().unwrap();
        }

        {
            let mmat = MmapMat::<f32>::open(&path).unwrap();
            assert_eq!(mmat[(0, 0)], 1.0f32);
            assert_eq!(mmat[(1, 1)], 2.0f32);
            assert_eq!(mmat[(2, 2)], 3.0f32);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_type_mismatch() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_type_mismatch.oxiblas");

        // Create as f64
        {
            let _mmat = MmapMatMut::<f64>::create(&path, 5, 5).unwrap();
        }

        // Try to open as f32 - should fail
        {
            let result = MmapMat::<f32>::open(&path);
            assert!(result.is_err());
            if let Err(MmapError::TypeMismatch { expected, found }) = result {
                assert_eq!(expected, ElementType::F32);
                assert_eq!(found, ElementType::F64);
            } else {
                panic!("Expected TypeMismatch error");
            }
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_builder() {
        use crate::Mat;

        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_builder.oxiblas");

        // Create a regular matrix
        let mat = Mat::<f64>::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        // Build mmap from matrix
        {
            let builder = MmapBuilder::<f64>::new(2, 3);
            let mmat = builder.from_mat(&path, &mat.as_ref()).unwrap();
            assert_eq!(mmat.shape(), (2, 3));
        }

        // Verify
        {
            let mmat = MmapMat::<f64>::open(&path).unwrap();
            assert_eq!(mmat[(0, 0)], 1.0);
            assert_eq!(mmat[(0, 2)], 3.0);
            assert_eq!(mmat[(1, 0)], 4.0);
            assert_eq!(mmat[(1, 2)], 6.0);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_read_dimensions() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_read_dims.oxiblas");

        {
            let _mmat = MmapMatMut::<f64>::create(&path, 100, 200).unwrap();
        }

        let (nrows, ncols) = read_dimensions(&path).unwrap();
        assert_eq!(nrows, 100);
        assert_eq!(ncols, 200);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_large_matrix() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_large.oxiblas");

        let nrows = 1000;
        let ncols = 500;

        // Create large matrix
        {
            let mut mmat = MmapMatMut::<f64>::create(&path, nrows, ncols).unwrap();

            // Set diagonal
            for i in 0..nrows.min(ncols) {
                mmat[(i, i)] = (i + 1) as f64;
            }

            // Set corners
            mmat[(0, 0)] = -1.0;
            mmat[(nrows - 1, ncols - 1)] = -2.0;

            mmat.flush().unwrap();
        }

        // Verify
        {
            let mmat = MmapMat::<f64>::open(&path).unwrap();
            assert_eq!(mmat.shape(), (nrows, ncols));
            assert_eq!(mmat[(0, 0)], -1.0);
            assert_eq!(mmat[(nrows - 1, ncols - 1)], -2.0);
            assert_eq!(mmat[(100, 100)], 101.0);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_fill() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_fill.oxiblas");

        {
            let mut mmat = MmapMatMut::<f64>::create(&path, 5, 5).unwrap();
            mmat.fill(42.0);
            mmat.flush().unwrap();
        }

        {
            let mmat = MmapMat::<f64>::open(&path).unwrap();
            for i in 0..5 {
                for j in 0..5 {
                    assert_eq!(mmat[(i, j)], 42.0);
                }
            }
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_write_mat() {
        use crate::Mat;

        let dir = std::env::temp_dir();
        let path = dir.join("test_write_mat.oxiblas");

        let mat = Mat::<f64>::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        write_mat(&path, &mat.as_ref()).unwrap();

        let mmat = MmapMat::<f64>::open(&path).unwrap();
        assert_eq!(mmat[(0, 0)], 1.0);
        assert_eq!(mmat[(0, 1)], 2.0);
        assert_eq!(mmat[(1, 0)], 3.0);
        assert_eq!(mmat[(1, 1)], 4.0);

        std::fs::remove_file(path).ok();
    }
}
