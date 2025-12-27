//! Banded matrix storage for matrices with limited bandwidth.
//!
//! Banded matrices store only the non-zero diagonals near the main diagonal,
//! reducing memory usage for matrices with a banded structure (common in
//! finite difference and finite element methods).
//!
//! # BLAS Storage Layout
//!
//! Following BLAS conventions, a banded matrix with `kl` subdiagonals and
//! `ku` superdiagonals is stored in a 2D array of size `(kl + ku + 1) × n`
//! where `n` is the number of columns.
//!
//! For a general banded matrix A(i,j), the element is stored at:
//! ```text
//! band_storage[ku + i - j, j]
//! ```
//!
//! # Example
//!
//! For a 5×5 tridiagonal matrix (kl=1, ku=1):
//! ```text
//! [a00  a01   0    0    0 ]     [ *  a01  a12  a23  a34]
//! [a10  a11  a12   0    0 ] --> [a00 a11  a22  a33  a44]  (band storage)
//! [ 0   a21  a22  a23   0 ]     [a10 a21  a32  a43   * ]
//! [ 0    0   a32  a33  a34]
//! [ 0    0    0   a43  a44]
//! ```

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Scalar;

/// A banded matrix using BLAS-style band storage.
///
/// # Storage
///
/// The matrix is stored in a compact 2D array where each diagonal occupies
/// one row. The number of stored rows is `kl + ku + 1` (subdiagonals +
/// main diagonal + superdiagonals).
///
/// # Example
///
/// ```
/// use oxiblas_matrix::banded::BandedMat;
///
/// // Create a 5x5 tridiagonal matrix (kl=1, ku=1)
/// let mut bm: BandedMat<f64> = BandedMat::zeros(5, 5, 1, 1);
///
/// // Set diagonal elements
/// for i in 0..5 {
///     bm.set(i, i, 2.0); // Main diagonal
///     if i > 0 {
///         bm.set(i, i - 1, -1.0); // Subdiagonal
///     }
///     if i < 4 {
///         bm.set(i, i + 1, -1.0); // Superdiagonal
///     }
/// }
///
/// assert_eq!(bm.get(0, 0), Some(&2.0));
/// assert_eq!(bm.get(1, 0), Some(&-1.0));
/// assert_eq!(bm.get(0, 1), Some(&-1.0));
/// assert_eq!(bm.get(0, 2), None); // Outside bandwidth
/// ```
#[derive(Clone)]
pub struct BandedMat<T: Scalar> {
    /// Band storage: (kl + ku + 1) × n, column-major.
    data: AlignedVec<T>,
    /// Number of rows in the logical matrix.
    nrows: usize,
    /// Number of columns in the logical matrix.
    ncols: usize,
    /// Number of subdiagonals (below main diagonal).
    kl: usize,
    /// Number of superdiagonals (above main diagonal).
    ku: usize,
    /// Leading dimension of band storage (rows in band storage).
    ldab: usize,
}

impl<T: Scalar> BandedMat<T> {
    /// Creates a new banded matrix filled with zeros.
    ///
    /// # Parameters
    /// - `nrows`: Number of rows in the logical matrix
    /// - `ncols`: Number of columns
    /// - `kl`: Number of subdiagonals
    /// - `ku`: Number of superdiagonals
    pub fn zeros(nrows: usize, ncols: usize, kl: usize, ku: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let ldab = kl + ku + 1;
        let total = ldab * ncols;

        BandedMat {
            data: AlignedVec::zeros(total),
            nrows,
            ncols,
            kl,
            ku,
            ldab,
        }
    }

    /// Creates a new banded matrix filled with a specific value.
    pub fn filled(nrows: usize, ncols: usize, kl: usize, ku: usize, value: T) -> Self {
        let ldab = kl + ku + 1;
        let total = ldab * ncols;

        BandedMat {
            data: AlignedVec::filled(total, value),
            nrows,
            ncols,
            kl,
            ku,
            ldab,
        }
    }

    /// Creates a banded matrix from a slice in BLAS band storage format.
    ///
    /// # Parameters
    /// - `nrows`: Number of rows in the logical matrix
    /// - `ncols`: Number of columns
    /// - `kl`: Number of subdiagonals
    /// - `ku`: Number of superdiagonals
    /// - `data`: Band storage data in column-major order
    ///
    /// # Panics
    /// Panics if the slice length doesn't match `(kl + ku + 1) * ncols`.
    pub fn from_slice(nrows: usize, ncols: usize, kl: usize, ku: usize, data: &[T]) -> Self {
        let ldab = kl + ku + 1;
        let expected = ldab * ncols;
        assert_eq!(
            data.len(),
            expected,
            "Slice length must equal (kl + ku + 1) * ncols = {}",
            expected
        );

        BandedMat {
            data: AlignedVec::from_slice(data),
            nrows,
            ncols,
            kl,
            ku,
            ldab,
        }
    }

    /// Returns the number of rows in the logical matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns in the logical matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the shape (nrows, ncols) of the logical matrix.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the number of subdiagonals.
    #[inline]
    pub fn kl(&self) -> usize {
        self.kl
    }

    /// Returns the number of superdiagonals.
    #[inline]
    pub fn ku(&self) -> usize {
        self.ku
    }

    /// Returns the bandwidth (kl + ku + 1).
    #[inline]
    pub fn bandwidth(&self) -> usize {
        self.ldab
    }

    /// Returns the leading dimension of band storage.
    #[inline]
    pub fn ldab(&self) -> usize {
        self.ldab
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    /// Returns true if the matrix is square.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Checks if an element (row, col) is within the bandwidth.
    #[inline]
    pub fn in_band(&self, row: usize, col: usize) -> bool {
        if row >= self.nrows || col >= self.ncols {
            return false;
        }

        // Element (i, j) is in band if:
        // j - ku <= i <= j + kl
        // Rearranging: j >= i - kl and j <= i + ku
        // Or equivalently: i - j <= kl and j - i <= ku

        let diff = row as isize - col as isize;
        diff >= -(self.ku as isize) && diff <= self.kl as isize
    }

    /// Computes the index in band storage for element (row, col).
    ///
    /// Returns `None` if the element is outside the bandwidth.
    #[inline]
    pub fn band_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.in_band(row, col) {
            return None;
        }

        // BLAS storage: band_row = ku + row - col
        // Index in column-major: band_row + col * ldab
        let band_row = self.ku + row - col;
        Some(band_row + col * self.ldab)
    }

    /// Returns a reference to the element at (row, col).
    ///
    /// Returns `None` if the element is outside the bandwidth.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.band_index(row, col).map(|idx| &self.data[idx])
    }

    /// Returns a mutable reference to the element at (row, col).
    ///
    /// Returns `None` if the element is outside the bandwidth.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.band_index(row, col).map(|idx| &mut self.data[idx])
    }

    /// Sets the element at (row, col).
    ///
    /// # Panics
    /// Panics if the element is outside the bandwidth.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let idx = self
            .band_index(row, col)
            .expect("Element outside bandwidth");
        self.data[idx] = value;
    }

    /// Returns a pointer to the band storage data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the band storage data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Returns the band storage as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns the band storage as a mutable slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns a specific band (diagonal) as a slice.
    ///
    /// `band_idx = 0` is the main diagonal, positive values are superdiagonals,
    /// negative values are subdiagonals.
    ///
    /// Returns the elements of the diagonal along with the starting column index.
    pub fn get_band(&self, band_idx: isize) -> Option<(&[T], usize)> {
        if band_idx < -(self.kl as isize) || band_idx > self.ku as isize {
            return None;
        }

        // Band row in storage: ku - band_idx
        let storage_row = (self.ku as isize - band_idx) as usize;

        // Determine the valid range
        let start_col = if band_idx >= 0 { band_idx as usize } else { 0 };

        let start_row = if band_idx >= 0 {
            0
        } else {
            (-band_idx) as usize
        };

        let len = (self.nrows - start_row).min(self.ncols - start_col);

        if len == 0 {
            return Some((&[], start_col));
        }

        // Collect the diagonal elements
        // Note: They're not contiguous in memory, so we can't return a simple slice
        // Instead, we return the starting pointer and the caller can access with stride ldab
        let start_idx = storage_row + start_col * self.ldab;
        let end_idx = storage_row + (start_col + len - 1) * self.ldab + 1;

        // This is a workaround - we return the underlying slice section
        // The actual diagonal elements are at indices start_idx, start_idx + ldab, ...
        Some((&self.data.as_slice()[start_idx..end_idx], start_col))
    }

    /// Returns the diagonal elements.
    pub fn diagonal(&self) -> Vec<T> {
        let len = self.nrows.min(self.ncols);
        (0..len).filter_map(|i| self.get(i, i).copied()).collect()
    }

    /// Sets the diagonal elements.
    pub fn set_diagonal(&mut self, diag: &[T]) {
        let len = self.nrows.min(self.ncols);
        assert!(
            diag.len() <= len,
            "Diagonal length exceeds matrix dimension"
        );

        for (i, &val) in diag.iter().enumerate() {
            if i < len {
                self.set(i, i, val);
            }
        }
    }

    /// Converts to a full dense matrix.
    pub fn to_dense(&self) -> crate::Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let mut mat = crate::Mat::zeros(self.nrows, self.ncols);

        for j in 0..self.ncols {
            // Only iterate over rows within the band
            let start_row = j.saturating_sub(self.ku);
            let end_row = (j + self.kl + 1).min(self.nrows);

            for i in start_row..end_row {
                if let Some(&val) = self.get(i, j) {
                    mat[(i, j)] = val;
                }
            }
        }

        mat
    }

    /// Creates a banded matrix from a dense matrix.
    ///
    /// Elements outside the specified bandwidth are ignored (assumed zero).
    pub fn from_dense(mat: &crate::MatRef<'_, T>, kl: usize, ku: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let mut banded = Self::zeros(nrows, ncols, kl, ku);

        for j in 0..ncols {
            let start_row = j.saturating_sub(ku);
            let end_row = (j + kl + 1).min(nrows);

            for i in start_row..end_row {
                banded.set(i, j, mat[(i, j)]);
            }
        }

        banded
    }

    /// Fills all band elements with a value.
    pub fn fill(&mut self, value: T) {
        for j in 0..self.ncols {
            let start_row = j.saturating_sub(self.ku);
            let end_row = (j + self.kl + 1).min(self.nrows);

            for i in start_row..end_row {
                self.set(i, j, value);
            }
        }
    }

    /// Scales all band elements by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for j in 0..self.ncols {
            let start_row = j.saturating_sub(self.ku);
            let end_row = (j + self.kl + 1).min(self.nrows);

            for i in start_row..end_row {
                if let Some(val) = self.get_mut(i, j) {
                    *val *= alpha;
                }
            }
        }
    }

    /// Creates a symmetric banded matrix (SB storage) from upper or lower triangle.
    ///
    /// For symmetric banded storage, we only store `k + 1` rows where `k` is
    /// the number of subdiagonals (or superdiagonals, they're equal).
    pub fn to_symmetric_banded(&self) -> SymmetricBandedMat<T>
    where
        T: bytemuck::Zeroable,
    {
        assert!(
            self.kl == self.ku,
            "Matrix must have equal number of sub- and superdiagonals for symmetric storage"
        );
        assert!(
            self.is_square(),
            "Matrix must be square for symmetric storage"
        );

        SymmetricBandedMat::from_banded(self)
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for BandedMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "BandedMat {}×{} (kl={}, ku={}) {{",
            self.nrows, self.ncols, self.kl, self.ku
        )?;

        let max_rows = 8.min(self.nrows);
        let max_cols = 8.min(self.ncols);

        for i in 0..max_rows {
            write!(f, "  [")?;
            for j in 0..max_cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                match self.get(i, j) {
                    Some(v) => write!(f, "{:8.4?}", v)?,
                    None => write!(f, "      0 ")?,
                }
            }
            if self.ncols > max_cols {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.nrows > max_rows {
            writeln!(f, "  ...")?;
        }
        write!(f, "}}")
    }
}

/// A symmetric banded matrix using BLAS-style storage.
///
/// For a symmetric banded matrix with bandwidth `k`, we only need to store
/// either the upper or lower triangle, requiring `(k + 1) × n` storage.
///
/// # Storage
///
/// **Upper storage**: Each column j stores elements A(max(0, j-k):j, j).
/// **Lower storage**: Each column j stores elements A(j:min(n-1, j+k), j).
#[derive(Clone)]
pub struct SymmetricBandedMat<T: Scalar> {
    /// Band storage: (k + 1) × n, column-major.
    data: AlignedVec<T>,
    /// Matrix dimension (n × n).
    n: usize,
    /// Half-bandwidth (number of sub/superdiagonals).
    k: usize,
    /// Upper or lower storage.
    uplo: super::packed::TriangularKind,
    /// Leading dimension.
    ldab: usize,
}

impl<T: Scalar> SymmetricBandedMat<T> {
    /// Creates a new symmetric banded matrix filled with zeros.
    ///
    /// # Parameters
    /// - `n`: Matrix dimension
    /// - `k`: Half-bandwidth (number of sub/superdiagonals)
    /// - `uplo`: Whether to store upper or lower triangle
    pub fn zeros(n: usize, k: usize, uplo: super::packed::TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let ldab = k + 1;
        let total = ldab * n;

        SymmetricBandedMat {
            data: AlignedVec::zeros(total),
            n,
            k,
            uplo,
            ldab,
        }
    }

    /// Creates a symmetric banded matrix from a general banded matrix.
    pub fn from_banded(banded: &BandedMat<T>) -> Self
    where
        T: bytemuck::Zeroable,
    {
        assert!(banded.is_square(), "Matrix must be square");
        assert_eq!(
            banded.kl(),
            banded.ku(),
            "Matrix must have equal sub/superdiagonals"
        );

        let n = banded.nrows();
        let k = banded.kl();
        let uplo = super::packed::TriangularKind::Upper;

        let mut sb = Self::zeros(n, k, uplo);

        for j in 0..n {
            let start_i = j.saturating_sub(k);
            for i in start_i..=j {
                if let Some(&val) = banded.get(i, j) {
                    sb.set(i, j, val);
                }
            }
        }

        sb
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the half-bandwidth.
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the storage type (upper or lower).
    #[inline]
    pub fn uplo(&self) -> super::packed::TriangularKind {
        self.uplo
    }

    /// Returns the leading dimension.
    #[inline]
    pub fn ldab(&self) -> usize {
        self.ldab
    }

    /// Checks if an element (row, col) is within the bandwidth.
    #[inline]
    pub fn in_band(&self, row: usize, col: usize) -> bool {
        if row >= self.n || col >= self.n {
            return false;
        }

        let diff = (row as isize - col as isize).unsigned_abs();
        diff <= self.k
    }

    /// Computes the index in band storage for element (row, col).
    ///
    /// For symmetric matrices, accessing (i, j) where i > j returns the same
    /// value as (j, i).
    pub fn band_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.in_band(row, col) {
            return None;
        }

        match self.uplo {
            super::packed::TriangularKind::Upper => {
                // Upper storage: element (i, j) where i <= j is at band_row = k + i - j
                let (i, j) = if row <= col { (row, col) } else { (col, row) };
                let band_row = self.k + i - j;
                Some(band_row + j * self.ldab)
            }
            super::packed::TriangularKind::Lower => {
                // Lower storage: element (i, j) where i >= j is at band_row = i - j
                let (i, j) = if row >= col { (row, col) } else { (col, row) };
                let band_row = i - j;
                Some(band_row + j * self.ldab)
            }
        }
    }

    /// Returns a reference to the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.band_index(row, col).map(|idx| &self.data[idx])
    }

    /// Sets the element at (row, col).
    ///
    /// For symmetric matrices, this also implicitly sets (col, row).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let idx = self
            .band_index(row, col)
            .expect("Element outside bandwidth");
        self.data[idx] = value;
    }

    /// Returns a pointer to the band storage.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the band storage.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Converts to a full dense symmetric matrix.
    pub fn to_dense(&self) -> crate::Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let mut mat = crate::Mat::zeros(self.n, self.n);

        for j in 0..self.n {
            let start_i = j.saturating_sub(self.k);
            let end_i = (j + self.k + 1).min(self.n);

            for i in start_i..end_i {
                if let Some(&val) = self.get(i, j) {
                    mat[(i, j)] = val;
                    if i != j {
                        mat[(j, i)] = val; // Symmetric
                    }
                }
            }
        }

        mat
    }

    /// Returns the diagonal elements.
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.n)
            .filter_map(|i| self.get(i, i).copied())
            .collect()
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for SymmetricBandedMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "SymmetricBandedMat {}×{} (k={}, {:?}) {{",
            self.n, self.n, self.k, self.uplo
        )?;

        let max_dim = 8.min(self.n);

        for i in 0..max_dim {
            write!(f, "  [")?;
            for j in 0..max_dim {
                if j > 0 {
                    write!(f, ", ")?;
                }
                match self.get(i, j) {
                    Some(v) => write!(f, "{:8.4?}", v)?,
                    None => write!(f, "      0 ")?,
                }
            }
            if self.n > max_dim {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.n > max_dim {
            writeln!(f, "  ...")?;
        }
        write!(f, "}}")
    }
}

/// A view into banded matrix data.
#[derive(Clone, Copy)]
pub struct BandedRef<'a, T: Scalar> {
    /// Pointer to band storage.
    ptr: *const T,
    /// Number of rows in logical matrix.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Number of subdiagonals.
    kl: usize,
    /// Number of superdiagonals.
    ku: usize,
    /// Leading dimension.
    ldab: usize,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Scalar> BandedRef<'a, T> {
    /// Creates a new banded reference.
    #[inline]
    pub fn new(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        kl: usize,
        ku: usize,
        ldab: usize,
    ) -> Self {
        BandedRef {
            ptr,
            nrows,
            ncols,
            kl,
            ku,
            ldab,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns kl (subdiagonals).
    #[inline]
    pub fn kl(&self) -> usize {
        self.kl
    }

    /// Returns ku (superdiagonals).
    #[inline]
    pub fn ku(&self) -> usize {
        self.ku
    }

    /// Returns the leading dimension.
    #[inline]
    pub fn ldab(&self) -> usize {
        self.ldab
    }

    /// Checks if element is in band.
    #[inline]
    pub fn in_band(&self, row: usize, col: usize) -> bool {
        if row >= self.nrows || col >= self.ncols {
            return false;
        }
        let diff = row as isize - col as isize;
        diff >= -(self.ku as isize) && diff <= self.kl as isize
    }

    /// Computes band index.
    #[inline]
    pub fn band_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.in_band(row, col) {
            return None;
        }
        let band_row = self.ku + row - col;
        Some(band_row + col * self.ldab)
    }

    /// Returns element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.band_index(row, col)
            .map(|idx| unsafe { &*self.ptr.add(idx) })
    }

    /// Returns pointer to band storage.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

unsafe impl<'a, T: Scalar + Send> Send for BandedRef<'a, T> {}
unsafe impl<'a, T: Scalar + Sync> Sync for BandedRef<'a, T> {}

/// A mutable view into banded matrix data.
pub struct BandedMut<'a, T: Scalar> {
    /// Pointer to band storage.
    ptr: *mut T,
    /// Number of rows in logical matrix.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Number of subdiagonals.
    kl: usize,
    /// Number of superdiagonals.
    ku: usize,
    /// Leading dimension.
    ldab: usize,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T: Scalar> BandedMut<'a, T> {
    /// Creates a new mutable banded reference.
    #[inline]
    pub fn new(ptr: *mut T, nrows: usize, ncols: usize, kl: usize, ku: usize, ldab: usize) -> Self {
        BandedMut {
            ptr,
            nrows,
            ncols,
            kl,
            ku,
            ldab,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns kl (subdiagonals).
    #[inline]
    pub fn kl(&self) -> usize {
        self.kl
    }

    /// Returns ku (superdiagonals).
    #[inline]
    pub fn ku(&self) -> usize {
        self.ku
    }

    /// Checks if element is in band.
    #[inline]
    pub fn in_band(&self, row: usize, col: usize) -> bool {
        if row >= self.nrows || col >= self.ncols {
            return false;
        }
        let diff = row as isize - col as isize;
        diff >= -(self.ku as isize) && diff <= self.kl as isize
    }

    /// Computes band index.
    #[inline]
    pub fn band_index(&self, row: usize, col: usize) -> Option<usize> {
        if !self.in_band(row, col) {
            return None;
        }
        let band_row = self.ku + row - col;
        Some(band_row + col * self.ldab)
    }

    /// Returns element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.band_index(row, col)
            .map(|idx| unsafe { &*self.ptr.add(idx) })
    }

    /// Returns mutable element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.band_index(row, col)
            .map(|idx| unsafe { &mut *self.ptr.add(idx) })
    }

    /// Sets element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let idx = self
            .band_index(row, col)
            .expect("Element outside bandwidth");
        unsafe {
            *self.ptr.add(idx) = value;
        }
    }

    /// Creates an immutable reborrow.
    #[inline]
    pub fn rb(&self) -> BandedRef<'_, T> {
        BandedRef::new(
            self.ptr, self.nrows, self.ncols, self.kl, self.ku, self.ldab,
        )
    }

    /// Creates a mutable reborrow.
    #[inline]
    pub fn rb_mut(&mut self) -> BandedMut<'_, T> {
        BandedMut::new(
            self.ptr, self.nrows, self.ncols, self.kl, self.ku, self.ldab,
        )
    }
}

unsafe impl<'a, T: Scalar + Send> Send for BandedMut<'a, T> {}
unsafe impl<'a, T: Scalar + Sync> Sync for BandedMut<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banded_basic() {
        // 4x4 tridiagonal matrix (kl=1, ku=1)
        let mut bm: BandedMat<f64> = BandedMat::zeros(4, 4, 1, 1);

        // Set Laplacian-like matrix
        // [2 -1  0  0]
        // [-1 2 -1  0]
        // [0 -1  2 -1]
        // [0  0 -1  2]
        for i in 0..4 {
            bm.set(i, i, 2.0);
            if i > 0 {
                bm.set(i, i - 1, -1.0);
            }
            if i < 3 {
                bm.set(i, i + 1, -1.0);
            }
        }

        // Check values
        assert_eq!(bm.get(0, 0), Some(&2.0));
        assert_eq!(bm.get(0, 1), Some(&-1.0));
        assert_eq!(bm.get(1, 0), Some(&-1.0));
        assert_eq!(bm.get(1, 1), Some(&2.0));
        assert_eq!(bm.get(0, 2), None); // Outside bandwidth
        assert_eq!(bm.get(2, 0), None);

        // Check diagonal
        let diag = bm.diagonal();
        assert_eq!(diag, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_banded_in_band() {
        let bm: BandedMat<f64> = BandedMat::zeros(5, 5, 2, 1);

        // kl=2, ku=1 means:
        // - 2 subdiagonals (i - j <= 2)
        // - 1 superdiagonal (j - i <= 1)

        assert!(bm.in_band(0, 0)); // Main diagonal
        assert!(bm.in_band(0, 1)); // Superdiagonal
        assert!(!bm.in_band(0, 2)); // Beyond ku

        assert!(bm.in_band(1, 0)); // First subdiagonal
        assert!(bm.in_band(2, 0)); // Second subdiagonal
        assert!(!bm.in_band(3, 0)); // Beyond kl

        assert!(!bm.in_band(5, 0)); // Out of bounds
        assert!(!bm.in_band(0, 5));
    }

    #[test]
    fn test_banded_to_dense() {
        let mut bm: BandedMat<f64> = BandedMat::zeros(3, 3, 1, 1);
        bm.set(0, 0, 1.0);
        bm.set(0, 1, 2.0);
        bm.set(1, 0, 3.0);
        bm.set(1, 1, 4.0);
        bm.set(1, 2, 5.0);
        bm.set(2, 1, 6.0);
        bm.set(2, 2, 7.0);

        let dense = bm.to_dense();

        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 1)], 2.0);
        assert_eq!(dense[(0, 2)], 0.0);
        assert_eq!(dense[(1, 0)], 3.0);
        assert_eq!(dense[(1, 1)], 4.0);
        assert_eq!(dense[(1, 2)], 5.0);
        assert_eq!(dense[(2, 0)], 0.0);
        assert_eq!(dense[(2, 1)], 6.0);
        assert_eq!(dense[(2, 2)], 7.0);
    }

    #[test]
    fn test_banded_from_dense() {
        use crate::Mat;

        let dense = Mat::from_rows(&[
            &[1.0, 2.0, 0.0, 0.0],
            &[3.0, 4.0, 5.0, 0.0],
            &[0.0, 6.0, 7.0, 8.0],
            &[0.0, 0.0, 9.0, 10.0],
        ]);

        let bm = BandedMat::from_dense(&dense.as_ref(), 1, 1);

        assert_eq!(bm.get(0, 0), Some(&1.0));
        assert_eq!(bm.get(0, 1), Some(&2.0));
        assert_eq!(bm.get(1, 0), Some(&3.0));
        assert_eq!(bm.get(1, 1), Some(&4.0));
        assert_eq!(bm.get(1, 2), Some(&5.0));
        assert_eq!(bm.get(2, 1), Some(&6.0));
        assert_eq!(bm.get(2, 2), Some(&7.0));
        assert_eq!(bm.get(2, 3), Some(&8.0));
        assert_eq!(bm.get(3, 2), Some(&9.0));
        assert_eq!(bm.get(3, 3), Some(&10.0));
    }

    #[test]
    fn test_banded_pentadiagonal() {
        // 6x6 pentadiagonal (kl=2, ku=2)
        let mut bm: BandedMat<f64> = BandedMat::zeros(6, 6, 2, 2);

        for i in 0..6 {
            bm.set(i, i, 4.0); // Main diagonal
        }
        for i in 1..6 {
            bm.set(i, i - 1, -1.0); // First subdiagonal
            bm.set(i - 1, i, -1.0); // First superdiagonal
        }
        for i in 2..6 {
            bm.set(i, i - 2, -0.5); // Second subdiagonal
            bm.set(i - 2, i, -0.5); // Second superdiagonal
        }

        // Check bandwidth
        assert_eq!(bm.bandwidth(), 5); // kl + ku + 1 = 2 + 2 + 1

        // Check values
        assert_eq!(bm.get(0, 0), Some(&4.0));
        assert_eq!(bm.get(2, 0), Some(&-0.5));
        assert_eq!(bm.get(0, 2), Some(&-0.5));
        assert_eq!(bm.get(3, 0), None); // Beyond kl
    }

    #[test]
    fn test_symmetric_banded() {
        use crate::packed::TriangularKind;

        let mut sb: SymmetricBandedMat<f64> =
            SymmetricBandedMat::zeros(4, 1, TriangularKind::Upper);

        // Set symmetric tridiagonal
        for i in 0..4 {
            sb.set(i, i, 2.0);
        }
        for i in 0..3 {
            sb.set(i, i + 1, -1.0); // Sets both (i, i+1) and (i+1, i)
        }

        // Check symmetry
        assert_eq!(sb.get(0, 1), Some(&-1.0));
        assert_eq!(sb.get(1, 0), Some(&-1.0)); // Same storage location

        let dense = sb.to_dense();
        assert_eq!(dense[(0, 1)], -1.0);
        assert_eq!(dense[(1, 0)], -1.0);
    }

    #[test]
    fn test_banded_scale() {
        let mut bm: BandedMat<f64> = BandedMat::zeros(3, 3, 1, 1);
        bm.set(0, 0, 1.0);
        bm.set(0, 1, 2.0);
        bm.set(1, 0, 3.0);
        bm.set(1, 1, 4.0);
        bm.set(1, 2, 5.0);
        bm.set(2, 1, 6.0);
        bm.set(2, 2, 7.0);

        bm.scale(2.0);

        assert_eq!(bm.get(0, 0), Some(&2.0));
        assert_eq!(bm.get(1, 1), Some(&8.0));
        assert_eq!(bm.get(2, 2), Some(&14.0));
    }

    #[test]
    fn test_banded_ref() {
        let mut bm: BandedMat<f64> = BandedMat::zeros(3, 3, 1, 1);
        bm.set(0, 0, 1.0);
        bm.set(1, 1, 2.0);
        bm.set(2, 2, 3.0);

        let bref = BandedRef::new(
            bm.as_ptr(),
            bm.nrows(),
            bm.ncols(),
            bm.kl(),
            bm.ku(),
            bm.ldab(),
        );

        assert_eq!(bref.get(0, 0), Some(&1.0));
        assert_eq!(bref.get(1, 1), Some(&2.0));
        assert_eq!(bref.get(2, 2), Some(&3.0));
    }

    #[test]
    fn test_banded_mut() {
        let mut bm: BandedMat<f64> = BandedMat::zeros(3, 3, 1, 1);

        {
            let mut bmut = BandedMut::new(
                bm.as_mut_ptr(),
                bm.nrows(),
                bm.ncols(),
                bm.kl(),
                bm.ku(),
                bm.ldab(),
            );

            bmut.set(0, 0, 10.0);
            bmut.set(1, 1, 20.0);
        }

        assert_eq!(bm.get(0, 0), Some(&10.0));
        assert_eq!(bm.get(1, 1), Some(&20.0));
    }

    #[test]
    fn test_banded_nonsquare() {
        // 4x6 rectangular banded matrix
        let mut bm: BandedMat<f64> = BandedMat::zeros(4, 6, 1, 2);

        assert_eq!(bm.shape(), (4, 6));
        assert!(!bm.is_square());

        // Set some values
        bm.set(0, 0, 1.0);
        bm.set(0, 1, 2.0);
        bm.set(0, 2, 3.0);
        bm.set(1, 0, 4.0);
        bm.set(1, 1, 5.0);
        bm.set(1, 2, 6.0);
        bm.set(1, 3, 7.0);

        let dense = bm.to_dense();
        assert_eq!(dense.shape(), (4, 6));
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 2)], 3.0);
        assert_eq!(dense[(1, 3)], 7.0);
    }
}
