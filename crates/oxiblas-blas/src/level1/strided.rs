//! Strided vector views and strided Level 1 BLAS routines.
//!
//! Reference BLAS defines every Level 1/2 routine in terms of an increment
//! (`incx`/`incy`) that describes how far apart, in elements, consecutive
//! logical vector entries live in memory. The contiguous entry points in the
//! sibling modules (`dot`, `axpy`, …) hard-code an increment of 1. This module
//! restores the full reference contract:
//!
//! * **Positive stride** — logical element `i` lives at physical offset
//!   `i * stride`.
//! * **Negative stride** — reference BLAS starts at physical offset
//!   `(1 - n) * incx` and walks *backwards*, so the *first* logical element
//!   reads the *highest* physical address. Concretely logical element `i` lives
//!   at `(n - 1 - i) * |stride|`. This is the exact convention Netlib uses so
//!   that, when both `x` and `y` carry negative increments, the i-th product in
//!   e.g. `ddot` pairs the same logical positions.
//! * **Zero stride** — logical element `i` always maps to physical offset 0,
//!   i.e. the same element is read (or written) `n` times. Reference BLAS
//!   (Netlib 3.12) permits this for the "product/copy" family
//!   (DOT/AXPY/COPY/SWAP/ROT/ROTM) and — since LAPACK 3.10's `?nrm2.f90` — for
//!   NRM2, which also walks a negative increment backwards (the norm of `n`
//!   reads of `x[0]` is `sqrt(n)·|x[0]|`). For SCAL/ASUM/IAMAX a non-positive
//!   increment is instead defined as a no-op / zero result, and the routines
//!   below honour that split.
//!
//! # CBLAS integration point
//!
//! The CBLAS compatibility layer (`crate::cblas`) receives raw
//! `*const T`/`*mut T` pointers together with an `i32` increment. It builds a
//! [`StridedSlice`] / [`StridedSliceMut`] via the `unsafe`
//! [`StridedSlice::from_raw_parts`] / [`StridedSliceMut::from_raw_parts`]
//! constructors and forwards to the `*_strided` routines here:
//! `cblas_scnrm2`/`cblas_dznrm2` ([`cnrm2_strided`]), `cblas_srot`/`cblas_drot`
//! ([`rot_strided`]), `cblas_srotm`/`cblas_drotm` ([`rotm_strided`]) and
//! `cblas_csrot`/`cblas_zdrot` ([`zdrot_strided`]). `cblas_sdsdot` and
//! `cblas_dsdot` build the same views for their strided branch and run the
//! reference double-precision loop over them. The raw-buffer contract is
//! exactly the reference one:
//!
//! * pass the array base pointer the caller holds — the *lowest* address the
//!   vector touches — for every sign of the increment;
//! * the view spans `(n - 1)·|inc| + 1` elements (just `1` when `inc == 0`);
//! * a negative increment starts at offset `(n - 1)·|inc|` and walks down to
//!   offset 0; a zero increment re-uses element 0 on every iteration;
//! * an extent that overflows `usize`, or whose elements span more than
//!   `isize::MAX` bytes, cannot describe a real allocation and yields an empty
//!   view. With the `i32` CBLAS arguments the first case needs a 32-bit
//!   `usize`, while the second is reachable on 64-bit targets too (`n` and
//!   `|inc|` near `2^31`). A wrapper that builds two views therefore checks both
//!   lengths before calling a paired routine, which asserts equal lengths.
//!
//! That yields genuine non-unit-stride and negative-increment correctness —
//! e.g. extracting a column of a row-major matrix — instead of merely
//! re-slicing a contiguous buffer.

use num_traits::{Float, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};

use super::RotmParams;
use super::nrm2::{nrm2_finalize, nrm2_fold};

/// Number of physical elements a strided vector spans.
///
/// For `len` logical elements with signed `stride`, the touched physical region
/// is `[0, (len - 1) * |stride|]`, i.e. `(len - 1) * |stride| + 1` elements
/// (both the positive- and negative-stride mappings stay inside this window).
/// Returns `None` on `usize` overflow so checked constructors can reject absurd
/// inputs without panicking.
#[inline]
fn strided_extent(len: usize, stride: isize) -> Option<usize> {
    if len == 0 {
        return Some(0);
    }
    (len - 1).checked_mul(stride.unsigned_abs())?.checked_add(1)
}

/// `true` when `count` elements of `T` span at most `isize::MAX` bytes: the size
/// limit `core::slice::from_raw_parts` puts on a slice, which no allocation can
/// exceed either. A zero-sized `T` is held to `isize::MAX` elements, stricter than
/// the slice rule but irrelevant for BLAS element types.
#[inline]
pub(crate) fn fits_isize_bytes<T>(count: usize) -> bool {
    count <= isize::MAX.unsigned_abs() / core::mem::size_of::<T>().max(1)
}

/// [`strided_extent`] for a view over a raw pointer: also `None` when the extent
/// spans more than `isize::MAX` bytes ([`fits_isize_bytes`]), so the raw-parts
/// constructors never hand `core::slice::from_raw_parts` an illegal length.
#[inline]
fn raw_parts_extent<T>(len: usize, stride: isize) -> Option<usize> {
    strided_extent(len, stride).filter(|&extent| fits_isize_bytes::<T>(extent))
}

/// An immutable strided view over a contiguous slice, modelling a BLAS vector
/// with increment `incx`.
///
/// # Invariants
///
/// * `len` is the number of *logical* elements.
/// * `stride` is the signed increment between consecutive logical elements,
///   measured in elements (never bytes).
/// * When `len > 0`, `data` spans at least `strided_extent(len, stride)`
///   elements, so every logical index maps in-bounds. All safe constructors
///   enforce this; the `unsafe` [`from_raw_parts`](Self::from_raw_parts) shifts
///   the obligation to the caller.
#[derive(Debug, Clone, Copy)]
pub struct StridedSlice<'a, T> {
    data: &'a [T],
    len: usize,
    stride: isize,
}

impl<'a, T> StridedSlice<'a, T> {
    /// Views a contiguous slice as a unit-stride vector (`incx == 1`).
    #[inline]
    #[must_use]
    pub fn from_slice(data: &'a [T]) -> Self {
        Self {
            len: data.len(),
            data,
            stride: 1,
        }
    }

    /// Builds a strided view over `data`, or `None` if `data` is too short to
    /// hold `len` logical elements at the given `stride`.
    #[inline]
    #[must_use]
    pub fn try_new(data: &'a [T], len: usize, stride: isize) -> Option<Self> {
        match strided_extent(len, stride) {
            Some(extent) if extent <= data.len() => Some(Self { data, len, stride }),
            _ => None,
        }
    }

    /// Builds a strided view directly from a base pointer, a logical length and
    /// a signed stride. This is the natural constructor for the CBLAS layer.
    ///
    /// `ptr` must be the *lowest* address the vector touches, i.e. the array
    /// base that reference BLAS / CBLAS callers already hold: they pass this
    /// same pointer for both signs of the increment. For `stride < 0` the
    /// reference walk begins at offset `(1 - n) * stride` — a *positive* offset
    /// from this base — and descends back to offset 0, so no address below
    /// `ptr` is ever accessed. Pass exactly the user's `x`/`incx` here.
    ///
    /// # Safety
    ///
    /// * `ptr` must be valid for reads of `strided_extent(len, stride)`
    ///   elements and that region must live for `'a`.
    /// * The region must be a single allocated object with no concurrent
    ///   mutation for `'a`.
    /// * An extent that overflows `usize`, or whose elements span more than
    ///   `isize::MAX` bytes, cannot describe a real allocation, so it is not an
    ///   error: such a `(len, stride)` yields an empty view (`len() == 0`) and
    ///   `ptr` is never accessed.
    #[inline]
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize, stride: isize) -> Self {
        let Some(extent) = raw_parts_extent::<T>(len, stride) else {
            // No real allocation can hold this vector (see `# Safety`). The CBLAS
            // `extern "C"` entry points reach this path, where a panic would
            // abort the C caller's process and an oversized slice would be
            // undefined behaviour, so return an empty view instead and leave
            // `ptr` untouched.
            return Self {
                data: &[],
                len: 0,
                stride,
            };
        };
        // SAFETY: contract of this function guarantees `ptr` is valid for
        // `extent` reads for `'a` and points into a single allocation, and
        // `raw_parts_extent` keeps `extent` elements within `isize::MAX` bytes.
        let data = unsafe { core::slice::from_raw_parts(ptr, extent) };
        Self { data, len, stride }
    }

    /// Number of logical elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view has no logical elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Signed stride (increment) of the view.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> isize {
        self.stride
    }

    /// Maps a logical index to its physical offset inside `data`.
    ///
    /// Negative strides reverse the walk so logical 0 is the highest physical
    /// offset (see the module docs for why this matches reference BLAS).
    #[inline]
    fn phys_index(&self, i: usize) -> usize {
        if self.stride >= 0 {
            i * self.stride.unsigned_abs()
        } else {
            (self.len - 1 - i) * self.stride.unsigned_abs()
        }
    }
}

impl<'a, T: Copy> StridedSlice<'a, T> {
    /// Reads logical element `i`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> T {
        self.data[self.phys_index(i)]
    }

    /// Returns the underlying contiguous slice when `stride == 1`, else `None`.
    /// Used to route unit-stride calls through the SIMD-optimised contiguous
    /// kernels so the fast path is bit-identical to the legacy entry points.
    #[inline]
    #[must_use]
    pub fn as_contiguous(&self) -> Option<&'a [T]> {
        if self.stride == 1 {
            Some(&self.data[..self.len])
        } else {
            None
        }
    }
}

/// A mutable strided view over a contiguous slice, modelling a BLAS output
/// vector with increment `incy`.
///
/// Carries the same invariants as [`StridedSlice`]. Because it holds a `&mut`
/// borrow, two distinct `StridedSliceMut` values can never alias through the
/// safe constructors; the `unsafe` [`from_raw_parts`](Self::from_raw_parts)
/// shifts the non-aliasing obligation to the caller (reference BLAS already
/// requires distinct `x`/`y` buffers).
#[derive(Debug)]
pub struct StridedSliceMut<'a, T> {
    data: &'a mut [T],
    len: usize,
    stride: isize,
}

impl<'a, T> StridedSliceMut<'a, T> {
    /// Views a contiguous mutable slice as a unit-stride vector (`incy == 1`).
    #[inline]
    #[must_use]
    pub fn from_slice(data: &'a mut [T]) -> Self {
        let len = data.len();
        Self {
            data,
            len,
            stride: 1,
        }
    }

    /// Builds a mutable strided view over `data`, or `None` if `data` is too
    /// short to hold `len` logical elements at the given `stride`.
    #[inline]
    #[must_use]
    pub fn try_new(data: &'a mut [T], len: usize, stride: isize) -> Option<Self> {
        match strided_extent(len, stride) {
            Some(extent) if extent <= data.len() => Some(Self { data, len, stride }),
            _ => None,
        }
    }

    /// Builds a mutable strided view directly from a base pointer.
    ///
    /// # Safety
    ///
    /// Same obligations as [`StridedSlice::from_raw_parts`] (including its empty
    /// view, with `ptr` never accessed, for an extent that overflows `usize` or
    /// spans more than `isize::MAX` bytes), plus: the region must be uniquely
    /// borrowed (no other live reference, and no aliasing with any other `x`/`y`
    /// passed to the same routine) for `'a`.
    #[inline]
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, stride: isize) -> Self {
        let Some(extent) = raw_parts_extent::<T>(len, stride) else {
            // Unsatisfiable extent: empty view, `ptr` untouched, no panic and no
            // oversized slice (see `StridedSlice::from_raw_parts`).
            return Self {
                data: &mut [],
                len: 0,
                stride,
            };
        };
        // SAFETY: contract of this function guarantees `ptr` is valid for
        // `extent` reads+writes for `'a`, uniquely borrowed, single allocation,
        // and `raw_parts_extent` keeps `extent` elements within `isize::MAX` bytes.
        let data = unsafe { core::slice::from_raw_parts_mut(ptr, extent) };
        Self { data, len, stride }
    }

    /// Number of logical elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view has no logical elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Signed stride (increment) of the view.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> isize {
        self.stride
    }

    /// Maps a logical index to its physical offset inside `data`.
    #[inline]
    fn phys_index(&self, i: usize) -> usize {
        if self.stride >= 0 {
            i * self.stride.unsigned_abs()
        } else {
            (self.len - 1 - i) * self.stride.unsigned_abs()
        }
    }
}

impl<'a, T: Copy> StridedSliceMut<'a, T> {
    /// Reads logical element `i`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> T {
        self.data[self.phys_index(i)]
    }

    /// Writes `value` into logical element `i`.
    #[inline]
    pub fn set(&mut self, i: usize, value: T) {
        let idx = self.phys_index(i);
        self.data[idx] = value;
    }

    /// Returns the underlying contiguous mutable slice when `stride == 1`, else
    /// `None`. Used to route unit-stride calls through the SIMD-optimised
    /// contiguous kernels.
    #[inline]
    pub fn as_contiguous_mut(&mut self) -> Option<&mut [T]> {
        if self.stride == 1 {
            Some(&mut self.data[..self.len])
        } else {
            None
        }
    }
}

// =============================================================================
// Product / copy family: honour negative AND zero increments.
// =============================================================================

/// Strided dot product `Σ x[i]·y[i]` (unconjugated).
///
/// General entry point behind the contiguous [`dot`](super::dot): honours
/// arbitrary positive, negative and zero increments per reference BLAS. A
/// unit-stride call is forwarded to the SIMD-optimised contiguous kernel so it
/// stays bit-identical to the legacy path.
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
#[must_use]
pub fn dot_strided<T: Field>(x: StridedSlice<'_, T>, y: StridedSlice<'_, T>) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 {
        return T::zero();
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous(), y.as_contiguous()) {
        return super::dot(xc, yc);
    }
    let mut acc = T::zero();
    for i in 0..n {
        acc += x.get(i) * y.get(i);
    }
    acc
}

/// Strided conjugated dot product `Σ conj(x[i])·y[i]`.
///
/// General entry point behind the contiguous [`dotc`](super::dotc). For real
/// types this equals [`dot_strided`].
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
#[must_use]
pub fn dotc_strided<T: Field>(x: StridedSlice<'_, T>, y: StridedSlice<'_, T>) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 {
        return T::zero();
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous(), y.as_contiguous()) {
        return super::dotc(xc, yc);
    }
    let mut acc = T::zero();
    for i in 0..n {
        acc += x.get(i).conj() * y.get(i);
    }
    acc
}

/// Strided AXPY `y ← α·x + y`.
///
/// General entry point behind the contiguous [`axpy`](super::axpy): honours
/// arbitrary positive, negative and zero increments.
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
pub fn axpy_strided<T: Field>(alpha: T, x: StridedSlice<'_, T>, mut y: StridedSliceMut<'_, T>) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 || alpha == T::zero() {
        return;
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous(), y.as_contiguous_mut()) {
        super::axpy(alpha, xc, yc);
        return;
    }
    for i in 0..n {
        let updated = alpha * x.get(i) + y.get(i);
        y.set(i, updated);
    }
}

/// Strided COPY `y ← x`.
///
/// General entry point behind the contiguous [`copy`](super::copy): honours
/// arbitrary positive, negative and zero increments.
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
pub fn copy_strided<T: Copy>(x: StridedSlice<'_, T>, mut y: StridedSliceMut<'_, T>) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 {
        return;
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous(), y.as_contiguous_mut()) {
        super::copy(xc, yc);
        return;
    }
    for i in 0..n {
        y.set(i, x.get(i));
    }
}

/// Strided SWAP `x ↔ y`.
///
/// General entry point behind the contiguous [`swap`](super::swap): honours
/// arbitrary positive, negative and zero increments. `x` and `y` must not
/// alias (a standing reference BLAS precondition; the safe view constructors
/// enforce it statically).
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
pub fn swap_strided<T: Copy>(mut x: StridedSliceMut<'_, T>, mut y: StridedSliceMut<'_, T>) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 {
        return;
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous_mut(), y.as_contiguous_mut()) {
        super::swap(xc, yc);
        return;
    }
    for i in 0..n {
        let xi = x.get(i);
        let yi = y.get(i);
        x.set(i, yi);
        y.set(i, xi);
    }
}

/// Strided plane rotation: applies `[c s; -s c]` to each pair `(x[i], y[i])`.
///
/// General entry point behind the contiguous [`rot`](super::rot): honours
/// arbitrary positive, negative and zero increments.
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
pub fn rot_strided<T: Float>(
    c: T,
    s: T,
    mut x: StridedSliceMut<'_, T>,
    mut y: StridedSliceMut<'_, T>,
) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 {
        return;
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous_mut(), y.as_contiguous_mut()) {
        super::rot(c, s, xc, yc);
        return;
    }
    for i in 0..n {
        let xi = x.get(i);
        let yi = y.get(i);
        x.set(i, c * xi + s * yi);
        y.set(i, c * yi - s * xi);
    }
}

/// Strided modified Givens rotation (`SROTM`/`DROTM` with increments).
///
/// Mirrors Netlib `drotm.f` (3.12) exactly:
///
/// * `n == 0` or `params.flag == -2` (the reference's `DFLAG + 2 == 0`):
///   no-op.
/// * `flag < 0`: `H = [[h11, h12], [h21, h22]]`.
/// * `flag == 0`: `H = [[1, h12], [h21, 1]]` (unit diagonal implied).
/// * otherwise (`flag > 0`, and also NaN): `H = [[h11, 1], [-1, h22]]`.
///
/// With `w = x[i]`, `z = y[i]` each pair becomes
/// `(w·h11 + z·h12, w·h21 + z·h22)`, the implied entries substituted as
/// above. Arbitrary positive, negative and zero increments are honoured (see
/// the module docs for the raw-buffer contract): a negative increment walks
/// its vector backwards, a zero increment re-reads and re-writes element 0
/// on every iteration, exactly like the reference loop. A unit-stride call
/// is forwarded to the contiguous [`rotm`](super::rotm), which uses the same
/// classification.
///
/// The reference `DPARAM` array is ordered `[flag, h11, h21, h12, h22]`
/// (column-major `H`), i.e.
/// `RotmParams { flag: p[0], h11: p[1], h21: p[2], h12: p[3], h22: p[4] }`.
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
pub fn rotm_strided<T: Float>(
    params: &RotmParams<T>,
    mut x: StridedSliceMut<'_, T>,
    mut y: StridedSliceMut<'_, T>,
) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    let zero = T::zero();
    let two = T::one() + T::one();
    if n == 0 || params.flag + two == zero {
        return;
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous_mut(), y.as_contiguous_mut()) {
        super::rotm(params, xc, yc);
        return;
    }
    let RotmParams {
        flag,
        h11,
        h12,
        h21,
        h22,
    } = *params;
    if flag < zero {
        for i in 0..n {
            let (w, z) = (x.get(i), y.get(i));
            x.set(i, w * h11 + z * h12);
            y.set(i, w * h21 + z * h22);
        }
    } else if flag == zero {
        for i in 0..n {
            let (w, z) = (x.get(i), y.get(i));
            x.set(i, w + z * h12);
            y.set(i, w * h21 + z);
        }
    } else {
        for i in 0..n {
            let (w, z) = (x.get(i), y.get(i));
            x.set(i, w * h11 + z);
            y.set(i, -w + h22 * z);
        }
    }
}

/// Strided complex plane rotation with a **real** cosine and sine
/// (`CSROT`/`ZDROT` with increments).
///
/// For each logical pair `x' = c·x + s·y`, `y' = c·y - s·x`, where `c` and
/// `s` scale the real and imaginary parts separately, exactly like Netlib
/// `zdrot.f`/`csrot.f` (3.12) -- see [`zdrot`](super::zdrot). Arbitrary
/// positive, negative and zero increments are honoured (see the module docs
/// for the raw-buffer contract): a negative increment walks its vector
/// backwards, a zero increment re-reads and re-writes element 0 on every
/// iteration, exactly like the reference loop. There is no quick return for
/// `c == 1, s == 0`; like the reference, every pair is recomputed. A
/// unit-stride call is forwarded to the contiguous [`zdrot`](super::zdrot).
///
/// # Panics
///
/// Panics if `x` and `y` have different logical lengths.
pub fn zdrot_strided<T: ComplexScalar>(
    c: T::Real,
    s: T::Real,
    mut x: StridedSliceMut<'_, T>,
    mut y: StridedSliceMut<'_, T>,
) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    let n = x.len();
    if n == 0 {
        return;
    }
    if let (Some(xc), Some(yc)) = (x.as_contiguous_mut(), y.as_contiguous_mut()) {
        super::zdrot(c, s, xc, yc);
        return;
    }
    for i in 0..n {
        let (xi, yi) = super::asum::zdrot_pair(c, s, x.get(i), y.get(i));
        x.set(i, xi);
        y.set(i, yi);
    }
}

// =============================================================================
// Reduction family. SCAL/ASUM/IAMAX/IAMIN: a non-positive increment is a
// no-op / zero result. NRM2 (Netlib >= 3.10): every increment is walked like
// the product family. Both match Netlib 3.12 (see module docs).
// =============================================================================

/// Strided SCAL `x ← α·x`.
///
/// General entry point behind the contiguous [`scal`](super::scal). Per
/// reference BLAS a non-positive increment (`stride <= 0`) is a **no-op**.
pub fn scal_strided<T: Field>(alpha: T, mut x: StridedSliceMut<'_, T>) {
    let n = x.len();
    if n == 0 || x.stride() <= 0 {
        return;
    }
    if let Some(xc) = x.as_contiguous_mut() {
        super::scal(alpha, xc);
        return;
    }
    for i in 0..n {
        let scaled = alpha * x.get(i);
        x.set(i, scaled);
    }
}

/// Strided NRM2 (Euclidean/L2 norm).
///
/// General entry point behind the contiguous [`nrm2`](super::nrm2). Follows
/// Netlib >= 3.10 (`dnrm2.f90`, unchanged in 3.12): every increment is
/// accepted — a negative stride walks the vector backwards from offset
/// `(n - 1)·|stride|`, and a zero stride reads `x[0]` on every iteration,
/// giving `sqrt(n)·|x[0]|`. (Pre-3.10 reference BLAS returned `0` for
/// `incx <= 0`.) Uses the same overflow-safe scaled-sum-of-squares recurrence
/// as the contiguous kernel, never a naive `Σx²`.
#[must_use]
pub fn nrm2_strided<T: Real>(x: StridedSlice<'_, T>) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }
    if let Some(xc) = x.as_contiguous() {
        return super::nrm2(xc);
    }
    if n == 1 {
        return Scalar::abs(x.get(0));
    }
    let mut scale = T::zero();
    let mut ssq = T::one();
    for i in 0..n {
        let abs_xi = Scalar::abs(x.get(i));
        // `!= zero` (not `> zero`): a NaN `abs_xi` must still enter this
        // branch and poison the accumulator, matching `nrm2_fold` in
        // `level1/nrm2.rs` (`>` is always false for NaN, which would
        // otherwise silently drop a NaN element from the norm).
        if abs_xi != T::zero() {
            if scale < abs_xi {
                let t = scale / abs_xi;
                ssq = T::one() + ssq * t * t;
                scale = abs_xi;
            } else {
                let t = if scale == abs_xi {
                    T::one()
                } else {
                    abs_xi / scale
                };
                ssq += t * t;
            }
        }
    }
    scale * Real::sqrt(ssq)
}

/// Strided complex NRM2 (`SCNRM2`/`DZNRM2` with an increment).
///
/// Same increment contract as [`nrm2_strided`] (Netlib >= 3.10
/// `dznrm2.f90`, unchanged in 3.12): a negative stride walks the vector
/// backwards from offset `(n - 1)·|stride|`, a zero stride reads `x[0]` on
/// every iteration (`sqrt(n)·|x[0]|`). The real and imaginary parts are
/// folded into the overflow-safe scaled sum of squares shared with
/// [`cnrm2`](super::cnrm2), to which a unit-stride call is forwarded.
#[must_use]
pub fn cnrm2_strided<T: ComplexScalar>(x: StridedSlice<'_, T>) -> T::Real {
    let n = x.len();
    if n == 0 {
        return T::Real::zero();
    }
    if let Some(xc) = x.as_contiguous() {
        return super::cnrm2(xc);
    }
    let mut state = (T::Real::zero(), T::Real::one());
    for i in 0..n {
        let z = x.get(i);
        state = nrm2_fold(nrm2_fold(state, z.real()), z.imag());
    }
    nrm2_finalize(state)
}

/// Strided ASUM (sum of absolute values, L1 norm).
///
/// General entry point behind the contiguous [`asum`](super::asum). Per
/// reference BLAS a non-positive increment returns `0`.
#[must_use]
pub fn asum_strided<T: Real>(x: StridedSlice<'_, T>) -> T {
    let n = x.len();
    if n == 0 || x.stride() <= 0 {
        return T::zero();
    }
    if let Some(xc) = x.as_contiguous() {
        return super::asum(xc);
    }
    let mut acc = T::zero();
    for i in 0..n {
        acc += Scalar::abs(x.get(i));
    }
    acc
}

/// Strided IAMAX (index of first element with maximum absolute value).
///
/// General entry point behind the contiguous [`iamax`](super::iamax). Per
/// reference BLAS a non-positive increment returns `0`.
///
/// For complex scalar types elements are ranked by
/// [`cabs1`](super::cabs1)`(z) = |Re(z)| + |Im(z)|` on **every** stride,
/// exactly like Netlib `ICAMAX`/`IZAMAX` and the contiguous [`iamax`](super::iamax)
/// (not by the modulus `|z|`); ties keep the first occurrence and a NaN never
/// displaces the running maximum. For real types this is plain `|x[i]|`.
#[must_use]
pub fn iamax_strided<T: Scalar>(x: StridedSlice<'_, T>) -> usize {
    let n = x.len();
    if n == 0 || x.stride() <= 0 {
        return 0;
    }
    if let Some(xc) = x.as_contiguous() {
        return super::iamax(xc);
    }
    let mut max_idx = 0;
    let mut max_val = super::cabs1(x.get(0));
    for i in 1..n {
        let abs_xi = super::cabs1(x.get(i));
        if abs_xi > max_val {
            max_val = abs_xi;
            max_idx = i;
        }
    }
    max_idx
}

/// Strided IAMIN (index of first element with minimum absolute value).
///
/// General entry point behind the contiguous [`iamin`](super::iamin). Per
/// reference BLAS a non-positive increment returns `0`.
///
/// Complex elements are ranked by [`cabs1`](super::cabs1) on every stride,
/// consistent with [`iamax_strided`] and the contiguous [`iamin`](super::iamin).
#[must_use]
pub fn iamin_strided<T: Scalar>(x: StridedSlice<'_, T>) -> usize {
    let n = x.len();
    if n == 0 || x.stride() <= 0 {
        return 0;
    }
    if let Some(xc) = x.as_contiguous() {
        return super::iamin(xc);
    }
    let mut min_idx = 0;
    let mut min_val = super::cabs1(x.get(0));
    for i in 1..n {
        let abs_xi = super::cabs1(x.get(i));
        if abs_xi < min_val {
            min_val = abs_xi;
            min_idx = i;
        }
    }
    min_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // Naive strided reference helpers written directly here (NOT by calling
    // the routines under test), per the regression-test requirement.
    // ---------------------------------------------------------------------

    /// Physical offset of logical element `i` under reference BLAS rules.
    fn ref_offset(i: usize, n: usize, inc: isize) -> usize {
        if inc >= 0 {
            i * inc.unsigned_abs()
        } else {
            (n - 1 - i) * inc.unsigned_abs()
        }
    }

    fn ref_dot(x: &[f64], incx: isize, y: &[f64], incy: isize, n: usize) -> f64 {
        let mut acc = 0.0;
        for i in 0..n {
            acc += x[ref_offset(i, n, incx)] * y[ref_offset(i, n, incy)];
        }
        acc
    }

    fn ref_asum(x: &[f64], incx: isize, n: usize) -> f64 {
        let mut acc = 0.0;
        for i in 0..n {
            acc += x[ref_offset(i, n, incx)].abs();
        }
        acc
    }

    fn ref_nrm2(x: &[f64], incx: isize, n: usize) -> f64 {
        let mut acc = 0.0;
        for i in 0..n {
            let v = x[ref_offset(i, n, incx)];
            acc += v * v;
        }
        acc.sqrt()
    }

    fn slice_view(data: &[f64], n: usize, inc: isize) -> StridedSlice<'_, f64> {
        StridedSlice::try_new(data, n, inc).expect("view fits")
    }

    fn slice_view_mut(data: &mut [f64], n: usize, inc: isize) -> StridedSliceMut<'_, f64> {
        StridedSliceMut::try_new(data, n, inc).expect("view fits")
    }

    // ---------------------------------------------------------------------
    // Extent / constructor validation
    // ---------------------------------------------------------------------

    #[test]
    fn test_strided_extent() {
        assert_eq!(strided_extent(0, 5), Some(0));
        assert_eq!(strided_extent(1, 5), Some(1));
        assert_eq!(strided_extent(4, 1), Some(4));
        assert_eq!(strided_extent(4, 2), Some(7)); // 3*2+1
        assert_eq!(strided_extent(4, -3), Some(10)); // 3*3+1
        assert_eq!(strided_extent(3, 0), Some(1));
    }

    #[test]
    fn test_try_new_bounds() {
        let data = [1.0, 2.0, 3.0, 4.0];
        // 3 elements stride 2 needs extent 5 > 4 -> None
        assert!(StridedSlice::try_new(&data, 3, 2).is_none());
        // 2 elements stride 2 needs extent 3 <= 4 -> Some
        assert!(StridedSlice::try_new(&data, 2, 2).is_some());
    }

    #[test]
    fn test_phys_index_negative() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        // n=3, stride=-2: logical 0 -> offset 4, 1 -> 2, 2 -> 0
        let v = slice_view(&data, 3, -2);
        assert_eq!(v.get(0), 50.0);
        assert_eq!(v.get(1), 30.0);
        assert_eq!(v.get(2), 10.0);
    }

    #[test]
    fn test_zero_stride_reads_first() {
        let data = [7.0, 8.0, 9.0];
        let v = slice_view(&data, 4, 0);
        for i in 0..4 {
            assert_eq!(v.get(i), 7.0);
        }
    }

    // ---------------------------------------------------------------------
    // DOT
    // ---------------------------------------------------------------------

    #[test]
    fn test_dot_unit_matches_contiguous() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        // Unit stride must be exactly the legacy contiguous result.
        let expected = super::super::dot(&x, &y);
        let got = dot_strided(StridedSlice::from_slice(&x), StridedSlice::from_slice(&y));
        assert_eq!(got, expected);
    }

    #[test]
    fn test_dot_stride2() {
        // x buffer holds 4 logical elements at stride 2.
        let xbuf = [1.0, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0];
        let ybuf = [10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0];
        let n = 4;
        let got = dot_strided(slice_view(&xbuf, n, 2), slice_view(&ybuf, n, 2));
        let expected = ref_dot(&xbuf, 2, &ybuf, 2, n);
        assert!((got - expected).abs() < 1e-12);
        assert!((expected - (10.0 + 40.0 + 90.0 + 160.0)).abs() < 1e-12);
    }

    #[test]
    fn test_dot_stride3() {
        let xbuf: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ybuf: Vec<f64> = (0..10).map(|i| (i * i) as f64).collect();
        let n = 4; // stride 3 -> offsets 0,3,6,9
        let got = dot_strided(slice_view(&xbuf, n, 3), slice_view(&ybuf, n, 3));
        let expected = ref_dot(&xbuf, 3, &ybuf, 3, n);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn test_dot_negative_stride() {
        let xbuf = [1.0, 2.0, 3.0, 4.0];
        let ybuf = [10.0, 20.0, 30.0, 40.0];
        let n = 4;
        // Negative stride on both: pairing is order-preserving so the value
        // equals the unit-stride dot of the reversed vectors == plain dot.
        let got = dot_strided(slice_view(&xbuf, n, -1), slice_view(&ybuf, n, -1));
        let expected = ref_dot(&xbuf, -1, &ybuf, -1, n);
        assert!((got - expected).abs() < 1e-12);
        assert!((got - super::super::dot(&xbuf, &ybuf)).abs() < 1e-12);
    }

    #[test]
    fn test_dot_mixed_strides() {
        // x: 3 elements stride 2 (offsets 0,2,4); y: 3 elements stride -1
        // (offsets 2,1,0). Different strides and different signs.
        let xbuf = [1.0, 0.0, 2.0, 0.0, 3.0];
        let ybuf = [100.0, 10.0, 1.0];
        let n = 3;
        let got = dot_strided(slice_view(&xbuf, n, 2), slice_view(&ybuf, n, -1));
        let expected = ref_dot(&xbuf, 2, &ybuf, -1, n);
        // logical: (1*1) + (2*10) + (3*100) = 1 + 20 + 300 = 321
        assert!((got - expected).abs() < 1e-12);
        assert!((got - 321.0).abs() < 1e-12);
    }

    #[test]
    fn test_dot_zero_stride_x() {
        // incx == 0 reads x[0] n times: sum = x[0]*(y0+y1+y2)
        let xbuf = [5.0];
        let ybuf = [1.0, 2.0, 3.0];
        let n = 3;
        let got = dot_strided(slice_view(&xbuf, n, 0), slice_view(&ybuf, n, 1));
        assert!((got - (5.0 * 6.0)).abs() < 1e-12);
    }

    // ---------------------------------------------------------------------
    // AXPY
    // ---------------------------------------------------------------------

    #[test]
    fn test_axpy_unit_matches_contiguous() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y_legacy = [0.5, 1.0, 1.5, 2.0];
        let mut y_strided = y_legacy;
        super::super::axpy(2.0, &x, &mut y_legacy);
        axpy_strided(
            2.0,
            StridedSlice::from_slice(&x),
            StridedSliceMut::from_slice(&mut y_strided),
        );
        assert_eq!(y_strided, y_legacy);
    }

    #[test]
    fn test_axpy_stride2_different_strides() {
        // x stride 3, y stride 2 (different strides).
        let xbuf = [1.0, 9.0, 9.0, 2.0, 9.0, 9.0, 3.0];
        let mut ybuf = [10.0, 0.0, 20.0, 0.0, 30.0];
        let n = 3;
        let mut expected = ybuf;
        // naive reference
        for i in 0..n {
            expected[ref_offset(i, n, 2)] += 2.0 * xbuf[ref_offset(i, n, 3)];
        }
        axpy_strided(
            2.0,
            slice_view(&xbuf, n, 3),
            slice_view_mut(&mut ybuf, n, 2),
        );
        for k in 0..ybuf.len() {
            assert!((ybuf[k] - expected[k]).abs() < 1e-12, "index {k}");
        }
    }

    #[test]
    fn test_axpy_negative_stride() {
        let xbuf = [1.0, 2.0, 3.0];
        let mut ybuf = [10.0, 20.0, 30.0];
        let n = 3;
        let mut expected = ybuf;
        for i in 0..n {
            expected[ref_offset(i, n, -1)] += 3.0 * xbuf[ref_offset(i, n, -1)];
        }
        axpy_strided(
            3.0,
            slice_view(&xbuf, n, -1),
            slice_view_mut(&mut ybuf, n, -1),
        );
        for k in 0..3 {
            assert!((ybuf[k] - expected[k]).abs() < 1e-12);
        }
    }

    // ---------------------------------------------------------------------
    // COPY / SWAP
    // ---------------------------------------------------------------------

    #[test]
    fn test_copy_negative_stride() {
        // Copy reversed: x logical order into y with incy = -1 reverses.
        let xbuf = [1.0, 2.0, 3.0];
        let mut ybuf = [0.0, 0.0, 0.0];
        let n = 3;
        copy_strided(slice_view(&xbuf, n, 1), slice_view_mut(&mut ybuf, n, -1));
        // logical i of y at offset (n-1-i); y[offset(i)] = x[i]
        assert_eq!(ybuf, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_swap_stride2() {
        let mut xbuf = [1.0, 0.0, 2.0, 0.0, 3.0];
        let mut ybuf = [10.0, 0.0, 20.0, 0.0, 30.0];
        let n = 3;
        swap_strided(
            slice_view_mut(&mut xbuf, n, 2),
            slice_view_mut(&mut ybuf, n, 2),
        );
        assert_eq!(xbuf, [10.0, 0.0, 20.0, 0.0, 30.0]);
        assert_eq!(ybuf, [1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    // ---------------------------------------------------------------------
    // SCAL (non-positive stride == no-op)
    // ---------------------------------------------------------------------

    #[test]
    fn test_scal_stride2() {
        let mut xbuf = [1.0, -7.0, 2.0, -7.0, 3.0];
        let n = 3;
        scal_strided(10.0, slice_view_mut(&mut xbuf, n, 2));
        assert_eq!(xbuf, [10.0, -7.0, 20.0, -7.0, 30.0]);
    }

    #[test]
    fn test_scal_nonpositive_is_noop() {
        let mut xbuf = [1.0, 2.0, 3.0];
        let before = xbuf;
        scal_strided(10.0, slice_view_mut(&mut xbuf, 3, -1));
        assert_eq!(xbuf, before, "stride < 0 must be a no-op");
        scal_strided(10.0, slice_view_mut(&mut xbuf, 3, 0));
        assert_eq!(xbuf, before, "stride == 0 must be a no-op");
    }

    // ---------------------------------------------------------------------
    // NRM2 / ASUM / IAMAX (non-positive stride == 0 result)
    // ---------------------------------------------------------------------

    #[test]
    fn test_nrm2_stride2() {
        let xbuf = [3.0, 0.0, 4.0];
        let n = 2;
        let got = nrm2_strided(slice_view(&xbuf, n, 2));
        let expected = ref_nrm2(&xbuf, 2, n);
        assert!((got - expected).abs() < 1e-12);
        assert!((got - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_issue_10_nrm2_strided_negative_stride_walks_backwards() {
        // Netlib >= 3.10 (`dnrm2.f90`) accepts incx < 0 and walks the vector
        // backwards from offset (n-1)*|incx|; the pre-fix code returned 0.
        let xbuf = [3.0, 4.0];
        assert_eq!(nrm2_strided(slice_view(&xbuf, 2, -1)), 5.0);
        // Reporter's case: dnrm2(1, [0.326], -1) == 0.326.
        assert_eq!(nrm2_strided(slice_view(&[0.326], 1, -1)), 0.326);
        // Same set of elements as the forward walk.
        let ybuf = [1.0, 9.0, -2.0, 9.0, 3.0, 9.0, -4.0];
        let backward = nrm2_strided(slice_view(&ybuf, 4, -2));
        let forward = nrm2_strided(slice_view(&ybuf, 4, 2));
        assert!((backward - forward).abs() <= 4.0 * f64::EPSILON * forward);
        assert!((backward - ref_nrm2(&ybuf, -2, 4)).abs() <= 4.0 * f64::EPSILON * forward);
    }

    #[test]
    fn test_issue_10_nrm2_strided_zero_stride_is_sqrt_n_abs_x0() {
        // incx == 0 reads x[0] on every iteration: sqrt(n) * |x[0]|.
        let xbuf = [-3.0, 100.0];
        assert_eq!(nrm2_strided(slice_view(&xbuf, 4, 0)), 6.0);
        assert_eq!(nrm2_strided(slice_view(&xbuf, 1, 0)), 3.0);
        // Scaled correctly: no overflow even though x[0]^2 overflows f64.
        let big = [1e300];
        let got = nrm2_strided(slice_view(&big, 9, 0));
        assert!(
            (got - 3e300).abs() <= 4.0 * f64::EPSILON * 3e300,
            "got {got}"
        );
    }

    #[test]
    fn test_issue_10_nrm2_strided_negative_stride_large_no_overflow() {
        let xbuf = [1e300, 0.0, -1e300, 0.0, 1e300];
        let got = nrm2_strided(slice_view(&xbuf, 3, -2));
        let expected = 3.0f64.sqrt() * 1e300;
        assert!(got.is_finite(), "norm overflowed to {got}");
        assert!((got - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }

    #[test]
    fn test_asum_stride3() {
        let xbuf = [1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 3.0];
        let n = 3;
        let got = asum_strided(slice_view(&xbuf, n, 3));
        let expected = ref_asum(&xbuf, 3, n);
        assert!((got - expected).abs() < 1e-12);
        assert!((got - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_asum_nonpositive_is_zero() {
        let xbuf = [1.0, 2.0, 3.0];
        assert_eq!(asum_strided(slice_view(&xbuf, 3, 0)), 0.0);
        assert_eq!(asum_strided(slice_view(&xbuf, 3, -1)), 0.0);
    }

    #[test]
    fn test_iamax_stride2() {
        // logical elements at offsets 0,2,4 = 1, 9, 3 -> max at logical 1
        let xbuf = [1.0, 100.0, 9.0, 100.0, 3.0];
        assert_eq!(iamax_strided(slice_view(&xbuf, 3, 2)), 1);
    }

    #[test]
    fn test_iamax_nonpositive_is_zero() {
        let xbuf = [1.0, 9.0, 3.0];
        assert_eq!(iamax_strided(slice_view(&xbuf, 3, -1)), 0);
        assert_eq!(iamax_strided(slice_view(&xbuf, 3, 0)), 0);
    }

    #[test]
    fn test_iamin_stride2() {
        let xbuf = [5.0, 100.0, 1.0, 100.0, 3.0];
        assert_eq!(iamin_strided(slice_view(&xbuf, 3, 2)), 1);
    }

    // ---------------------------------------------------------------------
    // Issue #11: complex IAMAX / IAMIN rank by cabs1 for every stride
    // ---------------------------------------------------------------------

    #[test]
    fn test_issue_11_iamax_strided_complex_uses_cabs1() {
        use num_complex::Complex64;
        // Reporter's case: x = [4+0i, 0, 3+2i], n = 2, inc = 2 selects 4
        // (cabs1 4) and 3+2i (cabs1 5). Netlib izamax returns 2 (1-based),
        // i.e. 1 here; ranking by modulus (4 > 3.606) picked 0.
        let x = [
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 2.0),
        ];
        let view = StridedSlice::try_new(&x, 2, 2).expect("view fits");
        assert_eq!(iamax_strided(view), 1);
        // Ties keep the first occurrence: cabs1(2+2i) == cabs1(4) == 4.
        let tie = [
            Complex64::new(2.0, 2.0),
            Complex64::new(9.0, 9.0),
            Complex64::new(4.0, 0.0),
        ];
        let view = StridedSlice::try_new(&tie, 2, 2).expect("view fits");
        assert_eq!(iamax_strided(view), 0);
    }

    #[test]
    fn test_issue_11_iamin_strided_complex_uses_cabs1() {
        use num_complex::Complex32;
        // Elements 3+2i (cabs1 5, modulus 3.606) and 4 (cabs1 4, modulus 4):
        // cabs1 ranks 4 as the minimum (index 1); the modulus picked 0.
        let x = [
            Complex32::new(3.0, 2.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        let view = StridedSlice::try_new(&x, 2, 3).expect("view fits");
        assert_eq!(iamin_strided(view), 1);
    }

    #[test]
    fn test_issue_11_iamax_iamin_strided_random_c64_match_gathered_contiguous() {
        use num_complex::Complex64;
        // xorshift64* -- deterministic, no external RNG dependency.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next_unit = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
            (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        for trial in 0..4000usize {
            let n = 1 + trial % 9;
            let stride = 2 + (trial / 9) % 4;
            let len = (n - 1) * stride + 1;
            let buf: Vec<Complex64> = (0..len)
                .map(|_| Complex64::new(next_unit(), next_unit()))
                .collect();
            let gathered: Vec<Complex64> = (0..n).map(|i| buf[i * stride]).collect();
            let view = StridedSlice::try_new(&buf, n, stride as isize).expect("view fits");
            assert_eq!(
                iamax_strided(view),
                super::super::iamax(&gathered),
                "iamax trial {trial}: n={n}, stride={stride}"
            );
            assert_eq!(
                iamin_strided(view),
                super::super::iamin(&gathered),
                "iamin trial {trial}: n={n}, stride={stride}"
            );
        }
    }

    // ---------------------------------------------------------------------
    // ROT
    // ---------------------------------------------------------------------

    #[test]
    fn test_rot_stride2() {
        let c = 0.6;
        let s = 0.8;
        let mut xbuf = [1.0, 0.0, 2.0];
        let mut ybuf = [3.0, 0.0, 4.0];
        let n = 2; // offsets 0,2
        let mut ex = xbuf;
        let mut ey = ybuf;
        for i in 0..n {
            let ox = ref_offset(i, n, 2);
            let xi = ex[ox];
            let yi = ey[ox];
            ex[ox] = c * xi + s * yi;
            ey[ox] = c * yi - s * xi;
        }
        rot_strided(
            c,
            s,
            slice_view_mut(&mut xbuf, n, 2),
            slice_view_mut(&mut ybuf, n, 2),
        );
        for k in 0..3 {
            assert!((xbuf[k] - ex[k]).abs() < 1e-12);
            assert!((ybuf[k] - ey[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rot_unit_matches_contiguous() {
        let c = 0.6;
        let s = 0.8;
        let mut x_legacy = [1.0, 2.0, 3.0];
        let mut y_legacy = [4.0, 5.0, 6.0];
        let mut x_strided = x_legacy;
        let mut y_strided = y_legacy;
        super::super::rot(c, s, &mut x_legacy, &mut y_legacy);
        rot_strided(
            c,
            s,
            StridedSliceMut::from_slice(&mut x_strided),
            StridedSliceMut::from_slice(&mut y_strided),
        );
        assert_eq!(x_strided, x_legacy);
        assert_eq!(y_strided, y_legacy);
    }

    // ---------------------------------------------------------------------
    // Raw-pointer constructor (the CBLAS integration path)
    // ---------------------------------------------------------------------

    #[test]
    fn test_from_raw_parts_negative_stride() {
        let xbuf = [1.0, 2.0, 3.0, 4.0];
        let ybuf = [10.0, 20.0, 30.0, 40.0];
        let n = 4;
        // SAFETY: pointers are valid for `n` elements at unit stride; the base
        // pointer is the lowest touched address for incx = -1.
        let got = unsafe {
            dot_strided(
                StridedSlice::from_raw_parts(xbuf.as_ptr(), n, -1),
                StridedSlice::from_raw_parts(ybuf.as_ptr(), n, -1),
            )
        };
        let expected = ref_dot(&xbuf, -1, &ybuf, -1, n);
        assert!((got - expected).abs() < 1e-12);
    }

    /// An extent that overflows `usize` cannot describe a real allocation, so
    /// the raw-parts constructors must return an empty view -- never panic (the
    /// `extern "C"` CBLAS entry points reach them) and never touch `ptr`.
    #[test]
    fn test_strided_from_raw_parts_overflowing_extent_is_empty() {
        // Non-null and aligned but not backed by any allocation: any access
        // through it would fault, so the test also proves `ptr` is left alone.
        let ptr = core::ptr::NonNull::<f64>::dangling().as_ptr();
        // `(usize::MAX - 1) * |stride| + 1` overflows `usize` for every stride
        // here; `isize::MIN` also covers the `unsigned_abs` edge.
        for stride in [2_isize, -2, isize::MIN] {
            assert_eq!(strided_extent(usize::MAX, stride), None, "stride {stride}");

            // SAFETY: an overflowing extent yields an empty view without
            // reading `ptr`, so the dangling pointer is never dereferenced.
            let view =
                unsafe { StridedSlice::from_raw_parts(ptr.cast_const(), usize::MAX, stride) };
            assert_eq!(view.len(), 0, "stride {stride}");
            assert!(view.is_empty(), "stride {stride}");
            assert_eq!(nrm2_strided(view), 0.0, "stride {stride}");

            // SAFETY: as above; the empty mutable view never reads or writes
            // `ptr`, so uniqueness and validity are vacuous.
            let view_mut = unsafe { StridedSliceMut::from_raw_parts(ptr, usize::MAX, stride) };
            assert_eq!(view_mut.len(), 0, "stride {stride}");
            assert!(view_mut.is_empty(), "stride {stride}");
            // `nrm2_strided` only takes an immutable view, so check instead that
            // a mutating routine treats the empty mutable view as a no-op.
            scal_strided(2.0, view_mut);
        }
    }

    /// The raw-parts constructors also reject an extent whose elements would span
    /// more than `isize::MAX` bytes (`slice::from_raw_parts` forbids it and no
    /// allocation is that large), even when the element count fits in `usize`.
    #[test]
    fn test_strided_raw_parts_extent_respects_isize_max_bytes() {
        let isize_max = isize::MAX.unsigned_abs();
        let max_f64 = isize_max / core::mem::size_of::<f64>();
        // Exact boundary at unit stride: the largest legal extent passes, one more
        // element fails.
        assert!(fits_isize_bytes::<f64>(max_f64));
        assert!(!fits_isize_bytes::<f64>(max_f64 + 1));
        assert_eq!(raw_parts_extent::<f64>(max_f64, 1), Some(max_f64));
        assert_eq!(raw_parts_extent::<f64>(max_f64 + 1, 1), None);
        // The same boundary through a stride: extent = (2 - 1)·|stride| + 1.
        let stride = isize::try_from(max_f64 - 1).expect("below isize::MAX");
        assert_eq!(raw_parts_extent::<f64>(2, stride), Some(max_f64));
        assert_eq!(raw_parts_extent::<f64>(2, -stride), Some(max_f64));
        assert_eq!(raw_parts_extent::<f64>(2, stride + 1), None);
        assert_eq!(raw_parts_extent::<f64>(2, -stride - 1), None);
        // The bound scales with the element size, and a zero-sized type is held to
        // `isize::MAX` elements.
        assert!(fits_isize_bytes::<u8>(max_f64 + 1));
        assert!(!fits_isize_bytes::<num_complex::Complex64>(max_f64));
        assert!(fits_isize_bytes::<()>(isize_max));
        assert!(!fits_isize_bytes::<()>(isize_max + 1));
        // The element-count helper behind `try_new` is unchanged.
        assert_eq!(strided_extent(max_f64 + 1, 1), Some(max_f64 + 1));

        // On the rejected side the constructors return an empty view and never
        // touch the (dangling) pointer. No view is built over an accepted extent:
        // one over unallocated memory would itself be undefined behaviour.
        let ptr = core::ptr::NonNull::<f64>::dangling().as_ptr();
        for (len, stride) in [(max_f64 + 1, 1), (2, stride + 1), (2, -stride - 1)] {
            // SAFETY: an extent beyond `isize::MAX` bytes yields an empty view
            // without reading `ptr`, so the dangling pointer is never dereferenced.
            let view = unsafe { StridedSlice::from_raw_parts(ptr.cast_const(), len, stride) };
            assert!(view.is_empty(), "len {len} stride {stride}");
            assert_eq!(nrm2_strided(view), 0.0, "len {len} stride {stride}");
            // SAFETY: as above; the empty mutable view never reads or writes
            // `ptr`, so uniqueness and validity are vacuous.
            let view_mut = unsafe { StridedSliceMut::from_raw_parts(ptr, len, stride) };
            assert!(view_mut.is_empty(), "len {len} stride {stride}");
            scal_strided(2.0, view_mut);
        }
    }

    // ---------------------------------------------------------------------
    // Issue #15: strided ROTM / complex ROT / complex NRM2
    // ---------------------------------------------------------------------

    /// Increment pairs exercising positive, negative, zero and mixed strides.
    const INC_PAIRS: [(isize, isize); 9] = [
        (1, 1),
        (2, 1),
        (1, 3),
        (-1, 2),
        (-2, -3),
        (0, 1),
        (1, 0),
        (0, 0),
        (-1, 0),
    ];

    /// Buffer length a reference-BLAS vector of `n` elements at `inc` needs.
    fn span(n: usize, inc: isize) -> usize {
        if n == 0 {
            0
        } else {
            (n - 1) * inc.unsigned_abs() + 1
        }
    }

    /// Naive reference `DROTM` loop written directly from Netlib `drotm.f`
    /// (start offsets via `ref_offset`, flag classification as in the
    /// reference: `flag + 2 == 0` identity, `< 0` full, `== 0`, else).
    fn ref_rotm(
        p: &RotmParams<f64>,
        x: &mut [f64],
        incx: isize,
        y: &mut [f64],
        incy: isize,
        n: usize,
    ) {
        if n == 0 || p.flag + 2.0 == 0.0 {
            return;
        }
        for i in 0..n {
            let (ox, oy) = (ref_offset(i, n, incx), ref_offset(i, n, incy));
            let (w, z) = (x[ox], y[oy]);
            let (nx, ny) = if p.flag < 0.0 {
                (w * p.h11 + z * p.h12, w * p.h21 + z * p.h22)
            } else if p.flag == 0.0 {
                (w + z * p.h12, w * p.h21 + z)
            } else {
                (w * p.h11 + z, -w + p.h22 * z)
            };
            x[ox] = nx;
            y[oy] = ny;
        }
    }

    #[test]
    fn test_issue_15_rotm_strided_matches_reference_loop() {
        let flags = [-1.0, 0.0, 1.0, -0.5, 0.5, f64::NAN, -2.0];
        for &flag in &flags {
            let p = RotmParams {
                flag,
                h11: 0.75,
                h12: -1.25,
                h21: 2.5,
                h22: -0.5,
            };
            for &(incx, incy) in &INC_PAIRS {
                for n in [0usize, 1, 2, 5] {
                    let xlen = span(n, incx).max(1);
                    let ylen = span(n, incy).max(1);
                    let x0: Vec<f64> = (0..xlen).map(|k| 1.0 + k as f64 * 0.5).collect();
                    let y0: Vec<f64> = (0..ylen).map(|k| -2.0 + k as f64 * 0.25).collect();
                    let (mut xr, mut yr) = (x0.clone(), y0.clone());
                    ref_rotm(&p, &mut xr, incx, &mut yr, incy, n);
                    let (mut xs, mut ys) = (x0, y0);
                    rotm_strided(
                        &p,
                        slice_view_mut(&mut xs, n, incx),
                        slice_view_mut(&mut ys, n, incy),
                    );
                    assert_eq!(xs, xr, "x: flag={flag}, incx={incx}, incy={incy}, n={n}");
                    assert_eq!(ys, yr, "y: flag={flag}, incx={incx}, incy={incy}, n={n}");
                }
            }
        }
    }

    #[test]
    fn test_issue_15_rotm_strided_flag_forms_hand_checked() {
        // flag = 0.5 (non-canonical, > 0): H = [[h11, 1], [-1, h22]].
        let p = RotmParams {
            flag: 0.5,
            h11: 2.0,
            h12: 100.0,
            h21: 100.0,
            h22: 3.0,
        };
        let mut x = [1.0, 9.0, 1.0];
        let mut y = [1.0, 9.0, 1.0];
        rotm_strided(
            &p,
            slice_view_mut(&mut x, 2, 2),
            slice_view_mut(&mut y, 2, -2),
        );
        assert_eq!(x, [3.0, 9.0, 3.0]);
        assert_eq!(y, [2.0, 9.0, 2.0]);

        // flag = -0.5 (< 0): full H, here a reflection-like [[0, 1], [1, 0]].
        let p = RotmParams {
            flag: -0.5,
            h11: 0.0,
            h12: 1.0,
            h21: 1.0,
            h22: 0.0,
        };
        let mut x = [1.0, 0.0, 2.0];
        let mut y = [5.0, 6.0];
        rotm_strided(
            &p,
            slice_view_mut(&mut x, 2, 2),
            slice_view_mut(&mut y, 2, 1),
        );
        assert_eq!(x, [5.0, 0.0, 6.0]);
        assert_eq!(y, [1.0, 2.0]);

        // flag = -2: identity, whatever the H entries hold.
        let p = RotmParams {
            flag: -2.0,
            h11: f64::NAN,
            h12: f64::NAN,
            h21: f64::NAN,
            h22: f64::NAN,
        };
        let mut x = [1.0, 2.0];
        let mut y = [3.0, 4.0];
        rotm_strided(
            &p,
            slice_view_mut(&mut x, 2, -1),
            slice_view_mut(&mut y, 2, 0),
        );
        assert_eq!((x, y), ([1.0, 2.0], [3.0, 4.0]));
    }

    #[test]
    fn test_issue_15_rotm_strided_zero_increment_reuses_element0() {
        // incx == 0 with flag == 0: x[0] <- x[0] + h12*y[i] for every i, and
        // y[i] <- h21*x[0](current) + y[i] -- the reference loop verbatim.
        let p = RotmParams {
            flag: 0.0,
            h11: 0.0,
            h12: 1.0,
            h21: 2.0,
            h22: 0.0,
        };
        let mut x = [1.0];
        let mut y = [1.0, 2.0, 3.0];
        rotm_strided(
            &p,
            slice_view_mut(&mut x, 3, 0),
            slice_view_mut(&mut y, 3, 1),
        );
        // i=0: w=1, z=1 -> x=2, y0=3; i=1: w=2, z=2 -> x=4, y1=6;
        // i=2: w=4, z=3 -> x=7, y2=11.
        assert_eq!(x, [7.0]);
        assert_eq!(y, [3.0, 6.0, 11.0]);
    }

    #[test]
    fn test_issue_15_rotm_strided_unit_stride_matches_contiguous() {
        let p = RotmParams {
            flag: -1.0,
            h11: 0.6,
            h12: 0.8,
            h21: -0.8,
            h22: 0.6,
        };
        let mut x_legacy = [1.0, 2.0, 3.0];
        let mut y_legacy = [4.0, 5.0, 6.0];
        let mut x_strided = x_legacy;
        let mut y_strided = y_legacy;
        super::super::rotm(&p, &mut x_legacy, &mut y_legacy);
        rotm_strided(
            &p,
            StridedSliceMut::from_slice(&mut x_strided),
            StridedSliceMut::from_slice(&mut y_strided),
        );
        assert_eq!((x_strided, y_strided), (x_legacy, y_legacy));
    }

    /// Naive reference `ZDROT` loop (Netlib `zdrot.f`): real `c`, `s` scale
    /// each component, as gfortran evaluates `C*ZX(IX) + S*ZY(IY)`.
    fn ref_zdrot(
        c: f64,
        s: f64,
        x: &mut [num_complex::Complex64],
        incx: isize,
        y: &mut [num_complex::Complex64],
        incy: isize,
        n: usize,
    ) {
        use num_complex::Complex64;
        for i in 0..n {
            let (ox, oy) = (ref_offset(i, n, incx), ref_offset(i, n, incy));
            let (xv, yv) = (x[ox], y[oy]);
            x[ox] = Complex64::new(c * xv.re + s * yv.re, c * xv.im + s * yv.im);
            y[oy] = Complex64::new(c * yv.re - s * xv.re, c * yv.im - s * xv.im);
        }
    }

    #[test]
    fn test_issue_15_zdrot_strided_matches_reference_loop() {
        use num_complex::Complex64;
        let (c, s) = (0.6, 0.8);
        for &(incx, incy) in &INC_PAIRS {
            for n in [0usize, 1, 3, 5] {
                let xlen = span(n, incx).max(1);
                let ylen = span(n, incy).max(1);
                let x0: Vec<Complex64> = (0..xlen)
                    .map(|k| Complex64::new(1.0 + k as f64, -0.5 * k as f64))
                    .collect();
                let y0: Vec<Complex64> = (0..ylen)
                    .map(|k| Complex64::new(-2.0 + 0.25 * k as f64, 3.0 - k as f64))
                    .collect();
                let (mut xr, mut yr) = (x0.clone(), y0.clone());
                ref_zdrot(c, s, &mut xr, incx, &mut yr, incy, n);
                let (mut xs, mut ys) = (x0, y0);
                zdrot_strided(
                    c,
                    s,
                    StridedSliceMut::try_new(&mut xs, n, incx).expect("view fits"),
                    StridedSliceMut::try_new(&mut ys, n, incy).expect("view fits"),
                );
                assert_eq!(xs, xr, "x: incx={incx}, incy={incy}, n={n}");
                assert_eq!(ys, yr, "y: incx={incx}, incy={incy}, n={n}");
            }
        }
    }

    #[test]
    fn test_issue_15_zdrot_strided_infinite_component_stays_nan_free() {
        use num_complex::Complex64;
        // Component-wise scaling: 0.6*(inf + 0i) + 0.8*(1 + 1i) = inf + 0.8i.
        // A complex multiply by (0.6 + 0i) would fabricate 0*inf = NaN.
        let mut x = [Complex64::new(f64::INFINITY, 0.0), Complex64::new(7.0, 7.0)];
        let mut y = [Complex64::new(1.0, 1.0)];
        zdrot_strided(
            0.6,
            0.8,
            StridedSliceMut::try_new(&mut x, 1, 2).expect("view fits"),
            StridedSliceMut::try_new(&mut y, 1, -1).expect("view fits"),
        );
        assert_eq!(x[0], Complex64::new(f64::INFINITY, 0.8));
        assert_eq!(x[1], Complex64::new(7.0, 7.0));
        assert_eq!(y[0], Complex64::new(f64::NEG_INFINITY, 0.6));
    }

    #[test]
    fn test_issue_15_zdrot_strided_unit_stride_matches_contiguous() {
        use num_complex::Complex32;
        let mut x_legacy = [Complex32::new(1.0, 2.0), Complex32::new(-3.0, 0.5)];
        let mut y_legacy = [Complex32::new(0.0, -1.0), Complex32::new(4.0, 4.0)];
        let mut x_strided = x_legacy;
        let mut y_strided = y_legacy;
        super::super::zdrot(0.8f32, -0.6f32, &mut x_legacy, &mut y_legacy);
        zdrot_strided(
            0.8f32,
            -0.6f32,
            StridedSliceMut::from_slice(&mut x_strided),
            StridedSliceMut::from_slice(&mut y_strided),
        );
        assert_eq!((x_strided, y_strided), (x_legacy, y_legacy));
    }

    #[test]
    fn test_issue_15_cnrm2_strided_matches_naive_for_every_stride() {
        use num_complex::Complex64;
        let buf: Vec<Complex64> = (0..13)
            .map(|k| Complex64::new(0.5 * k as f64 - 2.0, 1.5 - 0.25 * k as f64))
            .collect();
        for inc in [1isize, 2, 3, -1, -2, -4, 0] {
            for n in [0usize, 1, 2, 4] {
                if span(n, inc) > buf.len() {
                    continue;
                }
                let view = StridedSlice::try_new(&buf, n, inc).expect("view fits");
                let naive: f64 = (0..n)
                    .map(|i| buf[ref_offset(i, n, inc)].norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let got = cnrm2_strided(view);
                assert!(
                    (got - naive).abs() <= 8.0 * f64::EPSILON * naive.max(1.0),
                    "inc={inc}, n={n}: got {got}, naive {naive}"
                );
            }
        }
        // Unit stride is forwarded to the contiguous kernel bit-for-bit.
        let view = StridedSlice::from_slice(&buf);
        assert_eq!(cnrm2_strided(view), super::super::cnrm2(&buf));
    }

    #[test]
    fn test_issue_15_cnrm2_strided_zero_and_negative_increments() {
        use num_complex::{Complex32, Complex64};
        // incx == 0: sqrt(n) * |x[0]| = 2 * 5.
        let x = [Complex64::new(3.0, 4.0), Complex64::new(100.0, 0.0)];
        assert_eq!(
            cnrm2_strided(StridedSlice::try_new(&x, 4, 0).expect("view fits")),
            10.0
        );
        let xs = [Complex32::new(3.0, -4.0)];
        assert_eq!(
            cnrm2_strided(StridedSlice::try_new(&xs, 4, 0).expect("view fits")),
            10.0f32
        );
        // Negative increment, magnitudes whose squares overflow: no overflow.
        let big = [
            Complex64::new(1e300, 0.0),
            Complex64::new(-5.0, 5.0),
            Complex64::new(0.0, -1e300),
        ];
        let got = cnrm2_strided(StridedSlice::try_new(&big, 2, -2).expect("view fits"));
        let expected = std::f64::consts::SQRT_2 * 1e300;
        assert!(
            (got - expected).abs() <= 4.0 * f64::EPSILON * expected,
            "got {got}"
        );
        // ... and tiny ones whose squares underflow: no flush to zero.
        let tiny = [Complex64::new(3e-300, 4e-300); 3];
        let got = cnrm2_strided(StridedSlice::try_new(&tiny, 3, -1).expect("view fits"));
        let expected = 3.0f64.sqrt() * 5e-300;
        assert!(
            (got - expected).abs() <= 8.0 * f64::EPSILON * expected,
            "got {got}"
        );
    }
}
