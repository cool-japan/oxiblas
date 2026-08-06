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
//!   i.e. the same element is read `n` times. Reference BLAS permits this for
//!   the "product/copy" family (DOT/AXPY/COPY/SWAP/ROT); for the "reduction"
//!   family (SCAL/NRM2/ASUM/IAMAX) a non-positive increment is instead defined
//!   as a no-op / zero result, and the routines below honour that split.
//!
//! # Intended CBLAS integration point
//!
//! The CBLAS compatibility layer (`crate::cblas`) receives raw
//! `*const T`/`*mut T` pointers together with an `i32` increment. In a future
//! integration pass it should build a [`StridedSlice`] / [`StridedSliceMut`]
//! via the `unsafe` [`StridedSlice::from_raw_parts`] /
//! [`StridedSliceMut::from_raw_parts`] constructors (which internally form a
//! slice spanning the full strided extent `(n - 1) * |incx| + 1`) and forward
//! to the `*_strided` routines here. That yields genuine non-unit-stride and
//! negative-increment correctness — e.g. extracting a column of a row-major
//! matrix — instead of merely re-slicing a contiguous buffer.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};

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
    /// * `len` and `stride` must not overflow `usize` when forming the extent.
    #[inline]
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize, stride: isize) -> Self {
        let extent = strided_extent(len, stride).expect("strided extent must not overflow usize");
        // SAFETY: contract of this function guarantees `ptr` is valid for
        // `extent` reads for `'a` and points into a single allocation.
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
    /// Same obligations as [`StridedSlice::from_raw_parts`], plus: the region
    /// must be uniquely borrowed (no other live reference, and no aliasing with
    /// any other `x`/`y` passed to the same routine) for `'a`.
    #[inline]
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, stride: isize) -> Self {
        let extent = strided_extent(len, stride).expect("strided extent must not overflow usize");
        // SAFETY: contract of this function guarantees `ptr` is valid for
        // `extent` reads+writes for `'a`, uniquely borrowed, single allocation.
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

// =============================================================================
// Reduction family: a non-positive increment is a no-op / zero result,
// matching Netlib reference behaviour (see module docs).
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
/// General entry point behind the contiguous [`nrm2`](super::nrm2). Per
/// reference BLAS a non-positive increment returns `0`. Uses the same
/// overflow-safe scaled-sum-of-squares recurrence as the contiguous kernel.
#[must_use]
pub fn nrm2_strided<T: Real>(x: StridedSlice<'_, T>) -> T {
    let n = x.len();
    if n == 0 || x.stride() <= 0 {
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
    let mut max_val = Scalar::abs(x.get(0));
    for i in 1..n {
        let abs_xi = Scalar::abs(x.get(i));
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
    let mut min_val = Scalar::abs(x.get(0));
    for i in 1..n {
        let abs_xi = Scalar::abs(x.get(i));
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
    fn test_nrm2_nonpositive_is_zero() {
        let xbuf = [3.0, 4.0];
        assert_eq!(nrm2_strided(slice_view(&xbuf, 2, -1)), 0.0);
        assert_eq!(nrm2_strided(slice_view(&xbuf, 2, 0)), 0.0);
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
}
