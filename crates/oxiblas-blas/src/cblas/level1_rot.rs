//! CBLAS Level-1 plane rotations and the remaining reference Level-1 extras.
//!
//! C-ABI wrappers (issue #14) for the real rotation family `srotg`/`drotg`,
//! `srotmg`/`drotmg`, `srot`/`drot`, `srotm`/`drotm`, the complex
//! `crotg`/`zrotg` and `csrot`/`zdrot`, the double-accumulating dot products
//! `sdsdot`/`dsdot`, and the `|Re| + |Im|` helpers `scabs1`/`dcabs1`.
//!
//! Every wrapper forwards to the native routine of the same name in
//! [`crate::level1`] and follows netlib CBLAS / reference BLAS 3.12:
//!
//! * **Increments** (ROT, ROTM, CSROT, ZDROT, SDSDOT, DSDOT): every increment
//!   is valid. `x`/`y` are the array base the caller holds (the lowest address
//!   the vector touches) for either sign of the increment. A negative
//!   increment walks the vector back-to-front — logical element `i` lives at
//!   offset `(n - 1 - i)·|inc|`, the reference start `(1 - n)·inc` walked
//!   backwards — and a zero increment re-uses element 0 on every iteration,
//!   exactly like the reference loop (`drot` with `incx = 0` rotates `x[0]`
//!   against each `y` element in turn). The views are built with
//!   [`StridedSliceMut::from_raw_parts`] / [`StridedSlice::from_raw_parts`]
//!   and handed to the `*_strided` natives, which dispatch unit strides to the
//!   contiguous kernels; no gather/scatter copy is made (that would be wrong
//!   for a zero increment).
//! * **`n <= 0`** is a no-op for the rotations and returns the identity for
//!   the dots (`alpha` for SDSDOT, `0` for DSDOT).
//! * **Unsatisfiable extents**: an `(n, inc)` whose extent `(n - 1)·|inc| + 1`
//!   overflows `usize` (32-bit targets) or spans more than `isize::MAX` bytes
//!   (64-bit targets too, with `n` and `|inc|` near `2^31`) cannot describe a
//!   real allocation. [`StridedSlice::from_raw_parts`] turns it into an empty
//!   view, and the wrapper then behaves as for `n <= 0` instead of letting the
//!   natives' equal-length assert abort the C caller. The unit-stride branch of
//!   SDSDOT/DSDOT applies the same `isize::MAX`-byte bound to `n`, which only a
//!   32-bit target can exceed.
//! * **ROTMG** writes `P[0] = flag` and only the `H` entries that flag uses,
//!   as reference `?ROTMG` does: `flag = -1` writes `P[1..=4] = h11, h21,
//!   h12, h22` (column-major `H`), `flag = 0` writes `P[2] = h21`,
//!   `P[3] = h12`, `flag = 1` writes `P[1] = h11`, `P[4] = h22`, and
//!   `flag = -2` writes nothing else. ROTM reads only the entries its flag
//!   uses, with the same layout.
//! * **Null pointers**: a C ABI cannot report an error and a panic reaching it
//!   would abort the process, so a null operand makes the routine a no-op;
//!   value-returning routines return their identity (`alpha` for SDSDOT, `0`
//!   for DSDOT, SCABS1 and DCABS1).
//! * **Aliasing**: as in the reference, `x` and `y` (and the scalar operands
//!   of ROTG/ROTMG) must not overlap.

use crate::level1::{self, RotmParams, StridedSlice, StridedSliceMut};
use num_complex::{Complex32, Complex64};
use num_traits::Float;

// =============================================================================
// ROTG — generate a Givens rotation
// =============================================================================

/// Single precision Givens rotation generation (`SROTG`).
///
/// Computes `c`, `s` and `r` with `[c s; -s c]·[a; b] = [r; 0]` following the
/// reference `srotg.f90` (3.12), then overwrites `a` with `r` and `b` with the
/// reconstruction value `z` (`b == 0`: `c = 1, s = 0, z = 0, r = a`;
/// `a == 0, b != 0`: `c = 0, s = 1, r = b, z = 1`).
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `a`, `b`, `c` and `s` each point to one valid, properly aligned,
///   initialized `f32` that may be read and written (if any of them is null
///   the call is a no-op).
/// - The four pointers do not alias each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_srotg(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) {
    if a.is_null() || b.is_null() || c.is_null() || s.is_null() {
        return;
    }
    // SAFETY: all four pointers are non-null, valid and non-aliasing (contract).
    unsafe { rotg_impl(a, b, c, s) }
}

/// Double precision Givens rotation generation (`DROTG`).
///
/// Computes `c`, `s` and `r` with `[c s; -s c]·[a; b] = [r; 0]` following the
/// reference `drotg.f90` (3.12), then overwrites `a` with `r` and `b` with the
/// reconstruction value `z` (`b == 0`: `c = 1, s = 0, z = 0, r = a`;
/// `a == 0, b != 0`: `c = 0, s = 1, r = b, z = 1`).
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `a`, `b`, `c` and `s` each point to one valid, properly aligned,
///   initialized `f64` that may be read and written (if any of them is null
///   the call is a no-op).
/// - The four pointers do not alias each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_drotg(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64) {
    if a.is_null() || b.is_null() || c.is_null() || s.is_null() {
        return;
    }
    // SAFETY: all four pointers are non-null, valid and non-aliasing (contract).
    unsafe { rotg_impl(a, b, c, s) }
}

/// Shared `?ROTG` body: `a := r`, `b := z`, `c := c`, `s := s`.
///
/// # Safety
/// The four pointers must be non-null, valid for reads and writes of one `T`
/// each, and must not alias.
unsafe fn rotg_impl<T: Float>(a: *mut T, b: *mut T, c: *mut T, s: *mut T) {
    // SAFETY: guaranteed by this function's contract.
    unsafe {
        let g = level1::rotg(*a, *b);
        *a = g.r;
        *b = g.z;
        *c = g.c;
        *s = g.s;
    }
}

// =============================================================================
// ROTMG — generate a modified Givens rotation
// =============================================================================

/// Single precision modified Givens rotation generation (`SROTMG`).
///
/// Given the scaling factors `d1`, `d2` and the vector `(b1, b2)`, computes
/// the modified rotation `H` that zeroes the second component of
/// `(sqrt(d1)·b1, sqrt(d2)·b2)`, updates `d1`, `d2`, `b1` in place, and stores
/// the flag and the flag-dependent `H` entries in `p` (see the module docs for
/// the exact write mask; entries the flag does not use are left untouched).
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `d1`, `d2` and `b1` each point to one valid, properly aligned,
///   initialized `f32` that may be read and written.
/// - `p` points to a properly aligned array of 5 `f32` (`SPARAM`) that may
///   be written.
/// - If any of the pointers is null the call is a no-op; the four pointers do
///   not alias each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_srotmg(
    d1: *mut f32,
    d2: *mut f32,
    b1: *mut f32,
    b2: f32,
    p: *mut f32,
) {
    if d1.is_null() || d2.is_null() || b1.is_null() || p.is_null() {
        return;
    }
    // SAFETY: non-null, valid, non-aliasing scalars and a 5-element `p` (contract).
    unsafe { rotmg_impl(d1, d2, b1, b2, p) }
}

/// Double precision modified Givens rotation generation (`DROTMG`).
///
/// Given the scaling factors `d1`, `d2` and the vector `(b1, b2)`, computes
/// the modified rotation `H` that zeroes the second component of
/// `(sqrt(d1)·b1, sqrt(d2)·b2)`, updates `d1`, `d2`, `b1` in place, and stores
/// the flag and the flag-dependent `H` entries in `p` (see the module docs for
/// the exact write mask; entries the flag does not use are left untouched).
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `d1`, `d2` and `b1` each point to one valid, properly aligned,
///   initialized `f64` that may be read and written.
/// - `p` points to a properly aligned array of 5 `f64` (`DPARAM`) that may
///   be written.
/// - If any of the pointers is null the call is a no-op; the four pointers do
///   not alias each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_drotmg(
    d1: *mut f64,
    d2: *mut f64,
    b1: *mut f64,
    b2: f64,
    p: *mut f64,
) {
    if d1.is_null() || d2.is_null() || b1.is_null() || p.is_null() {
        return;
    }
    // SAFETY: non-null, valid, non-aliasing scalars and a 5-element `p` (contract).
    unsafe { rotmg_impl(d1, d2, b1, b2, p) }
}

/// Shared `?ROTMG` body.
///
/// # Safety
/// `d1`, `d2`, `b1` must be valid for reads and writes of one `T` each and `p`
/// for writes of 5 `T`; none of them may alias.
unsafe fn rotmg_impl<T: Float>(d1: *mut T, d2: *mut T, b1: *mut T, b2: T, p: *mut T) {
    // SAFETY: guaranteed by this function's contract.
    unsafe {
        let (mut dd1, mut dd2, mut x1) = (*d1, *d2, *b1);
        let params = level1::rotmg(&mut dd1, &mut dd2, &mut x1, b2);
        *d1 = dd1;
        *d2 = dd2;
        *b1 = x1;
        store_rotmg_params(&params, p);
    }
}

/// Stores `params` into a reference `PARAM` array exactly like `?ROTMG`:
/// `flag == -2` writes only `P[0]` (the reference returns before touching the
/// rest); otherwise `flag < 0` writes all four `H` entries, `flag == 0` the
/// off-diagonal pair and any other flag the diagonal pair, in the column-major
/// order `[flag, h11, h21, h12, h22]`.
///
/// # Safety
/// `p` must be valid for writes of 5 `T`.
unsafe fn store_rotmg_params<T: Float>(params: &RotmParams<T>, p: *mut T) {
    let zero = T::zero();
    let two = T::one() + T::one();
    // SAFETY: `p` holds 5 elements (contract); every offset below is < 5.
    unsafe {
        if params.flag + two != zero {
            if params.flag < zero {
                *p.add(1) = params.h11;
                *p.add(2) = params.h21;
                *p.add(3) = params.h12;
                *p.add(4) = params.h22;
            } else if params.flag == zero {
                *p.add(2) = params.h21;
                *p.add(3) = params.h12;
            } else {
                *p.add(1) = params.h11;
                *p.add(4) = params.h22;
            }
        }
        *p = params.flag;
    }
}

// =============================================================================
// ROT — apply a Givens rotation
// =============================================================================

/// Single precision plane rotation (`SROT`): for each logical pair,
/// `x' = c·x + s·y`, `y' = c·y - s·x`.
///
/// Any increment is valid, including zero and negative ones (see the module
/// docs). `n <= 0` is a no-op.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `f32`, and valid for
///   reads and writes of the `(n - 1)·|incx| + 1` elements starting at `x`
///   (the array base, the lowest address the strided walk touches for every
///   sign of `incx`).
/// - `y` satisfies the same requirement with `incy`.
/// - The regions of `x` and `y` do not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_srot(
    n: i32,
    x: *mut f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
    c: f32,
    s: f32,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    // SAFETY: n > 0, non-null operands covering their strided extents (contract).
    unsafe { rot_impl(n, x, incx, y, incy, c, s) }
}

/// Double precision plane rotation (`DROT`): for each logical pair,
/// `x' = c·x + s·y`, `y' = c·y - s·x`.
///
/// Any increment is valid, including zero and negative ones (see the module
/// docs). `n <= 0` is a no-op.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `f64`, and valid for
///   reads and writes of the `(n - 1)·|incx| + 1` elements starting at `x`
///   (the array base, the lowest address the strided walk touches for every
///   sign of `incx`).
/// - `y` satisfies the same requirement with `incy`.
/// - The regions of `x` and `y` do not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_drot(
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
    c: f64,
    s: f64,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    // SAFETY: n > 0, non-null operands covering their strided extents (contract).
    unsafe { rot_impl(n, x, incx, y, incy, c, s) }
}

/// Shared `?ROT` body: strided views over the caller's buffers, handed to
/// [`level1::rot_strided`] (unit strides go to the contiguous `rot` kernel).
///
/// # Safety
/// `n > 0`; `x`/`y` non-null, non-overlapping and valid for reads and writes
/// of `(n - 1)·|inc| + 1` elements each.
unsafe fn rot_impl<T: Float>(n: i32, x: *mut T, incx: i32, y: *mut T, incy: i32, c: T, s: T) {
    let len = n as usize;
    // SAFETY: guaranteed by this function's contract; an extent no allocation
    // can hold yields an empty view that never touches the pointer.
    let (xv, yv) = unsafe {
        (
            StridedSliceMut::from_raw_parts(x, len, incx as isize),
            StridedSliceMut::from_raw_parts(y, len, incy as isize),
        )
    };
    // `from_raw_parts` returns an empty view for an extent that overflows
    // `usize` (32-bit targets) or spans more than `isize::MAX` bytes (64-bit
    // targets too). A lone empty view would trip the equal-length assert in the
    // native, and a panic would abort the C caller, so do nothing.
    if xv.len() != len || yv.len() != len {
        return;
    }
    level1::rot_strided(c, s, xv, yv);
}

// =============================================================================
// ROTM — apply a modified Givens rotation
// =============================================================================

/// Single precision modified rotation (`SROTM`).
///
/// `p` is the reference `SPARAM` array `[flag, h11, h21, h12, h22]`
/// (column-major `H`). Classified exactly like Netlib `SROTM`:
/// `flag == -2` (or `n <= 0`) is a no-op, `flag < 0` uses the full `H`,
/// `flag == 0` the unit diagonal `[[1, h12], [h21, 1]]`, and any other flag
/// `[[h11, 1], [-1, h22]]`. Only the entries the flag uses are read. Any
/// increment is valid, including zero and negative ones.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `f32`, and valid for
///   reads and writes of the `(n - 1)·|incx| + 1` elements starting at `x`
///   (the array base, the lowest address the strided walk touches for every
///   sign of `incx`).
/// - `y` satisfies the same requirement with `incy`; `x` and `y` do not
///   overlap.
/// - `p` (when `n > 0`) is non-null, properly aligned and points to 5
///   readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_srotm(
    n: i32,
    x: *mut f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
    p: *const f32,
) {
    if n <= 0 || x.is_null() || y.is_null() || p.is_null() {
        return;
    }
    // SAFETY: n > 0, non-null operands covering their strided extents and a
    // 5-element `p` (contract).
    unsafe { rotm_impl(n, x, incx, y, incy, p) }
}

/// Double precision modified rotation (`DROTM`).
///
/// `p` is the reference `DPARAM` array `[flag, h11, h21, h12, h22]`
/// (column-major `H`). Classified exactly like Netlib `DROTM`:
/// `flag == -2` (or `n <= 0`) is a no-op, `flag < 0` uses the full `H`,
/// `flag == 0` the unit diagonal `[[1, h12], [h21, 1]]`, and any other flag
/// `[[h11, 1], [-1, h22]]`. Only the entries the flag uses are read. Any
/// increment is valid, including zero and negative ones.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `f64`, and valid for
///   reads and writes of the `(n - 1)·|incx| + 1` elements starting at `x`
///   (the array base, the lowest address the strided walk touches for every
///   sign of `incx`).
/// - `y` satisfies the same requirement with `incy`; `x` and `y` do not
///   overlap.
/// - `p` (when `n > 0`) is non-null, properly aligned and points to 5
///   readable `f64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_drotm(
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
    p: *const f64,
) {
    if n <= 0 || x.is_null() || y.is_null() || p.is_null() {
        return;
    }
    // SAFETY: n > 0, non-null operands covering their strided extents and a
    // 5-element `p` (contract).
    unsafe { rotm_impl(n, x, incx, y, incy, p) }
}

/// Shared `?ROTM` body.
///
/// Reads `PARAM` like the reference: the flag first, then only the `H`
/// entries that flag uses (the unused fields get the implied values, which
/// [`level1::rotm_strided`] never reads anyway).
///
/// # Safety
/// `n > 0`; `x`/`y` non-null, non-overlapping and valid for reads and writes
/// of `(n - 1)·|inc| + 1` elements each; `p` valid for reads of 5 `T`.
unsafe fn rotm_impl<T: Float>(n: i32, x: *mut T, incx: i32, y: *mut T, incy: i32, p: *const T) {
    let zero = T::zero();
    let one = T::one();
    // SAFETY: `p` holds 5 elements (contract); every offset below is < 5.
    let params = unsafe {
        let flag = *p;
        if flag + (one + one) == zero {
            // DFLAG + TWO == 0: identity, the reference returns at once.
            return;
        }
        if flag < zero {
            RotmParams {
                flag,
                h11: *p.add(1),
                h12: *p.add(3),
                h21: *p.add(2),
                h22: *p.add(4),
            }
        } else if flag == zero {
            RotmParams {
                flag,
                h11: one,
                h12: *p.add(3),
                h21: *p.add(2),
                h22: one,
            }
        } else {
            RotmParams {
                flag,
                h11: *p.add(1),
                h12: one,
                h21: -one,
                h22: *p.add(4),
            }
        }
    };
    let len = n as usize;
    // SAFETY: guaranteed by this function's contract.
    let (xv, yv) = unsafe {
        (
            StridedSliceMut::from_raw_parts(x, len, incx as isize),
            StridedSliceMut::from_raw_parts(y, len, incy as isize),
        )
    };
    // Unsatisfiable-extent fallback of `from_raw_parts` (see `rot_impl`).
    if xv.len() != len || yv.len() != len {
        return;
    }
    level1::rotm_strided(&params, xv, yv);
}

// =============================================================================
// CROTG / ZROTG — complex Givens rotation generation
// =============================================================================

/// Complex single precision Givens rotation generation (`CROTG`).
///
/// Computes the real cosine `c` and complex sine `s` with
/// `[c s; -conj(s) c]·[a; b] = [r; 0]` (reference `crotg.f90`, 3.12) and
/// overwrites `a` with `r`; `b` is only read. Conventions: `b == 0` gives
/// `c = 1, s = 0, r = a`; `a == 0, b != 0` gives `c = 0, r = |b|`,
/// `s = conj(b)/|b|`.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `a` and `s` each point to one valid, properly aligned, initialized
///   `Complex32` that may be written (`a` is also read), `b` to one readable
///   `Complex32`, and `c` to one writable `f32`.
/// - If any of the pointers is null the call is a no-op; `a`, `c` and `s` do
///   not alias each other or `b`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_crotg(
    a: *mut Complex32,
    b: *const Complex32,
    c: *mut f32,
    s: *mut Complex32,
) {
    if a.is_null() || b.is_null() || c.is_null() || s.is_null() {
        return;
    }
    // SAFETY: all pointers are non-null, valid and non-aliasing (contract).
    unsafe {
        let g = level1::crotg(*a, *b);
        *a = g.r;
        *c = g.c;
        *s = g.s;
    }
}

/// Complex double precision Givens rotation generation (`ZROTG`).
///
/// Computes the real cosine `c` and complex sine `s` with
/// `[c s; -conj(s) c]·[a; b] = [r; 0]` (reference `zrotg.f90`, 3.12) and
/// overwrites `a` with `r`; `b` is only read. Conventions: `b == 0` gives
/// `c = 1, s = 0, r = a`; `a == 0, b != 0` gives `c = 0, r = |b|`,
/// `s = conj(b)/|b|`.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `a` and `s` each point to one valid, properly aligned, initialized
///   `Complex64` that may be written (`a` is also read), `b` to one readable
///   `Complex64`, and `c` to one writable `f64`.
/// - If any of the pointers is null the call is a no-op; `a`, `c` and `s` do
///   not alias each other or `b`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zrotg(
    a: *mut Complex64,
    b: *const Complex64,
    c: *mut f64,
    s: *mut Complex64,
) {
    if a.is_null() || b.is_null() || c.is_null() || s.is_null() {
        return;
    }
    // SAFETY: all pointers are non-null, valid and non-aliasing (contract).
    unsafe {
        let g = level1::crotg(*a, *b);
        *a = g.r;
        *c = g.c;
        *s = g.s;
    }
}

// =============================================================================
// CSROT / ZDROT — real rotation applied to complex vectors
// =============================================================================

/// Complex single precision vectors rotated by a real rotation (`CSROT`):
/// `x' = c·x + s·y`, `y' = c·y - s·x` with real `c`, `s` scaling the real and
/// imaginary parts separately.
///
/// Any increment is valid, including zero and negative ones (see the module
/// docs). `n <= 0` is a no-op.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `Complex32`, and
///   valid for reads and writes of the `(n - 1)·|incx| + 1` elements starting
///   at `x` (the array base, the lowest address the strided walk touches for
///   every sign of `incx`).
/// - `y` satisfies the same requirement with `incy`.
/// - The regions of `x` and `y` do not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_csrot(
    n: i32,
    x: *mut Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
    c: f32,
    s: f32,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    let len = n as usize;
    // SAFETY: n > 0, non-null, non-overlapping operands covering their
    // strided extents (contract).
    let (xv, yv) = unsafe {
        (
            StridedSliceMut::from_raw_parts(x, len, incx as isize),
            StridedSliceMut::from_raw_parts(y, len, incy as isize),
        )
    };
    // Unsatisfiable-extent fallback of `from_raw_parts` (see `rot_impl`).
    if xv.len() != len || yv.len() != len {
        return;
    }
    level1::zdrot_strided(c, s, xv, yv);
}

/// Complex double precision vectors rotated by a real rotation (`ZDROT`):
/// `x' = c·x + s·y`, `y' = c·y - s·x` with real `c`, `s` scaling the real and
/// imaginary parts separately.
///
/// Any increment is valid, including zero and negative ones (see the module
/// docs). `n <= 0` is a no-op.
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `Complex64`, and
///   valid for reads and writes of the `(n - 1)·|incx| + 1` elements starting
///   at `x` (the array base, the lowest address the strided walk touches for
///   every sign of `incx`).
/// - `y` satisfies the same requirement with `incy`.
/// - The regions of `x` and `y` do not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdrot(
    n: i32,
    x: *mut Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
    c: f64,
    s: f64,
) {
    if n <= 0 || x.is_null() || y.is_null() {
        return;
    }
    let len = n as usize;
    // SAFETY: n > 0, non-null, non-overlapping operands covering their
    // strided extents (contract).
    let (xv, yv) = unsafe {
        (
            StridedSliceMut::from_raw_parts(x, len, incx as isize),
            StridedSliceMut::from_raw_parts(y, len, incy as isize),
        )
    };
    // Unsatisfiable-extent fallback of `from_raw_parts` (see `rot_impl`).
    if xv.len() != len || yv.len() != len {
        return;
    }
    level1::zdrot_strided(c, s, xv, yv);
}

// =============================================================================
// SDSDOT / DSDOT — single precision dot products accumulated in double
// =============================================================================

/// `alpha + x·y` for single precision vectors, accumulated in double precision
/// and rounded to `f32` once at the end (`SDSDOT`).
///
/// Reference `sdsdot.f`: the accumulator starts at `alpha` (in double) and
/// every product `dble(x)·dble(y)` is added in double. `n <= 0` (or a null
/// operand) returns `alpha`, and so does an `(n, inc)` that no allocation can
/// hold (see the module docs). Any increment is valid, including zero and
/// negative ones (see the module docs).
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `f32`, and valid for
///   reads of the `(n - 1)·|incx| + 1` elements starting at `x` (the array
///   base, the lowest address the strided walk touches for every sign of
///   `incx`).
/// - `y` satisfies the same requirement with `incy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sdsdot(
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f32 {
    if n <= 0 || x.is_null() || y.is_null() {
        return alpha;
    }
    let len = n as usize;
    if incx == 1 && incy == 1 {
        // `len` f32s beyond `isize::MAX` bytes (32-bit targets only) cannot be a
        // real buffer, and `slice::from_raw_parts` forbids such a length.
        if !level1::fits_isize_bytes::<f32>(len) {
            return alpha;
        }
        // SAFETY: unit strides; both operands hold `len` readable elements, and
        // `len` elements fit in `isize::MAX` bytes (checked above).
        let (xs, ys) = unsafe {
            (
                core::slice::from_raw_parts(x, len),
                core::slice::from_raw_parts(y, len),
            )
        };
        return level1::sdsdot(alpha, xs, ys);
    }
    // SAFETY: n > 0, non-null operands covering their strided extents (contract).
    let (xv, yv) = unsafe {
        (
            StridedSlice::from_raw_parts(x, len, incx as isize),
            StridedSlice::from_raw_parts(y, len, incy as isize),
        )
    };
    // Unsatisfiable-extent fallback of `from_raw_parts` (see `rot_impl`): a lone
    // empty view would index out of bounds in `strided_dot_f64`.
    if xv.len() != len || yv.len() != len {
        return alpha;
    }
    strided_dot_f64(f64::from(alpha), xv, yv) as f32
}

/// `x·y` for single precision vectors, accumulated and returned in double
/// precision (`DSDOT`).
///
/// Reference `dsdot.f`: the accumulator starts at `0` and every product
/// `dble(x)·dble(y)` is added in double. `n <= 0` (or a null operand) returns
/// `0`, and so does an `(n, inc)` that no allocation can hold (see the module
/// docs). Any increment is valid, including zero and negative ones (see the
/// module docs).
///
/// # Safety
///
/// This is a C ABI entry point; the raw pointers are trusted. The caller must
/// ensure:
///
/// - `x` (when `n > 0`) is non-null, properly aligned for `f32`, and valid for
///   reads of the `(n - 1)·|incx| + 1` elements starting at `x` (the array
///   base, the lowest address the strided walk touches for every sign of
///   `incx`).
/// - `y` satisfies the same requirement with `incy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsdot(
    n: i32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f64 {
    if n <= 0 || x.is_null() || y.is_null() {
        return 0.0;
    }
    let len = n as usize;
    if incx == 1 && incy == 1 {
        // `len` f32s beyond `isize::MAX` bytes (32-bit targets only) cannot be a
        // real buffer, and `slice::from_raw_parts` forbids such a length.
        if !level1::fits_isize_bytes::<f32>(len) {
            return 0.0;
        }
        // SAFETY: unit strides; both operands hold `len` readable elements, and
        // `len` elements fit in `isize::MAX` bytes (checked above).
        let (xs, ys) = unsafe {
            (
                core::slice::from_raw_parts(x, len),
                core::slice::from_raw_parts(y, len),
            )
        };
        return level1::dsdot(xs, ys);
    }
    // SAFETY: n > 0, non-null operands covering their strided extents (contract).
    let (xv, yv) = unsafe {
        (
            StridedSlice::from_raw_parts(x, len, incx as isize),
            StridedSlice::from_raw_parts(y, len, incy as isize),
        )
    };
    // Unsatisfiable-extent fallback of `from_raw_parts` (see `rot_impl`): a lone
    // empty view would index out of bounds in `strided_dot_f64`.
    if xv.len() != len || yv.len() != len {
        return 0.0;
    }
    strided_dot_f64(0.0, xv, yv)
}

/// The reference strided loop of `SDSDOT`/`DSDOT`: starting from `init`
/// (`dble(alpha)` / `0`), add `dble(x_i)·dble(y_i)` in double precision in
/// logical order (a zero increment re-reads element 0, a negative one walks
/// backwards), so the rounding sequence is the reference one.
fn strided_dot_f64(init: f64, x: StridedSlice<'_, f32>, y: StridedSlice<'_, f32>) -> f64 {
    let mut acc = init;
    for i in 0..x.len() {
        acc += f64::from(x.get(i)) * f64::from(y.get(i));
    }
    acc
}

// =============================================================================
// SCABS1 / DCABS1 — |Re(z)| + |Im(z)|
// =============================================================================

/// `|Re(z)| + |Im(z)|` for a single precision complex scalar (`SCABS1`).
///
/// A null `z` returns `0`.
///
/// # Safety
///
/// This is a C ABI entry point; `z` must be null or point to one valid,
/// properly aligned, initialized `Complex32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scabs1(z: *const Complex32) -> f32 {
    if z.is_null() {
        return 0.0;
    }
    // SAFETY: non-null and valid for one read (contract).
    level1::scabs1(unsafe { *z })
}

/// `|Re(z)| + |Im(z)|` for a double precision complex scalar (`DCABS1`).
///
/// A null `z` returns `0`.
///
/// # Safety
///
/// This is a C ABI entry point; `z` must be null or point to one valid,
/// properly aligned, initialized `Complex64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dcabs1(z: *const Complex64) -> f64 {
    if z.is_null() {
        return 0.0;
    }
    // SAFETY: non-null and valid for one read (contract).
    level1::dcabs1(unsafe { *z })
}

#[cfg(test)]
#[path = "level1_rot_tests.rs"]
mod tests;
