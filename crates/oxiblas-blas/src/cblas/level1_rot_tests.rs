//! Unit tests for the CBLAS Level-1 rotation wrappers and extras (issue #14).
//!
//! The reference loops below are written straight from the Netlib sources
//! (`drot.f`, `drotm.f`, `zdrot.f`, `sdsdot.f`, `dsdot.f`): the walk starts at
//! offset `(1 - n)·inc` for a negative increment (0 otherwise) and advances by
//! `inc` each iteration, so a zero increment revisits element 0. They never
//! use `StridedSlice`, so the wrappers are not tested against themselves.

use super::*;
use core::fmt::Debug;

/// Increments exercised by every strided test (reference BLAS allows all).
const INCS: [isize; 5] = [1, 2, -1, -3, 0];

/// Real test scalar: f32 or f64 with exact bit access.
trait TestFloat: Float + Debug {
    fn from_f64(v: f64) -> Self;
    fn bits(self) -> u64;
}

impl TestFloat for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl TestFloat for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

/// Netlib start offset of a strided walk of length `n`.
fn start(n: usize, inc: isize) -> isize {
    if inc < 0 { (1 - n as isize) * inc } else { 0 }
}

/// Number of elements a strided vector spans: `(n - 1)·|inc| + 1`.
fn extent(n: usize, inc: isize) -> usize {
    (n - 1) * inc.unsigned_abs() + 1
}

/// Distinct, non-trivial values for a vector buffer of `len` elements followed
/// by 2 guard slots holding `guard` (which no call may touch).
fn buffer<T: TestFloat>(len: usize, seed: usize, guard: T) -> Vec<T> {
    let mut v: Vec<T> = (0..len)
        .map(|i| T::from_f64(((7 * i + 3 * seed) % 13) as f64 * 0.375 - 2.0))
        .collect();
    v.push(guard);
    v.push(guard);
    v
}

fn bits_of<T: TestFloat>(v: &[T]) -> Vec<u64> {
    v.iter().map(|x| x.bits()).collect()
}

// =============================================================================
// ROT
// =============================================================================

type RotFn<T> = unsafe extern "C" fn(i32, *mut T, i32, *mut T, i32, T, T);

/// Netlib `DROT` loop.
fn ref_rot<T: TestFloat>(n: usize, x: &mut [T], incx: isize, y: &mut [T], incy: isize, c: T, s: T) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let (xi, yi) = (x[ix as usize], y[iy as usize]);
        let temp = c * xi + s * yi;
        y[iy as usize] = c * yi - s * xi;
        x[ix as usize] = temp;
        ix += incx;
        iy += incy;
    }
}

fn check_rot_increments<T: TestFloat>(func: RotFn<T>) {
    let (c, s) = (T::from_f64(0.6), T::from_f64(-0.8));
    let guard = T::from_f64(-777.0);
    for n in [1usize, 3, 5] {
        for incx in INCS {
            for incy in INCS {
                let mut x = buffer::<T>(extent(n, incx), 1, guard);
                let mut y = buffer::<T>(extent(n, incy), 2, guard);
                let (mut xr, mut yr) = (x.clone(), y.clone());
                ref_rot(n, &mut xr, incx, &mut yr, incy, c, s);
                // SAFETY: x / y span `extent(n, inc)` elements from their base.
                unsafe {
                    func(
                        n as i32,
                        x.as_mut_ptr(),
                        incx as i32,
                        y.as_mut_ptr(),
                        incy as i32,
                        c,
                        s,
                    );
                }
                assert_eq!(
                    bits_of(&x),
                    bits_of(&xr),
                    "rot x n={n} incx={incx} incy={incy}"
                );
                assert_eq!(
                    bits_of(&y),
                    bits_of(&yr),
                    "rot y n={n} incx={incx} incy={incy}"
                );
            }
        }
    }
}

#[test]
fn test_issue_14_srot_drot_every_increment_matches_reference_loop() {
    check_rot_increments::<f32>(cblas_srot);
    check_rot_increments::<f64>(cblas_drot);
}

#[test]
fn test_issue_14_rot_zero_increment_revisits_element_zero() {
    // incx = 0: x[0] is rotated against y[0], then (updated) against y[1].
    let mut x = [1.0f64];
    let mut y = [2.0f64, 3.0];
    // SAFETY: x spans 1 element (inc 0), y spans 2.
    unsafe { cblas_drot(2, x.as_mut_ptr(), 0, y.as_mut_ptr(), 1, 0.0, 1.0) };
    // Step 1 (c=0, s=1): x = y0 = 2, y0 = -1. Step 2: x = y1 = 3, y1 = -2.
    assert_eq!(x, [3.0]);
    assert_eq!(y, [-1.0, -2.0]);
}

#[test]
fn test_issue_14_rot_n_nonpositive_and_null_are_noops() {
    let mut x = [1.0f32, 2.0];
    let mut y = [3.0f32, 4.0];
    // SAFETY: n <= 0 / a null operand dereference nothing.
    unsafe {
        cblas_srot(0, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 0.0, 1.0);
        cblas_srot(-3, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 0.0, 1.0);
        cblas_srot(2, std::ptr::null_mut(), 1, y.as_mut_ptr(), 1, 0.0, 1.0);
        cblas_srot(2, x.as_mut_ptr(), 1, std::ptr::null_mut(), 1, 0.0, 1.0);
    }
    assert_eq!((x, y), ([1.0, 2.0], [3.0, 4.0]));
}

// =============================================================================
// ROTM
// =============================================================================

type RotmFn<T> = unsafe extern "C" fn(i32, *mut T, i32, *mut T, i32, *const T);

/// Netlib `DROTM` loop: `p = [flag, h11, h21, h12, h22]`, reading only the
/// entries the flag uses.
fn ref_rotm<T: TestFloat>(
    n: usize,
    x: &mut [T],
    incx: isize,
    y: &mut [T],
    incy: isize,
    p: &[T; 5],
) {
    let (zero, two) = (T::zero(), T::from_f64(2.0));
    let flag = p[0];
    if n == 0 || flag + two == zero {
        return;
    }
    let (mut kx, mut ky) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let (w, z) = (x[kx as usize], y[ky as usize]);
        let (xn, yn) = if flag < zero {
            (w * p[1] + z * p[3], w * p[2] + z * p[4])
        } else if flag == zero {
            (w + z * p[3], w * p[2] + z)
        } else {
            (w * p[1] + z, -w + p[4] * z)
        };
        x[kx as usize] = xn;
        y[ky as usize] = yn;
        kx += incx;
        ky += incy;
    }
}

fn check_rotm<T: TestFloat>(func: RotmFn<T>) {
    let nan = T::nan();
    let f = T::from_f64;
    // Unused entries hold NaN: they must never be read.
    let params: [[T; 5]; 6] = [
        [f(-1.0), f(2.0), f(-1.0), f(0.5), f(3.0)],
        [f(0.0), nan, f(-0.25), f(0.75), nan],
        [f(1.0), f(1.5), nan, nan, f(-0.5)],
        [f(-2.0), nan, nan, nan, nan],
        // Non-canonical flags are classified like the reference.
        [f(-0.5), f(2.0), f(-1.0), f(0.5), f(3.0)],
        [f(0.5), f(1.5), nan, nan, f(-0.5)],
    ];
    let guard = f(-777.0);
    for p in &params {
        for n in [1usize, 4] {
            for incx in INCS {
                for incy in INCS {
                    let mut x = buffer::<T>(extent(n, incx), 3, guard);
                    let mut y = buffer::<T>(extent(n, incy), 4, guard);
                    let (mut xr, mut yr) = (x.clone(), y.clone());
                    ref_rotm(n, &mut xr, incx, &mut yr, incy, p);
                    // SAFETY: x / y span `extent(n, inc)` elements; p holds 5.
                    unsafe {
                        func(
                            n as i32,
                            x.as_mut_ptr(),
                            incx as i32,
                            y.as_mut_ptr(),
                            incy as i32,
                            p.as_ptr(),
                        );
                    }
                    let what = format!("rotm p={p:?} n={n} incx={incx} incy={incy}");
                    assert_eq!(bits_of(&x), bits_of(&xr), "{what} (x)");
                    assert_eq!(bits_of(&y), bits_of(&yr), "{what} (y)");
                    assert!(x.iter().chain(y.iter()).all(|v| v.is_finite()), "{what}");
                }
            }
        }
    }
}

#[test]
fn test_issue_14_srotm_drotm_every_flag_and_increment_match_reference_loop() {
    check_rotm::<f32>(cblas_srotm);
    check_rotm::<f64>(cblas_drotm);
}

#[test]
fn test_issue_14_rotm_identity_flag_reads_only_param_zero() {
    // flag = -2: only P[0] may be read, so a 1-element P is enough.
    let p = [-2.0f64];
    let mut x = [1.0f64, 2.0];
    let mut y = [3.0f64, 4.0];
    // SAFETY: x, y hold 2 elements; with flag -2 only p[0] is read.
    unsafe { cblas_drotm(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, p.as_ptr()) };
    assert_eq!((x, y), ([1.0, 2.0], [3.0, 4.0]));
    // Null P / n <= 0: no-op.
    // SAFETY: nothing is dereferenced.
    unsafe {
        cblas_drotm(2, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, std::ptr::null());
        cblas_srotm(
            0,
            std::ptr::null_mut(),
            1,
            std::ptr::null_mut(),
            1,
            std::ptr::null(),
        );
    }
    assert_eq!((x, y), ([1.0, 2.0], [3.0, 4.0]));
}

// =============================================================================
// ROTG
// =============================================================================

type RotgFn<T> = unsafe extern "C" fn(*mut T, *mut T, *mut T, *mut T);

fn call_rotg<T: TestFloat>(func: RotgFn<T>, a: T, b: T) -> (T, T, T, T) {
    let (mut a, mut b, mut c, mut s) = (a, b, T::from_f64(9.0), T::from_f64(9.0));
    // SAFETY: four distinct, valid scalars.
    unsafe { func(&mut a, &mut b, &mut c, &mut s) };
    (a, b, c, s)
}

/// Transcription of the reference `drotg.f90` / `srotg.f90` (3.12), returning
/// `(r, z, c, s)`.
fn ref_rotg<T: TestFloat>(a: T, b: T) -> (T, T, T, T) {
    let (zero, one) = (T::zero(), T::one());
    let safmin = T::min_positive_value();
    let safmax = (one + one) / safmin;
    let (anorm, bnorm) = (a.abs(), b.abs());
    if bnorm == zero {
        (a, zero, one, zero)
    } else if anorm == zero {
        (b, one, zero, one)
    } else {
        let scl = safmax.min(safmin.max(anorm).max(bnorm));
        let sigma = if anorm > bnorm {
            one.copysign(a)
        } else {
            one.copysign(b)
        };
        let (sa, sb) = (a / scl, b / scl);
        let r = sigma * (scl * (sa * sa + sb * sb).sqrt());
        let (c, s) = (a / r, b / r);
        let z = if anorm > bnorm {
            s
        } else if c != zero {
            one / c
        } else {
            one
        };
        (r, z, c, s)
    }
}

fn check_rotg<T: TestFloat>(func: RotgFn<T>, big: f64) {
    let f = T::from_f64;
    let cases: Vec<(T, T)> = vec![
        (f(3.0), f(4.0)),
        (f(-3.0), f(4.0)),
        (f(4.0), f(-3.0)),
        (f(-1.25), f(-0.5)),
        (f(1e-3), f(7.0)),
        (f(2.0), f(2.0)),
        (f(0.0), f(-2.0)),
        (f(-0.0), f(5.0)),
        (f(-3.0), f(0.0)),
        (f(3.0), f(-0.0)),
        (f(0.0), f(0.0)),
        // Huge finite inputs: |a| + |b| overflows, max(|a|, |b|) does not.
        (f(big), f(big)),
        (f(-big), f(0.5 * big)),
        // Subnormal inputs.
        (
            T::min_positive_value() * f(0.25),
            T::min_positive_value() * f(0.5),
        ),
    ];
    for (a, b) in cases {
        let got = call_rotg(func, a, b);
        let want = ref_rotg(a, b);
        assert_eq!(
            [got.0.bits(), got.1.bits(), got.2.bits(), got.3.bits()],
            [want.0.bits(), want.1.bits(), want.2.bits(), want.3.bits()],
            "rotg({a:?}, {b:?}): got (r, z, c, s) = {got:?}, expected {want:?}"
        );
        // The rotation is genuine: finite, orthonormal and zeroing b.
        let (r, _, c, s) = got;
        assert!(
            r.is_finite() && c.is_finite() && s.is_finite(),
            "rotg({a:?}, {b:?})"
        );
        let tol = f(16.0) * T::epsilon();
        assert!((c * c + s * s - T::one()).abs() <= tol);
        let scale = a.abs().max(b.abs()).max(T::min_positive_value());
        assert!(((c * a + s * b) - r).abs() / scale <= tol);
        assert!((c * b - s * a).abs() / scale <= tol);
    }
}

#[test]
fn test_issue_14_srotg_drotg_match_reference_rotg_f90() {
    check_rotg::<f32>(cblas_srotg, 2e38);
    check_rotg::<f64>(cblas_drotg, 1e308);
}

#[test]
fn test_issue_14_rotg_special_cases_use_reference_conventions() {
    // b == 0: c = 1, s = +0, z = +0, r = a (even for a negative a).
    let (r, z, c, s) = call_rotg::<f64>(cblas_drotg, -3.0, 0.0);
    assert_eq!((r, z.to_bits(), c, s.to_bits()), (-3.0, 0, 1.0, 0));
    // a == 0, b != 0: c = +0, s = 1, r = b, z = 1 (even for a negative b).
    let (r, z, c, s) = call_rotg::<f32>(cblas_srotg, 0.0, -2.0);
    assert_eq!((r, z, c.to_bits(), s), (-2.0, 1.0, 0, 1.0));
    // Overflow-free: the reference keeps r finite where |a| + |b| overflows.
    let (r, z, c, s) = call_rotg::<f64>(cblas_drotg, 1e308, 1e308);
    assert!(
        (r - std::f64::consts::SQRT_2 * 1e308).abs() <= 1e293,
        "r = {r}"
    );
    assert!((c - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15 && c == s);
    assert!((z - std::f64::consts::SQRT_2).abs() < 1e-15);
    // Null operands: no-op.
    let mut v = [1.0f64, 2.0, 3.0];
    let base = v.as_mut_ptr();
    // SAFETY: a null operand makes the call a no-op (nothing is dereferenced).
    unsafe { cblas_drotg(base, base.add(1), base.add(2), std::ptr::null_mut()) };
    assert_eq!(v, [1.0, 2.0, 3.0]);
}

// =============================================================================
// ROTMG
// =============================================================================

type RotmgFn<T> = unsafe extern "C" fn(*mut T, *mut T, *mut T, T, *mut T);

/// Runs `?ROTMG` with `P` prefilled with `sentinel`; returns `(d1, d2, x1, P)`.
fn call_rotmg<T: TestFloat>(
    func: RotmgFn<T>,
    d1: T,
    d2: T,
    x1: T,
    y1: T,
    sentinel: T,
) -> (T, T, T, [T; 5]) {
    let (mut d1, mut d2, mut x1) = (d1, d2, x1);
    let mut p = [sentinel; 5];
    // SAFETY: three distinct scalars and a 5-element PARAM array.
    unsafe { func(&mut d1, &mut d2, &mut x1, y1, p.as_mut_ptr()) };
    (d1, d2, x1, p)
}

/// The full `H` encoded by `P` (implied entries substituted).
fn dense_h<T: TestFloat>(p: &[T; 5]) -> [[T; 2]; 2] {
    let (zero, one) = (T::zero(), T::one());
    if p[0] < zero {
        [[p[1], p[3]], [p[2], p[4]]]
    } else if p[0] == zero {
        [[one, p[3]], [p[2], one]]
    } else {
        [[p[1], one], [-one, p[4]]]
    }
}

/// Defining properties of a modified Givens rotation:
/// `H·[x1; y1] = [x1'; 0]` and `H^T·diag(d1', d2')·H = diag(d1, d2)`.
fn check_rotmg_properties<T: TestFloat>(inp: (T, T, T, T), out: (T, T, T, [T; 5]), tol: f64) {
    let (d1, d2, x1, y1) = inp;
    let (d1n, d2n, x1n, p) = out;
    let h = dense_h(&p);
    let tol = T::from_f64(tol);
    let rel = |got: T, want: T, scale: T| (got - want).abs() <= tol * scale.max(T::one());
    let first = h[0][0] * x1 + h[0][1] * y1;
    let second = h[1][0] * x1 + h[1][1] * y1;
    let mag = x1.abs().max(y1.abs());
    assert!(
        rel(first, x1n, x1n.abs().max(mag)),
        "x1' = {x1n:?}, H row 0 gives {first:?}"
    );
    assert!(rel(second, T::zero(), mag), "H row 1 gives {second:?}");
    // H^T D' H entries.
    let m00 = h[0][0] * h[0][0] * d1n + h[1][0] * h[1][0] * d2n;
    let m01 = h[0][0] * h[0][1] * d1n + h[1][0] * h[1][1] * d2n;
    let m11 = h[0][1] * h[0][1] * d1n + h[1][1] * h[1][1] * d2n;
    let dmag = d1.abs().max(d2.abs());
    assert!(rel(m00, d1, dmag) && rel(m01, T::zero(), dmag) && rel(m11, d2, dmag));
}

fn check_rotmg_masks<T: TestFloat>(func: RotmgFn<T>, tol: f64) {
    let f = T::from_f64;
    let sentinel = f(-1234.5);
    let s = sentinel.bits();

    // flag = 1 (|d1 x1^2| <= |d2 y1^2|): writes P[0], P[1], P[4] only.
    let inp = (f(1.0), f(2.0), f(1.0), f(2.0));
    let out = call_rotmg(func, inp.0, inp.1, inp.2, inp.3, sentinel);
    let p = out.3;
    assert_eq!(p[0], f(1.0));
    assert_eq!([p[2].bits(), p[3].bits()], [s, s], "flag 1 wrote P[2]/P[3]");
    assert_eq!((p[1], p[4]), (f(0.25), f(0.5)));
    check_rotmg_properties(inp, out, tol);

    // flag = 0 (|d1 x1^2| > |d2 y1^2|): writes P[0], P[2], P[3] only.
    let inp = (f(2.0), f(1.0), f(2.0), f(1.0));
    let out = call_rotmg(func, inp.0, inp.1, inp.2, inp.3, sentinel);
    let p = out.3;
    assert_eq!(p[0], f(0.0));
    assert_eq!([p[1].bits(), p[4].bits()], [s, s], "flag 0 wrote P[1]/P[4]");
    assert_eq!((p[2], p[3]), (f(-0.5), f(0.25)));
    check_rotmg_properties(inp, out, tol);

    // flag = -1 via rescaling (tiny d1): writes all of P.
    let inp = (f(1e-9), f(1.0), f(1.0), f(1e-6));
    let out = call_rotmg(func, inp.0, inp.1, inp.2, inp.3, sentinel);
    assert_eq!(out.3[0], f(-1.0));
    assert!(
        out.3.iter().all(|v| v.bits() != s),
        "flag -1 left a slot unwritten"
    );
    check_rotmg_properties(inp, out, tol);

    // flag = -1 via d1 < 0: H = 0 and d1 = d2 = x1 = 0.
    let (d1, d2, x1, p) = call_rotmg(func, f(-1.0), f(2.0), f(3.0), f(4.0), sentinel);
    assert_eq!(p, [f(-1.0), f(0.0), f(0.0), f(0.0), f(0.0)]);
    assert_eq!((d1, d2, x1), (f(0.0), f(0.0), f(0.0)));

    // flag = -2 (d2·y1 == 0): writes P[0] only; d1, d2, x1 unchanged.
    let (d1, d2, x1, p) = call_rotmg(func, f(1.5), f(2.0), f(3.0), f(0.0), sentinel);
    assert_eq!(p[0], f(-2.0));
    assert!(p[1..].iter().all(|v| v.bits() == s), "flag -2 wrote P[1..]");
    assert_eq!((d1, d2, x1), (f(1.5), f(2.0), f(3.0)));

    // Null operands: no-op.
    let mut p = [sentinel; 5];
    let (mut d1, mut d2) = (f(1.0), f(1.0));
    // SAFETY: a null operand makes the call a no-op.
    unsafe {
        func(
            &mut d1,
            &mut d2,
            std::ptr::null_mut(),
            f(1.0),
            p.as_mut_ptr(),
        )
    };
    assert!(p.iter().all(|v| v.bits() == s) && d1 == f(1.0) && d2 == f(1.0));
}

#[test]
fn test_issue_14_srotmg_drotmg_write_only_flag_entries() {
    check_rotmg_masks::<f32>(cblas_srotmg, 1e-5);
    check_rotmg_masks::<f64>(cblas_drotmg, 1e-13);
}

/// The reference range tests use the `DATA` constants verbatim — double:
/// `GAMSQ = 16777216.D0`, `RGAMSQ = 5.9604645D-8` (slightly above `2^-24`);
/// single: `GAMSQ = 1.67772E7` (below `2^24`), `RGAMSQ = 5.96046E-8` (below
/// `2^-24`) — while every rescaling multiplies by the exact `GAM**2 = 2^24`.
/// With `x1 = d2 = 1` and a tiny `y1`, the flag-0 update leaves `d1` exactly
/// unchanged (`u` rounds to 1), so only the range test decides the flag.
#[test]
fn test_issue_14_rotmg_range_tests_use_reference_data_constants() {
    let two24 = 16_777_216.0f64;

    // f64: 2^-24 < d1 <= 5.9604645e-8 -> rescaled (flag -1, d1·2^24).
    let d1 = 5.960_464_48e-8_f64;
    assert!(d1 > 1.0 / two24 && d1 <= 5.960_464_5e-8);
    let (d1n, _, x1n, p) = call_rotmg::<f64>(cblas_drotmg, d1, 1.0, 1.0, 1e-20, 7.0);
    assert_eq!(
        p[0], -1.0,
        "d1 = {d1:e} must be rescaled (RGAMSQ = 5.9604645D-8)"
    );
    assert_eq!((d1n, x1n, p[1]), (d1 * two24, 1.0 / 4096.0, 1.0 / 4096.0));

    // f32: 16777200 <= d1 < 2^24 -> rescaled down (flag -1, d1 / 2^24).
    let d1 = 16_777_208.0f32;
    let (d1n, _, x1n, p) = call_rotmg::<f32>(cblas_srotmg, d1, 1.0, 1.0, 1e-10, 7.0);
    assert_eq!(p[0], -1.0, "d1 = {d1} must be rescaled (GAMSQ = 1.67772E7)");
    assert_eq!((d1n, x1n, p[1]), (d1 / two24 as f32, 4096.0, 4096.0));

    // f32: d1 = 2^-24 lies above RGAMSQ = 5.96046E-8 -> no rescaling (flag 0).
    let d1 = 1.0 / two24 as f32;
    assert!(d1 > 5.960_46e-8_f32);
    let (d1n, _, _, p) = call_rotmg::<f32>(cblas_srotmg, d1, 1.0, 1.0, 1e-10, 7.0);
    assert_eq!(
        p[0], 0.0,
        "d1 = 2^-24 must not be rescaled (RGAMSQ = 5.96046E-8)"
    );
    assert_eq!(d1n, d1);
}

// =============================================================================
// CROTG / ZROTG
// =============================================================================

#[test]
fn test_issue_14_crotg_zrotg_conventions_and_rotation_property() {
    let z = |re: f64, im: f64| Complex64::new(re, im);
    // a = 0, b = 0: c = 1, s = 0, r = 0.
    let mut a = z(0.0, 0.0);
    let (mut c, mut s) = (9.0f64, z(9.0, 9.0));
    // SAFETY: valid scalars.
    unsafe { cblas_zrotg(&mut a, &z(0.0, 0.0), &mut c, &mut s) };
    assert_eq!((a, c, s), (z(0.0, 0.0), 1.0, z(0.0, 0.0)));
    // a = 0, b = 3 + 4i: c = 0, r = 5, s = 0.6 - 0.8i.
    let mut a = z(0.0, 0.0);
    // SAFETY: valid scalars.
    unsafe { cblas_zrotg(&mut a, &z(3.0, 4.0), &mut c, &mut s) };
    assert_eq!(c, 0.0);
    assert!((a - z(5.0, 0.0)).norm() < 1e-15 && (s - z(0.6, -0.8)).norm() < 1e-15);
    // General case: [c s; -conj(s) c] [a; b] = [r; 0], c^2 + |s|^2 = 1, b unchanged.
    for (a0, b0) in [(z(1.0, -2.0), z(0.5, 3.0)), (z(-4.0, 0.25), z(2.0, -2.0))] {
        let (mut a, b) = (a0, b0);
        // SAFETY: valid scalars; `b` is only read.
        unsafe { cblas_zrotg(&mut a, &b, &mut c, &mut s) };
        assert_eq!(b, b0);
        assert!((c * a0 + s * b0 - a).norm() < 1e-14 * (1.0 + a.norm()));
        assert!((-s.conj() * a0 + c * b0).norm() < 1e-14 * (1.0 + a.norm()));
        assert!((c * c + s.norm_sqr() - 1.0).abs() < 1e-14);
    }
    // Single precision: same conventions.
    let mut a = Complex32::new(0.0, 0.0);
    let (mut c, mut s) = (9.0f32, Complex32::new(9.0, 9.0));
    // SAFETY: valid scalars.
    unsafe { cblas_crotg(&mut a, &Complex32::new(3.0, 4.0), &mut c, &mut s) };
    assert_eq!(c, 0.0);
    assert!((a - Complex32::new(5.0, 0.0)).norm() < 1e-6);
    assert!((s - Complex32::new(0.6, -0.8)).norm() < 1e-6);
    // Null operand: no-op.
    let mut a = Complex32::new(1.0, 1.0);
    // SAFETY: a null operand makes the call a no-op.
    unsafe { cblas_crotg(&mut a, std::ptr::null(), &mut c, &mut s) };
    assert_eq!(a, Complex32::new(1.0, 1.0));
}

// =============================================================================
// CSROT / ZDROT
// =============================================================================

/// Netlib `ZDROT` loop with real `c`/`s` scaling the components separately.
fn ref_zdrot(
    n: usize,
    x: &mut [Complex64],
    incx: isize,
    y: &mut [Complex64],
    incy: isize,
    c: f64,
    s: f64,
) {
    let (mut ix, mut iy) = (start(n, incx), start(n, incy));
    for _ in 0..n {
        let (xi, yi) = (x[ix as usize], y[iy as usize]);
        let temp = Complex64::new(c * xi.re + s * yi.re, c * xi.im + s * yi.im);
        y[iy as usize] = Complex64::new(c * yi.re - s * xi.re, c * yi.im - s * xi.im);
        x[ix as usize] = temp;
        ix += incx;
        iy += incy;
    }
}

fn cbuffer(len: usize, seed: usize) -> Vec<Complex64> {
    let mut v: Vec<Complex64> = (0..len)
        .map(|i| {
            Complex64::new(
                ((5 * i + seed) % 9) as f64 * 0.5 - 2.0,
                ((3 * i + 2 * seed) % 7) as f64 * 0.25 - 0.75,
            )
        })
        .collect();
    v.push(Complex64::new(-777.0, 777.0));
    v
}

#[test]
fn test_issue_14_csrot_zdrot_every_increment_matches_reference_loop() {
    let (c, s) = (0.8f64, 0.6f64);
    let cbits = |v: &[Complex64]| -> Vec<(u64, u64)> {
        v.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
    };
    for n in [1usize, 3, 4] {
        for incx in INCS {
            for incy in INCS {
                let mut x = cbuffer(extent(n, incx), 1);
                let mut y = cbuffer(extent(n, incy), 2);
                let (mut xr, mut yr) = (x.clone(), y.clone());
                ref_zdrot(n, &mut xr, incx, &mut yr, incy, c, s);
                // Single precision copies (all values are exact in f32).
                let to32 = |v: &[Complex64]| -> Vec<Complex32> {
                    v.iter()
                        .map(|z| Complex32::new(z.re as f32, z.im as f32))
                        .collect()
                };
                let (mut x32, mut y32) = (to32(&x), to32(&y));
                // SAFETY: x / y span `extent(n, inc)` elements from their base.
                unsafe {
                    cblas_zdrot(
                        n as i32,
                        x.as_mut_ptr(),
                        incx as i32,
                        y.as_mut_ptr(),
                        incy as i32,
                        c,
                        s,
                    );
                    cblas_csrot(
                        n as i32,
                        x32.as_mut_ptr(),
                        incx as i32,
                        y32.as_mut_ptr(),
                        incy as i32,
                        c as f32,
                        s as f32,
                    );
                }
                let what = format!("zdrot n={n} incx={incx} incy={incy}");
                assert_eq!(cbits(&x), cbits(&xr), "{what} (x)");
                assert_eq!(cbits(&y), cbits(&yr), "{what} (y)");
                for (got, want) in x32.iter().zip(&xr).chain(y32.iter().zip(&yr)) {
                    let got = Complex64::new(f64::from(got.re), f64::from(got.im));
                    assert!(
                        (got - want).norm() <= 1e-5 * (1.0 + want.norm()),
                        "{what} (c)"
                    );
                }
            }
        }
    }
    // n <= 0 / null: no-op.
    let mut x = [Complex32::new(1.0, 2.0)];
    let mut y = [Complex32::new(3.0, 4.0)];
    // SAFETY: nothing is dereferenced (n <= 0 or a null operand).
    unsafe {
        cblas_csrot(0, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 0.0, 1.0);
        cblas_zdrot(
            -1,
            std::ptr::null_mut(),
            1,
            std::ptr::null_mut(),
            1,
            0.0,
            1.0,
        );
        cblas_csrot(1, x.as_mut_ptr(), 1, std::ptr::null_mut(), 1, 0.0, 1.0);
    }
    assert_eq!(
        (x, y),
        ([Complex32::new(1.0, 2.0)], [Complex32::new(3.0, 4.0)])
    );
}

// =============================================================================
// SDSDOT / DSDOT
// =============================================================================

/// Netlib `SDSDOT`/`DSDOT` loop: double accumulator starting at `init`.
fn ref_dsdot(init: f64, n: usize, x: &[f32], incx: isize, y: &[f32], incy: isize) -> f64 {
    let (mut kx, mut ky) = (start(n, incx), start(n, incy));
    let mut acc = init;
    for _ in 0..n {
        acc += f64::from(x[kx as usize]) * f64::from(y[ky as usize]);
        kx += incx;
        ky += incy;
    }
    acc
}

#[test]
fn test_issue_14_sdsdot_dsdot_every_increment_matches_reference_loop() {
    let alpha = 0.375f32;
    for n in [1usize, 2, 5] {
        for incx in INCS {
            for incy in INCS {
                // Values whose products need more than f32 precision.
                let x: Vec<f32> = (0..extent(n, incx))
                    .map(|i| 1.0 + (i as f32) * 1.0e-3 + 3.0e-7)
                    .collect();
                let y: Vec<f32> = (0..extent(n, incy))
                    .map(|i| 1.0 - (i as f32) * 7.0e-4 - 1.0e-7)
                    .collect();
                let want_d = ref_dsdot(0.0, n, &x, incx, &y, incy);
                let want_s = ref_dsdot(f64::from(alpha), n, &x, incx, &y, incy) as f32;
                // SAFETY: x / y span `extent(n, inc)` elements from their base.
                let (got_d, got_s) = unsafe {
                    (
                        cblas_dsdot(n as i32, x.as_ptr(), incx as i32, y.as_ptr(), incy as i32),
                        cblas_sdsdot(
                            n as i32,
                            alpha,
                            x.as_ptr(),
                            incx as i32,
                            y.as_ptr(),
                            incy as i32,
                        ),
                    )
                };
                assert_eq!(
                    got_d.to_bits(),
                    want_d.to_bits(),
                    "dsdot n={n} {incx} {incy}"
                );
                assert_eq!(
                    got_s.to_bits(),
                    want_s.to_bits(),
                    "sdsdot n={n} {incx} {incy}"
                );
            }
        }
    }
}

#[test]
fn test_issue_14_sdsdot_accumulates_alpha_and_products_in_double() {
    // In f32: 0.5 + 2^24 rounds to 2^24, (+1) rounds back to 2^24, (-2^24) -> 0.
    // In double (reference): 0.5 + 2^24 + 1 - 2^24 = 1.5 exactly.
    let x = [16_777_216.0f32, 1.0, -16_777_216.0];
    let y = [1.0f32; 3];
    // SAFETY: both vectors hold 3 elements; strided calls span 3 elements too.
    unsafe {
        assert_eq!(cblas_sdsdot(3, 0.5, x.as_ptr(), 1, y.as_ptr(), 1), 1.5);
        assert_eq!(cblas_sdsdot(3, 0.5, x.as_ptr(), -1, y.as_ptr(), -1), 1.5);
        assert_eq!(cblas_dsdot(3, x.as_ptr(), 1, y.as_ptr(), 0), 1.0);
        // n <= 0 and null operands return the identity.
        assert_eq!(cblas_sdsdot(0, 2.5, x.as_ptr(), 1, y.as_ptr(), 1), 2.5);
        assert_eq!(cblas_sdsdot(-1, 2.5, x.as_ptr(), 1, y.as_ptr(), 1), 2.5);
        assert_eq!(
            cblas_sdsdot(3, 2.5, std::ptr::null(), 1, y.as_ptr(), 1),
            2.5
        );
        assert_eq!(cblas_dsdot(0, x.as_ptr(), 1, y.as_ptr(), 1), 0.0);
        assert_eq!(cblas_dsdot(3, x.as_ptr(), 1, std::ptr::null(), 1), 0.0);
    }
}

// =============================================================================
// SCABS1 / DCABS1
// =============================================================================

#[test]
fn test_issue_14_scabs1_dcabs1_values_and_null() {
    let cases = [
        (3.0, -4.0, 7.0),
        (-0.0, 0.0, 0.0),
        (-1.5, -2.25, 3.75),
        (1e300, 0.0, 1e300),
    ];
    for (re, im, want) in cases {
        // SAFETY: valid scalars.
        unsafe {
            assert_eq!(cblas_dcabs1(&Complex64::new(re, im)), want);
            assert_eq!(
                cblas_scabs1(&Complex32::new(re as f32, im as f32)),
                (re as f32).abs() + (im as f32).abs()
            );
        }
    }
    // SAFETY: null is documented to return 0.
    unsafe {
        assert_eq!(cblas_scabs1(std::ptr::null()), 0.0);
        assert_eq!(cblas_dcabs1(std::ptr::null()), 0.0);
    }
}

// =============================================================================
// Unsatisfiable extents
// =============================================================================

/// `n = |inc| ≈ 2^31` gives the extent `(n - 1)·|inc| + 1 ≈ 2^62` elements. That
/// overflows a 32-bit `usize`; on a 64-bit target it fits in `usize`, but at 4, 8
/// or 16 bytes per element it spans more than `isize::MAX` bytes, which no
/// allocation can and `slice::from_raw_parts` forbids. Either way
/// `StridedSlice[Mut]::from_raw_parts` must return an empty view without touching
/// the (dangling) pointer, and every wrapper must then return its identity instead
/// of panicking or building an invalid slice. Runs on 32- and 64-bit targets.
#[test]
fn test_issue_14_wrappers_unsatisfiable_extent_is_noop() {
    use crate::cblas::{cblas_dznrm2, cblas_scnrm2};
    use std::ptr::NonNull;

    const N: i32 = i32::MAX;
    let dangling_f32 = NonNull::<f32>::dangling().as_ptr();
    let dangling_f64 = NonNull::<f64>::dangling().as_ptr();
    let dangling_c32 = NonNull::<Complex32>::dangling().as_ptr();
    let dangling_c64 = NonNull::<Complex64>::dangling().as_ptr();
    let p = [-1.0f64, 0.5, 0.25, -0.25, 0.5];
    let ps = p.map(|v| v as f32);

    for inc in [i32::MAX, i32::MIN] {
        // SAFETY: the dangling operand has an unsatisfiable extent, so every
        // wrapper must return before it reads or writes through that pointer; the
        // other operand spans its single element (`inc == 0`).
        unsafe {
            // ROT, with the dangling operand in either position.
            let mut v = [2.5f64];
            cblas_drot(N, dangling_f64, inc, v.as_mut_ptr(), 0, 0.6, -0.8);
            cblas_drot(N, v.as_mut_ptr(), 0, dangling_f64, inc, 0.6, -0.8);
            assert_eq!(v, [2.5], "drot inc={inc}");
            let mut v = [2.5f32];
            cblas_srot(N, dangling_f32, inc, v.as_mut_ptr(), 0, 0.6, -0.8);
            cblas_srot(N, v.as_mut_ptr(), 0, dangling_f32, inc, 0.6, -0.8);
            assert_eq!(v, [2.5], "srot inc={inc}");

            // ROTM with a non-identity flag (-2 would return before any view).
            let mut v = [2.5f64];
            cblas_drotm(N, dangling_f64, inc, v.as_mut_ptr(), 0, p.as_ptr());
            cblas_drotm(N, v.as_mut_ptr(), 0, dangling_f64, inc, p.as_ptr());
            assert_eq!(v, [2.5], "drotm inc={inc}");
            let mut v = [2.5f32];
            cblas_srotm(N, dangling_f32, inc, v.as_mut_ptr(), 0, ps.as_ptr());
            cblas_srotm(N, v.as_mut_ptr(), 0, dangling_f32, inc, ps.as_ptr());
            assert_eq!(v, [2.5], "srotm inc={inc}");

            // CSROT / ZDROT.
            let mut v = [Complex32::new(2.5, -1.5)];
            cblas_csrot(N, dangling_c32, inc, v.as_mut_ptr(), 0, 0.6, -0.8);
            cblas_csrot(N, v.as_mut_ptr(), 0, dangling_c32, inc, 0.6, -0.8);
            assert_eq!(v, [Complex32::new(2.5, -1.5)], "csrot inc={inc}");
            let mut v = [Complex64::new(2.5, -1.5)];
            cblas_zdrot(N, dangling_c64, inc, v.as_mut_ptr(), 0, 0.6, -0.8);
            cblas_zdrot(N, v.as_mut_ptr(), 0, dangling_c64, inc, 0.6, -0.8);
            assert_eq!(v, [Complex64::new(2.5, -1.5)], "zdrot inc={inc}");

            // SDSDOT / DSDOT return their identities (`alpha` / `0`).
            let v = [3.0f32];
            let sdsdot_x = cblas_sdsdot(N, 0.5, dangling_f32, inc, v.as_ptr(), 0);
            let sdsdot_y = cblas_sdsdot(N, 0.5, v.as_ptr(), 0, dangling_f32, inc);
            assert_eq!((sdsdot_x, sdsdot_y), (0.5, 0.5), "sdsdot inc={inc}");
            let dsdot_x = cblas_dsdot(N, dangling_f32, inc, v.as_ptr(), 0);
            let dsdot_y = cblas_dsdot(N, v.as_ptr(), 0, dangling_f32, inc);
            assert_eq!((dsdot_x, dsdot_y), (0.0, 0.0), "dsdot inc={inc}");

            // SCNRM2 / DZNRM2 see an empty vector, whose norm is 0.
            assert_eq!(cblas_scnrm2(N, dangling_c32, inc), 0.0, "scnrm2 inc={inc}");
            assert_eq!(cblas_dznrm2(N, dangling_c64, inc), 0.0, "dznrm2 inc={inc}");
        }
    }
}

/// 32-bit targets: `n = 2^20` with `incx = 2^13` has the extent
/// `(n - 1)·incx + 1 ≈ 2^33`, which overflows a 32-bit `usize`, and with
/// `incx = 2^11` the extent `≈ 2^31` fits in `usize` but its 4- to 16-byte
/// elements span more than `isize::MAX` bytes. Either way
/// `StridedSlice[Mut]::from_raw_parts` degrades that view to an empty one. An
/// increment of 0 keeps the other view at the full logical length `n` over a
/// single element. Without the length guard in the wrappers the natives'
/// equal-length assert (or `strided_dot_f64` indexing the empty view) would
/// panic, which aborts a C caller. The unit-stride SDSDOT/DSDOT branch with
/// `n = 2^29` f32s (`2^31` bytes) must return the identity as well.
#[cfg(target_pointer_width = "32")]
#[test]
fn test_issue_14_paired_wrappers_overflowing_extent_is_noop_on_32_bit() {
    use crate::cblas::cblas_dznrm2;
    use std::ptr::NonNull;

    const N: i32 = 1 << 20;
    const INCX: i32 = 1 << 13;
    let dangling_f32 = NonNull::<f32>::dangling().as_ptr();
    let dangling_f64 = NonNull::<f64>::dangling().as_ptr();
    let dangling_c32 = NonNull::<Complex32>::dangling().as_ptr();
    let dangling_c64 = NonNull::<Complex64>::dangling().as_ptr();
    let p = [-1.0f64, 0.5, 0.25, -0.25, 0.5];

    // SAFETY: the dangling operand has an unsatisfiable extent, so every wrapper
    // must return before it reads or writes through that pointer; the other
    // operand spans its single element (`inc == 0`).
    unsafe {
        // ROT: x overflows, then y overflows.
        let mut y = [2.5f64];
        cblas_drot(N, dangling_f64, INCX, y.as_mut_ptr(), 0, 0.6, -0.8);
        assert_eq!(y, [2.5]);
        let mut x = [2.5f64];
        cblas_drot(N, x.as_mut_ptr(), 0, dangling_f64, INCX, 0.6, -0.8);
        assert_eq!(x, [2.5]);

        // ROTM with a non-identity flag (-2 would return before any view).
        let mut y = [2.5f64];
        cblas_drotm(N, dangling_f64, INCX, y.as_mut_ptr(), 0, p.as_ptr());
        assert_eq!(y, [2.5]);

        // CSROT / ZDROT.
        let mut yc = [Complex32::new(2.5, -1.5)];
        cblas_csrot(N, dangling_c32, INCX, yc.as_mut_ptr(), 0, 0.6, -0.8);
        assert_eq!(yc, [Complex32::new(2.5, -1.5)]);
        let mut yz = [Complex64::new(2.5, -1.5)];
        cblas_zdrot(N, dangling_c64, INCX, yz.as_mut_ptr(), 0, 0.6, -0.8);
        assert_eq!(yz, [Complex64::new(2.5, -1.5)]);

        // SDSDOT / DSDOT return their identities (`alpha` / `0`).
        let ys = [3.0f32];
        assert_eq!(
            cblas_sdsdot(N, 0.5, dangling_f32, INCX, ys.as_ptr(), 0),
            0.5
        );
        assert_eq!(
            cblas_sdsdot(N, 0.5, ys.as_ptr(), 0, dangling_f32, INCX),
            0.5
        );
        assert_eq!(cblas_dsdot(N, dangling_f32, INCX, ys.as_ptr(), 0), 0.0);
        assert_eq!(cblas_dsdot(N, ys.as_ptr(), 0, dangling_f32, INCX), 0.0);
    }

    // `incx = 2^11`: the extent `(2^20 - 1)·2^11 + 1 ≈ 2^31` elements fits in a
    // 32-bit `usize` but spans `≈ 2^33` (f32) to `≈ 2^35` (Complex64) bytes.
    const INCX_BYTES: i32 = 1 << 11;
    // The unit-stride SDSDOT/DSDOT branch: `2^29` f32s span `2^31` bytes.
    const N_UNIT: i32 = 1 << 29;
    // SAFETY: as above -- the dangling operand's extent exceeds `isize::MAX`
    // bytes, so no wrapper may read or write through it; the other operand spans
    // its single element (`inc == 0`).
    unsafe {
        let mut v = [2.5f64];
        cblas_drot(N, dangling_f64, INCX_BYTES, v.as_mut_ptr(), 0, 0.6, -0.8);
        cblas_drot(N, v.as_mut_ptr(), 0, dangling_f64, INCX_BYTES, 0.6, -0.8);
        assert_eq!(v, [2.5]);
        let mut v = [2.5f64];
        cblas_drotm(N, dangling_f64, INCX_BYTES, v.as_mut_ptr(), 0, p.as_ptr());
        assert_eq!(v, [2.5]);
        let mut v = [Complex32::new(2.5, -1.5)];
        cblas_csrot(N, dangling_c32, INCX_BYTES, v.as_mut_ptr(), 0, 0.6, -0.8);
        assert_eq!(v, [Complex32::new(2.5, -1.5)]);
        let mut v = [Complex64::new(2.5, -1.5)];
        cblas_zdrot(N, dangling_c64, INCX_BYTES, v.as_mut_ptr(), 0, 0.6, -0.8);
        assert_eq!(v, [Complex64::new(2.5, -1.5)]);
        let ys = [3.0f32];
        let sdsdot = cblas_sdsdot(N, 0.5, dangling_f32, INCX_BYTES, ys.as_ptr(), 0);
        assert_eq!(sdsdot, 0.5);
        assert_eq!(
            cblas_dsdot(N, ys.as_ptr(), 0, dangling_f32, INCX_BYTES),
            0.0
        );
        assert_eq!(cblas_dznrm2(N, dangling_c64, INCX_BYTES), 0.0);

        assert_eq!(
            cblas_sdsdot(N_UNIT, 0.5, dangling_f32, 1, dangling_f32, 1),
            0.5
        );
        assert_eq!(cblas_dsdot(N_UNIT, dangling_f32, 1, dangling_f32, 1), 0.0);
    }
}
