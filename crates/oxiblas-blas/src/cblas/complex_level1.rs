//! CBLAS-compatible complex Level-1 routines.
//!
//! Rounds out the complex vector operations: scaling (`cscal`/`zscal`,
//! `csscal`/`zdscal`), `caxpy`/`zaxpy`, `ccopy`/`zcopy`, `cswap`/`zswap`,
//! the real-valued reductions `scnrm2`/`dznrm2`, `scasum`/`dzasum`, and the
//! index reductions `icamax`/`izamax`.
//!
//! Unit-stride paths defer to the SIMD-accelerated native kernels
//! (`level1::scal`/`axpy`/`copy`/`swap`); strided paths use scalar loops.
//! Increments follow the reference BLAS convention, including negative
//! increments (see [`vec_offset`]).

use crate::level1;
use num_complex::{Complex, Complex32, Complex64};
use num_traits::Float;

/// Pointer offset (in elements) of logical index `i` for a length-`n` vector
/// with increment `inc`, honoring the reference BLAS negative-increment rule.
#[inline]
fn vec_offset(i: usize, n: usize, inc: isize) -> isize {
    if inc >= 0 {
        i as isize * inc
    } else {
        (i as isize - (n as isize - 1)) * inc
    }
}

// =============================================================================
// SCAL — x = alpha * x  (complex alpha)
// =============================================================================

/// Complex single precision scaling by a complex scalar.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cscal(
    n: i32,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    if n <= 0 {
        return;
    }
    scal_c(n as usize, *alpha, x, incx as isize);
}

/// Complex double precision scaling by a complex scalar.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zscal(
    n: i32,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    if n <= 0 {
        return;
    }
    scal_c(n as usize, *alpha, x, incx as isize);
}

unsafe fn scal_c<F: Float>(n: usize, alpha: Complex<F>, x: *mut Complex<F>, inc: isize) {
    // Generic over `F`, so we cannot name the `Complex<F>: Field` bound the SIMD
    // kernel needs; a scalar loop is used for both unit and strided access.
    for i in 0..n {
        let p = x.offset(vec_offset(i, n, inc));
        *p = *p * alpha;
    }
}

// =============================================================================
// (D)SCAL — x = alpha * x  (real alpha, complex x): csscal / zdscal
// =============================================================================

/// Complex single precision scaling by a real scalar.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_csscal(n: i32, alpha: f32, x: *mut Complex32, incx: i32) {
    if n <= 0 {
        return;
    }
    dscal_c(n as usize, alpha, x, incx as isize);
}

/// Complex double precision scaling by a real scalar.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zdscal(n: i32, alpha: f64, x: *mut Complex64, incx: i32) {
    if n <= 0 {
        return;
    }
    dscal_c(n as usize, alpha, x, incx as isize);
}

unsafe fn dscal_c<F: Float>(n: usize, alpha: F, x: *mut Complex<F>, inc: isize) {
    for i in 0..n {
        let p = x.offset(vec_offset(i, n, inc));
        *p = (*p).scale(alpha);
    }
}

// =============================================================================
// AXPY — y = alpha * x + y  (complex alpha)
// =============================================================================

/// Complex single precision AXPY.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_caxpy(
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    axpy_c(n as usize, *alpha, x, incx as isize, y, incy as isize);
}

/// Complex double precision AXPY.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zaxpy(
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    axpy_c(n as usize, *alpha, x, incx as isize, y, incy as isize);
}

unsafe fn axpy_c<F: Float>(
    n: usize,
    alpha: Complex<F>,
    x: *const Complex<F>,
    incx: isize,
    y: *mut Complex<F>,
    incy: isize,
) {
    // Scalar loop for both unit and strided access (see `scal_c`).
    for i in 0..n {
        let xi = *x.offset(vec_offset(i, n, incx));
        let yp = y.offset(vec_offset(i, n, incy));
        *yp = *yp + alpha * xi;
    }
}

// =============================================================================
// COPY — y = x
// =============================================================================

/// Complex single precision copy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ccopy(
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    copy_c(n as usize, x, incx as isize, y, incy as isize);
}

/// Complex double precision copy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zcopy(
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    copy_c(n as usize, x, incx as isize, y, incy as isize);
}

unsafe fn copy_c<T: Copy>(n: usize, x: *const T, incx: isize, y: *mut T, incy: isize) {
    if incx == 1 && incy == 1 {
        let xs = std::slice::from_raw_parts(x, n);
        let ys = std::slice::from_raw_parts_mut(y, n);
        level1::copy(xs, ys);
        return;
    }
    for i in 0..n {
        *y.offset(vec_offset(i, n, incy)) = *x.offset(vec_offset(i, n, incx));
    }
}

// =============================================================================
// SWAP — x <-> y
// =============================================================================

/// Complex single precision swap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cswap(
    n: i32,
    x: *mut Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    swap_c(n as usize, x, incx as isize, y, incy as isize);
}

/// Complex double precision swap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zswap(
    n: i32,
    x: *mut Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    swap_c(n as usize, x, incx as isize, y, incy as isize);
}

unsafe fn swap_c<T: Copy>(n: usize, x: *mut T, incx: isize, y: *mut T, incy: isize) {
    if incx == 1 && incy == 1 {
        let xs = std::slice::from_raw_parts_mut(x, n);
        let ys = std::slice::from_raw_parts_mut(y, n);
        level1::swap(xs, ys);
        return;
    }
    for i in 0..n {
        let xp = x.offset(vec_offset(i, n, incx));
        let yp = y.offset(vec_offset(i, n, incy));
        std::ptr::swap(xp, yp);
    }
}

// =============================================================================
// NRM2 — Euclidean norm of a complex vector (real result)
// =============================================================================

/// Complex single precision Euclidean norm (`scnrm2`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scnrm2(n: i32, x: *const Complex32, incx: i32) -> f32 {
    if n <= 0 || incx <= 0 {
        return 0.0;
    }
    nrm2_c(n as usize, x, incx as isize)
}

/// Complex double precision Euclidean norm (`dznrm2`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dznrm2(n: i32, x: *const Complex64, incx: i32) -> f64 {
    if n <= 0 || incx <= 0 {
        return 0.0;
    }
    nrm2_c(n as usize, x, incx as isize)
}

/// Classical LAPACK-style scaled accumulation, applied over the interleaved
/// real and imaginary components, matching Netlib `?znrm2` overflow behavior.
unsafe fn nrm2_c<F: Float>(n: usize, x: *const Complex<F>, inc: isize) -> F {
    let mut scale = F::zero();
    let mut ssq = F::one();
    for i in 0..n {
        let z = *x.offset(vec_offset(i, n, inc));
        for comp in [z.re, z.im] {
            if comp != F::zero() {
                let a = comp.abs();
                if scale < a {
                    let r = scale / a;
                    ssq = F::one() + ssq * r * r;
                    scale = a;
                } else {
                    let r = a / scale;
                    ssq = ssq + r * r;
                }
            }
        }
    }
    scale * ssq.sqrt()
}

// =============================================================================
// ASUM — sum of |re| + |im| over the vector (real result)
// =============================================================================

/// Complex single precision `scasum`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_scasum(n: i32, x: *const Complex32, incx: i32) -> f32 {
    if n <= 0 || incx <= 0 {
        return 0.0;
    }
    asum_c(n as usize, x, incx as isize)
}

/// Complex double precision `dzasum`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dzasum(n: i32, x: *const Complex64, incx: i32) -> f64 {
    if n <= 0 || incx <= 0 {
        return 0.0;
    }
    asum_c(n as usize, x, incx as isize)
}

unsafe fn asum_c<F: Float>(n: usize, x: *const Complex<F>, inc: isize) -> F {
    let mut acc = F::zero();
    for i in 0..n {
        let z = *x.offset(vec_offset(i, n, inc));
        acc = acc + z.re.abs() + z.im.abs();
    }
    acc
}

// =============================================================================
// IAMAX — index of element with largest |re| + |im| (0-based)
// =============================================================================

/// Complex single precision `icamax` (0-based index).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_icamax(n: i32, x: *const Complex32, incx: i32) -> i32 {
    if n <= 0 || incx <= 0 {
        return 0;
    }
    iamax_c(n as usize, x, incx as isize) as i32
}

/// Complex double precision `izamax` (0-based index).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_izamax(n: i32, x: *const Complex64, incx: i32) -> i32 {
    if n <= 0 || incx <= 0 {
        return 0;
    }
    iamax_c(n as usize, x, incx as isize) as i32
}

unsafe fn iamax_c<F: Float>(n: usize, x: *const Complex<F>, inc: isize) -> usize {
    let mut max_idx = 0usize;
    let z0 = *x;
    let mut max_val = z0.re.abs() + z0.im.abs();
    for i in 1..n {
        let z = *x.offset(vec_offset(i, n, inc));
        let v = z.re.abs() + z.im.abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cblas_zscal() {
        let alpha = Complex64::new(2.0, 1.0);
        let mut x = [Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0)];
        unsafe {
            cblas_zscal(2, &alpha, x.as_mut_ptr(), 1);
        }
        // (2+i)(1+i) = 2 + 2i + i - 1 = 1 + 3i
        // (2+i)(0-2i) = -4i - 2i^2 = 2 - 4i
        assert!((x[0].re - 1.0).abs() < 1e-12 && (x[0].im - 3.0).abs() < 1e-12);
        assert!((x[1].re - 2.0).abs() < 1e-12 && (x[1].im + 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zdscal() {
        let mut x = [Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)];
        unsafe {
            cblas_zdscal(2, 0.5, x.as_mut_ptr(), 1);
        }
        assert!((x[0].re - 0.5).abs() < 1e-12 && (x[0].im + 1.0).abs() < 1e-12);
        assert!((x[1].re - 1.5).abs() < 1e-12 && (x[1].im - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zaxpy_strided() {
        let alpha = Complex64::new(1.0, 0.0);
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)];
        // y strided incy=2
        let mut y = [
            Complex64::new(0.0, 0.0),
            Complex64::new(9.0, 9.0),
            Complex64::new(0.0, 0.0),
        ];
        unsafe {
            cblas_zaxpy(2, &alpha, x.as_ptr(), 1, y.as_mut_ptr(), 2);
        }
        assert!((y[0].re - 1.0).abs() < 1e-12 && (y[0].im - 1.0).abs() < 1e-12);
        assert!((y[2].re - 2.0).abs() < 1e-12 && (y[2].im - 2.0).abs() < 1e-12);
        // untouched
        assert!((y[1].re - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zaxpy_matches_native_unit() {
        let alpha = Complex64::new(1.5, -0.5);
        let x = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 3.0),
        ];
        let mut y = [
            Complex64::new(0.5, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(1.0, -1.0),
        ];
        let mut y_ref = y;
        unsafe {
            cblas_zaxpy(3, &alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        level1::axpy(alpha, &x, &mut y_ref);
        for (g, r) in y.iter().zip(y_ref.iter()) {
            assert!((g.re - r.re).abs() < 1e-12 && (g.im - r.im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_zcopy_zswap() {
        let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        unsafe {
            cblas_zcopy(2, x.as_ptr(), 1, y.as_mut_ptr(), 1);
        }
        assert_eq!(y, x);
        let mut a = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let mut b = [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];
        unsafe {
            cblas_zswap(2, a.as_mut_ptr(), 1, b.as_mut_ptr(), 1);
        }
        assert_eq!(a, [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]);
        assert_eq!(b, [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
    }

    #[test]
    fn test_cblas_dznrm2() {
        // z = [3+4i] -> |z| = 5 ; [3+4i, 0] -> 5
        let x = [Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)];
        let r = unsafe { cblas_dznrm2(2, x.as_ptr(), 1) };
        assert!((r - 5.0).abs() < 1e-12);
        // [1+1i, 1+1i] -> sqrt(1+1+1+1) = 2
        let y = [Complex64::new(1.0, 1.0), Complex64::new(1.0, 1.0)];
        let r2 = unsafe { cblas_dznrm2(2, y.as_ptr(), 1) };
        assert!((r2 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_dzasum() {
        let x = [Complex64::new(1.0, -2.0), Complex64::new(-3.0, 4.0)];
        let r = unsafe { cblas_dzasum(2, x.as_ptr(), 1) };
        // |1|+|−2| + |−3|+|4| = 1+2+3+4 = 10
        assert!((r - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_izamax() {
        let x = [
            Complex64::new(1.0, 1.0),  // 2
            Complex64::new(3.0, -3.0), // 6
            Complex64::new(0.0, 5.0),  // 5
        ];
        let idx = unsafe { cblas_izamax(3, x.as_ptr(), 1) };
        assert_eq!(idx, 1);
    }
}
