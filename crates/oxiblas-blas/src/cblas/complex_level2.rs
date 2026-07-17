//! CBLAS-compatible complex Level-2 routines.
//!
//! Provides `cgemv`/`zgemv` (general matrix-vector, marshaled manually like the
//! real `cblas_dgemv`) and `chemv`/`zhemv` (Hermitian matrix-vector, wrapping
//! the native `level2::hemv`).
//!
//! # Row-major Hermitian identity
//!
//! For `hemv`, reinterpreting the row-major buffer column-major yields
//! `A^T = conj(A)` (Hermitian, opposite triangle). Conjugating the whole
//! equation `y = alpha*A*x + beta*y` gives
//! `conj(y) = conj(alpha)*conj(A)*conj(x) + conj(beta)*conj(y)`, i.e. a
//! column-major `hemv` on `conj(A)` with conjugated `alpha`, `beta`, `x` and a
//! conjugated result. That is exactly the row-major mapping used below.

use super::types::*;
use crate::level2;
use num_complex::{Complex, Complex32, Complex64};
use num_traits::Float;
use oxiblas_matrix::MatRef;

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

/// Gathers a strided complex vector into a contiguous `Vec`, optionally
/// conjugating each element (used by the row-major Hermitian path).
unsafe fn gather_c<F: Float>(
    x: *const Complex<F>,
    n: usize,
    inc: isize,
    conj: bool,
) -> Vec<Complex<F>> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let z = *x.offset(vec_offset(i, n, inc));
        v.push(if conj { z.conj() } else { z });
    }
    v
}

// =============================================================================
// GEMV — y = alpha*op(A)*x + beta*y  (complex)
// =============================================================================

/// Complex single precision GEMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_cgemv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    gemv_c(
        layout, trans, m as usize, n as usize, *alpha, a, lda as usize, x, incx as isize, *beta, y,
        incy as isize,
    );
}

/// Complex double precision GEMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zgemv(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: i32,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    gemv_c(
        layout, trans, m as usize, n as usize, *alpha, a, lda as usize, x, incx as isize, *beta, y,
        incy as isize,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemv_c<F: Float>(
    layout: CblasLayout,
    trans: CblasTranspose,
    m: usize,
    n: usize,
    alpha: Complex<F>,
    a: *const Complex<F>,
    lda: usize,
    x: *const Complex<F>,
    incx: isize,
    beta: Complex<F>,
    y: *mut Complex<F>,
    incy: isize,
) {
    // op(A) is M x N for NoTrans, else N x M; length of y then x follow.
    let (leny, lenx) = match trans {
        CblasTranspose::NoTrans => (m, n),
        _ => (n, m),
    };
    let zero = Complex::new(F::zero(), F::zero());

    // Scale y by beta (beta == 0 must not read/propagate uninitialized y).
    for i in 0..leny {
        let p = y.offset(vec_offset(i, leny, incy));
        *p = if beta == zero { zero } else { *p * beta };
    }

    for i in 0..leny {
        let mut acc = zero;
        for j in 0..lenx {
            // opA(i,j): NoTrans -> A[i,j]; Trans/ConjTrans -> A[j,i] (conjugated
            // for ConjTrans). A[r,c] indexing follows the storage layout.
            let (ar, ac) = match trans {
                CblasTranspose::NoTrans => (i, j),
                _ => (j, i),
            };
            let raw = match layout {
                CblasLayout::ColMajor => *a.add(ar + ac * lda),
                CblasLayout::RowMajor => *a.add(ar * lda + ac),
            };
            let a_elem = match trans {
                CblasTranspose::ConjTrans => raw.conj(),
                _ => raw,
            };
            let xj = *x.offset(vec_offset(j, lenx, incx));
            acc = acc + a_elem * xj;
        }
        let p = y.offset(vec_offset(i, leny, incy));
        *p = *p + alpha * acc;
    }
}

// =============================================================================
// HEMV — y = alpha*A*x + beta*y  (A Hermitian) — wraps native hemv
// =============================================================================

/// Complex single precision HEMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_chemv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: i32,
    x: *const Complex32,
    incx: i32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    hemv_c(
        layout, uplo, n as usize, *alpha, a, lda as usize, x, incx as isize, *beta, y,
        incy as isize,
    );
}

/// Complex double precision HEMV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_zhemv(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: i32,
    x: *const Complex64,
    incx: i32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: i32,
) {
    if n <= 0 {
        return;
    }
    hemv_c(
        layout, uplo, n as usize, *alpha, a, lda as usize, x, incx as isize, *beta, y,
        incy as isize,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn hemv_c<F: Float + oxiblas_core::scalar::Real>(
    layout: CblasLayout,
    uplo: CblasUplo,
    n: usize,
    alpha: Complex<F>,
    a: *const Complex<F>,
    lda: usize,
    x: *const Complex<F>,
    incx: isize,
    beta: Complex<F>,
    y: *mut Complex<F>,
    incy: isize,
) where
    Complex<F>: oxiblas_core::scalar::Field,
{
    match layout {
        CblasLayout::ColMajor => {
            let x_vec = gather_c(x, n, incx, false);
            let mut y_vec = gather_c(y, n, incy, false);
            let a_ref = MatRef::<Complex<F>>::new(a, n, n, lda);
            let _ = level2::hemv(map_hemv_uplo(uplo), alpha, a_ref, &x_vec, beta, &mut y_vec);
            scatter_c(y, &y_vec, n, incy, false);
        }
        CblasLayout::RowMajor => {
            // Conjugated column-major solve on conj(A) with flipped uplo.
            let x_vec = gather_c(x, n, incx, true);
            let mut y_vec = gather_c(y, n, incy, true);
            let a_ref = MatRef::<Complex<F>>::new(a, n, n, lda);
            let _ = level2::hemv(
                map_hemv_uplo(flip_uplo(uplo)),
                alpha.conj(),
                a_ref,
                &x_vec,
                beta.conj(),
                &mut y_vec,
            );
            // Conjugate the result back into y.
            scatter_c(y, &y_vec, n, incy, true);
        }
    }
}

unsafe fn scatter_c<F: Float>(dst: *mut Complex<F>, src: &[Complex<F>], n: usize, inc: isize, conj: bool) {
    for (i, &val) in src.iter().enumerate().take(n) {
        *dst.offset(vec_offset(i, n, inc)) = if conj { val.conj() } else { val };
    }
}

#[inline]
fn map_hemv_uplo(uplo: CblasUplo) -> level2::HemvUplo {
    match uplo {
        CblasUplo::Upper => level2::HemvUplo::Upper,
        CblasUplo::Lower => level2::HemvUplo::Lower,
    }
}

#[inline]
fn flip_uplo(uplo: CblasUplo) -> CblasUplo {
    match uplo {
        CblasUplo::Upper => CblasUplo::Lower,
        CblasUplo::Lower => CblasUplo::Upper,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cblas_zgemv_notrans() {
        // A = [[1+i, 2],[0, 1-i]] col-major: [1+i, 0, 2, 1-i]
        let a = [
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ];
        let x = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);
        unsafe {
            cblas_zgemv(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                2,
                2,
                &alpha,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                &beta,
                y.as_mut_ptr(),
                1,
            );
        }
        // y[0] = (1+i)*1 + 2*(i) = 1+i + 2i = 1 + 3i
        // y[1] = 0*1 + (1-i)*(i) = i - i^2 = 1 + i
        assert!((y[0].re - 1.0).abs() < 1e-12 && (y[0].im - 3.0).abs() < 1e-12);
        assert!((y[1].re - 1.0).abs() < 1e-12 && (y[1].im - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zgemv_conjtrans() {
        let a = [
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ];
        let x = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut y = [Complex64::new(0.0, 0.0); 2];
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);
        unsafe {
            cblas_zgemv(
                CblasLayout::ColMajor,
                CblasTranspose::ConjTrans,
                2,
                2,
                &alpha,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                &beta,
                y.as_mut_ptr(),
                1,
            );
        }
        // op = A^H. A = [[1+i, 2],[0+2i, 1-i]]
        // A^H = [[1-i, -2i],[2, 1+i]]
        // y[0] = (1-i)*1 + (-2i)*1 = 1 - 3i
        // y[1] = 2*1 + (1+i)*1 = 3 + i
        assert!((y[0].re - 1.0).abs() < 1e-12 && (y[0].im + 3.0).abs() < 1e-12);
        assert!((y[1].re - 3.0).abs() < 1e-12 && (y[1].im - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cblas_zhemv_matches_native() {
        // Hermitian A (2x2), lower stored, real diagonal.
        let a = [
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0), // unused upper
            Complex64::new(3.0, 0.0),
        ];
        let x = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
        let mut y = [Complex64::new(0.5, 0.5), Complex64::new(-1.0, 0.0)];
        let mut y_ref = y;
        let alpha = Complex64::new(1.0, 0.5);
        let beta = Complex64::new(0.5, 0.0);
        unsafe {
            cblas_zhemv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                2,
                &alpha,
                a.as_ptr(),
                2,
                x.as_ptr(),
                1,
                &beta,
                y.as_mut_ptr(),
                1,
            );
            let a_ref = MatRef::<Complex64>::new(a.as_ptr(), 2, 2, 2);
            let _ = level2::hemv(level2::HemvUplo::Lower, alpha, a_ref, &x, beta, &mut y_ref);
        }
        for (g, r) in y.iter().zip(y_ref.iter()) {
            assert!((g.re - r.re).abs() < 1e-12 && (g.im - r.im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cblas_zhemv_rowmajor_hermitian_result() {
        // Row-major upper == the conjugate-transformed column-major solve.
        // Cross-check row-major Upper against column-major Lower on the same
        // logical Hermitian operator A = [[2, 1+i],[1-i, 3]].
        // Column-major Lower stores: (0,0)=2, (1,0)=1-i, (1,1)=3.
        let a_col_lower = [
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        // Row-major Upper stores: (0,0)=2, (0,1)=1+i, (1,1)=3.
        // Row-major buffer (row stride 2): [2, 1+i, junk, 3].
        let a_row_upper = [
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let x = [Complex64::new(1.0, 2.0), Complex64::new(-1.0, 1.0)];
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);
        let mut y_col = [Complex64::new(0.0, 0.0); 2];
        let mut y_row = [Complex64::new(0.0, 0.0); 2];
        unsafe {
            cblas_zhemv(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                2,
                &alpha,
                a_col_lower.as_ptr(),
                2,
                x.as_ptr(),
                1,
                &beta,
                y_col.as_mut_ptr(),
                1,
            );
            cblas_zhemv(
                CblasLayout::RowMajor,
                CblasUplo::Upper,
                2,
                &alpha,
                a_row_upper.as_ptr(),
                2,
                x.as_ptr(),
                1,
                &beta,
                y_row.as_mut_ptr(),
                1,
            );
        }
        for (r, c) in y_row.iter().zip(y_col.iter()) {
            assert!(
                (r.re - c.re).abs() < 1e-12 && (r.im - c.im).abs() < 1e-12,
                "row {r:?} col {c:?}"
            );
        }
    }
}
