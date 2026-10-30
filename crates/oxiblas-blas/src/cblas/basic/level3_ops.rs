//! CBLAS Level 3 (matrix-matrix) real operations: GEMM.

use super::super::types::*;
use super::super::validate::gemm_params_valid;
use crate::level3;
/// Double precision GEMM: C = alpha * op(A) * op(B) + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f64`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dgemm(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
    {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    // Create matrices from raw pointers and call our internal GEMM
    match layout {
        CblasLayout::ColMajor => {
            gemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            // For row-major: C = A * B in row-major is equivalent to C^T = B^T * A^T in col-major
            // So we swap A and B and swap their transposes
            let new_transa = transb;
            let new_transb = transa;
            gemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
        }
    }
}

/// Internal GEMM for column-major layout.
///
/// Uses optimized SIMD implementation for NoTrans/NoTrans case.
unsafe fn gemm_raw_colmajor(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    // Fast path: NoTrans/NoTrans - use optimized SIMD implementation
    if matches!(transa, CblasTranspose::NoTrans) && matches!(transb, CblasTranspose::NoTrans) {
        use oxiblas_matrix::{MatMut, MatRef};

        // A is m×k with leading dimension lda
        let a_ref = MatRef::<f64>::new(a, m, k, lda);
        // B is k×n with leading dimension ldb
        let b_ref = MatRef::<f64>::new(b, k, n, ldb);
        // C is m×n with leading dimension ldc
        let c_mut = MatMut::<f64>::new(c.cast::<f64>(), m, n, ldc);

        level3::gemm(alpha, a_ref, b_ref, beta, c_mut);
        return;
    }

    // Fallback: handle transpose cases with scalar implementation.
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite rather than multiply (0*NaN would be NaN).
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == 0.0 {
                *cp = 0.0;
            } else {
                *cp *= beta;
            }
        }
    }

    // Reference BLAS reads neither A nor B when alpha == 0 (C = beta * C);
    // `alpha * b` / `a * 0` would otherwise turn NaN/Inf operands into NaN.
    if k == 0 || alpha == 0.0 {
        return;
    }

    // Compute C += alpha * op(A) * op(B)
    for j in 0..n {
        for p in 0..k {
            // Get B element based on transpose
            let b_val = match transb {
                CblasTranspose::NoTrans => *b.add(p + j * ldb),
                CblasTranspose::Trans | CblasTranspose::ConjTrans => *b.add(j + p * ldb),
            };
            let temp = alpha * b_val;

            for i in 0..m {
                // Get A element based on transpose
                let a_val = match transa {
                    CblasTranspose::NoTrans => *a.add(i + p * lda),
                    CblasTranspose::Trans | CblasTranspose::ConjTrans => *a.add(p + i * lda),
                };
                let cp = c.add(i + j * ldc);
                *cp += a_val * temp;
            }
        }
    }
}

/// Single precision GEMM: C = alpha * op(A) * op(B) + beta * C.
///
/// # Safety
///
/// This is a C ABI entry point. Scalar arguments (dimensions, leading
/// dimensions, enum flags) are validated before any pointer is
/// dereferenced, but — like every BLAS/LAPACK ABI — the raw pointers
/// themselves are trusted. The caller must ensure:
///
/// - `a` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `lda`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `lda` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `b` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from at every `ldb`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldb` for internal consistency but cannot verify
///   the pointer's actual allocation size).
/// - `c` must be non-null, properly aligned for `f32`, and point to
///   a buffer large enough to be read from and written to at every `ldc`-strided offset
///   this function computes, consistent with `layout` and the declared
///   matrix dimensions (the parameter-validation helper this function
///   calls checks `ldc` for internal consistency but cannot verify
///   the pointer's actual allocation size).
///
/// - No two pointer parameters may alias in a way that violates Rust's
///   aliasing rules unless this function's documented semantics
///   explicitly allow it (for example, an in-place call with an output
///   pointer equal to an input pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_sgemm(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    if !gemm_params_valid(layout, transa, transb, m, n, k, lda, ldb, ldc)
        || a.is_null()
        || b.is_null()
        || c.is_null()
    {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    match layout {
        CblasLayout::ColMajor => {
            sgemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            let new_transa = transb;
            let new_transb = transa;
            sgemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
        }
    }
}

/// Internal SGEMM for column-major layout.
///
/// Uses optimized SIMD implementation for NoTrans/NoTrans case.
unsafe fn sgemm_raw_colmajor(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Fast path: NoTrans/NoTrans - use optimized SIMD implementation
    if matches!(transa, CblasTranspose::NoTrans) && matches!(transb, CblasTranspose::NoTrans) {
        use oxiblas_matrix::{MatMut, MatRef};

        // A is m×k with leading dimension lda
        let a_ref = MatRef::<f32>::new(a, m, k, lda);
        // B is k×n with leading dimension ldb
        let b_ref = MatRef::<f32>::new(b, k, n, ldb);
        // C is m×n with leading dimension ldc
        let c_mut = MatMut::<f32>::new(c.cast::<f32>(), m, n, ldc);

        level3::gemm(alpha, a_ref, b_ref, beta, c_mut);
        return;
    }

    // Fallback: handle transpose cases with scalar implementation.
    // Scale C by beta. When beta == 0 the BLAS contract says C need not be
    // initialized, so we overwrite rather than multiply (0*NaN would be NaN).
    for j in 0..n {
        for i in 0..m {
            let cp = c.add(i + j * ldc);
            if beta == 0.0 {
                *cp = 0.0;
            } else {
                *cp *= beta;
            }
        }
    }

    // Reference BLAS reads neither A nor B when alpha == 0 (C = beta * C).
    if k == 0 || alpha == 0.0 {
        return;
    }

    // Compute C += alpha * op(A) * op(B)
    for j in 0..n {
        for p in 0..k {
            let b_val = match transb {
                CblasTranspose::NoTrans => *b.add(p + j * ldb),
                CblasTranspose::Trans | CblasTranspose::ConjTrans => *b.add(j + p * ldb),
            };
            let temp = alpha * b_val;

            for i in 0..m {
                let a_val = match transa {
                    CblasTranspose::NoTrans => *a.add(i + p * lda),
                    CblasTranspose::Trans | CblasTranspose::ConjTrans => *a.add(p + i * lda),
                };
                let cp = c.add(i + j * ldc);
                *cp += a_val * temp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{cblas_dgemm, cblas_sgemm};
    use crate::cblas::types::{CblasLayout, CblasTranspose};
    use crate::cblas::{cblas_cgemm, cblas_zgemm};
    use num_complex::{Complex32, Complex64};

    /// Layout / transpose combinations covering the ColMajor NoTrans/NoTrans
    /// fast path (into `level3::gemm`), the scalar transpose fallback and the
    /// RowMajor operand swap, at a small-path (5) and a blocked (40) size.
    const CASES: [(CblasLayout, CblasTranspose, CblasTranspose, i32); 8] = [
        (
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            5,
        ),
        (
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            40,
        ),
        (
            CblasLayout::ColMajor,
            CblasTranspose::Trans,
            CblasTranspose::NoTrans,
            5,
        ),
        (
            CblasLayout::ColMajor,
            CblasTranspose::Trans,
            CblasTranspose::NoTrans,
            40,
        ),
        (
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::ConjTrans,
            40,
        ),
        (
            CblasLayout::RowMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            40,
        ),
        (
            CblasLayout::RowMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::Trans,
            40,
        ),
        (
            CblasLayout::RowMajor,
            CblasTranspose::ConjTrans,
            CblasTranspose::Trans,
            5,
        ),
    ];

    // Issue #9: reference BLAS reads neither A nor B when alpha == 0
    // (C := beta * C), and never reads C when beta == 0.

    #[test]
    fn test_issue_9_cblas_dgemm_alpha_zero_ignores_nan_operands() {
        for &(layout, ta, tb, n) in &CASES {
            let len = (n * n) as usize;
            let a = vec![f64::NAN; len];
            let b = vec![f64::INFINITY; len];
            for &(beta, c0, expected) in &[(0.5, 2.0, 1.0), (0.0, f64::NAN, 0.0)] {
                let mut c = vec![c0; len];
                // SAFETY: every buffer holds n*n elements with leading dimension n.
                unsafe {
                    cblas_dgemm(
                        layout,
                        ta,
                        tb,
                        n,
                        n,
                        n,
                        0.0,
                        a.as_ptr(),
                        n,
                        b.as_ptr(),
                        n,
                        beta,
                        c.as_mut_ptr(),
                        n,
                    );
                }
                assert!(
                    c.iter().all(|&v| v.to_bits() == f64::to_bits(expected)),
                    "dgemm {layout:?} {ta:?}/{tb:?} n={n} beta={beta}: {:?}",
                    &c[..4]
                );
            }
        }
    }

    #[test]
    fn test_issue_9_cblas_sgemm_alpha_zero_ignores_nan_operands() {
        for &(layout, ta, tb, n) in &CASES {
            let len = (n * n) as usize;
            let a = vec![f32::NAN; len];
            let b = vec![f32::NEG_INFINITY; len];
            for &(beta, c0, expected) in &[(0.5f32, 2.0f32, 1.0f32), (0.0, f32::NAN, 0.0)] {
                let mut c = vec![c0; len];
                // SAFETY: every buffer holds n*n elements with leading dimension n.
                unsafe {
                    cblas_sgemm(
                        layout,
                        ta,
                        tb,
                        n,
                        n,
                        n,
                        0.0,
                        a.as_ptr(),
                        n,
                        b.as_ptr(),
                        n,
                        beta,
                        c.as_mut_ptr(),
                        n,
                    );
                }
                assert!(
                    c.iter().all(|&v| v.to_bits() == f32::to_bits(expected)),
                    "sgemm {layout:?} {ta:?}/{tb:?} n={n} beta={beta}: {:?}",
                    &c[..4]
                );
            }
        }
    }

    #[test]
    fn test_issue_9_cblas_zgemm_alpha_zero_ignores_nan_operands() {
        let zero = Complex64::new(0.0, 0.0);
        let nan = Complex64::new(f64::NAN, f64::NAN);
        for &(layout, ta, tb, n) in &CASES {
            let len = (n * n) as usize;
            let a = vec![nan; len];
            let b = vec![Complex64::new(f64::INFINITY, 1.0); len];
            let cases = [
                (
                    Complex64::new(0.5, 0.0),
                    Complex64::new(2.0, -4.0),
                    Complex64::new(1.0, -2.0),
                ),
                (zero, nan, zero),
            ];
            for &(beta, c0, expected) in &cases {
                let mut c = vec![c0; len];
                // SAFETY: every buffer holds n*n elements with leading dimension
                // n; alpha/beta point to live locals.
                unsafe {
                    cblas_zgemm(
                        layout,
                        ta,
                        tb,
                        n,
                        n,
                        n,
                        &zero,
                        a.as_ptr(),
                        n,
                        b.as_ptr(),
                        n,
                        &beta,
                        c.as_mut_ptr(),
                        n,
                    );
                }
                assert!(
                    c.iter().all(|&v| v == expected),
                    "zgemm {layout:?} {ta:?}/{tb:?} n={n} beta={beta}: {:?}",
                    &c[..2]
                );
            }
        }
    }

    #[test]
    fn test_issue_9_cblas_cgemm_alpha_zero_ignores_nan_operands() {
        let zero = Complex32::new(0.0, 0.0);
        let nan = Complex32::new(f32::NAN, f32::NAN);
        for &(layout, ta, tb, n) in &CASES {
            let len = (n * n) as usize;
            let a = vec![nan; len];
            let b = vec![Complex32::new(1.0, f32::INFINITY); len];
            let cases = [
                (
                    Complex32::new(0.5, 0.0),
                    Complex32::new(2.0, 4.0),
                    Complex32::new(1.0, 2.0),
                ),
                (zero, nan, zero),
            ];
            for &(beta, c0, expected) in &cases {
                let mut c = vec![c0; len];
                // SAFETY: every buffer holds n*n elements with leading dimension
                // n; alpha/beta point to live locals.
                unsafe {
                    cblas_cgemm(
                        layout,
                        ta,
                        tb,
                        n,
                        n,
                        n,
                        &zero,
                        a.as_ptr(),
                        n,
                        b.as_ptr(),
                        n,
                        &beta,
                        c.as_mut_ptr(),
                        n,
                    );
                }
                assert!(
                    c.iter().all(|&v| v == expected),
                    "cgemm {layout:?} {ta:?}/{tb:?} n={n} beta={beta}: {:?}",
                    &c[..2]
                );
            }
        }
    }

    #[test]
    fn test_issue_9_cblas_dgemm_beta_zero_ignores_nan_c() {
        // A = B = ones: every entry of op(A) * op(B) is n.
        for &(layout, ta, tb, n) in &CASES {
            let len = (n * n) as usize;
            let ones = vec![1.0f64; len];
            let mut c = vec![f64::NAN; len];
            // SAFETY: every buffer holds n*n elements with leading dimension n.
            unsafe {
                cblas_dgemm(
                    layout,
                    ta,
                    tb,
                    n,
                    n,
                    n,
                    1.0,
                    ones.as_ptr(),
                    n,
                    ones.as_ptr(),
                    n,
                    0.0,
                    c.as_mut_ptr(),
                    n,
                );
            }
            assert!(
                c.iter().all(|&v| v == f64::from(n)),
                "dgemm beta=0 {layout:?} {ta:?}/{tb:?} n={n}: {:?}",
                &c[..4]
            );
        }
    }
}
