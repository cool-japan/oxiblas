//! CBLAS Level 3 (matrix-matrix) real operations: GEMM.

use super::super::types::*;
use super::super::validate::gemm_params_valid;
use crate::level3;
use crate::level3::{GemmKernel, SMALL_THRESHOLD, Trans};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};
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

/// Maps a CBLAS transpose flag to the GEMM driver's.
#[inline]
fn to_trans(trans: CblasTranspose) -> Trans {
    match trans {
        CblasTranspose::NoTrans => Trans::NoTrans,
        CblasTranspose::Trans => Trans::Trans,
        CblasTranspose::ConjTrans => Trans::ConjTrans,
    }
}

/// Internal GEMM for column-major layout: `C = alpha * op(A) * op(B) + beta * C`.
///
/// `NoTrans`/`NoTrans`, and any transpose combination with more than
/// [`SMALL_THRESHOLD`] (32³) multiply-adds, run on the blocked,
/// transpose-aware GEMM ([`level3::gemm::gemm_transposed`]), which reads
/// transposed operands in place while packing. Smaller transposed problems,
/// and transposed problems with a single output column (`n == 1`, a GEMV),
/// keep the reference-BLAS loop of [`small_transposed_colmajor`]: there it is
/// as fast as the GEMM driver, and faster for some skinny shapes. Both keep
/// the reference-BLAS semantics: `alpha == 0`
/// or `k == 0` gives `C = beta * C` without reading A or B, and `beta == 0`
/// never reads C. `ConjTrans` equals `Trans` for the real types used here.
///
/// # Safety
///
/// The column-major contract of `cblas_dgemm`/`cblas_sgemm` after parameter
/// validation: the stored A is `m × k` (`NoTrans`) or `k × m` (otherwise)
/// with leading dimension `lda >= max(1, rows)`, B likewise `k × n` or `n × k`
/// with `ldb`, and C is `m × n` with `ldc >= max(1, m)`; every strided element
/// of them is in bounds, C is writable, and C does not overlap A or B.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_raw_colmajor<T: Field + GemmKernel + bytemuck::Zeroable>(
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    b: *const T,
    ldb: usize,
    beta: T,
    c: *mut T,
    ldc: usize,
) {
    let (trans_a, trans_b) = (to_trans(transa), to_trans(transb));
    let transposed = trans_a != Trans::NoTrans || trans_b != Trans::NoTrans;
    // A single output column is a GEMV: the loop streams op(A) once, while the
    // blocked driver would pad op(B) to a whole NR-column micro-panel (on an
    // Apple M3 a 33 x 1024 x 1 call took 1.3x the loop's time there).
    if transposed && (n == 1 || m.saturating_mul(n).saturating_mul(k) <= SMALL_THRESHOLD) {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            small_transposed_colmajor(
                trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
        }
        return;
    }
    // Physical (stored) shapes of the operands.
    let (a_rows, a_cols) = if trans_a == Trans::NoTrans {
        (m, k)
    } else {
        (k, m)
    };
    let (b_rows, b_cols) = if trans_b == Trans::NoTrans {
        (k, n)
    } else {
        (n, k)
    };
    // SAFETY: by this function's contract each pointer covers its stored
    // matrix at the given shape and leading dimension (which is at least the
    // row count, as validated by `gemm_params_valid`), so the column-major
    // views address only in-bounds elements; C is exclusively ours for the
    // call and overlaps neither A nor B.
    let (a_ref, b_ref, c_mut) = unsafe {
        (
            MatRef::<T>::new(a, a_rows, a_cols, lda),
            MatRef::<T>::new(b, b_rows, b_cols, ldb),
            MatMut::<T>::new(c, m, n, ldc),
        )
    };
    level3::gemm::gemm_transposed(trans_a, trans_b, alpha, a_ref, b_ref, beta, c_mut);
}

/// Small transposed GEMM (`m * n * k <= SMALL_THRESHOLD`, at least one operand
/// transposed), column-major on raw pointers: the reference-BLAS order,
/// `C = beta * C` and then `C[:, j] += op(A)[:, p] * (alpha * op(B)[p, j])`.
///
/// This is the loop `cblas_dgemm`/`cblas_sgemm` used for every transposed
/// call before 0.2.3; it stays for the small problems, where the blocked
/// driver brings no gain (`gemm_raw_colmajor`). `beta == 0` overwrites C
/// without reading it, and A and B are not read when `alpha == 0` or `k == 0`.
///
/// # Safety
///
/// As for [`gemm_raw_colmajor`].
#[allow(clippy::too_many_arguments)]
unsafe fn small_transposed_colmajor<T: Field>(
    trans_a: Trans,
    trans_b: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    b: *const T,
    ldb: usize,
    beta: T,
    c: *mut T,
    ldc: usize,
) {
    // SAFETY: every offset below addresses an element of the stored C
    // (`i < m`, `j < n`), A (`i + p * lda` or `p + i * lda` for the stored
    // `m × k` / `k × m` shape) or B (likewise), which this function's contract
    // puts in bounds; C is writable and does not overlap A or B.
    unsafe {
        // Scale C by beta. When beta == 0 the BLAS contract says C need not
        // be initialized, so it is overwritten (0 * NaN would be NaN).
        for j in 0..n {
            for i in 0..m {
                let cp = c.add(i + j * ldc);
                if beta == T::zero() {
                    *cp = T::zero();
                } else {
                    *cp *= beta;
                }
            }
        }

        // Reference BLAS reads neither A nor B when alpha == 0 (C = beta * C);
        // `alpha * b` / `a * 0` would otherwise turn NaN/Inf operands into NaN.
        if k == 0 || alpha == T::zero() {
            return;
        }

        for j in 0..n {
            for p in 0..k {
                let b_val = match trans_b {
                    Trans::NoTrans => *b.add(p + j * ldb),
                    Trans::Trans | Trans::ConjTrans => *b.add(j + p * ldb),
                };
                let temp = alpha * b_val;
                for i in 0..m {
                    let a_val = match trans_a {
                        Trans::NoTrans => *a.add(i + p * lda),
                        Trans::Trans | Trans::ConjTrans => *a.add(p + i * lda),
                    };
                    *c.add(i + j * ldc) += a_val * temp;
                }
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
            gemm_raw_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        CblasLayout::RowMajor => {
            let new_transa = transb;
            let new_transb = transa;
            gemm_raw_colmajor(
                new_transa, new_transb, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc,
            );
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
    /// path, the transposed paths (the small loop at size 5, the blocked
    /// transpose-aware GEMM at size 40) and the RowMajor operand swap.
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

    const TRANSPOSES: [CblasTranspose; 3] = [
        CblasTranspose::NoTrans,
        CblasTranspose::Trans,
        CblasTranspose::ConjTrans,
    ];

    /// Storage of one operand: `rows × cols` logical, in `layout`, with a
    /// leading dimension padded by `pad`.
    struct Stored {
        layout: CblasLayout,
        rows: usize,
        cols: usize,
        ld: usize,
    }

    impl Stored {
        fn new(layout: CblasLayout, rows: usize, cols: usize, pad: usize) -> Self {
            let ld = match layout {
                CblasLayout::ColMajor => rows.max(1) + pad,
                CblasLayout::RowMajor => cols.max(1) + pad,
            };
            Self {
                layout,
                rows,
                cols,
                ld,
            }
        }
        fn len(&self) -> usize {
            match self.layout {
                CblasLayout::ColMajor => self.ld * self.cols.max(1),
                CblasLayout::RowMajor => self.ld * self.rows.max(1),
            }
        }
        fn idx(&self, r: usize, c: usize) -> usize {
            match self.layout {
                CblasLayout::ColMajor => r + c * self.ld,
                CblasLayout::RowMajor => r * self.ld + c,
            }
        }
        /// Element `op(X)[r, c]` of a stored operand.
        fn op_idx(&self, trans: CblasTranspose, r: usize, c: usize) -> usize {
            match trans {
                CblasTranspose::NoTrans => self.idx(r, c),
                CblasTranspose::Trans | CblasTranspose::ConjTrans => self.idx(c, r),
            }
        }
    }

    fn stored_operand(
        layout: CblasLayout,
        trans: CblasTranspose,
        rows: usize,
        cols: usize,
        pad: usize,
    ) -> Stored {
        match trans {
            CblasTranspose::NoTrans => Stored::new(layout, rows, cols, pad),
            CblasTranspose::Trans | CblasTranspose::ConjTrans => {
                Stored::new(layout, cols, rows, pad)
            }
        }
    }

    fn value(seed: usize) -> f64 {
        ((seed.wrapping_mul(2_654_435_761) >> 9) % 2003) as f64 / 1001.5 - 1.0
    }

    /// Runs `cblas_dgemm` and `cblas_sgemm` for one case and compares every
    /// element of C (including the padding between columns or rows, which
    /// must stay untouched) with a naive reference.
    #[allow(clippy::too_many_arguments)]
    fn check_cblas_case(
        layout: CblasLayout,
        ta: CblasTranspose,
        tb: CblasTranspose,
        (m, n, k): (usize, usize, usize),
        alpha: f64,
        beta: f64,
    ) {
        let sa = stored_operand(layout, ta, m, k, 3);
        let sb = stored_operand(layout, tb, k, n, 1);
        let sc = Stored::new(layout, m, n, 2);
        let a: Vec<f64> = (0..sa.len()).map(|i| value(i + 17)).collect();
        let b: Vec<f64> = (0..sb.len()).map(|i| value(i + 5_003)).collect();
        // Padding entries of C hold a sentinel; with beta == 0 the live
        // entries start as NaN, which must not propagate.
        let mut c0: Vec<f64> = vec![-7.25; sc.len()];
        for j in 0..n {
            for i in 0..m {
                c0[sc.idx(i, j)] = if beta == 0.0 {
                    f64::NAN
                } else {
                    value(i * 31 + j * 7 + 99)
                };
            }
        }
        let mut expected = c0.clone();
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[sa.op_idx(ta, i, p)] * b[sb.op_idx(tb, p, j)];
                }
                let e = &mut expected[sc.idx(i, j)];
                *e = if alpha == 0.0 { 0.0 } else { alpha * acc }
                    + if beta == 0.0 { 0.0 } else { beta * *e };
            }
        }
        let ctx = format!("{layout:?} {ta:?}/{tb:?} m={m} n={n} k={k} alpha={alpha} beta={beta}");
        let dim = |x: usize| x as i32;

        let mut c = c0.clone();
        // SAFETY: each buffer holds its stored operand with the leading
        // dimension passed, as built by `Stored`.
        unsafe {
            cblas_dgemm(
                layout,
                ta,
                tb,
                dim(m),
                dim(n),
                dim(k),
                alpha,
                a.as_ptr(),
                dim(sa.ld),
                b.as_ptr(),
                dim(sb.ld),
                beta,
                c.as_mut_ptr(),
                dim(sc.ld),
            );
        }
        for (i, (&got, &exp)) in c.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-12 * (k as f64 + 1.0),
                "dgemm {ctx}: index {i}: {got} vs {exp}"
            );
        }

        let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let mut c32: Vec<f32> = c0.iter().map(|&v| v as f32).collect();
        // SAFETY: as above, with the f32 copies of the same buffers.
        unsafe {
            cblas_sgemm(
                layout,
                ta,
                tb,
                dim(m),
                dim(n),
                dim(k),
                alpha as f32,
                a32.as_ptr(),
                dim(sa.ld),
                b32.as_ptr(),
                dim(sb.ld),
                beta as f32,
                c32.as_mut_ptr(),
                dim(sc.ld),
            );
        }
        for (i, (&got, &exp)) in c32.iter().zip(&expected).enumerate() {
            assert!(
                (f64::from(got) - exp).abs() <= 1e-5 * (k as f64 + 1.0),
                "sgemm {ctx}: index {i}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn test_cblas_real_gemm_all_transposes_both_layouts() {
        // Non-square shapes, a small-path size, blocked sizes with partial
        // micro-tiles, k == 0, and single-column / single-row outputs above
        // 32^3 (the n == 1 loop branch and the m == 1 blocked path; the
        // RowMajor swap exchanges the two).
        let shapes = [
            (7usize, 13usize, 29usize),
            (3, 2, 1),
            (1, 1, 1),
            (37, 41, 43),
            (70, 9, 65),
            (5, 6, 0),
            (40, 1, 900),
            (1, 40, 900),
        ];
        for layout in [CblasLayout::ColMajor, CblasLayout::RowMajor] {
            for ta in TRANSPOSES {
                for tb in TRANSPOSES {
                    for &shape in &shapes {
                        check_cblas_case(layout, ta, tb, shape, 1.25, 0.0);
                        check_cblas_case(layout, ta, tb, shape, -0.5, 0.75);
                        check_cblas_case(layout, ta, tb, shape, 0.0, 0.5);
                    }
                }
            }
        }
    }

    fn cvalue(seed: usize) -> Complex64 {
        Complex64::new(value(seed), value(seed.wrapping_add(7_919)))
    }

    /// Element `op(X)[r, c]` of a stored complex operand (conjugated for
    /// `ConjTrans`).
    fn op_c64(x: &[Complex64], s: &Stored, trans: CblasTranspose, r: usize, c: usize) -> Complex64 {
        let v = x[s.op_idx(trans, r, c)];
        if matches!(trans, CblasTranspose::ConjTrans) {
            v.conj()
        } else {
            v
        }
    }

    /// `cblas_zgemm` and `cblas_cgemm` against a naive reference for one case,
    /// padding of C included.
    #[allow(clippy::too_many_arguments)]
    fn check_cblas_complex_case(
        layout: CblasLayout,
        ta: CblasTranspose,
        tb: CblasTranspose,
        (m, n, k): (usize, usize, usize),
        alpha: Complex64,
        beta: Complex64,
    ) {
        let zero = Complex64::new(0.0, 0.0);
        let sa = stored_operand(layout, ta, m, k, 2);
        let sb = stored_operand(layout, tb, k, n, 3);
        let sc = Stored::new(layout, m, n, 1);
        let a: Vec<Complex64> = (0..sa.len()).map(|i| cvalue(i + 11)).collect();
        let b: Vec<Complex64> = (0..sb.len()).map(|i| cvalue(i + 3_001)).collect();
        let sentinel = Complex64::new(-7.25, 3.5);
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let mut c0 = vec![sentinel; sc.len()];
        for j in 0..n {
            for i in 0..m {
                c0[sc.idx(i, j)] = if beta == zero {
                    nan
                } else {
                    cvalue(i * 37 + j * 5 + 77)
                };
            }
        }
        let mut expected = c0.clone();
        for j in 0..n {
            for i in 0..m {
                let mut acc = zero;
                for p in 0..k {
                    acc += op_c64(&a, &sa, ta, i, p) * op_c64(&b, &sb, tb, p, j);
                }
                let e = &mut expected[sc.idx(i, j)];
                let scaled = if alpha == zero { zero } else { alpha * acc };
                *e = if beta == zero {
                    scaled
                } else {
                    scaled + beta * *e
                };
            }
        }
        let ctx = format!("{layout:?} {ta:?}/{tb:?} m={m} n={n} k={k} alpha={alpha} beta={beta}");
        let dim = |x: usize| x as i32;

        let mut c = c0.clone();
        // SAFETY: each buffer holds its stored operand with the leading
        // dimension passed; alpha/beta point to live locals.
        unsafe {
            cblas_zgemm(
                layout,
                ta,
                tb,
                dim(m),
                dim(n),
                dim(k),
                &alpha,
                a.as_ptr(),
                dim(sa.ld),
                b.as_ptr(),
                dim(sb.ld),
                &beta,
                c.as_mut_ptr(),
                dim(sc.ld),
            );
        }
        for (i, (got, exp)) in c.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).norm() <= 1e-12 * (k as f64 + 1.0),
                "zgemm {ctx}: index {i}: {got} vs {exp}"
            );
        }

        let to32 = |v: &Complex64| Complex32::new(v.re as f32, v.im as f32);
        let a32: Vec<Complex32> = a.iter().map(to32).collect();
        let b32: Vec<Complex32> = b.iter().map(to32).collect();
        let mut c32: Vec<Complex32> = c0.iter().map(to32).collect();
        let (alpha32, beta32) = (to32(&alpha), to32(&beta));
        // SAFETY: as above, with the Complex32 copies.
        unsafe {
            cblas_cgemm(
                layout,
                ta,
                tb,
                dim(m),
                dim(n),
                dim(k),
                &alpha32,
                a32.as_ptr(),
                dim(sa.ld),
                b32.as_ptr(),
                dim(sb.ld),
                &beta32,
                c32.as_mut_ptr(),
                dim(sc.ld),
            );
        }
        for (i, (got, exp)) in c32.iter().zip(&expected).enumerate() {
            let got = Complex64::new(f64::from(got.re), f64::from(got.im));
            assert!(
                (got - exp).norm() <= 2e-5 * (k as f64 + 1.0),
                "cgemm {ctx}: index {i}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn test_cblas_complex_gemm_all_transposes_both_layouts() {
        let shapes = [
            (7usize, 13usize, 29usize),
            (3, 2, 1),
            (37, 41, 43),
            (5, 6, 0),
        ];
        let zero = Complex64::new(0.0, 0.0);
        for layout in [CblasLayout::ColMajor, CblasLayout::RowMajor] {
            for ta in TRANSPOSES {
                for tb in TRANSPOSES {
                    for &shape in &shapes {
                        check_cblas_complex_case(
                            layout,
                            ta,
                            tb,
                            shape,
                            Complex64::new(1.25, -0.5),
                            zero,
                        );
                        check_cblas_complex_case(
                            layout,
                            ta,
                            tb,
                            shape,
                            Complex64::new(-0.5, 0.25),
                            Complex64::new(0.75, -1.0),
                        );
                        check_cblas_complex_case(
                            layout,
                            ta,
                            tb,
                            shape,
                            zero,
                            Complex64::new(0.5, 0.5),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_cblas_real_gemm_alpha_zero_non_square_ignores_nan_operands() {
        // Non-square transposed shapes with NaN/Inf operands and alpha == 0:
        // C = beta * C (beta == 0 writes zeros without reading the NaN C).
        let (m, n, k) = (9usize, 4usize, 6usize);
        for layout in [CblasLayout::ColMajor, CblasLayout::RowMajor] {
            for ta in TRANSPOSES {
                for tb in TRANSPOSES {
                    let sa = stored_operand(layout, ta, m, k, 2);
                    let sb = stored_operand(layout, tb, k, n, 2);
                    let sc = Stored::new(layout, m, n, 2);
                    let a = vec![f64::NAN; sa.len()];
                    let b = vec![f64::INFINITY; sb.len()];
                    for &(beta, c_live, expected) in &[(0.5, 2.0, 1.0), (0.0, f64::NAN, 0.0)] {
                        let mut c = vec![-7.25; sc.len()];
                        for j in 0..n {
                            for i in 0..m {
                                c[sc.idx(i, j)] = c_live;
                            }
                        }
                        // SAFETY: buffers hold the stored operands at the
                        // leading dimensions passed.
                        unsafe {
                            cblas_dgemm(
                                layout,
                                ta,
                                tb,
                                m as i32,
                                n as i32,
                                k as i32,
                                0.0,
                                a.as_ptr(),
                                sa.ld as i32,
                                b.as_ptr(),
                                sb.ld as i32,
                                beta,
                                c.as_mut_ptr(),
                                sc.ld as i32,
                            );
                        }
                        for j in 0..n {
                            for i in 0..m {
                                assert_eq!(
                                    c[sc.idx(i, j)].to_bits(),
                                    f64::to_bits(expected),
                                    "{layout:?} {ta:?}/{tb:?} beta={beta} ({i},{j})"
                                );
                            }
                        }
                        assert_eq!(
                            c.iter().filter(|&&v| v == -7.25).count(),
                            sc.len() - m * n,
                            "{layout:?} {ta:?}/{tb:?}: padding written"
                        );
                    }
                }
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
