// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! NEON kernels for GEMM shapes that a 64-row SME2 tile cannot use well: one
//! row (`m == 1`) or one column (`n == 1`). All matrices are column-major;
//! each kernel streams the large operand once, unpacked.

use core::arch::aarch64::*;

/// Applies `c = alpha * acc + beta * c` (C is not read when `beta == 0`).
#[inline(always)]
fn finish<T>(c: &mut T, acc: T, alpha: T, beta: T)
where
    T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + PartialEq + Default,
{
    *c = if beta == T::default() {
        alpha * acc
    } else {
        alpha * acc + beta * *c
    };
}

macro_rules! thin_kernels {
    ($m:ident, $t:ty, $lanes:expr, $vec:ty, $dup:ident, $ldr:ident, $ld:ident, $st:ident, $fma:ident, $add:ident, $addv:ident) => {
        pub(crate) mod $m {
            use super::*;

            /// `y = alpha * A * x + beta * y` (one column, `n == 1`): A is
            /// `m x k` with column stride `lda`, `x` has `k` and `y` `m`
            /// elements. The columns of A are read in order, four at a time,
            /// and added into an accumulator of `m` elements (L1-resident).
            ///
            /// # Safety
            ///
            /// The pointers must be valid for the given shapes and strides.
            #[allow(clippy::too_many_arguments)]
            pub(crate) unsafe fn one_col(
                m: usize,
                k: usize,
                alpha: $t,
                a: *const $t,
                lda: usize,
                x: *const $t,
                beta: $t,
                y: *mut $t,
            ) {
                let mv = m / $lanes * $lanes;
                let mut acc = vec![0.0 as $t; m];
                let s = acc.as_mut_ptr();
                let mut p = 0;
                while p + 4 <= k {
                    // SAFETY: columns p..p + 4 of A have m rows; acc has m.
                    unsafe {
                        let (x0, x1, x2, x3) = (
                            $dup(*x.add(p)),
                            $dup(*x.add(p + 1)),
                            $dup(*x.add(p + 2)),
                            $dup(*x.add(p + 3)),
                        );
                        let (c0, c1, c2, c3) = (
                            a.add(p * lda),
                            a.add((p + 1) * lda),
                            a.add((p + 2) * lda),
                            a.add((p + 3) * lda),
                        );
                        let mut i = 0;
                        while i < mv {
                            let mut v = $ld(s.add(i));
                            v = $fma(v, $ld(c0.add(i)), x0);
                            v = $fma(v, $ld(c1.add(i)), x1);
                            v = $fma(v, $ld(c2.add(i)), x2);
                            v = $fma(v, $ld(c3.add(i)), x3);
                            $st(s.add(i), v);
                            i += $lanes;
                        }
                        for i in mv..m {
                            *s.add(i) += *c0.add(i) * *x.add(p)
                                + *c1.add(i) * *x.add(p + 1)
                                + *c2.add(i) * *x.add(p + 2)
                                + *c3.add(i) * *x.add(p + 3);
                        }
                    }
                    p += 4;
                }
                while p < k {
                    // SAFETY: column p of A has m rows.
                    unsafe {
                        let xv = *x.add(p);
                        let col = a.add(p * lda);
                        for i in 0..m {
                            *s.add(i) += *col.add(i) * xv;
                        }
                    }
                    p += 1;
                }
                for (i, &v) in acc.iter().enumerate() {
                    // SAFETY: row i is inside y.
                    finish(unsafe { &mut *y.add(i) }, v, alpha, beta);
                }
            }

            /// Dot product of `k` elements of `a` and `b`.
            #[inline(always)]
            unsafe fn dot(a: *const $t, b: *const $t, k: usize) -> $t {
                let kv = k / (4 * $lanes) * (4 * $lanes);
                // SAFETY: the caller guarantees k elements behind both.
                unsafe {
                    let (mut s0, mut s1, mut s2, mut s3) =
                        ($dup(0.0), $dup(0.0), $dup(0.0), $dup(0.0));
                    let mut p = 0;
                    while p < kv {
                        s0 = $fma(s0, $ld(a.add(p)), $ld(b.add(p)));
                        s1 = $fma(s1, $ld(a.add(p + $lanes)), $ld(b.add(p + $lanes)));
                        s2 = $fma(s2, $ld(a.add(p + 2 * $lanes)), $ld(b.add(p + 2 * $lanes)));
                        s3 = $fma(s3, $ld(a.add(p + 3 * $lanes)), $ld(b.add(p + 3 * $lanes)));
                        p += 4 * $lanes;
                    }
                    let mut r = $addv($add($add(s0, s1), $add(s2, s3)));
                    while p < k {
                        r += *a.add(p) * *b.add(p);
                        p += 1;
                    }
                    r
                }
            }

            /// `C = alpha * A * B + beta * C` for one row (`m == 1`): each
            /// entry is the dot product of the row of A with a column of B,
            /// which are contiguous; A's row is first copied (stride `lda`).
            ///
            /// # Safety
            ///
            /// The pointers must be valid for the given shapes and strides.
            #[allow(clippy::too_many_arguments)]
            pub(crate) unsafe fn one_row(
                n: usize,
                k: usize,
                alpha: $t,
                a: *const $t,
                lda: usize,
                b: *const $t,
                ldb: usize,
                beta: $t,
                c: *mut $t,
                ldc: usize,
            ) {
                // SAFETY: entries p * lda of the row of A, p < k.
                let row: Vec<$t> = (0..k).map(|p| unsafe { *a.add(p * lda) }).collect();
                for j in 0..n {
                    // SAFETY: column j of B has k rows; (0, j) is inside C.
                    unsafe {
                        let d = dot(row.as_ptr(), b.add(j * ldb), k);
                        finish(&mut *c.add(j * ldc), d, alpha, beta);
                    }
                }
            }
        }
    };
}

thin_kernels!(
    f64k,
    f64,
    2,
    float64x2_t,
    vdupq_n_f64,
    vld1q_dup_f64,
    vld1q_f64,
    vst1q_f64,
    vfmaq_f64,
    vaddq_f64,
    vaddvq_f64
);
thin_kernels!(
    f32k,
    f32,
    4,
    float32x4_t,
    vdupq_n_f32,
    vld1q_dup_f32,
    vld1q_f32,
    vst1q_f32,
    vfmaq_f32,
    vaddq_f32,
    vaddvq_f32
);
