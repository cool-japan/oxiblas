// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The f32 16×6 AVX2 + FMA micro-kernel (x86-64 only).
//!
//! It is never the canonical f32 kernel (`f32::micro_kernel_dispatch()` stays
//! the 8×8 kernel on the AVX2 tier); a GEMM call's plan carries it when
//! `kernel_for_problem` picks it for the problem's shape.
//!
//! # Bitwise contract
//!
//! Every element of the tile is computed exactly like the 8×8 AVX2 kernel
//! (`micro_kernel_f32_avx2` in the parent module) computes it:
//!
//! 1. `acc = 0`, then `acc = fma(a[p], b[p], acc)` for `p = 0, 1, …, k - 1`
//!    in this order (one rounding per step; the ×4 unrolling only groups the
//!    steps, every lane keeps its own chain);
//! 2. `acc = acc * alpha` (one rounding) only when `alpha != 1`;
//! 3. `beta == 0`: `c = acc` (C is not read); `beta == 1`: `c = acc + c`;
//!    otherwise `c = fma(c, beta, acc)`.
//!
//! For the same KC partition a GEMM therefore gives the bits of the 8×8
//! kernel with this one, except the payloads of NaN results; only the packed
//! panel shape differs. NaN results sit at the same positions with either
//! kernel, but their payloads may differ: the compiler may commute the first
//! two operands of an FMA (it picks `vfmadd132` / `vfmadd213` / `vfmadd231`
//! freely), and that order decides which NaN operand x86 propagates, so
//! writing the intrinsics in the 8×8 kernel's operand order does not fix the
//! payload.

use super::{MicroKernel, MicroKernelShape};
use core::arch::x86_64::{
    _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_broadcast_ss, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};

/// Rows of the register tile (two ymm vectors of eight f32).
const MR: usize = 16;
/// Columns of the register tile.
const NR: usize = 6;
/// Prefetch distance in k steps (as in the f64 8×6 AVX2 kernel).
const PREFETCH_DIST: usize = 12;

/// The 16×6 kernel with its shape, for a plan's [`MicroKernel`].
#[inline]
pub(super) fn micro_kernel_16x6() -> MicroKernel<f32> {
    MicroKernel {
        shape: MicroKernelShape { mr: MR, nr: NR },
        kernel: micro_kernel_f32_avx2_16x6,
    }
}

/// AVX2 + FMA micro-kernel for f32 (16×6):
/// `C = alpha * A~ * B~ + beta * C` on one 16 × 6 tile.
///
/// Twelve ymm accumulators (two row halves × six columns); per k step two
/// loads of A, six broadcasts of B and twelve FMAs; the k loop is unrolled
/// by four with a remainder loop, and the A and B panels are prefetched
/// `PREFETCH_DIST` steps ahead. The epilogue is the one of the 8×8 AVX2 f32
/// kernel (see the module documentation for the per-element order).
///
/// # Safety
///
/// - The CPU supports AVX2 and FMA (the plan's tier is `SimdTier::Avx2Fma`
///   and the canonical f32 kernel is the 8×8 AVX2 kernel, which is only
///   selected when both are detected).
/// - `a` is readable for `16 * k` f32: the packed `16 × k` row panel, element
///   `(i, p)` at `a + 16 * p + i`.
/// - `b` is readable for `6 * k` f32: the packed `k × 6` column panel, element
///   `(p, j)` at `b + 6 * p + j`.
/// - `c` points to a 16 × 6 column-major tile with leading dimension
///   `c_stride >= 16`: `C(i, j)` at `c + i + j * c_stride` for `i < 16`,
///   `j < 6`, all writable, and readable unless `beta == 0` (then C is not
///   read). Nothing outside those 96 elements is read or written (the drivers'
///   edge scratch tile has `c_stride == 16`).
/// - `a`, `b` and the tile of `c` do not overlap. With `k == 0`, `a` and `b`
///   are not read.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn micro_kernel_f32_avx2_16x6(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    let mut acc0_lo = _mm256_setzero_ps();
    let mut acc0_hi = _mm256_setzero_ps();
    let mut acc1_lo = _mm256_setzero_ps();
    let mut acc1_hi = _mm256_setzero_ps();
    let mut acc2_lo = _mm256_setzero_ps();
    let mut acc2_hi = _mm256_setzero_ps();
    let mut acc3_lo = _mm256_setzero_ps();
    let mut acc3_hi = _mm256_setzero_ps();
    let mut acc4_lo = _mm256_setzero_ps();
    let mut acc4_hi = _mm256_setzero_ps();
    let mut acc5_lo = _mm256_setzero_ps();
    let mut acc5_hi = _mm256_setzero_ps();
    macro_rules! fma_iter {
        ($offset:expr) => {{
            let a_ptr = a.add($offset * MR);
            let b_ptr = b.add($offset * NR);
            let a_lo = _mm256_loadu_ps(a_ptr);
            let a_hi = _mm256_loadu_ps(a_ptr.add(8));
            let b0 = _mm256_broadcast_ss(&*b_ptr);
            acc0_lo = _mm256_fmadd_ps(a_lo, b0, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(a_hi, b0, acc0_hi);
            let b1 = _mm256_broadcast_ss(&*b_ptr.add(1));
            acc1_lo = _mm256_fmadd_ps(a_lo, b1, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(a_hi, b1, acc1_hi);
            let b2 = _mm256_broadcast_ss(&*b_ptr.add(2));
            acc2_lo = _mm256_fmadd_ps(a_lo, b2, acc2_lo);
            acc2_hi = _mm256_fmadd_ps(a_hi, b2, acc2_hi);
            let b3 = _mm256_broadcast_ss(&*b_ptr.add(3));
            acc3_lo = _mm256_fmadd_ps(a_lo, b3, acc3_lo);
            acc3_hi = _mm256_fmadd_ps(a_hi, b3, acc3_hi);
            let b4 = _mm256_broadcast_ss(&*b_ptr.add(4));
            acc4_lo = _mm256_fmadd_ps(a_lo, b4, acc4_lo);
            acc4_hi = _mm256_fmadd_ps(a_hi, b4, acc4_hi);
            let b5 = _mm256_broadcast_ss(&*b_ptr.add(5));
            acc5_lo = _mm256_fmadd_ps(a_lo, b5, acc5_lo);
            acc5_hi = _mm256_fmadd_ps(a_hi, b5, acc5_hi);
        }};
    }
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            // Inside the panels: step `pf_base + 3 < k`, so the 16 f32 of A
            // at `pf_base * MR + 16` and the B step at `pf_base * NR` exist.
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            _mm_prefetch(a.add(pf_base * MR) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a.add(pf_base * MR + 16) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.add(pf_base * NR) as *const i8, _MM_HINT_T0);
        }
        fma_iter!(base);
        fma_iter!(base + 1);
        fma_iter!(base + 2);
        fma_iter!(base + 3);
    }
    let base = k_unroll * 4;
    for r in 0..k_remainder {
        fma_iter!(base + r);
    }
    if alpha != 1.0 {
        let alpha_vec = _mm256_set1_ps(alpha);
        acc0_lo = _mm256_mul_ps(acc0_lo, alpha_vec);
        acc0_hi = _mm256_mul_ps(acc0_hi, alpha_vec);
        acc1_lo = _mm256_mul_ps(acc1_lo, alpha_vec);
        acc1_hi = _mm256_mul_ps(acc1_hi, alpha_vec);
        acc2_lo = _mm256_mul_ps(acc2_lo, alpha_vec);
        acc2_hi = _mm256_mul_ps(acc2_hi, alpha_vec);
        acc3_lo = _mm256_mul_ps(acc3_lo, alpha_vec);
        acc3_hi = _mm256_mul_ps(acc3_hi, alpha_vec);
        acc4_lo = _mm256_mul_ps(acc4_lo, alpha_vec);
        acc4_hi = _mm256_mul_ps(acc4_hi, alpha_vec);
        acc5_lo = _mm256_mul_ps(acc5_lo, alpha_vec);
        acc5_hi = _mm256_mul_ps(acc5_hi, alpha_vec);
    }
    if beta == 0.0 {
        macro_rules! store {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {{
                let c_col = c.add($col * c_stride);
                _mm256_storeu_ps(c_col, $acc_lo);
                _mm256_storeu_ps(c_col.add(8), $acc_hi);
            }};
        }
        store!(0, acc0_lo, acc0_hi);
        store!(1, acc1_lo, acc1_hi);
        store!(2, acc2_lo, acc2_hi);
        store!(3, acc3_lo, acc3_hi);
        store!(4, acc4_lo, acc4_hi);
        store!(5, acc5_lo, acc5_hi);
        return;
    }
    if beta == 1.0 {
        macro_rules! store_add {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {{
                let c_col = c.add($col * c_stride);
                _mm256_storeu_ps(c_col, _mm256_add_ps($acc_lo, _mm256_loadu_ps(c_col)));
                _mm256_storeu_ps(
                    c_col.add(8),
                    _mm256_add_ps($acc_hi, _mm256_loadu_ps(c_col.add(8))),
                );
            }};
        }
        store_add!(0, acc0_lo, acc0_hi);
        store_add!(1, acc1_lo, acc1_hi);
        store_add!(2, acc2_lo, acc2_hi);
        store_add!(3, acc3_lo, acc3_hi);
        store_add!(4, acc4_lo, acc4_hi);
        store_add!(5, acc5_lo, acc5_hi);
        return;
    }
    let beta_vec = _mm256_set1_ps(beta);
    macro_rules! store_fma {
        ($col:expr, $acc_lo:expr, $acc_hi:expr) => {{
            let c_col = c.add($col * c_stride);
            let c_lo = _mm256_loadu_ps(c_col);
            let c_hi = _mm256_loadu_ps(c_col.add(8));
            _mm256_storeu_ps(c_col, _mm256_fmadd_ps(c_lo, beta_vec, $acc_lo));
            _mm256_storeu_ps(c_col.add(8), _mm256_fmadd_ps(c_hi, beta_vec, $acc_hi));
        }};
    }
    store_fma!(0, acc0_lo, acc0_hi);
    store_fma!(1, acc1_lo, acc1_hi);
    store_fma!(2, acc2_lo, acc2_hi);
    store_fma!(3, acc3_lo, acc3_hi);
    store_fma!(4, acc4_lo, acc4_hi);
    store_fma!(5, acc5_lo, acc5_hi);
}
