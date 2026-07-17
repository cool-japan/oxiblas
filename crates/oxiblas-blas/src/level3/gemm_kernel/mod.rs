// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GEMM micro-kernels.
//!
//! Micro-kernels are the innermost computational routines that operate
//! on small MR × NR blocks. They are optimized for specific architectures
//! using SIMD intrinsics.

// Allow MSRV incompatibility for AVX-512 intrinsics that require Rust 1.89+
// We gate these with runtime CPU feature detection
#![allow(clippy::incompatible_msrv)]

use num_complex::{Complex32, Complex64};
use oxiblas_core::scalar::Field;
use oxiblas_core::simd::dispatch::GemmKernelKind;
#[cfg(target_arch = "x86_64")]
use super::gemm_kernel_sse42::{micro_kernel_f32_sse42, micro_kernel_f64_sse42};
/// Describes the shape of a micro-kernel.
#[derive(Debug, Clone, Copy)]
pub struct MicroKernelShape {
    /// Number of rows processed by the micro-kernel.
    pub mr: usize,
    /// Number of columns processed by the micro-kernel.
    pub nr: usize,
}
/// The single source of truth for which micro-kernel this build + CPU uses.
///
/// WHY this indirection exists: `micro_kernel_shape()` decides how the A and B
/// panels are packed, while `micro_kernel()` consumes those packed panels. The
/// two MUST agree on exactly which kernel runs. If they disagree — e.g. the
/// shape selects the 8×6 AVX2 layout but dispatch executes the 4×2 SSE4.2
/// kernel — the kernel strides through the packed data with the wrong `MR`/`NR`
/// and silently produces garbage. That is precisely the bug an AVX2-without-FMA
/// CPU used to hit: the shape selector only checked `avx2`, but dispatch
/// required `avx2 && fma`, so such a CPU got the 8×6 shape yet ran the 4×2
/// kernel. Deriving BOTH `micro_kernel_shape()` and `micro_kernel()` from this
/// one function makes that mismatch unrepresentable by construction.
///
/// It also honors the crate's SIMD-control features so they change the kernel
/// that actually executes, not merely the capability-reporting API:
/// `force-scalar` forces the portable kernel; `max-simd-128` / `max-simd-256`
/// clamp the SIMD width (see [`cap_kernel_kind`]).
#[inline]
fn selected_kernel_kind() -> GemmKernelKind {
    #[cfg(feature = "force-scalar")] { GemmKernelKind::Scalar }
    #[cfg(not(feature = "force-scalar"))]
    {
        let detected = oxiblas_core::simd::dispatch::KernelSelector::select()
            .gemm_f64_kernel;
        cap_kernel_kind(detected)
    }
}
/// Clamp a runtime-detected kernel kind to the compile-time SIMD width ceiling.
///
/// WHY: the `max-simd-128` / `max-simd-256` features must actually constrain the
/// executed kernel, not just what the capability API reports. Without this, a
/// build that opts into `max-simd-128` would still run 256-/512-bit kernels.
#[cfg(not(feature = "force-scalar"))]
#[inline]
fn cap_kernel_kind(kind: GemmKernelKind) -> GemmKernelKind {
    #[cfg(feature = "max-simd-128")]
    {
        match kind {
            GemmKernelKind::Avx512 | GemmKernelKind::Avx2 => {
                #[cfg(target_arch = "x86_64")] { GemmKernelKind::Sse42 }
                #[cfg(not(target_arch = "x86_64"))] { GemmKernelKind::Scalar }
            }
            other => other,
        }
    }
    #[cfg(all(feature = "max-simd-256", not(feature = "max-simd-128")))]
    {
        match kind {
            GemmKernelKind::Avx512 => GemmKernelKind::Avx2,
            other => other,
        }
    }
    #[cfg(not(any(feature = "max-simd-128", feature = "max-simd-256")))] { kind }
}
/// Trait for types that have GEMM micro-kernel implementations.
pub trait GemmKernel: Field {
    /// Returns the micro-kernel shape for this type.
    fn micro_kernel_shape() -> MicroKernelShape;
    /// Executes the micro-kernel.
    ///
    /// Computes: C = alpha * A * B + beta * C
    ///
    /// # Arguments
    ///
    /// * `k` - Number of iterations (columns of A, rows of B)
    /// * `alpha` - Scalar multiplier for A * B
    /// * `a` - Pointer to packed A data (MR × K, packed row-panel)
    /// * `b` - Pointer to packed B data (K × NR, packed column-panel)
    /// * `beta` - Scalar multiplier for C
    /// * `c` - Pointer to C matrix
    /// * `c_stride` - Row stride of C
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All pointers are valid and properly aligned
    /// - The arrays have sufficient length
    /// - No aliasing between A, B, and C
    unsafe fn micro_kernel(
        k: usize,
        alpha: Self,
        a: *const Self,
        b: *const Self,
        beta: Self,
        c: *mut Self,
        c_stride: usize,
    );
}
impl GemmKernel for f64 {
    fn micro_kernel_shape() -> MicroKernelShape {
        match selected_kernel_kind() {
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx512 => MicroKernelShape { mr: 16, nr: 6 },
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx2 => MicroKernelShape { mr: 8, nr: 6 },
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Sse42 => MicroKernelShape { mr: 4, nr: 2 },
            #[cfg(target_arch = "aarch64")]
            GemmKernelKind::Neon => MicroKernelShape { mr: 8, nr: 6 },
            _ => MicroKernelShape { mr: 4, nr: 4 },
        }
    }
    #[inline]
    unsafe fn micro_kernel(
        k: usize,
        alpha: Self,
        a: *const Self,
        b: *const Self,
        beta: Self,
        c: *mut Self,
        c_stride: usize,
    ) {
        match selected_kernel_kind() {
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx512 => {
                micro_kernel_f64_avx512(k, alpha, a, b, beta, c, c_stride)
            }
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx2 => {
                micro_kernel_f64_avx2(k, alpha, a, b, beta, c, c_stride)
            }
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Sse42 => {
                micro_kernel_f64_sse42(k, alpha, a, b, beta, c, c_stride)
            }
            #[cfg(target_arch = "aarch64")]
            GemmKernelKind::Neon => {
                micro_kernel_f64_neon(k, alpha, a, b, beta, c, c_stride)
            }
            _ => micro_kernel_f64_scalar(k, alpha, a, b, beta, c, c_stride),
        }
    }
}
/// Scalar micro-kernel for f64.
#[inline]
unsafe fn micro_kernel_f64_scalar(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    c_stride: usize,
) {
    const MR: usize = 4;
    const NR: usize = 4;
    let mut acc = [[0.0f64; NR]; MR];
    for p in 0..k {
        let a_ptr = a.add(p * MR);
        let b_ptr = b.add(p * NR);
        for i in 0..MR {
            let a_val = *a_ptr.add(i);
            for j in 0..NR {
                let b_val = *b_ptr.add(j);
                acc[i][j] = a_val.mul_add(b_val, acc[i][j]);
            }
        }
    }
    if beta == 0.0 {
        for j in 0..NR {
            for i in 0..MR {
                *c.add(i + j * c_stride) = alpha * acc[i][j];
            }
        }
    } else {
        for j in 0..NR {
            for i in 0..MR {
                let c_ptr = c.add(i + j * c_stride);
                let c_val = *c_ptr;
                *c_ptr = alpha.mul_add(acc[i][j], beta * c_val);
            }
        }
    }
}
/// AVX2 micro-kernel for f64 (8×6).
///
/// Uses 12 ymm accumulator registers (6 columns × 2 row groups).
/// Uses 2-way loop unrolling and software prefetching for optimal performance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_kernel_f64_avx2(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    c_stride: usize,
) {
    use core::arch::x86_64::*;
    const MR: usize = 8;
    const NR: usize = 6;
    const PREFETCH_DIST: usize = 12;
    let mut acc0_lo = _mm256_setzero_pd();
    let mut acc0_hi = _mm256_setzero_pd();
    let mut acc1_lo = _mm256_setzero_pd();
    let mut acc1_hi = _mm256_setzero_pd();
    let mut acc2_lo = _mm256_setzero_pd();
    let mut acc2_hi = _mm256_setzero_pd();
    let mut acc3_lo = _mm256_setzero_pd();
    let mut acc3_hi = _mm256_setzero_pd();
    let mut acc4_lo = _mm256_setzero_pd();
    let mut acc4_hi = _mm256_setzero_pd();
    let mut acc5_lo = _mm256_setzero_pd();
    let mut acc5_hi = _mm256_setzero_pd();
    macro_rules! fma_iter {
        ($offset:expr) => {
            { let a_ptr = a.add($offset * MR); let b_ptr = b.add($offset * NR); let a_lo
            = _mm256_loadu_pd(a_ptr); let a_hi = _mm256_loadu_pd(a_ptr.add(4)); let b0 =
            _mm256_broadcast_sd(&* b_ptr); acc0_lo = _mm256_fmadd_pd(a_lo, b0, acc0_lo);
            acc0_hi = _mm256_fmadd_pd(a_hi, b0, acc0_hi); let b1 = _mm256_broadcast_sd(&*
            b_ptr.add(1)); acc1_lo = _mm256_fmadd_pd(a_lo, b1, acc1_lo); acc1_hi =
            _mm256_fmadd_pd(a_hi, b1, acc1_hi); let b2 = _mm256_broadcast_sd(&* b_ptr
            .add(2)); acc2_lo = _mm256_fmadd_pd(a_lo, b2, acc2_lo); acc2_hi =
            _mm256_fmadd_pd(a_hi, b2, acc2_hi); let b3 = _mm256_broadcast_sd(&* b_ptr
            .add(3)); acc3_lo = _mm256_fmadd_pd(a_lo, b3, acc3_lo); acc3_hi =
            _mm256_fmadd_pd(a_hi, b3, acc3_hi); let b4 = _mm256_broadcast_sd(&* b_ptr
            .add(4)); acc4_lo = _mm256_fmadd_pd(a_lo, b4, acc4_lo); acc4_hi =
            _mm256_fmadd_pd(a_hi, b4, acc4_hi); let b5 = _mm256_broadcast_sd(&* b_ptr
            .add(5)); acc5_lo = _mm256_fmadd_pd(a_lo, b5, acc5_lo); acc5_hi =
            _mm256_fmadd_pd(a_hi, b5, acc5_hi); }
        };
    }
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            _mm_prefetch(a.add(pf_base * MR) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a.add(pf_base * MR + 32) as *const i8, _MM_HINT_T0);
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
        let alpha_vec = _mm256_set1_pd(alpha);
        acc0_lo = _mm256_mul_pd(acc0_lo, alpha_vec);
        acc0_hi = _mm256_mul_pd(acc0_hi, alpha_vec);
        acc1_lo = _mm256_mul_pd(acc1_lo, alpha_vec);
        acc1_hi = _mm256_mul_pd(acc1_hi, alpha_vec);
        acc2_lo = _mm256_mul_pd(acc2_lo, alpha_vec);
        acc2_hi = _mm256_mul_pd(acc2_hi, alpha_vec);
        acc3_lo = _mm256_mul_pd(acc3_lo, alpha_vec);
        acc3_hi = _mm256_mul_pd(acc3_hi, alpha_vec);
        acc4_lo = _mm256_mul_pd(acc4_lo, alpha_vec);
        acc4_hi = _mm256_mul_pd(acc4_hi, alpha_vec);
        acc5_lo = _mm256_mul_pd(acc5_lo, alpha_vec);
        acc5_hi = _mm256_mul_pd(acc5_hi, alpha_vec);
    }
    if beta == 0.0 {
        _mm256_storeu_pd(c, acc0_lo);
        _mm256_storeu_pd(c.add(4), acc0_hi);
        _mm256_storeu_pd(c.add(c_stride), acc1_lo);
        _mm256_storeu_pd(c.add(c_stride + 4), acc1_hi);
        _mm256_storeu_pd(c.add(2 * c_stride), acc2_lo);
        _mm256_storeu_pd(c.add(2 * c_stride + 4), acc2_hi);
        _mm256_storeu_pd(c.add(3 * c_stride), acc3_lo);
        _mm256_storeu_pd(c.add(3 * c_stride + 4), acc3_hi);
        _mm256_storeu_pd(c.add(4 * c_stride), acc4_lo);
        _mm256_storeu_pd(c.add(4 * c_stride + 4), acc4_hi);
        _mm256_storeu_pd(c.add(5 * c_stride), acc5_lo);
        _mm256_storeu_pd(c.add(5 * c_stride + 4), acc5_hi);
    } else if beta == 1.0 {
        macro_rules! store_add {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
                { let c_col = c.add($col * c_stride); _mm256_storeu_pd(c_col,
                _mm256_add_pd($acc_lo, _mm256_loadu_pd(c_col))); _mm256_storeu_pd(c_col
                .add(4), _mm256_add_pd($acc_hi, _mm256_loadu_pd(c_col.add(4))),); }
            };
        }
        store_add!(0, acc0_lo, acc0_hi);
        store_add!(1, acc1_lo, acc1_hi);
        store_add!(2, acc2_lo, acc2_hi);
        store_add!(3, acc3_lo, acc3_hi);
        store_add!(4, acc4_lo, acc4_hi);
        store_add!(5, acc5_lo, acc5_hi);
    } else {
        let beta_vec = _mm256_set1_pd(beta);
        macro_rules! store_fma {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
                { let c_col = c.add($col * c_stride); let c_lo = _mm256_loadu_pd(c_col);
                let c_hi = _mm256_loadu_pd(c_col.add(4)); _mm256_storeu_pd(c_col,
                _mm256_fmadd_pd(c_lo, beta_vec, $acc_lo)); _mm256_storeu_pd(c_col.add(4),
                _mm256_fmadd_pd(c_hi, beta_vec, $acc_hi)); }
            };
        }
        store_fma!(0, acc0_lo, acc0_hi);
        store_fma!(1, acc1_lo, acc1_hi);
        store_fma!(2, acc2_lo, acc2_hi);
        store_fma!(3, acc3_lo, acc3_hi);
        store_fma!(4, acc4_lo, acc4_hi);
        store_fma!(5, acc5_lo, acc5_hi);
    }
}
/// AVX-512 micro-kernel for f64 (16×6).
///
/// Uses 12 zmm registers for accumulators (6 columns × 2 row groups).
/// Each zmm holds 8 f64, so 16 rows per column.
/// Optimized with 4-way unrolling and software prefetching.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn micro_kernel_f64_avx512(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    c_stride: usize,
) {
    use core::arch::x86_64::*;
    const MR: usize = 16;
    const NR: usize = 6;
    const PREFETCH_DIST: usize = 16;
    let mut acc0_lo = _mm512_setzero_pd();
    let mut acc0_hi = _mm512_setzero_pd();
    let mut acc1_lo = _mm512_setzero_pd();
    let mut acc1_hi = _mm512_setzero_pd();
    let mut acc2_lo = _mm512_setzero_pd();
    let mut acc2_hi = _mm512_setzero_pd();
    let mut acc3_lo = _mm512_setzero_pd();
    let mut acc3_hi = _mm512_setzero_pd();
    let mut acc4_lo = _mm512_setzero_pd();
    let mut acc4_hi = _mm512_setzero_pd();
    let mut acc5_lo = _mm512_setzero_pd();
    let mut acc5_hi = _mm512_setzero_pd();
    macro_rules! fma_iter {
        ($offset:expr) => {
            { let a_ptr = a.add($offset * MR); let b_ptr = b.add($offset * NR); let a_lo
            = _mm512_loadu_pd(a_ptr); let a_hi = _mm512_loadu_pd(a_ptr.add(8)); let b0 =
            _mm512_set1_pd(* b_ptr); acc0_lo = _mm512_fmadd_pd(a_lo, b0, acc0_lo);
            acc0_hi = _mm512_fmadd_pd(a_hi, b0, acc0_hi); let b1 = _mm512_set1_pd(* b_ptr
            .add(1)); acc1_lo = _mm512_fmadd_pd(a_lo, b1, acc1_lo); acc1_hi =
            _mm512_fmadd_pd(a_hi, b1, acc1_hi); let b2 = _mm512_set1_pd(* b_ptr.add(2));
            acc2_lo = _mm512_fmadd_pd(a_lo, b2, acc2_lo); acc2_hi = _mm512_fmadd_pd(a_hi,
            b2, acc2_hi); let b3 = _mm512_set1_pd(* b_ptr.add(3)); acc3_lo =
            _mm512_fmadd_pd(a_lo, b3, acc3_lo); acc3_hi = _mm512_fmadd_pd(a_hi, b3,
            acc3_hi); let b4 = _mm512_set1_pd(* b_ptr.add(4)); acc4_lo =
            _mm512_fmadd_pd(a_lo, b4, acc4_lo); acc4_hi = _mm512_fmadd_pd(a_hi, b4,
            acc4_hi); let b5 = _mm512_set1_pd(* b_ptr.add(5)); acc5_lo =
            _mm512_fmadd_pd(a_lo, b5, acc5_lo); acc5_hi = _mm512_fmadd_pd(a_hi, b5,
            acc5_hi); }
        };
    }
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            _mm_prefetch(a.add(pf_base * MR) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a.add(pf_base * MR + 64) as *const i8, _MM_HINT_T0);
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
        let alpha_vec = _mm512_set1_pd(alpha);
        acc0_lo = _mm512_mul_pd(acc0_lo, alpha_vec);
        acc0_hi = _mm512_mul_pd(acc0_hi, alpha_vec);
        acc1_lo = _mm512_mul_pd(acc1_lo, alpha_vec);
        acc1_hi = _mm512_mul_pd(acc1_hi, alpha_vec);
        acc2_lo = _mm512_mul_pd(acc2_lo, alpha_vec);
        acc2_hi = _mm512_mul_pd(acc2_hi, alpha_vec);
        acc3_lo = _mm512_mul_pd(acc3_lo, alpha_vec);
        acc3_hi = _mm512_mul_pd(acc3_hi, alpha_vec);
        acc4_lo = _mm512_mul_pd(acc4_lo, alpha_vec);
        acc4_hi = _mm512_mul_pd(acc4_hi, alpha_vec);
        acc5_lo = _mm512_mul_pd(acc5_lo, alpha_vec);
        acc5_hi = _mm512_mul_pd(acc5_hi, alpha_vec);
    }
    if beta == 0.0 {
        _mm512_storeu_pd(c, acc0_lo);
        _mm512_storeu_pd(c.add(8), acc0_hi);
        _mm512_storeu_pd(c.add(c_stride), acc1_lo);
        _mm512_storeu_pd(c.add(c_stride + 8), acc1_hi);
        _mm512_storeu_pd(c.add(2 * c_stride), acc2_lo);
        _mm512_storeu_pd(c.add(2 * c_stride + 8), acc2_hi);
        _mm512_storeu_pd(c.add(3 * c_stride), acc3_lo);
        _mm512_storeu_pd(c.add(3 * c_stride + 8), acc3_hi);
        _mm512_storeu_pd(c.add(4 * c_stride), acc4_lo);
        _mm512_storeu_pd(c.add(4 * c_stride + 8), acc4_hi);
        _mm512_storeu_pd(c.add(5 * c_stride), acc5_lo);
        _mm512_storeu_pd(c.add(5 * c_stride + 8), acc5_hi);
    } else if beta == 1.0 {
        macro_rules! store_add {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
                { let c_col = c.add($col * c_stride); _mm512_storeu_pd(c_col,
                _mm512_add_pd($acc_lo, _mm512_loadu_pd(c_col))); _mm512_storeu_pd(c_col
                .add(8), _mm512_add_pd($acc_hi, _mm512_loadu_pd(c_col.add(8))),); }
            };
        }
        store_add!(0, acc0_lo, acc0_hi);
        store_add!(1, acc1_lo, acc1_hi);
        store_add!(2, acc2_lo, acc2_hi);
        store_add!(3, acc3_lo, acc3_hi);
        store_add!(4, acc4_lo, acc4_hi);
        store_add!(5, acc5_lo, acc5_hi);
    } else {
        let beta_vec = _mm512_set1_pd(beta);
        macro_rules! store_fma {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
                { let c_col = c.add($col * c_stride); let c_lo = _mm512_loadu_pd(c_col);
                let c_hi = _mm512_loadu_pd(c_col.add(8)); _mm512_storeu_pd(c_col,
                _mm512_fmadd_pd(c_lo, beta_vec, $acc_lo)); _mm512_storeu_pd(c_col.add(8),
                _mm512_fmadd_pd(c_hi, beta_vec, $acc_hi)); }
            };
        }
        store_fma!(0, acc0_lo, acc0_hi);
        store_fma!(1, acc1_lo, acc1_hi);
        store_fma!(2, acc2_lo, acc2_hi);
        store_fma!(3, acc3_lo, acc3_hi);
        store_fma!(4, acc4_lo, acc4_hi);
        store_fma!(5, acc5_lo, acc5_hi);
    }
}
/// Software prefetch for aarch64.
/// Uses PRFM instruction to prefetch data to L1 cache.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_read<T>(ptr: *const T) {
    core::arch::asm!(
        "prfm pldl1keep, [{ptr}]", ptr = in (reg) ptr, options(nostack, preserves_flags)
    );
}
/// NEON micro-kernel for f64 (8×6).
///
/// Optimized for Apple Silicon using 24 NEON registers for accumulators.
/// Uses 4-way loop unrolling and aggressive software prefetching.
/// Apple Silicon M1/M2/M3 have 4 FMLA units and large L2 cache.
#[cfg(target_arch = "aarch64")]
unsafe fn micro_kernel_f64_neon(
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    c_stride: usize,
) {
    use core::arch::aarch64::{
        vaddq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64, vmulq_f64, vst1q_f64,
    };
    const MR: usize = 8;
    const NR: usize = 6;
    const PREFETCH_DIST: usize = 10;
    let mut acc00 = vdupq_n_f64(0.0);
    let mut acc01 = vdupq_n_f64(0.0);
    let mut acc02 = vdupq_n_f64(0.0);
    let mut acc03 = vdupq_n_f64(0.0);
    let mut acc10 = vdupq_n_f64(0.0);
    let mut acc11 = vdupq_n_f64(0.0);
    let mut acc12 = vdupq_n_f64(0.0);
    let mut acc13 = vdupq_n_f64(0.0);
    let mut acc20 = vdupq_n_f64(0.0);
    let mut acc21 = vdupq_n_f64(0.0);
    let mut acc22 = vdupq_n_f64(0.0);
    let mut acc23 = vdupq_n_f64(0.0);
    let mut acc30 = vdupq_n_f64(0.0);
    let mut acc31 = vdupq_n_f64(0.0);
    let mut acc32 = vdupq_n_f64(0.0);
    let mut acc33 = vdupq_n_f64(0.0);
    let mut acc40 = vdupq_n_f64(0.0);
    let mut acc41 = vdupq_n_f64(0.0);
    let mut acc42 = vdupq_n_f64(0.0);
    let mut acc43 = vdupq_n_f64(0.0);
    let mut acc50 = vdupq_n_f64(0.0);
    let mut acc51 = vdupq_n_f64(0.0);
    let mut acc52 = vdupq_n_f64(0.0);
    let mut acc53 = vdupq_n_f64(0.0);
    macro_rules! fma_iter {
        ($offset:expr) => {
            { let a_ptr = a.add($offset * MR); let b_ptr = b.add($offset * NR); let a0 =
            vld1q_f64(a_ptr); let a1 = vld1q_f64(a_ptr.add(2)); let a2 = vld1q_f64(a_ptr
            .add(4)); let a3 = vld1q_f64(a_ptr.add(6)); let b0 = vdupq_n_f64(* b_ptr);
            acc00 = vfmaq_f64(acc00, a0, b0); acc01 = vfmaq_f64(acc01, a1, b0); let b1 =
            vdupq_n_f64(* b_ptr.add(1)); acc02 = vfmaq_f64(acc02, a2, b0); acc03 =
            vfmaq_f64(acc03, a3, b0); acc10 = vfmaq_f64(acc10, a0, b1); acc11 =
            vfmaq_f64(acc11, a1, b1); let b2 = vdupq_n_f64(* b_ptr.add(2)); acc12 =
            vfmaq_f64(acc12, a2, b1); acc13 = vfmaq_f64(acc13, a3, b1); acc20 =
            vfmaq_f64(acc20, a0, b2); acc21 = vfmaq_f64(acc21, a1, b2); let b3 =
            vdupq_n_f64(* b_ptr.add(3)); acc22 = vfmaq_f64(acc22, a2, b2); acc23 =
            vfmaq_f64(acc23, a3, b2); acc30 = vfmaq_f64(acc30, a0, b3); acc31 =
            vfmaq_f64(acc31, a1, b3); let b4 = vdupq_n_f64(* b_ptr.add(4)); acc32 =
            vfmaq_f64(acc32, a2, b3); acc33 = vfmaq_f64(acc33, a3, b3); acc40 =
            vfmaq_f64(acc40, a0, b4); acc41 = vfmaq_f64(acc41, a1, b4); let b5 =
            vdupq_n_f64(* b_ptr.add(5)); acc42 = vfmaq_f64(acc42, a2, b4); acc43 =
            vfmaq_f64(acc43, a3, b4); acc50 = vfmaq_f64(acc50, a0, b5); acc51 =
            vfmaq_f64(acc51, a1, b5); acc52 = vfmaq_f64(acc52, a2, b5); acc53 =
            vfmaq_f64(acc53, a3, b5); }
        };
    }
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            prefetch_read(a.add(pf_base * MR));
            prefetch_read(a.add(pf_base * MR + 32));
            prefetch_read(b.add(pf_base * NR));
            prefetch_read(b.add(pf_base * NR + 16));
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
        let alpha_vec = vdupq_n_f64(alpha);
        acc00 = vmulq_f64(acc00, alpha_vec);
        acc01 = vmulq_f64(acc01, alpha_vec);
        acc02 = vmulq_f64(acc02, alpha_vec);
        acc03 = vmulq_f64(acc03, alpha_vec);
        acc10 = vmulq_f64(acc10, alpha_vec);
        acc11 = vmulq_f64(acc11, alpha_vec);
        acc12 = vmulq_f64(acc12, alpha_vec);
        acc13 = vmulq_f64(acc13, alpha_vec);
        acc20 = vmulq_f64(acc20, alpha_vec);
        acc21 = vmulq_f64(acc21, alpha_vec);
        acc22 = vmulq_f64(acc22, alpha_vec);
        acc23 = vmulq_f64(acc23, alpha_vec);
        acc30 = vmulq_f64(acc30, alpha_vec);
        acc31 = vmulq_f64(acc31, alpha_vec);
        acc32 = vmulq_f64(acc32, alpha_vec);
        acc33 = vmulq_f64(acc33, alpha_vec);
        acc40 = vmulq_f64(acc40, alpha_vec);
        acc41 = vmulq_f64(acc41, alpha_vec);
        acc42 = vmulq_f64(acc42, alpha_vec);
        acc43 = vmulq_f64(acc43, alpha_vec);
        acc50 = vmulq_f64(acc50, alpha_vec);
        acc51 = vmulq_f64(acc51, alpha_vec);
        acc52 = vmulq_f64(acc52, alpha_vec);
        acc53 = vmulq_f64(acc53, alpha_vec);
    }
    macro_rules! store_column {
        ($col:expr, $acc0:expr, $acc1:expr, $acc2:expr, $acc3:expr) => {
            { let c_col = c.add($col * c_stride); vst1q_f64(c_col, $acc0);
            vst1q_f64(c_col.add(2), $acc1); vst1q_f64(c_col.add(4), $acc2);
            vst1q_f64(c_col.add(6), $acc3); }
        };
    }
    macro_rules! store_column_add {
        ($col:expr, $acc0:expr, $acc1:expr, $acc2:expr, $acc3:expr) => {
            { let c_col = c.add($col * c_stride); vst1q_f64(c_col, vaddq_f64($acc0,
            vld1q_f64(c_col))); vst1q_f64(c_col.add(2), vaddq_f64($acc1, vld1q_f64(c_col
            .add(2)))); vst1q_f64(c_col.add(4), vaddq_f64($acc2, vld1q_f64(c_col
            .add(4)))); vst1q_f64(c_col.add(6), vaddq_f64($acc3, vld1q_f64(c_col
            .add(6)))); }
        };
    }
    macro_rules! store_column_fma {
        ($col:expr, $acc0:expr, $acc1:expr, $acc2:expr, $acc3:expr, $beta_vec:expr) => {
            { let c_col = c.add($col * c_stride); let c0 = vld1q_f64(c_col); let c1 =
            vld1q_f64(c_col.add(2)); let c2 = vld1q_f64(c_col.add(4)); let c3 =
            vld1q_f64(c_col.add(6)); vst1q_f64(c_col, vfmaq_f64($acc0, c0, $beta_vec));
            vst1q_f64(c_col.add(2), vfmaq_f64($acc1, c1, $beta_vec)); vst1q_f64(c_col
            .add(4), vfmaq_f64($acc2, c2, $beta_vec)); vst1q_f64(c_col.add(6),
            vfmaq_f64($acc3, c3, $beta_vec)); }
        };
    }
    if beta == 0.0 {
        store_column!(0, acc00, acc01, acc02, acc03);
        store_column!(1, acc10, acc11, acc12, acc13);
        store_column!(2, acc20, acc21, acc22, acc23);
        store_column!(3, acc30, acc31, acc32, acc33);
        store_column!(4, acc40, acc41, acc42, acc43);
        store_column!(5, acc50, acc51, acc52, acc53);
    } else if beta == 1.0 {
        store_column_add!(0, acc00, acc01, acc02, acc03);
        store_column_add!(1, acc10, acc11, acc12, acc13);
        store_column_add!(2, acc20, acc21, acc22, acc23);
        store_column_add!(3, acc30, acc31, acc32, acc33);
        store_column_add!(4, acc40, acc41, acc42, acc43);
        store_column_add!(5, acc50, acc51, acc52, acc53);
    } else {
        let beta_vec = vdupq_n_f64(beta);
        store_column_fma!(0, acc00, acc01, acc02, acc03, beta_vec);
        store_column_fma!(1, acc10, acc11, acc12, acc13, beta_vec);
        store_column_fma!(2, acc20, acc21, acc22, acc23, beta_vec);
        store_column_fma!(3, acc30, acc31, acc32, acc33, beta_vec);
        store_column_fma!(4, acc40, acc41, acc42, acc43, beta_vec);
        store_column_fma!(5, acc50, acc51, acc52, acc53, beta_vec);
    }
}
impl GemmKernel for f32 {
    fn micro_kernel_shape() -> MicroKernelShape {
        match selected_kernel_kind() {
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx512 => MicroKernelShape { mr: 16, nr: 16 },
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx2 => MicroKernelShape { mr: 8, nr: 8 },
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Sse42 => MicroKernelShape { mr: 4, nr: 4 },
            #[cfg(target_arch = "aarch64")]
            GemmKernelKind::Neon => MicroKernelShape { mr: 8, nr: 8 },
            _ => MicroKernelShape { mr: 4, nr: 4 },
        }
    }
    #[inline]
    unsafe fn micro_kernel(
        k: usize,
        alpha: Self,
        a: *const Self,
        b: *const Self,
        beta: Self,
        c: *mut Self,
        c_stride: usize,
    ) {
        match selected_kernel_kind() {
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx512 => {
                micro_kernel_f32_avx512(k, alpha, a, b, beta, c, c_stride)
            }
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx2 => {
                micro_kernel_f32_avx2(k, alpha, a, b, beta, c, c_stride)
            }
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Sse42 => {
                micro_kernel_f32_sse42(k, alpha, a, b, beta, c, c_stride)
            }
            #[cfg(target_arch = "aarch64")]
            GemmKernelKind::Neon => {
                micro_kernel_f32_neon(k, alpha, a, b, beta, c, c_stride)
            }
            _ => micro_kernel_f32_scalar(k, alpha, a, b, beta, c, c_stride),
        }
    }
}
/// Scalar micro-kernel for f32.
#[inline]
unsafe fn micro_kernel_f32_scalar(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    const MR: usize = 4;
    const NR: usize = 4;
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..k {
        let a_ptr = a.add(p * MR);
        let b_ptr = b.add(p * NR);
        for i in 0..MR {
            let a_val = *a_ptr.add(i);
            for j in 0..NR {
                let b_val = *b_ptr.add(j);
                acc[i][j] = a_val.mul_add(b_val, acc[i][j]);
            }
        }
    }
    if beta == 0.0 {
        for j in 0..NR {
            for i in 0..MR {
                *c.add(i + j * c_stride) = alpha * acc[i][j];
            }
        }
    } else {
        for j in 0..NR {
            for i in 0..MR {
                let c_ptr = c.add(i + j * c_stride);
                let c_val = *c_ptr;
                *c_ptr = alpha.mul_add(acc[i][j], beta * c_val);
            }
        }
    }
}
/// AVX2 micro-kernel for f32 (8×8).
///
/// Uses 8 ymm accumulator registers with 4-way loop unrolling and software prefetching.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_kernel_f32_avx2(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    use core::arch::x86_64::*;
    const MR: usize = 8;
    const NR: usize = 8;
    const PREFETCH_DIST: usize = 12;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    macro_rules! fma_iter {
        ($offset:expr) => {
            { let a_ptr = a.add($offset * MR); let b_ptr = b.add($offset * NR); let a_vec
            = _mm256_loadu_ps(a_ptr); let b0 = _mm256_broadcast_ss(&* b_ptr); acc0 =
            _mm256_fmadd_ps(a_vec, b0, acc0); let b1 = _mm256_broadcast_ss(&* b_ptr
            .add(1)); acc1 = _mm256_fmadd_ps(a_vec, b1, acc1); let b2 =
            _mm256_broadcast_ss(&* b_ptr.add(2)); acc2 = _mm256_fmadd_ps(a_vec, b2,
            acc2); let b3 = _mm256_broadcast_ss(&* b_ptr.add(3)); acc3 =
            _mm256_fmadd_ps(a_vec, b3, acc3); let b4 = _mm256_broadcast_ss(&* b_ptr
            .add(4)); acc4 = _mm256_fmadd_ps(a_vec, b4, acc4); let b5 =
            _mm256_broadcast_ss(&* b_ptr.add(5)); acc5 = _mm256_fmadd_ps(a_vec, b5,
            acc5); let b6 = _mm256_broadcast_ss(&* b_ptr.add(6)); acc6 =
            _mm256_fmadd_ps(a_vec, b6, acc6); let b7 = _mm256_broadcast_ss(&* b_ptr
            .add(7)); acc7 = _mm256_fmadd_ps(a_vec, b7, acc7); }
        };
    }
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            _mm_prefetch(a.add(pf_base * MR) as *const i8, _MM_HINT_T0);
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
        acc0 = _mm256_mul_ps(acc0, alpha_vec);
        acc1 = _mm256_mul_ps(acc1, alpha_vec);
        acc2 = _mm256_mul_ps(acc2, alpha_vec);
        acc3 = _mm256_mul_ps(acc3, alpha_vec);
        acc4 = _mm256_mul_ps(acc4, alpha_vec);
        acc5 = _mm256_mul_ps(acc5, alpha_vec);
        acc6 = _mm256_mul_ps(acc6, alpha_vec);
        acc7 = _mm256_mul_ps(acc7, alpha_vec);
    }
    if beta == 0.0 {
        _mm256_storeu_ps(c, acc0);
        _mm256_storeu_ps(c.add(c_stride), acc1);
        _mm256_storeu_ps(c.add(2 * c_stride), acc2);
        _mm256_storeu_ps(c.add(3 * c_stride), acc3);
        _mm256_storeu_ps(c.add(4 * c_stride), acc4);
        _mm256_storeu_ps(c.add(5 * c_stride), acc5);
        _mm256_storeu_ps(c.add(6 * c_stride), acc6);
        _mm256_storeu_ps(c.add(7 * c_stride), acc7);
        return;
    }
    if beta == 1.0 {
        macro_rules! store_add {
            ($col:expr, $acc:expr) => {
                { let c_col = c.add($col * c_stride); _mm256_storeu_ps(c_col,
                _mm256_add_ps($acc, _mm256_loadu_ps(c_col))); }
            };
        }
        store_add!(0, acc0);
        store_add!(1, acc1);
        store_add!(2, acc2);
        store_add!(3, acc3);
        store_add!(4, acc4);
        store_add!(5, acc5);
        store_add!(6, acc6);
        store_add!(7, acc7);
        return;
    }
    let beta_vec = _mm256_set1_ps(beta);
    macro_rules! store_col {
        ($col:expr, $acc:expr) => {
            { let c_col = c.add($col * c_stride); let c_vec = _mm256_loadu_ps(c_col); let
            res = _mm256_fmadd_ps(c_vec, beta_vec, $acc); _mm256_storeu_ps(c_col, res); }
        };
    }
    store_col!(0, acc0);
    store_col!(1, acc1);
    store_col!(2, acc2);
    store_col!(3, acc3);
    store_col!(4, acc4);
    store_col!(5, acc5);
    store_col!(6, acc6);
    store_col!(7, acc7);
}
/// AVX-512 micro-kernel for f32 (16×16).
///
/// Uses 16 zmm registers for accumulators (16 columns, each holding 16 rows).
/// Each zmm holds 16 f32 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn micro_kernel_f32_avx512(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    use core::arch::x86_64::*;
    const MR: usize = 16;
    const NR: usize = 16;
    const PREFETCH_DIST: usize = 8;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();
    let mut acc9 = _mm512_setzero_ps();
    let mut acc10 = _mm512_setzero_ps();
    let mut acc11 = _mm512_setzero_ps();
    let mut acc12 = _mm512_setzero_ps();
    let mut acc13 = _mm512_setzero_ps();
    let mut acc14 = _mm512_setzero_ps();
    let mut acc15 = _mm512_setzero_ps();
    macro_rules! fma_iter {
        ($offset:expr) => {
            { let a_ptr = a.add($offset * MR); let b_ptr = b.add($offset * NR); let a_vec
            = _mm512_loadu_ps(a_ptr); let b0 = _mm512_set1_ps(* b_ptr); acc0 =
            _mm512_fmadd_ps(a_vec, b0, acc0); let b1 = _mm512_set1_ps(* b_ptr.add(1));
            acc1 = _mm512_fmadd_ps(a_vec, b1, acc1); let b2 = _mm512_set1_ps(* b_ptr
            .add(2)); acc2 = _mm512_fmadd_ps(a_vec, b2, acc2); let b3 = _mm512_set1_ps(*
            b_ptr.add(3)); acc3 = _mm512_fmadd_ps(a_vec, b3, acc3); let b4 =
            _mm512_set1_ps(* b_ptr.add(4)); acc4 = _mm512_fmadd_ps(a_vec, b4, acc4); let
            b5 = _mm512_set1_ps(* b_ptr.add(5)); acc5 = _mm512_fmadd_ps(a_vec, b5, acc5);
            let b6 = _mm512_set1_ps(* b_ptr.add(6)); acc6 = _mm512_fmadd_ps(a_vec, b6,
            acc6); let b7 = _mm512_set1_ps(* b_ptr.add(7)); acc7 = _mm512_fmadd_ps(a_vec,
            b7, acc7); let b8 = _mm512_set1_ps(* b_ptr.add(8)); acc8 =
            _mm512_fmadd_ps(a_vec, b8, acc8); let b9 = _mm512_set1_ps(* b_ptr.add(9));
            acc9 = _mm512_fmadd_ps(a_vec, b9, acc9); let b10 = _mm512_set1_ps(* b_ptr
            .add(10)); acc10 = _mm512_fmadd_ps(a_vec, b10, acc10); let b11 =
            _mm512_set1_ps(* b_ptr.add(11)); acc11 = _mm512_fmadd_ps(a_vec, b11, acc11);
            let b12 = _mm512_set1_ps(* b_ptr.add(12)); acc12 = _mm512_fmadd_ps(a_vec,
            b12, acc12); let b13 = _mm512_set1_ps(* b_ptr.add(13)); acc13 =
            _mm512_fmadd_ps(a_vec, b13, acc13); let b14 = _mm512_set1_ps(* b_ptr
            .add(14)); acc14 = _mm512_fmadd_ps(a_vec, b14, acc14); let b15 =
            _mm512_set1_ps(* b_ptr.add(15)); acc15 = _mm512_fmadd_ps(a_vec, b15, acc15);
            }
        };
    }
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            _mm_prefetch(a.add(pf_base * MR) as *const i8, _MM_HINT_T0);
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
        let alpha_vec = _mm512_set1_ps(alpha);
        acc0 = _mm512_mul_ps(acc0, alpha_vec);
        acc1 = _mm512_mul_ps(acc1, alpha_vec);
        acc2 = _mm512_mul_ps(acc2, alpha_vec);
        acc3 = _mm512_mul_ps(acc3, alpha_vec);
        acc4 = _mm512_mul_ps(acc4, alpha_vec);
        acc5 = _mm512_mul_ps(acc5, alpha_vec);
        acc6 = _mm512_mul_ps(acc6, alpha_vec);
        acc7 = _mm512_mul_ps(acc7, alpha_vec);
        acc8 = _mm512_mul_ps(acc8, alpha_vec);
        acc9 = _mm512_mul_ps(acc9, alpha_vec);
        acc10 = _mm512_mul_ps(acc10, alpha_vec);
        acc11 = _mm512_mul_ps(acc11, alpha_vec);
        acc12 = _mm512_mul_ps(acc12, alpha_vec);
        acc13 = _mm512_mul_ps(acc13, alpha_vec);
        acc14 = _mm512_mul_ps(acc14, alpha_vec);
        acc15 = _mm512_mul_ps(acc15, alpha_vec);
    }
    if beta == 0.0 {
        _mm512_storeu_ps(c, acc0);
        _mm512_storeu_ps(c.add(c_stride), acc1);
        _mm512_storeu_ps(c.add(2 * c_stride), acc2);
        _mm512_storeu_ps(c.add(3 * c_stride), acc3);
        _mm512_storeu_ps(c.add(4 * c_stride), acc4);
        _mm512_storeu_ps(c.add(5 * c_stride), acc5);
        _mm512_storeu_ps(c.add(6 * c_stride), acc6);
        _mm512_storeu_ps(c.add(7 * c_stride), acc7);
        _mm512_storeu_ps(c.add(8 * c_stride), acc8);
        _mm512_storeu_ps(c.add(9 * c_stride), acc9);
        _mm512_storeu_ps(c.add(10 * c_stride), acc10);
        _mm512_storeu_ps(c.add(11 * c_stride), acc11);
        _mm512_storeu_ps(c.add(12 * c_stride), acc12);
        _mm512_storeu_ps(c.add(13 * c_stride), acc13);
        _mm512_storeu_ps(c.add(14 * c_stride), acc14);
        _mm512_storeu_ps(c.add(15 * c_stride), acc15);
    } else if beta == 1.0 {
        macro_rules! store_add {
            ($col:expr, $acc:expr) => {
                { let c_col = c.add($col * c_stride); _mm512_storeu_ps(c_col,
                _mm512_add_ps($acc, _mm512_loadu_ps(c_col))); }
            };
        }
        store_add!(0, acc0);
        store_add!(1, acc1);
        store_add!(2, acc2);
        store_add!(3, acc3);
        store_add!(4, acc4);
        store_add!(5, acc5);
        store_add!(6, acc6);
        store_add!(7, acc7);
        store_add!(8, acc8);
        store_add!(9, acc9);
        store_add!(10, acc10);
        store_add!(11, acc11);
        store_add!(12, acc12);
        store_add!(13, acc13);
        store_add!(14, acc14);
        store_add!(15, acc15);
    } else {
        let beta_vec = _mm512_set1_ps(beta);
        macro_rules! store_fma {
            ($col:expr, $acc:expr) => {
                { let c_col = c.add($col * c_stride); let c_vec = _mm512_loadu_ps(c_col);
                _mm512_storeu_ps(c_col, _mm512_fmadd_ps(c_vec, beta_vec, $acc)); }
            };
        }
        store_fma!(0, acc0);
        store_fma!(1, acc1);
        store_fma!(2, acc2);
        store_fma!(3, acc3);
        store_fma!(4, acc4);
        store_fma!(5, acc5);
        store_fma!(6, acc6);
        store_fma!(7, acc7);
        store_fma!(8, acc8);
        store_fma!(9, acc9);
        store_fma!(10, acc10);
        store_fma!(11, acc11);
        store_fma!(12, acc12);
        store_fma!(13, acc13);
        store_fma!(14, acc14);
        store_fma!(15, acc15);
    }
}
/// NEON micro-kernel for f32 (8×8).
///
/// Uses 4-way loop unrolling and aggressive software prefetching.
#[cfg(target_arch = "aarch64")]
unsafe fn micro_kernel_f32_neon(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    c_stride: usize,
) {
    use core::arch::aarch64::{
        vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
    };
    const MR: usize = 8;
    const NR: usize = 8;
    const PREFETCH_DIST: usize = 10;
    let mut acc0_lo = vdupq_n_f32(0.0);
    let mut acc0_hi = vdupq_n_f32(0.0);
    let mut acc1_lo = vdupq_n_f32(0.0);
    let mut acc1_hi = vdupq_n_f32(0.0);
    let mut acc2_lo = vdupq_n_f32(0.0);
    let mut acc2_hi = vdupq_n_f32(0.0);
    let mut acc3_lo = vdupq_n_f32(0.0);
    let mut acc3_hi = vdupq_n_f32(0.0);
    let mut acc4_lo = vdupq_n_f32(0.0);
    let mut acc4_hi = vdupq_n_f32(0.0);
    let mut acc5_lo = vdupq_n_f32(0.0);
    let mut acc5_hi = vdupq_n_f32(0.0);
    let mut acc6_lo = vdupq_n_f32(0.0);
    let mut acc6_hi = vdupq_n_f32(0.0);
    let mut acc7_lo = vdupq_n_f32(0.0);
    let mut acc7_hi = vdupq_n_f32(0.0);
    let k_unroll = k / 4;
    let k_remainder = k % 4;
    macro_rules! fma_iteration {
        ($a_ptr:expr, $b_ptr:expr) => {
            { let a_lo = vld1q_f32($a_ptr); let a_hi = vld1q_f32($a_ptr .add(4)); let b0
            = vdupq_n_f32(*$b_ptr); acc0_lo = vfmaq_f32(acc0_lo, a_lo, b0); acc0_hi =
            vfmaq_f32(acc0_hi, a_hi, b0); let b1 = vdupq_n_f32(*$b_ptr .add(1)); acc1_lo
            = vfmaq_f32(acc1_lo, a_lo, b1); acc1_hi = vfmaq_f32(acc1_hi, a_hi, b1); let
            b2 = vdupq_n_f32(*$b_ptr .add(2)); acc2_lo = vfmaq_f32(acc2_lo, a_lo, b2);
            acc2_hi = vfmaq_f32(acc2_hi, a_hi, b2); let b3 = vdupq_n_f32(*$b_ptr
            .add(3)); acc3_lo = vfmaq_f32(acc3_lo, a_lo, b3); acc3_hi =
            vfmaq_f32(acc3_hi, a_hi, b3); let b4 = vdupq_n_f32(*$b_ptr .add(4)); acc4_lo
            = vfmaq_f32(acc4_lo, a_lo, b4); acc4_hi = vfmaq_f32(acc4_hi, a_hi, b4); let
            b5 = vdupq_n_f32(*$b_ptr .add(5)); acc5_lo = vfmaq_f32(acc5_lo, a_lo, b5);
            acc5_hi = vfmaq_f32(acc5_hi, a_hi, b5); let b6 = vdupq_n_f32(*$b_ptr
            .add(6)); acc6_lo = vfmaq_f32(acc6_lo, a_lo, b6); acc6_hi =
            vfmaq_f32(acc6_hi, a_hi, b6); let b7 = vdupq_n_f32(*$b_ptr .add(7)); acc7_lo
            = vfmaq_f32(acc7_lo, a_lo, b7); acc7_hi = vfmaq_f32(acc7_hi, a_hi, b7); }
        };
    }
    for p in 0..k_unroll {
        let base = p * 4;
        if p + PREFETCH_DIST / 4 < k_unroll {
            let pf_base = (p + PREFETCH_DIST / 4) * 4;
            prefetch_read(a.add(pf_base * MR));
            prefetch_read(a.add(pf_base * MR + 16));
            prefetch_read(b.add(pf_base * NR));
            prefetch_read(b.add(pf_base * NR + 16));
        }
        fma_iteration!(a.add(base * MR), b.add(base * NR));
        fma_iteration!(a.add((base + 1) * MR), b.add((base + 1) * NR));
        fma_iteration!(a.add((base + 2) * MR), b.add((base + 2) * NR));
        fma_iteration!(a.add((base + 3) * MR), b.add((base + 3) * NR));
    }
    for r in 0..k_remainder {
        let p = k_unroll * 4 + r;
        fma_iteration!(a.add(p * MR), b.add(p * NR));
    }
    let alpha_is_one = alpha == 1.0;
    if !alpha_is_one {
        let alpha_vec = vdupq_n_f32(alpha);
        acc0_lo = vmulq_f32(acc0_lo, alpha_vec);
        acc0_hi = vmulq_f32(acc0_hi, alpha_vec);
        acc1_lo = vmulq_f32(acc1_lo, alpha_vec);
        acc1_hi = vmulq_f32(acc1_hi, alpha_vec);
        acc2_lo = vmulq_f32(acc2_lo, alpha_vec);
        acc2_hi = vmulq_f32(acc2_hi, alpha_vec);
        acc3_lo = vmulq_f32(acc3_lo, alpha_vec);
        acc3_hi = vmulq_f32(acc3_hi, alpha_vec);
        acc4_lo = vmulq_f32(acc4_lo, alpha_vec);
        acc4_hi = vmulq_f32(acc4_hi, alpha_vec);
        acc5_lo = vmulq_f32(acc5_lo, alpha_vec);
        acc5_hi = vmulq_f32(acc5_hi, alpha_vec);
        acc6_lo = vmulq_f32(acc6_lo, alpha_vec);
        acc6_hi = vmulq_f32(acc6_hi, alpha_vec);
        acc7_lo = vmulq_f32(acc7_lo, alpha_vec);
        acc7_hi = vmulq_f32(acc7_hi, alpha_vec);
    }
    if beta == 0.0 {
        macro_rules! store_direct {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
                { let c_col = c.add($col * c_stride); vst1q_f32(c_col, $acc_lo);
                vst1q_f32(c_col.add(4), $acc_hi); }
            };
        }
        store_direct!(0, acc0_lo, acc0_hi);
        store_direct!(1, acc1_lo, acc1_hi);
        store_direct!(2, acc2_lo, acc2_hi);
        store_direct!(3, acc3_lo, acc3_hi);
        store_direct!(4, acc4_lo, acc4_hi);
        store_direct!(5, acc5_lo, acc5_hi);
        store_direct!(6, acc6_lo, acc6_hi);
        store_direct!(7, acc7_lo, acc7_hi);
        return;
    }
    if beta == 1.0 {
        macro_rules! store_add {
            ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
                { let c_col = c.add($col * c_stride); let c_lo = vld1q_f32(c_col); let
                c_hi = vld1q_f32(c_col.add(4)); vst1q_f32(c_col, vaddq_f32($acc_lo,
                c_lo)); vst1q_f32(c_col.add(4), vaddq_f32($acc_hi, c_hi)); }
            };
        }
        store_add!(0, acc0_lo, acc0_hi);
        store_add!(1, acc1_lo, acc1_hi);
        store_add!(2, acc2_lo, acc2_hi);
        store_add!(3, acc3_lo, acc3_hi);
        store_add!(4, acc4_lo, acc4_hi);
        store_add!(5, acc5_lo, acc5_hi);
        store_add!(6, acc6_lo, acc6_hi);
        store_add!(7, acc7_lo, acc7_hi);
        return;
    }
    let beta_vec = vdupq_n_f32(beta);
    macro_rules! scale_and_store {
        ($col:expr, $acc_lo:expr, $acc_hi:expr) => {
            { let c_col = c.add($col * c_stride); let c_lo = vld1q_f32(c_col); let c_hi =
            vld1q_f32(c_col.add(4)); vst1q_f32(c_col, vfmaq_f32($acc_lo, c_lo,
            beta_vec)); vst1q_f32(c_col.add(4), vfmaq_f32($acc_hi, c_hi, beta_vec)); }
        };
    }
    scale_and_store!(0, acc0_lo, acc0_hi);
    scale_and_store!(1, acc1_lo, acc1_hi);
    scale_and_store!(2, acc2_lo, acc2_hi);
    scale_and_store!(3, acc3_lo, acc3_hi);
    scale_and_store!(4, acc4_lo, acc4_hi);
    scale_and_store!(5, acc5_lo, acc5_hi);
    scale_and_store!(6, acc6_lo, acc6_hi);
    scale_and_store!(7, acc7_lo, acc7_hi);
}
impl GemmKernel for Complex64 {
    fn micro_kernel_shape() -> MicroKernelShape {
        MicroKernelShape { mr: 2, nr: 2 }
    }
    #[inline]
    unsafe fn micro_kernel(
        k: usize,
        alpha: Self,
        a: *const Self,
        b: *const Self,
        beta: Self,
        c: *mut Self,
        c_stride: usize,
    ) {
        micro_kernel_c64_scalar(k, alpha, a, b, beta, c, c_stride);
    }
}
/// Scalar micro-kernel for Complex64 (2×2 tile).
///
/// Computes C = alpha * A * B + beta * C for a 2-row × 2-column tile.
/// A is a (k × 2) packed column-major panel, B is a (k × 2) packed panel.
#[inline]
unsafe fn micro_kernel_c64_scalar(
    k: usize,
    alpha: Complex64,
    a: *const Complex64,
    b: *const Complex64,
    beta: Complex64,
    c: *mut Complex64,
    c_stride: usize,
) {
    const MR: usize = 2;
    const NR: usize = 2;
    let zero = Complex64::new(0.0, 0.0);
    let mut acc = [[zero; NR]; MR];
    for p in 0..k {
        let a_ptr = a.add(p * MR);
        let b_ptr = b.add(p * NR);
        for i in 0..MR {
            let a_val = *a_ptr.add(i);
            for j in 0..NR {
                let b_val = *b_ptr.add(j);
                acc[i][j] = Complex64::new(
                    acc[i][j].re + a_val.re * b_val.re - a_val.im * b_val.im,
                    acc[i][j].im + a_val.re * b_val.im + a_val.im * b_val.re,
                );
            }
        }
    }
    let beta_is_zero = beta.re == 0.0 && beta.im == 0.0;
    for j in 0..NR {
        for i in 0..MR {
            let c_ptr = c.add(i + j * c_stride);
            *c_ptr = if beta_is_zero {
                alpha * acc[i][j]
            } else {
                alpha * acc[i][j] + beta * (*c_ptr)
            };
        }
    }
}
impl GemmKernel for Complex32 {
    fn micro_kernel_shape() -> MicroKernelShape {
        MicroKernelShape { mr: 2, nr: 2 }
    }
    #[inline]
    unsafe fn micro_kernel(
        k: usize,
        alpha: Self,
        a: *const Self,
        b: *const Self,
        beta: Self,
        c: *mut Self,
        c_stride: usize,
    ) {
        micro_kernel_c32_scalar(k, alpha, a, b, beta, c, c_stride);
    }
}
/// Scalar micro-kernel for Complex32 (2×2 tile).
#[inline]
unsafe fn micro_kernel_c32_scalar(
    k: usize,
    alpha: Complex32,
    a: *const Complex32,
    b: *const Complex32,
    beta: Complex32,
    c: *mut Complex32,
    c_stride: usize,
) {
    const MR: usize = 2;
    const NR: usize = 2;
    let zero = Complex32::new(0.0, 0.0);
    let mut acc = [[zero; NR]; MR];
    for p in 0..k {
        let a_ptr = a.add(p * MR);
        let b_ptr = b.add(p * NR);
        for i in 0..MR {
            let a_val = *a_ptr.add(i);
            for j in 0..NR {
                let b_val = *b_ptr.add(j);
                acc[i][j] = Complex32::new(
                    acc[i][j].re + a_val.re * b_val.re - a_val.im * b_val.im,
                    acc[i][j].im + a_val.re * b_val.im + a_val.im * b_val.re,
                );
            }
        }
    }
    let beta_is_zero = beta.re == 0.0 && beta.im == 0.0;
    for j in 0..NR {
        for i in 0..MR {
            let c_ptr = c.add(i + j * c_stride);
            *c_ptr = if beta_is_zero {
                alpha * acc[i][j]
            } else {
                alpha * acc[i][j] + beta * (*c_ptr)
            };
        }
    }
}

#[cfg(test)]
mod tests;
