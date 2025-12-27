//! Small matrix GEMM specializations.
//!
//! This module provides optimized GEMM implementations for small matrices
//! (dimensions < 32) where the overhead of packing and blocked algorithms
//! outweighs their benefits.
//!
//! For small matrices, we use:
//! - Unrolled loops to reduce loop overhead
//! - Register blocking to maximize register utilization
//! - SIMD when available and beneficial
//! - Direct memory access (no packing)

use oxiblas_core::scalar::Field;
use oxiblas_matrix::{MatMut, MatRef};

/// Threshold for using small matrix specialization.
///
/// When m * n * k is below this threshold, small matrix kernels are used.
pub const SMALL_THRESHOLD: usize = 32 * 32 * 32;

/// Performs small matrix GEMM with unrolled loops.
///
/// C = alpha * A * B + beta * C
///
/// This is optimized for matrices where m, n, k < 32.
pub fn gemm_small<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // Handle beta scaling
    if beta == T::zero() {
        c.fill_zero();
    } else if beta != T::one() {
        c.scale(beta);
    }

    if alpha == T::zero() {
        return;
    }

    // Dispatch to size-specific implementations
    // For very small matrices, use fully unrolled versions
    match (m, n, k) {
        // 2x2 case
        (2, 2, kk) if kk <= 32 => gemm_2x2_k(alpha, a, b, c, kk),
        // 3x3 case
        (3, 3, kk) if kk <= 32 => gemm_3x3_k(alpha, a, b, c, kk),
        // 4x4 case
        (4, 4, kk) if kk <= 32 => gemm_4x4_k(alpha, a, b, c, kk),
        // Small matrices using register blocking
        _ if m <= 8 && n <= 8 => gemm_small_blocked_8(alpha, a, b, c, m, n, k),
        _ if m <= 16 && n <= 16 => gemm_small_blocked_16(alpha, a, b, c, m, n, k),
        // General small matrix case
        _ => gemm_small_general(alpha, a, b, c, m, n, k),
    }
}

/// 2x2 GEMM with fully unrolled k-loop.
#[inline(always)]
fn gemm_2x2_k<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    k: usize,
) {
    // Use 4 accumulators for the 2x2 result
    let mut c00 = T::zero();
    let mut c01 = T::zero();
    let mut c10 = T::zero();
    let mut c11 = T::zero();

    // Unroll k-loop by 4 when possible
    let k4 = k / 4;
    let k_rem = k % 4;

    for p in 0..k4 {
        let pp = p * 4;

        // Load A column vectors for 4 k-iterations
        let a00 = a[(0, pp)];
        let a10 = a[(1, pp)];
        let a01 = a[(0, pp + 1)];
        let a11 = a[(1, pp + 1)];
        let a02 = a[(0, pp + 2)];
        let a12 = a[(1, pp + 2)];
        let a03 = a[(0, pp + 3)];
        let a13 = a[(1, pp + 3)];

        // Load B row vectors for 4 k-iterations
        let b00 = b[(pp, 0)];
        let b01 = b[(pp, 1)];
        let b10 = b[(pp + 1, 0)];
        let b11 = b[(pp + 1, 1)];
        let b20 = b[(pp + 2, 0)];
        let b21 = b[(pp + 2, 1)];
        let b30 = b[(pp + 3, 0)];
        let b31 = b[(pp + 3, 1)];

        // Accumulate
        c00 = c00 + a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
        c01 = c01 + a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
        c10 = c10 + a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
        c11 = c11 + a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    }

    // Handle remaining k iterations
    for p in (k - k_rem)..k {
        let a0 = a[(0, p)];
        let a1 = a[(1, p)];
        let b0 = b[(p, 0)];
        let b1 = b[(p, 1)];

        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }

    // Store result with alpha scaling
    let val00 = c[(0, 0)] + alpha * c00;
    let val01 = c[(0, 1)] + alpha * c01;
    let val10 = c[(1, 0)] + alpha * c10;
    let val11 = c[(1, 1)] + alpha * c11;

    c.set(0, 0, val00);
    c.set(0, 1, val01);
    c.set(1, 0, val10);
    c.set(1, 1, val11);
}

/// 3x3 GEMM with unrolled loops.
#[inline(always)]
fn gemm_3x3_k<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    k: usize,
) {
    // 9 accumulators for 3x3 result
    let mut acc = [[T::zero(); 3]; 3];

    // Unroll k-loop by 2
    let k2 = k / 2;
    let k_rem = k % 2;

    for p in 0..k2 {
        let pp = p * 2;

        // Load A columns
        let a0 = [a[(0, pp)], a[(1, pp)], a[(2, pp)]];
        let a1 = [a[(0, pp + 1)], a[(1, pp + 1)], a[(2, pp + 1)]];

        // Load B rows
        let b0 = [b[(pp, 0)], b[(pp, 1)], b[(pp, 2)]];
        let b1 = [b[(pp + 1, 0)], b[(pp + 1, 1)], b[(pp + 1, 2)]];

        // Outer product accumulation
        for i in 0..3 {
            for j in 0..3 {
                acc[i][j] = acc[i][j] + a0[i] * b0[j] + a1[i] * b1[j];
            }
        }
    }

    // Handle remaining
    for p in (k - k_rem)..k {
        for i in 0..3 {
            let ai = a[(i, p)];
            for j in 0..3 {
                acc[i][j] += ai * b[(p, j)];
            }
        }
    }

    // Store with alpha
    for i in 0..3 {
        for j in 0..3 {
            let val = c[(i, j)] + alpha * acc[i][j];
            c.set(i, j, val);
        }
    }
}

/// 4x4 GEMM with register blocking.
#[inline(always)]
fn gemm_4x4_k<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    k: usize,
) {
    // 16 accumulators for 4x4 result, stored as 4 columns of 4 elements
    let mut c0 = [T::zero(); 4];
    let mut c1 = [T::zero(); 4];
    let mut c2 = [T::zero(); 4];
    let mut c3 = [T::zero(); 4];

    for p in 0..k {
        // Load A column
        let a_col = [a[(0, p)], a[(1, p)], a[(2, p)], a[(3, p)]];

        // Load B row
        let b_row = [b[(p, 0)], b[(p, 1)], b[(p, 2)], b[(p, 3)]];

        // Rank-1 update
        for i in 0..4 {
            c0[i] += a_col[i] * b_row[0];
            c1[i] += a_col[i] * b_row[1];
            c2[i] += a_col[i] * b_row[2];
            c3[i] += a_col[i] * b_row[3];
        }
    }

    // Store with alpha
    for i in 0..4 {
        c.set(i, 0, c[(i, 0)] + alpha * c0[i]);
        c.set(i, 1, c[(i, 1)] + alpha * c1[i]);
        c.set(i, 2, c[(i, 2)] + alpha * c2[i]);
        c.set(i, 3, c[(i, 3)] + alpha * c3[i]);
    }
}

/// Small matrix GEMM with 4x4 register blocking for m,n <= 8.
fn gemm_small_blocked_8<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    m: usize,
    n: usize,
    k: usize,
) {
    const BLOCK: usize = 4;

    // Process in 4x4 blocks
    for jb in (0..n).step_by(BLOCK) {
        let jn = BLOCK.min(n - jb);
        for ib in (0..m).step_by(BLOCK) {
            let im = BLOCK.min(m - ib);

            // Accumulator for this block
            let mut acc = [[T::zero(); 4]; 4];

            // Compute A[ib:ib+im, :] * B[:, jb:jb+jn]
            for p in 0..k {
                for i in 0..im {
                    let a_val = a[(ib + i, p)];
                    for j in 0..jn {
                        acc[i][j] += a_val * b[(p, jb + j)];
                    }
                }
            }

            // Store results
            for i in 0..im {
                for j in 0..jn {
                    let val = c[(ib + i, jb + j)] + alpha * acc[i][j];
                    c.set(ib + i, jb + j, val);
                }
            }
        }
    }
}

/// Small matrix GEMM with 8x8 register blocking for m,n <= 16.
fn gemm_small_blocked_16<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    m: usize,
    n: usize,
    k: usize,
) {
    const BLOCK: usize = 8;

    // Process in 8x8 blocks
    for jb in (0..n).step_by(BLOCK) {
        let jn = BLOCK.min(n - jb);
        for ib in (0..m).step_by(BLOCK) {
            let im = BLOCK.min(m - ib);

            // Accumulator for this block
            let mut acc = [[T::zero(); 8]; 8];

            // Compute A[ib:ib+im, :] * B[:, jb:jb+jn]
            for p in 0..k {
                for i in 0..im {
                    let a_val = a[(ib + i, p)];
                    for j in 0..jn {
                        acc[i][j] += a_val * b[(p, jb + j)];
                    }
                }
            }

            // Store results
            for i in 0..im {
                for j in 0..jn {
                    let val = c[(ib + i, jb + j)] + alpha * acc[i][j];
                    c.set(ib + i, jb + j, val);
                }
            }
        }
    }
}

/// General small matrix GEMM with 8x4 blocking.
///
/// Uses outer-product formulation for better cache behavior on small matrices.
fn gemm_small_general<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    m: usize,
    n: usize,
    k: usize,
) {
    const MR: usize = 8;
    const NR: usize = 4;

    // Process in MRxNR blocks
    for jb in (0..n).step_by(NR) {
        let jn = NR.min(n - jb);
        for ib in (0..m).step_by(MR) {
            let im = MR.min(m - ib);

            // Small local accumulator (avoids repeated writes to C)
            let mut acc = [[T::zero(); NR]; MR];

            // Outer product loop
            for p in 0..k {
                // Load A column segment
                for i in 0..im {
                    let a_val = a[(ib + i, p)];
                    // Rank-1 update
                    for j in 0..jn {
                        acc[i][j] += a_val * b[(p, jb + j)];
                    }
                }
            }

            // Store results with alpha
            for i in 0..im {
                for j in 0..jn {
                    let val = c[(ib + i, jb + j)] + alpha * acc[i][j];
                    c.set(ib + i, jb + j, val);
                }
            }
        }
    }
}

/// SIMD-optimized small GEMM for f64 (when NEON or AVX2 is available).
#[cfg(target_arch = "aarch64")]
pub fn gemm_small_f64_simd(
    alpha: f64,
    a: &MatRef<'_, f64>,
    b: &MatRef<'_, f64>,
    c: &mut MatMut<'_, f64>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // Handle beta scaling is done by caller

    if alpha == 0.0 {
        return;
    }

    // For very small matrices, SIMD overhead isn't worth it
    if m * n * k < 64 {
        gemm_small(alpha, a, b, 0.0, c);
        return;
    }

    const MR: usize = 4;
    const NR: usize = 4;

    // Process in 4x4 blocks using NEON
    for jb in (0..n).step_by(NR) {
        let jn = NR.min(n - jb);
        for ib in (0..m).step_by(MR) {
            let im = MR.min(m - ib);

            if im == MR && jn == NR {
                // Full 4x4 block - use SIMD
                unsafe {
                    gemm_4x4_neon(alpha, a, b, c, ib, jb, k);
                }
            } else {
                // Partial block - scalar
                let mut acc = [[0.0f64; NR]; MR];
                for p in 0..k {
                    for i in 0..im {
                        let a_val = a[(ib + i, p)];
                        for j in 0..jn {
                            acc[i][j] += a_val * b[(p, jb + j)];
                        }
                    }
                }
                for i in 0..im {
                    for j in 0..jn {
                        let val = alpha.mul_add(acc[i][j], c[(ib + i, jb + j)]);
                        c.set(ib + i, jb + j, val);
                    }
                }
            }
        }
    }
}

/// 4x4 NEON kernel for f64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x4_neon(
    alpha: f64,
    a: &MatRef<'_, f64>,
    b: &MatRef<'_, f64>,
    c: &mut MatMut<'_, f64>,
    i: usize,
    j: usize,
    k: usize,
) {
    use std::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vmulq_f64, vst1q_f64};

    // 4x4 accumulator using 8 vector registers (2 per column)
    let mut c0_lo = vdupq_n_f64(0.0);
    let mut c0_hi = vdupq_n_f64(0.0);
    let mut c1_lo = vdupq_n_f64(0.0);
    let mut c1_hi = vdupq_n_f64(0.0);
    let mut c2_lo = vdupq_n_f64(0.0);
    let mut c2_hi = vdupq_n_f64(0.0);
    let mut c3_lo = vdupq_n_f64(0.0);
    let mut c3_hi = vdupq_n_f64(0.0);

    let a_row_stride = a.row_stride();
    let b_row_stride = b.row_stride();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for p in 0..k {
        // Load A column [i:i+4, p] - need to gather due to column stride
        let a_base = a_ptr.add(i + p * a_row_stride);
        let a0 = *a_base;
        let a1 = *a_base.add(1);
        let a2 = *a_base.add(2);
        let a3 = *a_base.add(3);

        let a_lo = vld1q_f64([a0, a1].as_ptr());
        let a_hi = vld1q_f64([a2, a3].as_ptr());

        // Load B row [p, j:j+4] - need to gather due to column stride
        let b_base = b_ptr.add(p + j * b_row_stride);
        let b0 = vdupq_n_f64(*b_base);
        let b1 = vdupq_n_f64(*b_base.add(b_row_stride));
        let b2 = vdupq_n_f64(*b_base.add(2 * b_row_stride));
        let b3 = vdupq_n_f64(*b_base.add(3 * b_row_stride));

        // Rank-1 update using FMA
        c0_lo = vfmaq_f64(c0_lo, a_lo, b0);
        c0_hi = vfmaq_f64(c0_hi, a_hi, b0);
        c1_lo = vfmaq_f64(c1_lo, a_lo, b1);
        c1_hi = vfmaq_f64(c1_hi, a_hi, b1);
        c2_lo = vfmaq_f64(c2_lo, a_lo, b2);
        c2_hi = vfmaq_f64(c2_hi, a_hi, b2);
        c3_lo = vfmaq_f64(c3_lo, a_lo, b3);
        c3_hi = vfmaq_f64(c3_hi, a_hi, b3);
    }

    // Scale by alpha and add to C
    let alpha_v = vdupq_n_f64(alpha);
    c0_lo = vmulq_f64(c0_lo, alpha_v);
    c0_hi = vmulq_f64(c0_hi, alpha_v);
    c1_lo = vmulq_f64(c1_lo, alpha_v);
    c1_hi = vmulq_f64(c1_hi, alpha_v);
    c2_lo = vmulq_f64(c2_lo, alpha_v);
    c2_hi = vmulq_f64(c2_hi, alpha_v);
    c3_lo = vmulq_f64(c3_lo, alpha_v);
    c3_hi = vmulq_f64(c3_hi, alpha_v);

    // Extract and store
    let c_row_stride = c.row_stride();
    let c_ptr = c.as_ptr().cast_mut();

    // Column 0
    let c_col0 = c_ptr.add(i + j * c_row_stride);
    let mut tmp = [0.0f64; 4];
    vst1q_f64(tmp.as_mut_ptr(), c0_lo);
    *c_col0 += tmp[0];
    *c_col0.add(1) += tmp[1];
    vst1q_f64(tmp.as_mut_ptr(), c0_hi);
    *c_col0.add(2) += tmp[0];
    *c_col0.add(3) += tmp[1];

    // Column 1
    let c_col1 = c_ptr.add(i + (j + 1) * c_row_stride);
    vst1q_f64(tmp.as_mut_ptr(), c1_lo);
    *c_col1 += tmp[0];
    *c_col1.add(1) += tmp[1];
    vst1q_f64(tmp.as_mut_ptr(), c1_hi);
    *c_col1.add(2) += tmp[0];
    *c_col1.add(3) += tmp[1];

    // Column 2
    let c_col2 = c_ptr.add(i + (j + 2) * c_row_stride);
    vst1q_f64(tmp.as_mut_ptr(), c2_lo);
    *c_col2 += tmp[0];
    *c_col2.add(1) += tmp[1];
    vst1q_f64(tmp.as_mut_ptr(), c2_hi);
    *c_col2.add(2) += tmp[0];
    *c_col2.add(3) += tmp[1];

    // Column 3
    let c_col3 = c_ptr.add(i + (j + 3) * c_row_stride);
    vst1q_f64(tmp.as_mut_ptr(), c3_lo);
    *c_col3 += tmp[0];
    *c_col3.add(1) += tmp[1];
    vst1q_f64(tmp.as_mut_ptr(), c3_hi);
    *c_col3.add(2) += tmp[0];
    *c_col3.add(3) += tmp[1];
}

/// SIMD-optimized small GEMM for f32 (when NEON or AVX2 is available).
#[cfg(target_arch = "aarch64")]
pub fn gemm_small_f32_simd(
    alpha: f32,
    a: &MatRef<'_, f32>,
    b: &MatRef<'_, f32>,
    c: &mut MatMut<'_, f32>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    if alpha == 0.0 {
        return;
    }

    // For very small matrices, SIMD overhead isn't worth it
    if m * n * k < 64 {
        gemm_small(alpha, a, b, 0.0, c);
        return;
    }

    const MR: usize = 4;
    const NR: usize = 4;

    // Process in 4x4 blocks using NEON
    for jb in (0..n).step_by(NR) {
        let jn = NR.min(n - jb);
        for ib in (0..m).step_by(MR) {
            let im = MR.min(m - ib);

            if im == MR && jn == NR {
                // Full 4x4 block - use SIMD
                unsafe {
                    gemm_4x4_neon_f32(alpha, a, b, c, ib, jb, k);
                }
            } else {
                // Partial block - scalar
                let mut acc = [[0.0f32; NR]; MR];
                for p in 0..k {
                    for i in 0..im {
                        let a_val = a[(ib + i, p)];
                        for j in 0..jn {
                            acc[i][j] += a_val * b[(p, jb + j)];
                        }
                    }
                }
                for i in 0..im {
                    for j in 0..jn {
                        let val = alpha.mul_add(acc[i][j], c[(ib + i, jb + j)]);
                        c.set(ib + i, jb + j, val);
                    }
                }
            }
        }
    }
}

/// 4x4 NEON kernel for f32.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gemm_4x4_neon_f32(
    alpha: f32,
    a: &MatRef<'_, f32>,
    b: &MatRef<'_, f32>,
    c: &mut MatMut<'_, f32>,
    i: usize,
    j: usize,
    k: usize,
) {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    // 4x4 accumulator using 4 vector registers (one per column)
    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);

    let a_row_stride = a.row_stride();
    let b_row_stride = b.row_stride();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for p in 0..k {
        // Load A column [i:i+4, p] - gather due to column stride
        let a_base = a_ptr.add(i + p * a_row_stride);
        let a_vec = vld1q_f32([*a_base, *a_base.add(1), *a_base.add(2), *a_base.add(3)].as_ptr());

        // Load B row [p, j:j+4] - gather due to column stride
        let b_base = b_ptr.add(p + j * b_row_stride);
        let b0 = vdupq_n_f32(*b_base);
        let b1 = vdupq_n_f32(*b_base.add(b_row_stride));
        let b2 = vdupq_n_f32(*b_base.add(2 * b_row_stride));
        let b3 = vdupq_n_f32(*b_base.add(3 * b_row_stride));

        // Rank-1 update using FMA
        c0 = vfmaq_f32(c0, a_vec, b0);
        c1 = vfmaq_f32(c1, a_vec, b1);
        c2 = vfmaq_f32(c2, a_vec, b2);
        c3 = vfmaq_f32(c3, a_vec, b3);
    }

    // Scale by alpha and add to C
    let alpha_v = vdupq_n_f32(alpha);
    c0 = vmulq_f32(c0, alpha_v);
    c1 = vmulq_f32(c1, alpha_v);
    c2 = vmulq_f32(c2, alpha_v);
    c3 = vmulq_f32(c3, alpha_v);

    // Store (add to existing C values)
    let c_row_stride = c.row_stride();
    let c_ptr = c.as_ptr().cast_mut();

    let mut tmp = [0.0f32; 4];

    // Column 0
    let c_col0 = c_ptr.add(i + j * c_row_stride);
    vst1q_f32(tmp.as_mut_ptr(), c0);
    *c_col0 += tmp[0];
    *c_col0.add(1) += tmp[1];
    *c_col0.add(2) += tmp[2];
    *c_col0.add(3) += tmp[3];

    // Column 1
    let c_col1 = c_ptr.add(i + (j + 1) * c_row_stride);
    vst1q_f32(tmp.as_mut_ptr(), c1);
    *c_col1 += tmp[0];
    *c_col1.add(1) += tmp[1];
    *c_col1.add(2) += tmp[2];
    *c_col1.add(3) += tmp[3];

    // Column 2
    let c_col2 = c_ptr.add(i + (j + 2) * c_row_stride);
    vst1q_f32(tmp.as_mut_ptr(), c2);
    *c_col2 += tmp[0];
    *c_col2.add(1) += tmp[1];
    *c_col2.add(2) += tmp[2];
    *c_col2.add(3) += tmp[3];

    // Column 3
    let c_col3 = c_ptr.add(i + (j + 3) * c_row_stride);
    vst1q_f32(tmp.as_mut_ptr(), c3);
    *c_col3 += tmp[0];
    *c_col3.add(1) += tmp[1];
    *c_col3.add(2) += tmp[2];
    *c_col3.add(3) += tmp[3];
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_gemm_2x2() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let mut c: Mat<f64> = Mat::zeros(2, 2);

        gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

        // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_3x3() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let b: Mat<f64> = Mat::eye(3);
        let mut c: Mat<f64> = Mat::zeros(3, 3);

        gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

        // A*I = A
        for i in 0..3 {
            for j in 0..3 {
                assert!((c[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_4x4() {
        let a: Mat<f64> = Mat::filled(4, 4, 2.0);
        let b: Mat<f64> = Mat::filled(4, 4, 3.0);
        let mut c: Mat<f64> = Mat::zeros(4, 4);

        gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

        // Each element = 4 * (2*3) = 24
        for i in 0..4 {
            for j in 0..4 {
                assert!((c[(i, j)] - 24.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_small_8x8() {
        let n = 8;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

        // Each element = n * (1*1) = n
        for i in 0..n {
            for j in 0..n {
                assert!((c[(i, j)] - n as f64).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_small_16x16() {
        let n = 16;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

        for i in 0..n {
            for j in 0..n {
                assert!((c[(i, j)] - n as f64).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_small_general() {
        let m = 20;
        let k = 15;
        let n = 25;
        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_small(1.0, &a.as_ref(), &b.as_ref(), 0.0, &mut c.as_mut());

        for i in 0..m {
            for j in 0..n {
                assert!((c[(i, j)] - k as f64).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let mut c: Mat<f64> = Mat::from_rows(&[&[10.0, 20.0], &[30.0, 40.0]]);

        // C = 2*A*B + 0.5*C = 2*A + 0.5*C
        gemm_small(2.0, &a.as_ref(), &b.as_ref(), 0.5, &mut c.as_mut());

        // Expected: 2*[[1,2],[3,4]] + 0.5*[[10,20],[30,40]]
        //         = [[2,4],[6,8]] + [[5,10],[15,20]]
        //         = [[7,14],[21,28]]
        assert!((c[(0, 0)] - 7.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 14.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 21.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[5.0f32, 6.0], &[7.0, 8.0]]);
        let mut c: Mat<f32> = Mat::zeros(2, 2);

        gemm_small(1.0f32, &a.as_ref(), &b.as_ref(), 0.0f32, &mut c.as_mut());

        assert!((c[(0, 0)] - 19.0).abs() < 1e-5);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-5);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-5);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_small_f64_simd() {
        let n = 8;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm_small_f64_simd(1.0, &a.as_ref(), &b.as_ref(), &mut c.as_mut());

        for i in 0..n {
            for j in 0..n {
                assert!((c[(i, j)] - n as f64).abs() < 1e-10);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_small_f32_simd() {
        let n = 8;
        let a: Mat<f32> = Mat::filled(n, n, 1.0);
        let b: Mat<f32> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f32> = Mat::zeros(n, n);

        gemm_small_f32_simd(1.0, &a.as_ref(), &b.as_ref(), &mut c.as_mut());

        for i in 0..n {
            for j in 0..n {
                assert!((c[(i, j)] - n as f32).abs() < 1e-5);
            }
        }
    }
}
