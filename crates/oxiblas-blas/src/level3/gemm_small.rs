//! Small matrix GEMM specializations.
//!
//! This module provides optimized GEMM implementations for small problems
//! (by default `m * n * k <= SMALL_THRESHOLD`, i.e. at most 32³ multiply-adds
//! however the dimensions are split; a routing policy may raise the ceiling
//! for problems its tier's SIMD small kernels take, see
//! `GemmPolicy::small_simd_max_work`) where the overhead of packing and
//! blocked algorithms outweighs their benefits.
//!
//! For small matrices, we use:
//! - Unrolled loops to reduce loop overhead
//! - Register blocking to maximize register utilization
//! - SIMD when available and beneficial
//! - Direct memory access (no packing)
//!
//! SIMD small kernels per tier (`f64`/`f32`, column-major operands):
//! - NEON (aarch64): 4×4 tiles from `m, n >= 4` (`gemm_small_f64_simd` /
//!   `gemm_small_f32_simd`), under every policy.
//! - AVX2 + FMA (x86-64): f64 8×4 and f32 16×4 register tiles with masked row
//!   tails, from `m >= 5`, `n >= 4`, only under a policy with
//!   `small_simd_x86` (the x86 candidate policy; the default policy of a
//!   normal build keeps the portable loops there, exactly as 0.2.3). Every
//!   element is computed by the same fused sequence wherever it sits, so the
//!   results are position independent (see `gemm_small_avx2.rs`).
//! - SSE4.2 and portable builds: the portable loops.

use super::gemm::{op_dims, read_op};
use super::gemm_kernel::GemmKernel;
use super::gemm_packing::{PackBuffer, PackSlot};
use super::gemm_plan::{DEFAULT_POLICY, GemmContext, GemmPolicy, SimdTier};
use super::trsm::Trans;
use crate::level1::typed::{same_matmut, same_matref, same_value};
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{MatMut, MatRef};

#[cfg(target_arch = "x86_64")]
#[path = "gemm_small_avx2.rs"]
mod avx2;

/// Threshold for using small matrix specialization.
///
/// Under the default policy, GEMM sends a `NoTrans`/`NoTrans` problem with
/// `m * n * k` at most this threshold (32³) to the small path, on every SIMD
/// tier. (The candidate policies first send thin problems, `m == 1` or
/// `n == 1`, to the thin route, and, on a tier whose SIMD small kernels take
/// the shape, a problem with operands that are not column-major between 8³
/// and 32³ to the copy-to-column-major small path.) A routing policy
/// may raise the ceiling for column-major `f64`/`f32` problems that a tier's
/// SIMD small kernels take (`small_simd_max_work`): the x86-64 candidate
/// policy (`--cfg oxiblas_gemm_candidate`) uses 80³ for the AVX2 + FMA small
/// kernels; the default policy keeps 32³ everywhere.
pub const SMALL_THRESHOLD: usize = 32 * 32 * 32;

/// Performs small matrix GEMM with unrolled loops.
///
/// C = alpha * A * B + beta * C
///
/// This is optimized for small problems (the blocked GEMM routes here when
/// `m * n * k <= SMALL_THRESHOLD`). As in reference BLAS, C is not read when
/// `beta == 0` and A/B are not read when `alpha == 0`.
///
/// It runs with this build's default routing policy. On aarch64, `f64`/`f32`
/// problems with `m, n >= 4` and column-major operands (`col_stride() == 1`)
/// run on the NEON kernels (`gemm_small_f64_simd` / `gemm_small_f32_simd`).
/// On x86-64 with AVX2 and FMA, the same problems with `m >= 5`, `n >= 4` run
/// on the AVX2 + FMA small kernels only when the crate is built with the
/// candidate policy (`--cfg oxiblas_gemm_candidate`); the default policy
/// keeps the portable loops there, bit for bit as in 0.2.3. All other cases
/// (other element types, strided or transposed views, SSE4.2, `force-scalar`
/// builds, and x86 builds capped by `max-simd-128`) use the portable loops.
pub fn gemm_small<T: Field>(
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) {
    gemm_small_with(GemmContext::detect(DEFAULT_POLICY), alpha, a, b, beta, c);
}

/// [`gemm_small`] for the SIMD tier and policy of a GEMM plan.
///
/// `C = alpha * A * B + beta * C` with C not read when `beta == 0` and A/B
/// not read when `alpha == 0`. Beta is applied first (C is zero-filled
/// without being read when `beta == 0`); then the SIMD small kernels of
/// `ctx`'s tier accumulate `alpha * A * B` when they apply (`f64`/`f32` with
/// column-major operands and consistent shapes):
/// - NEON: from `m, n >= 4`;
/// - AVX2 + FMA: from `m >= 5`, `n >= 4`, when `ctx.policy.small_simd_x86`
///   (the same rule as the router's `RouteInputs::simd_small`); per element
///   `acc = fma(A[i, p], B[p, j], acc)` for `p` ascending from `+0`, then
///   `C[i, j] = fma(acc, alpha, C[i, j])`.
///
/// Everything else (SSE4.2 and portable tiers, the AVX2 tier under a policy
/// without `small_simd_x86`, other element types, strided operands) takes
/// the portable loops.
pub(crate) fn gemm_small_with<T: Field>(
    ctx: GemmContext,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) {
    // Handle beta scaling (C is not read when beta == 0)
    if beta == T::zero() {
        c.fill_zero();
    } else if beta != T::one() {
        c.scale(beta);
    }

    if alpha == T::zero() {
        return;
    }

    if !small_simd(ctx, alpha, a, b, c) {
        gemm_small_acc(alpha, a, b, c);
    }
}

/// `C = alpha * op(A) * op(B) + beta * C` for a transposed or strided small
/// problem on the column-major small kernels.
///
/// `op(A)` (`m × k`) and `op(B)` (`k × n`) are copied to column-major scratch
/// (leading dimensions `m` and `k`) in this thread's A and B packing buffers;
/// an operand that is already `NoTrans` with `col_stride() == 1` is used in
/// place. Then [`gemm_small_with`] runs on the copies, so alpha, beta and C
/// behave exactly as for `NoTrans`/`NoTrans`, and the result is bit for bit
/// that of `gemm_small_with` on column-major operands holding the same values.
///
/// Returns `false` without touching C when the shapes are inconsistent or a
/// scratch view cannot be formed; the caller then uses the transpose-aware
/// loops.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_small_trans_copy<T: Field + GemmKernel + bytemuck::Zeroable>(
    ctx: GemmContext,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) -> bool {
    let (m, k) = op_dims(a, trans_a);
    let (kb, n) = op_dims(b, trans_b);
    if kb != k || c.nrows() != m || c.ncols() != n {
        return false;
    }
    let a_buf = (!is_plain_column_major(a, trans_a))
        .then(|| copy_to_column_major(a, trans_a, m, k, PackSlot::A));
    let b_buf = (!is_plain_column_major(b, trans_b))
        .then(|| copy_to_column_major(b, trans_b, k, n, PackSlot::B));
    let a_view = match &a_buf {
        Some(buf) => match MatRef::from_column_major(buf.as_slice(), m, k) {
            Some(view) => view,
            None => return false,
        },
        None => *a,
    };
    let b_view = match &b_buf {
        Some(buf) => match MatRef::from_column_major(buf.as_slice(), k, n) {
            Some(view) => view,
            None => return false,
        },
        None => *b,
    };
    gemm_small_with(ctx, alpha, &a_view, &b_view, beta, c);
    true
}

/// `op(x)` is `x` itself in plain column-major storage.
#[inline]
fn is_plain_column_major<T: Scalar>(x: &MatRef<'_, T>, trans: Trans) -> bool {
    trans == Trans::NoTrans && x.col_stride() == 1
}

/// `op(src)` (`rows × cols`) copied into this thread's packing buffer of
/// `slot` as a column-major matrix with leading dimension `rows` (every
/// element of the first `rows · cols` is written).
fn copy_to_column_major<T: Field + GemmKernel + bytemuck::Zeroable>(
    src: &MatRef<'_, T>,
    trans: Trans,
    rows: usize,
    cols: usize,
    slot: PackSlot,
) -> PackBuffer<T> {
    let mut buf = PackBuffer::<T>::take(slot, rows * cols);
    copy_op_column_major(
        src,
        trans,
        &mut buf.as_mut_slice()[..rows * cols],
        rows,
        cols,
    );
    buf
}

/// `dst[i + j·rows] = op(src)[i, j]` for every `i < rows`, `j < cols`.
///
/// The loops follow the contiguous direction of the source: column by column
/// when the columns of `op(src)` are contiguous, otherwise in 4 × 4 blocks
/// that read along the rows of `op(src)` (the stored columns of a transposed
/// operand) and write four destination columns.
fn copy_op_column_major<T: Field>(
    src: &MatRef<'_, T>,
    trans: Trans,
    dst: &mut [T],
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    // Distance in `src` between op(src)[i, j] and op(src)[i + 1, j].
    let down = if trans == Trans::NoTrans {
        src.col_stride()
    } else {
        src.row_stride()
    };
    if down == 1 || rows == 1 {
        for (j, column) in dst.chunks_exact_mut(rows).take(cols).enumerate() {
            for (i, d) in column.iter_mut().enumerate() {
                *d = read_op(src, trans, i, j);
            }
        }
        return;
    }
    const BLOCK: usize = 4;
    for j0 in (0..cols).step_by(BLOCK) {
        let j1 = (j0 + BLOCK).min(cols);
        for i0 in (0..rows).step_by(BLOCK) {
            for i in i0..(i0 + BLOCK).min(rows) {
                for j in j0..j1 {
                    dst[i + j * rows] = read_op(src, trans, i, j);
                }
            }
        }
    }
}

/// `C += alpha * A * B` on the SIMD small kernels of `ctx`'s tier when `T`
/// is `f64` or `f32`, the operands are column-major with consistent shapes,
/// and the tier's kernels take the shape; `false` (nothing done) otherwise.
///
/// The typed views are formed here, on every target, with the
/// `level1::typed` hand-off; the per-type functions below choose the kernel
/// by tier.
#[inline]
fn small_simd<T: Field>(
    ctx: GemmContext,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
) -> bool {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // The SIMD kernels address the operands through raw pointers with unit
    // row-to-row strides (plain column-major A and B; `MatMut` always has
    // them), so they need those strides and consistent shapes.
    let eligible = a.col_stride() == 1
        && b.col_stride() == 1
        && b.nrows() == k
        && c.nrows() == m
        && c.ncols() == n;
    if !eligible {
        return false;
    }

    if let (Some(a64), Some(b64), Some(alpha64)) = (
        same_matref::<T, f64>(*a),
        same_matref::<T, f64>(*b),
        same_value::<T, f64>(alpha),
    ) && let Some(mut c64) = same_matmut::<T, f64>(c.rb_mut())
    {
        return small_simd_f64(ctx, alpha64, &a64, &b64, &mut c64);
    }
    if let (Some(a32), Some(b32), Some(alpha32)) = (
        same_matref::<T, f32>(*a),
        same_matref::<T, f32>(*b),
        same_value::<T, f32>(alpha),
    ) && let Some(mut c32) = same_matmut::<T, f32>(c.rb_mut())
    {
        return small_simd_f32(ctx, alpha32, &a32, &b32, &mut c32);
    }
    false
}

/// [`small_simd`] for `f64` (column-major operands, consistent shapes):
/// the kernels of `ctx`'s tier, or `false` (nothing done).
#[inline]
fn small_simd_f64(
    ctx: GemmContext,
    alpha: f64,
    a: &MatRef<'_, f64>,
    b: &MatRef<'_, f64>,
    c: &mut MatMut<'_, f64>,
) -> bool {
    match ctx.tier() {
        SimdTier::Neon => neon_small_f64(alpha, a, b, c),
        SimdTier::Avx2Fma => {
            avx2_small_applies(&ctx.policy, a.nrows(), b.ncols())
                && avx2_small_f64(ctx, alpha, a, b, c)
        }
        SimdTier::Sse42 | SimdTier::Portable => false,
    }
}

/// [`small_simd`] for `f32` (column-major operands, consistent shapes):
/// the kernels of `ctx`'s tier, or `false` (nothing done).
#[inline]
fn small_simd_f32(
    ctx: GemmContext,
    alpha: f32,
    a: &MatRef<'_, f32>,
    b: &MatRef<'_, f32>,
    c: &mut MatMut<'_, f32>,
) -> bool {
    match ctx.tier() {
        SimdTier::Neon => neon_small_f32(alpha, a, b, c),
        SimdTier::Avx2Fma => {
            avx2_small_applies(&ctx.policy, a.nrows(), b.ncols())
                && avx2_small_f32(ctx, alpha, a, b, c)
        }
        SimdTier::Sse42 | SimdTier::Portable => false,
    }
}

/// Whether the AVX2 + FMA small kernels take an `m × n` product under
/// `policy`: only with `policy.small_simd_x86` (off in the legacy policy, so
/// the default build keeps the 0.2.3 portable loops), and from `m >= 5`,
/// `n >= 4`. The router's `RouteInputs::simd_small` applies the same rule to
/// the AVX2 + FMA tier.
#[inline]
const fn avx2_small_applies(policy: &GemmPolicy, m: usize, n: usize) -> bool {
    policy.small_simd_x86 && m >= 5 && n >= 4
}

/// The AVX2 + FMA `f64` small kernel on column-major operands with
/// consistent shapes, when `ctx` is on that tier; `false` otherwise, and
/// always in builds for other architectures (where the tier never occurs).
#[inline]
fn avx2_small_f64(
    ctx: GemmContext,
    alpha: f64,
    a: &MatRef<'_, f64>,
    b: &MatRef<'_, f64>,
    c: &mut MatMut<'_, f64>,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
        if ctx.tier() != SimdTier::Avx2Fma
            || a.col_stride() != 1
            || b.col_stride() != 1
            || b.nrows() != k
            || c.nrows() != m
            || c.ncols() != n
        {
            return false;
        }
        let (lda, ldb, ldc) = (a.row_stride(), b.row_stride(), c.row_stride());
        let c_ptr = c.as_mut_ptr();
        // SAFETY: `GemmContext::tier()` is `Avx2Fma` only when AVX2 and FMA
        // were detected at run time and the build allows 256-bit vectors
        // (`SimdTier::from_kind` on the capped kernel kind), so both target
        // features of the kernel are present. A is `m × k` and B `k × n` with
        // unit row-to-row stride (checked above), so `A[i, p]` is the view's
        // element at `as_ptr() + i + p·lda` and `B[p, j]` at
        // `as_ptr() + p + j·ldb`; C is `m × n` (checked) and a `MatMut`
        // always has unit row-to-row stride, so `C[i, j]` is at
        // `c_ptr + i + j·ldc`. The kernel touches only those elements (masked
        // tails, no access outside the three areas), C is borrowed exclusively
        // through `c`, and the borrows of `a` / `b` keep A and B readable and
        // disjoint from C for the call.
        unsafe {
            avx2::small_f64_avx2(m, n, k, alpha, a.as_ptr(), lda, b.as_ptr(), ldb, c_ptr, ldc);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ctx, alpha, a, b, c);
        false
    }
}

/// The AVX2 + FMA `f32` small kernel (see [`avx2_small_f64`]).
#[inline]
fn avx2_small_f32(
    ctx: GemmContext,
    alpha: f32,
    a: &MatRef<'_, f32>,
    b: &MatRef<'_, f32>,
    c: &mut MatMut<'_, f32>,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
        if ctx.tier() != SimdTier::Avx2Fma
            || a.col_stride() != 1
            || b.col_stride() != 1
            || b.nrows() != k
            || c.nrows() != m
            || c.ncols() != n
        {
            return false;
        }
        let (lda, ldb, ldc) = (a.row_stride(), b.row_stride(), c.row_stride());
        let c_ptr = c.as_mut_ptr();
        // SAFETY: as in `avx2_small_f64`: the `Avx2Fma` tier guarantees AVX2
        // and FMA at run time under a build that allows 256 bits, and the
        // checked shapes and unit row-to-row strides put every element the
        // kernel accesses (`A[i, p]`, `B[p, j]`, `C[i, j]` for `i < m`,
        // `p < k`, `j < n`) inside the views; C is borrowed exclusively.
        unsafe {
            avx2::small_f32_avx2(m, n, k, alpha, a.as_ptr(), lda, b.as_ptr(), ldb, c_ptr, ldc);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ctx, alpha, a, b, c);
        false
    }
}

/// The NEON `f64` small kernel when the problem has a full 4×4 tile;
/// `false` otherwise, and always in builds without it (other architectures,
/// `force-scalar`, where the NEON tier never occurs).
#[inline]
fn neon_small_f64(
    alpha: f64,
    a: &MatRef<'_, f64>,
    b: &MatRef<'_, f64>,
    c: &mut MatMut<'_, f64>,
) -> bool {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        if a.nrows() >= 4 && b.ncols() >= 4 {
            gemm_small_f64_simd(alpha, a, b, c);
            return true;
        }
        false
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
    {
        let _ = (alpha, a, b, c);
        false
    }
}

/// The NEON `f32` small kernel when the problem has a full 4×4 tile;
/// `false` otherwise, and always in builds without it.
#[inline]
fn neon_small_f32(
    alpha: f32,
    a: &MatRef<'_, f32>,
    b: &MatRef<'_, f32>,
    c: &mut MatMut<'_, f32>,
) -> bool {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        if a.nrows() >= 4 && b.ncols() >= 4 {
            gemm_small_f32_simd(alpha, a, b, c);
            return true;
        }
        false
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
    {
        let _ = (alpha, a, b, c);
        false
    }
}

/// Portable small GEMM accumulation: `C += alpha * A * B` (no beta).
///
/// Dispatches to the size-specialized loops; operands are read through the
/// view's `Index`, so arbitrary strides (e.g. transposed views) are honoured.
fn gemm_small_acc<T: Field>(alpha: T, a: &MatRef<'_, T>, b: &MatRef<'_, T>, c: &mut MatMut<'_, T>) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

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

/// NEON (aarch64) small GEMM for f64: `C += alpha * A * B`.
///
/// Beta is handled by the caller (see [`gemm_small`]); C is accumulated into,
/// never overwritten. Operands that are not plain column-major
/// (`col_stride() != 1`) and very small problems take the portable loops.
///
/// # Panics
///
/// Panics if `B.nrows != A.ncols`, `C.nrows != A.nrows` or
/// `C.ncols != B.ncols`.
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

    // The NEON tiles address A, B and C through raw pointers, so the shapes
    // must be consistent.
    assert_eq!(b.nrows(), k, "B.nrows must equal A.ncols");
    assert_eq!(c.nrows(), m, "C.nrows must equal A.nrows");
    assert_eq!(c.ncols(), n, "C.ncols must equal B.ncols");

    // Handle beta scaling is done by caller

    if alpha == 0.0 {
        return;
    }

    // For very small matrices SIMD overhead isn't worth it, and the tiles
    // assume unit row-to-row strides (column-major A and B).
    if m * n * k < 64 || a.col_stride() != 1 || b.col_stride() != 1 {
        gemm_small_acc(alpha, a, b, c);
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
                // SAFETY: the tile is full (`ib + 4 <= m`, `jb + 4 <= n`), the
                // shapes were asserted above and A/B have unit row-to-row
                // strides, so every element the kernel reads (A[ib..ib+4, ..k],
                // B[..k, jb..jb+4]) and writes (C[ib..ib+4, jb..jb+4]) lies in
                // its view.
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

/// 4x4 NEON kernel for f64:
/// `C[i..i+4, j..j+4] += alpha * A[i..i+4, ..k] * B[..k, j..j+4]`.
///
/// # Safety
///
/// `i + 4 <= A.nrows == C.nrows`, `j + 4 <= B.ncols == C.ncols`,
/// `k == A.ncols == B.nrows`, and A and B must have unit row-to-row stride
/// (`col_stride() == 1`).
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

/// NEON (aarch64) small GEMM for f32: `C += alpha * A * B`.
///
/// See [`gemm_small_f64_simd`]: beta is handled by the caller, and strided or
/// very small operands take the portable loops.
///
/// # Panics
///
/// Panics if `B.nrows != A.ncols`, `C.nrows != A.nrows` or
/// `C.ncols != B.ncols`.
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

    // The NEON tiles address A, B and C through raw pointers, so the shapes
    // must be consistent.
    assert_eq!(b.nrows(), k, "B.nrows must equal A.ncols");
    assert_eq!(c.nrows(), m, "C.nrows must equal A.nrows");
    assert_eq!(c.ncols(), n, "C.ncols must equal B.ncols");

    if alpha == 0.0 {
        return;
    }

    // For very small matrices SIMD overhead isn't worth it, and the tiles
    // assume unit row-to-row strides (column-major A and B).
    if m * n * k < 64 || a.col_stride() != 1 || b.col_stride() != 1 {
        gemm_small_acc(alpha, a, b, c);
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
                // SAFETY: the tile is full (`ib + 4 <= m`, `jb + 4 <= n`), the
                // shapes were asserted above and A/B have unit row-to-row
                // strides, so every element the kernel reads and writes lies
                // in its view.
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

/// 4x4 NEON kernel for f32 (same contract as [`gemm_4x4_neon`]).
///
/// # Safety
///
/// `i + 4 <= A.nrows == C.nrows`, `j + 4 <= B.ncols == C.ncols`,
/// `k == A.ncols == B.nrows`, and A and B must have unit row-to-row stride.
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
#[path = "gemm_small_tests.rs"]
mod tests;
