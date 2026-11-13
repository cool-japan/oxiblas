//! Thin GEMM products (`m == 1` or `n == 1`) computed as one GEMV.
//!
//! # Contract
//!
//! For `f64` / `f32`, `C = alpha·op(A)·op(B) + beta·C` with `n == 1` or
//! `m == 1` is exactly one GEMV on the corresponding vectors (the GEMM core
//! has already returned for `k == 0` and `alpha == 0`):
//!
//! - `n == 1`, including `m == n == 1`: `gemv(op_A, A, x = op(B)[:, 0],
//!   y = C[:, 0])`.
//! - `m == 1` (and `n > 1`): `gemv(op_B, B, x = op(A)[0, :], y = C[0, :])`
//!   with the GEMV transpose `Trans` for `op(B) = B` and `NoTrans` for
//!   `op(B) = Bᵀ` (because `C[0, :]ᵀ = op(B)ᵀ·op(A)[0, :]ᵀ`). The row of C has
//!   the stride of C's leading dimension; it is gathered into a contiguous
//!   buffer and scattered back.
//!
//! As in [`gemv`](fn@crate::level2::gemv), `y` is first scaled by `beta`: set to
//! zero without being read when `beta == 0`, and replaced by `beta * y` when
//! `beta` is neither 0 nor 1. Then `y += alpha·(product)` with the
//! per-element operation sequence of the aarch64 NEON GEMV kernels:
//!
//! - **axpy form** (GEMV `NoTrans` on a column-major `S`): `t_j = alpha·x[j]`,
//!   then `y[i] = S[i, j]·t_j + y[i]` for `j = 0, 1, …` in order.
//! - **dot form** (GEMV `Trans`): `y[j] = y[j] + alpha·d_j`, where `d_j` is
//!   the dot product of column `j` of `S` with `x` in the NEON order. With
//!   `L` lanes per 128-bit vector (2 for `f64`, 4 for `f32`), rows go to
//!   `4·L` partial sums `acc[q][l]`: in steps of `4·L` rows, row
//!   `i + q·L + l` goes to `acc[q][l]`; then, if at least `2·L` rows remain,
//!   the next `2·L` rows go to `acc[0]` and `acc[1]` the same way; then, if
//!   at least `L` rows remain, the next `L` rows go to `acc[2]`. The lanes are
//!   combined as `v[l] = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l])`,
//!   then pairwise (`v[0] + v[1]` for `f64`, `(v[0] + v[1]) + (v[2] + v[3])`
//!   for `f32`), and the remaining rows are added one at a time in order.
//!
//! The product `t_j = alpha·x[j]` of the axpy form and the final update
//! `y[j] = y[j] + alpha·d_j` of the dot form are a plain multiply (and a
//! plain add) on every tier: the NEON kernels write `y[j] += alpha * d_j`,
//! which is not fused. Every other multiply-add of these sequences (the axpy
//! chain, the partial sums and the row tail of the dot form) is fused on the
//! NEON and AVX2 + FMA tiers, and in the portable kernels where the target
//! has a native FMA (aarch64, and x86-64 built with the `fma` target
//! feature); elsewhere the portable kernels multiply and then add, because a
//! software FMA would make them slower than the blocked driver. On aarch64
//! (the NEON tier, and the portable kernels, which fuse there) the results
//! are therefore bit-identical to the NEON GEMV kernels, and so to
//! [`gemv`](fn@crate::level2::gemv) on the same column-major operands in
//! builds without `force-scalar`, where `gemv` runs those kernels (under
//! `force-scalar`, `gemv` takes its own blocked loops, in another order). The
//! AVX2 + FMA tier gives the bits of the NEON kernels except the sign and
//! payload of NaN results: each instruction set has its own default NaN (an
//! invalid operation such as `0·∞` gives a positive quiet NaN on aarch64 and
//! a negative one on x86-64), while NaN results sit at the same positions on
//! both. On every tier the result of an element depends only on its own
//! operands, never on its position, on alignment, on the size of the part
//! of `y` it is computed in, or on the thread count.
//!
//! # Kernels per tier
//!
//! The kernel table is chosen from the plan's `SimdTier`:
//!
//! - `Neon`: the aarch64 NEON GEMV kernels of `level2::gemv::neon`
//!   (`notrans_*` for the axpy form, `trans_*` for the dot form).
//! - `Avx2Fma`: the x86-64 AVX2 + FMA kernels of `gemm_thin_avx2.rs`, which
//!   hold the dot form's `4·L` partial sums in two ymm registers per column.
//! - `Sse42` and `Portable`: the portable kernels of this module.
//!
//! # Layout
//!
//! The large operand (A for `n == 1`, B for `m == 1`) is read in place: as
//! stored when its rows are adjacent (`col_stride() == 1`), or as its
//! column-major transpose with the GEMV transpose flipped when its columns are
//! adjacent (`row_stride() == 1`, e.g. a `transpose()` view). Other layouts
//! are never routed here. The vector `x` is read in place when it is
//! contiguous and gathered into the thread's packing buffer otherwise. No
//! column is skipped for a zero `x[j]`: NaN and infinity in the large operand
//! propagate as in the reference product.
//!
//! # Parallel split
//!
//! When the policy's thin work gate (`GemmPolicy::thin_par_gate`) opens for
//! the caller's `Par`, `y` is split into contiguous parts of whole 64-element
//! runs, one per thread, and every part runs the same kernel on its rows
//! (axpy form) or columns (dot form) of the large operand. The kernels are
//! position-independent, so the result is bit for bit the sequential one.

use super::gemm_kernel::GemmKernel;
use super::gemm_packing::{PackBuffer, PackSlot};
use super::gemm_plan::{GemmContext, SimdTier};
use super::trsm::Trans;
use crate::level1::typed::same_value;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{MatMut, MatRef};

/// Whether multiply-adds are fused: the target has a native FMA.
const FUSED: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// The two GEMV forms on a column-major matrix `S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// GEMV `NoTrans`: `y += alpha·S·x`, column by column.
    Axpy,
    /// GEMV `Trans`: `y[j] += alpha·(S[:, j]·x)`.
    Dot,
}

/// A GEMV kernel of one form: `(alpha, S, x, y)` with a column-major `S`
/// (`col_stride() == 1`).
type ThinKernel<T> = fn(T, MatRef<'_, T>, &[T], &mut [T]);

/// The GEMV kernels for one element type.
#[derive(Clone, Copy)]
struct ThinKernels<T: Scalar> {
    /// `y += alpha·S·x` (`x.len() == S.ncols`, `y.len() == S.nrows`).
    axpy: ThinKernel<T>,
    /// `y += alpha·Sᵀ·x` (`x.len() == S.nrows`, `y.len() == S.ncols`).
    dot: ThinKernel<T>,
}

/// The portable kernels for `f64`.
const PORTABLE_F64: ThinKernels<f64> = ThinKernels {
    axpy: axpy_portable::<f64>,
    dot: dot_portable::<f64>,
};

/// The portable kernels for `f32`.
const PORTABLE_F32: ThinKernels<f32> = ThinKernels {
    axpy: axpy_portable::<f32>,
    dot: dot_portable::<f32>,
};

/// The NEON GEMV kernels for `f64`.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
const NEON_F64: ThinKernels<f64> = ThinKernels {
    axpy: crate::level2::gemv::neon::notrans_f64,
    dot: crate::level2::gemv::neon::trans_f64,
};

/// The NEON GEMV kernels for `f32`.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
const NEON_F32: ThinKernels<f32> = ThinKernels {
    axpy: crate::level2::gemv::neon::notrans_f32,
    dot: crate::level2::gemv::neon::trans_f32,
};

/// The AVX2 + FMA kernels for `f64`.
#[cfg(target_arch = "x86_64")]
const AVX2_F64: ThinKernels<f64> = ThinKernels {
    axpy: avx2::axpy_f64,
    dot: avx2::dot_f64,
};

/// The AVX2 + FMA kernels for `f32`.
#[cfg(target_arch = "x86_64")]
const AVX2_F32: ThinKernels<f32> = ThinKernels {
    axpy: avx2::axpy_f32,
    dot: avx2::dot_f32,
};

/// The `f64` kernels of `tier` (module docs, "Kernels per tier").
///
/// A tier whose kernels are not compiled into this build (NEON off aarch64
/// or under `force-scalar`, AVX2 + FMA off x86-64) gets the portable kernels;
/// a `GemmContext` never reports such a tier.
fn tier_kernels_f64(tier: SimdTier) -> ThinKernels<f64> {
    match tier {
        #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
        SimdTier::Neon => NEON_F64,
        #[cfg(target_arch = "x86_64")]
        SimdTier::Avx2Fma => AVX2_F64,
        _ => PORTABLE_F64,
    }
}

/// The `f32` kernels of `tier` (as [`tier_kernels_f64`]).
fn tier_kernels_f32(tier: SimdTier) -> ThinKernels<f32> {
    match tier {
        #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
        SimdTier::Neon => NEON_F32,
        #[cfg(target_arch = "x86_64")]
        SimdTier::Avx2Fma => AVX2_F32,
        _ => PORTABLE_F32,
    }
}

/// The kernels of `tier` for `T`, or `None` when `T` is neither `f64` nor
/// `f32`.
///
/// The tables are selected by type identity (`level1::typed::same_value`),
/// so `T` needs no extra trait bound.
fn thin_kernels<T: Field>(tier: SimdTier) -> Option<ThinKernels<T>> {
    same_value::<ThinKernels<f64>, ThinKernels<T>>(tier_kernels_f64(tier))
        .or_else(|| same_value::<ThinKernels<f32>, ThinKernels<T>>(tier_kernels_f32(tier)))
}

/// Computes `C = alpha·op(A)·op(B) + beta·C` for `m == 1` or `n == 1` as one
/// GEMV (see the module docs for the contract).
///
/// `ctx` is the plan's tier and policy, `par` the caller's parallelism (see
/// the module docs for the parallel split). Returns `false` without touching
/// C when the problem is not thin, `T` is neither `f64` nor `f32`, or the
/// large operand has no unit stride. The operand shapes must be consistent
/// (the GEMM core checks them), and `alpha == 0` / `k == 0` must have been
/// handled by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_thin<T: Field + GemmKernel + bytemuck::Zeroable>(
    ctx: GemmContext,
    trans_a: Trans,
    trans_b: Trans,
    alpha: T,
    a: &MatRef<'_, T>,
    b: &MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    par: Par,
) -> bool {
    let Some(kernels) = thin_kernels::<T>(ctx.tier()) else {
        return false;
    };
    let (m, n) = (c.nrows(), c.ncols());
    if m == 0 || n == 0 {
        return false;
    }
    if n == 1 {
        // gemv(op_A, A, x = op(B)[:, 0], y = C[:, 0]).
        let Some((s, form)) = normalize(*a, trans_a == Trans::NoTrans) else {
            return false;
        };
        let x_view = op_vector(*b, trans_b, false);
        let mut x_buf = None;
        let x = vector_in_place_or_gathered(x_view, &mut x_buf);
        let Some(y) = c.col_as_slice_mut(0) else {
            return false;
        };
        scale_like_gemv(beta, y);
        run_split(kernels, form, alpha, s, x, y, thin_threads(&ctx, &s, par));
        return true;
    }
    if m == 1 {
        // gemv(op_B, B, x = op(A)[0, :], y = C[0, :]): `Trans` on B for
        // op(B) = B, `NoTrans` for op(B) = Bᵀ.
        let Some((s, form)) = normalize(*b, trans_b != Trans::NoTrans) else {
            return false;
        };
        let x_view = op_vector(*a, trans_a, true);
        let mut x_buf = None;
        let x = vector_in_place_or_gathered(x_view, &mut x_buf);
        let mut y_buf = PackBuffer::<T>::take(PackSlot::B, n);
        let y = &mut y_buf.as_mut_slice()[..n];
        if beta == T::zero() {
            // C is not read.
            y.fill(T::zero());
        } else {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj = c[(0, j)];
            }
            scale_like_gemv(beta, y);
        }
        run_split(kernels, form, alpha, s, x, y, thin_threads(&ctx, &s, par));
        for (j, &yj) in y.iter().enumerate() {
            c.set(0, j, yj);
        }
        return true;
    }
    false
}

/// `y := beta·y` exactly as [`gemv`](fn@crate::level2::gemv) does it: zeros
/// without reading `y` when `beta == 0`, `beta * y` when `beta` is neither 0
/// nor 1.
#[inline]
fn scale_like_gemv<T: Field>(beta: T, y: &mut [T]) {
    if beta == T::zero() {
        for yi in y.iter_mut() {
            *yi = T::zero();
        }
    } else if beta != T::one() {
        for yi in y.iter_mut() {
            *yi = beta * *yi;
        }
    }
}

/// The large operand as a column-major view, with the GEMV form on it.
///
/// `notrans` is whether the GEMV on the stored operand is `NoTrans`. A view
/// with adjacent rows is used as stored; one with adjacent columns is read as
/// its transpose, which flips the GEMV transpose.
fn normalize<'a, T: Scalar>(stored: MatRef<'a, T>, notrans: bool) -> Option<(MatRef<'a, T>, Form)> {
    let form = |notrans: bool| if notrans { Form::Axpy } else { Form::Dot };
    if stored.col_stride() == 1 {
        Some((stored, form(notrans)))
    } else if stored.row_stride() == 1 {
        Some((stored.transpose(), form(!notrans)))
    } else {
        None
    }
}

/// The vector of a thin product as a `len × 1` view of its stored operand:
/// the only column of `op(X)`, or (`op_is_row`) the only row of `op(X)`.
///
/// For the real element types of this path `op(X)` is `X` for `NoTrans` and
/// `Xᵀ` otherwise.
fn op_vector<'a, T: Scalar>(stored: MatRef<'a, T>, trans: Trans, op_is_row: bool) -> MatRef<'a, T> {
    if (trans == Trans::NoTrans) != op_is_row {
        stored
    } else {
        stored.transpose()
    }
}

/// The elements of the `len × 1` view `v` as a slice: in place when they are
/// contiguous, otherwise gathered into this thread's A packing buffer (held in
/// `buf`).
fn vector_in_place_or_gathered<'v, T: Field + GemmKernel + bytemuck::Zeroable>(
    v: MatRef<'v, T>,
    buf: &'v mut Option<PackBuffer<T>>,
) -> &'v [T] {
    if let Some(x) = v.col_as_slice(0) {
        return x;
    }
    let len = v.nrows();
    let gathered = buf.insert(PackBuffer::<T>::take(PackSlot::A, len));
    let x = &mut gathered.as_mut_slice()[..len];
    for (p, xp) in x.iter_mut().enumerate() {
        *xp = v[(p, 0)];
    }
    x
}

/// Threads for the thin product on the large operand `s`: the thread count
/// of `par` when the policy's thin work gate opens for `s.nrows() ·
/// s.ncols()` multiply-adds, otherwise 1.
fn thin_threads<T: Scalar>(ctx: &GemmContext, s: &MatRef<'_, T>, par: Par) -> usize {
    let work = s.nrows().saturating_mul(s.ncols());
    if ctx
        .policy
        .thin_par_threshold()
        .should_parallelize(work, par)
    {
        par.num_threads()
    } else {
        1
    }
}

/// Length of each part of `y` when `len` elements are split for `threads`
/// threads: whole runs of [`SPLIT_ALIGN`] elements.
#[cfg(feature = "parallel")]
fn part_len(len: usize, threads: usize) -> usize {
    len.div_ceil(threads.max(1)).div_ceil(SPLIT_ALIGN).max(1) * SPLIT_ALIGN
}

/// Every part of a split `y` but the last is a multiple of this many
/// elements (a multiple of every vector step of the kernels).
#[cfg(feature = "parallel")]
const SPLIT_ALIGN: usize = 64;

/// Runs the kernel of `form` with `y` split into parts of whole
/// [`SPLIT_ALIGN`]-element runs, one per thread (sequentially for one
/// thread or a `y` of one part). Bit for bit the sequential result: every
/// part runs the same position-independent kernel on its rows (axpy form) or
/// columns (dot form) of `s`.
#[cfg(feature = "parallel")]
fn run_split<T: Scalar + Send + Sync>(
    kernels: ThinKernels<T>,
    form: Form,
    alpha: T,
    s: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    threads: usize,
) {
    use rayon::prelude::*;

    let len = part_len(y.len(), threads);
    if threads <= 1 || len >= y.len() {
        run(kernels, form, alpha, s, x, y);
        return;
    }
    let (rows, cols) = (s.nrows(), s.ncols());
    y.par_chunks_mut(len).enumerate().for_each(|(p, part)| {
        let start = p * len;
        let sub = match form {
            Form::Axpy => s.submatrix(start, 0, part.len(), cols),
            Form::Dot => s.submatrix(0, start, rows, part.len()),
        };
        run(kernels, form, alpha, sub, x, part);
    });
}

/// Without `parallel` there is one thread: the kernel of `form` on all of
/// `y`.
#[cfg(not(feature = "parallel"))]
fn run_split<T: Scalar>(
    kernels: ThinKernels<T>,
    form: Form,
    alpha: T,
    s: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    _threads: usize,
) {
    run(kernels, form, alpha, s, x, y);
}

/// Runs the kernel of `form`.
#[inline]
fn run<T: Scalar>(
    kernels: ThinKernels<T>,
    form: Form,
    alpha: T,
    s: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
) {
    match form {
        Form::Axpy => (kernels.axpy)(alpha, s, x, y),
        Form::Dot => (kernels.dot)(alpha, s, x, y),
    }
}

// ---------------------------------------------------------------------------
// Portable kernels
// ---------------------------------------------------------------------------

/// The element types of the portable kernels.
trait ThinElem: Field {
    /// `self·b + c`, fused where the target has a native FMA.
    fn madd(self, b: Self, c: Self) -> Self;

    /// The dot product `Σ c[i]·x[i]` in the NEON order (module docs).
    fn dot(c: &[Self], x: &[Self]) -> Self;
}

impl ThinElem for f64 {
    #[inline(always)]
    fn madd(self, b: Self, c: Self) -> Self {
        if FUSED {
            self.mul_add(b, c)
        } else {
            self * b + c
        }
    }

    #[inline]
    fn dot(c: &[Self], x: &[Self]) -> Self {
        dot_neon_order::<Self, 2>(c, x)
    }
}

impl ThinElem for f32 {
    #[inline(always)]
    fn madd(self, b: Self, c: Self) -> Self {
        if FUSED {
            self.mul_add(b, c)
        } else {
            self * b + c
        }
    }

    #[inline]
    fn dot(c: &[Self], x: &[Self]) -> Self {
        dot_neon_order::<Self, 4>(c, x)
    }
}

/// Column `j` of a column-major view (empty if `j` is out of range).
#[inline]
fn column<'a, T: Scalar>(s: &MatRef<'a, T>, j: usize) -> &'a [T] {
    s.col_as_slice(j).unwrap_or_default()
}

/// `y += alpha·S·x` (axpy form, module docs) for a column-major `S`.
///
/// Four columns are applied per pass over `y`, with the per-element sequence
/// of one column at a time.
///
/// # Panics
///
/// Panics unless `s.col_stride() == 1`, `x.len() == s.ncols()` and
/// `y.len() == s.nrows()`.
fn axpy_portable<E: ThinElem>(alpha: E, s: MatRef<'_, E>, x: &[E], y: &mut [E]) {
    assert!(
        s.col_stride() == 1 && x.len() == s.ncols() && y.len() == s.nrows(),
        "axpy_portable: operands do not match a column-major view"
    );
    let n = x.len();
    let mut j = 0;
    while j + 4 <= n {
        let (t0, t1, t2, t3) = (
            alpha * x[j],
            alpha * x[j + 1],
            alpha * x[j + 2],
            alpha * x[j + 3],
        );
        let (c0, c1, c2, c3) = (
            column(&s, j),
            column(&s, j + 1),
            column(&s, j + 2),
            column(&s, j + 3),
        );
        for ((((yi, &a0), &a1), &a2), &a3) in y.iter_mut().zip(c0).zip(c1).zip(c2).zip(c3) {
            let v = a0.madd(t0, *yi);
            let v = a1.madd(t1, v);
            let v = a2.madd(t2, v);
            *yi = a3.madd(t3, v);
        }
        j += 4;
    }
    while j < n {
        let t = alpha * x[j];
        for (yi, &aij) in y.iter_mut().zip(column(&s, j)) {
            *yi = aij.madd(t, *yi);
        }
        j += 1;
    }
}

/// `y += alpha·Sᵀ·x` (dot form, module docs) for a column-major `S`.
///
/// # Panics
///
/// Panics unless `s.col_stride() == 1`, `x.len() == s.nrows()` and
/// `y.len() == s.ncols()`.
fn dot_portable<E: ThinElem>(alpha: E, s: MatRef<'_, E>, x: &[E], y: &mut [E]) {
    assert!(
        s.col_stride() == 1 && x.len() == s.nrows() && y.len() == s.ncols(),
        "dot_portable: operands do not match a column-major view"
    );
    for (j, yj) in y.iter_mut().enumerate() {
        let d = E::dot(column(&s, j), x);
        *yj = *yj + alpha * d;
    }
}

/// `Σ c[i]·x[i]` over the common length, in the NEON order with `L` lanes
/// per vector (module docs).
#[inline]
fn dot_neon_order<E: ThinElem, const L: usize>(c: &[E], x: &[E]) -> E {
    let len = c.len().min(x.len());
    let (c, x) = (&c[..len], &x[..len]);
    let zero = E::zero();
    let mut acc = [[zero; L]; 4];
    let step = 4 * L;
    let body = len - len % step;
    let (c_body, mut c_rest) = c.split_at(body);
    let (x_body, mut x_rest) = x.split_at(body);
    for (cs, xs) in c_body.chunks_exact(step).zip(x_body.chunks_exact(step)) {
        for ((accq, cq), xq) in acc
            .iter_mut()
            .zip(cs.chunks_exact(L))
            .zip(xs.chunks_exact(L))
        {
            for ((a, &cv), &xv) in accq.iter_mut().zip(cq).zip(xq) {
                *a = cv.madd(xv, *a);
            }
        }
    }
    if c_rest.len() >= 2 * L {
        let (c2, c_tail) = c_rest.split_at(2 * L);
        let (x2, x_tail) = x_rest.split_at(2 * L);
        for ((accq, cq), xq) in acc
            .iter_mut()
            .take(2)
            .zip(c2.chunks_exact(L))
            .zip(x2.chunks_exact(L))
        {
            for ((a, &cv), &xv) in accq.iter_mut().zip(cq).zip(xq) {
                *a = cv.madd(xv, *a);
            }
        }
        (c_rest, x_rest) = (c_tail, x_tail);
    }
    if c_rest.len() >= L {
        let (c1, c_tail) = c_rest.split_at(L);
        let (x1, x_tail) = x_rest.split_at(L);
        for ((a, &cv), &xv) in acc[2].iter_mut().zip(c1).zip(x1) {
            *a = cv.madd(xv, *a);
        }
        (c_rest, x_rest) = (c_tail, x_tail);
    }
    let mut lanes = [zero; L];
    for (l, v) in lanes.iter_mut().enumerate() {
        *v = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    }
    let mut s = pairwise_sum(lanes);
    for (&cv, &xv) in c_rest.iter().zip(x_rest) {
        s = cv.madd(xv, s);
    }
    s
}

/// `v[0] + v[1]` for two lanes, `(v[0] + v[1]) + (v[2] + v[3])` for four:
/// adjacent pairs, repeatedly (the NEON across-vector add).
#[inline]
fn pairwise_sum<E: Field, const L: usize>(mut v: [E; L]) -> E {
    let mut len = L;
    while len > 1 {
        let half = len / 2;
        for p in 0..half {
            v[p] = v[2 * p] + v[2 * p + 1];
        }
        len = half;
    }
    v.first().copied().unwrap_or_else(E::zero)
}

#[cfg(target_arch = "x86_64")]
#[path = "gemm_thin_avx2.rs"]
mod avx2;

#[cfg(test)]
#[path = "gemm_thin_tests.rs"]
mod tests;
