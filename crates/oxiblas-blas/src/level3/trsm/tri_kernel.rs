//! Register-tiled kernels for the diagonal blocks of TRSM and TRMM.
//!
//! Both routines are reduced (see [`super::tri_driver`]) to one canonical
//! problem on a diagonal block: a *lower* triangular `kb x kb` matrix `L`
//! applied to a packed right-hand-side micro-panel `X` of `kb` rows and `NR`
//! columns,
//!
//! - TRSM: overwrite `X` with `L⁻¹ X` (forward substitution),
//! - TRMM: overwrite `X` with `alpha · L X`.
//!
//! # Packed layouts
//!
//! `L` is packed by [`super::tri_pack::pack_tri_block`] into row panels of
//! `MR` rows. Panel `p` (rows `p·MR .. p·MR + MR`, zero-padded past `kb`)
//! holds, for every column `k < p·MR`, the `MR` entries `L[p·MR + r, k]`
//! (the *rectangular* part, `p·MR · MR` values), followed by the `MR x MR`
//! diagonal tile stored column by column (`tile[c·MR + r] = L[p·MR + r,
//! p·MR + c]` for `r > c`, zero for `r < c`). The diagonal slot `tile[c·MR +
//! c]` holds what the kernel needs: the reciprocal of the diagonal (TRSM),
//! the diagonal itself (TRSM in exact-division mode, and TRMM), or nothing
//! for a unit diagonal (the kernels never read it then).
//!
//! `X` is packed row-major: `xp[i·NR + c]`, `kb` rounded up to a multiple of
//! `MR` rows (padding rows are zero), `NR` columns (padding columns are zero
//! and never written back).
//!
//! # Arithmetic
//!
//! Every column of `X` is processed by exactly the same sequence of
//! operations, independently of the other columns of the micro-panel, so
//! the result for a right-hand side does not depend on which micro-panel or
//! thread handled it. On aarch64 the kernels for `f32`/`f64`/`Complex32`/
//! `Complex64` use fused multiply-adds (one rounding per update): explicit
//! NEON for the real types (portable kernels with fused scalar operations
//! under `force-scalar`, and for the complex types). Other targets, and
//! element types other than those four, use plain multiplies and
//! subtractions.

use core::any::Any;
use num_complex::{Complex32, Complex64};
use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Multiply-add primitives of the tile kernels.
pub(crate) trait TileOps<T> {
    /// Returns `c - a·b`.
    fn nmadd(a: T, b: T, c: T) -> T;
    /// Returns `c + a·b`.
    fn madd(a: T, b: T, c: T) -> T;
}

/// Plain (separately rounded) arithmetic, valid for every [`Field`].
pub(crate) struct PlainOps;

impl<T: Field> TileOps<T> for PlainOps {
    #[inline(always)]
    fn nmadd(a: T, b: T, c: T) -> T {
        c - a * b
    }

    #[inline(always)]
    fn madd(a: T, b: T, c: T) -> T {
        c + a * b
    }
}

/// Fused multiply-adds. Only built for aarch64, where `mul_add` is a single
/// hardware instruction (on targets without FMA it would be a libm call).
#[cfg(target_arch = "aarch64")]
pub(crate) struct FusedOps;

#[cfg(target_arch = "aarch64")]
impl TileOps<f64> for FusedOps {
    #[inline(always)]
    fn nmadd(a: f64, b: f64, c: f64) -> f64 {
        (-a).mul_add(b, c)
    }

    #[inline(always)]
    fn madd(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
}

#[cfg(target_arch = "aarch64")]
impl TileOps<f32> for FusedOps {
    #[inline(always)]
    fn nmadd(a: f32, b: f32, c: f32) -> f32 {
        (-a).mul_add(b, c)
    }

    #[inline(always)]
    fn madd(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

/// Complex `c ∓ a·b` with every real product fused into the accumulation.
#[cfg(target_arch = "aarch64")]
macro_rules! fused_complex_ops {
    ($c:ty) => {
        impl TileOps<$c> for FusedOps {
            #[inline(always)]
            fn nmadd(a: $c, b: $c, c: $c) -> $c {
                // c - a·b = (c.re - a.re·b.re + a.im·b.im) + (c.im - a.re·b.im - a.im·b.re)i
                <$c>::new(
                    a.im.mul_add(b.im, (-a.re).mul_add(b.re, c.re)),
                    (-a.im).mul_add(b.re, (-a.re).mul_add(b.im, c.im)),
                )
            }

            #[inline(always)]
            fn madd(a: $c, b: $c, c: $c) -> $c {
                // c + a·b = (c.re + a.re·b.re - a.im·b.im) + (c.im + a.re·b.im + a.im·b.re)i
                <$c>::new(
                    (-a.im).mul_add(b.im, a.re.mul_add(b.re, c.re)),
                    a.im.mul_add(b.re, a.re.mul_add(b.im, c.im)),
                )
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
fused_complex_ops!(Complex64);
#[cfg(target_arch = "aarch64")]
fused_complex_ops!(Complex32);

/// Ops used for the four fast element types on this target.
#[cfg(target_arch = "aarch64")]
type FastOps = FusedOps;
#[cfg(not(target_arch = "aarch64"))]
type FastOps = PlainOps;

/// Register tile (`MR` rows, `NR` right-hand sides) of the `f64` kernels:
/// sixteen accumulator registers out of aarch64's 32.
#[cfg(target_arch = "aarch64")]
const F64_TILE: (usize, usize) = (4, 8);
/// Register tile of the `f32` kernels.
#[cfg(target_arch = "aarch64")]
const F32_TILE: (usize, usize) = (8, 8);
/// Smaller tiles where only 16 vector registers exist (x86_64 SSE/AVX).
#[cfg(not(target_arch = "aarch64"))]
const F64_TILE: (usize, usize) = (4, 4);
#[cfg(not(target_arch = "aarch64"))]
const F32_TILE: (usize, usize) = (4, 8);

// The NEON kernels hard-code their tile; it must be the one the packing uses.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
const _: () = assert!(
    F64_TILE.0 == super::tri_kernel_neon::F64_SHAPE.0
        && F64_TILE.1 == super::tri_kernel_neon::F64_SHAPE.1
        && F32_TILE.0 == super::tri_kernel_neon::F32_SHAPE.0
        && F32_TILE.1 == super::tri_kernel_neon::F32_SHAPE.1,
    "NEON kernel tiles must match the packing tiles"
);
/// Register tile of the complex kernels.
const COMPLEX_TILE: (usize, usize) = (4, 2);
/// Register tile of the generic fallback kernels (other element types).
const GENERIC_TILE: (usize, usize) = (4, 4);

/// Returns `true` when `T` and `U` are the same type.
#[inline(always)]
fn same_type<T: 'static, U: 'static>() -> bool {
    core::any::TypeId::of::<T>() == core::any::TypeId::of::<U>()
}

/// The `(MR, NR)` register tile the kernels use for element type `T`; the
/// packing routines lay the triangle and the right-hand sides out for it.
#[inline]
pub(crate) fn tile_shape<T: 'static>() -> (usize, usize) {
    if same_type::<T, f64>() {
        F64_TILE
    } else if same_type::<T, f32>() {
        F32_TILE
    } else if same_type::<T, Complex64>() || same_type::<T, Complex32>() {
        COMPLEX_TILE
    } else {
        GENERIC_TILE
    }
}

/// How the diagonal of a TRSM block is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SolveDiag {
    /// Unit diagonal: never read.
    Unit,
    /// Multiply by the packed reciprocal of the diagonal.
    Reciprocal,
    /// Divide by the packed diagonal (used when a reciprocal is unusable; see
    /// `tri_driver::nonunit_diag`), robustly for complex types.
    Divide,
}

/// Number of packed elements of a `kb x kb` block for an `mr`-row tile.
#[inline]
pub(crate) fn packed_tri_len(kb: usize, mr: usize) -> usize {
    let np = kb.div_ceil(mr);
    mr * mr * np * (np + 1) / 2
}

/// Solves `L · X = X` in place on one packed micro-panel (forward
/// substitution), dispatching on the element type.
///
/// `tri` is the packed block (see the module docs), `kb` its order and `xp`
/// the packed micro-panel laid out for [`tile_shape::<T>()`](tile_shape).
/// The buffers are taken as `Vec`s because `Any` downcasting needs a sized
/// `'static` type; that is how the element type is dispatched without
/// `unsafe`.
#[allow(clippy::ptr_arg)]
pub(crate) fn trsm_micro_panel<T: Field>(
    tri: &Vec<T>,
    kb: usize,
    xp: &mut Vec<T>,
    diag: SolveDiag,
) {
    macro_rules! go {
        ($t:ty, $ops:ty, $tile:expr) => {
            if let (Some(t), Some(x)) = (
                (tri as &dyn Any).downcast_ref::<Vec<$t>>(),
                (xp as &mut dyn Any).downcast_mut::<Vec<$t>>(),
            ) {
                trsm_dispatch_diag::<$t, $ops, { $tile.0 }, { $tile.1 }>(t, kb, x, diag);
                return;
            }
        };
    }
    // Explicit NEON kernels for the real types (the exact-division mode is
    // rare and stays on the portable kernel, which has the same tile shape).
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        macro_rules! neon {
            ($t:ty, $kernel:ident) => {
                if diag != SolveDiag::Divide {
                    if let (Some(t), Some(x)) = (
                        (tri as &dyn Any).downcast_ref::<Vec<$t>>(),
                        (xp as &mut dyn Any).downcast_mut::<Vec<$t>>(),
                    ) {
                        if diag == SolveDiag::Unit {
                            super::tri_kernel_neon::$kernel::<true>(t, kb, x);
                        } else {
                            super::tri_kernel_neon::$kernel::<false>(t, kb, x);
                        }
                        return;
                    }
                }
            };
        }
        neon!(f64, trsm_panel_f64);
        neon!(f32, trsm_panel_f32);
    }
    go!(f64, FastOps, F64_TILE);
    go!(f32, FastOps, F32_TILE);
    go!(Complex64, FastOps, COMPLEX_TILE);
    go!(Complex32, FastOps, COMPLEX_TILE);
    trsm_dispatch_diag::<T, PlainOps, { GENERIC_TILE.0 }, { GENERIC_TILE.1 }>(tri, kb, xp, diag);
}

/// Computes `X := alpha · L · X` in place on one packed micro-panel
/// (`alpha == None` means 1 and skips the scaling), dispatching on the
/// element type (`Vec` parameters as in [`trsm_micro_panel`]).
#[allow(clippy::ptr_arg)]
pub(crate) fn trmm_micro_panel<T: Field>(
    tri: &Vec<T>,
    kb: usize,
    xp: &mut Vec<T>,
    unit: bool,
    alpha: Option<T>,
) {
    macro_rules! go {
        ($t:ty, $ops:ty, $tile:expr) => {
            if let (Some(t), Some(x)) = (
                (tri as &dyn Any).downcast_ref::<Vec<$t>>(),
                (xp as &mut dyn Any).downcast_mut::<Vec<$t>>(),
            ) {
                // `alpha` is a `T`, which is `$t` here; `Any` converts it safely.
                let alpha = alpha.and_then(|al| (&al as &dyn Any).downcast_ref::<$t>().copied());
                if unit {
                    trmm_panel::<$t, $ops, { $tile.0 }, { $tile.1 }, true>(t, kb, x, alpha);
                } else {
                    trmm_panel::<$t, $ops, { $tile.0 }, { $tile.1 }, false>(t, kb, x, alpha);
                }
                return;
            }
        };
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        macro_rules! neon {
            ($t:ty, $kernel:ident) => {
                if let (Some(t), Some(x)) = (
                    (tri as &dyn Any).downcast_ref::<Vec<$t>>(),
                    (xp as &mut dyn Any).downcast_mut::<Vec<$t>>(),
                ) {
                    let alpha =
                        alpha.and_then(|al| (&al as &dyn Any).downcast_ref::<$t>().copied());
                    if unit {
                        super::tri_kernel_neon::$kernel::<true>(t, kb, x, alpha);
                    } else {
                        super::tri_kernel_neon::$kernel::<false>(t, kb, x, alpha);
                    }
                    return;
                }
            };
        }
        neon!(f64, trmm_panel_f64);
        neon!(f32, trmm_panel_f32);
    }
    go!(f64, FastOps, F64_TILE);
    go!(f32, FastOps, F32_TILE);
    go!(Complex64, FastOps, COMPLEX_TILE);
    go!(Complex32, FastOps, COMPLEX_TILE);
    if unit {
        trmm_panel::<T, PlainOps, { GENERIC_TILE.0 }, { GENERIC_TILE.1 }, true>(tri, kb, xp, alpha);
    } else {
        trmm_panel::<T, PlainOps, { GENERIC_TILE.0 }, { GENERIC_TILE.1 }, false>(
            tri, kb, xp, alpha,
        );
    }
}

/// Selects the monomorphized TRSM kernel for a diagonal mode.
#[inline(always)]
fn trsm_dispatch_diag<T: Field, O: TileOps<T>, const MR: usize, const NR: usize>(
    tri: &[T],
    kb: usize,
    xp: &mut [T],
    diag: SolveDiag,
) {
    match diag {
        SolveDiag::Unit => trsm_panel::<T, O, MR, NR, true, false>(tri, kb, xp),
        SolveDiag::Reciprocal => trsm_panel::<T, O, MR, NR, false, false>(tri, kb, xp),
        SolveDiag::Divide => trsm_panel::<T, O, MR, NR, false, true>(tri, kb, xp),
    }
}

/// Forward substitution on one packed micro-panel.
///
/// For each row tile `p` the already solved rows `0 .. p·MR` are applied
/// with a register-tiled rank-1 loop (`acc -= L[tile, k] ⊗ X[k, :]`), then
/// the `MR x MR` diagonal tile is solved in registers and the tile is stored
/// back, so later tiles read the solved values.
///
/// Kept out of line so that every instantiation is optimized on its own
/// (fully unrolled tile loops, accumulators in registers).
#[inline(never)]
fn trsm_panel<
    T: Field,
    O: TileOps<T>,
    const MR: usize,
    const NR: usize,
    const UNIT: bool,
    const DIV: bool,
>(
    tri: &[T],
    kb: usize,
    xp: &mut [T],
) {
    let np = kb.div_ceil(MR);
    let mut toff = 0;
    for p in 0..np {
        let k = p * MR;
        let rect_len = k * MR;
        let (rect, rest) = tri[toff..].split_at(rect_len);
        let (tile, _) = rest.split_at(MR * MR);
        toff += rect_len + MR * MR;

        let (solved, rest) = xp.split_at_mut(k * NR);
        let (cur, _) = rest.split_at_mut(MR * NR);
        let (cur_rows, _) = cur.as_chunks_mut::<NR>();

        let mut acc = [[T::zero(); NR]; MR];
        for (acc_r, row) in acc.iter_mut().zip(cur_rows.iter()) {
            *acc_r = *row;
        }

        let (rect_cols, _) = rect.as_chunks::<MR>();
        let (solved_rows, _) = solved.as_chunks::<NR>();
        for (lk, xk) in rect_cols.iter().zip(solved_rows.iter()) {
            for r in 0..MR {
                let l = lk[r];
                for c in 0..NR {
                    acc[r][c] = O::nmadd(l, xk[c], acc[r][c]);
                }
            }
        }

        let (tile_cols, _) = tile.as_chunks::<MR>();
        for (c, col) in tile_cols.iter().enumerate() {
            if !UNIT {
                let d = col[c];
                for j in 0..NR {
                    acc[c][j] = if DIV {
                        div_diag(acc[c][j], d)
                    } else {
                        acc[c][j] * d
                    };
                }
            }
            let xc = acc[c];
            for r in (c + 1)..MR {
                let l = col[r];
                for j in 0..NR {
                    acc[r][j] = O::nmadd(l, xc[j], acc[r][j]);
                }
            }
        }

        for (row, acc_r) in cur_rows.iter_mut().zip(acc.iter()) {
            *row = *acc_r;
        }
    }
}

/// `x / d` in the exact-division mode of [`trsm_panel`].
///
/// Real types divide directly (one correctly rounded operation), exactly as
/// before. For complex types the `/` of `num_complex` forms `|d|²`, which
/// overflows for `|d|` above about `1.3e154` (`1.8e19` for `Complex32`) and
/// underflows for tiny `|d|`: the very diagonals that select this mode. They
/// are divided with [`ladiv`] instead, which never forms `|d|²`.
#[inline]
pub(crate) fn div_diag<T: Field>(x: T, d: T) -> T {
    if T::is_real() {
        x / d
    } else {
        let (re, im) = ladiv(x.real(), x.imag(), d.real(), d.imag());
        T::from_real_imag(re, im)
    }
}

/// `(a + ib) / (c + id)` without harmful intermediate overflow or underflow:
/// the robust Smith algorithm of Baudin and Smith (2012), as in LAPACK
/// `?LADIV`. Operands near the overflow threshold are halved and tiny ones
/// scaled up by `2 / ε²`; every scaling is by a power of two, so it is exact,
/// and the quotient is computed from the ratio of the divisor's components.
fn ladiv<R: Real>(a: R, b: R, c: R, d: R) -> (R, R) {
    let one = R::one();
    let two = one + one;
    let half = one / two;
    let ov = <R as Scalar>::max_value();
    let un = <R as Scalar>::min_positive();
    // LAPACK's relative machine precision (half the spacing above 1).
    let eps = <R as Scalar>::epsilon() * half;
    let be = two / (eps * eps);
    let ab = Float::max(Float::abs(a), Float::abs(b));
    let cd = Float::max(Float::abs(c), Float::abs(d));
    let (mut aa, mut bb, mut cc, mut dd) = (a, b, c, d);
    let mut s = one;
    if ab >= half * ov {
        aa = half * aa;
        bb = half * bb;
        s = two * s;
    }
    if cd >= half * ov {
        cc = half * cc;
        dd = half * dd;
        s = half * s;
    }
    if ab <= un * two / eps {
        aa = aa * be;
        bb = bb * be;
        s = s / be;
    }
    if cd <= un * two / eps {
        cc = cc * be;
        dd = dd * be;
        s = s * be;
    }
    let (p, q) = if Float::abs(d) <= Float::abs(c) {
        ladiv1(aa, bb, cc, dd)
    } else {
        // (a + ib) / (c + id) = conj((b + ia) / (d + ic)).
        let (p, q) = ladiv1(bb, aa, dd, cc);
        (p, -q)
    };
    (p * s, q * s)
}

/// [`ladiv`] for `|d| <= |c|` (LAPACK `?LADIV1`).
fn ladiv1<R: Real>(a: R, b: R, c: R, d: R) -> (R, R) {
    let r = d / c;
    let t = R::one() / (c + d * r);
    (ladiv2(a, b, c, d, r, t), ladiv2(b, -a, c, d, r, t))
}

/// `(a + b·r)·t`, with the terms regrouped when `r` or `b·r` underflows to
/// zero (LAPACK `?LADIV2`).
fn ladiv2<R: Real>(a: R, b: R, c: R, d: R, r: R, t: R) -> R {
    if r != R::zero() {
        let br = b * r;
        if br != R::zero() {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        }
    } else {
        (a + d * (b / c)) * t
    }
}

/// Lower-triangular multiply on one packed micro-panel, in place.
///
/// Row tiles are processed bottom-up: tile `p` only reads rows
/// `0 .. (p + 1)·MR`, none of which has been overwritten yet. Kept out of
/// line for the same reason as [`trsm_panel`].
#[inline(never)]
fn trmm_panel<T: Field, O: TileOps<T>, const MR: usize, const NR: usize, const UNIT: bool>(
    tri: &[T],
    kb: usize,
    xp: &mut [T],
    alpha: Option<T>,
) {
    let np = kb.div_ceil(MR);
    for p in (0..np).rev() {
        let k = p * MR;
        let toff = MR * MR * p * (p + 1) / 2;
        let rect_len = k * MR;
        let (rect, rest) = tri[toff..].split_at(rect_len);
        let (tile, _) = rest.split_at(MR * MR);

        let (prev, rest) = xp.split_at_mut(k * NR);
        let (cur, _) = rest.split_at_mut(MR * NR);
        let (cur_rows, _) = cur.as_chunks_mut::<NR>();

        let mut acc = [[T::zero(); NR]; MR];

        let (rect_cols, _) = rect.as_chunks::<MR>();
        let (prev_rows, _) = prev.as_chunks::<NR>();
        for (lk, xk) in rect_cols.iter().zip(prev_rows.iter()) {
            for r in 0..MR {
                let l = lk[r];
                for c in 0..NR {
                    acc[r][c] = O::madd(l, xk[c], acc[r][c]);
                }
            }
        }

        let (tile_cols, _) = tile.as_chunks::<MR>();
        for (c, (col, xc)) in tile_cols.iter().zip(cur_rows.iter()).enumerate() {
            if UNIT {
                for j in 0..NR {
                    acc[c][j] += xc[j];
                }
            } else {
                let d = col[c];
                for j in 0..NR {
                    acc[c][j] = O::madd(d, xc[j], acc[c][j]);
                }
            }
            for r in (c + 1)..MR {
                let l = col[r];
                for j in 0..NR {
                    acc[r][j] = O::madd(l, xc[j], acc[r][j]);
                }
            }
        }

        if let Some(al) = alpha {
            for acc_r in acc.iter_mut() {
                for v in acc_r.iter_mut() {
                    *v = al * *v;
                }
            }
        }

        for (row, acc_r) in cur_rows.iter_mut().zip(acc.iter()) {
            *row = *acc_r;
        }
    }
}
