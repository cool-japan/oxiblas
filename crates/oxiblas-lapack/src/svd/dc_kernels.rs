//! Dense building blocks shared by the divide-and-conquer SVD kernels.
//!
//! [`DenseOps`] abstracts the one Level-3 operation the SVD kernels need,
//! `C = α·op(A)·op(B) + β·C`:
//!
//! - [`BlasOps`] forwards it to the optimised `oxiblas_blas` GEMM, which is
//!   only available for the [`GemmKernel`] scalars (`f32`, `f64`). It carries
//!   the [`Par`] mode every GEMM of one SVD call runs with (`Par::Seq` for
//!   `SvdDc::compute`); the parallel GEMM driver splits only the output, so
//!   its results are bitwise identical to the sequential ones.
//! - [`PortableOps`] is a column-oriented fallback for every other real scalar
//!   (e.g. `f16`).
//!
//! The public SVD front-ends keep their `T: Field + Real + Zeroable` bounds and
//! select the backend once per call with [`same_ref`] / [`same_mut`], which are
//! safe `core::any::Any` downcasts: when `T` is `f64` (or `f32`) the owned
//! buffers are viewed as `f64` (`f32`) buffers and the GEMM-backed kernel runs;
//! otherwise the portable kernel runs on the original type.
//!
//! The remaining helpers are column-oriented Level-1 kernels on contiguous
//! slices. They use independent accumulators and fused multiply-adds
//! (`Real::mul_add`) so that the compiler can vectorise them into `fmla`
//! instructions for `f32`/`f64` without changing their meaning for other
//! scalar types. The scaling helpers ([`pow2_exponent`], [`scale_pow2`])
//! multiply by powers of two only, so scaling and unscaling are exact barring
//! overflow and underflow.

use core::any::Any;
use core::cmp::Ordering;

use oxiblas_blas::level3::Trans;
use oxiblas_blas::level3::gemm::gemm_transposed_with_par;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Provider of the Level-3 operation used by the SVD kernels.
pub(crate) trait DenseOps<T: Field>: Copy {
    /// `C = alpha · op(A) · op(B) + beta · C` where `op(X) = Xᵀ` when the
    /// corresponding flag is set.
    ///
    /// `beta == 0` overwrites `C` without reading it.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        c: MatMut<'_, T>,
    );
}

/// GEMM backend for the scalars with an optimised `oxiblas_blas` kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlasOps {
    /// Parallelism of every GEMM issued through this backend.
    pub(crate) par: Par,
}

impl BlasOps {
    /// The sequential backend (what `SvdDc::compute` uses).
    pub(crate) const SEQ: Self = Self { par: Par::Seq };
}

impl<T: Field + GemmKernel + bytemuck::Zeroable> DenseOps<T> for BlasOps {
    #[inline]
    fn gemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        c: MatMut<'_, T>,
    ) {
        let ta = if trans_a {
            Trans::Trans
        } else {
            Trans::NoTrans
        };
        let tb = if trans_b {
            Trans::Trans
        } else {
            Trans::NoTrans
        };
        gemm_transposed_with_par(ta, tb, alpha, a, b, beta, c, self.par);
    }
}

/// Column-oriented GEMM for scalars without an optimised kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortableOps;

impl<T: Field> DenseOps<T> for PortableOps {
    fn gemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        mut c: MatMut<'_, T>,
    ) {
        let m = c.nrows();
        let n = c.ncols();
        let k = if trans_a { a.nrows() } else { a.ncols() };

        for j in 0..n {
            for i in 0..m {
                c[(i, j)] = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * c[(i, j)]
                };
            }
        }
        if alpha == T::zero() || k == 0 {
            return;
        }

        let b_at = |l: usize, j: usize| if trans_b { b[(j, l)] } else { b[(l, j)] };
        if trans_a {
            // C[i, j] += alpha · Σ_l A[l, i] · op(B)[l, j]: walks columns of A.
            for j in 0..n {
                for i in 0..m {
                    let mut acc = T::zero();
                    for l in 0..k {
                        acc = acc + a[(l, i)] * b_at(l, j);
                    }
                    c[(i, j)] = c[(i, j)] + alpha * acc;
                }
            }
        } else {
            // C[:, j] += (alpha · op(B)[l, j]) · A[:, l]: column axpys.
            for j in 0..n {
                for l in 0..k {
                    let s = alpha * b_at(l, j);
                    for i in 0..m {
                        c[(i, j)] = c[(i, j)] + s * a[(i, l)];
                    }
                }
            }
        }
    }
}

/// Views `value` as a `D` when `S` and `D` are the same type.
#[inline]
pub(crate) fn same_ref<S: 'static, D: 'static>(value: &S) -> Option<&D> {
    (value as &dyn Any).downcast_ref::<D>()
}

/// Views `value` as a mutable `D` when `S` and `D` are the same type.
#[inline]
pub(crate) fn same_mut<S: 'static, D: 'static>(value: &mut S) -> Option<&mut D> {
    (value as &mut dyn Any).downcast_mut::<D>()
}

/// Absolute value of a real scalar.
#[inline]
pub(crate) fn fabs<R: Real>(x: R) -> R {
    if x < R::zero() { -x } else { x }
}

/// Converts an `f64` constant into `R` (zero if unrepresentable).
#[inline]
pub(crate) fn real_const<R: Real>(v: f64) -> R {
    R::from_f64(v).unwrap_or_else(R::zero)
}

/// LAPACK's `dlamch('E')`: the unit roundoff, half of [`Scalar::epsilon`]
/// (`2⁻⁵³` for `f64`, `2⁻²⁴` for `f32`).
#[inline]
pub(crate) fn lapack_eps<R: Real>() -> R {
    <R as Scalar>::epsilon() * real_const::<R>(0.5)
}

/// Fortran `SIGN(a, b)`: `|a|` with the sign of `b` (`+|a|` when `b` is a
/// zero of either sign).
#[inline]
pub(crate) fn sign_of<R: Real>(a: R, b: R) -> R {
    if b < R::zero() { -fabs(a) } else { fabs(a) }
}

/// Largest exponent applied in one step of [`scale_pow2`], chosen so that
/// `2^±k` is a normal number of `R` (511 for `f64`, 63 for `f32`).
fn pow2_step<R: Real>() -> i32 {
    let emax = num_traits::Float::log2(<R as Scalar>::max_value());
    let half = <R as Real>::floor(emax * real_const::<R>(0.5));
    num_traits::ToPrimitive::to_i32(&half).unwrap_or(1).max(1)
}

/// `2^k` for `|k| ≤ pow2_step::<R>()` (exact: a power of two converts
/// exactly between floating-point formats that can represent it).
fn pow2<R: Real>(k: i32) -> R {
    R::from_f64(f64::powi(2.0, k)).unwrap_or_else(R::one)
}

/// The exponent `k` for which `x · 2^k` lies in `[2^-0.5, 2^0.5]`, for a
/// finite `x > 0` (zero otherwise).
pub(crate) fn pow2_exponent<R: Real>(x: R) -> i32 {
    if !(x > R::zero() && x.is_finite()) {
        return 0;
    }
    let l = <R as Real>::round(num_traits::Float::log2(x));
    -num_traits::ToPrimitive::to_i32(&l).unwrap_or(0)
}

/// Multiplies every element of `x` by `2^k`.
///
/// The factor is applied in steps of at most [`pow2_step`] so that it stays
/// representable; each step is exact unless an element overflows or
/// underflows.
pub(crate) fn scale_pow2<R: Real>(x: &mut [R], k: i32) {
    let step = pow2_step::<R>();
    let mut rem = k;
    while rem != 0 {
        let s = rem.clamp(-step, step);
        let f = pow2::<R>(s);
        for v in x.iter_mut() {
            *v = *v * f;
        }
        rem -= s;
    }
}

/// `x · 2^k` for one value (see [`scale_pow2`]).
#[inline]
pub(crate) fn mul_pow2<R: Real>(x: R, k: i32) -> R {
    let mut a = [x];
    scale_pow2(&mut a, k);
    a[0]
}

/// Largest absolute value in `x`, NaN if any element is NaN (zero for an
/// empty slice).
pub(crate) fn max_abs<R: Real>(x: &[R]) -> R {
    let mut m = R::zero();
    for &v in x {
        if v.is_nan() {
            return v;
        }
        m = m.max(fabs(v));
    }
    m
}

/// Plane rotation of two columns of `m` over the rows `r0..r0 + len`:
/// `x ← c·x + s·y` and `y ← c·y − s·x` for `x = m[r0.., a]`, `y = m[r0.., b]`
/// (LAPACK `drot`; `a ≠ b`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rot_cols<R: Field + Real>(
    m: &mut Mat<R>,
    a: usize,
    b: usize,
    r0: usize,
    len: usize,
    c: R,
    s: R,
) {
    if a == b || len == 0 {
        return;
    }
    let ld = m.row_stride();
    let data = m.raw_data_mut();
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (left, right) = data.split_at_mut(hi * ld);
    let col_lo = &mut left[lo * ld + r0..lo * ld + r0 + len];
    let col_hi = &mut right[r0..r0 + len];
    let (x, y) = if a < b {
        (col_lo, col_hi)
    } else {
        (col_hi, col_lo)
    };
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let t = c * *xi + s * *yi;
        *yi = c * *yi - s * *xi;
        *xi = t;
    }
}

/// Swaps two columns of `m` over the rows `r0..r0 + len`.
pub(crate) fn swap_cols<R: Scalar>(m: &mut Mat<R>, a: usize, b: usize, r0: usize, len: usize) {
    if a == b || len == 0 {
        return;
    }
    let ld = m.row_stride();
    let data = m.raw_data_mut();
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (left, right) = data.split_at_mut(hi * ld);
    left[lo * ld + r0..lo * ld + r0 + len].swap_with_slice(&mut right[r0..r0 + len]);
}

/// Comparator for sorting real values ascending, with NaN placed last.
///
/// It is a total preorder (`±0` compare equal, as do all NaNs, which compare
/// greater than every number), which is what `slice::sort_by` requires: the
/// standard sorts may panic on an inconsistent comparator, such as a plain
/// `partial_cmp(..).unwrap_or(Equal)`, which makes a NaN equal to everything.
/// The `partial_cmp` below only ever sees two non-NaN values.
#[inline]
pub(crate) fn cmp_asc_nan_last<R: Real>(a: R, b: R) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Total order for sorting real values descending, with NaN placed last.
#[inline]
pub(crate) fn cmp_desc_nan_last<R: Real>(a: R, b: R) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

/// `Σ x[i]·y[i]` over the common length, using eight independent accumulators.
///
/// Products are accumulated with fused multiply-adds (`Real::mul_add`, a
/// single `fmadd` for `f32`/`f64`).
#[inline]
pub(crate) fn dot<T: Field + Real>(x: &[T], y: &[T]) -> T {
    let len = x.len().min(y.len());
    let (xc, xr) = x[..len].as_chunks::<8>();
    let (yc, yr) = y[..len].as_chunks::<8>();
    let mut acc = [T::zero(); 8];
    for (a, b) in xc.iter().zip(yc) {
        for ((s, &p), &q) in acc.iter_mut().zip(a).zip(b) {
            *s = <T as Real>::mul_add(p, q, *s);
        }
    }
    let mut tail = T::zero();
    for (&p, &q) in xr.iter().zip(yr) {
        tail = <T as Real>::mul_add(p, q, tail);
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// `[x₀·y, x₁·y, x₂·y, x₃·y]` in one pass over `y` (two accumulators per column).
#[inline]
pub(crate) fn dot4<T: Field + Real>(x: [&[T]; 4], y: &[T]) -> [T; 4] {
    let len = y
        .len()
        .min(x[0].len())
        .min(x[1].len())
        .min(x[2].len())
        .min(x[3].len());
    let (yc, yr) = y[..len].as_chunks::<2>();
    let (c0, r0) = x[0][..len].as_chunks::<2>();
    let (c1, r1) = x[1][..len].as_chunks::<2>();
    let (c2, r2) = x[2][..len].as_chunks::<2>();
    let (c3, r3) = x[3][..len].as_chunks::<2>();
    let mut acc = [[T::zero(); 2]; 4];
    for ((((yy, a), b), c), d) in yc.iter().zip(c0).zip(c1).zip(c2).zip(c3) {
        for l in 0..2 {
            acc[0][l] = <T as Real>::mul_add(a[l], yy[l], acc[0][l]);
            acc[1][l] = <T as Real>::mul_add(b[l], yy[l], acc[1][l]);
            acc[2][l] = <T as Real>::mul_add(c[l], yy[l], acc[2][l]);
            acc[3][l] = <T as Real>::mul_add(d[l], yy[l], acc[3][l]);
        }
    }
    let mut out = [
        acc[0][0] + acc[0][1],
        acc[1][0] + acc[1][1],
        acc[2][0] + acc[2][1],
        acc[3][0] + acc[3][1],
    ];
    if let (Some(&yl), Some(&a), Some(&b), Some(&c), Some(&d)) =
        (yr.first(), r0.first(), r1.first(), r2.first(), r3.first())
    {
        out[0] = <T as Real>::mul_add(a, yl, out[0]);
        out[1] = <T as Real>::mul_add(b, yl, out[1]);
        out[2] = <T as Real>::mul_add(c, yl, out[2]);
        out[3] = <T as Real>::mul_add(d, yl, out[3]);
    }
    out
}

/// `y += alpha · x` over the common length.
#[inline]
pub(crate) fn axpy<T: Field + Real>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = <T as Real>::mul_add(alpha, xi, *yi);
    }
}

/// `y += a0·x0 + a1·x1 + a2·x2 + a3·x3` over the common length (one pass over `y`).
#[inline]
pub(crate) fn axpy4<T: Field + Real>(a: [T; 4], x: [&[T]; 4], y: &mut [T]) {
    let len = y
        .len()
        .min(x[0].len())
        .min(x[1].len())
        .min(x[2].len())
        .min(x[3].len());
    let (x0, x1, x2, x3) = (&x[0][..len], &x[1][..len], &x[2][..len], &x[3][..len]);
    for (i, yi) in y[..len].iter_mut().enumerate() {
        let t = <T as Real>::mul_add(a[0], x0[i], *yi);
        let t = <T as Real>::mul_add(a[1], x1[i], t);
        let t = <T as Real>::mul_add(a[2], x2[i], t);
        *yi = <T as Real>::mul_add(a[3], x3[i], t);
    }
}

/// `x *= alpha`.
#[inline]
pub(crate) fn scal<T: Field>(alpha: T, x: &mut [T]) {
    for xi in x.iter_mut() {
        *xi = alpha * *xi;
    }
}

/// `x /= denom`, multiplying by the reciprocal when it is finite and
/// dividing element-wise otherwise (a subnormal `denom` has no finite
/// reciprocal, while `xᵢ / denom` is still representable).
#[inline]
pub(crate) fn scal_div<T: Field + Real>(denom: T, x: &mut [T]) {
    let inv = T::one() / denom;
    if inv.is_finite() {
        scal(inv, x);
    } else {
        for xi in x.iter_mut() {
            *xi = *xi / denom;
        }
    }
}

/// Euclidean norm, rescaling when the plain sum of squares under- or overflows.
pub(crate) fn nrm2<T: Field + Real>(x: &[T]) -> T {
    let ss = dot(x, x);
    if ss.is_nan() {
        return ss;
    }
    let small = <T as Scalar>::min_positive() / <T as Scalar>::epsilon();
    if ss.is_finite() && ss >= small {
        return <T as Real>::sqrt(ss);
    }
    let mut scale = T::zero();
    for &v in x {
        let a = fabs(v);
        if a > scale {
            scale = a;
        }
    }
    if scale == T::zero() || !scale.is_finite() {
        return scale;
    }
    // Divide (rather than multiply by 1/scale): a subnormal `scale` has no
    // finite reciprocal, but every |v / scale| ≤ 1 is representable.
    let mut acc = T::zero();
    for &v in x {
        let t = v / scale;
        acc = acc + t * t;
    }
    scale * <T as Real>::sqrt(acc)
}

/// Writes `dst = srcᵀ` with a cache-friendly tiled traversal.
///
/// `dst` must be `src.ncols() × src.nrows()`.
pub(crate) fn transpose_into<T: Scalar>(src: MatRef<'_, T>, mut dst: MatMut<'_, T>) {
    const TILE: usize = 32;
    let m = src.nrows();
    let n = src.ncols();
    debug_assert_eq!((dst.nrows(), dst.ncols()), (n, m));
    for jb in (0..n).step_by(TILE) {
        let je = (jb + TILE).min(n);
        for ib in (0..m).step_by(TILE) {
            let ie = (ib + TILE).min(m);
            for i in ib..ie {
                for j in jb..je {
                    dst[(j, i)] = src[(i, j)];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn dot_axpy_match_naive() {
        let x: Vec<f64> = (0..37).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = (0..37).map(|i| (i as f64 * 0.11).cos()).collect();
        let naive: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        assert!((dot(&x, &y) - naive).abs() < 1e-13);

        let mut z = y.clone();
        axpy(2.5, &x, &mut z);
        for i in 0..37 {
            assert!((z[i] - (y[i] + 2.5 * x[i])).abs() < 1e-15);
        }

        let neg: Vec<f64> = x.iter().map(|v| -v).collect();
        let d4 = dot4([&x, &y, &neg, &x], &y);
        let naive_yy: f64 = y.iter().map(|a| a * a).sum();
        assert!((d4[0] - naive).abs() < 1e-13);
        assert!((d4[1] - naive_yy).abs() < 1e-13);
        assert!((d4[2] + naive).abs() < 1e-13);
        assert!((d4[3] - naive).abs() < 1e-13);

        let mut w = y.clone();
        axpy4([1.0, -2.0, 0.5, 3.0], [&x, &y, &x, &y], &mut w);
        for i in 0..37 {
            let expect = y[i] + x[i] - 2.0 * y[i] + 0.5 * x[i] + 3.0 * y[i];
            assert!((w[i] - expect).abs() < 1e-14);
        }
    }

    #[test]
    fn nrm2_rescales_extreme_values() {
        let tiny = [3e-200f64, 4e-200];
        assert!((nrm2(&tiny) / 5e-200 - 1.0).abs() < 1e-14);
        let huge = [3e200f64, 4e200];
        assert!((nrm2(&huge) / 5e200 - 1.0).abs() < 1e-14);
        assert_eq!(nrm2(&[0.0f64, 0.0]), 0.0);
        assert!(nrm2(&[1.0f64, f64::NAN]).is_nan());
    }

    #[test]
    fn portable_gemm_matches_blas() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.5], &[-1.0, 2.0], &[0.25, 1.0]]);
        for &(ta, tb) in &[(false, false), (true, false), (false, true), (true, true)] {
            let (am, bm) = match (ta, tb) {
                (false, false) => (a.clone(), b.clone()),
                (true, false) => (a.transpose(), b.clone()),
                (false, true) => (a.clone(), b.transpose()),
                (true, true) => (a.transpose(), b.transpose()),
            };
            let mut c1 = Mat::filled(2, 2, 1.0f64);
            let mut c2 = Mat::filled(2, 2, 1.0f64);
            PortableOps.gemm(ta, tb, 2.0, am.as_ref(), bm.as_ref(), 0.5, c1.as_mut());
            BlasOps::SEQ.gemm(ta, tb, 2.0, am.as_ref(), bm.as_ref(), 0.5, c2.as_mut());
            for i in 0..2 {
                for j in 0..2 {
                    assert!((c1[(i, j)] - c2[(i, j)]).abs() < 1e-13, "{ta} {tb}");
                }
            }
        }
    }

    #[test]
    fn transpose_into_roundtrip() {
        let mut a = Mat::<f64>::zeros(45, 70);
        for j in 0..70 {
            for i in 0..45 {
                a[(i, j)] = (i * 100 + j) as f64;
            }
        }
        let mut t = Mat::<f64>::zeros(70, 45);
        transpose_into(a.as_ref(), t.as_mut());
        for j in 0..70 {
            for i in 0..45 {
                assert_eq!(t[(j, i)], a[(i, j)]);
            }
        }
    }

    #[test]
    fn pow2_scaling_is_exact_and_reversible() {
        let mut x = [3.0e-300f64, -1.5, 7.25e290, 0.0];
        let orig = x;
        for &(k, mag) in &[
            (1100i32, 1e-300f64),
            (-1100, 1e300),
            (7, 1.0),
            (-511, 1.0),
            (0, 1.0),
        ] {
            let mut y = [mag, 3.0 * mag, -0.75 * mag];
            let base = y;
            scale_pow2(&mut y, k);
            assert!(y.iter().all(|v| v.is_finite() && *v != 0.0), "k={k}");
            scale_pow2(&mut y, -k);
            assert_eq!(y, base, "k={k}");
        }
        let k = pow2_exponent(7.25e290f64);
        scale_pow2(&mut x, k);
        assert!(x[2] >= 0.70 && x[2] <= 1.42, "scaled max {}", x[2]);
        scale_pow2(&mut x, -k);
        assert_eq!(x[1..], orig[1..]);
        assert_eq!(pow2_exponent(0.0f64), 0);
        assert_eq!(pow2_exponent(f64::NAN), 0);
        assert_eq!(mul_pow2(1.0f32, 3), 8.0);
        assert!(max_abs(&[1.0f64, f64::NAN]).is_nan());
        assert_eq!(max_abs(&[-4.0f64, 2.0]), 4.0);
    }

    #[test]
    fn column_rotation_and_swap() {
        let mut m = Mat::<f64>::zeros(3, 3);
        for j in 0..3 {
            for i in 0..3 {
                m[(i, j)] = (3 * j + i) as f64;
            }
        }
        let (c, s) = (0.6, 0.8);
        let x0: Vec<f64> = (0..3).map(|i| m[(i, 2)]).collect();
        let y0: Vec<f64> = (0..3).map(|i| m[(i, 0)]).collect();
        rot_cols(&mut m, 2, 0, 1, 2, c, s);
        assert_eq!(m[(0, 2)], x0[0]);
        assert_eq!(m[(0, 0)], y0[0]);
        for i in 1..3 {
            assert!((m[(i, 2)] - (c * x0[i] + s * y0[i])).abs() < 1e-14);
            assert!((m[(i, 0)] - (c * y0[i] - s * x0[i])).abs() < 1e-14);
        }
        let before = m.clone();
        swap_cols(&mut m, 0, 1, 0, 3);
        for i in 0..3 {
            assert_eq!(m[(i, 0)], before[(i, 1)]);
            assert_eq!(m[(i, 1)], before[(i, 0)]);
        }
    }

    #[test]
    fn nan_aware_orderings_are_total() {
        let mut v = [3.0f64, f64::NAN, -1.0, 2.0, f64::NAN, 0.0];
        v.sort_by(|a, b| cmp_asc_nan_last(*a, *b));
        assert_eq!(&v[..4], &[-1.0, 0.0, 2.0, 3.0]);
        assert!(v[4].is_nan() && v[5].is_nan());
        v.sort_by(|a, b| cmp_desc_nan_last(*a, *b));
        assert_eq!(&v[..4], &[3.0, 2.0, 0.0, -1.0]);
        assert!(v[4].is_nan() && v[5].is_nan());
    }
}
