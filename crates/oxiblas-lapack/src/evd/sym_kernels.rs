//! Shared building blocks of the dense symmetric eigensolvers
//! ([`SymmetricEvd`](super::SymmetricEvd) and
//! [`SymmetricEvdDc`](super::SymmetricEvdDc)).
//!
//! - [`MatMul`]: the matrix-product back end. [`BlasMatMul`] routes to the
//!   `oxiblas_blas` GEMM (only `f32`/`f64` qualify, see [`GemmKernel`]) with
//!   the [`Par`] the driver was given: `Par::Seq` for the plain `compute`
//!   entry points, the caller's choice for `compute_with_par`. Every `Par`
//!   gives bitwise the same products. [`NaiveMatMul`] is a plain loop used
//!   for every other scalar type (it ignores `par`).
//! - [`with_gemm_backend`]: picks the back end for a scalar type at run time
//!   by downcasting the owned work buffers (no `unsafe`).
//! - Level-1/2 kernels on column-major slices (`dot`, `axpy`, the lower
//!   triangle `symv`/`syr2`) and the Householder generator [`larfg`].
//!
//! All dense storage is column-major: element `(i, j)` of a block starting at
//! offset `off` with leading dimension `ld` lives at `off + i + j * ld`.

use core::any::Any;

use oxiblas_blas::level3::gemm::gemm_transposed_with_par;
use oxiblas_blas::level3::gemm_kernel::GemmKernel;
use oxiblas_blas::level3::trsm::Trans;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Matrix-product back end of the symmetric eigensolvers.
pub(crate) trait MatMul<T: Field> {
    /// `true` when the blocked, GEMM-rich algorithms should be used.
    const BLOCKED: bool;

    /// `c = alpha · op(a) · op(b) + beta · c`, where `op(x)` is `xᵀ` when the
    /// corresponding flag is set. `c` is not read when `beta == 0`.
    ///
    /// `par` is the parallelism of the product; the result is bitwise the
    /// same for every `Par`.
    fn gemm(
        trans_a: bool,
        trans_b: bool,
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        c: MatMut<'_, T>,
        par: Par,
    );
}

/// GEMM-backed products through `oxiblas_blas` (`f32`, `f64`).
pub(crate) struct BlasMatMul;

impl<T: Field + GemmKernel + bytemuck::Zeroable> MatMul<T> for BlasMatMul {
    const BLOCKED: bool = true;

    /// `oxiblas_blas` GEMM with `par`, whose parallel path is bitwise
    /// identical to its sequential one.
    #[inline]
    fn gemm(
        trans_a: bool,
        trans_b: bool,
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        c: MatMut<'_, T>,
        par: Par,
    ) {
        let op = |t: bool| if t { Trans::Trans } else { Trans::NoTrans };
        gemm_transposed_with_par(op(trans_a), op(trans_b), alpha, a, b, beta, c, par);
    }
}

/// Plain triple-loop products for scalar types without a GEMM kernel.
pub(crate) struct NaiveMatMul;

impl<T: Field> MatMul<T> for NaiveMatMul {
    const BLOCKED: bool = false;

    fn gemm(
        trans_a: bool,
        trans_b: bool,
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        mut c: MatMut<'_, T>,
        _par: Par,
    ) {
        let m = c.nrows();
        let n = c.ncols();
        let k = if trans_a { a.nrows() } else { a.ncols() };
        let a_at = |i: usize, p: usize| if trans_a { a[(p, i)] } else { a[(i, p)] };
        let b_at = |p: usize, j: usize| if trans_b { b[(j, p)] } else { b[(p, j)] };
        for j in 0..n {
            for i in 0..m {
                let mut acc = T::zero();
                for p in 0..k {
                    acc += a_at(i, p) * b_at(p, j);
                }
                let old = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * c[(i, j)]
                };
                c[(i, j)] = old + alpha * acc;
            }
        }
    }
}

/// A solver pipeline that can run with any matrix-product back end.
pub(crate) trait Pipeline {
    /// Result of the run.
    type Output;

    /// Runs the pipeline on the owned buffers with back end `G`.
    fn run<T, G>(
        self,
        work: &mut Mat<T>,
        values: &mut Vec<T>,
        vectors: &mut Mat<T>,
    ) -> Self::Output
    where
        T: Field + Real + bytemuck::Zeroable,
        G: MatMul<T>;
}

/// Runs `pipeline` on `(work, values, vectors)` with the best matrix-product
/// back end for `T`.
///
/// `f64` and `f32` are routed to the `oxiblas_blas` GEMM by downcasting the
/// owned buffers (they are `'static`, so [`Any`] applies and no `unsafe` is
/// needed); every other scalar type runs the generic path with
/// [`NaiveMatMul`].
pub(crate) fn with_gemm_backend<T, P>(
    work: &mut Mat<T>,
    values: &mut Vec<T>,
    vectors: &mut Mat<T>,
    pipeline: P,
) -> P::Output
where
    T: Field + Real + bytemuck::Zeroable,
    P: Pipeline,
{
    if let (Some(w), Some(v), Some(z)) = (
        (work as &mut dyn Any).downcast_mut::<Mat<f64>>(),
        (values as &mut dyn Any).downcast_mut::<Vec<f64>>(),
        (vectors as &mut dyn Any).downcast_mut::<Mat<f64>>(),
    ) {
        return pipeline.run::<f64, BlasMatMul>(w, v, z);
    }
    if let (Some(w), Some(v), Some(z)) = (
        (work as &mut dyn Any).downcast_mut::<Mat<f32>>(),
        (values as &mut dyn Any).downcast_mut::<Vec<f32>>(),
        (vectors as &mut dyn Any).downcast_mut::<Mat<f32>>(),
    ) {
        return pipeline.run::<f32, BlasMatMul>(w, v, z);
    }
    pipeline.run::<T, NaiveMatMul>(work, values, vectors)
}

/// Copies the upper triangle of the symmetric matrix `a` into the lower
/// triangle of a new `n × n` matrix (`w[(j, i)] = a[(i, j)]` for `i <= j`).
///
/// The strict upper triangle of the result is zero. The copy is tiled so the
/// strided reads of `a` stay in cache.
pub(crate) fn lower_from_upper<T: Field + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Mat<T> {
    const TILE: usize = 32;
    let n = a.nrows();
    let mut w = Mat::zeros(n, n);
    let ldw = w.row_stride();
    let wd = w.raw_data_mut();
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + TILE).min(n);
        let mut i0 = 0;
        while i0 < j1 {
            let i1 = (i0 + TILE).min(j1);
            for j in j0..j1 {
                let top = i1.min(j + 1);
                if i0 >= top {
                    continue;
                }
                match a.col_as_slice(j) {
                    Some(col) => {
                        for (i, &v) in col[i0..top].iter().enumerate() {
                            wd[(i0 + i) * ldw + j] = v;
                        }
                    }
                    None => {
                        for i in i0..top {
                            wd[i * ldw + j] = a[(i, j)];
                        }
                    }
                }
            }
            i0 = i1;
        }
        j0 = j1;
    }
    w
}

/// Small exact constants of a real scalar type.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Consts<T> {
    /// `1/2`.
    pub(crate) half: T,
    /// Machine epsilon (`<T as Scalar>::epsilon()`).
    pub(crate) eps: T,
    /// Smallest positive normal number.
    pub(crate) safmin: T,
}

impl<T: Field + Real> Consts<T> {
    /// The constants of `T`.
    #[inline]
    pub(crate) fn new() -> Self {
        let two = T::one() + T::one();
        Self {
            half: T::one() / two,
            eps: <T as Scalar>::epsilon(),
            safmin: <T as Scalar>::min_positive(),
        }
    }
}

/// Scales the lower triangle of `a` (`n × n`) into the safe range of LAPACK
/// `DSYEVD` (max-norm in `[sqrt(smlnum), sqrt(bignum)]`,
/// `smlnum = safmin / eps`) by a power of two, so the reduction can neither
/// overflow nor lose accuracy to underflow; returns the factor applied
/// (`1` when the norm is already in range, zero or not finite).
///
/// Eigenvalues of the scaled matrix are divided by the factor afterwards;
/// a power of two makes both steps exact.
pub(crate) fn scale_into_safe_range<T: Field + Real>(a: &mut Mat<T>) -> T {
    let n = a.nrows();
    let ld = a.row_stride();
    let data = a.raw_data_mut();
    let mut anrm = T::zero();
    for j in 0..n {
        for &v in &data[j * ld + j..j * ld + n] {
            let av = Scalar::abs(v);
            if av > anrm {
                anrm = av;
            }
        }
    }
    let c = Consts::<T>::new();
    let smlnum = c.safmin / c.eps;
    let rmin = Real::sqrt(smlnum);
    let rmax = Real::sqrt(T::one() / smlnum);
    let two = T::one() + T::one();
    let sigma = if anrm > T::zero() && anrm < rmin {
        two * pow2_below(rmin / anrm)
    } else if anrm > rmax && anrm.is_finite() {
        T::one() / (two * pow2_below(anrm / rmax))
    } else {
        return T::one();
    };
    for j in 0..n {
        for v in data[j * ld + j..j * ld + n].iter_mut() {
            *v *= sigma;
        }
    }
    sigma
}

/// The largest power of two not above `x` (`1` unless `x` is positive and
/// finite).
pub(crate) fn pow2_below<T: Field + Real>(x: T) -> T {
    let two = T::one() + T::one();
    let mut s = T::one();
    if !(x > T::zero()) || !x.is_finite() {
        return s;
    }
    let mut guard = 0;
    while s * two <= x && guard < 1 << 14 {
        s *= two;
        guard += 1;
    }
    while s > x && guard < 1 << 14 {
        s /= two;
        guard += 1;
    }
    s
}

/// `Σ x_i y_i` with four independent accumulators.
#[inline]
pub(crate) fn dot<T: Field + Real>(x: &[T], y: &[T]) -> T {
    let len = x.len().min(y.len());
    let (x, y) = (&x[..len], &y[..len]);
    let mut acc = [T::zero(); 4];
    let mut xc = x.chunks_exact(4);
    let mut yc = y.chunks_exact(4);
    for (xs, ys) in (&mut xc).zip(&mut yc) {
        acc[0] = Real::mul_add(xs[0], ys[0], acc[0]);
        acc[1] = Real::mul_add(xs[1], ys[1], acc[1]);
        acc[2] = Real::mul_add(xs[2], ys[2], acc[2]);
        acc[3] = Real::mul_add(xs[3], ys[3], acc[3]);
    }
    let mut tail = T::zero();
    for (&a, &b) in xc.remainder().iter().zip(yc.remainder()) {
        tail = Real::mul_add(a, b, tail);
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `y += alpha · x`.
#[inline]
pub(crate) fn axpy<T: Field + Real>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = Real::mul_add(alpha, xi, *yi);
    }
}

/// `x *= alpha`.
#[inline]
pub(crate) fn scal<T: Field>(alpha: T, x: &mut [T]) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// Euclidean norm with scaled accumulation (no overflow or harmful
/// underflow for extreme magnitudes); NaN propagates.
pub(crate) fn nrm2<T: Field + Real>(x: &[T]) -> T {
    let mut scale = T::zero();
    let mut ssq = T::one();
    for &v in x {
        let a = Scalar::abs(v);
        // `!=` (not `>`) so a NaN entry enters the branch and propagates.
        if a != T::zero() {
            if scale < a {
                let t = scale / a;
                ssq = T::one() + ssq * t * t;
                scale = a;
            } else {
                let t = a / scale;
                ssq += t * t;
            }
        }
    }
    scale * Real::sqrt(ssq)
}

/// Generates an elementary reflector (LAPACK `DLARFG`).
///
/// Given `alpha` and `x`, computes `beta` and `tau` with
/// `(I - tau · v vᵀ) [alpha; x] = [beta; 0]`, `v = [1; x']`. On return `x`
/// holds `x'`. `tau = 0` (and `beta = alpha`) when `x` is already zero, so
/// the reflector is the identity.
pub(crate) fn larfg<T: Field + Real>(alpha: T, x: &mut [T]) -> (T, T) {
    let mut xnorm = nrm2(x);
    if xnorm == T::zero() {
        return (alpha, T::zero());
    }
    let c = Consts::<T>::new();
    let signed = |a: T, norm: T| if a >= T::zero() { -norm } else { norm };
    let mut alpha = alpha;
    let mut beta = signed(alpha, Real::hypot(alpha, xnorm));
    let safmin = c.safmin / c.eps;
    let mut knt = 0usize;
    if Scalar::abs(beta) < safmin {
        // beta may be inaccurate: rescale x and recompute (at most 20 times).
        let rsafmn = T::one() / safmin;
        loop {
            knt += 1;
            scal(rsafmn, x);
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(Scalar::abs(beta) < safmin) || knt >= 20 {
                break;
            }
        }
        xnorm = nrm2(x);
        beta = signed(alpha, Real::hypot(alpha, xnorm));
    }
    let tau = (beta - alpha) / beta;
    scal(T::one() / (alpha - beta), x);
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau)
}

/// `y[0..n] = A · x[0..n]` for the symmetric `n × n` matrix `A` whose lower
/// triangle is stored at `a[off + i + j * ld]` (`i >= j`). The strict upper
/// triangle is never read.
///
/// Four columns are processed per pass: each pass reads its columns once and
/// fuses the `A·x` update of the rows below with the transposed dot products.
pub(crate) fn symv_lower<T: Field + Real>(
    n: usize,
    a: &[T],
    off: usize,
    ld: usize,
    x: &[T],
    y: &mut [T],
) {
    const W: usize = 4;
    let x = &x[..n];
    let y = &mut y[..n];
    for v in y.iter_mut() {
        *v = T::zero();
    }
    let mut j = 0;
    while j + W <= n {
        // Diagonal block (lower entries only).
        for c in 0..W {
            let xc = x[j + c];
            let mut t = a[off + (j + c) + (j + c) * ld] * xc;
            for r in (c + 1)..W {
                let arc = a[off + (j + r) + (j + c) * ld];
                y[j + r] = Real::mul_add(arc, xc, y[j + r]);
                t = Real::mul_add(arc, x[j + r], t);
            }
            y[j + c] += t;
        }
        // Rows below the block.
        let r0 = j + W;
        if r0 < n {
            let len = n - r0;
            let col = |c: usize| &a[off + r0 + (j + c) * ld..off + r0 + (j + c) * ld + len];
            let xk = [x[j], x[j + 1], x[j + 2], x[j + 3]];
            let s = symv4_rows([col(0), col(1), col(2), col(3)], &x[r0..], &mut y[r0..], xk);
            for (c, sc) in s.iter().enumerate() {
                y[j + c] += *sc;
            }
        }
        j += W;
    }
    // Remaining columns.
    while j < n {
        let xj = x[j];
        let mut t = a[off + j + j * ld] * xj;
        if j + 1 < n {
            let len = n - j - 1;
            let col = &a[off + j + 1 + j * ld..off + j + 1 + j * ld + len];
            for ((yi, &ai), &xi) in y[j + 1..].iter_mut().zip(col).zip(&x[j + 1..]) {
                *yi = Real::mul_add(ai, xj, *yi);
                t = Real::mul_add(ai, xi, t);
            }
        }
        y[j] += t;
        j += 1;
    }
}

/// Row kernel of [`symv_lower`] for four columns `c`: for every row `r`,
/// `y[r] += Σ_k c[k][r]·xk[k]` and `s[k] += c[k][r]·x[r]`; returns `s`.
///
/// Chunks of four rows with array accumulators vectorise to two-lane FMAs
/// and keep the four dot products off the FMA latency chain.
#[inline]
fn symv4_rows<T: Field + Real>(c: [&[T]; 4], xs: &[T], ys: &mut [T], xk: [T; 4]) -> [T; 4] {
    let len = ys.len();
    let n4 = len - len % 4;
    let [c0, c1, c2, c3] = c.map(|col| &col[..len]);
    let xs = &xs[..len];
    let [x0, x1, x2, x3] = xk;
    let mut s0 = [T::zero(); 4];
    let mut s1 = [T::zero(); 4];
    let mut s2 = [T::zero(); 4];
    let mut s3 = [T::zero(); 4];
    {
        let it = c0[..n4]
            .chunks_exact(4)
            .zip(c1[..n4].chunks_exact(4))
            .zip(c2[..n4].chunks_exact(4))
            .zip(c3[..n4].chunks_exact(4))
            .zip(xs[..n4].chunks_exact(4))
            .zip(ys[..n4].chunks_exact_mut(4));
        for (((((a0, a1), a2), a3), xv), yv) in it {
            for t in 0..4 {
                let mut yt = yv[t];
                yt = Real::mul_add(a0[t], x0, yt);
                yt = Real::mul_add(a1[t], x1, yt);
                yt = Real::mul_add(a2[t], x2, yt);
                yt = Real::mul_add(a3[t], x3, yt);
                yv[t] = yt;
                s0[t] = Real::mul_add(a0[t], xv[t], s0[t]);
                s1[t] = Real::mul_add(a1[t], xv[t], s1[t]);
                s2[t] = Real::mul_add(a2[t], xv[t], s2[t]);
                s3[t] = Real::mul_add(a3[t], xv[t], s3[t]);
            }
        }
    }
    for r in n4..len {
        let xr = xs[r];
        let (u0, u1, u2, u3) = (c0[r], c1[r], c2[r], c3[r]);
        ys[r] = Real::mul_add(
            u3,
            x3,
            Real::mul_add(u2, x2, Real::mul_add(u1, x1, Real::mul_add(u0, x0, ys[r]))),
        );
        s0[0] = Real::mul_add(u0, xr, s0[0]);
        s1[0] = Real::mul_add(u1, xr, s1[0]);
        s2[0] = Real::mul_add(u2, xr, s2[0]);
        s3[0] = Real::mul_add(u3, xr, s3[0]);
    }
    let sum = |v: [T; 4]| (v[0] + v[1]) + (v[2] + v[3]);
    [sum(s0), sum(s1), sum(s2), sum(s3)]
}

/// Symmetric rank-2 update of the lower triangle (LAPACK `DSYR2`, lower):
/// `A += alpha · (x yᵀ + y xᵀ)` for the `n × n` block at `a[off..]`.
pub(crate) fn syr2_lower<T: Field + Real>(
    n: usize,
    a: &mut [T],
    off: usize,
    ld: usize,
    alpha: T,
    x: &[T],
    y: &[T],
) {
    let x = &x[..n];
    let y = &y[..n];
    for j in 0..n {
        let ax = alpha * x[j];
        let ay = alpha * y[j];
        let col = &mut a[off + j + j * ld..off + n + j * ld];
        for ((aij, &xi), &yi) in col.iter_mut().zip(&x[j..]).zip(&y[j..]) {
            *aij = Real::mul_add(xi, ay, Real::mul_add(yi, ax, *aij));
        }
    }
}

/// Indices `0..values.len()` sorted by ascending value (stable, total order;
/// NaN sorts last).
pub(crate) fn argsort<T: Field + Real>(values: &[T], out: &mut Vec<usize>) {
    out.clear();
    out.extend(0..values.len());
    out.sort_by(|&x, &y| {
        values[x]
            .partial_cmp(&values[y])
            .unwrap_or_else(|| values[x].is_nan().cmp(&values[y].is_nan()))
    });
}

/// Permutes the columns of the `nrows × perm.len()` block at `data[off..]`
/// in place so that new column `j` is old column `perm[j]`.
///
/// Follows the cycles of the permutation with one saved column, so every
/// column moves once (`O(n · nrows)` copies, no second matrix).
pub(crate) fn permute_columns<T: Copy>(
    data: &mut [T],
    off: usize,
    ld: usize,
    nrows: usize,
    perm: &[usize],
    saved: &mut Vec<T>,
    visited: &mut Vec<bool>,
) {
    let n = perm.len();
    visited.clear();
    visited.resize(n, false);
    for start in 0..n {
        if visited[start] || perm[start] == start {
            visited[start] = true;
            continue;
        }
        saved.clear();
        saved.extend_from_slice(&data[off + start * ld..off + start * ld + nrows]);
        let mut dst = start;
        loop {
            visited[dst] = true;
            let src = perm[dst];
            if src == start {
                data[off + dst * ld..off + dst * ld + nrows].copy_from_slice(saved);
                break;
            }
            data.copy_within(off + src * ld..off + src * ld + nrows, off + dst * ld);
            dst = src;
        }
    }
}
