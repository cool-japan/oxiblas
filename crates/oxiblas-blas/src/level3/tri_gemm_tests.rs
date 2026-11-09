//! Unit tests for the triangle-only blocked kernel ([`super::tri_update`]),
//! plus the helpers shared with the SYRK / SYR2K / GEMMT API tests
//! (`syrk_family_tests.rs`).
//!
//! The test data are small dyadic rationals (multiples of 1/4 in
//! `[-1.5, 1.5]`), so every product, sum and scaling by the test scalars is
//! exact in f32 and f64 for the sizes used: results can be compared with the
//! reference regardless of the summation order, and an entry computed from
//! the wrong operands or written to the wrong place cannot hide in rounding.

use super::*;
use crate::level3::gemm::{gemm_transposed, gemm_transposed_with_blocking};
use num_complex::{Complex32, Complex64};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

// =============================================================================
// Shared helpers
// =============================================================================

/// Element types under test, with a widening to `Complex64` for the
/// reference arithmetic.
pub(crate) trait Elem: Field + GemmKernel + bytemuck::Zeroable + core::fmt::Debug {
    /// Unit roundoff scale used by the error bound.
    const EPS: f64;
    /// Whether the type is complex (then `ConjTrans` conjugates).
    const COMPLEX: bool;
    fn from_c64(z: Complex64) -> Self;
    fn to_c64(self) -> Complex64;
    fn nan() -> Self;
    /// Raw bits of every component (for bitwise-unchanged checks).
    fn bits(self) -> [u64; 2];
}

impl Elem for f64 {
    const EPS: f64 = f64::EPSILON;
    const COMPLEX: bool = false;
    fn from_c64(z: Complex64) -> Self {
        z.re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn bits(self) -> [u64; 2] {
        [self.to_bits(), 0]
    }
}

impl Elem for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    const COMPLEX: bool = false;
    fn from_c64(z: Complex64) -> Self {
        z.re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn bits(self) -> [u64; 2] {
        [u64::from(self.to_bits()), 0]
    }
}

impl Elem for Complex64 {
    const EPS: f64 = f64::EPSILON;
    const COMPLEX: bool = true;
    fn from_c64(z: Complex64) -> Self {
        z
    }
    fn to_c64(self) -> Complex64 {
        self
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn bits(self) -> [u64; 2] {
        [self.re.to_bits(), self.im.to_bits()]
    }
}

impl Elem for Complex32 {
    const EPS: f64 = f32::EPSILON as f64;
    const COMPLEX: bool = true;
    fn from_c64(z: Complex64) -> Self {
        Complex32::new(z.re as f32, z.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
    fn nan() -> Self {
        Complex32::new(f32::NAN, f32::NAN)
    }
    fn bits(self) -> [u64; 2] {
        [u64::from(self.re.to_bits()), u64::from(self.im.to_bits())]
    }
}

/// Deterministic dyadic test value in `[-1.5, 1.5]` (both components).
pub(crate) fn val(seed: usize, i: usize, j: usize) -> Complex64 {
    let re = ((i * 7 + j * 3 + seed * 5) % 13) as f64 * 0.25 - 1.5;
    let im = ((i * 5 + j * 11 + seed * 3) % 11) as f64 * 0.25 - 1.25;
    Complex64::new(re, im)
}

/// An `r × c` matrix of [`val`] entries.
pub(crate) fn mat<T: Elem>(seed: usize, r: usize, c: usize) -> Mat<T> {
    let mut m = Mat::<T>::zeros(r, c);
    for j in 0..c {
        for i in 0..r {
            m[(i, j)] = T::from_c64(val(seed, i, j));
        }
    }
    m
}

/// An `r × c` matrix of NaN.
pub(crate) fn nan_mat<T: Elem>(r: usize, c: usize) -> Mat<T> {
    Mat::<T>::filled(r, c, T::nan())
}

/// Whether `(i, j)` lies in the `uplo` triangle.
pub(crate) fn in_tri(uplo: Uplo, i: usize, j: usize) -> bool {
    match uplo {
        Uplo::Lower => i >= j,
        Uplo::Upper => i <= j,
    }
}

/// `op(x)[r, c]`, widened.
pub(crate) fn op_at<T: Elem>(x: &MatRef<'_, T>, t: Trans, r: usize, c: usize) -> Complex64 {
    match t {
        Trans::NoTrans => x[(r, c)].to_c64(),
        Trans::Trans => x[(c, r)].to_c64(),
        Trans::ConjTrans => {
            let z = x[(c, r)].to_c64();
            if T::COMPLEX { z.conj() } else { z }
        }
    }
}

/// Initial C for a case: the triangle holds [`val`] entries (or NaN when
/// `beta == 0`, which must never be read); the other triangle holds NaN, Inf
/// and signed zeros, which must stay bitwise untouched.
pub(crate) fn initial_c<T: Elem>(uplo: Uplo, n: usize, beta_is_zero: bool) -> Mat<T> {
    let mut c = Mat::<T>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            c[(i, j)] = if in_tri(uplo, i, j) {
                if beta_is_zero {
                    T::nan()
                } else {
                    T::from_c64(val(9, i, j))
                }
            } else {
                match (i + 2 * j) % 4 {
                    0 => T::nan(),
                    1 => T::from_c64(Complex64::new(f64::INFINITY, f64::NEG_INFINITY)),
                    2 => T::from_c64(Complex64::new(-0.0, -0.0)),
                    _ => T::from_c64(val(4, i, j)),
                }
            };
        }
    }
    c
}

/// One k-segment of a reference computation: stored operands and transposes.
pub(crate) type RefSeg<'a, T> = (MatRef<'a, T>, Trans, MatRef<'a, T>, Trans);

/// Checks `got` (an `n × n` view) against
/// `beta·before + alpha·Σ op(A)·op(B)` on the `uplo` triangle and requires the
/// other triangle to be bitwise equal to `before`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_tri<T: Elem>(
    what: &str,
    uplo: Uplo,
    alpha: T,
    beta: T,
    segs: &[RefSeg<'_, T>],
    before: &MatRef<'_, T>,
    got: &MatRef<'_, T>,
) {
    let n = got.nrows();
    let (alpha_w, beta_w) = (alpha.to_c64(), beta.to_c64());
    let k_total: usize = segs.iter().map(|s| op_dims(&s.0, s.1).1).sum();
    for j in 0..n {
        for i in 0..n {
            let g = got[(i, j)];
            if !in_tri(uplo, i, j) {
                assert_eq!(
                    g.bits(),
                    before[(i, j)].bits(),
                    "{what}: C[{i},{j}] outside the triangle changed"
                );
                continue;
            }
            let mut acc = Complex64::new(0.0, 0.0);
            let mut mag = 0.0f64;
            if alpha != T::zero() {
                for &(a, ta, b, tb) in segs {
                    let k = op_dims(&a, ta).1;
                    for l in 0..k {
                        let p = op_at(&a, ta, i, l) * op_at(&b, tb, l, j);
                        acc += p;
                        mag += p.norm();
                    }
                }
            }
            let c_term = if beta == T::zero() {
                Complex64::new(0.0, 0.0)
            } else {
                beta_w * before[(i, j)].to_c64()
            };
            let want = alpha_w * acc + c_term;
            let scale = alpha_w.norm() * mag + c_term.norm() + 1.0;
            let tol = 8.0 * (k_total as f64 + 4.0) * T::EPS * scale;
            let err = (g.to_c64() - want).norm();
            assert!(
                err <= tol,
                "{what}: C[{i},{j}] = {:?}, expected {want} (err {err:e}, tol {tol:e})",
                g.to_c64()
            );
        }
    }
}

/// The 16 `(alpha, beta)` pairs over `{0, 1, -1, 2.5}`.
pub(crate) fn scalar_pairs() -> Vec<(f64, f64)> {
    let vals = [0.0, 1.0, -1.0, 2.5];
    let mut out = Vec::new();
    for &a in &vals {
        for &b in &vals {
            out.push((a, b));
        }
    }
    out
}

/// A complex scalar for the complex types (the real types keep `re`).
pub(crate) fn scalar<T: Elem>(x: f64) -> T {
    // A non-real factor for the complex types makes the complex products
    // non-trivial; 0 and 1 stay exactly 0 and 1.
    if x == 0.0 || x == 1.0 {
        T::from_c64(Complex64::new(x, 0.0))
    } else {
        T::from_c64(Complex64::new(x, 0.5))
    }
}

// =============================================================================
// Kernel tests
// =============================================================================

/// Runs `tri_update` (optionally with forced blocking) and checks it against
/// the reference, with NaN planted wherever it must not be read.
#[allow(clippy::too_many_arguments)]
fn run_case<T: Elem>(
    uplo: Uplo,
    n: usize,
    ks: &[usize],
    trans: (Trans, Trans),
    two_operands: bool,
    alpha: f64,
    beta: f64,
    blocking: Option<&GemmBlocking>,
) {
    let (ta, tb) = trans;
    let (alpha_t, beta_t) = (scalar::<T>(alpha), scalar::<T>(beta));
    // Stored operands per segment: alternately (X, Y) and (Y, X) like SYR2K
    // when `two_operands`, otherwise independent matrices.
    let mut store = Vec::new();
    for (s, &k) in ks.iter().enumerate() {
        let (ar, ac) = if ta == Trans::NoTrans { (n, k) } else { (k, n) };
        let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
        let (a, b) = if alpha == 0.0 {
            (nan_mat::<T>(ar, ac), nan_mat::<T>(br, bc))
        } else if two_operands {
            (mat::<T>(1 + s, ar, ac), mat::<T>(2 + s, br, bc))
        } else {
            (mat::<T>(3 * s + 1, ar, ac), mat::<T>(3 * s + 2, br, bc))
        };
        store.push((a, b));
    }
    let segs: Vec<KSegment<'_, T>> = store
        .iter()
        .map(|(a, b)| KSegment::new(a.as_ref(), ta, b.as_ref(), tb))
        .collect();
    let refs: Vec<RefSeg<'_, T>> = store
        .iter()
        .map(|(a, b)| (a.as_ref(), ta, b.as_ref(), tb))
        .collect();
    let before = initial_c::<T>(uplo, n, beta == 0.0);
    let mut c = before.clone();
    match blocking {
        Some(b) => tri_update_with_blocking(uplo, alpha_t, &segs, beta_t, c.as_mut(), b),
        None => tri_update(uplo, alpha_t, &segs, beta_t, c.as_mut()),
    }
    let what = format!(
        "{} {uplo:?} {ta:?}/{tb:?} n={n} ks={ks:?} alpha={alpha} beta={beta} blocking={:?}",
        core::any::type_name::<T>(),
        blocking.map(|b| (b.mc, b.kc, b.nc))
    );
    check_tri(
        &what,
        uplo,
        alpha_t,
        beta_t,
        &refs,
        &before.as_ref(),
        &c.as_ref(),
    );
}

/// Tiny blocking targets: several MC, NC and KC blocks even for small `n`.
fn tiny_blocking<T: Elem>() -> GemmBlocking {
    let shape = T::micro_kernel_dispatch().shape;
    GemmBlocking {
        mc: 2 * shape.mr,
        kc: 8,
        nc: 2 * shape.nr,
    }
}

fn forced_blocking_sweep<T: Elem>() {
    let tiny = tiny_blocking::<T>();
    let pairs = scalar_pairs();
    let mut idx = 0usize;
    for uplo in [Uplo::Lower, Uplo::Upper] {
        for &n in &[1usize, 2, 7, 23, 50] {
            for ks in [&[1usize][..], &[9], &[17], &[40], &[17, 17], &[9, 3]] {
                for trans in [
                    (Trans::NoTrans, Trans::Trans),
                    (Trans::Trans, Trans::NoTrans),
                    (Trans::NoTrans, Trans::NoTrans),
                    (Trans::Trans, Trans::Trans),
                ] {
                    let (alpha, beta) = pairs[idx % pairs.len()];
                    idx += 1;
                    run_case::<T>(uplo, n, ks, trans, ks.len() == 2, alpha, beta, Some(&tiny));
                }
            }
        }
    }
}

#[test]
fn tri_update_forced_blocking_f64() {
    forced_blocking_sweep::<f64>();
}

#[test]
fn tri_update_forced_blocking_f32() {
    forced_blocking_sweep::<f32>();
}

#[test]
fn tri_update_forced_blocking_c64() {
    forced_blocking_sweep::<Complex64>();
}

#[test]
fn tri_update_forced_blocking_c32() {
    forced_blocking_sweep::<Complex32>();
}

/// Every `(alpha, beta)` pair at default blocking, sizes that are not
/// multiples of any kernel's MR or NR.
fn scalar_cross<T: Elem>() {
    for (alpha, beta) in scalar_pairs() {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            run_case::<T>(
                uplo,
                45,
                &[29],
                (Trans::NoTrans, Trans::Trans),
                false,
                alpha,
                beta,
                None,
            );
            run_case::<T>(
                uplo,
                19,
                &[13, 13],
                (Trans::Trans, Trans::NoTrans),
                true,
                alpha,
                beta,
                None,
            );
        }
    }
}

#[test]
fn tri_update_scalar_cross_real() {
    scalar_cross::<f64>();
    scalar_cross::<f32>();
}

#[test]
fn tri_update_scalar_cross_complex() {
    scalar_cross::<Complex64>();
    scalar_cross::<Complex32>();
}

/// With GEMM's blocking, `beta == 0` and `beta == 1` results are bitwise
/// equal to the triangle of `gemm_transposed` (signed zeros aside): the same
/// packers, kernel and accumulation order.
fn bitwise_vs_gemm<T: Elem>(blocking: Option<GemmBlocking>) {
    // n²k above GEMM's small-matrix threshold (32³), so GEMM is blocked too.
    for &(n, k) in &[(64usize, 40usize), (97, 33), (70, 130)] {
        for uplo in [Uplo::Lower, Uplo::Upper] {
            for (ta, tb) in [
                (Trans::NoTrans, Trans::Trans),
                (Trans::Trans, Trans::NoTrans),
                (Trans::NoTrans, Trans::NoTrans),
                (Trans::ConjTrans, Trans::Trans),
            ] {
                let (ar, ac) = if ta == Trans::NoTrans { (n, k) } else { (k, n) };
                let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
                let a = mat::<T>(5, ar, ac);
                let b = mat::<T>(6, br, bc);
                let alpha = scalar::<T>(-1.0);
                for beta_f in [0.0, 1.0] {
                    let beta = scalar::<T>(beta_f);
                    let c0 = mat::<T>(7, n, n);
                    let mut full = c0.clone();
                    match &blocking {
                        Some(bl) => gemm_transposed_with_blocking(
                            ta,
                            tb,
                            alpha,
                            a.as_ref(),
                            b.as_ref(),
                            beta,
                            full.as_mut(),
                            Par::Seq,
                            bl,
                        ),
                        None => gemm_transposed(
                            ta,
                            tb,
                            alpha,
                            a.as_ref(),
                            b.as_ref(),
                            beta,
                            full.as_mut(),
                        ),
                    }
                    let mut tri = c0.clone();
                    let seg = [KSegment::new(a.as_ref(), ta, b.as_ref(), tb)];
                    match &blocking {
                        Some(bl) => {
                            tri_update_with_blocking(uplo, alpha, &seg, beta, tri.as_mut(), bl)
                        }
                        None => tri_update(uplo, alpha, &seg, beta, tri.as_mut()),
                    }
                    for j in 0..n {
                        for i in 0..n {
                            let (t, f) = (tri[(i, j)].to_c64(), full[(i, j)].to_c64());
                            if in_tri(uplo, i, j) {
                                // `==` treats -0 and +0 as equal, nothing else.
                                assert!(
                                    t.re == f.re && t.im == f.im,
                                    "{} {uplo:?} {ta:?}/{tb:?} n={n} k={k} beta={beta_f}: \
                                     C[{i},{j}] = {t} vs GEMM {f}",
                                    core::any::type_name::<T>()
                                );
                            } else {
                                assert_eq!(tri[(i, j)].bits(), c0[(i, j)].bits());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tri_update_bitwise_equals_gemm_real() {
    bitwise_vs_gemm::<f64>(None);
    bitwise_vs_gemm::<f32>(None);
    bitwise_vs_gemm::<f64>(Some(tiny_blocking::<f64>()));
    bitwise_vs_gemm::<f32>(Some(tiny_blocking::<f32>()));
}

#[test]
fn tri_update_bitwise_equals_gemm_complex() {
    bitwise_vs_gemm::<Complex64>(None);
    bitwise_vs_gemm::<Complex32>(None);
    bitwise_vs_gemm::<Complex64>(Some(tiny_blocking::<Complex64>()));
}

/// Transposed and strided operand views go through the element-wise packers
/// and must give the same result as contiguous storage; C may be a submatrix
/// whose padding must stay bitwise untouched.
fn views<T: Elem>() {
    let (n, k) = (37usize, 21usize);
    for uplo in [Uplo::Lower, Uplo::Upper] {
        // A stored transposed (k × n), viewed as n × k: col_stride != 1.
        let at = mat::<T>(11, k, n);
        let a_view = at.as_ref().transpose();
        // B as a strided submatrix of a larger matrix.
        let big_b = mat::<T>(12, n + 5, k + 3);
        let b_view = big_b.as_ref().submatrix(2, 1, n, k);
        // C as a submatrix of a NaN-padded larger matrix.
        let (pr, pc) = (3usize, 2usize);
        let mut big_c = nan_mat::<T>(n + 6, n + 4);
        let c_init = initial_c::<T>(uplo, n, false);
        for j in 0..n {
            for i in 0..n {
                big_c[(pr + i, pc + j)] = c_init[(i, j)];
            }
        }
        let big_before = big_c.clone();
        let (alpha, beta) = (scalar::<T>(2.5), scalar::<T>(-1.0));
        let segs = [
            KSegment::new(a_view, Trans::NoTrans, b_view, Trans::Trans),
            KSegment::new(b_view, Trans::NoTrans, a_view, Trans::Trans),
        ];
        {
            let c_view = big_c.as_mut().submatrix(pr, pc, n, n);
            tri_update(uplo, alpha, &segs, beta, c_view);
        }
        let refs: Vec<RefSeg<'_, T>> = segs
            .iter()
            .map(|s| (s.a, s.trans_a, s.b, s.trans_b))
            .collect();
        let what = format!("views {} {uplo:?}", core::any::type_name::<T>());
        check_tri(
            &what,
            uplo,
            alpha,
            beta,
            &refs,
            &big_before.as_ref().submatrix(pr, pc, n, n),
            &big_c.as_ref().submatrix(pr, pc, n, n),
        );
        for j in 0..n + 4 {
            for i in 0..n + 6 {
                let inside = (pr..pr + n).contains(&i) && (pc..pc + n).contains(&j);
                if !inside {
                    assert_eq!(
                        big_c[(i, j)].bits(),
                        big_before[(i, j)].bits(),
                        "{what}: padding C[{i},{j}] changed"
                    );
                }
            }
        }
    }
}

#[test]
fn tri_update_views() {
    views::<f64>();
    views::<f32>();
    views::<Complex64>();
    views::<Complex32>();
}

/// `beta == 1` must only ever add to C: an infinite component of C stays
/// infinite and the other component finite (the complex kernels' `(1, 0)·C`
/// forms `0·∞` for the other component, which makes it NaN).
#[test]
fn tri_update_beta_one_only_adds() {
    let n = 40;
    let k = 12;
    let a = mat::<Complex64>(1, n, k);
    let mut c = Mat::<Complex64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            c[(i, j)] = Complex64::new(f64::INFINITY, -1.0);
        }
    }
    let seg = [KSegment::new(
        a.as_ref(),
        Trans::NoTrans,
        a.as_ref(),
        Trans::Trans,
    )];
    tri_update(
        Uplo::Lower,
        Complex64::new(1.0, 0.0),
        &seg,
        Complex64::new(1.0, 0.0),
        c.as_mut(),
    );
    for j in 0..n {
        for i in j..n {
            assert_eq!(c[(i, j)].re, f64::INFINITY, "C[{i},{j}] = {}", c[(i, j)]);
            assert!(c[(i, j)].im.is_finite(), "C[{i},{j}] = {}", c[(i, j)]);
        }
    }
}

/// `scale_triangle`: `beta == 0` zero-fills without reading, `beta == 1` is
/// bitwise a no-op, and the other triangle is never touched.
#[test]
fn scale_triangle_semantics() {
    for uplo in [Uplo::Lower, Uplo::Upper] {
        for beta in [0.0f64, 1.0, -1.0, 2.5] {
            let n = 9;
            let before = initial_c::<f64>(uplo, n, beta == 0.0);
            let mut c = before.clone();
            scale_triangle(uplo, beta, &mut c.as_mut());
            for j in 0..n {
                for i in 0..n {
                    if in_tri(uplo, i, j) && beta != 1.0 {
                        let want = if beta == 0.0 {
                            0.0
                        } else {
                            beta * before[(i, j)]
                        };
                        assert_eq!(c[(i, j)].to_bits(), want.to_bits(), "{uplo:?} beta={beta}");
                    } else {
                        assert_eq!(c[(i, j)].to_bits(), before[(i, j)].to_bits());
                    }
                }
            }
        }
    }
}
