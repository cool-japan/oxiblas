//! Tests for the per-call overhead work in the GEMM driver: packing buffers
//! reused across calls (stale contents, element types, nesting, size policy),
//! the edge-tile scratch tile, the hoisted micro-kernel dispatch, and the
//! transposed operands packed by the fast packers.

use super::*;
use crate::level3::gemm_packing::{
    PACK_CACHE_MAX_BYTES, PackSlot, pack_cache_bytes, pack_cache_capacity,
};
use num_complex::{Complex32, Complex64};
use oxiblas_matrix::Mat;

/// Element helpers for the reference checks.
trait Elem: Field + GemmKernel + bytemuck::Zeroable {
    fn sample(seed: usize) -> Self;
    fn nan() -> Self;
    fn is_finite_elem(self) -> bool;
    /// Absolute tolerance for a dot product of length `k` of O(1) values.
    fn tol(k: usize) -> f64;
    fn dist(self, other: Self) -> f64;
}

fn unit(seed: usize) -> f64 {
    ((seed.wrapping_mul(2_654_435_761) >> 9) % 2003) as f64 / 1001.5 - 1.0
}

impl Elem for f64 {
    fn sample(seed: usize) -> Self {
        unit(seed)
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn is_finite_elem(self) -> bool {
        self.is_finite()
    }
    fn tol(k: usize) -> f64 {
        1e-13 * k as f64 + 1e-13
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl Elem for f32 {
    fn sample(seed: usize) -> Self {
        unit(seed) as f32
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn is_finite_elem(self) -> bool {
        self.is_finite()
    }
    fn tol(k: usize) -> f64 {
        4e-6 * k as f64 + 1e-6
    }
    fn dist(self, other: Self) -> f64 {
        f64::from((self - other).abs())
    }
}

impl Elem for Complex64 {
    fn sample(seed: usize) -> Self {
        Complex64::new(unit(seed), unit(seed.wrapping_add(7_919)))
    }
    fn nan() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn is_finite_elem(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn tol(k: usize) -> f64 {
        4e-13 * k as f64 + 1e-13
    }
    fn dist(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

impl Elem for Complex32 {
    fn sample(seed: usize) -> Self {
        Complex32::new(unit(seed) as f32, unit(seed.wrapping_add(7_919)) as f32)
    }
    fn nan() -> Self {
        Complex32::new(f32::NAN, f32::NAN)
    }
    fn is_finite_elem(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn tol(k: usize) -> f64 {
        1.6e-5 * k as f64 + 1e-6
    }
    fn dist(self, other: Self) -> f64 {
        f64::from((self - other).norm())
    }
}

fn fill<T: Elem>(rows: usize, cols: usize, seed: usize) -> Mat<T> {
    let mut out: Mat<T> = Mat::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            out[(i, j)] = T::sample(i * 7_919 + j * 104_729 + seed * 1_299_709);
        }
    }
    out
}

/// Stored operand for `op(X)` of shape `rows × cols`.
fn operand<T: Elem>(trans: Trans, rows: usize, cols: usize, seed: usize) -> Mat<T> {
    if trans == Trans::NoTrans {
        fill(rows, cols, seed)
    } else {
        fill(cols, rows, seed)
    }
}

/// Naive `alpha * op(A) * op(B) + beta * C0`, with C0 not read when
/// `beta == 0`; never calls the blocked GEMM.
#[allow(clippy::too_many_arguments)]
fn reference<T: Elem>(
    ta: Trans,
    tb: Trans,
    alpha: T,
    a: &Mat<T>,
    b: &Mat<T>,
    beta: T,
    c0: &Mat<T>,
    (m, k, n): (usize, usize, usize),
) -> Mat<T> {
    let mut out: Mat<T> = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let mut acc = T::zero();
            for p in 0..k {
                acc += read_op(&a.as_ref(), ta, i, p) * read_op(&b.as_ref(), tb, p, j);
            }
            out[(i, j)] = if beta == T::zero() {
                alpha * acc
            } else {
                alpha * acc + beta * c0[(i, j)]
            };
        }
    }
    out
}

fn assert_close<T: Elem>(got: &Mat<T>, expected: &Mat<T>, k: usize, ctx: &str) {
    let tol = T::tol(k);
    for j in 0..got.ncols() {
        for i in 0..got.nrows() {
            let (g, e) = (got[(i, j)], expected[(i, j)]);
            assert!(g.is_finite_elem(), "{ctx}: non-finite at ({i},{j}): {g}");
            assert!(
                g.dist(e) <= tol,
                "{ctx}: ({i},{j}) got {g}, expected {e} (tol {tol})"
            );
        }
    }
}

/// Runs `C = alpha * op(A) * op(B) + beta * C` through the public entry point
/// and checks it against the reference. With `beta == 0`, C starts as NaN so a
/// read of C would show.
fn check_case<T: Elem>(
    ta: Trans,
    tb: Trans,
    (m, k, n): (usize, usize, usize),
    beta: T,
    seed: usize,
) {
    let a = operand::<T>(ta, m, k, seed);
    let b = operand::<T>(tb, k, n, seed + 1);
    let alpha = T::sample(seed + 2);
    let c0 = if beta == T::zero() {
        Mat::filled(m, n, T::nan())
    } else {
        fill::<T>(m, n, seed + 3)
    };
    let expected = reference(ta, tb, alpha, &a, &b, beta, &c0, (m, k, n));
    let mut c = c0.clone();
    gemm_transposed(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
    assert_close(
        &c,
        &expected,
        k,
        &format!("{} {ta:?}/{tb:?} {m}x{k}x{n}", core::any::type_name::<T>()),
    );
}

/// Fills this thread's packing buffers with NaN by running a large GEMM on
/// NaN operands (its result is irrelevant).
fn poison_pack_buffers<T: Elem>(m: usize, k: usize, n: usize) {
    let a = Mat::filled(m, k, T::nan());
    let b = Mat::filled(k, n, T::nan());
    let mut c: Mat<T> = Mat::zeros(m, n);
    gemm(T::one(), a.as_ref(), b.as_ref(), T::zero(), c.as_mut());
    // Transposed operands go through the other packers; poison those paths too.
    let at = Mat::filled(k, m, T::nan());
    let bt = Mat::filled(n, k, T::nan());
    gemm_transposed(
        Trans::Trans,
        Trans::Trans,
        T::one(),
        at.as_ref(),
        bt.as_ref(),
        T::zero(),
        c.as_mut(),
    );
}

/// A smaller GEMM with partial edge tiles in both dimensions, run on buffers
/// that a previous call left full of NaN, must see none of it.
fn check_no_stale_data<T: Elem>() {
    poison_pack_buffers::<T>(200, 180, 190);
    let shape = T::micro_kernel_shape();
    // One row and one column of partial micro-tiles (whatever MR / NR are).
    let m = 8 * shape.mr + shape.mr / 2 + 1;
    let n = 8 * shape.nr + shape.nr / 2 + 1;
    let all = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    for &ta in &all {
        for &tb in &all {
            check_case::<T>(ta, tb, (m, 61, n), T::zero(), 11);
            check_case::<T>(ta, tb, (m, 61, n), T::sample(5), 12);
        }
    }
}

#[test]
fn test_reused_buffers_hold_no_stale_data_f64() {
    check_no_stale_data::<f64>();
}

#[test]
fn test_reused_buffers_hold_no_stale_data_f32() {
    check_no_stale_data::<f32>();
}

#[test]
fn test_reused_buffers_hold_no_stale_data_c64() {
    check_no_stale_data::<Complex64>();
}

#[test]
fn test_reused_buffers_hold_no_stale_data_c32() {
    check_no_stale_data::<Complex32>();
}

#[test]
fn test_small_transposed_paths_match_reference() {
    // m * n * k <= SMALL_THRESHOLD: `gemm_small_trans` (axpy order when op(A)
    // has unit-stride columns, dot order otherwise).
    fn sweep<T: Elem>(betas: &[T]) {
        let all = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (5, 7, 3),
            (8, 8, 8),
            (2, 31, 3),
            (31, 2, 30),
        ] {
            assert!(m * n * k <= SMALL_THRESHOLD);
            for &ta in &all {
                for &tb in &all {
                    if ta == Trans::NoTrans && tb == Trans::NoTrans {
                        continue;
                    }
                    for &beta in betas {
                        check_case::<T>(ta, tb, (m, k, n), beta, m + 3 * k + 7 * n);
                    }
                }
            }
        }
    }
    sweep::<f64>(&[0.0, 1.0, -0.5]);
    sweep::<f32>(&[0.0, 0.75]);
    sweep::<Complex64>(&[Complex64::new(0.0, 0.0), Complex64::new(0.5, -0.25)]);
    sweep::<Complex32>(&[Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)]);

    // A strided (transposed-view) A with NoTrans takes the dot order.
    let (m, k, n) = (6usize, 9usize, 5usize);
    let stored = fill::<f64>(k, m, 91);
    let a_view = stored.as_ref().transpose();
    let mut a_copy: Mat<f64> = Mat::zeros(m, k);
    a_copy.copy_from(&a_view);
    let b = fill::<f64>(n, k, 92);
    let c0 = Mat::filled(m, n, f64::NAN);
    let expected = reference(
        Trans::NoTrans,
        Trans::Trans,
        0.5,
        &a_copy,
        &b,
        0.0,
        &c0,
        (m, k, n),
    );
    let mut c = c0.clone();
    gemm_transposed(
        Trans::NoTrans,
        Trans::Trans,
        0.5,
        a_view,
        b.as_ref(),
        0.0,
        c.as_mut(),
    );
    assert_close(&c, &expected, k, "strided A, small path");
}

#[test]
fn test_alternating_element_types_on_one_thread() {
    // Each element type has its own slots; interleaving them (with growing
    // and shrinking sizes) must never hand one type another's buffer.
    for (round, &(m, k, n)) in [(70usize, 50usize, 66usize), (40, 90, 35), (130, 70, 129)]
        .iter()
        .enumerate()
    {
        check_case::<f32>(Trans::NoTrans, Trans::NoTrans, (m, k, n), 0.0, round);
        check_case::<f64>(Trans::NoTrans, Trans::Trans, (m, k, n), 0.0, round);
        check_case::<Complex64>(
            Trans::ConjTrans,
            Trans::NoTrans,
            (m, k, n),
            Complex64::new(0.0, 0.0),
            round,
        );
        check_case::<Complex32>(
            Trans::NoTrans,
            Trans::NoTrans,
            (m, k, n),
            Complex32::new(0.5, -0.25),
            round,
        );
        check_case::<f64>(Trans::NoTrans, Trans::NoTrans, (m, k, n), 1.0, round);
    }
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
}

#[test]
fn test_edge_tiles_every_remainder() {
    // m and n run through every remainder modulo MR and NR (blocked path:
    // m * n * k > SMALL_THRESHOLD), with beta = 0 (NaN C), 1 and -0.5.
    fn sweep<T: Elem>(betas: &[T]) {
        let shape = T::micro_kernel_shape();
        let k = 40;
        // Whole micro-panels plus a remainder, large enough for the blocked
        // path with every kernel shape.
        let (m0, n0) = (64 / shape.mr * shape.mr, 48 / shape.nr * shape.nr);
        for dm in 0..shape.mr {
            for dn in 0..shape.nr {
                let (m, n) = (m0 + dm, n0 + dn);
                assert!(m * n * k > SMALL_THRESHOLD);
                for &beta in betas {
                    check_case::<T>(
                        Trans::NoTrans,
                        Trans::NoTrans,
                        (m, k, n),
                        beta,
                        dm * 31 + dn,
                    );
                }
            }
        }
    }
    sweep::<f64>(&[0.0, 1.0, -0.5]);
    sweep::<f32>(&[0.0, 1.0, -0.5]);
    sweep::<Complex64>(&[Complex64::new(0.0, 0.0), Complex64::new(-0.5, 0.75)]);
}

/// Transposed operands with unit row stride are packed by the fast packers
/// (A^T through the B packer, B^T through the A packer); a transposed *view*
/// of the same data still goes element-wise through `read_op`. Both must
/// produce the same packed panels as an explicitly transposed copy, so all
/// three results agree bitwise.
fn check_transposed_paths_bitwise<T: Elem>((m, k, n): (usize, usize, usize)) {
    let real = T::is_real();
    let a_nt = fill::<T>(m, k, 21);
    let b_nt = fill::<T>(k, n, 22);
    // Stored transposes (explicit copies), so op(A) = A_stored^T = a_nt.
    let mut a_st: Mat<T> = Mat::zeros(k, m);
    let mut b_st: Mat<T> = Mat::zeros(n, k);
    for i in 0..m {
        for p in 0..k {
            a_st[(p, i)] = a_nt[(i, p)];
        }
    }
    for p in 0..k {
        for j in 0..n {
            b_st[(j, p)] = b_nt[(p, j)];
        }
    }
    let alpha = T::sample(3);
    let beta = T::sample(4);
    let c0 = fill::<T>(m, n, 23);

    let mut c_nn = c0.clone();
    gemm(alpha, a_nt.as_ref(), b_nt.as_ref(), beta, c_nn.as_mut());

    let trans_modes: &[Trans] = if real {
        &[Trans::Trans, Trans::ConjTrans]
    } else {
        &[Trans::Trans]
    };
    for &t in trans_modes {
        let mut c_tt = c0.clone();
        gemm_transposed(
            t,
            t,
            alpha,
            a_st.as_ref(),
            b_st.as_ref(),
            beta,
            c_tt.as_mut(),
        );
        let mut c_tn = c0.clone();
        gemm_transposed(
            t,
            Trans::NoTrans,
            alpha,
            a_st.as_ref(),
            b_nt.as_ref(),
            beta,
            c_tn.as_mut(),
        );
        let mut c_nt = c0.clone();
        gemm_transposed(
            Trans::NoTrans,
            t,
            alpha,
            a_nt.as_ref(),
            b_st.as_ref(),
            beta,
            c_nt.as_mut(),
        );
        for j in 0..n {
            for i in 0..m {
                let e = c_nn[(i, j)];
                assert!(c_tt[(i, j)] == e, "{t:?}/{t:?} differs at ({i},{j})");
                assert!(c_tn[(i, j)] == e, "{t:?}/NoTrans differs at ({i},{j})");
                assert!(c_nt[(i, j)] == e, "NoTrans/{t:?} differs at ({i},{j})");
            }
        }
    }

    // Transposed views: element-wise packing, same panels.
    let mut c_views = c0.clone();
    gemm(
        alpha,
        a_st.as_ref().transpose(),
        b_st.as_ref().transpose(),
        beta,
        c_views.as_mut(),
    );
    for j in 0..n {
        for i in 0..m {
            assert!(
                c_views[(i, j)] == c_nn[(i, j)],
                "view path differs at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_transposed_fast_packers_bitwise_f64() {
    check_transposed_paths_bitwise::<f64>((67, 45, 53));
    // k > 512 takes the auto-tuned blocking with several KC blocks.
    check_transposed_paths_bitwise::<f64>((37, 700, 29));
}

#[test]
fn test_transposed_fast_packers_bitwise_f32() {
    check_transposed_paths_bitwise::<f32>((67, 45, 53));
    check_transposed_paths_bitwise::<f32>((70, 1100, 9));
}

#[test]
fn test_transposed_fast_packers_bitwise_complex() {
    // Complex `Trans` uses the fast packers too; `ConjTrans` stays element-wise
    // (checked against the reference in `check_no_stale_data`).
    check_transposed_paths_bitwise::<Complex64>((45, 33, 37));
    check_transposed_paths_bitwise::<Complex32>((45, 33, 37));
}

#[test]
fn test_strided_c_view_and_border_untouched() {
    // C is a view into a larger matrix: the kernel's direct stores, the edge
    // copy-back and beta handling must stay inside it.
    for &beta in &[0.0f64, 0.75] {
        let (m, k, n) = (69usize, 50usize, 61usize);
        let a = fill::<f64>(m, k, 31);
        let b = fill::<f64>(k, n, 32);
        let big0 = fill::<f64>(m + 7, n + 5, 33);
        let c0 = {
            let mut c0: Mat<f64> = Mat::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    c0[(i, j)] = big0[(i + 3, j + 2)];
                }
            }
            c0
        };
        let expected = reference(
            Trans::NoTrans,
            Trans::NoTrans,
            1.5,
            &a,
            &b,
            beta,
            &c0,
            (m, k, n),
        );
        let mut big = big0.clone();
        gemm(
            1.5,
            a.as_ref(),
            b.as_ref(),
            beta,
            big.as_mut().submatrix(3, 2, m, n),
        );
        for j in 0..n + 5 {
            for i in 0..m + 7 {
                let inside = (3..3 + m).contains(&i) && (2..2 + n).contains(&j);
                if inside {
                    let (g, e) = (big[(i, j)], expected[(i - 3, j - 2)]);
                    assert!((g - e).abs() <= f64::tol(k), "({i},{j}): {g} vs {e}");
                } else {
                    assert_eq!(
                        big[(i, j)].to_bits(),
                        big0[(i, j)].to_bits(),
                        "border ({i},{j}) written"
                    );
                }
            }
        }
    }
}

#[test]
fn test_pack_cache_stays_bounded_after_oversized_gemm() {
    // 8 x 416 x 4096 (short-wide): the packed B panel is 416 x 4098 f64,
    // about 13.6 MB, above the cache limit, so it must be freed on return.
    let (m, k, n) = (8usize, 416usize, 4096usize);
    let a = fill::<f64>(m, k, 41);
    let b = fill::<f64>(k, n, 42);
    let mut c: Mat<f64> = Mat::zeros(m, n);
    gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
    assert!(pack_cache_capacity::<f64>(PackSlot::B) * 8 <= PACK_CACHE_MAX_BYTES);
    // Spot-check the last 96 columns.
    let mut b_tail: Mat<f64> = Mat::zeros(k, 96);
    b_tail.copy_from(&b.as_ref().submatrix(0, 4000, k, 96));
    let expected = reference(
        Trans::NoTrans,
        Trans::NoTrans,
        1.0,
        &a,
        &b_tail,
        0.0,
        &Mat::zeros(m, 96),
        (m, k, 96),
    );
    for j in 0..96 {
        for i in 0..m {
            let (g, e) = (c[(i, 4000 + j)], expected[(i, j)]);
            assert!((g - e).abs() <= f64::tol(k), "({i},{j}): {g} vs {e}");
        }
    }
    // The next, small call still works and caches normally.
    check_case::<f64>(Trans::NoTrans, Trans::NoTrans, (70, 60, 50), 0.0, 43);
    assert!(pack_cache_capacity::<f64>(PackSlot::B) > 0);
}

#[test]
fn test_dispatch_resolved_once_matches_per_tile_dispatch() {
    // `macro_panel_multiply_with` calls the resolved kernel; the trait's
    // `micro_kernel` re-dispatches. Same shape and bitwise-same tiles.
    fn check<T: Elem>() {
        let kernel = T::micro_kernel_dispatch();
        let shape = T::micro_kernel_shape();
        assert_eq!(kernel.shape, shape);
        let (mr, nr, k) = (shape.mr, shape.nr, 37usize);
        let a: Vec<T> = (0..mr * k).map(|i| T::sample(i)).collect();
        let b: Vec<T> = (0..k * nr).map(|i| T::sample(i + 999)).collect();
        for &beta in &[T::zero(), T::sample(7)] {
            let c0: Vec<T> = (0..mr * nr).map(|i| T::sample(i + 77)).collect();
            let mut c1 = c0.clone();
            let mut c2 = c0.clone();
            let alpha = T::sample(5);
            // SAFETY: `a` holds MR * k, `b` holds k * NR and each C buffer
            // holds MR * NR elements with leading dimension MR, the layout
            // both calls read and write; the buffers do not overlap.
            unsafe {
                (kernel.kernel)(k, alpha, a.as_ptr(), b.as_ptr(), beta, c1.as_mut_ptr(), mr);
                T::micro_kernel(k, alpha, a.as_ptr(), b.as_ptr(), beta, c2.as_mut_ptr(), mr);
            }
            assert!(
                c1 == c2,
                "{}: dispatch mismatch",
                core::any::type_name::<T>()
            );
        }
    }
    check::<f64>();
    check::<f32>();
    check::<Complex64>();
    check::<Complex32>();
}

#[cfg(feature = "parallel")]
#[test]
fn test_nested_gemms_inside_parallel_gemm() {
    use rayon::prelude::*;

    // An outer parallel GEMM runs while sequential and parallel GEMMs run in
    // other tasks of the same pool: work stealing makes a worker that is
    // waiting inside one GEMM run tile tasks or whole GEMMs of another, so
    // the per-thread slots are taken while already on loan.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("thread pool");
    let (bm, bk, bn) = (260usize, 230usize, 250usize);
    let big_a = fill::<f64>(bm, bk, 51);
    let big_b = fill::<f64>(bk, bn, 52);
    let big_ref = reference(
        Trans::NoTrans,
        Trans::NoTrans,
        1.0,
        &big_a,
        &big_b,
        0.0,
        &Mat::zeros(bm, bn),
        (bm, bk, bn),
    );
    let small: Vec<(Mat<f64>, Mat<f64>, Mat<f64>)> = (0..12)
        .map(|i| {
            let (m, k, n) = (60 + i, 50 + 2 * i, 70 - i);
            let a = fill::<f64>(m, k, 60 + i);
            let b = fill::<f64>(k, n, 80 + i);
            let r = reference(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a,
                &b,
                0.0,
                &Mat::zeros(m, n),
                (m, k, n),
            );
            (a, b, r)
        })
        .collect();

    pool.install(|| {
        let (big_c, smalls) = rayon::join(
            || {
                let mut c = Mat::filled(bm, bn, f64::NAN);
                gemm_with_par(
                    1.0,
                    big_a.as_ref(),
                    big_b.as_ref(),
                    0.0,
                    c.as_mut(),
                    Par::Rayon,
                );
                c
            },
            || {
                small
                    .par_iter()
                    .enumerate()
                    .map(|(i, (a, b, _))| {
                        let mut c = Mat::filled(a.nrows(), b.ncols(), f64::NAN);
                        let par = if i % 2 == 0 { Par::Seq } else { Par::Rayon };
                        gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), par);
                        c
                    })
                    .collect::<Vec<_>>()
            },
        );
        assert_close(&big_c, &big_ref, bk, "outer parallel GEMM");
        for (i, (c, (_, b, r))) in smalls.iter().zip(&small).enumerate() {
            assert_close(c, r, b.nrows(), &format!("nested GEMM {i}"));
        }
    });
}
