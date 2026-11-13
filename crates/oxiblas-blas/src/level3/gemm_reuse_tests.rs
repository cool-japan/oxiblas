//! Tests for the per-call overhead work in the GEMM driver: packing buffers
//! reused across calls (stale contents, element types, nesting, size policy),
//! the edge-tile scratch tile, the hoisted micro-kernel dispatch, and the
//! transposed operands packed by the fast packers.

use super::*;
use crate::level3::gemm_packing::{
    PACK_CACHE_MAX_BYTES, PackSlot, pack_cache_bytes, pack_cache_capacity,
};
use crate::level3::gemm_plan::GemmPolicy;
use crate::level3::gemm_small::SMALL_THRESHOLD;
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
fn check_case<T: Elem>(ta: Trans, tb: Trans, dims: (usize, usize, usize), beta: T, seed: usize) {
    check_case_on(None, ta, tb, dims, beta, seed);
}

/// [`check_case`] on the blocked route: the build's plan for the problem
/// (`DEFAULT_POLICY`, its kernel and blocking) with [`GemmRoute::Blocked`]
/// forced, so the packing buffers and the blocked driver are exercised under
/// every policy (the x86 candidate sends products up to 80³ to its small
/// kernels).
fn check_case_blocked<T: Elem>(
    ta: Trans,
    tb: Trans,
    dims: (usize, usize, usize),
    beta: T,
    seed: usize,
) {
    check_case_on(Some(GemmRoute::Blocked), ta, tb, dims, beta, seed);
}

/// [`check_case`] through the public entry point (`route == None`) or
/// through [`gemm_planned`] with `route` forced.
fn check_case_on<T: Elem>(
    route: Option<GemmRoute>,
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
    match route {
        None => gemm_transposed(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut()),
        Some(route) => {
            let plan = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m, n, k).with_forced_route(route);
            gemm_planned(
                ta,
                tb,
                alpha,
                a.as_ref(),
                b.as_ref(),
                beta,
                c.as_mut(),
                Par::Seq,
                BlockingSource::Default,
                &plan,
            );
        }
    }
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
            check_case_blocked::<T>(ta, tb, (m, 61, n), T::zero(), 11);
            check_case_blocked::<T>(ta, tb, (m, 61, n), T::sample(5), 12);
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
        check_case_blocked::<f32>(Trans::NoTrans, Trans::NoTrans, (m, k, n), 0.0, round);
        check_case_blocked::<f64>(Trans::NoTrans, Trans::Trans, (m, k, n), 0.0, round);
        check_case_blocked::<Complex64>(
            Trans::ConjTrans,
            Trans::NoTrans,
            (m, k, n),
            Complex64::new(0.0, 0.0),
            round,
        );
        check_case_blocked::<Complex32>(
            Trans::NoTrans,
            Trans::NoTrans,
            (m, k, n),
            Complex32::new(0.5, -0.25),
            round,
        );
        check_case_blocked::<f64>(Trans::NoTrans, Trans::NoTrans, (m, k, n), 1.0, round);
    }
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
}

#[test]
fn test_edge_tiles_every_remainder() {
    // m and n run through every remainder modulo MR and NR on the blocked
    // route (forced: the x86 candidate's small route reaches 80³), with
    // beta = 0 (NaN C), 1 and -0.5. The plan's kernel may differ from the
    // canonical one, so the remainders run up to the larger MR and NR of the
    // two (every MR and NR of this crate's kernels divides the larger one,
    // or is at most it: 8 | 16, 6 <= 8).
    fn sweep<T: Elem>(betas: &[T]) {
        let k = 40;
        // 64 and 48 are whole micro-panels of every kernel shape (MR in
        // {2, 4, 8, 16}, NR in {2, 4, 6, 8, 16}).
        let (m0, n0) = (64, 48);
        let canonical = T::micro_kernel_shape();
        let planned = GemmPlan::<T>::for_problem(DEFAULT_POLICY, m0, n0, k)
            .kernel
            .shape;
        for dm in 0..canonical.mr.max(planned.mr) {
            for dn in 0..canonical.nr.max(planned.nr) {
                let (m, n) = (m0 + dm, n0 + dn);
                assert!(m * n * k > SMALL_THRESHOLD);
                for &beta in betas {
                    check_case_blocked::<T>(
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
///
/// Every product must take the blocked route under every policy (above the
/// largest small-route ceiling, the x86 candidate's 80³), or the
/// column-major `NoTrans` reference would run on the small kernels.
fn check_transposed_paths_bitwise<T: Elem>((m, k, n): (usize, usize, usize)) {
    assert!(
        m * n * k > crate::level3::gemm_plan::CANDIDATE_X86.small_simd_max_work,
        "{m}x{k}x{n} must take the blocked route"
    );
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
    check_transposed_paths_bitwise::<f64>((83, 75, 89));
    // k > 512 takes the auto-tuned blocking with several KC blocks.
    check_transposed_paths_bitwise::<f64>((37, 700, 29));
}

#[test]
fn test_transposed_fast_packers_bitwise_f32() {
    check_transposed_paths_bitwise::<f32>((83, 75, 89));
    check_transposed_paths_bitwise::<f32>((70, 1100, 9));
}

#[test]
fn test_transposed_fast_packers_bitwise_complex() {
    // Complex `Trans` uses the fast packers too; `ConjTrans` stays element-wise
    // (checked against the reference in `check_no_stale_data`).
    check_transposed_paths_bitwise::<Complex64>((83, 75, 89));
    check_transposed_paths_bitwise::<Complex32>((83, 75, 89));
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
    // 8 x 416 x 4096 (short-wide) with the auto-tuned blocking passed
    // explicitly, so no B-block cap can shrink it (a capping policy would
    // make this test vacuous): the packed B panel is about 416 x 4098 f64,
    // some 13.6 MB, above the cache limit, so it must be freed on return.
    let (m, k, n) = (8usize, 416usize, 4096usize);
    let shape = f64::micro_kernel_shape();
    let targets = GemmBlocking::auto_tuned::<f64>(m, k, n, &shape);
    let fitted = targets.fit_to_dims(m, k, n, &shape);
    let b_bytes = fitted.kc * fitted.nc.div_ceil(shape.nr) * shape.nr * 8;
    assert!(
        b_bytes > PACK_CACHE_MAX_BYTES,
        "{fitted:?}: the packed B block ({b_bytes} bytes) must exceed the cache limit"
    );
    let a = fill::<f64>(m, k, 41);
    let b = fill::<f64>(k, n, 42);
    let mut c: Mat<f64> = Mat::zeros(m, n);
    gemm_with_blocking(
        1.0,
        a.as_ref(),
        b.as_ref(),
        0.0,
        c.as_mut(),
        Par::Seq,
        &targets,
    );
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
    assert_eq!(
        pack_cache_capacity::<f64>(PackSlot::B),
        0,
        "the oversized B buffer must be freed"
    );
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
    // The next, small blocked call still works and caches normally.
    check_case_blocked::<f64>(Trans::NoTrans, Trans::NoTrans, (70, 60, 50), 0.0, 43);
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

    // The outer GEMM runs through the public entry point, then with each pack
    // scope forced (the per-tile scope moves both packing buffers onto the
    // workers, which are the threads that also run the nested GEMMs).
    let outer_scopes = [
        None,
        Some(crate::level3::gemm_plan::PackScope::Shared),
        Some(crate::level3::gemm_plan::PackScope::PerTile),
    ];
    for outer_scope in outer_scopes {
        pool.install(|| run_nested_round(outer_scope, (&big_a, &big_b, &big_ref), &small));
    }
}

/// One round of [`test_nested_gemms_inside_parallel_gemm`]: the outer GEMM
/// (`scope`: through the public entry point when `None`, else through the
/// blocked route with that pack scope forced) joined with the nested ones.
#[cfg(feature = "parallel")]
fn run_nested_round(
    scope: Option<crate::level3::gemm_plan::PackScope>,
    (big_a, big_b, big_ref): (&Mat<f64>, &Mat<f64>, &Mat<f64>),
    small: &[(Mat<f64>, Mat<f64>, Mat<f64>)],
) {
    use rayon::prelude::*;

    let (bm, bk, bn) = (big_a.nrows(), big_a.ncols(), big_b.ncols());
    {
        let (big_c, smalls) = rayon::join(
            || {
                let mut c = Mat::filled(bm, bn, f64::NAN);
                match scope {
                    None => gemm_with_par(
                        1.0,
                        big_a.as_ref(),
                        big_b.as_ref(),
                        0.0,
                        c.as_mut(),
                        Par::Rayon,
                    ),
                    Some(scope) => gemm_planned(
                        Trans::NoTrans,
                        Trans::NoTrans,
                        1.0,
                        big_a.as_ref(),
                        big_b.as_ref(),
                        0.0,
                        c.as_mut(),
                        Par::Rayon,
                        BlockingSource::Default,
                        &GemmPlan::<f64>::for_problem(DEFAULT_POLICY, bm, bn, bk)
                            .with_forced_route(GemmRoute::Blocked)
                            .with_forced_scope(scope),
                    ),
                }
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
        assert_close(
            &big_c,
            big_ref,
            bk,
            &format!("outer parallel GEMM {scope:?}"),
        );
        for (i, (c, (_, b, r))) in smalls.iter().zip(small).enumerate() {
            assert_close(c, r, b.nrows(), &format!("nested GEMM {i} ({scope:?})"));
        }
    }
}

// ------------------------------------------------------------------
// Issue #23: the B-block cap changes NC only (never a result bit), and the
// packed B block of a thin-M product then stays in the per-thread cache.
// ------------------------------------------------------------------

/// Plans forcing the blocked route, without the B-block cap and with it in
/// `scope`.
fn b_cap_plans_in<T: Elem>(scope: BCapScope) -> (GemmPlan<T>, GemmPlan<T>) {
    use crate::level3::gemm_plan::LEGACY_POLICY;
    let uncapped = GemmPlan::<T>::new(LEGACY_POLICY).with_forced_route(GemmRoute::Blocked);
    let capped = GemmPlan::<T>::new(GemmPolicy {
        b_cap: scope,
        ..LEGACY_POLICY
    })
    .with_forced_route(GemmRoute::Blocked);
    (uncapped, capped)
}

/// [`b_cap_plans_in`] with the cap in the sequential driver.
fn b_cap_plans<T: Elem>() -> (GemmPlan<T>, GemmPlan<T>) {
    b_cap_plans_in(BCapScope::Sequential)
}

/// Exact bit patterns of the real element types.
trait Bits: Elem {
    fn bits(self) -> u64;
}

impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

fn bits_eq<T: Bits>(got: &Mat<T>, expected: &Mat<T>) -> bool {
    (0..got.ncols())
        .all(|j| (0..got.nrows()).all(|i| got[(i, j)].bits() == expected[(i, j)].bits()))
}

#[test]
fn test_issue_23_b_cap_result_bitwise_unchanged() {
    fn check<T: Bits>() {
        let shape = T::micro_kernel_shape();
        let elem = core::mem::size_of::<T>();
        // Single pass (all rows in one MC block) with two KC blocks and
        // more than 4 MiB of packed B; then two MC blocks (one of MR rows,
        // one of a single row), one KC block and more than 8 MiB of packed
        // B. The byte thresholds fix kc·n, so the work scales with m only:
        // both shapes use the fewest rows that keep their case.
        let kc = 256;
        let single = (
            (shape.mr, 2 * kc, (4 << 20) / (kc * elem) + 452),
            GemmBlocking {
                mc: 64,
                kc,
                nc: 1 << 20,
            },
        );
        let shallow = (
            (shape.mr + 1, 2 * kc, (8 << 20) / (2 * kc * elem) + 200),
            GemmBlocking {
                mc: shape.mr,
                kc: 2 * kc,
                nc: 1 << 20,
            },
        );
        for ((m, k, n), targets) in [single, shallow] {
            let ctx = format!("{} {m}x{k}x{n}", core::any::type_name::<T>());
            let fitted = targets.fit_to_dims(m, k, n, &shape);
            let a = fill::<T>(m, k, 71);
            let b = fill::<T>(k, n, 72);
            for beta in [T::zero(), T::sample(73)] {
                let c0 = if beta == T::zero() {
                    Mat::filled(m, n, T::nan())
                } else {
                    fill::<T>(m, n, 74)
                };
                let run = |plan: &GemmPlan<T>, par: Par| {
                    let mut c = c0.clone();
                    gemm_planned(
                        Trans::NoTrans,
                        Trans::NoTrans,
                        T::sample(75),
                        a.as_ref(),
                        b.as_ref(),
                        beta,
                        c.as_mut(),
                        par,
                        BlockingSource::Heuristic(targets),
                        plan,
                    );
                    c
                };
                // The uncapped sequential result, the same for every cap
                // scope.
                let (uncapped, _) = b_cap_plans_in::<T>(BCapScope::Sequential);
                let c_ref = run(&uncapped, Par::Seq);
                // `All` also hands the capped NC to the 2D parallel driver.
                for scope in [BCapScope::Sequential, BCapScope::All] {
                    let ctx = format!("{ctx} {scope:?} beta={beta}");
                    let (_, capped) = b_cap_plans_in::<T>(scope);
                    let blocks =
                        call_blocking(BlockingSource::Heuristic(targets), &capped, m, k, n);
                    let capped_blocks = *blocks.sequential();
                    assert!(capped_blocks.nc < fitted.nc, "{ctx}: the cap must engage");
                    assert_eq!((capped_blocks.mc, capped_blocks.kc), (fitted.mc, fitted.kc));
                    let c = run(&capped, Par::Seq);
                    assert!(bits_eq(&c, &c_ref), "{ctx} Seq: bits moved");
                    #[cfg(feature = "parallel")]
                    {
                        use crate::level3::gemm_parallel::scope_runs;
                        use crate::level3::gemm_plan::PackScope;
                        // The parallel NC is capped exactly under `All`.
                        let parallel_nc = blocks.parallel().nc;
                        if scope == BCapScope::All {
                            assert_eq!(parallel_nc, capped_blocks.nc, "{ctx}");
                        } else {
                            assert_eq!(parallel_nc, fitted.nc, "{ctx}");
                        }
                        // Both pack scopes of the 2D driver, which must run
                        // (counted per scope on this thread).
                        for pack in [PackScope::Shared, PackScope::PerTile] {
                            let plan = capped.with_forced_scope(pack);
                            let before = scope_runs();
                            let c = run(&plan, Par::RayonWith(4));
                            let after = scope_runs();
                            let ran = match pack {
                                PackScope::Shared => (after.0 - before.0, after.1 - before.1),
                                PackScope::PerTile => (after.1 - before.1, after.0 - before.0),
                            };
                            assert_eq!(ran, (1, 0), "{ctx} {pack:?}: 2D driver (this, other)");
                            assert!(bits_eq(&c, &c_ref), "{ctx} {pack:?}: bits moved");
                        }
                    }
                }
            }
        }
    }
    check::<f64>();
    check::<f32>();
}

#[test]
fn test_issue_23_thin_m_b_buffer_stays_cached() {
    // Thin-M (all rows in one MC block): the uncapped packed B block is
    // larger than the per-thread cache limit, so it was freed after every
    // call (0.2.3); with the cap NC shrinks to at most 2 MiB of packed B and
    // the buffer stays cached.
    let shape = f64::micro_kernel_shape();
    let kc = 256usize;
    let (m, k) = (16usize, kc);
    let n = PACK_CACHE_MAX_BYTES / (kc * 8) + 2 * shape.nr;
    let targets = GemmBlocking {
        mc: 64,
        kc,
        nc: 1 << 20,
    };
    let (uncapped, capped) = b_cap_plans::<f64>();
    let a = fill::<f64>(m, k, 81);
    let b = fill::<f64>(k, n, 82);
    let run_with = |plan: &GemmPlan<f64>, source: BlockingSource| {
        let mut c: Mat<f64> = Mat::filled(m, n, f64::NAN);
        gemm_planned(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
            Par::Seq,
            source,
            plan,
        );
        c
    };
    let run = |plan: &GemmPlan<f64>| run_with(plan, BlockingSource::Heuristic(targets));

    let c_uncapped = run(&uncapped);
    assert_eq!(
        pack_cache_capacity::<f64>(PackSlot::B),
        0,
        "the uncapped B block must exceed the cache limit (0.2.3 frees it)"
    );

    let c_capped = run(&capped);
    let cached = pack_cache_capacity::<f64>(PackSlot::B);
    assert!(cached > 0, "the capped B block must stay cached");
    assert!(cached * 8 <= 2 << 20, "capped B block of {cached} elements");
    assert!(pack_cache_bytes() <= PACK_CACHE_MAX_BYTES);
    assert!(bits_eq(&c_capped, &c_uncapped), "the cap moved result bits");

    // The next call reuses the cached buffer.
    let c_again = run(&capped);
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), cached);
    assert!(bits_eq(&c_again, &c_uncapped));

    // Explicit (caller-chosen) blocking is never capped, even under a
    // capping plan: its packed B block exceeds the cache limit again.
    let c_explicit = run_with(&capped, BlockingSource::Explicit(targets));
    assert_eq!(
        pack_cache_capacity::<f64>(PackSlot::B),
        0,
        "explicit blocking must keep the caller's NC"
    );
    assert!(bits_eq(&c_explicit, &c_uncapped));
}

#[test]
fn test_issue_23_b_cap_shallow_leaves_room_for_a() {
    // Shallow case (several MC blocks, two KC blocks): the sequential driver
    // returns its B buffer to the cache before its A buffer, so a B block
    // capped at the whole cache limit (issue #23's 8 MiB against the 8 MiB
    // cache) makes the cache free A after every call. The rule caps B at the
    // room left beside A instead. Scaled down by 16 so the test stays cheap:
    // a 512 KiB cache, a 512 KiB shallow cap and a 64 KiB A block.
    use crate::level3::autotune::{BBlockCaps, ISSUE_23_B_CAPS, capped_nc_with};
    use crate::level3::gemm_packing::set_pack_cache_limit;
    let limit = 512usize << 10;
    let caps = BBlockCaps {
        shallow_cap: limit,
        ..ISSUE_23_B_CAPS
    };
    let policy = GemmPolicy {
        b_cap: BCapScope::Sequential,
        b_caps: caps,
        ..crate::level3::gemm_plan::LEGACY_POLICY
    };
    let capped = GemmPlan::<f64>::new(policy).with_forced_route(GemmRoute::Blocked);
    let (uncapped, _) = b_cap_plans::<f64>();
    let shape = capped.kernel.shape;
    let (mc, kc) = (64usize, 128usize);
    assert_eq!(mc % shape.mr, 0);
    // n is two blocks of the plain cap's NC, so that cap would give a B block
    // of (almost) the whole cache.
    let plain_nc = limit / (kc * 8) / shape.nr * shape.nr;
    let (m, k, n) = (2 * mc, 2 * kc, 2 * plain_nc);
    let targets = GemmBlocking {
        mc,
        kc,
        nc: 1 << 20,
    };
    let a_bytes = mc * kc * 8;

    let _guard = set_pack_cache_limit(limit);
    let fitted = targets.fit_to_dims(m, k, n, &shape);
    assert_eq!((fitted.mc, fitted.kc, fitted.nc), (mc, kc, n));
    let blocks = *call_blocking(BlockingSource::Heuristic(targets), &capped, m, k, n).sequential();
    let rule = |cache_limit| capped_nc_with(m, k, n, mc, kc, n, shape.nr, 8, &caps, cache_limit);
    assert_eq!(
        blocks.nc,
        rule(limit),
        "call_blocking uses this thread's limit"
    );
    // The plain cap (room ignored, as with a twice larger cache) leaves no
    // room for A; the rule does.
    assert_eq!(rule(2 * limit), plain_nc);
    assert!(kc * plain_nc * 8 + a_bytes > limit);
    assert!(blocks.nc < plain_nc);
    assert!(kc * blocks.nc.div_ceil(shape.nr) * shape.nr * 8 + a_bytes <= limit);

    let a = fill::<f64>(m, k, 91);
    let b = fill::<f64>(k, n, 92);
    let run = |plan: &GemmPlan<f64>| {
        let mut c: Mat<f64> = Mat::filled(m, n, f64::NAN);
        gemm_planned(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c.as_mut(),
            Par::Seq,
            BlockingSource::Heuristic(targets),
            plan,
        );
        c
    };
    let c_uncapped = run(&uncapped);
    let c_capped = run(&capped);
    assert!(bits_eq(&c_capped, &c_uncapped), "the cap moved result bits");
    // Both buffers stay cached, within the limit.
    let cached_a = pack_cache_capacity::<f64>(PackSlot::A);
    let cached_b = pack_cache_capacity::<f64>(PackSlot::B);
    assert_eq!(cached_a, mc * kc, "the A buffer was freed");
    assert!(cached_b > 0, "the B buffer was freed");
    assert!(pack_cache_bytes() <= limit);
    // And the next call reuses both.
    let c_again = run(&capped);
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::A), cached_a);
    assert_eq!(pack_cache_capacity::<f64>(PackSlot::B), cached_b);
    assert!(bits_eq(&c_again, &c_uncapped));
}
