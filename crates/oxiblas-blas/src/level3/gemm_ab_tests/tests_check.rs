//! The comparisons of the correctness pass: the reference, the tolerance
//! and the bitwise check, with corrupted results at each route group's
//! largest `k`.

use super::super::trsm::Trans;
use super::check::{
    AbReal, Operands, Reference, SEED_A, SEED_B, check_bitwise, check_reference, initial_c,
    operand, reference, stored_shape, tolerance,
};
use super::specs::{Beta, TransPair};
use oxiblas_matrix::Mat;

/// `op(x)[i, j]` of the stored `x`.
fn op_at<T: AbReal>(x: &Mat<T>, trans: Trans, i: usize, j: usize) -> T {
    let (r, c) = stored_shape(trans, i, j);
    x[(r, c)]
}

/// The operands of `trans` `m × k` by `k × n`, their reference, and the
/// product in `T` by recursive summation in natural order (a correct, if
/// naive, kernel).
struct Fixture<T: AbReal> {
    trans: TransPair,
    dims: [usize; 3],
    a: Mat<T>,
    b: Mat<T>,
    reference: Reference,
    naive: Mat<T>,
}

impl<T: AbReal> Fixture<T> {
    fn new(trans: TransPair, [m, k, n]: [usize; 3]) -> Self {
        let (a, b) = (
            operand::<T>(trans.a, m, k, SEED_A),
            operand::<T>(trans.b, k, n, SEED_B),
        );
        let ops = Operands {
            trans_a: trans.a,
            trans_b: trans.b,
            a: &a,
            b: &b,
            c0: None,
        };
        let reference = reference(&ops, [m, k, n]);
        let mut naive = Mat::<T>::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                let mut acc = T::zero();
                for p in 0..k {
                    acc = acc + op_at(&a, trans.a, i, p) * op_at(&b, trans.b, p, j);
                }
                naive[(i, j)] = acc;
            }
        }
        Self {
            trans,
            dims: [m, k, n],
            a,
            b,
            reference,
            naive,
        }
    }

    /// The term `op(A)[i, p]·op(B)[p, j]` of largest magnitude of `(i, j)`.
    fn largest_term(&self, i: usize, j: usize) -> T {
        let k = self.dims[1];
        let term = |p| op_at(&self.a, self.trans.a, i, p) * op_at(&self.b, self.trans.b, p, j);
        let pick = |best: T, p| {
            if term(p).val().abs() > best.val().abs() {
                term(p)
            } else {
                best
            }
        };
        (0..k).fold(T::zero(), pick)
    }

    /// The corruptions every reference check must reject.
    fn corruptions(&self) -> Vec<(&'static str, Mat<T>)> {
        let [m, k, n] = self.dims;
        let u = T::UNIT_ROUNDOFF;
        let edit = |f: &dyn Fn(&mut Mat<T>)| {
            let mut c = self.naive.clone();
            f(&mut c);
            c
        };
        let big = self.largest_term(0, 0);
        let scale = T::of(1.0 + 64.0 * u * (k as f64).sqrt());
        let mut out = vec![
            ("one term dropped", edit(&|c| c[(0, 0)] = c[(0, 0)] - big)),
            (
                "one term counted twice",
                edit(&|c| c[(0, 0)] = c[(0, 0)] + big),
            ),
            ("all zero", Mat::<T>::zeros(m, n)),
            ("scaled by 1 + 64u√k", edit(&|c| scale_all(c, scale))),
        ];
        if n >= 2 {
            out.push((
                "one element from the next column",
                edit(&|c| c[(0, 0)] = c[(0, 1)]),
            ));
        }
        out
    }
}

/// Multiplies every element of `c` by `factor`.
fn scale_all<T: AbReal>(c: &mut Mat<T>, factor: T) {
    for j in 0..c.ncols() {
        for i in 0..c.nrows() {
            c[(i, j)] = c[(i, j)] * factor;
        }
    }
}

/// Checks that the naive product passes and every corruption fails.
fn check_fixture<T: AbReal>(trans: TransPair, dims: [usize; 3]) {
    let fx = Fixture::<T>::new(trans, dims);
    let at = format!("{}_{dims:?}_{}", trans.name(), core::any::type_name::<T>());
    let worst = check_reference(&fx.naive, &fx.reference);
    assert!(
        worst.as_ref().is_ok_and(|w| *w <= 0.5),
        "{at}: naive {worst:?}"
    );
    for (name, corrupt) in fx.corruptions() {
        let result = check_reference(&corrupt, &fx.reference);
        assert!(result.is_err(), "{at}: {name} passes ({result:?})");
    }
    // A relative error of 1e-3 (f32 only: f64 has far more room).
    if T::UNIT_ROUNDOFF > 1e-10 {
        let mut off = fx.naive.clone();
        scale_all(&mut off, T::of(1.0 + 1e-3));
        assert!(
            check_reference(&off, &fx.reference).is_err(),
            "{at}: 1e-3 passes"
        );
    }
}

#[test]
fn test_gemm_ab_check_rejects_corruption_thin() {
    // The thin groups' largest k (2^20), as a row of two elements.
    check_fixture::<f32>(TransPair::NN, [1, 1 << 20, 2]);
    check_fixture::<f64>(TransPair::NN, [1, 1 << 20, 2]);
}

#[test]
fn test_gemm_ab_check_rejects_corruption_small_simd() {
    // The small SIMD group's largest k: 8 × 512 × 8.
    check_fixture::<f32>(TransPair::NN, [8, 512, 8]);
    check_fixture::<f64>(TransPair::NN, [8, 512, 8]);
}

#[test]
fn test_gemm_ab_check_rejects_corruption_trans_copy() {
    // The small transposed copy group's largest k: 8 × 128 × 16 (TN, NT).
    for trans in [TransPair::TN, TransPair::NT] {
        check_fixture::<f32>(trans, [8, 128, 16]);
        check_fixture::<f64>(trans, [8, 128, 16]);
    }
}

#[test]
fn test_gemm_ab_reference_every_transpose_and_beta() {
    // The reference against a plain f64 sum (exact enough for k = 7) for
    // every transpose pair, with and without beta = 1.
    for trans in [TransPair::NN, TransPair::TN, TransPair::NT, TransPair::TT] {
        let [m, k, n] = [3, 7, 2];
        let (a, b) = (
            operand::<f32>(trans.a, m, k, 1),
            operand::<f32>(trans.b, k, n, 2),
        );
        let c0 = initial_c::<f32>(m, n, Beta::One);
        for with_c in [false, true] {
            let ops = Operands {
                trans_a: trans.a,
                trans_b: trans.b,
                a: &a,
                b: &b,
                c0: with_c.then_some(&c0),
            };
            let r = reference(&ops, [m, k, n]);
            assert_eq!((r.m, r.n, r.terms), (m, n, k + usize::from(with_c)));
            for j in 0..n {
                for i in 0..m {
                    let term = |p| op_at(&a, trans.a, i, p).val() * op_at(&b, trans.b, p, j).val();
                    let c = if with_c { c0[(i, j)].val() } else { 0.0 };
                    let want: f64 = (0..k).map(term).sum::<f64>() + c;
                    let abs: f64 = (0..k).map(|p| term(p).abs()).sum::<f64>() + c.abs();
                    let e = r.elems[i + j * m];
                    assert!(
                        (e.value - want).abs() <= 1e-14,
                        "{} ({i},{j})",
                        trans.name()
                    );
                    assert!((e.abs_sum - abs).abs() <= 1e-14 * abs);
                }
            }
        }
    }
    // beta = 0 starts from NaN (an element not written stays NaN).
    assert!(initial_c::<f64>(2, 2, Beta::Zero)[(1, 1)].is_nan());
}

#[test]
fn test_gemm_ab_reference_is_compensated() {
    // 1 + 2^-60 − 1 cancels in plain f64 but not in the compensated sum.
    let mut a = Mat::<f64>::zeros(1, 3);
    let mut b = Mat::<f64>::zeros(3, 1);
    for (p, (x, y)) in [(1.0, 1.0), (2f64.powi(-30), 2f64.powi(-30)), (-1.0, 1.0)]
        .iter()
        .enumerate()
    {
        a[(0, p)] = *x;
        b[(p, 0)] = *y;
    }
    let ops = Operands {
        trans_a: Trans::NoTrans,
        trans_b: Trans::NoTrans,
        a: &a,
        b: &b,
        c0: None,
    };
    let r = reference(&ops, [1, 3, 1]);
    assert_eq!(r.elems[0].value, 2f64.powi(-60));
}

#[test]
fn test_gemm_ab_tolerance_terms() {
    let u = f64::EPSILON / 2.0;
    let elem = |value: f64, abs_sum: f64, sq_sum: f64| super::check::RefElem {
        value,
        abs_sum,
        sq_sum,
    };
    // Few terms: the rigorous bound is the smaller one.
    let few = tolerance(&elem(1.0, 4.0, 4.0), 4, u);
    let gamma6 = 6.0 * u / (1.0 - 6.0 * u);
    assert!((few - (gamma6 * 4.0 + 2.0 * u)).abs() <= 1e-30, "{few:e}");
    // Many terms: the statistical bound, 8·u·√k·N2.
    let k = 1 << 20;
    let many = tolerance(&elem(0.0, k as f64 / 4.0, k as f64 / 9.0), k, u);
    let want = 8.0 * u * (k as f64).sqrt() * (k as f64 / 9.0).sqrt();
    assert!((many - want).abs() <= 1e-12 * want, "{many:e} {want:e}");
    // γ beyond 1/u is infinite, so the statistical bound applies.
    let huge = tolerance(
        &elem(0.0, 1.0, 1.0),
        usize::MAX / 2,
        f32::EPSILON as f64 / 2.0,
    );
    assert!(huge.is_finite());
}

#[test]
fn test_gemm_ab_compare_bitwise() {
    let base = operand::<f64>(Trans::NoTrans, 5, 4, 9);
    let copy = || operand::<f64>(Trans::NoTrans, 5, 4, 9);
    assert_eq!(check_bitwise(&copy(), &base), Ok(()));
    let mut ulp = copy();
    ulp[(3, 2)] = f64::from_bits(ulp[(3, 2)].to_bits() + 1);
    assert!(check_bitwise(&ulp, &base).is_err());
    let mut zero = copy();
    zero[(0, 0)] = 0.0;
    let mut neg_zero = copy();
    neg_zero[(0, 0)] = -0.0;
    assert!(
        check_bitwise(&zero, &neg_zero).is_err(),
        "signed zeros differ"
    );
    let mut f32_ulp = operand::<f32>(Trans::Trans, 3, 3, 4);
    let f32_base = f32_ulp.clone();
    f32_ulp[(1, 1)] = f32::from_bits(f32_ulp[(1, 1)].to_bits() + 1);
    assert!(check_bitwise(&f32_ulp, &f32_base).is_err());
    let other = operand::<f64>(Trans::NoTrans, 4, 5, 9);
    assert!(check_bitwise(&other, &base).is_err());
    // A NaN fails the reference check.
    let fx = Fixture::<f64>::new(TransPair::NN, [2, 3, 2]);
    let mut nan = fx.naive.clone();
    nan[(1, 1)] = f64::NAN;
    assert!(check_reference(&nan, &fx.reference).is_err());
    assert!(check_reference(&fx.naive, &fx.reference).is_ok());
}
