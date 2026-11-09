//! Tests of the blocked, triangle-only TRMM: every flag combination against
//! a dense oracle and the previous (textbook) implementation, alpha
//! handling, in-place ordering across blocks, unit diagonals with garbage,
//! special values and strided views.

use super::tri_test_util::*;
use super::types::{Diag, Side, Trans, Uplo};
use crate::level3::trmm::{
    TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo, trmm, trmm_in_place, trmm_left_naive, trmm_right_naive,
};
use num_complex::{Complex32, Complex64};
use oxiblas_matrix::Mat;

const ORDERS: [usize; 20] = [
    1, 2, 3, 5, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 193, 257, 300,
];

const ALPHAS: [f64; 4] = [0.0, 1.0, -1.0, 2.5];

fn tside(side: Side) -> TrmmSide {
    match side {
        Side::Left => TrmmSide::Left,
        Side::Right => TrmmSide::Right,
    }
}

fn tuplo(uplo: Uplo) -> TrmmUplo {
    match uplo {
        Uplo::Lower => TrmmUplo::Lower,
        Uplo::Upper => TrmmUplo::Upper,
    }
}

fn ttrans(trans: Trans) -> TrmmTrans {
    match trans {
        Trans::NoTrans => TrmmTrans::NoTrans,
        Trans::Trans => TrmmTrans::Trans,
        Trans::ConjTrans => TrmmTrans::ConjTrans,
    }
}

fn tdiag(diag: Diag) -> TrmmDiag {
    match diag {
        Diag::NonUnit => TrmmDiag::NonUnit,
        Diag::Unit => TrmmDiag::Unit,
    }
}

/// Runs the in-place TRMM of one configuration.
fn run<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    alpha: T,
    a: &Mat<T>,
    b: &Mat<T>,
) -> Mat<T> {
    let (side, uplo, trans, diag) = flags;
    let mut out = b.clone();
    trmm_in_place(
        tside(side),
        tuplo(uplo),
        ttrans(trans),
        tdiag(diag),
        alpha,
        a.as_ref(),
        out.as_mut(),
    )
    .expect("consistent shapes");
    out
}

/// The previous (textbook) implementation of one configuration.
fn run_previous<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    alpha: T,
    a: &Mat<T>,
    b: &Mat<T>,
) -> Mat<T> {
    let (side, uplo, trans, diag) = flags;
    let mut out = b.clone();
    let (m, n) = (b.nrows(), b.ncols());
    match side {
        Side::Left => trmm_left_naive(
            tuplo(uplo),
            ttrans(trans),
            tdiag(diag),
            alpha,
            a.as_ref(),
            out.as_mut(),
            m,
            n,
        ),
        Side::Right => trmm_right_naive(
            tuplo(uplo),
            ttrans(trans),
            tdiag(diag),
            alpha,
            a.as_ref(),
            out.as_mut(),
            m,
            n,
        ),
    }
    out
}

fn check_trmm<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    s: usize,
    nrhs: usize,
    alpha: T,
    rng: &mut Lcg,
) {
    let (side, uplo, _, _) = flags;
    let (m, n) = rhs_shape(side, s, nrhs);
    let nan = T::from_parts(f64::NAN, f64::NAN);
    let a = tri_fixture::<T>(s, uplo, rng, nan);
    let b = dense_fixture::<T>(m, n, rng);
    let out = run(flags, alpha, &a, &b);
    if alpha == T::zero() {
        for j in 0..n {
            for i in 0..m {
                assert!(
                    out[(i, j)] == T::zero(),
                    "{flags:?}: alpha == 0 must give 0"
                );
            }
        }
        return;
    }
    assert_trmm_result(flags, alpha, &a, &b, &out, "trmm");
    let old = run_previous(flags, alpha, &a, &b);
    let tol = T::TOL * 64.0 * (s as f64 + 4.0);
    for j in 0..n {
        for i in 0..m {
            let (got, want) = (out[(i, j)].to_c64(), old[(i, j)].to_c64());
            assert!(
                (got - want).norm() <= tol * (1.0 + want.norm()),
                "{flags:?} s={s}: new {got} vs previous {want} at ({i},{j})"
            );
        }
    }
}

fn sweep<T: TestScalar>(orders: &[usize], seed: u64) {
    let mut rng = Lcg(seed);
    for (oi, &s) in orders.iter().enumerate() {
        for (fi, flags) in all_flags().into_iter().enumerate() {
            let nrhs = if (oi + fi) % 5 == 0 {
                2 * 128 + 5
            } else {
                1 + (oi * 5 + fi) % 9
            };
            let alpha = T::from_parts(ALPHAS[(oi + fi) % ALPHAS.len()], 0.0);
            check_trmm::<T>(flags, s, nrhs, alpha, &mut rng);
        }
    }
}

#[test]
fn trmm_all_combinations_f64_orders_1_to_300() {
    sweep::<f64>(&ORDERS, 0x7244_0001);
}

#[test]
fn trmm_all_combinations_f32() {
    sweep::<f32>(&[1, 4, 9, 33, 64, 65, 130], 0x7244_0002);
}

#[test]
fn trmm_all_combinations_c64() {
    sweep::<Complex64>(&[1, 3, 8, 64, 65, 129], 0x7244_0003);
}

#[test]
fn trmm_all_combinations_c32() {
    sweep::<Complex32>(&[2, 7, 66, 131], 0x7244_0004);
}

/// Every alpha (including a genuinely complex one) for every combination.
#[test]
fn trmm_every_alpha_every_combination() {
    let mut rng = Lcg(0x7244_0005);
    for flags in all_flags() {
        for alpha in ALPHAS {
            check_trmm::<f64>(flags, 70, 11, alpha, &mut rng);
        }
        check_trmm::<Complex64>(flags, 70, 3, Complex64::new(0.5, -2.0), &mut rng);
    }
}

/// `trmm` (out of place) equals `trmm_in_place` on a copy.
#[test]
fn trmm_out_of_place_equals_in_place() {
    let mut rng = Lcg(0x7244_0006);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        let (m, n) = rhs_shape(side, 90, 7);
        let a = tri_fixture::<f64>(90, uplo, &mut rng, f64::NAN);
        let b = dense_fixture::<f64>(m, n, &mut rng);
        let x = trmm(
            tside(side),
            tuplo(uplo),
            ttrans(trans),
            tdiag(diag),
            -1.5,
            a.as_ref(),
            b.as_ref(),
        )
        .expect("ok");
        let y = run(flags, -1.5, &a, &b);
        assert_bitwise_eq(&x, &y, "trmm vs trmm_in_place");
    }
}

/// `Diag::Unit` never reads the diagonal or the other triangle.
#[test]
fn trmm_unit_diagonal_with_garbage_matches_explicit_ones() {
    let mut rng = Lcg(0x7244_0007);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        if diag == Diag::NonUnit {
            continue;
        }
        for &s in &[5usize, 64, 100] {
            let mut garbage = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
            let mut ones = garbage.clone();
            for k in 0..s {
                garbage[(k, k)] = f64::NAN;
                ones[(k, k)] = 1.0;
            }
            let (m, n) = rhs_shape(side, s, 6);
            let b = dense_fixture::<f64>(m, n, &mut rng);
            let x = run((side, uplo, trans, Diag::Unit), 2.0, &garbage, &b);
            let y = run((side, uplo, trans, Diag::NonUnit), 2.0, &ones, &b);
            for j in 0..n {
                for i in 0..m {
                    assert!(x[(i, j)].is_finite(), "{flags:?}: garbage leaked");
                    assert!(
                        (x[(i, j)] - y[(i, j)]).abs() <= 1e-13 * (1.0 + y[(i, j)].abs()),
                        "{flags:?}"
                    );
                }
            }
        }
    }
}

/// The multiply only touches the triangle: Inf/NaN in B or in the
/// referenced part of A reach exactly the positions the textbook
/// (triangle-only) implementation reaches.
#[test]
fn trmm_nan_inf_propagation_matches_textbook_implementation() {
    let mut rng = Lcg(0x7244_0008);
    for &s in &[6usize, 70, 140] {
        for flags in all_flags() {
            let (side, uplo, _, diag) = flags;
            let (m, n) = rhs_shape(side, s, 5);
            let mut a = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
            let mut b = dense_fixture::<f64>(m, n, &mut rng);
            for r in 0..5 {
                let g = (r * 41 + 3) % s;
                let v = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY][r % 3];
                match side {
                    Side::Left => b[(g, r)] = v,
                    Side::Right => b[(r, g)] = v,
                }
            }
            if s > 2 {
                let (i, j) = if uplo == Uplo::Lower {
                    (s - 1, 0)
                } else {
                    (0, s - 1)
                };
                a[(i, j)] = f64::INFINITY;
            }
            if diag == Diag::NonUnit {
                a[(s / 2, s / 2)] = f64::NEG_INFINITY;
            }
            let new = run(flags, 1.0, &a, &b);
            let old = run_previous(flags, 1.0, &a, &b);
            for j in 0..n {
                for i in 0..m {
                    let (x, y) = (new[(i, j)], old[(i, j)]);
                    assert_eq!(
                        (x.is_nan(), x.is_infinite(), x.is_infinite() && x > 0.0),
                        (y.is_nan(), y.is_infinite(), y.is_infinite() && y > 0.0),
                        "{flags:?} s={s}: class mismatch at ({i},{j}): new {x} vs textbook {y}"
                    );
                }
            }
        }
    }
}

/// A as a transposed view or strided submatrix, and B inside a larger buffer.
#[test]
fn trmm_strided_and_transposed_views() {
    let mut rng = Lcg(0x7244_0009);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        let s = 70;
        let (m, n) = rhs_shape(side, s, 9);
        let a = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
        let b = dense_fixture::<f64>(m, n, &mut rng);
        let reference = run(flags, 0.75, &a, &b);

        // Strided A.
        let mut big: Mat<f64> = Mat::filled(s + 6, s + 2, f64::NAN);
        for j in 0..s {
            for i in 0..s {
                big[(i + 2, j + 1)] = a[(i, j)];
            }
        }
        let mut x = b.clone();
        trmm_in_place(
            tside(side),
            tuplo(uplo),
            ttrans(trans),
            tdiag(diag),
            0.75,
            big.as_ref().submatrix(2, 1, s, s),
            x.as_mut(),
        )
        .expect("ok");
        assert_bitwise_eq(&x, &reference, "strided A");

        // Transposed view of the transposed storage (tolerance: GEMM's
        // small-matrix path depends on the operand strides).
        let mut at: Mat<f64> = Mat::zeros(s, s);
        for j in 0..s {
            for i in 0..s {
                at[(i, j)] = a[(j, i)];
            }
        }
        let mut y = b.clone();
        trmm_in_place(
            tside(side),
            tuplo(uplo),
            ttrans(trans),
            tdiag(diag),
            0.75,
            at.as_ref().transpose(),
            y.as_mut(),
        )
        .expect("ok");
        for j in 0..n {
            for i in 0..m {
                assert!(
                    (y[(i, j)] - reference[(i, j)]).abs()
                        <= 1e-12 * (1.0 + reference[(i, j)].abs()),
                    "{flags:?}: transposed view"
                );
            }
        }

        // B inside a larger buffer.
        let mut bb: Mat<f64> = Mat::filled(m + 5, n + 2, 3.5);
        for j in 0..n {
            for i in 0..m {
                bb[(i + 3, j + 1)] = b[(i, j)];
            }
        }
        trmm_in_place(
            tside(side),
            tuplo(uplo),
            ttrans(trans),
            tdiag(diag),
            0.75,
            a.as_ref(),
            bb.as_mut().submatrix(3, 1, m, n),
        )
        .expect("ok");
        for j in 0..n + 2 {
            for i in 0..m + 5 {
                if (3..3 + m).contains(&i) && (1..1 + n).contains(&j) {
                    assert_eq!(bb[(i, j)].to_bits(), reference[(i - 3, j - 1)].to_bits());
                } else {
                    assert_eq!(bb[(i, j)], 3.5, "{flags:?}: write outside the B view");
                }
            }
        }
    }
}

/// Complex `alpha == 1` is not applied as a multiply: an infinite entry of
/// B that only meets the unit diagonal stays exactly infinite.
#[test]
fn trmm_complex_alpha_one_and_unit_diagonal_keep_infinities() {
    // Upper unit triangle with NaN on the (unread) diagonal and below it.
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let inf = Complex64::new(f64::INFINITY, 0.0);
    let mut a2: Mat<Complex64> = Mat::filled(3, 3, nan);
    a2[(0, 1)] = Complex64::new(0.5, 0.0);
    a2[(0, 2)] = Complex64::new(0.25, 0.0);
    a2[(1, 2)] = Complex64::new(2.0, 0.0);
    let mut b2: Mat<Complex64> = Mat::zeros(3, 1);
    b2[(2, 0)] = Complex64::new(1.0, 0.0);
    b2[(0, 0)] = inf;
    trmm_in_place(
        TrmmSide::Left,
        TrmmUplo::Upper,
        TrmmTrans::NoTrans,
        TrmmDiag::Unit,
        Complex64::new(1.0, 0.0),
        a2.as_ref(),
        b2.as_mut(),
    )
    .expect("ok");
    assert_eq!(b2[(0, 0)].re, f64::INFINITY);
    assert_eq!(b2[(0, 0)].im, 0.0);
    assert_eq!(b2[(1, 0)], Complex64::new(2.0, 0.0));
    assert_eq!(b2[(2, 0)], Complex64::new(1.0, 0.0));
}
