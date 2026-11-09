//! Tests of the blocked TRSM driver: every flag combination against
//! residual and previous-implementation oracles, special values, singular
//! factors, unit diagonals with garbage, strided/transposed views and
//! parallel-versus-sequential bitwise equality.

use super::generic::trsm_naive;
use super::tri_test_util::*;
use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use super::{trsm, trsm_c32, trsm_c64, trsm_c64_in_place, trsm_in_place, trsm_in_place_with_par};
use num_complex::{Complex32, Complex64};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

/// Orders crossing the 128-row diagonal blocks, the 4/8-row tiles and the
/// 128-wide right-hand-side panels.
const ORDERS: [usize; 27] = [
    1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 33, 63, 64, 65, 96, 127, 128, 129, 191, 192, 193, 255,
    256, 257, 300,
];

const ALPHAS: [f64; 4] = [0.0, 1.0, -1.0, 2.5];

/// Solves one configuration with `trsm` and checks it against the residual
/// oracle and against the previous (textbook) implementation.
fn check_trsm<T: TestScalar>(
    flags: (Side, Uplo, Trans, Diag),
    s: usize,
    nrhs: usize,
    alpha: T,
    rng: &mut Lcg,
) {
    let (side, uplo, trans, diag) = flags;
    let (m, n) = rhs_shape(side, s, nrhs);
    let nan = T::from_parts(f64::NAN, f64::NAN);
    let a = tri_fixture::<T>(s, uplo, rng, nan);
    let b = dense_fixture::<T>(m, n, rng);
    let x = trsm(side, uplo, trans, diag, alpha, a.as_ref(), b.as_ref())
        .expect("fixture diagonal is non-zero");
    if alpha == T::zero() {
        for j in 0..n {
            for i in 0..m {
                assert!(x[(i, j)] == T::zero(), "{flags:?}: alpha == 0 must give 0");
            }
        }
        return;
    }
    assert_trsm_residual(flags, alpha, &a, &b, &x, "trsm");

    // Previous implementation on the prescaled right-hand side.
    let mut old = scaled(alpha, &b);
    trsm_naive(side, uplo, trans, diag, a.as_ref(), old.as_mut(), m, n)
        .expect("fixture diagonal is non-zero");
    let tol = T::TOL * 64.0 * (s as f64 + 4.0);
    for j in 0..n {
        for i in 0..m {
            let (got, want) = (x[(i, j)].to_c64(), old[(i, j)].to_c64());
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
            // Rectangular B: the number of right-hand sides varies with the
            // order and combination; every 5th case spans several panels.
            let nrhs = if (oi + fi) % 5 == 0 {
                2 * 128 + 3
            } else {
                1 + (oi * 3 + fi) % 9
            };
            let alpha = T::from_parts(ALPHAS[(oi + fi) % ALPHAS.len()], 0.0);
            check_trsm::<T>(flags, s, nrhs, alpha, &mut rng);
        }
    }
}

#[test]
fn trsm_all_combinations_f64_orders_1_to_300() {
    sweep::<f64>(&ORDERS, 0x7253_0001);
}

#[test]
fn trsm_all_combinations_f32() {
    sweep::<f32>(&[1, 5, 8, 17, 63, 64, 65, 130, 200], 0x7253_0002);
}

#[test]
fn trsm_all_combinations_c64() {
    sweep::<Complex64>(&[1, 3, 4, 9, 64, 65, 129], 0x7253_0003);
}

#[test]
fn trsm_all_combinations_c32() {
    sweep::<Complex32>(&[2, 5, 64, 70, 130], 0x7253_0004);
}

/// Every alpha for every combination at one block-crossing order.
#[test]
fn trsm_every_alpha_every_combination() {
    let mut rng = Lcg(0x7253_0005);
    for flags in all_flags() {
        for alpha in ALPHAS {
            check_trsm::<f64>(flags, 70, 11, alpha, &mut rng);
        }
    }
}

/// `trsm(alpha, B)` is exactly `trsm_in_place` on the prescaled `alpha·B`.
#[test]
fn trsm_out_of_place_equals_in_place_on_prescaled_rhs() {
    let mut rng = Lcg(0x7253_0006);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        let (m, n) = rhs_shape(side, 77, 13);
        let a = tri_fixture::<f64>(77, uplo, &mut rng, 0.0);
        let b = dense_fixture::<f64>(m, n, &mut rng);
        let x = trsm(side, uplo, trans, diag, -2.5, a.as_ref(), b.as_ref()).expect("non-singular");
        let mut y = scaled(-2.5, &b);
        trsm_in_place(side, uplo, trans, diag, a.as_ref(), y.as_mut()).expect("non-singular");
        assert_bitwise_eq(&x, &y, "trsm vs in-place");
    }
}

/// The complex entry points and the generic one run the same code.
#[test]
fn trsm_c64_entry_points_match_generic() {
    let mut rng = Lcg(0x7253_0007);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        let (m, n) = rhs_shape(side, 70, 5);
        let a = tri_fixture::<Complex64>(70, uplo, &mut rng, Complex64::new(0.0, 0.0));
        let b = dense_fixture::<Complex64>(m, n, &mut rng);
        let alpha = Complex64::new(0.5, -1.5);
        let x = trsm_c64(side, uplo, trans, diag, alpha, a.as_ref(), b.as_ref()).expect("ok");
        let y = trsm(side, uplo, trans, diag, alpha, a.as_ref(), b.as_ref()).expect("ok");
        assert_bitwise_eq(&x, &y, "trsm_c64 vs trsm");
        let mut z = scaled(alpha, &b);
        trsm_c64_in_place(side, uplo, trans, diag, a.as_ref(), z.as_mut()).expect("ok");
        assert_bitwise_eq(&x, &z, "trsm_c64 vs trsm_c64_in_place");
        assert_trsm_residual(flags, alpha, &a, &b, &x, "trsm_c64");
    }
    // c32 through the generic entry point as well.
    let a = tri_fixture::<Complex32>(9, Uplo::Upper, &mut rng, Complex32::new(0.0, 0.0));
    let b = dense_fixture::<Complex32>(9, 4, &mut rng);
    let one = Complex32::new(1.0, 0.0);
    let flags = (Side::Left, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit);
    let x = trsm_c32(
        flags.0,
        flags.1,
        flags.2,
        flags.3,
        one,
        a.as_ref(),
        b.as_ref(),
    )
    .expect("ok");
    let y = trsm(
        flags.0,
        flags.1,
        flags.2,
        flags.3,
        one,
        a.as_ref(),
        b.as_ref(),
    )
    .expect("ok");
    assert_bitwise_eq(&x, &y, "trsm_c32 vs trsm");
}

/// NaN/Inf in B and in the referenced part of A propagate to exactly the
/// same positions (NaN vs ±Inf vs finite) as with the previous
/// implementation.
#[test]
fn trsm_nan_inf_propagation_matches_previous_implementation() {
    let mut rng = Lcg(0x7253_0008);
    for &s in &[7usize, 70, 150] {
        for flags in all_flags() {
            let (side, uplo, trans, diag) = flags;
            let (m, n) = rhs_shape(side, s, 6);
            let mut a = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
            let mut b = dense_fixture::<f64>(m, n, &mut rng);
            // Special values in B (one per right-hand side, varied rows) ...
            for r in 0..6 {
                let g = (r * 37 + 5) % s;
                let v = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY][r % 3];
                match side {
                    Side::Left => b[(g, r)] = v,
                    Side::Right => b[(r, g)] = v,
                }
            }
            // ... an infinite off-diagonal entry in the referenced triangle ...
            if s > 2 {
                let (i, j) = (s - 1, 1);
                let (i, j) = if uplo == Uplo::Lower { (i, j) } else { (j, i) };
                a[(i, j)] = f64::INFINITY;
            }
            // ... and an infinite diagonal entry (its reciprocal is 0).
            if diag == Diag::NonUnit && s > 3 {
                a[(s / 2, s / 2)] = f64::NEG_INFINITY;
            }
            let mut new = b.clone();
            trsm_in_place(side, uplo, trans, diag, a.as_ref(), new.as_mut()).expect("ok");
            let mut old = b.clone();
            trsm_naive(side, uplo, trans, diag, a.as_ref(), old.as_mut(), m, n).expect("ok");
            for j in 0..n {
                for i in 0..m {
                    let (x, y) = (new[(i, j)], old[(i, j)]);
                    assert_eq!(
                        (x.is_nan(), x.is_infinite(), x.is_infinite() && x > 0.0),
                        (y.is_nan(), y.is_infinite(), y.is_infinite() && y > 0.0),
                        "{flags:?} s={s}: class mismatch at ({i},{j}): new {x} vs previous {y}"
                    );
                }
            }
        }
    }
}

/// A subnormal diagonal entry has no finite reciprocal; the solver then
/// divides exactly and still returns the finite solution.
#[test]
fn trsm_subnormal_diagonal_uses_exact_division() {
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        if diag == Diag::Unit {
            continue;
        }
        for &s in &[1usize, 5, 70] {
            let mut rng = Lcg(0x7253_0009 + s as u64);
            let mut a = tri_fixture::<f64>(s, uplo, &mut rng, 0.0);
            let k = s / 2;
            a[(k, k)] = 4.0e-310; // 1 / 4e-310 overflows to +Inf
            let (m, n) = rhs_shape(side, s, 3);
            // RHS small enough that the quotient is representable.
            let mut b: Mat<f64> = Mat::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    b[(i, j)] = 1.0e-300 * (1.0 + rng.unit());
                }
            }
            let mut x = b.clone();
            trsm_in_place(side, uplo, trans, diag, a.as_ref(), x.as_mut()).expect("non-singular");
            let mut old = b.clone();
            trsm_naive(side, uplo, trans, diag, a.as_ref(), old.as_mut(), m, n).expect("ok");
            for j in 0..n {
                for i in 0..m {
                    assert_eq!(
                        x[(i, j)].is_finite(),
                        old[(i, j)].is_finite(),
                        "{flags:?} s={s}"
                    );
                    if old[(i, j)].is_finite() {
                        let rel = (x[(i, j)] - old[(i, j)]).abs() / old[(i, j)].abs().max(1e-300);
                        assert!(
                            rel < 1e-10,
                            "{flags:?} s={s}: {} vs {}",
                            x[(i, j)],
                            old[(i, j)]
                        );
                    }
                }
            }
        }
    }
}

/// A zero on a non-unit diagonal is `Singular` for every combination and
/// position, B is left untouched, and the unit-diagonal solve ignores it.
#[test]
fn trsm_singular_factor_reported_and_b_untouched() {
    let mut rng = Lcg(0x7253_000A);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        for &s in &[1usize, 5, 64, 130] {
            for &k in &[0, s / 2, s - 1] {
                let mut a = tri_fixture::<f64>(s, uplo, &mut rng, 0.0);
                a[(k, k)] = if k % 2 == 0 { 0.0 } else { -0.0 };
                let (m, n) = rhs_shape(side, s, 4);
                let b = dense_fixture::<f64>(m, n, &mut rng);
                let mut x = b.clone();
                let r = trsm_in_place(side, uplo, trans, diag, a.as_ref(), x.as_mut());
                if diag == Diag::Unit {
                    assert!(r.is_ok(), "{flags:?}: unit diagonal never reads A[k,k]");
                } else {
                    assert_eq!(r, Err(TrsmError::Singular), "{flags:?} s={s} k={k}");
                    assert_bitwise_eq(&x, &b, "B must be untouched on error");
                    let y = trsm(side, uplo, trans, diag, 2.0, a.as_ref(), b.as_ref());
                    assert!(matches!(y, Err(TrsmError::Singular)));
                    // alpha == 0 never reads A; an empty B never solves.
                    assert!(trsm(side, uplo, trans, diag, 0.0, a.as_ref(), b.as_ref()).is_ok());
                    let (em, en) = rhs_shape(side, s, 0);
                    let mut empty: Mat<f64> = Mat::zeros(em, en);
                    assert!(
                        trsm_in_place(side, uplo, trans, diag, a.as_ref(), empty.as_mut()).is_ok()
                    );
                }
            }
        }
    }
}

/// `Diag::Unit` never reads the diagonal (nor the other triangle): with NaN
/// there it gives bitwise the result of a non-unit solve with ones on the
/// diagonal.
#[test]
fn trsm_unit_diagonal_with_garbage_matches_explicit_ones() {
    let mut rng = Lcg(0x7253_000B);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        if diag == Diag::NonUnit {
            continue;
        }
        for &s in &[3usize, 64, 100] {
            let mut garbage = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
            let mut ones = garbage.clone();
            for k in 0..s {
                garbage[(k, k)] = f64::NAN;
                ones[(k, k)] = 1.0;
            }
            let (m, n) = rhs_shape(side, s, 9);
            let b = dense_fixture::<f64>(m, n, &mut rng);
            let mut x = b.clone();
            trsm_in_place(side, uplo, trans, Diag::Unit, garbage.as_ref(), x.as_mut()).expect("ok");
            let mut y = b.clone();
            trsm_in_place(side, uplo, trans, Diag::NonUnit, ones.as_ref(), y.as_mut()).expect("ok");
            assert_bitwise_eq(&x, &y, "unit with garbage vs non-unit with ones");
            for j in 0..n {
                for i in 0..m {
                    assert!(x[(i, j)].is_finite(), "{flags:?}: garbage leaked");
                }
            }
        }
    }
    // Complex: same property.
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let mut garbage = tri_fixture::<Complex64>(70, Uplo::Lower, &mut rng, nan);
    let mut ones = garbage.clone();
    for k in 0..70 {
        garbage[(k, k)] = nan;
        ones[(k, k)] = Complex64::new(1.0, 0.0);
    }
    let b = dense_fixture::<Complex64>(70, 5, &mut rng);
    for trans in TRANSES {
        let mut x = b.clone();
        trsm_in_place(
            Side::Left,
            Uplo::Lower,
            trans,
            Diag::Unit,
            garbage.as_ref(),
            x.as_mut(),
        )
        .expect("ok");
        let mut y = b.clone();
        trsm_in_place(
            Side::Left,
            Uplo::Lower,
            trans,
            Diag::NonUnit,
            ones.as_ref(),
            y.as_mut(),
        )
        .expect("ok");
        for j in 0..5 {
            for i in 0..70 {
                assert!(
                    x[(i, j)].re.is_finite() && x[(i, j)].im.is_finite(),
                    "{trans:?}"
                );
                assert!((x[(i, j)] - y[(i, j)]).norm() <= 1e-12, "{trans:?}");
            }
        }
    }
}

/// A as a transposed view or a strided submatrix, and B as a submatrix of a
/// larger buffer (ldb > m): same results as compact operands, and the rest of
/// the B buffer is untouched.
#[test]
fn trsm_strided_and_transposed_views() {
    let mut rng = Lcg(0x7253_000C);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        let s = 75;
        let (m, n) = rhs_shape(side, s, 10);
        let a = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
        let b = dense_fixture::<f64>(m, n, &mut rng);
        let mut reference = b.clone();
        trsm_in_place(side, uplo, trans, diag, a.as_ref(), reference.as_mut()).expect("ok");

        // A stored transposed: solving with the flipped triangle and the
        // toggled transpose flag is the same problem.
        let mut at: Mat<f64> = Mat::zeros(s, s);
        for j in 0..s {
            for i in 0..s {
                at[(i, j)] = a[(j, i)];
            }
        }
        let flipped = if uplo == Uplo::Lower {
            Uplo::Upper
        } else {
            Uplo::Lower
        };
        if trans != Trans::ConjTrans {
            let toggled = if trans == Trans::NoTrans {
                Trans::Trans
            } else {
                Trans::NoTrans
            };
            let mut x = b.clone();
            trsm_in_place(side, flipped, toggled, diag, at.as_ref(), x.as_mut()).expect("ok");
            let tol = 1e-12;
            for j in 0..n {
                for i in 0..m {
                    assert!(
                        (x[(i, j)] - reference[(i, j)]).abs()
                            <= tol * (1.0 + reference[(i, j)].abs()),
                        "{flags:?}: stored transpose"
                    );
                }
            }
            // The same through a zero-copy transposed view of `at`. (Not
            // bitwise: GEMM's small-matrix path picks a different kernel for
            // operands whose rows are not adjacent in memory.)
            let mut y = b.clone();
            trsm_in_place(side, uplo, trans, diag, at.as_ref().transpose(), y.as_mut())
                .expect("ok");
            for j in 0..n {
                for i in 0..m {
                    assert!(
                        (y[(i, j)] - reference[(i, j)]).abs()
                            <= tol * (1.0 + reference[(i, j)].abs()),
                        "{flags:?}: transposed view of A"
                    );
                }
            }
        }

        // A as a strided submatrix of a larger matrix.
        let mut big: Mat<f64> = Mat::filled(s + 9, s + 4, f64::NAN);
        for j in 0..s {
            for i in 0..s {
                big[(i + 5, j + 3)] = a[(i, j)];
            }
        }
        let mut x = b.clone();
        trsm_in_place(
            side,
            uplo,
            trans,
            diag,
            big.as_ref().submatrix(5, 3, s, s),
            x.as_mut(),
        )
        .expect("ok");
        assert_bitwise_eq(&x, &reference, "strided A");

        // B as a submatrix of a larger buffer.
        let mut bb: Mat<f64> = Mat::filled(m + 7, n + 3, -7.25);
        for j in 0..n {
            for i in 0..m {
                bb[(i + 4, j + 1)] = b[(i, j)];
            }
        }
        trsm_in_place(
            side,
            uplo,
            trans,
            diag,
            a.as_ref(),
            bb.as_mut().submatrix(4, 1, m, n),
        )
        .expect("ok");
        for j in 0..n + 3 {
            for i in 0..m + 7 {
                let inside = (4..4 + m).contains(&i) && (1..1 + n).contains(&j);
                if inside {
                    assert_eq!(bb[(i, j)].to_bits(), reference[(i - 4, j - 1)].to_bits());
                } else {
                    assert_eq!(bb[(i, j)], -7.25, "{flags:?}: write outside the B view");
                }
            }
        }
    }
}

/// Parallel solves are bitwise identical to sequential ones (several full
/// panels plus a ragged one), on an explicit 4-thread pool.
#[cfg(feature = "parallel")]
#[test]
fn trsm_parallel_is_bitwise_identical_to_sequential() {
    use core::sync::atomic::Ordering;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("thread pool");
    let runs_before = super::tri_driver::PARALLEL_RUNS.load(Ordering::Relaxed);
    let mut rng = Lcg(0x7253_000D);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        // Two diagonal blocks, so every panel also runs an in-place GEMM update.
        let s = 136;
        let nrhs = 3 * super::tri_driver::RHS_PANEL + 37;
        let (m, n) = rhs_shape(side, s, nrhs);
        let a = tri_fixture::<f64>(s, uplo, &mut rng, f64::NAN);
        let b = dense_fixture::<f64>(m, n, &mut rng);
        let mut seq = b.clone();
        trsm_in_place(side, uplo, trans, diag, a.as_ref(), seq.as_mut()).expect("ok");
        let mut par = b.clone();
        pool.install(|| {
            trsm_in_place_with_par(
                side,
                uplo,
                trans,
                diag,
                a.as_ref(),
                par.as_mut(),
                Par::Rayon,
            )
        })
        .expect("ok");
        assert_bitwise_eq(&par, &seq, "Par::Rayon vs sequential");
        let mut par4 = b.clone();
        pool.install(|| {
            trsm_in_place_with_par(
                side,
                uplo,
                trans,
                diag,
                a.as_ref(),
                par4.as_mut(),
                Par::RayonWith(4),
            )
        })
        .expect("ok");
        assert_bitwise_eq(&par4, &seq, "Par::RayonWith(4) vs sequential");
        let out = pool
            .install(|| {
                super::trsm_with_par(
                    side,
                    uplo,
                    trans,
                    diag,
                    1.5,
                    a.as_ref(),
                    b.as_ref(),
                    Par::Rayon,
                )
            })
            .expect("ok");
        let reference = trsm(side, uplo, trans, diag, 1.5, a.as_ref(), b.as_ref()).expect("ok");
        assert_bitwise_eq(&out, &reference, "trsm_with_par vs trsm");
    }
    // Complex as well (s² · nrhs above the parallel work gate, several
    // panels plus a ragged one, both sides).
    let nrhs = 3 * super::tri_driver::RHS_PANEL + 37;
    let a = tri_fixture::<Complex64>(136, Uplo::Upper, &mut rng, Complex64::new(f64::NAN, 0.0));
    let b = dense_fixture::<Complex64>(136, nrhs, &mut rng);
    let br = dense_fixture::<Complex64>(nrhs, 136, &mut rng);
    for trans in TRANSES {
        let mut seq = br.clone();
        trsm_in_place(
            Side::Right,
            Uplo::Upper,
            trans,
            Diag::NonUnit,
            a.as_ref(),
            seq.as_mut(),
        )
        .expect("ok");
        let mut par = br.clone();
        pool.install(|| {
            trsm_in_place_with_par(
                Side::Right,
                Uplo::Upper,
                trans,
                Diag::NonUnit,
                a.as_ref(),
                par.as_mut(),
                Par::Rayon,
            )
        })
        .expect("ok");
        assert_bitwise_eq(&par, &seq, "complex Right parallel vs sequential");

        let mut seq = b.clone();
        trsm_in_place(
            Side::Left,
            Uplo::Upper,
            trans,
            Diag::NonUnit,
            a.as_ref(),
            seq.as_mut(),
        )
        .expect("ok");
        let mut par = b.clone();
        pool.install(|| {
            trsm_in_place_with_par(
                Side::Left,
                Uplo::Upper,
                trans,
                Diag::NonUnit,
                a.as_ref(),
                par.as_mut(),
                Par::Rayon,
            )
        })
        .expect("ok");
        assert_bitwise_eq(&par, &seq, "complex parallel vs sequential");
    }
    // Many right-hand sides: wide (256) panels, still several plus a ragged one.
    let nrhs = 4 * super::tri_driver::RHS_PANEL_WIDE + 37;
    assert_eq!(
        super::tri_driver::rhs_panel(nrhs),
        super::tri_driver::RHS_PANEL_WIDE
    );
    for flags in [
        (Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit),
        (Side::Left, Uplo::Upper, Trans::Trans, Diag::Unit),
        (Side::Right, Uplo::Lower, Trans::Trans, Diag::NonUnit),
        (Side::Right, Uplo::Upper, Trans::NoTrans, Diag::Unit),
    ] {
        let (side, uplo, trans, diag) = flags;
        let (m, n) = rhs_shape(side, 136, nrhs);
        let a = tri_fixture::<f64>(136, uplo, &mut rng, f64::NAN);
        let b = dense_fixture::<f64>(m, n, &mut rng);
        let mut seq = b.clone();
        trsm_in_place(side, uplo, trans, diag, a.as_ref(), seq.as_mut()).expect("ok");
        let mut par = b.clone();
        pool.install(|| {
            trsm_in_place_with_par(
                side,
                uplo,
                trans,
                diag,
                a.as_ref(),
                par.as_mut(),
                Par::Rayon,
            )
        })
        .expect("ok");
        assert_bitwise_eq(&par, &seq, "wide panels: Par::Rayon vs sequential");
        if side == Side::Left && diag == Diag::NonUnit {
            assert_trsm_residual(flags, 1.0, &a, &b, &seq, "wide panels");
        }
    }
    // 24 real combinations x 3 parallel calls + 3 x 2 complex calls + 4 wide.
    let runs = super::tri_driver::PARALLEL_RUNS.load(Ordering::Relaxed) - runs_before;
    assert!(
        runs >= 24 * 3 + 6 + 4,
        "only {runs} calls took the parallel path"
    );
}

/// Errors are reported identically by the parallel entry point.
#[test]
fn trsm_with_par_validates_and_reports_singular() {
    let a = Mat::from_rows(&[&[1.0f64, 0.0], &[2.0, 0.0]]);
    let mut b: Mat<f64> = Mat::filled(2, 300, 1.0);
    let r = trsm_in_place_with_par(
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        a.as_ref(),
        b.as_mut(),
        Par::default(),
    );
    assert_eq!(r, Err(TrsmError::Singular));
    let mut wrong: Mat<f64> = Mat::zeros(3, 3);
    let r = trsm_in_place_with_par(
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        a.as_ref(),
        wrong.as_mut(),
        Par::Seq,
    );
    assert_eq!(r, Err(TrsmError::DimensionMismatch));
}

/// Complex special values: non-finite results appear at exactly the
/// positions the previous (textbook) solver produces them. Whether such an
/// entry is NaN or ±Inf can differ: the kernels use fused complex
/// multiply-adds and multiply by the reciprocal of the diagonal, and the
/// off-diagonal updates go through the 3M GEMM (with `beta = 1 + 0i`), all of
/// which combine infinite components in a different order than the textbook
/// `sum -= a·x; x = sum / d`.
#[test]
fn trsm_complex_nan_inf_reach_the_same_positions_as_previous_implementation() {
    let mut rng = Lcg(0x7253_00EE);
    for &s in &[7usize, 70, 150] {
        for flags in all_flags() {
            let (side, uplo, trans, diag) = flags;
            let (m, n) = rhs_shape(side, s, 6);
            let nan = Complex64::new(f64::NAN, f64::NAN);
            let a = tri_fixture::<Complex64>(s, uplo, &mut rng, nan);
            let mut b = dense_fixture::<Complex64>(m, n, &mut rng);
            for r in 0..6 {
                let g = (r * 37 + 5) % s;
                let v = [
                    Complex64::new(f64::INFINITY, 0.0),
                    Complex64::new(f64::NAN, 1.0),
                    Complex64::new(0.5, f64::NEG_INFINITY),
                ][r % 3];
                match side {
                    Side::Left => b[(g, r)] = v,
                    Side::Right => b[(r, g)] = v,
                }
            }
            let mut new = b.clone();
            trsm_in_place(side, uplo, trans, diag, a.as_ref(), new.as_mut()).expect("ok");
            let mut old = b.clone();
            trsm_naive(side, uplo, trans, diag, a.as_ref(), old.as_mut(), m, n).expect("ok");
            for j in 0..n {
                for i in 0..m {
                    let (x, y) = (new[(i, j)], old[(i, j)]);
                    let finite = |z: Complex64| z.re.is_finite() && z.im.is_finite();
                    assert_eq!(
                        finite(x),
                        finite(y),
                        "{flags:?} s={s}: ({i},{j}): new {x} vs previous {y}"
                    );
                    if finite(y) {
                        assert!((x - y).norm() <= 1e-9 * (1.0 + y.norm()), "{flags:?} s={s}");
                    }
                }
            }
        }
    }
}

// =============================================================================
// Diagonals whose reciprocal is unusable (exact-division mode)
// =============================================================================

/// `B = op(T)·X` (Left) or `X·op(T)` (Right) for the triangular `a`,
/// evaluated in `Complex64` and rounded to `T`.
fn rhs_of<T: TestScalar>(flags: (Side, Uplo, Trans, Diag), a: &Mat<T>, x: &Mat<T>) -> Mat<T> {
    let (prod, _) = tri_product(flags, a, x);
    let (m, n) = (x.nrows(), x.ncols());
    let mut b: Mat<T> = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let z = prod[i + j * m];
            b[(i, j)] = T::from_parts(z.re, z.im);
        }
    }
    b
}

/// `a` with its `uplo` triangle (diagonal included) multiplied by the power
/// of two `scale`, which is exact; the other triangle keeps its junk.
fn scaled_triangle<T: TestScalar>(a: &Mat<T>, uplo: Uplo, scale: f64) -> Mat<T> {
    let s = T::from_parts(scale, 0.0);
    let mut out = a.clone();
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if i == j || (uplo == Uplo::Lower) == (i > j) {
                out[(i, j)] = s * a[(i, j)];
            }
        }
    }
    out
}

/// Scaling the triangle and the right-hand sides by the same power of two
/// leaves the solution unchanged. With a diagonal of modulus `scale·[1, 1.5]`
/// the reciprocals are unusable for the extreme scales below, so the solver
/// divides; it must still return the solution of the unscaled system (which
/// multiplies by well-scaled reciprocals) to working accuracy.
fn check_extreme_diagonals<T: TestScalar>(scales: &[f64], seed: u64) {
    let mut rng = Lcg(seed);
    for flags in all_flags() {
        let (side, uplo, trans, diag) = flags;
        if diag == Diag::Unit {
            // The diagonal is never read.
            continue;
        }
        for &s in &[1usize, 5, 70, 130] {
            let junk = T::from_parts(f64::NAN, f64::NAN);
            let a = tri_fixture::<T>(s, uplo, &mut rng, junk);
            let (m, n) = rhs_shape(side, s, 3);
            let x0 = dense_fixture::<T>(m, n, &mut rng);
            let b = rhs_of(flags, &a, &x0);
            let mut want = b.clone();
            trsm_in_place(side, uplo, trans, diag, a.as_ref(), want.as_mut())
                .expect("non-singular");
            let tol = T::TOL * 8.0 * (s as f64 + 4.0);
            for &scale in scales {
                let a_s = scaled_triangle(&a, uplo, scale);
                let mut got = scaled(T::from_parts(scale, 0.0), &b);
                trsm_in_place(side, uplo, trans, diag, a_s.as_ref(), got.as_mut())
                    .expect("non-singular");
                for j in 0..n {
                    for i in 0..m {
                        let (g, w) = (got[(i, j)].to_c64(), want[(i, j)].to_c64());
                        assert!(
                            (g - w).norm() <= tol * (1.0 + w.norm()),
                            "{} {flags:?} s={s} scale={scale:e}: ({i},{j}): {g} vs {w}",
                            core::any::type_name::<T>()
                        );
                    }
                }
            }
        }
    }
}

/// Powers of two `2^k` for the exponents `ks`.
fn pow2s(ks: &[i32]) -> Vec<f64> {
    ks.iter().map(|&k| 2f64.powi(k)).collect()
}

/// Complex64 diagonals of modulus `2^k·[1, 1.5]`: `|d|²` overflows (`k =
/// 600, 1000`: the reciprocal is 0), is subnormal (`k = -535`: the reciprocal
/// keeps a few digits) or underflows to zero (`k = -600, -1000`: the
/// reciprocal is not finite and the plain complex quotient is NaN).
#[test]
fn trsm_c64_huge_and_tiny_diagonals() {
    check_extreme_diagonals::<Complex64>(&pow2s(&[1000, 600, -535, -600, -1000]), 0x7253_00F1);
}

/// Complex32 counterpart: `|d|²` overflows for `k = 70, 100`, is subnormal
/// for `k = -72` and underflows to zero for `k = -80, -100`.
#[test]
fn trsm_c32_huge_and_tiny_diagonals() {
    check_extreme_diagonals::<Complex32>(&pow2s(&[100, 70, -72, -80, -100]), 0x7253_00F2);
}

/// The real types accept the same extreme diagonals: a real reciprocal is
/// correctly rounded, so the multiply mode serves them all.
#[test]
fn trsm_real_huge_and_tiny_diagonals() {
    check_extreme_diagonals::<f64>(&pow2s(&[1000, 600, -535, -600, -1000]), 0x7253_00F3);
    check_extreme_diagonals::<f32>(&pow2s(&[100, 70, -72, -80, -100]), 0x7253_00F4);
}

/// The reported case: `d = 1e200`, `b = 1e200` gives `x = 1` (it gave 0),
/// through both complex entry points, on both sides.
#[test]
fn trsm_complex_unit_quotient_of_huge_values() {
    for side in [Side::Left, Side::Right] {
        let a = Mat::from_rows(&[&[Complex64::new(1e200, 0.0)]]);
        let b = Mat::from_rows(&[&[Complex64::new(1e200, 0.0)]]);
        let one = Complex64::new(1.0, 0.0);
        let (lo, nt, nu) = (Uplo::Lower, Trans::NoTrans, Diag::NonUnit);
        let x = trsm_c64(side, lo, nt, nu, one, a.as_ref(), b.as_ref()).expect("non-singular");
        assert!(
            (x[(0, 0)] - one).norm() <= 4.0 * f64::EPSILON,
            "{side:?}: {}",
            x[(0, 0)]
        );
        // conj(1e30 + 1e30i)·x = 2e30  =>  x = 1 + i.
        let a = Mat::from_rows(&[&[Complex32::new(1e30, 1e30)]]);
        let b = Mat::from_rows(&[&[Complex32::new(2e30, 0.0)]]);
        let one = Complex32::new(1.0, 0.0);
        let (up, ct) = (Uplo::Upper, Trans::ConjTrans);
        let x = trsm_c32(side, up, ct, nu, one, a.as_ref(), b.as_ref()).expect("non-singular");
        let want = Complex32::new(1.0, 1.0);
        assert!(
            (x[(0, 0)] - want).norm() <= 8.0 * f32::EPSILON,
            "{side:?}: {}",
            x[(0, 0)]
        );
    }
}

/// Which diagonals select the exact-division mode. The real rule is
/// unchanged (only a non-finite reciprocal); complex diagonals whose squared
/// modulus overflows, is subnormal or underflows divide as well.
#[test]
fn trsm_nonunit_diagonal_mode_selection() {
    use super::tri_driver::nonunit_diag;
    use super::tri_kernel::SolveDiag;
    let mode_f64 = |d: &[f64]| nonunit_diag(d.to_vec()).0;
    let mode_f32 = |d: &[f32]| nonunit_diag(d.to_vec()).0;
    let mode_c64 = |d: &[(f64, f64)]| {
        nonunit_diag(d.iter().map(|&(re, im)| Complex64::new(re, im)).collect()).0
    };
    let mode_c32 = |d: &[(f32, f32)]| {
        nonunit_diag(d.iter().map(|&(re, im)| Complex32::new(re, im)).collect()).0
    };
    let (recip, divide) = (SolveDiag::Reciprocal, SolveDiag::Divide);
    assert_eq!(mode_f64(&[1e300, -1e-300, 3.0, f64::MAX]), recip);
    assert_eq!(mode_f64(&[1.0, 4e-310]), divide);
    assert_eq!(mode_f32(&[3e38, -1e-38, 0.5]), recip);
    assert_eq!(mode_f32(&[1.0, 1e-40]), divide);
    assert_eq!(
        mode_c64(&[(1.0, 0.5), (1e150, -1e150), (2e-154, 0.0)]),
        recip
    );
    assert_eq!(mode_c64(&[(1.0, 0.0), (1e200, 0.0)]), divide);
    assert_eq!(mode_c64(&[(3e-157, 1e-157)]), divide);
    assert_eq!(mode_c64(&[(0.0, 1e-200)]), divide);
    assert_eq!(mode_c32(&[(1e19, 0.0), (1e-19, 1e-19)]), recip);
    assert_eq!(mode_c32(&[(1e20, 0.0)]), divide);
    assert_eq!(mode_c32(&[(1e-21, 0.0)]), divide);
    // Non-finite entries keep the reciprocal (NaN / Inf propagate).
    assert_eq!(mode_c64(&[(f64::INFINITY, 0.0), (f64::NAN, 1.0)]), recip);
    assert_eq!(mode_f64(&[f64::NEG_INFINITY]), recip);
}

/// The robust division agrees with the plain one where that is accurate, and
/// stays accurate where `|d|²` overflows or underflows.
#[test]
fn trsm_robust_complex_division() {
    use super::tri_kernel::div_diag;
    let close =
        |got: Complex64, want: Complex64| (got - want).norm() <= 8.0 * f64::EPSILON * want.norm();
    let mut rng = Lcg(0x7253_00F5);
    for _ in 0..200 {
        let x = Complex64::new(rng.unit(), rng.unit());
        let d = Complex64::new(rng.unit(), rng.unit());
        let want = x / d;
        assert!(close(div_diag(x, d), want), "{x} / {d}");
        // The same quotient with both operands scaled to the extremes.
        for k in [-1000, -600, -520, 520, 600, 1000, 1020] {
            let s = 2f64.powi(k);
            let (xs, ds) = (x * s, d * s);
            assert!(close(div_diag(xs, ds), want), "k={k}: {xs} / {ds}");
        }
    }
    // Real types divide plainly (bitwise).
    assert_eq!(div_diag(1.0f64, 3.0).to_bits(), (1.0f64 / 3.0).to_bits());
    assert_eq!(
        div_diag(1e-300f64, 4e-310).to_bits(),
        (1e-300f64 / 4e-310).to_bits()
    );
    assert_eq!(div_diag(2.0f32, 7.0).to_bits(), (2.0f32 / 7.0).to_bits());
    let c = div_diag(Complex32::new(3e30, 4e30), Complex32::new(1e30, 2e30));
    assert!(
        (c - Complex32::new(2.2, -0.4)).norm() <= 8.0 * f32::EPSILON,
        "{c}"
    );
}
