//! Tests for the GEMM driver.

use super::*;
use oxiblas_matrix::Mat;

#[test]
fn test_gemm_identity() {
    let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let b: Mat<f64> = Mat::eye(2);
    let mut c: Mat<f64> = Mat::zeros(2, 2);

    gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((c[(0, 1)] - 2.0).abs() < 1e-10);
    assert!((c[(1, 0)] - 3.0).abs() < 1e-10);
    assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
}

#[test]
fn test_gemm_small() {
    // A = [1 2 3]    B = [1 4]
    //     [4 5 6]        [2 5]
    //                    [3 6]
    // A * B = [1*1+2*2+3*3  1*4+2*5+3*6] = [14 32]
    //         [4*1+5*2+6*3  4*4+5*5+6*6]   [32 77]

    let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    let b: Mat<f64> = Mat::from_rows(&[&[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]]);
    let mut c: Mat<f64> = Mat::zeros(2, 2);

    gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    assert!((c[(0, 0)] - 14.0).abs() < 1e-10);
    assert!((c[(0, 1)] - 32.0).abs() < 1e-10);
    assert!((c[(1, 0)] - 32.0).abs() < 1e-10);
    assert!((c[(1, 1)] - 77.0).abs() < 1e-10);
}

#[test]
fn test_gemm_with_alpha_beta() {
    let a: Mat<f64> = Mat::filled(2, 2, 1.0);
    let b: Mat<f64> = Mat::filled(2, 2, 2.0);
    let mut c: Mat<f64> = Mat::filled(2, 2, 10.0);

    // C = 2 * A * B + 3 * C
    // A * B = [4 4; 4 4] (each element is 1*2 + 1*2)
    // Result = 2 * [4 4; 4 4] + 3 * [10 10; 10 10] = [8 8; 8 8] + [30 30; 30 30] = [38 38; 38 38]
    gemm(2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (c[(i, j)] - 38.0).abs() < 1e-10,
                "c[{},{}] = {}",
                i,
                j,
                c[(i, j)]
            );
        }
    }
}

#[test]
fn test_gemm_larger() {
    let n = 64;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let b: Mat<f64> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f64> = Mat::zeros(n, n);

    gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    // Each element should be n (sum of n ones)
    for i in 0..n {
        for j in 0..n {
            assert!(
                (c[(i, j)] - n as f64).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                n
            );
        }
    }
}

#[test]
fn test_gemm_f32() {
    let a: Mat<f32> = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
    let b: Mat<f32> = Mat::from_rows(&[&[5.0f32, 6.0], &[7.0, 8.0]]);
    let mut c: Mat<f32> = Mat::zeros(2, 2);

    // A * B = [1*5+2*7  1*6+2*8] = [19 22]
    //         [3*5+4*7  3*6+4*8]   [43 50]

    gemm(1.0f32, a.as_ref(), b.as_ref(), 0.0f32, c.as_mut());

    assert!((c[(0, 0)] - 19.0).abs() < 1e-5);
    assert!((c[(0, 1)] - 22.0).abs() < 1e-5);
    assert!((c[(1, 0)] - 43.0).abs() < 1e-5);
    assert!((c[(1, 1)] - 50.0).abs() < 1e-5);
}

#[test]
fn test_gemm_parallel() {
    // Test with a larger matrix to trigger parallel execution
    let n = 256;
    let a: Mat<f64> = Mat::filled(n, n, 1.0);
    let b: Mat<f64> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f64> = Mat::zeros(n, n);

    #[cfg(feature = "parallel")]
    {
        gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
    }
    #[cfg(not(feature = "parallel"))]
    {
        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
    }

    // Each element should be n (sum of n ones)
    for i in 0..n {
        for j in 0..n {
            assert!(
                (c[(i, j)] - n as f64).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                n
            );
        }
    }
}

#[test]
fn test_asymmetric_blocking_parameters() {
    use crate::level3::gemm_kernel::MicroKernelShape;

    let shape = MicroKernelShape { mr: 8, nr: 6 };

    // Test balanced matrix (should use standard blocking)
    let blocking = GemmBlocking::asymmetric::<f64>(100, 100, 100, &shape);
    assert!(blocking.mc > 0);
    assert!(blocking.kc > 0);
    assert!(blocking.nc > 0);

    // Test tall-thin matrix (m >> k, m >> n)
    let blocking_tall = GemmBlocking::asymmetric::<f64>(1000, 10, 20, &shape);
    // Should have larger MC for tall matrices
    assert!(blocking_tall.mc >= 8); // At least mr

    // Test short-wide matrix (n >> m, n >> k)
    let blocking_wide = GemmBlocking::asymmetric::<f64>(20, 10, 1000, &shape);
    // Should have larger NC for wide matrices
    assert!(blocking_wide.nc >= 6); // At least nr

    // Test inner-product dominated (k >> m, k >> n)
    let blocking_inner = GemmBlocking::asymmetric::<f64>(20, 1000, 20, &shape);
    // Should have larger KC for inner-product
    assert!(blocking_inner.kc >= 128);
}

#[test]
fn test_gemm_asymmetric_tall_thin() {
    // Tall-thin: A is 500x10, B is 10x50
    let m = 500;
    let k = 10;
    let n = 50;

    let a: Mat<f64> = Mat::filled(m, k, 1.0);
    let b: Mat<f64> = Mat::filled(k, n, 2.0);
    let mut c: Mat<f64> = Mat::zeros(m, n);

    gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    // Each element should be k * 1.0 * 2.0 = 20.0
    let expected = k as f64 * 2.0;
    for i in 0..m {
        for j in 0..n {
            assert!(
                (c[(i, j)] - expected).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                expected
            );
        }
    }
}

#[test]
fn test_gemm_asymmetric_short_wide() {
    // Short-wide: A is 20x10, B is 10x500
    let m = 20;
    let k = 10;
    let n = 500;

    let a: Mat<f64> = Mat::filled(m, k, 1.0);
    let b: Mat<f64> = Mat::filled(k, n, 2.0);
    let mut c: Mat<f64> = Mat::zeros(m, n);

    gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    let expected = k as f64 * 2.0;
    for i in 0..m {
        for j in 0..n {
            assert!(
                (c[(i, j)] - expected).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                expected
            );
        }
    }
}

#[test]
fn test_gemm_asymmetric_inner_product() {
    // Inner-product dominated: A is 20x500, B is 500x20
    let m = 20;
    let k = 500;
    let n = 20;

    let a: Mat<f64> = Mat::filled(m, k, 1.0);
    let b: Mat<f64> = Mat::filled(k, n, 2.0);
    let mut c: Mat<f64> = Mat::zeros(m, n);

    gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    let expected = k as f64 * 2.0;
    for i in 0..m {
        for j in 0..n {
            assert!(
                (c[(i, j)] - expected).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                expected
            );
        }
    }
}

#[test]
fn test_gemm_asymmetric_panel_panel() {
    // Panel-panel: A is 200x10, B is 10x200 (m and n large, k small)
    let m = 200;
    let k = 10;
    let n = 200;

    let a: Mat<f64> = Mat::filled(m, k, 1.0);
    let b: Mat<f64> = Mat::filled(k, n, 2.0);
    let mut c: Mat<f64> = Mat::zeros(m, n);

    gemm_asymmetric(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

    let expected = k as f64 * 2.0;
    for i in 0..m {
        for j in 0..n {
            assert!(
                (c[(i, j)] - expected).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                expected
            );
        }
    }
}

#[test]
fn test_gemm_asymmetric_with_alpha_beta() {
    let m = 100;
    let k = 10;
    let n = 200;

    let a: Mat<f64> = Mat::filled(m, k, 1.0);
    let b: Mat<f64> = Mat::filled(k, n, 2.0);
    let mut c: Mat<f64> = Mat::filled(m, n, 5.0);

    // C = 2 * A * B + 3 * C
    // A * B each element = 10 * 2 = 20
    // Result = 2 * 20 + 3 * 5 = 40 + 15 = 55
    gemm_asymmetric(2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());

    let expected = 2.0 * (k as f64 * 2.0) + 3.0 * 5.0;
    for i in 0..m {
        for j in 0..n {
            assert!(
                (c[(i, j)] - expected).abs() < 1e-10,
                "c[{},{}] = {}, expected {}",
                i,
                j,
                c[(i, j)],
                expected
            );
        }
    }
}

// =====================================================================
// Transpose-aware GEMM + non-constant-data regression tests.
//
// The pre-existing tests above fill matrices with a single constant, which
// cannot distinguish a correct kernel from one with a transposed-vs-not
// indexing bug (all products of equal values are equal). The tests below
// use deterministic, seeded, *non-symmetric* fills so index bugs surface,
// and check every result against an independent naive triple-loop reference
// that does NOT call gemm() (avoiding a self-consistency-only test).
// =====================================================================

/// Deterministic non-constant fill. Every element is distinct and the
/// pattern is non-symmetric, so `a[(i,j)] != a[(j,i)]` for `i != j`.
fn pattern_f64(rows: usize, cols: usize, seed: usize) -> Mat<f64> {
    let mut m = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m[(i, j)] = ((i * cols + j) as f64) * 0.05 + (seed as f64) * 0.25 + 1.0;
        }
    }
    m
}

fn pattern_c64(rows: usize, cols: usize, seed: usize) -> Mat<Complex64> {
    let mut m = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let re = ((i * cols + j) as f64) * 0.05 + (seed as f64) * 0.25 + 1.0;
            let im = ((i * cols + 2 * j) as f64) * 0.03 - 0.5 + (seed as f64) * 0.1;
            m[(i, j)] = Complex64::new(re, im);
        }
    }
    m
}

fn pattern_c32(rows: usize, cols: usize, seed: usize) -> Mat<Complex32> {
    let mut m = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let re = ((i * cols + j) as f32) * 0.05 + (seed as f32) * 0.25 + 1.0;
            let im = ((i * cols + 2 * j) as f32) * 0.03 - 0.5 + (seed as f32) * 0.1;
            m[(i, j)] = Complex32::new(re, im);
        }
    }
    m
}

/// Reads `op(mat)[r, c]` for reals (conjugation is the identity on reals).
fn op_read_f64(mat: &Mat<f64>, trans: Trans, r: usize, c: usize) -> f64 {
    match trans {
        Trans::NoTrans => mat[(r, c)],
        Trans::Trans | Trans::ConjTrans => mat[(c, r)],
    }
}

/// Independent naive reference: `C = alpha * op(A) * op(B) + beta * C_init`.
/// Never calls gemm() — this is the ground truth the kernel is checked against.
fn naive_trans_f64(
    trans_a: Trans,
    trans_b: Trans,
    alpha: f64,
    a: &Mat<f64>,
    b: &Mat<f64>,
    beta: f64,
    c_init: &Mat<f64>,
    m: usize,
    k: usize,
    n: usize,
) -> Mat<f64> {
    let mut out = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += op_read_f64(a, trans_a, i, p) * op_read_f64(b, trans_b, p, j);
            }
            out[(i, j)] = alpha * acc + beta * c_init[(i, j)];
        }
    }
    out
}

fn op_read_c64(mat: &Mat<Complex64>, trans: Trans, r: usize, c: usize) -> Complex64 {
    match trans {
        Trans::NoTrans => mat[(r, c)],
        Trans::Trans => mat[(c, r)],
        Trans::ConjTrans => mat[(c, r)].conj(),
    }
}

fn op_read_c32(mat: &Mat<Complex32>, trans: Trans, r: usize, c: usize) -> Complex32 {
    match trans {
        Trans::NoTrans => mat[(r, c)],
        Trans::Trans => mat[(c, r)],
        Trans::ConjTrans => mat[(c, r)].conj(),
    }
}

#[test]
fn test_gemm_blocked_nonconstant() {
    // m*n*k > SMALL_THRESHOLD forces the blocked (not small) path.
    let (m, k, n) = (80usize, 64usize, 72usize);
    assert!(m * n * k > SMALL_THRESHOLD);
    let a = pattern_f64(m, k, 1);
    let b = pattern_f64(k, n, 2);
    let c_init = pattern_f64(m, n, 3);
    let mut c = pattern_f64(m, n, 3);
    let (alpha, beta) = (0.5, 2.0);

    gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
    let expected = naive_trans_f64(
        Trans::NoTrans,
        Trans::NoTrans,
        alpha,
        &a,
        &b,
        beta,
        &c_init,
        m,
        k,
        n,
    );

    for i in 0..m {
        for j in 0..n {
            let (got, exp) = (c[(i, j)], expected[(i, j)]);
            assert!(
                (got - exp).abs() <= 1e-9 * exp.abs().max(1.0),
                "blocked ({},{}): got {} exp {}",
                i,
                j,
                got,
                exp
            );
        }
    }
}

#[test]
fn test_gemm_parallel_nonconstant() {
    // Large enough to cross the parallel work threshold when the feature is on.
    let (m, k, n) = (96usize, 80usize, 88usize);
    let a = pattern_f64(m, k, 4);
    let b = pattern_f64(k, n, 5);
    let c_init = pattern_f64(m, n, 6);
    let mut c = pattern_f64(m, n, 6);
    let (alpha, beta) = (1.0, 0.0);

    #[cfg(feature = "parallel")]
    {
        gemm_with_par(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut(), Par::Rayon);
    }
    #[cfg(not(feature = "parallel"))]
    {
        gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
    }

    let expected = naive_trans_f64(
        Trans::NoTrans,
        Trans::NoTrans,
        alpha,
        &a,
        &b,
        beta,
        &c_init,
        m,
        k,
        n,
    );
    for i in 0..m {
        for j in 0..n {
            let (got, exp) = (c[(i, j)], expected[(i, j)]);
            assert!(
                (got - exp).abs() <= 1e-9 * exp.abs().max(1.0),
                "parallel ({},{}): got {} exp {}",
                i,
                j,
                got,
                exp
            );
        }
    }
}

#[test]
fn test_gemm_transposed_nn_matches_gemm() {
    // The NoTrans/NoTrans path must be identical to plain gemm().
    let (m, k, n) = (50usize, 40usize, 44usize);
    let a = pattern_f64(m, k, 7);
    let b = pattern_f64(k, n, 8);
    let mut c1 = pattern_f64(m, n, 9);
    let mut c2 = pattern_f64(m, n, 9);

    gemm(1.3, a.as_ref(), b.as_ref(), 0.4, c1.as_mut());
    gemm_transposed(
        Trans::NoTrans,
        Trans::NoTrans,
        1.3,
        a.as_ref(),
        b.as_ref(),
        0.4,
        c2.as_mut(),
    );

    for i in 0..m {
        for j in 0..n {
            assert!(
                (c1[(i, j)] - c2[(i, j)]).abs() < 1e-12,
                "NN mismatch vs gemm at ({},{})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_gemm_transposed_real_all_combos() {
    let combos = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    // Small size (<= SMALL_THRESHOLD) exercises gemm_small_trans; the large
    // size exercises the blocked transpose-aware packers.
    for &(m, k, n) in &[(6usize, 5usize, 7usize), (48usize, 40usize, 36usize)] {
        for &ta in &combos {
            for &tb in &combos {
                // op(A) is m x k, so A is stored m x k (NoTrans) or k x m (transposed).
                let a = if ta == Trans::NoTrans {
                    pattern_f64(m, k, 1)
                } else {
                    pattern_f64(k, m, 1)
                };
                let b = if tb == Trans::NoTrans {
                    pattern_f64(k, n, 2)
                } else {
                    pattern_f64(n, k, 2)
                };
                let c_init = pattern_f64(m, n, 3);
                let mut c = pattern_f64(m, n, 3);
                let (alpha, beta) = (1.5, -0.75);

                gemm_transposed(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
                let expected = naive_trans_f64(ta, tb, alpha, &a, &b, beta, &c_init, m, k, n);

                for i in 0..m {
                    for j in 0..n {
                        let (got, exp) = (c[(i, j)], expected[(i, j)]);
                        assert!(
                            (got - exp).abs() <= 1e-9 * exp.abs().max(1.0),
                            "combo {:?}/{:?} size {}x{}x{} at ({},{}): got {} exp {}",
                            ta,
                            tb,
                            m,
                            k,
                            n,
                            i,
                            j,
                            got,
                            exp
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gemm_transposed_conjtrans_c64() {
    // Size > SMALL_THRESHOLD exercises the optimized 3M complex path.
    let (m, k, n) = (36usize, 40usize, 32usize);
    let combos = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    let alpha = Complex64::new(1.25, -0.5);
    let beta = Complex64::new(-0.75, 0.25);

    for &ta in &combos {
        for &tb in &combos {
            let a = if ta == Trans::NoTrans {
                pattern_c64(m, k, 1)
            } else {
                pattern_c64(k, m, 1)
            };
            let b = if tb == Trans::NoTrans {
                pattern_c64(k, n, 2)
            } else {
                pattern_c64(n, k, 2)
            };
            let c_init = pattern_c64(m, n, 3);
            let mut c = pattern_c64(m, n, 3);

            gemm_transposed_c64(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());

            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for p in 0..k {
                        acc += op_read_c64(&a, ta, i, p) * op_read_c64(&b, tb, p, j);
                    }
                    let exp = alpha * acc + beta * c_init[(i, j)];
                    let got = c[(i, j)];
                    assert!(
                        (got - exp).norm() <= 1e-6 * exp.norm().max(1.0),
                        "c64 {:?}/{:?} at ({},{}): got {} exp {}",
                        ta,
                        tb,
                        i,
                        j,
                        got,
                        exp
                    );
                }
            }
        }
    }
}

#[test]
fn test_gemm_transposed_conjtrans_c32() {
    // Small size keeps the naive 3M path (accurate for f32 comparison).
    let (m, k, n) = (20usize, 16usize, 24usize);
    let combos = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    let alpha = Complex32::new(0.75, 0.5);
    let beta = Complex32::new(-0.25, -0.5);

    for &ta in &combos {
        for &tb in &combos {
            let a = if ta == Trans::NoTrans {
                pattern_c32(m, k, 1)
            } else {
                pattern_c32(k, m, 1)
            };
            let b = if tb == Trans::NoTrans {
                pattern_c32(k, n, 2)
            } else {
                pattern_c32(n, k, 2)
            };
            let c_init = pattern_c32(m, n, 3);
            let mut c = pattern_c32(m, n, 3);

            gemm_transposed_c32(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());

            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex32::new(0.0, 0.0);
                    for p in 0..k {
                        acc += op_read_c32(&a, ta, i, p) * op_read_c32(&b, tb, p, j);
                    }
                    let exp = alpha * acc + beta * c_init[(i, j)];
                    let got = c[(i, j)];
                    assert!(
                        (got - exp).norm() <= 1e-3 * exp.norm().max(1.0),
                        "c32 {:?}/{:?} at ({},{}): got {} exp {}",
                        ta,
                        tb,
                        i,
                        j,
                        got,
                        exp
                    );
                }
            }
        }
    }
}

#[test]
fn test_gemm_blocking_custom_clamps_and_runs() {
    let shape = MicroKernelShape { mr: 8, nr: 6 };

    // Degenerate requests must clamp up to a valid micro-tile, not collapse
    // to a zero block size (which previously panicked inside gemm).
    let deg = GemmBlocking::custom(3, 0, 2, &shape);
    assert_eq!(deg.mc, 8, "mc must clamp to mr");
    assert_eq!(deg.nc, 6, "nc must clamp to nr");
    assert_eq!(deg.kc, 1, "kc must clamp to >= 1");

    let all_zero = GemmBlocking::custom(0, 0, 0, &shape);
    assert!(all_zero.mc >= 8 && all_zero.nc >= 6 && all_zero.kc >= 1);

    // A real GEMM driven by the degenerate blocking must neither panic nor
    // produce a wrong result (it is merely inefficient).
    let (m, k, n) = (40usize, 40usize, 40usize);
    assert!(m * n * k > SMALL_THRESHOLD);
    let a = pattern_f64(m, k, 1);
    let b = pattern_f64(k, n, 2);
    let c_init = pattern_f64(m, n, 3);
    let mut c = pattern_f64(m, n, 3);

    gemm_with_blocking(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Seq, &deg);
    let expected = naive_trans_f64(
        Trans::NoTrans,
        Trans::NoTrans,
        1.0,
        &a,
        &b,
        0.0,
        &c_init,
        m,
        k,
        n,
    );

    for i in 0..m {
        for j in 0..n {
            let (got, exp) = (c[(i, j)], expected[(i, j)]);
            assert!(
                (got - exp).abs() <= 1e-9 * exp.abs().max(1.0),
                "degenerate-blocking ({},{}): got {} exp {}",
                i,
                j,
                got,
                exp
            );
        }
    }
}

fn assert_close_f64(got: &Mat<f64>, exp: &Mat<f64>, what: &str) {
    for i in 0..exp.nrows() {
        for j in 0..exp.ncols() {
            let (g, e) = (got[(i, j)], exp[(i, j)]);
            assert!(
                (g - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{what} ({i},{j}): got {g} exp {e}"
            );
        }
    }
}

#[test]
fn test_gemm_small_sizes_on_blocked_path() {
    // Sizes from SMALL_BLOCKED_MIN to SMALL_THRESHOLD, all transpose combinations, beta != 0.
    let combos = [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
        (Trans::Trans, Trans::Trans),
    ];
    for &(m, k, n) in &[
        (12, 12, 12),
        (16, 16, 16),
        (31, 23, 29),
        (32, 32, 32),
        (8, 40, 6),
    ] {
        for &(ta, tb) in &combos {
            let a = if ta == Trans::NoTrans {
                pattern_f64(m, k, 1)
            } else {
                pattern_f64(k, m, 1)
            };
            let b = if tb == Trans::NoTrans {
                pattern_f64(k, n, 2)
            } else {
                pattern_f64(n, k, 2)
            };
            let c_init = pattern_f64(m, n, 3);
            let mut c = pattern_f64(m, n, 3);
            gemm_transposed(ta, tb, 0.5, a.as_ref(), b.as_ref(), -1.5, c.as_mut());
            let exp = naive_trans_f64(ta, tb, 0.5, &a, &b, -1.5, &c_init, m, k, n);
            assert_close_f64(&c, &exp, &format!("{m}x{k}x{n} {ta:?} {tb:?}"));
        }
    }
}

#[test]
fn test_gemm_small_alpha_zero_ignores_a_and_b() {
    // alpha == 0 must not read A or B, also at sizes that otherwise use the blocked path.
    let n = 16;
    let a: Mat<f64> = Mat::filled(n, n, f64::NAN);
    let b: Mat<f64> = Mat::filled(n, n, 1.0);
    let mut c: Mat<f64> = Mat::filled(n, n, 2.0);
    gemm(0.0, a.as_ref(), b.as_ref(), 0.5, c.as_mut());
    for i in 0..n {
        for j in 0..n {
            assert_eq!(c[(i, j)], 1.0, "c[{i},{j}]");
        }
    }
}
