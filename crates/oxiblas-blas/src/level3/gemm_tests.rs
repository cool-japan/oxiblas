//! Unit tests for the blocked GEMM driver, its blocking parameters and the
//! transpose-aware entry points.

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
    // m*n*k > SMALL_THRESHOLD_SIMD forces the blocked (not small) path.
    let (m, k, n) = (96usize, 64usize, 88usize);
    assert!(m * n * k > SMALL_THRESHOLD_SIMD);
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
    // Crosses the parallel gate on 4 threads (>= 128^3 multiply-adds and
    // >= 80^3 per thread) when the feature is on, whatever the core count.
    let (m, k, n) = (141usize, 135usize, 139usize);
    let a = pattern_f64(m, k, 4);
    let b = pattern_f64(k, n, 5);
    let c_init = pattern_f64(m, n, 6);
    let mut c = pattern_f64(m, n, 6);
    let (alpha, beta) = (1.0, 0.0);

    #[cfg(feature = "parallel")]
    {
        let par = Par::RayonWith(4);
        assert!(PARALLEL_GATE.should_parallelize(m * n * k, par));
        gemm_with_par(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut(), par);
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
    let (m, k, n) = (88usize, 80usize, 80usize);
    assert!(m * n * k > SMALL_THRESHOLD_SIMD);
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

// =====================================================================
// Issue #9: reference BLAS reads neither A nor B when `alpha == 0`, and
// never reads C when `beta == 0`, so NaN/Inf in an ignored operand must
// not leak into the result. `n = 5` takes the small-matrix path, 40 the
// blocked path except where the SIMD small kernels take it (x86_64:
// 40^3 <= SMALL_THRESHOLD_SIMD), and 100 the blocked path.
// =====================================================================

/// Sizes covering the small-matrix path and the blocked path.
const ISSUE_9_SIZES: [usize; 3] = [5, 40, 100];

/// Asserts `got == expected` element-wise with exact (bitwise) equality.
fn assert_bits_eq_f64(got: &Mat<f64>, expected: &Mat<f64>, what: &str) {
    assert_eq!(got.shape(), expected.shape(), "{what}: shape");
    for j in 0..got.ncols() {
        for i in 0..got.nrows() {
            assert_eq!(
                got[(i, j)].to_bits(),
                expected[(i, j)].to_bits(),
                "{what} at ({i},{j}): got {} expected {}",
                got[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
fn test_issue_9_gemm_alpha_zero_ignores_nan_operands() {
    for &n in &ISSUE_9_SIZES {
        for &poison in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            // Reporter's case: gemm(0.0, A = NaN, B = 1, beta = 0.5, C = 2) == 1.
            let a = Mat::filled(n, n, poison);
            let ones = Mat::filled(n, n, 1.0);
            let mut c = Mat::filled(n, n, 2.0);
            gemm(0.0, a.as_ref(), ones.as_ref(), 0.5, c.as_mut());
            assert_bits_eq_f64(&c, &Mat::filled(n, n, 1.0), "alpha=0 beta=0.5");

            // Both operands poisoned; beta in {-0.75, 1, 0}.
            let b = Mat::filled(n, n, poison);
            let c_init = pattern_f64(n, n, 3);
            let mut c = pattern_f64(n, n, 3);
            gemm(0.0, a.as_ref(), b.as_ref(), -0.75, c.as_mut());
            let mut expected = pattern_f64(n, n, 3);
            for j in 0..n {
                for i in 0..n {
                    expected[(i, j)] = -0.75 * c_init[(i, j)];
                }
            }
            assert_bits_eq_f64(&c, &expected, "alpha=0 beta=-0.75");

            let mut c = pattern_f64(n, n, 3);
            gemm(0.0, a.as_ref(), b.as_ref(), 1.0, c.as_mut());
            assert_bits_eq_f64(&c, &c_init, "alpha=0 beta=1");

            let mut c = Mat::filled(n, n, f64::NAN);
            gemm(0.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
            assert_bits_eq_f64(&c, &Mat::zeros(n, n), "alpha=0 beta=0");
        }

        // f32 through the same funnel.
        let a32: Mat<f32> = Mat::filled(n, n, f32::NAN);
        let b32: Mat<f32> = Mat::filled(n, n, f32::INFINITY);
        let mut c32: Mat<f32> = Mat::filled(n, n, 2.0);
        gemm(0.0f32, a32.as_ref(), b32.as_ref(), 0.5f32, c32.as_mut());
        for j in 0..n {
            for i in 0..n {
                assert_eq!(
                    c32[(i, j)].to_bits(),
                    1.0f32.to_bits(),
                    "f32 n={n} ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn test_issue_9_gemm_transposed_alpha_zero_ignores_nan_operands() {
    let combos = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
    for &n in &ISSUE_9_SIZES {
        // Rectangular so a transposed operand has a different stored shape.
        let (m, k) = (n + 3, n + 1);
        for &ta in &combos {
            for &tb in &combos {
                let a = if ta == Trans::NoTrans {
                    Mat::filled(m, k, f64::NAN)
                } else {
                    Mat::filled(k, m, f64::NAN)
                };
                let b = if tb == Trans::NoTrans {
                    Mat::filled(k, n, f64::INFINITY)
                } else {
                    Mat::filled(n, k, f64::INFINITY)
                };
                let mut c = Mat::filled(m, n, 2.0);
                gemm_transposed(ta, tb, 0.0, a.as_ref(), b.as_ref(), 0.5, c.as_mut());
                assert_bits_eq_f64(&c, &Mat::filled(m, n, 1.0), "gemm_transposed alpha=0");
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_issue_9_gemm_parallel_alpha_zero_ignores_nan_operands() {
    // Large enough for the parallel driver once alpha != 0.
    let n = 160;
    let a = Mat::filled(n, n, f64::NAN);
    let b = Mat::filled(n, n, f64::NAN);
    for &par in &[Par::Rayon, Par::RayonWith(3)] {
        let mut c = Mat::filled(n, n, 2.0);
        gemm_with_par(0.0, a.as_ref(), b.as_ref(), 0.5, c.as_mut(), par);
        assert_bits_eq_f64(&c, &Mat::filled(n, n, 1.0), "parallel alpha=0");
    }
}

#[test]
fn test_issue_9_gemm_beta_zero_ignores_nan_c() {
    let combos = [Trans::NoTrans, Trans::Trans];
    for &n in &ISSUE_9_SIZES {
        let (m, k) = (n + 2, n + 1);
        for &ta in &combos {
            for &tb in &combos {
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
                let zeros = Mat::zeros(m, n);
                let mut c = Mat::filled(m, n, f64::NAN);
                gemm_transposed(ta, tb, 1.5, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
                let expected = naive_trans_f64(ta, tb, 1.5, &a, &b, 0.0, &zeros, m, k, n);
                for j in 0..n {
                    for i in 0..m {
                        let (got, exp) = (c[(i, j)], expected[(i, j)]);
                        assert!(
                            got.is_finite() && (got - exp).abs() <= 1e-9 * exp.abs().max(1.0),
                            "beta=0 {ta:?}/{tb:?} n={n} ({i},{j}): got {got} exp {exp}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_issue_9_gemm_batched_alpha_zero_ignores_nan_operands() {
    use crate::level3::batched::{Transpose, gemm_batched, gemm_batched_c32, gemm_batched_c64};

    for &n in &[5usize, 40] {
        for &(ta, tb) in &[
            (Transpose::NoTrans, Transpose::NoTrans),
            (Transpose::Trans, Transpose::ConjTrans),
        ] {
            let a = Mat::filled(n, n, f64::NAN);
            let b = Mat::filled(n, n, f64::INFINITY);
            let mut c0 = Mat::filled(n, n, 2.0);
            let mut c1 = Mat::filled(n, n, f64::NAN);
            {
                let mut cs = [c0.as_mut(), c1.as_mut()];
                let result = gemm_batched(
                    ta,
                    tb,
                    0.0,
                    &[a.as_ref(), a.as_ref()],
                    &[b.as_ref(), b.as_ref()],
                    0.5,
                    &mut cs,
                );
                assert_eq!(result, Ok(()));
            }
            // c1 started as NaN and beta * NaN stays NaN; only c0 is checked.
            assert_bits_eq_f64(&c0, &Mat::filled(n, n, 1.0), "batched alpha=0");

            let mut c2 = Mat::filled(n, n, f64::NAN);
            {
                let mut cs = [c2.as_mut()];
                let result = gemm_batched(ta, tb, 0.0, &[a.as_ref()], &[b.as_ref()], 0.0, &mut cs);
                assert_eq!(result, Ok(()));
            }
            assert_bits_eq_f64(&c2, &Mat::zeros(n, n), "batched alpha=0 beta=0");

            let nan64 = Complex64::new(f64::NAN, f64::NAN);
            let a64 = Mat::filled(n, n, nan64);
            let mut cz = Mat::filled(n, n, Complex64::new(2.0, -4.0));
            {
                let mut cs = [cz.as_mut()];
                let result = gemm_batched_c64(
                    ta,
                    tb,
                    Complex64::new(0.0, 0.0),
                    &[a64.as_ref()],
                    &[a64.as_ref()],
                    Complex64::new(0.5, 0.0),
                    &mut cs,
                );
                assert_eq!(result, Ok(()));
            }
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(cz[(i, j)], Complex64::new(1.0, -2.0), "c64 n={n} ({i},{j})");
                }
            }

            let nan32 = Complex32::new(f32::NAN, f32::NAN);
            let a32 = Mat::filled(n, n, nan32);
            let mut cc = Mat::filled(n, n, nan32);
            {
                let mut cs = [cc.as_mut()];
                let result = gemm_batched_c32(
                    ta,
                    tb,
                    Complex32::new(0.0, 0.0),
                    &[a32.as_ref()],
                    &[a32.as_ref()],
                    Complex32::new(0.0, 0.0),
                    &mut cs,
                );
                assert_eq!(result, Ok(()));
            }
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(cc[(i, j)], Complex32::new(0.0, 0.0), "c32 n={n} ({i},{j})");
                }
            }
        }
    }
}

// =====================================================================
// Latent bug: the NoTrans fast packers address `op(X)[r, c]` as
// `ptr + r + c * row_stride`, i.e. they assume `col_stride == 1`.
// `MatRef::transpose()` produces genuine strided views (col_stride != 1),
// so a blocked GEMM on such a view used to read the wrong elements.
// =====================================================================

/// Owned column-major copy of a (possibly strided) view.
fn copy_view_f64(view: MatRef<'_, f64>) -> Mat<f64> {
    let mut out = Mat::zeros(view.nrows(), view.ncols());
    for j in 0..view.ncols() {
        for i in 0..view.nrows() {
            out[(i, j)] = view[(i, j)];
        }
    }
    out
}

#[test]
fn test_gemm_blocked_transposed_view_operands() {
    for &(m, k, n) in &[(96usize, 64usize, 96usize), (80, 72, 96), (37, 301, 50)] {
        assert!(
            m * n * k > SMALL_THRESHOLD_SIMD,
            "must take the blocked path"
        );
        let a_store = pattern_f64(k, m, 1); // A = a_store^T is m x k
        let b_store = pattern_f64(n, k, 2); // B = b_store^T is k x n
        let a_view = a_store.as_ref().transpose();
        let b_view = b_store.as_ref().transpose();
        assert_ne!(a_view.col_stride(), 1);
        assert_ne!(b_view.col_stride(), 1);
        let a_copy = copy_view_f64(a_view);
        let b_copy = copy_view_f64(b_view);

        let mut c_view = pattern_f64(m, n, 3);
        let mut c_copy = pattern_f64(m, n, 3);
        gemm(1.25, a_view, b_view, -0.5, c_view.as_mut());
        gemm(
            1.25,
            a_copy.as_ref(),
            b_copy.as_ref(),
            -0.5,
            c_copy.as_mut(),
        );
        // Identical packed panels => bitwise identical results.
        assert_bits_eq_f64(&c_view, &c_copy, "strided view vs copy");

        // Mixed: strided A view with a contiguous B, and the double transpose
        // op(view^T) of a strided view through the transpose-aware entry point.
        let mut c_mixed = pattern_f64(m, n, 3);
        gemm(1.25, a_view, b_copy.as_ref(), -0.5, c_mixed.as_mut());
        assert_bits_eq_f64(&c_mixed, &c_copy, "strided A only");

        let a_double = a_copy.as_ref().transpose(); // k x m view, op() = A
        let mut c_double = pattern_f64(m, n, 3);
        gemm_transposed(
            Trans::Trans,
            Trans::NoTrans,
            1.25,
            a_double,
            b_copy.as_ref(),
            -0.5,
            c_double.as_mut(),
        );
        assert_bits_eq_f64(&c_double, &c_copy, "double transpose");

        #[cfg(feature = "parallel")]
        {
            let mut c_par = pattern_f64(m, n, 3);
            gemm_with_par(
                1.25,
                a_view,
                b_view,
                -0.5,
                c_par.as_mut(),
                Par::RayonWith(3),
            );
            assert_bits_eq_f64(&c_par, &c_copy, "parallel strided view");
        }
    }
}

// =====================================================================
// Issue #4: blocks that fit a dimension must not be split. With an 8x6
// micro-kernel (f64 on NEON / AVX2) a dimension of 256 is not a multiple
// of NR = 6; it used to be rounded down to 252 (blocks [252, 4]) and the
// adaptive tall-thin branch shrank an already-clamped NC to 168
// (blocks [168, 88]). Every extra NC block re-packs all of A.
// =====================================================================

#[test]
fn test_issue_4_auto_tuned_does_not_split_fitting_dims() {
    let shape = MicroKernelShape { mr: 8, nr: 6 };
    for &(m, k, n) in &[(1024usize, 1024usize, 256usize), (4096, 512, 256)] {
        let blk = GemmBlocking::auto_tuned::<f64>(m, k, n, &shape);
        assert_eq!(
            n.div_ceil(blk.nc),
            1,
            "({m},{k},{n}): NC = {} splits n = {n} into {} blocks",
            blk.nc,
            n.div_ceil(blk.nc)
        );
    }
}

#[test]
fn test_issue_4_explicit_blocking_is_fitted_to_dims() {
    let shape = MicroKernelShape { mr: 8, nr: 6 };
    // `for_kernel` (used by Strassen / TRMM / TRSM) is dimension-free:
    // f64 targets MC = 576, KC = 448, NC = 2046.
    let fk = GemmBlocking::for_kernel::<f64>(&shape);
    assert_eq!((fk.mc, fk.kc, fk.nc), (576, 448, 2046));

    let fit = fk.fit_to_dims(37, 500, 2050, &shape);
    assert_eq!(fit.mc, 37, "fits: one block");
    assert_eq!(fit.kc, 256, "k = 500: [256, 244], not [448, 52]");
    assert_eq!(fit.nc, 1026, "n = 2050: [1026, 1024], not [2046, 4]");
    // Idempotent.
    let again = fit.fit_to_dims(37, 500, 2050, &shape);
    assert_eq!((again.mc, again.kc, again.nc), (fit.mc, fit.kc, fit.nc));

    // `asymmetric` finalizes its per-shape targets the same way.
    for &(m, k, n) in &[
        (4096usize, 64usize, 256usize),
        (20, 10, 1000),
        (20, 1000, 20),
        (1000, 1000, 250),
    ] {
        let blk = GemmBlocking::asymmetric::<f64>(m, k, n, &shape);
        for (b, dim, unit) in [(blk.mc, m, 8), (blk.kc, k, 8), (blk.nc, n, 6)] {
            assert!(b >= 1 && b <= dim, "({m},{k},{n}) {blk:?}");
            assert!(b == dim || b % unit == 0, "({m},{k},{n}) {blk:?}");
            let blocks = dim.div_ceil(b);
            assert!(blocks * b - dim < blocks * unit, "({m},{k},{n}) {blk:?}");
        }
    }
    // Tall-thin: n = 256 fits the NC target and must not be split.
    assert_eq!(
        GemmBlocking::asymmetric::<f64>(4096, 64, 256, &shape).nc,
        256
    );
}

#[test]
fn test_issue_4_fitted_blocking_matches_reference() {
    // Awkward dimensions that used to produce sliver blocks with the
    // explicit `for_kernel` blocking: k = 500 and n = 2050 against
    // KC = 448 / NC = (2048 / NR) * NR.
    let (m, k, n) = (9usize, 500usize, 2050usize);
    let a = pattern_f64(m, k, 1);
    let b = pattern_f64(k, n, 2);
    let c_init = pattern_f64(m, n, 3);
    let mut c = pattern_f64(m, n, 3);
    let shape = f64::micro_kernel_shape();
    let blocking = GemmBlocking::for_kernel::<f64>(&shape);
    gemm_with_blocking(
        0.5,
        a.as_ref(),
        b.as_ref(),
        -1.0,
        c.as_mut(),
        Par::Seq,
        &blocking,
    );
    let expected = naive_trans_f64(
        Trans::NoTrans,
        Trans::NoTrans,
        0.5,
        &a,
        &b,
        -1.0,
        &c_init,
        m,
        k,
        n,
    );
    for j in 0..n {
        for i in 0..m {
            let (got, exp) = (c[(i, j)], expected[(i, j)]);
            assert!(
                (got - exp).abs() <= 1e-9 * exp.abs().max(1.0),
                "({i},{j}): got {got} exp {exp}"
            );
        }
    }
}
