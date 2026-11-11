//! Tests for the GEMV SIMD kernels and the parallel drivers.
//!
//! * Integer-valued data (every product and partial sum exact) must give the
//!   naive result bit for bit for every shape, tail, transpose, view, alpha and
//!   beta, on every target and feature set.
//! * On aarch64 the NEON kernels are checked against scalar emulations of their
//!   documented per-element arithmetic, bit for bit.
//! * Random data are checked against a compensated reference with the
//!   `γ_k·|alpha|·Σ|A_ij·x_j|` bound.
//! * Parallel results equal sequential ones bit for bit (same per-element
//!   arithmetic), including the generic complex Trans/ConjTrans paths.

use super::*;
use oxiblas_matrix::Mat;

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// Small integer in `[-k, k]`.
    fn int(&mut self, k: i64) -> f64 {
        ((self.next_u64() % (2 * k as u64 + 1)) as i64 - k) as f64
    }
}

const TRANSES: [GemvTrans; 3] = [GemvTrans::NoTrans, GemvTrans::Trans, GemvTrans::ConjTrans];

/// Shapes covering every vector-step / tail combination of both kernels
/// (8/16-row steps, 2/4-row steps, scalar rows, 4-column groups, column tail).
const DIMS: [usize; 16] = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 33, 40, 67];

/// `op(A)` dimensions `(rows, cols)` of a stored `m x n` matrix.
fn op_dims(trans: GemvTrans, m: usize, n: usize) -> (usize, usize) {
    match trans {
        GemvTrans::NoTrans => (m, n),
        GemvTrans::Trans | GemvTrans::ConjTrans => (n, m),
    }
}

/// Naive `beta·y + alpha·op(A)·x` in plain f64 arithmetic (exact for the
/// integer data used with it); `beta == 0` overwrites `y`.
fn naive(
    trans: GemvTrans,
    alpha: f64,
    a: MatRef<'_, f64>,
    x: &[f64],
    beta: f64,
    y0: &[f64],
) -> Vec<f64> {
    let (rows, cols) = op_dims(trans, a.nrows(), a.ncols());
    assert_eq!(x.len(), cols);
    (0..rows)
        .map(|r| {
            let mut s = 0.0;
            for (c, &xc) in x.iter().enumerate() {
                let aij = match trans {
                    GemvTrans::NoTrans => a[(r, c)],
                    GemvTrans::Trans | GemvTrans::ConjTrans => a[(c, r)],
                };
                s += aij * xc;
            }
            let base = if beta == 0.0 { 0.0 } else { beta * y0[r] };
            base + alpha * s
        })
        .collect()
}

/// Every transpose / shape / view / alpha / beta with exact integer data:
/// `gemv` and `gemv_with_par(Seq)` must equal the naive result bit for bit,
/// and `beta == 0` must discard NaN in `y`.
#[test]
fn test_gemv_integer_data_exact_all_shapes_f64() {
    let mut rng = Rng(0x1234_5678);
    for &m in &DIMS {
        for &n in &DIMS {
            // A column-major submatrix with lda > m and a plain matrix.
            let mut big: Mat<f64> = Mat::zeros(m + 5, n + 2);
            for j in 0..n + 2 {
                for i in 0..m + 5 {
                    big[(i, j)] = rng.int(9);
                }
            }
            let views = [
                big.as_ref().submatrix(3, 1, m, n),
                big.as_ref().submatrix(0, 0, m, n),
            ];
            for a in views {
                for trans in TRANSES {
                    let (rows, cols) = op_dims(trans, m, n);
                    let x: Vec<f64> = (0..cols).map(|_| rng.int(7)).collect();
                    let y0: Vec<f64> = (0..rows).map(|_| rng.int(5)).collect();
                    for (alpha, beta) in
                        [(1.0, 0.0), (1.5, -0.5), (-2.0, 1.0), (0.0, 2.0), (1.0, 1.0)]
                    {
                        let mut y = if beta == 0.0 {
                            vec![f64::NAN; rows]
                        } else {
                            y0.clone()
                        };
                        let expected = naive(trans, alpha, a, &x, beta, &y0);
                        gemv(trans, alpha, a, &x, beta, &mut y);
                        assert_eq!(
                            y, expected,
                            "gemv {trans:?} {m}x{n} alpha={alpha} beta={beta}"
                        );
                        let mut y = if beta == 0.0 {
                            vec![f64::NAN; rows]
                        } else {
                            y0.clone()
                        };
                        gemv_with_par(trans, alpha, a, &x, beta, &mut y, Par::Seq);
                        assert_eq!(y, expected, "gemv_with_par {trans:?} {m}x{n}");
                    }
                }
            }
        }
    }
}

/// f32 counterpart of the exact integer test (results compared as f32).
#[test]
fn test_gemv_integer_data_exact_all_shapes_f32() {
    let mut rng = Rng(0x8765_4321);
    for &m in &DIMS {
        for &n in &DIMS {
            let mut big: Mat<f32> = Mat::zeros(m + 3, n + 1);
            for j in 0..n + 1 {
                for i in 0..m + 3 {
                    big[(i, j)] = rng.int(9) as f32;
                }
            }
            let a = big.as_ref().submatrix(2, 1, m, n);
            let mut a64: Mat<f64> = Mat::zeros(m, n);
            for j in 0..n {
                for i in 0..m {
                    a64[(i, j)] = f64::from(a[(i, j)]);
                }
            }
            for trans in TRANSES {
                let (rows, cols) = op_dims(trans, m, n);
                let x: Vec<f32> = (0..cols).map(|_| rng.int(7) as f32).collect();
                let y0: Vec<f32> = (0..rows).map(|_| rng.int(5) as f32).collect();
                let xd: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
                let yd: Vec<f64> = y0.iter().map(|&v| f64::from(v)).collect();
                for (alpha, beta) in [(1.0f32, 0.0f32), (1.5, -0.5), (-2.0, 1.0)] {
                    let expected: Vec<f32> = naive(
                        trans,
                        f64::from(alpha),
                        a64.as_ref(),
                        &xd,
                        f64::from(beta),
                        &yd,
                    )
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                    let mut y = if beta == 0.0 {
                        vec![f32::NAN; rows]
                    } else {
                        y0.clone()
                    };
                    gemv(trans, alpha, a, &x, beta, &mut y);
                    assert_eq!(
                        y, expected,
                        "gemv f32 {trans:?} {m}x{n} alpha={alpha} beta={beta}"
                    );
                }
            }
        }
    }
}

/// Very tall and very wide matrices (long column streams, many column groups)
/// stay exact.
#[test]
fn test_gemv_integer_data_exact_tall_matrices() {
    let mut rng = Rng(99);
    for (m, n) in [(4096 + 37, 9), (8192 + 21, 6), (3, 9000)] {
        let mut a: Mat<f64> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a[(i, j)] = rng.int(3);
            }
        }
        for trans in TRANSES {
            let (rows, cols) = op_dims(trans, m, n);
            let x: Vec<f64> = (0..cols).map(|_| rng.int(3)).collect();
            let y0: Vec<f64> = (0..rows).map(|_| rng.int(3)).collect();
            let expected = naive(trans, 1.0, a.as_ref(), &x, 1.0, &y0);
            let mut y = y0.clone();
            gemv(trans, 1.0, a.as_ref(), &x, 1.0, &mut y);
            assert_eq!(y, expected, "{trans:?} {m}x{n}");
            let a32: Mat<f32> = {
                let mut t: Mat<f32> = Mat::zeros(m, n);
                for j in 0..n {
                    for i in 0..m {
                        t[(i, j)] = a[(i, j)] as f32;
                    }
                }
                t
            };
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let mut y32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
            gemv(trans, 1.0f32, a32.as_ref(), &x32, 1.0f32, &mut y32);
            let e32: Vec<f32> = expected.iter().map(|&v| v as f32).collect();
            assert_eq!(y32, e32, "f32 {trans:?} {m}x{n}");
        }
    }
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, (a - (s - z)) + (b - z))
}

/// Compensated (`Dot2`) dot product: accurate to about `u·|result|`.
fn dot2(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut p, mut s) = (0.0f64, 0.0f64);
    for (a, b) in pairs {
        let h = a * b;
        let r = a.mul_add(b, -h);
        let (q, e) = two_sum(p, h);
        p = q;
        s += e + r;
    }
    p + s
}

/// Random data: every `y` element is within
/// `γ_{k+2}·(|alpha|·Σ|A·x| + |beta·y0|)` of a compensated reference, with
/// `k` the inner dimension (any summation order meets this bound; the alpha
/// pre-scaling, the beta product and the final addition add two roundings).
#[test]
fn test_gemv_random_data_error_bound() {
    let mut rng = Rng(0xFEED_BEEF);
    let u = f64::EPSILON / 2.0;
    let u32_ = f64::from(f32::EPSILON) / 2.0;
    for (m, n) in [
        (1, 1),
        (5, 300),
        (300, 5),
        (129, 131),
        (1000, 64),
        (64, 1000),
    ] {
        let mut a: Mat<f64> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a[(i, j)] = rng.sym();
            }
        }
        let mut a32: Mat<f32> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a32[(i, j)] = a[(i, j)] as f32;
            }
        }
        for trans in TRANSES {
            let (rows, cols) = op_dims(trans, m, n);
            let x: Vec<f64> = (0..cols).map(|_| rng.sym()).collect();
            let y0: Vec<f64> = (0..rows).map(|_| rng.sym()).collect();
            let (alpha, beta) = (0.75, -1.25);
            let mut y = y0.clone();
            gemv(trans, alpha, a.as_ref(), &x, beta, &mut y);
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let y032: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
            let mut y32 = y032.clone();
            gemv(
                trans,
                alpha as f32,
                a32.as_ref(),
                &x32,
                beta as f32,
                &mut y32,
            );
            let rows_iter = y.iter().zip(&y0).zip(y32.iter().zip(&y032)).enumerate();
            for (r, ((&yr, &y0r), (&y32r, &y032r))) in rows_iter {
                let elem = |c: usize| match trans {
                    GemvTrans::NoTrans => (r, c),
                    GemvTrans::Trans | GemvTrans::ConjTrans => (c, r),
                };
                let dot = dot2(x.iter().enumerate().map(|(c, &xc)| (a[elem(c)], xc)));
                let exact = beta * y0r + alpha * dot;
                let mag: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(c, &xc)| (a[elem(c)] * xc).abs())
                    .sum::<f64>()
                    * alpha.abs()
                    + (beta * y0r).abs();
                let k = (cols + 2) as f64;
                let tol = k * u / (1.0 - k * u) * mag + u * exact.abs();
                assert!(
                    (yr - exact).abs() <= tol,
                    "f64 {trans:?} {m}x{n} row {r}: {yr} vs {exact}"
                );

                let pairs32 = x32
                    .iter()
                    .enumerate()
                    .map(|(c, &xc)| (f64::from(a32[elem(c)]), f64::from(xc)));
                let dot32 = dot2(pairs32.clone());
                let exact32 =
                    f64::from(beta as f32) * f64::from(y032r) + f64::from(alpha as f32) * dot32;
                let mag32: f64 = pairs32.map(|(p, q)| (p * q).abs()).sum::<f64>() * alpha.abs()
                    + (beta * f64::from(y032r)).abs();
                let tol32 = k * u32_ / (1.0 - k * u32_) * mag32 + u32_ * exact32.abs();
                assert!(
                    (f64::from(y32r) - exact32).abs() <= tol32,
                    "f32 {trans:?} {m}x{n} row {r}: {y32r} vs {exact32}"
                );
            }
        }
    }
}

/// Inf/NaN in A or x propagate through the kernels exactly like the reference
/// (no zero-skipping in either direction).
#[test]
fn test_gemv_kernels_propagate_nonfinite() {
    for (m, n) in [(9, 7), (40, 33)] {
        for trans in TRANSES {
            let (rows, cols) = op_dims(trans, m, n);
            let mut a: Mat<f64> = Mat::filled(m, n, 1.0);
            // A[2, 3] = Inf with the matching x entry zero: 0·Inf = NaN.
            a[(2, 3)] = f64::INFINITY;
            let mut x = vec![1.0f64; cols];
            let (row_hit, col_zero) = match trans {
                GemvTrans::NoTrans => (2, 3),
                GemvTrans::Trans | GemvTrans::ConjTrans => (3, 2),
            };
            x[col_zero] = 0.0;
            let mut y = vec![0.0f64; rows];
            gemv(trans, 1.0, a.as_ref(), &x, 0.0, &mut y);
            assert!(y[row_hit].is_nan(), "{trans:?} {m}x{n}: {}", y[row_hit]);
            for (r, v) in y.iter().enumerate() {
                if r != row_hit {
                    assert!(v.is_finite(), "{trans:?} {m}x{n} row {r}: {v}");
                }
            }
        }
    }
}

/// Scalar emulation of the NEON NoTrans arithmetic: `y[i] = fma(alpha·x[j],
/// A[i,j], y[i])` for `j = 0..n` in order, after the beta step.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
fn notrans_emulation_f64(
    alpha: f64,
    a: MatRef<'_, f64>,
    x: &[f64],
    beta: f64,
    y0: &[f64],
) -> Vec<f64> {
    (0..a.nrows())
        .map(|i| {
            let mut s = if beta == 0.0 {
                0.0
            } else if beta == 1.0 {
                y0[i]
            } else {
                beta * y0[i]
            };
            for (j, &xj) in x.iter().enumerate() {
                s = a[(i, j)].mul_add(alpha * xj, s);
            }
            s
        })
        .collect()
}

/// Scalar emulation of the NEON Trans arithmetic for one f64 column: four
/// two-lane accumulators over 8-row steps, the 4/2/1-row tail, the pairwise
/// reduction, then `y += alpha·dot`.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
fn trans_column_emulation_f64(col: &[f64], x: &[f64]) -> f64 {
    let m = col.len();
    let mut acc = [[0.0f64; 2]; 4];
    let body = m - m % 8;
    let mut i = 0;
    while i < body {
        for (k, lanes) in acc.iter_mut().enumerate() {
            for (l, lane) in lanes.iter_mut().enumerate() {
                let r = i + 2 * k + l;
                *lane = col[r].mul_add(x[r], *lane);
            }
        }
        i += 8;
    }
    if m - i >= 4 {
        for (k, lanes) in acc.iter_mut().take(2).enumerate() {
            for (l, lane) in lanes.iter_mut().enumerate() {
                let r = i + 2 * k + l;
                *lane = col[r].mul_add(x[r], *lane);
            }
        }
        i += 4;
    }
    if m - i >= 2 {
        for (l, lane) in acc[2].iter_mut().enumerate() {
            *lane = col[i + l].mul_add(x[i + l], *lane);
        }
        i += 2;
    }
    let lane = |l: usize| (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    let mut s = lane(0) + lane(1);
    if i < m {
        s = col[i].mul_add(x[i], s);
    }
    s
}

/// f32 counterpart of [`trans_column_emulation_f64`] (four-lane accumulators,
/// 16-row steps, 8/4-row and scalar tails; the lane sum is pairwise).
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
fn trans_column_emulation_f32(col: &[f32], x: &[f32]) -> f32 {
    let m = col.len();
    let mut acc = [[0.0f32; 4]; 4];
    let body = m - m % 16;
    let mut i = 0;
    while i < body {
        for (k, lanes) in acc.iter_mut().enumerate() {
            for (l, lane) in lanes.iter_mut().enumerate() {
                let r = i + 4 * k + l;
                *lane = col[r].mul_add(x[r], *lane);
            }
        }
        i += 16;
    }
    if m - i >= 8 {
        for (k, lanes) in acc.iter_mut().take(2).enumerate() {
            for (l, lane) in lanes.iter_mut().enumerate() {
                let r = i + 4 * k + l;
                *lane = col[r].mul_add(x[r], *lane);
            }
        }
        i += 8;
    }
    if m - i >= 4 {
        for (l, lane) in acc[2].iter_mut().enumerate() {
            *lane = col[i + l].mul_add(x[i + l], *lane);
        }
        i += 4;
    }
    let v: Vec<f32> = (0..4)
        .map(|l| (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]))
        .collect();
    let mut s = (v[0] + v[1]) + (v[2] + v[3]);
    while i < m {
        s = col[i].mul_add(x[i], s);
        i += 1;
    }
    s
}

/// The NEON kernels reproduce their documented arithmetic bit for bit on
/// random data, for every row/column tail and for tall and wide shapes.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[test]
fn test_gemv_neon_kernels_match_scalar_emulation() {
    let mut rng = Rng(0xABCD_EF01);
    let mut shapes: Vec<(usize, usize)> = DIMS
        .iter()
        .flat_map(|&m| DIMS.iter().map(move |&n| (m, n)))
        .collect();
    shapes.extend([(4096 + 17, 5), (8192 + 3, 3), (1001, 1003)]);
    for (m, n) in shapes {
        let mut a: Mat<f64> = Mat::zeros(m, n);
        let mut a32: Mat<f32> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a[(i, j)] = rng.sym();
                a32[(i, j)] = rng.sym() as f32;
            }
        }
        let (alpha, beta) = (1.25, 0.5);

        // NoTrans.
        let x: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let y0: Vec<f64> = (0..m).map(|_| rng.sym()).collect();
        let mut y = y0.clone();
        gemv(GemvTrans::NoTrans, alpha, a.as_ref(), &x, beta, &mut y);
        assert_eq!(
            y,
            notrans_emulation_f64(alpha, a.as_ref(), &x, beta, &y0),
            "NoTrans {m}x{n}"
        );

        // Trans (f64 and f32).
        let xt: Vec<f64> = (0..m).map(|_| rng.sym()).collect();
        let yt0: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let mut yt = yt0.clone();
        gemv(GemvTrans::Trans, alpha, a.as_ref(), &xt, beta, &mut yt);
        let xt32: Vec<f32> = xt.iter().map(|&v| v as f32).collect();
        let yt032: Vec<f32> = yt0.iter().map(|&v| v as f32).collect();
        let mut yt32 = yt032.clone();
        gemv(
            GemvTrans::ConjTrans,
            alpha as f32,
            a32.as_ref(),
            &xt32,
            beta as f32,
            &mut yt32,
        );
        let cols_iter = yt.iter().zip(&yt0).zip(yt32.iter().zip(&yt032)).enumerate();
        for (j, ((&got, &y0j), (&got32, &y032j))) in cols_iter {
            let col: Vec<f64> = (0..m).map(|i| a[(i, j)]).collect();
            let want = beta * y0j + alpha * trans_column_emulation_f64(&col, &xt);
            assert_eq!(got.to_bits(), want.to_bits(), "Trans {m}x{n} col {j}");
            let col32: Vec<f32> = (0..m).map(|i| a32[(i, j)]).collect();
            let want32 =
                beta as f32 * y032j + alpha as f32 * trans_column_emulation_f32(&col32, &xt32);
            assert_eq!(
                got32.to_bits(),
                want32.to_bits(),
                "Trans f32 {m}x{n} col {j}"
            );
        }
    }
}

/// Parallel GEMV equals the sequential result bit for bit wherever both use
/// the same per-element arithmetic: the aarch64 f64/f32 kernels (rows / columns
/// split between threads) and the portable `NoTrans` loops (complex,
/// column-major and transposed views), for several thread counts. Where the
/// arithmetic differs (the per-thread Trans/ConjTrans reduction, the AVX2 f64
/// `NoTrans` kernel) the results must agree to rounding.
#[cfg(feature = "parallel")]
#[test]
fn test_gemv_parallel_matches_sequential_bitwise() {
    use num_complex::Complex64;

    let mut rng = Rng(0x5151_5151);
    for (m, n) in [
        (300usize, 301usize),
        (517, 129),
        (129, 517),
        (1000, 70),
        (70, 1000),
    ] {
        let mut a: Mat<f64> = Mat::zeros(m, n);
        let mut a32: Mat<f32> = Mat::zeros(m, n);
        let mut ac: Mat<Complex64> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a[(i, j)] = rng.sym();
                a32[(i, j)] = rng.sym() as f32;
                ac[(i, j)] = Complex64::new(rng.sym(), rng.sym());
            }
        }
        for trans in TRANSES {
            let (rows, cols) = op_dims(trans, m, n);
            let x: Vec<f64> = (0..cols).map(|_| rng.sym()).collect();
            let y0: Vec<f64> = (0..rows).map(|_| rng.sym()).collect();
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let y032: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
            let xc: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, -0.5 * v)).collect();
            let yc0: Vec<Complex64> = y0.iter().map(|&v| Complex64::new(0.25 * v, v)).collect();
            let (alpha, beta) = (0.7, -1.1);
            let (alphac, betac) = (Complex64::new(0.7, 0.2), Complex64::new(-1.1, 0.3));

            let mut seq = y0.clone();
            gemv(trans, alpha, a.as_ref(), &x, beta, &mut seq);
            let mut seq32 = y032.clone();
            gemv(
                trans,
                alpha as f32,
                a32.as_ref(),
                &x32,
                beta as f32,
                &mut seq32,
            );
            let mut seqc = yc0.clone();
            gemv(trans, alphac, ac.as_ref(), &xc, betac, &mut seqc);
            // Transposed complex view: op dims of a.transpose() are swapped.
            let tr_trans = match trans {
                GemvTrans::NoTrans => GemvTrans::Trans,
                GemvTrans::Trans | GemvTrans::ConjTrans => GemvTrans::NoTrans,
            };
            let act = ac.as_ref().transpose();
            let mut seqt = yc0.clone();
            gemv(tr_trans, alphac, act, &xc, betac, &mut seqt);

            let notrans = trans == GemvTrans::NoTrans;
            let f64_bitwise = if notrans {
                f64_notrans_shares_arithmetic()
            } else {
                neon_parallel_trans()
            };
            let f32_bitwise = notrans || neon_parallel_trans();
            let c64_bitwise = notrans;
            let tr_bitwise = tr_trans == GemvTrans::NoTrans;
            let f64_close = |p: &f64, q: &f64| (p - q).abs() <= 1e-12 * q.abs().max(1.0);
            let f32_close = |p: &f32, q: &f32| (p - q).abs() <= 1e-4 * q.abs().max(1.0);
            let c64_close =
                |p: &Complex64, q: &Complex64| (p - q).norm() <= 1e-12 * q.norm().max(1.0);

            for threads in [2usize, 3, 4, 7] {
                let par = Par::RayonWith(threads);
                let what = format!("{trans:?} {m}x{n} threads={threads}");
                let mut y = y0.clone();
                gemv_with_par(trans, alpha, a.as_ref(), &x, beta, &mut y, par);
                assert_par_matches(&y, &seq, f64_bitwise, f64_close, &format!("f64 {what}"));
                let mut y = y032.clone();
                gemv_with_par(
                    trans,
                    alpha as f32,
                    a32.as_ref(),
                    &x32,
                    beta as f32,
                    &mut y,
                    par,
                );
                assert_par_matches(&y, &seq32, f32_bitwise, f32_close, &format!("f32 {what}"));
                let mut y = yc0.clone();
                gemv_with_par(trans, alphac, ac.as_ref(), &xc, betac, &mut y, par);
                assert_par_matches(&y, &seqc, c64_bitwise, c64_close, &format!("c64 {what}"));
                let mut y = yc0.clone();
                gemv_with_par(tr_trans, alphac, act, &xc, betac, &mut y, par);
                assert_par_matches(
                    &y,
                    &seqt,
                    tr_bitwise,
                    c64_close,
                    &format!("c64 transposed view {tr_trans:?} {what}"),
                );
            }
        }
    }
}

/// Whether the parallel f64/f32 Trans/ConjTrans driver reproduces the
/// sequential per-element arithmetic: on aarch64 (without `force-scalar`) it
/// runs the NEON kernel on disjoint column ranges; the portable driver keeps
/// per-thread partial sums with a final reduction.
#[cfg(feature = "parallel")]
fn neon_parallel_trans() -> bool {
    cfg!(all(target_arch = "aarch64", not(feature = "force-scalar")))
}

/// `got == want` bit for bit when `bitwise`, otherwise elementwise `close`.
#[cfg(feature = "parallel")]
fn assert_par_matches<T: PartialEq + core::fmt::Debug>(
    got: &[T],
    want: &[T],
    bitwise: bool,
    close: impl Fn(&T, &T) -> bool,
    what: &str,
) {
    if bitwise {
        assert_eq!(got, want, "{what}");
    } else {
        assert_eq!(got.len(), want.len(), "{what}");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(close(g, w), "{what} [{k}]: {g:?} vs {w:?}");
        }
    }
}

/// Whether the sequential and the parallel f64 `NoTrans` drivers share their
/// per-element arithmetic. They do everywhere except on `x86_64` with AVX2 and
/// FMA, where the sequential driver runs the AVX2 kernel and the parallel one
/// the portable loop (behaviour that predates the NEON kernels).
#[cfg(feature = "parallel")]
fn f64_notrans_shares_arithmetic() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// The private parallel drivers handle small and uneven splits (more threads
/// than rows/columns, partial vector steps at every boundary).
#[cfg(feature = "parallel")]
#[test]
fn test_gemv_parallel_small_and_uneven_splits() {
    let mut rng = Rng(77);
    for (m, n) in [(1usize, 1usize), (3, 5), (9, 17), (33, 4), (4, 33)] {
        let mut a: Mat<f64> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                a[(i, j)] = rng.sym();
            }
        }
        for trans in TRANSES {
            let (rows, cols) = op_dims(trans, m, n);
            let x: Vec<f64> = (0..cols).map(|_| rng.sym()).collect();
            let y0: Vec<f64> = (0..rows).map(|_| rng.sym()).collect();
            let mut seq = y0.clone();
            gemv(trans, 1.5, a.as_ref(), &x, 0.5, &mut seq);
            let bitwise = if trans == GemvTrans::NoTrans {
                true
            } else {
                neon_parallel_trans()
            };
            let close = |p: &f64, q: &f64| (p - q).abs() <= 1e-12 * q.abs().max(1.0);
            for threads in [2usize, 5, 64] {
                let mut y = y0.clone();
                gemv_parallel(
                    trans,
                    1.5,
                    a.as_ref(),
                    &x,
                    0.5,
                    &mut y,
                    Par::RayonWith(threads),
                );
                let what = format!("{trans:?} {m}x{n} threads={threads}");
                assert_par_matches(&y, &seq, bitwise, close, &what);
            }
        }
    }
}
