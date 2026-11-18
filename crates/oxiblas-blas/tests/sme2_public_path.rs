//! The public GEMM and GEMV paths with the `sme2` feature, built as a user
//! builds them: the library without `cfg(test)`.
//!
//! Each test makes public `gemm` / `gemv` calls large enough for the SME2
//! backend (past the blocked-route threshold of GEMM, `32³` multiply-adds,
//! and past the GEMV crossover), sequential and parallel, and compares the
//! results with a plain reference. The data are small integers and the
//! scalars powers of two, so every result is exact and the comparison is
//! equality: it holds for SME2 and NEON alike. In a debug build these calls
//! also run the `debug_assert` of the SME2 driver's split (a validated
//! layout runs no NEON block in an SME2 piece), which exists only in builds
//! without `cfg(test)`.
//!
//! # Gate and marker lines
//!
//! The library's test support (`test_support`) is not available here, so
//! the gate of every test follows its rules with the public query
//! `sme2_active()` standing for the backend:
//!
//! - `sme2_active()` is asked first: it waits for the settled verdict of
//!   the first-use checks, so the public calls after it see the backend as
//!   the gate saw it.
//! - `OXIBLAS_TEST_REQUIRE_SME2` set (to anything other than an empty
//!   string or `0`) and the backend inactive: the test fails.
//! - `OXIBLAS_TEST_EXPECT_NO_SME2` set and the backend active: the test
//!   fails.
//! - `OXIBLAS_TEST_SME2_MARKERS` set: the gate writes one line straight to
//!   the standard error handle (not captured by libtest),
//!   `sme2-gate: ran <test>` when the backend is active and
//!   `sme2-gate: skipped <test>` otherwise. (`sme2_active()` is the f32
//!   backend; the f64 tests' products use SME2 only with FEAT_SME_F64F64,
//!   which `tools/sme2_qemu.sh --hardware` checks from the detection report.)
//!
//! The results are checked in both cases: without the backend the same
//! calls run on NEON and must give the same exact values.

#![cfg(all(feature = "sme2", target_arch = "aarch64"))]

use oxiblas_blas::level2::{GemvTrans, gemv, gemv_with_par};
use oxiblas_blas::level3::gemm::{gemm, gemm_transposed_with_par, gemm_with_par};
use oxiblas_blas::level3::{Trans, sme2_active};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

/// Whether an environment variable is set to something other than an empty
/// string or `0`.
fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| !value.is_empty() && value != "0")
}

/// The gate of the module docs: whether the backend is active, after the
/// checks of the two environment switches and with the marker line.
fn gate(test: &str) -> bool {
    use std::io::Write;
    let active = sme2_active();
    if env_flag("OXIBLAS_TEST_REQUIRE_SME2") && !active {
        panic!("{test}: the SME2 backend is not active but OXIBLAS_TEST_REQUIRE_SME2 is set");
    }
    if env_flag("OXIBLAS_TEST_EXPECT_NO_SME2") && active {
        panic!("{test}: the SME2 backend is active but OXIBLAS_TEST_EXPECT_NO_SME2 is set");
    }
    if env_flag("OXIBLAS_TEST_SME2_MARKERS") {
        let word = if active { "ran" } else { "skipped" };
        let line = format!("sme2-gate: {word} {test}\n");
        let _ = std::io::stderr().lock().write_all(line.as_bytes());
    }
    active
}

/// A small integer in `-4..=4` for position `(i, j)` and `salt`.
fn small(i: usize, j: usize, salt: usize) -> f64 {
    ((i * 7 + j * 3 + salt) % 9) as f64 - 4.0
}

/// The `Par` values of the calls: sequential, and with the `parallel`
/// feature rayon's global pool.
fn pars() -> Vec<Par> {
    #[cfg(feature = "parallel")]
    {
        vec![Par::Seq, Par::Rayon]
    }
    #[cfg(not(feature = "parallel"))]
    {
        vec![Par::Seq]
    }
}

/// An `rows × cols` matrix of small integers.
fn matrix<T: From<i8> + oxiblas_core::scalar::Field + bytemuck::Zeroable>(
    rows: usize,
    cols: usize,
    salt: usize,
) -> Mat<T> {
    let mut mat = Mat::<T>::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            mat[(i, j)] = T::from(small(i, j, salt) as i8);
        }
    }
    mat
}

/// The exact `alpha · op(A) · op(B) + beta · C` in `f64`.
fn reference_gemm(
    (trans_a, trans_b): (bool, bool),
    alpha: f64,
    a: &dyn Fn(usize, usize) -> f64,
    b: &dyn Fn(usize, usize) -> f64,
    beta: f64,
    c: &dyn Fn(usize, usize) -> f64,
    (m, n, k): (usize, usize, usize),
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let x = if trans_a { a(p, i) } else { a(i, p) };
                let y = if trans_b { b(j, p) } else { b(p, j) };
                sum += x * y;
            }
            out[i + j * m] = alpha * sum + beta * c(i, j);
        }
    }
    out
}

/// One type's public GEMM calls: `gemm`, `gemm_with_par` (sequential and
/// parallel) and `gemm_transposed_with_par` with every transpose
/// combination.
fn gemm_calls<T>(test: &str, to_f64: fn(T) -> f64)
where
    T: From<i8> + oxiblas_core::scalar::Field + oxiblas_blas::level3::gemm_kernel::GemmKernel,
    T: bytemuck::Zeroable + Copy,
{
    let (m, n, k) = (150, 70, 90);
    let alpha = T::from(1) / T::from(2);
    let beta = T::from(1) / T::from(4);
    let pars = pars();
    let combos = [
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::Trans, Trans::Trans),
    ];
    for (index, (trans_a, trans_b)) in combos.into_iter().enumerate() {
        let (ta, tb) = (trans_a == Trans::Trans, trans_b == Trans::Trans);
        let a = if ta {
            matrix::<T>(k, m, 1)
        } else {
            matrix::<T>(m, k, 1)
        };
        let b = if tb {
            matrix::<T>(n, k, 2)
        } else {
            matrix::<T>(k, n, 2)
        };
        let c0 = matrix::<T>(m, n, 3);
        let want = reference_gemm(
            (ta, tb),
            0.5,
            &|i, j| to_f64(a[(i, j)]),
            &|i, j| to_f64(b[(i, j)]),
            0.25,
            &|i, j| to_f64(c0[(i, j)]),
            (m, n, k),
        );
        for &par in &pars {
            let mut runs: Vec<(&str, Mat<T>)> = Vec::new();
            let mut c = matrix::<T>(m, n, 3);
            gemm_transposed_with_par(
                trans_a,
                trans_b,
                alpha,
                a.as_ref(),
                b.as_ref(),
                beta,
                c.as_mut(),
                par,
            );
            runs.push(("gemm_transposed_with_par", c));
            if index == 0 {
                let mut c = matrix::<T>(m, n, 3);
                gemm_with_par(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut(), par);
                runs.push(("gemm_with_par", c));
                let mut c = matrix::<T>(m, n, 3);
                gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
                runs.push(("gemm", c));
            }
            for (name, c) in runs {
                for j in 0..n {
                    for i in 0..m {
                        assert_eq!(
                            to_f64(c[(i, j)]),
                            want[i + j * m],
                            "{test}: {name} {trans_a:?}/{trans_b:?} {par:?} C({i}, {j})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_sme2_public_path_gemm_f32() {
    let test = "test_sme2_public_path_gemm_f32";
    gate(test);
    gemm_calls::<f32>(test, f64::from);
}

#[test]
fn test_sme2_public_path_gemm_f64() {
    let test = "test_sme2_public_path_gemm_f64";
    gate(test);
    gemm_calls::<f64>(test, |x| x);
}

/// One type's public GEMV calls past the crossover: `gemv` and
/// `gemv_with_par` (sequential, and parallel on a matrix large enough for
/// its parallel path), every transpose.
fn gemv_calls<T>(test: &str, to_f64: fn(T) -> f64)
where
    T: From<i8> + oxiblas_core::scalar::Field + bytemuck::Zeroable + Copy + Send + Sync,
{
    let alpha = T::from(1) / T::from(2);
    let beta = T::from(1) / T::from(4);
    for (m, n) in [(200, 150), (300, 280)] {
        let a = matrix::<T>(m, n, 4);
        for trans in [GemvTrans::NoTrans, GemvTrans::Trans, GemvTrans::ConjTrans] {
            let (x_len, y_len) = match trans {
                GemvTrans::NoTrans => (n, m),
                GemvTrans::Trans | GemvTrans::ConjTrans => (m, n),
            };
            let x: Vec<T> = (0..x_len).map(|p| T::from(small(p, 0, 5) as i8)).collect();
            let y0: Vec<T> = (0..y_len).map(|i| T::from(small(i, 1, 6) as i8)).collect();
            let want: Vec<f64> = (0..y_len)
                .map(|i| {
                    let dot: f64 = (0..x_len)
                        .map(|p| {
                            let entry = match trans {
                                GemvTrans::NoTrans => a[(i, p)],
                                GemvTrans::Trans | GemvTrans::ConjTrans => a[(p, i)],
                            };
                            to_f64(entry) * to_f64(x[p])
                        })
                        .sum();
                    0.5 * dot + 0.25 * to_f64(y0[i])
                })
                .collect();
            let mut results = Vec::new();
            let mut y = y0.clone();
            gemv(trans, alpha, a.as_ref(), &x, beta, &mut y);
            results.push(("gemv", y));
            for par in pars() {
                let mut y = y0.clone();
                gemv_with_par(trans, alpha, a.as_ref(), &x, beta, &mut y, par);
                results.push(("gemv_with_par", y));
            }
            for (name, y) in results {
                for (i, (&got, &expected)) in y.iter().zip(&want).enumerate() {
                    assert_eq!(
                        to_f64(got),
                        expected,
                        "{test}: {name} {trans:?} {m}x{n} y[{i}]"
                    );
                }
            }
        }
    }
}

#[test]
fn test_sme2_public_path_gemv_f32() {
    let test = "test_sme2_public_path_gemv_f32";
    gate(test);
    gemv_calls::<f32>(test, f64::from);
}

#[test]
fn test_sme2_public_path_gemv_f64() {
    let test = "test_sme2_public_path_gemv_f64";
    gate(test);
    gemv_calls::<f64>(test, |x| x);
}
