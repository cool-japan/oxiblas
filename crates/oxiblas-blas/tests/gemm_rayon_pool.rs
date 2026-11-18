//! A GEMM that the parallel gate keeps sequential must not build rayon's
//! global thread pool: a program may still configure the pool afterwards
//! (as with 0.2.3).
//!
//! This file holds a single test so that its binary runs nothing else that
//! could build the pool first (each integration-test file is its own binary).

#![cfg(feature = "parallel")]

use oxiblas_blas::level3::gemm::gemm_with_par;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

#[test]
fn test_gemm_gated_rayon_call_leaves_the_global_pool_unbuilt() {
    // 100³ multiply-adds: above every small-kernel ceiling (32³ by default,
    // 80³ under the x86 candidate policy's AVX2 small kernels) and outside
    // the candidates' copy range (8³..=32³), so the blocked driver runs, and
    // below the parallel gate (128³), so it runs sequentially.
    let n = 100;
    let a: Mat<f64> = Mat::filled(n, n, 0.5);
    let b: Mat<f64> = Mat::filled(n, n, 2.0);
    let mut c: Mat<f64> = Mat::zeros(n, n);
    gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
    // Every element is 100 · 0.5 · 2 = 100 (exact in f64).
    assert_eq!(c[(n - 1, n - 1)], 100.0);
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global();
    assert!(
        built.is_ok(),
        "a gated GEMM built rayon's global pool: {built:?}"
    );
}
