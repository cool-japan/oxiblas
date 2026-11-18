//! The x86-64 blocked route of the public GEMM, through the plan, on the
//! AVX2 + FMA tier.
//!
//! The default blocking of a call reads the cache description of the CPU
//! (`autotune::get_cache_info`), which on x86-64 runs the `cpuid`
//! instruction, so Miri cannot run a public call with the default blocking.
//! A call with explicit blocking (`gemm_transposed_with_blocking`,
//! `BlockingSource::Explicit`) never asks for that description: its targets
//! are the caller's, fitted to the problem by arithmetic, the B-block cap is
//! off, and the sequential driver, the packers and the micro-kernels read no
//! cache description. These tests make such a call on a product just above
//! the small threshold, which takes the blocked route, and check:
//! - the result, exactly, against a plain triple loop (small integers, so
//!   every product and partial sum is exact whatever the kernel's order);
//! - that the call resolved exactly one micro-kernel, and that it was the
//!   AVX2 kernel of the type (its register-tile shape, which differs from
//!   the shape of every other kernel of the type).
//!
//! On any other tier they print `SKIPPED` (or fail, when
//! `OXIBLAS_REQUIRE_AVX2` is set; see [`avx2_fma_tier_or_skip`]). Under
//! Miri on x86-64 they run with
//! `RUSTFLAGS=-Ctarget-feature=+avx2,+fma OXIBLAS_REQUIRE_AVX2=1
//! MIRIFLAGS=-Zmiri-env-forward=OXIBLAS_REQUIRE_AVX2`.

use super::call_probe::{last_resolved, snapshot};
use super::{
    DEFAULT_POLICY, F32Avx2Policy, GemmRoute, RouteInputs, SimdTier, avx2_fma_tier_or_skip,
    choose_route,
};
use crate::level3::gemm::{GemmBlocking, gemm_transposed_with_blocking};
use crate::level3::gemm_kernel::{GemmKernel, MicroKernelShape, f32_avx2_picks_16x6};
use crate::level3::trsm::Trans;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;

/// `op(A)` is `m × k`, `op(B)` is `k × n`: 33 · 32 · 32 = 33 792 multiply-adds,
/// just above the small threshold of 32³.
const DIMS: (usize, usize, usize) = (33, 32, 32);

/// A small integer (−3 to 3) as `T`.
fn small<T: Field>(i: usize, j: usize, salt: usize) -> T {
    let x = ((i * 5 + j * 3 + salt) % 7) as i32 - 3;
    let mut v = T::zero();
    for _ in 0..x.unsigned_abs() {
        v = if x < 0 { v - T::one() } else { v + T::one() };
    }
    v
}

/// One public call `C = 2·A·Bᵀ + 3·C` with explicit blocking on the
/// AVX2 + FMA tier; returns the shape of the micro-kernel the call resolved
/// after checking the route, the result and the number of resolutions.
fn explicit_blocked_call<T: GemmKernel + bytemuck::Zeroable + core::fmt::Debug>(
    test: &str,
    tile: MicroKernelShape,
) -> Option<MicroKernelShape> {
    let (m, k, n) = DIMS;
    let (ta, tb) = (Trans::NoTrans, Trans::Trans);
    let mut a: Mat<T> = Mat::zeros(m, k);
    let mut b: Mat<T> = Mat::zeros(n, k);
    let mut c: Mat<T> = Mat::zeros(m, n);
    for j in 0..k {
        for i in 0..m {
            a[(i, j)] = small(i, j, 1);
        }
        for i in 0..n {
            b[(i, j)] = small(i, j, 4);
        }
    }
    for j in 0..n {
        for i in 0..m {
            c[(i, j)] = small(i, j, 2);
        }
    }
    let two = T::one() + T::one();
    let three = two + T::one();
    let mut expected = c.clone();
    for j in 0..n {
        for i in 0..m {
            let mut dot = T::zero();
            for p in 0..k {
                dot = dot + a[(i, p)] * b[(j, p)];
            }
            expected[(i, j)] = two * dot + three * c[(i, j)];
        }
    }

    // The default policy sends this call to the blocked route.
    let inp = RouteInputs::new(
        ta,
        tb,
        &a.as_ref(),
        &b.as_ref(),
        m,
        n,
        k,
        SimdTier::current(),
    );
    assert_eq!(
        choose_route(&inp, &DEFAULT_POLICY),
        GemmRoute::Blocked,
        "{test}: {m}x{k}x{n} NoTrans/Trans must take the blocked route"
    );

    // Small blocks, so the call has several MC, KC and NC blocks and edge
    // tiles in both directions.
    let blocking = GemmBlocking::custom(2 * tile.mr, 8, 2 * tile.nr, &tile);
    let before = snapshot();
    gemm_transposed_with_blocking(
        ta,
        tb,
        two,
        a.as_ref(),
        b.as_ref(),
        three,
        c.as_mut(),
        Par::Seq,
        &blocking,
    );
    let counts = snapshot().since(before);
    assert_eq!(
        counts.kernel_resolutions, 1,
        "{test}: the blocked call must resolve its micro-kernel exactly once"
    );
    for j in 0..n {
        for i in 0..m {
            assert_eq!(
                c[(i, j)],
                expected[(i, j)],
                "{test}: C({i}, {j}) differs from the triple loop"
            );
        }
    }
    last_resolved()
}

/// The f64 call runs the 8×6 AVX2 kernel (the SSE4.2 kernel is 4×2, the
/// portable one 4×4, the AVX-512 one 16×6).
#[test]
fn test_gemm_plan_explicit_blocked_call_runs_avx2_kernel_f64() {
    let test = "test_gemm_plan_explicit_blocked_call_runs_avx2_kernel_f64";
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let avx2 = MicroKernelShape { mr: 8, nr: 6 };
    for other in [
        MicroKernelShape { mr: 4, nr: 2 },
        MicroKernelShape { mr: 4, nr: 4 },
        MicroKernelShape { mr: 16, nr: 6 },
    ] {
        assert_ne!(avx2, other);
    }
    assert_eq!(explicit_blocked_call::<f64>(test, avx2), Some(avx2));
}

/// The f32 call runs an AVX2 kernel: the 8×8 kernel, or the 16×6 kernel
/// where the policy picks it (the SSE4.2 and portable kernels are 4×4, the
/// AVX-512 one 16×16).
#[test]
fn test_gemm_plan_explicit_blocked_call_runs_avx2_kernel_f32() {
    let test = "test_gemm_plan_explicit_blocked_call_runs_avx2_kernel_f32";
    if !avx2_fma_tier_or_skip(test) {
        return;
    }
    let (m, k, n) = DIMS;
    let canonical = MicroKernelShape { mr: 8, nr: 8 };
    let avx2 = if f32_avx2_picks_16x6(
        SimdTier::Avx2Fma,
        DEFAULT_POLICY.f32_avx2,
        canonical,
        m,
        n,
        k,
    ) {
        MicroKernelShape { mr: 16, nr: 6 }
    } else {
        canonical
    };
    if DEFAULT_POLICY.f32_avx2 == F32Avx2Policy::Legacy8x8 {
        assert_eq!(avx2, canonical);
    }
    for other in [
        MicroKernelShape { mr: 4, nr: 4 },
        MicroKernelShape { mr: 16, nr: 16 },
    ] {
        assert_ne!(avx2, other);
    }
    assert_eq!(explicit_blocked_call::<f32>(test, avx2), Some(avx2));
}
