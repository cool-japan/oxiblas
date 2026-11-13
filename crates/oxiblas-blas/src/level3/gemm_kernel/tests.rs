//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use num_complex::{Complex32, Complex64};
use oxiblas_core::simd::dispatch::GemmKernelKind;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // ------------------------------------------------------------------
    // Shape sanity: minimum register-block dimensions.
    //
    // NOTE: f64's SSE4.2 kernel is 4×2, so its NR minimum is 2 (the previous
    // `nr >= 4` assertion only ever passed because the dev CPU happened to have
    // AVX2 — it would have failed on an SSE4.2-only host).
    // ------------------------------------------------------------------
    #[test]
    fn test_micro_kernel_shape() {
        let shape_f64 = f64::micro_kernel_shape();
        assert!(shape_f64.mr >= 4);
        assert!(shape_f64.nr >= 2);

        let shape_f32 = f32::micro_kernel_shape();
        assert!(shape_f32.mr >= 4);
        assert!(shape_f32.nr >= 4);
    }

    // ------------------------------------------------------------------
    // Finding 1 (regression): the packed-panel shape MUST correspond to the
    // kernel that dispatch actually runs. Encoding the exact (mr, nr) expected
    // for each `GemmKernelKind` here means that if the shape selector and the
    // dispatch selector ever diverge again (the AVX2-without-FMA garbage bug),
    // this test fails loudly instead of silently corrupting GEMM output.
    // ------------------------------------------------------------------
    #[test]
    fn test_shape_matches_selected_kernel_kind() {
        let kind = selected_kernel_kind();
        let (expected_f64, expected_f32): ((usize, usize), (usize, usize)) = match kind {
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx512 => ((16, 6), (16, 16)),
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Avx2 => ((8, 6), (8, 8)),
            #[cfg(target_arch = "x86_64")]
            GemmKernelKind::Sse42 => ((4, 2), (4, 4)),
            #[cfg(target_arch = "aarch64")]
            GemmKernelKind::Neon => ((8, 6), (8, 8)),
            GemmKernelKind::Scalar => ((4, 4), (4, 4)),
            other => panic!("kernel kind {other:?} cannot be selected on this target"),
        };

        let s64 = f64::micro_kernel_shape();
        let s32 = f32::micro_kernel_shape();
        assert_eq!(
            (s64.mr, s64.nr),
            expected_f64,
            "f64 shape/dispatch mismatch"
        );
        assert_eq!(
            (s32.mr, s32.nr),
            expected_f32,
            "f32 shape/dispatch mismatch"
        );
    }

    // Generates an end-to-end round-trip test helper for a floating-point type:
    // it packs A/B panels at the host's selected `MR`/`NR`, runs the dispatched
    // `micro_kernel`, and compares against an independent scalar reference that
    // honors reference-BLAS `beta == 0` "C not referenced" semantics.
    //
    // A shape/dispatch mismatch (finding 1) makes the kernel read the packed
    // panels with the wrong stride -> gross mismatch here. A `beta == 0` C-read
    // bug (finding 3) turns a NaN-initialized C into NaN output -> caught by the
    // `is_finite` assertion.
    macro_rules! kernel_roundtrip_helper {
        ($helper:ident, $ty:ty, $tol_rel:expr, $tol_abs:expr) => {
            fn $helper(k: usize, alpha: $ty, beta: $ty, c_init: $ty) {
                let shape = <$ty as GemmKernel>::micro_kernel_shape();
                let mr = shape.mr;
                let nr = shape.nr;
                let stride = mr;

                // Packed row-panel A (mr rows per k-step) and col-panel B.
                let a: Vec<$ty> = (0..k * mr).map(|i| ((i % 7) as $ty) * 0.1 - 0.3).collect();
                let b: Vec<$ty> = (0..k * nr).map(|i| ((i % 5) as $ty) * 0.1 - 0.2).collect();

                let mut c_ref = vec![c_init; mr * nr];
                let mut c_test = c_ref.clone();

                // Independent reference (naive, non-FMA accumulation).
                for j in 0..nr {
                    for i in 0..mr {
                        let mut s: $ty = 0.0;
                        for p in 0..k {
                            s += a[p * mr + i] * b[p * nr + j];
                        }
                        let idx = i + j * stride;
                        c_ref[idx] = if beta == 0.0 {
                            alpha * s
                        } else {
                            alpha * s + beta * c_ref[idx]
                        };
                    }
                }

                // SAFETY: `micro_kernel_shape()` and `micro_kernel` both use
                // the kernel picked by `selected_kernel_kind()`, so `a` (k *
                // mr elements) and `b` (k * nr) are packed for exactly the
                // kernel that runs, and `c_test` holds its `mr x nr` tile with
                // leading dimension `stride = mr`; the three vectors are
                // distinct allocations.
                unsafe {
                    <$ty as GemmKernel>::micro_kernel(
                        k,
                        alpha,
                        a.as_ptr(),
                        b.as_ptr(),
                        beta,
                        c_test.as_mut_ptr(),
                        stride,
                    );
                }

                for (idx, (&r, &t)) in c_ref.iter().zip(c_test.iter()).enumerate() {
                    assert!(
                        t.is_finite(),
                        "non-finite kernel output at {idx} (k={k}, alpha={alpha}, beta={beta})"
                    );
                    let diff = (r - t).abs();
                    let tol = r.abs() * $tol_rel + $tol_abs;
                    assert!(
                        diff <= tol,
                        "mismatch at {idx}: ref={r}, kernel={t}, diff={diff} (k={k}, alpha={alpha}, beta={beta})"
                    );
                }
            }
        };
    }

    kernel_roundtrip_helper!(check_host_f64, f64, 1e-10, 1e-10);
    kernel_roundtrip_helper!(check_host_f32, f32, 1e-4, 1e-4);

    #[test]
    fn test_host_kernel_f64_matches_reference() {
        for &k in &[0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
            check_host_f64(k, 1.0, 0.0, 0.0);
            check_host_f64(k, 1.0, 1.0, 0.5);
            check_host_f64(k, 2.5, 0.0, 0.0);
            check_host_f64(k, 0.7, 0.3, 1.25);
            // beta == 0 must NOT read C: a NaN-filled C must yield a finite result.
            check_host_f64(k, 1.0, 0.0, f64::NAN);
            check_host_f64(k, -1.5, 0.0, f64::NAN);
        }
    }

    #[test]
    fn test_host_kernel_f32_matches_reference() {
        for &k in &[0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64] {
            check_host_f32(k, 1.0, 0.0, 0.0);
            check_host_f32(k, 1.0, 1.0, 0.5);
            check_host_f32(k, 2.5, 0.0, 0.0);
            check_host_f32(k, 0.7, 0.3, 1.25);
            check_host_f32(k, 1.0, 0.0, f32::NAN);
            check_host_f32(k, -1.5, 0.0, f32::NAN);
        }
    }

    // ------------------------------------------------------------------
    // Finding 3 (regression), host-CPU-independent: the SCALAR fallback kernels
    // are always callable regardless of what the host CPU supports, so we test
    // them directly (bypassing dispatch). With beta == 0 and a NaN-filled C the
    // output must be the finite `alpha * A * B`, never NaN.
    // ------------------------------------------------------------------
    #[test]
    fn test_scalar_f64_beta_zero_ignores_c() {
        const MR: usize = 4;
        const NR: usize = 4;
        let k = 5_usize;
        let a: Vec<f64> = (0..k * MR).map(|i| (i as f64) * 0.1 - 0.3).collect();
        let b: Vec<f64> = (0..k * NR).map(|i| (i as f64) * 0.1 - 0.2).collect();
        let mut c = vec![f64::NAN; MR * NR];

        // SAFETY: the scalar 4 x 4 kernel needs no CPU feature; `a` holds k *
        // MR and `b` k * NR packed elements, and `c` holds the MR x NR tile
        // with leading dimension MR.
        unsafe {
            micro_kernel_f64_scalar(k, 1.5, a.as_ptr(), b.as_ptr(), 0.0, c.as_mut_ptr(), MR);
        }

        for j in 0..NR {
            for i in 0..MR {
                let mut s = 0.0_f64;
                for p in 0..k {
                    s += a[p * MR + i] * b[p * NR + j];
                }
                let got = c[i + j * MR];
                assert!(got.is_finite(), "scalar f64 beta==0 produced NaN");
                assert!((got - 1.5 * s).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_scalar_f32_beta_zero_ignores_c() {
        const MR: usize = 4;
        const NR: usize = 4;
        let k = 5_usize;
        let a: Vec<f32> = (0..k * MR).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.1 - 0.2).collect();
        let mut c = vec![f32::NAN; MR * NR];

        // SAFETY: the scalar 4 x 4 kernel needs no CPU feature; `a` holds k *
        // MR and `b` k * NR packed elements, and `c` holds the MR x NR tile
        // with leading dimension MR.
        unsafe {
            micro_kernel_f32_scalar(k, 1.5, a.as_ptr(), b.as_ptr(), 0.0, c.as_mut_ptr(), MR);
        }

        for j in 0..NR {
            for i in 0..MR {
                let mut s = 0.0_f32;
                for p in 0..k {
                    s += a[p * MR + i] * b[p * NR + j];
                }
                let got = c[i + j * MR];
                assert!(got.is_finite(), "scalar f32 beta==0 produced NaN");
                assert!((got - 1.5 * s).abs() < 1e-4);
            }
        }
    }

    // ------------------------------------------------------------------
    // Finding 1 (regression), host-CPU-independent: directly exercise the
    // SSE4.2-shaped kernels (f64 = 4×2, f32 = 4×4) that an AVX2-without-FMA CPU
    // now correctly falls back to. This proves the 4×2 / 4×4 shape+kernel pair
    // is self-consistent and correct even on a dev/CI machine whose CPU never
    // takes the SSE path through dispatch. Also covers the beta == 0 NaN case.
    // ------------------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sse42_shaped_kernels_bypass_dispatch() {
        if !is_x86_feature_detected!("sse4.2") {
            return; // Cannot execute SSE4.2 instructions on this CPU.
        }

        // --- f64, 4×2 ---
        {
            const MR: usize = 4;
            const NR: usize = 2;
            for &(alpha, beta, c_init) in &[
                (1.0_f64, 0.0_f64, 0.0_f64),
                (2.5, 0.0, f64::NAN), // beta == 0 must ignore NaN C
                (1.0, 1.0, 0.5),
                (0.7, 0.3, 1.25),
            ] {
                let k = 6_usize;
                let a: Vec<f64> = (0..k * MR).map(|i| (i as f64) * 0.1 - 0.3).collect();
                let b: Vec<f64> = (0..k * NR).map(|i| (i as f64) * 0.1 - 0.2).collect();
                let mut c = vec![c_init; MR * NR];

                // SAFETY: SSE4.2 was detected at the top of the test and SSE2
                // is part of the x86_64 baseline (the kernel's target
                // features); `a` holds k * MR and `b` k * NR packed elements
                // and `c` the 4 x 2 tile with leading dimension MR.
                unsafe {
                    super::micro_kernel_f64_sse42(
                        k,
                        alpha,
                        a.as_ptr(),
                        b.as_ptr(),
                        beta,
                        c.as_mut_ptr(),
                        MR,
                    );
                }

                for j in 0..NR {
                    for i in 0..MR {
                        let mut s = 0.0_f64;
                        for p in 0..k {
                            s += a[p * MR + i] * b[p * NR + j];
                        }
                        let expected = if beta == 0.0 {
                            alpha * s
                        } else {
                            alpha * s + beta * c_init
                        };
                        let got = c[i + j * MR];
                        assert!(got.is_finite());
                        assert!(
                            (got - expected).abs() < expected.abs() * 1e-12 + 1e-12,
                            "sse42 f64 mismatch: got={got}, expected={expected}"
                        );
                    }
                }
            }
        }

        // --- f32, 4×4 ---
        {
            const MR: usize = 4;
            const NR: usize = 4;
            for &(alpha, beta, c_init) in &[
                (1.0_f32, 0.0_f32, 0.0_f32),
                (2.5, 0.0, f32::NAN),
                (1.0, 1.0, 0.5),
                (0.7, 0.3, 1.25),
            ] {
                let k = 6_usize;
                let a: Vec<f32> = (0..k * MR).map(|i| (i as f32) * 0.1 - 0.3).collect();
                let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.1 - 0.2).collect();
                let mut c = vec![c_init; MR * NR];

                // SAFETY: SSE4.2 was detected at the top of the test and SSE2
                // is part of the x86_64 baseline (the kernel's target
                // features); `a` holds k * MR and `b` k * NR packed elements
                // and `c` the 4 x 4 tile with leading dimension MR.
                unsafe {
                    super::micro_kernel_f32_sse42(
                        k,
                        alpha,
                        a.as_ptr(),
                        b.as_ptr(),
                        beta,
                        c.as_mut_ptr(),
                        MR,
                    );
                }

                for j in 0..NR {
                    for i in 0..MR {
                        let mut s = 0.0_f32;
                        for p in 0..k {
                            s += a[p * MR + i] * b[p * NR + j];
                        }
                        let expected = if beta == 0.0 {
                            alpha * s
                        } else {
                            alpha * s + beta * c_init
                        };
                        let got = c[i + j * MR];
                        assert!(got.is_finite());
                        assert!(
                            (got - expected).abs() < expected.abs() * 1e-4 + 1e-4,
                            "sse42 f32 mismatch: got={got}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Finding 2 (regression): the SIMD-control features must change the kernel
    // that actually executes, not just the capability-reporting API. These
    // tests only compile+run under the corresponding feature, and are what the
    // `cargo test --features force-scalar` (etc.) verification exercises.
    // ------------------------------------------------------------------
    #[cfg(feature = "force-scalar")]
    #[test]
    fn test_force_scalar_forces_scalar_kernel() {
        assert_eq!(selected_kernel_kind(), GemmKernelKind::Scalar);
        let s64 = f64::micro_kernel_shape();
        let s32 = f32::micro_kernel_shape();
        assert_eq!((s64.mr, s64.nr), (4, 4));
        assert_eq!((s32.mr, s32.nr), (4, 4));
    }

    #[cfg(all(feature = "max-simd-128", not(feature = "force-scalar")))]
    #[test]
    fn test_max_simd_128_caps_width() {
        let kind = selected_kernel_kind();
        assert_ne!(kind, GemmKernelKind::Avx2, "max-simd-128 must forbid AVX2");
        assert_ne!(
            kind,
            GemmKernelKind::Avx512,
            "max-simd-128 must forbid AVX-512"
        );
    }

    #[cfg(all(
        feature = "max-simd-256",
        not(feature = "max-simd-128"),
        not(feature = "force-scalar")
    ))]
    #[test]
    fn test_max_simd_256_caps_width() {
        assert_ne!(
            selected_kernel_kind(),
            GemmKernelKind::Avx512,
            "max-simd-256 must forbid AVX-512"
        );
    }
    #[test]
    fn test_micro_kernel_shape_complex() {
        let shape_c64 = Complex64::micro_kernel_shape();
        assert_eq!(shape_c64.mr, 2);
        assert_eq!(shape_c64.nr, 2);

        let shape_c32 = Complex32::micro_kernel_shape();
        assert_eq!(shape_c32.mr, 2);
        assert_eq!(shape_c32.nr, 2);
    }

    /// Verify the Complex64 micro-kernel computes the correct product.
    ///
    /// A = [[1+2i, 3+4i],  (k=2, mr=2 column panel after packing: [1+2i, 5+6i, 3+4i, 7+8i])
    ///      [5+6i, 7+8i]]
    /// B = [[1+0i, 0+1i],  (k=2, nr=2 row panel: [1+0i, 0+1i, 2+0i, 0+2i])
    ///      [2+0i, 0+2i]]
    ///
    /// Row 0 of A * B:
    ///   C[0,0] = (1+2i)(1+0i) + (3+4i)(2+0i) = (1+2i) + (6+8i) = 7+10i
    ///   C[0,1] = (1+2i)(0+1i) + (3+4i)(0+2i) = (-2+1i) + (-8+6i) = -10+7i
    /// Row 1 of A * B:
    ///   C[1,0] = (5+6i)(1+0i) + (7+8i)(2+0i) = (5+6i) + (14+16i) = 19+22i
    ///   C[1,1] = (5+6i)(0+1i) + (7+8i)(0+2i) = (-6+5i) + (-16+14i) = -22+19i
    #[test]
    fn test_micro_kernel_c64_correctness() {
        use num_complex::Complex64;

        // Packed A panel: k=2, mr=2, column-major in the tile
        // a[p*MR + i] for p in 0..k, i in 0..MR
        // p=0: a[0]=1+2i (row 0), a[1]=5+6i (row 1)
        // p=1: a[2]=3+4i (row 0), a[3]=7+8i (row 1)
        let a = [
            Complex64::new(1.0, 2.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(7.0, 8.0),
        ];

        // Packed B panel: k=2, nr=2
        // b[p*NR + j] for p in 0..k, j in 0..NR
        // p=0: b[0]=1+0i (col 0), b[1]=0+1i (col 1)
        // p=1: b[2]=2+0i (col 0), b[3]=0+2i (col 1)
        let b = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 2.0),
        ];

        // C is a 2×2 column-major matrix stored with stride 2
        let zero = Complex64::new(0.0, 0.0);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);

        let mut c = [zero; 4]; // c[i + j*2]

        // SAFETY: the scalar complex kernel is 2 x 2 and needs no CPU feature:
        // with k = 2, `a` and `b` hold the 4 packed elements each it reads and
        // `c` the 2 x 2 tile with leading dimension 2.
        unsafe {
            micro_kernel_c64_scalar(2, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), 2);
        }

        // c[0+0*2]=c[0] = C[0,0] = 7+10i
        // c[1+0*2]=c[1] = C[1,0] = 19+22i
        // c[0+1*2]=c[2] = C[0,1] = -10+7i
        // c[1+1*2]=c[3] = C[1,1] = -22+19i
        let tol = 1e-12;
        assert!((c[0].re - 7.0).abs() < tol, "C[0,0].re got {}", c[0].re);
        assert!((c[0].im - 10.0).abs() < tol, "C[0,0].im got {}", c[0].im);
        assert!((c[1].re - 19.0).abs() < tol, "C[1,0].re got {}", c[1].re);
        assert!((c[1].im - 22.0).abs() < tol, "C[1,0].im got {}", c[1].im);
        assert!((c[2].re - (-10.0)).abs() < tol, "C[0,1].re got {}", c[2].re);
        assert!((c[2].im - 7.0).abs() < tol, "C[0,1].im got {}", c[2].im);
        assert!((c[3].re - (-22.0)).abs() < tol, "C[1,1].re got {}", c[3].re);
        assert!((c[3].im - 19.0).abs() < tol, "C[1,1].im got {}", c[3].im);
    }

    #[test]
    fn test_micro_kernel_c64_alpha_beta_scaling() {
        use num_complex::Complex64;

        // Test alpha scaling: 2×2 with identity-like A and a scalar B
        // a[0,0] = 1+0i, all other a = 0
        // b[0,0] = 5+3i, all other b = 0
        // acc[0][0] = (1+0i)*(5+3i) = 5+3i
        // With alpha=2+0i, beta=0: C[0,0] = 2*(5+3i) = 10+6i
        let a2 = [
            Complex64::new(1.0, 0.0), // p=0, row 0
            Complex64::new(0.0, 0.0), // p=0, row 1
            Complex64::new(0.0, 0.0), // p=1, row 0
            Complex64::new(0.0, 0.0), // p=1, row 1
        ];
        let b2 = [
            Complex64::new(5.0, 3.0), // p=0, col 0
            Complex64::new(0.0, 0.0), // p=0, col 1
            Complex64::new(0.0, 0.0), // p=1, col 0
            Complex64::new(0.0, 0.0), // p=1, col 1
        ];
        let alpha2 = Complex64::new(2.0, 0.0);
        let beta2 = Complex64::new(0.0, 0.0);
        let mut c2 = [Complex64::new(99.0, 99.0); 4];
        // SAFETY: the scalar complex kernel is 2 x 2: with k = 2, `a2` and
        // `b2` hold the 4 packed elements each it reads and `c2` the 2 x 2
        // tile with leading dimension 2.
        unsafe {
            micro_kernel_c64_scalar(
                2,
                alpha2,
                a2.as_ptr(),
                b2.as_ptr(),
                beta2,
                c2.as_mut_ptr(),
                2,
            );
        }
        let tol = 1e-12;
        // alpha * acc[0][0] + 0*c = 2*(5+3i) = 10+6i
        assert!(
            (c2[0].re - 10.0).abs() < tol,
            "alpha*acc[0,0].re = {}",
            c2[0].re
        );
        assert!(
            (c2[0].im - 6.0).abs() < tol,
            "alpha*acc[0,0].im = {}",
            c2[0].im
        );
        // beta=0 so all other C entries are zeroed regardless of initial value
        assert!(c2[1].re.abs() < tol, "C[1,0].re = {}", c2[1].re);
        assert!(c2[2].re.abs() < tol, "C[0,1].re = {}", c2[2].re);
        assert!(c2[3].re.abs() < tol, "C[1,1].re = {}", c2[3].re);

        // Test beta accumulation: same A,B but with beta=1 and non-zero C
        // C = alpha*acc + 1*C_init: C[0,0] = 10+6i + 1+1i = 11+7i
        let mut c3 = [
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let beta3 = Complex64::new(1.0, 0.0);
        // SAFETY: the scalar complex kernel is 2 x 2: with k = 2, `a2` and
        // `b2` hold the 4 packed elements each it reads and `c3` the 2 x 2
        // tile with leading dimension 2.
        unsafe {
            micro_kernel_c64_scalar(
                2,
                alpha2,
                a2.as_ptr(),
                b2.as_ptr(),
                beta3,
                c3.as_mut_ptr(),
                2,
            );
        }
        assert!(
            (c3[0].re - 11.0).abs() < tol,
            "beta=1 C[0,0].re = {}",
            c3[0].re
        );
        assert!(
            (c3[0].im - 7.0).abs() < tol,
            "beta=1 C[0,0].im = {}",
            c3[0].im
        );
    }

    #[test]
    fn test_micro_kernel_c32_correctness() {
        use num_complex::Complex32;

        // Same test as c64 but with f32 precision
        let a = [
            Complex32::new(1.0, 2.0),
            Complex32::new(5.0, 6.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(7.0, 8.0),
        ];
        let b = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 2.0),
        ];
        let zero = Complex32::new(0.0, 0.0);
        let alpha = Complex32::new(1.0, 0.0);
        let beta = Complex32::new(0.0, 0.0);
        let mut c = [zero; 4];

        // SAFETY: the scalar complex kernel is 2 x 2 and needs no CPU feature:
        // with k = 2, `a` and `b` hold the 4 packed elements each it reads and
        // `c` the 2 x 2 tile with leading dimension 2.
        unsafe {
            micro_kernel_c32_scalar(2, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), 2);
        }

        let tol = 1e-5_f32;
        assert!((c[0].re - 7.0).abs() < tol, "C[0,0].re got {}", c[0].re);
        assert!((c[0].im - 10.0).abs() < tol, "C[0,0].im got {}", c[0].im);
        assert!((c[1].re - 19.0).abs() < tol, "C[1,0].re got {}", c[1].re);
        assert!((c[1].im - 22.0).abs() < tol, "C[1,0].im got {}", c[1].im);
        assert!((c[2].re - (-10.0)).abs() < tol, "C[0,1].re got {}", c[2].re);
        assert!((c[2].im - 7.0).abs() < tol, "C[0,1].im got {}", c[2].im);
        assert!((c[3].re - (-22.0)).abs() < tol, "C[1,1].re got {}", c[3].re);
        assert!((c[3].im - 19.0).abs() < tol, "C[1,1].im got {}", c[3].im);
    }

    #[test]
    fn test_micro_kernel_c64_beta_zero_does_not_read_c() {
        use num_complex::Complex64;

        // a[0,0] = 1+0i, all other a = 0; b[0,0] = 5+3i, all other b = 0
        // acc[0][0] = (1+0i)*(5+3i) = 5+3i
        let a = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let b = [
            Complex64::new(5.0, 3.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let alpha = Complex64::new(2.0, 0.0);
        let beta = Complex64::new(0.0, 0.0);
        // C is poisoned with NaN everywhere; beta==0 must mean it is never read.
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let mut c = [nan; 4];
        // SAFETY: the scalar complex kernel is 2 x 2: with k = 2, `a` and `b`
        // hold the 4 packed elements each it reads and `c` the 2 x 2 tile with
        // leading dimension 2.
        unsafe {
            micro_kernel_c64_scalar(2, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), 2);
        }
        for (idx, val) in c.iter().enumerate() {
            assert!(
                val.re.is_finite() && val.im.is_finite(),
                "c[{idx}] = {val:?} leaked NaN from beta=0-scaled C"
            );
        }
        let tol = 1e-12;
        assert!((c[0].re - 10.0).abs() < tol);
        assert!((c[0].im - 6.0).abs() < tol);
    }

    #[test]
    fn test_micro_kernel_c32_beta_zero_does_not_read_c() {
        use num_complex::Complex32;

        let a = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ];
        let b = [
            Complex32::new(5.0, 3.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ];
        let alpha = Complex32::new(2.0, 0.0);
        let beta = Complex32::new(0.0, 0.0);
        let nan = Complex32::new(f32::NAN, f32::NAN);
        let mut c = [nan; 4];
        // SAFETY: the scalar complex kernel is 2 x 2: with k = 2, `a` and `b`
        // hold the 4 packed elements each it reads and `c` the 2 x 2 tile with
        // leading dimension 2.
        unsafe {
            micro_kernel_c32_scalar(2, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr(), 2);
        }
        for (idx, val) in c.iter().enumerate() {
            assert!(
                val.re.is_finite() && val.im.is_finite(),
                "c[{idx}] = {val:?} leaked NaN from beta=0-scaled C"
            );
        }
        let tol = 1e-5_f32;
        assert!((c[0].re - 10.0).abs() < tol);
        assert!((c[0].im - 6.0).abs() < tol);
    }

    // ------------------------------------------------------------------
    // The per-problem kernel of a plan (`GemmPlan::for_problem`).
    // ------------------------------------------------------------------
    #[test]
    fn test_gemm_kernel_for_problem_choice() {
        use crate::level3::gemm_plan::{F32Avx2Policy, SimdTier};
        const TIERS: [SimdTier; 4] = [
            SimdTier::Portable,
            SimdTier::Sse42,
            SimdTier::Neon,
            SimdTier::Avx2Fma,
        ];
        // (24, 4096, 64) and (4096, 24, 64) get different answers from the
        // padding model (8×8 and 16×6), so a swap of m and n anywhere
        // between `kernel_for_problem` and the rule shows.
        const SHAPES: [(usize, usize, usize); 7] = [
            (1, 1, 1),
            (12, 4096, 4096),
            (16, 16, 16),
            (4096, 8, 4096),
            (77, 61, 3),
            (24, 4096, 64),
            (4096, 24, 64),
        ];
        const TILE_16X6: MicroKernelShape = MicroKernelShape { mr: 16, nr: 6 };
        const TILE_8X8: MicroKernelShape = MicroKernelShape { mr: 8, nr: 8 };
        fn check<T: GemmKernel>() -> usize {
            let canonical = T::micro_kernel_shape();
            let is_f32 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>();
            let mut switched = 0;
            for tier in TIERS {
                for (m, n, k) in SHAPES {
                    let legacy = kernel_for_problem::<T>(tier, F32Avx2Policy::Legacy8x8, m, n, k);
                    assert_eq!(legacy.shape, canonical, "{tier:?} legacy {m}x{n}x{k}");
                    // Only f32 on the AVX2 + FMA tier with the 8×8 AVX2 kernel
                    // as its canonical kernel switches, and only on x86-64
                    // (the only target with the 16×6 kernel), where the
                    // padding model does not keep 8×8.
                    let switches = is_f32
                        && cfg!(target_arch = "x86_64")
                        && tier == SimdTier::Avx2Fma
                        && canonical == TILE_8X8
                        && !f32_avx2_prefers_8x8(m, n, k);
                    let expected = if switches { TILE_16X6 } else { canonical };
                    let adaptive = kernel_for_problem::<T>(tier, F32Avx2Policy::Adaptive, m, n, k);
                    assert_eq!(adaptive.shape, expected, "{tier:?} adaptive {m}x{n}x{k}");
                    switched += usize::from(switches);
                }
            }
            switched
        }
        assert_eq!(check::<f64>(), 0);
        assert_eq!(check::<Complex64>(), 0);
        assert_eq!(check::<Complex32>(), 0);
        let f32_switches = check::<f32>();
        // (12, 4096), (16, 16), (77, 61) and (4096, 24) take 16×6 on an 8×8
        // AVX2 host; (24, 4096) keeps 8×8.
        assert!(f32_avx2_prefers_8x8(24, 4096, 64));
        assert!(!f32_avx2_prefers_8x8(4096, 24, 64));
        let canonical_is_8x8 = f32::micro_kernel_shape() == TILE_8X8;
        let expected = if cfg!(target_arch = "x86_64") && canonical_is_8x8 {
            4
        } else {
            0
        };
        assert_eq!(f32_switches, expected);

        // The same two shapes through `GemmPlan::for_problem`, which passes
        // (m, n, k) on to `kernel_for_problem`: on this host's tier, only an
        // x86-64 AVX2 + FMA host with the 8×8 canonical kernel switches, and
        // only for (4096, 24).
        use crate::level3::gemm_plan::{CANDIDATE_X86, GemmPlan};
        let host_switches = cfg!(target_arch = "x86_64")
            && SimdTier::current() == SimdTier::Avx2Fma
            && canonical_is_8x8;
        let canonical = f32::micro_kernel_shape();
        let wide = GemmPlan::<f32>::for_problem(CANDIDATE_X86, 24, 4096, 64);
        assert_eq!(wide.kernel.shape, canonical, "for_problem 24x4096x64");
        let tall = GemmPlan::<f32>::for_problem(CANDIDATE_X86, 4096, 24, 64);
        let tall_expected = if host_switches { TILE_16X6 } else { canonical };
        assert_eq!(tall.kernel.shape, tall_expected, "for_problem 4096x24x64");
    }
}

/// The f32 16×6 AVX2 kernel: its selection rule (pure, every target) and the
/// kernel itself (x86-64, AVX2 + FMA tier).
mod tests_f32_16x6 {
    use super::*;
    use crate::level3::gemm_plan::{F32Avx2Policy, SimdTier};

    const TILE_8X8: MicroKernelShape = MicroKernelShape { mr: 8, nr: 8 };
    const TILE_16X16: MicroKernelShape = MicroKernelShape { mr: 16, nr: 16 };
    const TILE_4X4: MicroKernelShape = MicroKernelShape { mr: 4, nr: 4 };

    /// `(m, n, prefers 8×8)` of the padding model with R = 0.8 (k does not
    /// matter without a work floor).
    const TABLE: [(usize, usize, bool); 7] = [
        (8, 4096, true),
        (4096, 8, true),
        (4096, 4096, false),
        (12, 4096, false),
        (16, 16, false),
        (24, 4096, true),
        (40, 4096, false),
    ];

    #[test]
    fn test_gemm_kernel_f32_avx2_shape_selection_table() {
        assert_eq!(F32_AVX2_8X8_REL_PEAK_PERMILLE, 800);
        assert_eq!(F32_AVX2_16X6_MIN_WORK, 0);
        for (m, n, prefers_8x8) in TABLE {
            for k in [1, 64, 4096] {
                assert_eq!(f32_avx2_prefers_8x8(m, n, k), prefers_8x8, "{m}x{n}x{k}");
                // The policy turns the model into the choice; Legacy8x8
                // always keeps the canonical 8×8 kernel.
                assert_eq!(
                    f32_avx2_picks_16x6(
                        SimdTier::Avx2Fma,
                        F32Avx2Policy::Adaptive,
                        TILE_8X8,
                        m,
                        n,
                        k
                    ),
                    !prefers_8x8,
                    "adaptive {m}x{n}x{k}"
                );
                assert!(
                    !f32_avx2_picks_16x6(
                        SimdTier::Avx2Fma,
                        F32Avx2Policy::Legacy8x8,
                        TILE_8X8,
                        m,
                        n,
                        k
                    ),
                    "legacy {m}x{n}x{k}"
                );
            }
        }
        // Degenerate dimensions never panic (the drivers return before).
        for (m, n) in [(0, 0), (0, 7), (7, 0)] {
            assert!(!f32_avx2_prefers_8x8(m, n, 5), "{m}x{n}");
        }
        // Dimensions far beyond any real matrix must not panic. Both costs
        // saturate there, so the answer is arbitrary and is not asserted.
        std::hint::black_box(f32_avx2_prefers_8x8(usize::MAX, usize::MAX, usize::MAX));
    }

    #[test]
    fn test_gemm_kernel_f32_avx2_model_parameters() {
        // The relative peak is a real input: at R = 1 (8×8 as fast as 16×6)
        // any extra 16×6 padding keeps 8×8, at R = 0.8 it does not.
        assert!(!f32_avx2_prefers_8x8_with(800, 0, 12, 4096, 64));
        assert!(f32_avx2_prefers_8x8_with(1000, 0, 12, 4096, 64));
        // Equal padding never prefers 8×8 for R <= 1.
        assert!(!f32_avx2_prefers_8x8_with(1000, 0, 48, 48, 64));
        // The work floor keeps 8×8 strictly below it, whatever the padding.
        assert!(f32_avx2_prefers_8x8_with(800, 1024, 16, 16, 3));
        assert!(!f32_avx2_prefers_8x8_with(800, 1024, 16, 16, 4));
        assert!(!f32_avx2_prefers_8x8_with(800, 0, 16, 16, 0));
    }

    #[test]
    fn test_gemm_kernel_f32_avx2_choice_rule() {
        let policies = [F32Avx2Policy::Legacy8x8, F32Avx2Policy::Adaptive];
        let tiers = [
            SimdTier::Portable,
            SimdTier::Sse42,
            SimdTier::Neon,
            SimdTier::Avx2Fma,
        ];
        // A shape the model gives to 16×6 and one it keeps on 8×8.
        let (wide, narrow) = ((4096, 4096, 512), (8, 4096, 512));
        assert!(!f32_avx2_prefers_8x8(wide.0, wide.1, wide.2));
        assert!(f32_avx2_prefers_8x8(narrow.0, narrow.1, narrow.2));
        for policy in policies {
            for tier in tiers {
                for canonical in [TILE_8X8, TILE_16X16, TILE_4X4] {
                    for (m, n, k) in [wide, narrow] {
                        let picks = f32_avx2_picks_16x6(tier, policy, canonical, m, n, k);
                        // Only the AVX2 + FMA tier, the adaptive policy and
                        // the canonical 8×8 kernel switch, and only for the
                        // shape the model gives to 16×6. In particular an
                        // AVX-512 host (canonical 16×16 on the same tier)
                        // keeps its kernel.
                        let expected = policy == F32Avx2Policy::Adaptive
                            && tier == SimdTier::Avx2Fma
                            && canonical == TILE_8X8
                            && (m, n, k) == wide;
                        assert_eq!(
                            picks, expected,
                            "{policy:?} {tier:?} {canonical:?} {m}x{n}x{k}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        use super::*;
        use crate::level3::gemm::{BlockingSource, GemmBlocking, gemm_planned};
        use crate::level3::gemm_plan::{
            CANDIDATE_X86, GemmPlan, GemmRoute, LEGACY_POLICY, avx2_fma_tier_or_skip,
        };
        use crate::level3::trsm::Trans;
        use oxiblas_core::parallel::Par;
        use oxiblas_matrix::Mat;

        const TILE_16X6: MicroKernelShape = MicroKernelShape { mr: 16, nr: 6 };

        /// A quiet NaN with a payload that no arithmetic produces.
        const SENTINEL_BITS: u32 = 0x7FC1_2345;

        /// A deterministic value in [-0.5, 0.5) that is not a short dyadic
        /// fraction, so products and sums round.
        fn sample(seed: usize) -> f32 {
            (seed.wrapping_mul(7_919) % 1_009) as f32 / 1_009.0 - 0.5
        }

        /// The per-element operation order of the AVX2 f32 kernels (8×8 and
        /// 16×6) on a `mr × nr` tile with leading dimension `ldc`: FMA chain
        /// over p from zero, `* alpha` when `alpha != 1`, then store / add /
        /// `fma(c, beta, acc)`.
        #[allow(clippy::too_many_arguments)]
        fn reference_tile(
            k: usize,
            alpha: f32,
            a: &[f32],
            b: &[f32],
            beta: f32,
            c: &mut [f32],
            ldc: usize,
            (mr, nr): (usize, usize),
        ) {
            for j in 0..nr {
                for i in 0..mr {
                    let mut acc = 0.0_f32;
                    for p in 0..k {
                        acc = a[p * mr + i].mul_add(b[p * nr + j], acc);
                    }
                    if alpha != 1.0 {
                        acc *= alpha;
                    }
                    let cij = &mut c[i + j * ldc];
                    *cij = if beta == 0.0 {
                        acc
                    } else if beta == 1.0 {
                        acc + *cij
                    } else {
                        cij.mul_add(beta, acc)
                    };
                }
            }
        }

        /// Runs `kernel` (shape `mr × nr`) on one tile of leading dimension
        /// `ldc > mr` inside a sentinel border and checks every element of the
        /// buffer bitwise against [`reference_tile`]: the tile must match and
        /// the border must stay untouched. `a` and `b` hold exactly `mr · k` /
        /// `nr · k` values followed by NaN, so a read past the panels shows.
        /// Under Miri the panels have no NaN tail: they end exactly at the end
        /// of their allocations, so Miri reports any read, and any pointer
        /// arithmetic (such as a prefetch address), past the panels.
        fn check_kernel_tile(kernel: MicroKernel<f32>, k: usize, alpha: f32, beta: f32) {
            let (mr, nr) = (kernel.shape.mr, kernel.shape.nr);
            let ldc = mr + 5;
            let (row0, col0) = (3, 1);
            let cols = nr + 2;
            let tail = if cfg!(miri) { 0 } else { 2 * mr * nr };
            let mut a: Vec<f32> = (0..mr * k).map(|i| sample(i + 11)).collect();
            a.resize(mr * k + tail, f32::NAN);
            let mut b: Vec<f32> = (0..nr * k).map(|i| sample(i * 3 + 5)).collect();
            b.resize(nr * k + tail, f32::NAN);
            let sentinel = f32::from_bits(SENTINEL_BITS);
            let mut c = vec![sentinel; ldc * cols];
            let inside = |idx: usize| {
                let (i, j) = (idx % ldc, idx / ldc);
                (row0..row0 + mr).contains(&i) && (col0..col0 + nr).contains(&j)
            };
            for (idx, v) in c.iter_mut().enumerate() {
                if inside(idx) {
                    // beta == 0 must not read C: NaN there would leak.
                    *v = if beta == 0.0 {
                        f32::NAN
                    } else {
                        sample(idx + 101) * 4.0
                    };
                }
            }
            let mut expected = c.clone();
            let offset = row0 + col0 * ldc;
            reference_tile(
                k,
                alpha,
                &a,
                &b,
                beta,
                &mut expected[offset..],
                ldc,
                (mr, nr),
            );
            // SAFETY: the caller passed the AVX2 + FMA tier gate, so the CPU
            // has AVX2 and FMA (both AVX2 f32 kernels need nothing else); `a`
            // holds `mr · k` and `b` `nr · k` packed values (plus a tail
            // outside Miri);
            // the `mr × nr` tile at `offset` with leading dimension `ldc`
            // lies inside `c` (`offset + (nr - 1) · ldc + mr <= ldc · cols`);
            // the three vectors are distinct allocations.
            unsafe {
                (kernel.kernel)(
                    k,
                    alpha,
                    a.as_ptr(),
                    b.as_ptr(),
                    beta,
                    c.as_mut_ptr().add(offset),
                    ldc,
                );
            }
            for (idx, (got, want)) in c.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{mr}x{nr} k={k} alpha={alpha} beta={beta}: element ({}, {}) {}: got {got}, expected {want}",
                    idx % ldc,
                    idx / ldc,
                    if inside(idx) {
                        "of the tile"
                    } else {
                        "of the border"
                    }
                );
            }
        }

        const KERNEL_KS: [usize; 15] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64];
        const ALPHA_BETA: [(f32, f32); 7] = [
            (1.0, 0.0),
            (-1.5, 0.0),
            (1.0, 1.0),
            (2.5, 1.0),
            (0.7, 0.3),
            (1.0, -0.5),
            (-0.25, 2.0),
        ];

        #[test]
        fn test_gemm_kernel_avx2_f32_16x6_matches_reference() {
            if !avx2_fma_tier_or_skip("test_gemm_kernel_avx2_f32_16x6_matches_reference") {
                return;
            }
            let k16x6 = avx2_f32::micro_kernel_16x6();
            assert_eq!(k16x6.shape, TILE_16X6);
            // The 8×8 kernel follows the same per-element order (the 16×6
            // epilogue is copied from it).
            let k8x8 = f32_kernel_for(GemmKernelKind::Avx2);
            assert_eq!(k8x8.shape, TILE_8X8);
            for k in KERNEL_KS {
                for (alpha, beta) in ALPHA_BETA {
                    check_kernel_tile(k16x6, k, alpha, beta);
                    check_kernel_tile(k8x8, k, alpha, beta);
                }
            }
        }

        /// `alpha · op(A) · op(B) + beta · C0` through `gemm_planned`.
        fn planned(
            plan: &GemmPlan<f32>,
            (ta, tb): (Trans, Trans),
            (m, k, n): (usize, usize, usize),
            (alpha, beta): (f32, f32),
            par: Par,
            source: BlockingSource,
        ) -> Mat<f32> {
            let operand = |trans: Trans, rows: usize, cols: usize, seed: usize| {
                let (r, c) = if trans == Trans::NoTrans {
                    (rows, cols)
                } else {
                    (cols, rows)
                };
                let mut x: Mat<f32> = Mat::zeros(r, c);
                for j in 0..c {
                    for i in 0..r {
                        x[(i, j)] = sample(i * 131 + j * 7 + seed);
                    }
                }
                x
            };
            let a = operand(ta, m, k, 1);
            let b = operand(tb, k, n, 2);
            let mut c: Mat<f32> = if beta == 0.0 {
                Mat::filled(m, n, f32::NAN)
            } else {
                operand(Trans::NoTrans, m, n, 3)
            };
            gemm_planned(
                ta,
                tb,
                alpha,
                a.as_ref(),
                b.as_ref(),
                beta,
                c.as_mut(),
                par,
                source,
                plan,
            );
            c
        }

        fn assert_bitwise(got: &Mat<f32>, want: &Mat<f32>, ctx: &str) {
            assert_eq!((got.nrows(), got.ncols()), (want.nrows(), want.ncols()));
            for j in 0..want.ncols() {
                for i in 0..want.nrows() {
                    assert_eq!(
                        got[(i, j)].to_bits(),
                        want[(i, j)].to_bits(),
                        "{ctx}: ({i}, {j}) got {}, expected {}",
                        got[(i, j)],
                        want[(i, j)]
                    );
                }
            }
        }

        fn assert_finite(c: &Mat<f32>, ctx: &str) {
            for j in 0..c.ncols() {
                for i in 0..c.nrows() {
                    assert!(c[(i, j)].is_finite(), "{ctx}: ({i}, {j}) beta == 0 read C");
                }
            }
        }

        /// Plans forcing the blocked route with the 8×8 and with the 16×6
        /// kernel (LEGACY policy otherwise).
        fn blocked_plans() -> (GemmPlan<f32>, GemmPlan<f32>) {
            let base = GemmPlan::<f32>::new(LEGACY_POLICY).with_forced_route(GemmRoute::Blocked);
            let p8x8 = GemmPlan {
                kernel: f32_kernel_for(GemmKernelKind::Avx2),
                ..base
            };
            let p16x6 = GemmPlan {
                kernel: avx2_f32::micro_kernel_16x6(),
                ..base
            };
            assert_eq!(p8x8.kernel.shape, TILE_8X8);
            assert_eq!(p16x6.kernel.shape, TILE_16X6);
            (p8x8, p16x6)
        }

        const TRANS: [(Trans, Trans); 4] = [
            (Trans::NoTrans, Trans::NoTrans),
            (Trans::Trans, Trans::NoTrans),
            (Trans::NoTrans, Trans::Trans),
            (Trans::Trans, Trans::Trans),
        ];
        /// `(alpha, beta)`; C is NaN-filled when beta == 0. The last beta is
        /// not a power of two, so `c · beta` rounds and only a fused
        /// `fma(c, beta, acc)` in both kernels gives equal bits.
        const SCALARS: [(f32, f32); 4] = [(0.75, 0.0), (1.0, 1.0), (-1.25, -0.5), (0.3, 0.7)];

        fn pars() -> Vec<Par> {
            #[cfg(feature = "parallel")]
            {
                vec![Par::Seq, Par::RayonWith(3)]
            }
            #[cfg(not(feature = "parallel"))]
            {
                vec![Par::Seq]
            }
        }

        #[test]
        fn test_gemm_kernel_avx2_f32_16x6_bitwise_equals_8x8() {
            if !avx2_fma_tier_or_skip("test_gemm_kernel_avx2_f32_16x6_bitwise_equals_8x8") {
                return;
            }
            let (p8x8, p16x6) = blocked_plans();
            // (a) Explicit blocking: three KC blocks (170 by 64), several MC
            // and NC blocks, and edge tiles of both kernels in both
            // dimensions (131 = 8·16 + 3, 101 = 16·6 + 5 = 12·8 + 5).
            let explicit = BlockingSource::Explicit(GemmBlocking {
                mc: 48,
                kc: 64,
                nc: 36,
            });
            // (b) Default blocking: with a dimension of 512 or more (the
            // auto-tuned path, KC from the canonical kernel; 700 deep, edges
            // in both dimensions), and with all dimensions below 512.
            let cases = [
                ((131, 170, 101), explicit, "explicit"),
                ((40, 700, 91), BlockingSource::Default, "default auto-tuned"),
                (
                    (100, 300, 77),
                    BlockingSource::Default,
                    "default for_kernel",
                ),
            ];
            for (dims, source, name) in cases {
                for trans in TRANS {
                    for scalars in SCALARS {
                        for par in pars() {
                            let want = planned(&p8x8, trans, dims, scalars, par, source);
                            let got = planned(&p16x6, trans, dims, scalars, par, source);
                            let ctx = format!("{name} {dims:?} {trans:?} {scalars:?} {par:?}");
                            assert_bitwise(&got, &want, &ctx);
                            if scalars.1 == 0.0 {
                                assert_finite(&got, &ctx);
                            }
                        }
                    }
                }
            }
        }

        #[cfg(feature = "parallel")]
        #[test]
        fn test_gemm_kernel_f32_avx2_narrow_c_parallel_bitwise() {
            if !avx2_fma_tier_or_skip("test_gemm_kernel_f32_avx2_narrow_c_parallel_bitwise") {
                return;
            }
            let (_, p16x6) = blocked_plans();
            // m below one 16-row tile, and n below one 6-column tile; both
            // above the parallel gate (128³ in total, 80³ per thread).
            for dims in [(10, 400, 600), (700, 800, 4)] {
                for trans in [TRANS[0], TRANS[3]] {
                    for scalars in [SCALARS[0], SCALARS[2]] {
                        let source = BlockingSource::Default;
                        let seq = planned(&p16x6, trans, dims, scalars, Par::Seq, source);
                        let par = planned(&p16x6, trans, dims, scalars, Par::RayonWith(3), source);
                        assert_bitwise(&par, &seq, &format!("{dims:?} {trans:?} {scalars:?}"));
                    }
                }
            }
        }

        #[cfg(feature = "parallel")]
        #[test]
        fn test_gemm_kernel_f32_avx2_16x6_both_scopes_bitwise() {
            use crate::level3::gemm::PARALLEL_GATE;
            use crate::level3::gemm_parallel::scope_runs;
            use crate::level3::gemm_plan::PackScope;
            if !avx2_fma_tier_or_skip("test_gemm_kernel_f32_avx2_16x6_both_scopes_bitwise") {
                return;
            }
            // The real 16×6 kernel on the 2D driver with both pack scopes
            // (the x86 candidate takes the per-tile scope from five threads)
            // and 2, 3, 5 and 8 threads: every run gives the bits of the
            // sequential 16×6 run, which gives the bits of the sequential 8×8
            // run. Each shape is above the 8-thread parallel gate (80³ per
            // thread), so every run reaches the 2D driver with the forced
            // scope (counted).
            let (p8x8, p16x6) = blocked_plans();
            let explicit = BlockingSource::Explicit(GemmBlocking {
                mc: 48,
                kc: 64,
                nc: 36,
            });
            let cases = [
                // Three KC blocks, three MC blocks and six NC blocks (the
                // last narrower), edge tiles of 16×6 in both dimensions.
                ((133, 170, 197), explicit, "explicit blocks"),
                // C narrower than one 16-row tile.
                ((10, 600, 700), BlockingSource::Default, "m < 16"),
                // C narrower than one 6-column tile.
                ((700, 1500, 4), BlockingSource::Default, "n < 6"),
            ];
            // beta == 0 with a NaN C, and a general (alpha, beta).
            let scalars = [SCALARS[0], SCALARS[3]];
            for ((m, k, n), source, name) in cases {
                for trans in [TRANS[0], TRANS[3]] {
                    for scalar in scalars {
                        let ctx = format!("{name} {:?} {trans:?} {scalar:?}", (m, k, n));
                        let seq = planned(&p16x6, trans, (m, k, n), scalar, Par::Seq, source);
                        let seq8 = planned(&p8x8, trans, (m, k, n), scalar, Par::Seq, source);
                        assert_bitwise(&seq, &seq8, &format!("{ctx}: Seq 16x6 vs Seq 8x8"));
                        if scalar.1 == 0.0 {
                            assert_finite(&seq, &ctx);
                        }
                        for scope in [PackScope::Shared, PackScope::PerTile] {
                            let plan = p16x6.with_forced_scope(scope);
                            for threads in [2usize, 3, 5, 8] {
                                let par = Par::RayonWith(threads);
                                assert!(
                                    PARALLEL_GATE.should_parallelize(m * n * k, par),
                                    "{ctx}: gate, t={threads}"
                                );
                                let before = scope_runs();
                                let got = planned(&plan, trans, (m, k, n), scalar, par, source);
                                let after = scope_runs();
                                let ran = match scope {
                                    PackScope::Shared => (after.0 - before.0, after.1 - before.1),
                                    PackScope::PerTile => (after.1 - before.1, after.0 - before.0),
                                };
                                let leg = format!("{ctx} {scope:?} t={threads}");
                                assert_eq!(ran, (1, 0), "{leg}: 2D driver (this scope, other)");
                                assert_bitwise(&got, &seq, &leg);
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn test_gemm_kernel_f32_avx2_candidate_plan_equals_legacy_bitwise() {
            if !avx2_fma_tier_or_skip(
                "test_gemm_kernel_f32_avx2_candidate_plan_equals_legacy_bitwise",
            ) {
                return;
            }
            // Blocked under both policies: neither thin, above the x86
            // candidate's small ceiling (80³) and the copy range (32³).
            let (m, k, n) = (130, 120, 150);
            assert!(m * n * k > 80 * 80 * 80);
            let candidate = GemmPlan::<f32>::for_problem(CANDIDATE_X86, m, n, k);
            let legacy = GemmPlan::<f32>::new(LEGACY_POLICY);
            let canonical = f32::micro_kernel_shape();
            assert_eq!(legacy.kernel.shape, canonical);
            if canonical == TILE_8X8 {
                assert!(!f32_avx2_prefers_8x8(m, n, k));
                assert_eq!(
                    candidate.kernel.shape, TILE_16X6,
                    "the candidate picks 16x6"
                );
            } else {
                // An AVX-512 host keeps its canonical 16×16 kernel.
                assert_eq!(candidate.kernel.shape, canonical);
            }
            for trans in [TRANS[0], TRANS[3]] {
                for scalars in SCALARS {
                    for par in pars() {
                        let source = BlockingSource::Default;
                        let want = planned(&legacy, trans, (m, k, n), scalars, par, source);
                        let got = planned(&candidate, trans, (m, k, n), scalars, par, source);
                        assert_bitwise(&got, &want, &format!("{trans:?} {scalars:?} {par:?}"));
                    }
                }
            }
        }
    }
}
