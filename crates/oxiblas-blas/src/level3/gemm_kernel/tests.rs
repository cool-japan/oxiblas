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
            _ => ((4, 4), (4, 4)),
        };

        let s64 = f64::micro_kernel_shape();
        let s32 = f32::micro_kernel_shape();
        assert_eq!((s64.mr, s64.nr), expected_f64, "f64 shape/dispatch mismatch");
        assert_eq!((s32.mr, s32.nr), expected_f32, "f32 shape/dispatch mismatch");
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

}
