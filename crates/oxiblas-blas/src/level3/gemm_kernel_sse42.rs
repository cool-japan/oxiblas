//! SSE4.2 GEMM micro-kernels (128-bit, x86-64).
//!
//! These micro-kernels fill the performance gap between scalar and AVX2 on
//! older x86-64 processors that have SSE4.2 but not AVX2/FMA (e.g. pre-Haswell
//! Nehalem/Sandy Bridge/Ivy Bridge, some Intel Atom variants).
//!
//! # Kernel shapes
//!
//! | Type | MR | NR | Registers used |
//! |------|----|----|----------------|
//! | f64  |  4 |  2 | 8 × __m128d    |
//! | f32  |  4 |  4 | 8 × __m128     |
//!
//! The shapes are intentionally modest: `__m128d` holds 2 f64 lanes and
//! `__m128` holds 4 f32 lanes, so we use 2-way unrolling on the column
//! dimension to keep register pressure reasonable while still beating scalar.
//!
//! # Safety contract
//!
//! Every `pub(super)` function in this module is `unsafe`.  Callers must
//! ensure:
//! - All raw pointers are valid, non-null, and point to initialised memory.
//! - The pointed-to arrays have sufficient length (see individual docs).
//! - No aliasing between `a`, `b`, and `c` pointers.
//! - The CPU supports SSE4.2 (checked at call-site via
//!   `is_x86_feature_detected!`).

#![allow(clippy::incompatible_msrv)]

// This entire module is x86-64-only.
#[cfg(target_arch = "x86_64")]
mod inner {
    use core::arch::x86_64::*;

    // =========================================================================
    // f64 micro-kernel  (4 × 2, using __m128d)
    // =========================================================================

    /// SSE4.2 micro-kernel for f64: 4 rows × 2 columns.
    ///
    /// Accumulator layout (8 `__m128d` registers):
    ///
    /// ```text
    /// col 0: acc0[0..2], acc1[0..2]   (rows 0-1, rows 2-3)
    /// col 1: acc2[0..2], acc3[0..2]   (rows 0-1, rows 2-3)
    /// ```
    ///
    /// The inner loop runs `k` iterations.  Each iteration loads:
    /// - `a[p*4 .. p*4+4]` as two `__m128d` vectors (2 f64 each).
    /// - `b[p*2]` and `b[p*2+1]` as scalar broadcasts.
    ///
    /// # Safety
    ///
    /// - `a` must point to at least `k * 4` valid `f64` values.
    /// - `b` must point to at least `k * 2` valid `f64` values.
    /// - `c` must point to a matrix with at least 4 rows and 2 columns,
    ///   where column `j` starts at `c + j * c_stride`.
    #[target_feature(enable = "sse4.2", enable = "sse2")]
    pub(crate) unsafe fn micro_kernel_f64_sse42(
        k: usize,
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        c_stride: usize,
    ) {
        const MR: usize = 4;
        const NR: usize = 2;

        // 4 rows × 2 columns = 8 accumulator slots; 4 __m128d registers.
        // col 0
        let mut acc0 = _mm_setzero_pd(); // rows 0-1
        let mut acc1 = _mm_setzero_pd(); // rows 2-3
        // col 1
        let mut acc2 = _mm_setzero_pd(); // rows 0-1
        let mut acc3 = _mm_setzero_pd(); // rows 2-3

        // -------------------------------------------------------
        // Main accumulation loop with 2-way unrolling.
        // -------------------------------------------------------
        let k2 = k / 2;
        let k_rem = k % 2;

        for p in 0..k2 {
            let base = p * 2;

            // --- iteration base ---
            {
                let a_ptr = a.add(base * MR);
                let b_ptr = b.add(base * NR);

                let a_lo = _mm_loadu_pd(a_ptr); // rows 0-1
                let a_hi = _mm_loadu_pd(a_ptr.add(2)); // rows 2-3

                let b0 = _mm_set1_pd(*b_ptr);
                let b1 = _mm_set1_pd(*b_ptr.add(1));

                acc0 = _mm_add_pd(acc0, _mm_mul_pd(a_lo, b0));
                acc1 = _mm_add_pd(acc1, _mm_mul_pd(a_hi, b0));
                acc2 = _mm_add_pd(acc2, _mm_mul_pd(a_lo, b1));
                acc3 = _mm_add_pd(acc3, _mm_mul_pd(a_hi, b1));
            }

            // --- iteration base + 1 ---
            {
                let a_ptr = a.add((base + 1) * MR);
                let b_ptr = b.add((base + 1) * NR);

                let a_lo = _mm_loadu_pd(a_ptr);
                let a_hi = _mm_loadu_pd(a_ptr.add(2));

                let b0 = _mm_set1_pd(*b_ptr);
                let b1 = _mm_set1_pd(*b_ptr.add(1));

                acc0 = _mm_add_pd(acc0, _mm_mul_pd(a_lo, b0));
                acc1 = _mm_add_pd(acc1, _mm_mul_pd(a_hi, b0));
                acc2 = _mm_add_pd(acc2, _mm_mul_pd(a_lo, b1));
                acc3 = _mm_add_pd(acc3, _mm_mul_pd(a_hi, b1));
            }
        }

        // --- remainder (0 or 1 iterations) ---
        for r in 0..k_rem {
            let p = k2 * 2 + r;
            let a_ptr = a.add(p * MR);
            let b_ptr = b.add(p * NR);

            let a_lo = _mm_loadu_pd(a_ptr);
            let a_hi = _mm_loadu_pd(a_ptr.add(2));

            let b0 = _mm_set1_pd(*b_ptr);
            let b1 = _mm_set1_pd(*b_ptr.add(1));

            acc0 = _mm_add_pd(acc0, _mm_mul_pd(a_lo, b0));
            acc1 = _mm_add_pd(acc1, _mm_mul_pd(a_hi, b0));
            acc2 = _mm_add_pd(acc2, _mm_mul_pd(a_lo, b1));
            acc3 = _mm_add_pd(acc3, _mm_mul_pd(a_hi, b1));
        }

        // -------------------------------------------------------
        // Scale by alpha.
        // -------------------------------------------------------
        if alpha != 1.0 {
            let av = _mm_set1_pd(alpha);
            acc0 = _mm_mul_pd(acc0, av);
            acc1 = _mm_mul_pd(acc1, av);
            acc2 = _mm_mul_pd(acc2, av);
            acc3 = _mm_mul_pd(acc3, av);
        }

        // -------------------------------------------------------
        // Write C = alpha*A*B + beta*C.
        // -------------------------------------------------------
        if beta == 0.0 {
            _mm_storeu_pd(c, acc0);
            _mm_storeu_pd(c.add(2), acc1);
            _mm_storeu_pd(c.add(c_stride), acc2);
            _mm_storeu_pd(c.add(c_stride + 2), acc3);
        } else if beta == 1.0 {
            let c0 = _mm_loadu_pd(c);
            let c1 = _mm_loadu_pd(c.add(2));
            let c2 = _mm_loadu_pd(c.add(c_stride));
            let c3 = _mm_loadu_pd(c.add(c_stride + 2));
            _mm_storeu_pd(c, _mm_add_pd(acc0, c0));
            _mm_storeu_pd(c.add(2), _mm_add_pd(acc1, c1));
            _mm_storeu_pd(c.add(c_stride), _mm_add_pd(acc2, c2));
            _mm_storeu_pd(c.add(c_stride + 2), _mm_add_pd(acc3, c3));
        } else {
            let bv = _mm_set1_pd(beta);
            let c0 = _mm_loadu_pd(c);
            let c1 = _mm_loadu_pd(c.add(2));
            let c2 = _mm_loadu_pd(c.add(c_stride));
            let c3 = _mm_loadu_pd(c.add(c_stride + 2));
            _mm_storeu_pd(c, _mm_add_pd(acc0, _mm_mul_pd(c0, bv)));
            _mm_storeu_pd(c.add(2), _mm_add_pd(acc1, _mm_mul_pd(c1, bv)));
            _mm_storeu_pd(c.add(c_stride), _mm_add_pd(acc2, _mm_mul_pd(c2, bv)));
            _mm_storeu_pd(c.add(c_stride + 2), _mm_add_pd(acc3, _mm_mul_pd(c3, bv)));
        }
    }

    // =========================================================================
    // f32 micro-kernel  (4 × 4, using __m128)
    // =========================================================================

    /// SSE4.2 micro-kernel for f32: 4 rows × 4 columns.
    ///
    /// Accumulator layout (4 `__m128` registers, one per column):
    ///
    /// ```text
    /// col 0: acc0[0..4]  (rows 0-3)
    /// col 1: acc1[0..4]  (rows 0-3)
    /// col 2: acc2[0..4]  (rows 0-3)
    /// col 3: acc3[0..4]  (rows 0-3)
    /// ```
    ///
    /// The inner loop runs `k` iterations.  Each iteration loads:
    /// - `a[p*4 .. p*4+4]` as one `__m128` vector (4 f32).
    /// - `b[p*4]` through `b[p*4+3]` as scalar broadcasts.
    ///
    /// # Safety
    ///
    /// - `a` must point to at least `k * 4` valid `f32` values.
    /// - `b` must point to at least `k * 4` valid `f32` values.
    /// - `c` must point to a matrix with at least 4 rows and 4 columns,
    ///   where column `j` starts at `c + j * c_stride`.
    #[target_feature(enable = "sse4.2", enable = "sse2")]
    pub(crate) unsafe fn micro_kernel_f32_sse42(
        k: usize,
        alpha: f32,
        a: *const f32,
        b: *const f32,
        beta: f32,
        c: *mut f32,
        c_stride: usize,
    ) {
        const MR: usize = 4;
        const NR: usize = 4;

        // 4 rows × 4 columns = 16 accumulator slots; 4 __m128 registers.
        let mut acc0 = _mm_setzero_ps(); // col 0
        let mut acc1 = _mm_setzero_ps(); // col 1
        let mut acc2 = _mm_setzero_ps(); // col 2
        let mut acc3 = _mm_setzero_ps(); // col 3

        // -------------------------------------------------------
        // Main accumulation loop with 2-way unrolling.
        // -------------------------------------------------------
        let k2 = k / 2;
        let k_rem = k % 2;

        for p in 0..k2 {
            let base = p * 2;

            // --- iteration base ---
            {
                let a_ptr = a.add(base * MR);
                let b_ptr = b.add(base * NR);

                let a_vec = _mm_loadu_ps(a_ptr);

                let b0 = _mm_set1_ps(*b_ptr);
                let b1 = _mm_set1_ps(*b_ptr.add(1));
                let b2 = _mm_set1_ps(*b_ptr.add(2));
                let b3 = _mm_set1_ps(*b_ptr.add(3));

                acc0 = _mm_add_ps(acc0, _mm_mul_ps(a_vec, b0));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(a_vec, b1));
                acc2 = _mm_add_ps(acc2, _mm_mul_ps(a_vec, b2));
                acc3 = _mm_add_ps(acc3, _mm_mul_ps(a_vec, b3));
            }

            // --- iteration base + 1 ---
            {
                let a_ptr = a.add((base + 1) * MR);
                let b_ptr = b.add((base + 1) * NR);

                let a_vec = _mm_loadu_ps(a_ptr);

                let b0 = _mm_set1_ps(*b_ptr);
                let b1 = _mm_set1_ps(*b_ptr.add(1));
                let b2 = _mm_set1_ps(*b_ptr.add(2));
                let b3 = _mm_set1_ps(*b_ptr.add(3));

                acc0 = _mm_add_ps(acc0, _mm_mul_ps(a_vec, b0));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(a_vec, b1));
                acc2 = _mm_add_ps(acc2, _mm_mul_ps(a_vec, b2));
                acc3 = _mm_add_ps(acc3, _mm_mul_ps(a_vec, b3));
            }
        }

        // --- remainder (0 or 1 iterations) ---
        for r in 0..k_rem {
            let p = k2 * 2 + r;
            let a_ptr = a.add(p * MR);
            let b_ptr = b.add(p * NR);

            let a_vec = _mm_loadu_ps(a_ptr);

            let b0 = _mm_set1_ps(*b_ptr);
            let b1 = _mm_set1_ps(*b_ptr.add(1));
            let b2 = _mm_set1_ps(*b_ptr.add(2));
            let b3 = _mm_set1_ps(*b_ptr.add(3));

            acc0 = _mm_add_ps(acc0, _mm_mul_ps(a_vec, b0));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(a_vec, b1));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(a_vec, b2));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(a_vec, b3));
        }

        // -------------------------------------------------------
        // Scale by alpha.
        // -------------------------------------------------------
        if alpha != 1.0 {
            let av = _mm_set1_ps(alpha);
            acc0 = _mm_mul_ps(acc0, av);
            acc1 = _mm_mul_ps(acc1, av);
            acc2 = _mm_mul_ps(acc2, av);
            acc3 = _mm_mul_ps(acc3, av);
        }

        // -------------------------------------------------------
        // Write C = alpha*A*B + beta*C.
        // -------------------------------------------------------
        if beta == 0.0 {
            _mm_storeu_ps(c, acc0);
            _mm_storeu_ps(c.add(c_stride), acc1);
            _mm_storeu_ps(c.add(2 * c_stride), acc2);
            _mm_storeu_ps(c.add(3 * c_stride), acc3);
        } else if beta == 1.0 {
            let c0 = _mm_loadu_ps(c);
            let c1 = _mm_loadu_ps(c.add(c_stride));
            let c2 = _mm_loadu_ps(c.add(2 * c_stride));
            let c3 = _mm_loadu_ps(c.add(3 * c_stride));
            _mm_storeu_ps(c, _mm_add_ps(acc0, c0));
            _mm_storeu_ps(c.add(c_stride), _mm_add_ps(acc1, c1));
            _mm_storeu_ps(c.add(2 * c_stride), _mm_add_ps(acc2, c2));
            _mm_storeu_ps(c.add(3 * c_stride), _mm_add_ps(acc3, c3));
        } else {
            let bv = _mm_set1_ps(beta);
            let c0 = _mm_loadu_ps(c);
            let c1 = _mm_loadu_ps(c.add(c_stride));
            let c2 = _mm_loadu_ps(c.add(2 * c_stride));
            let c3 = _mm_loadu_ps(c.add(3 * c_stride));
            _mm_storeu_ps(c, _mm_add_ps(acc0, _mm_mul_ps(c0, bv)));
            _mm_storeu_ps(c.add(c_stride), _mm_add_ps(acc1, _mm_mul_ps(c1, bv)));
            _mm_storeu_ps(c.add(2 * c_stride), _mm_add_ps(acc2, _mm_mul_ps(c2, bv)));
            _mm_storeu_ps(c.add(3 * c_stride), _mm_add_ps(acc3, _mm_mul_ps(c3, bv)));
        }
    }
} // mod inner

// Re-export so `gemm_kernel.rs` can call them directly under `#[cfg(target_arch = "x86_64")]`.
#[cfg(target_arch = "x86_64")]
pub(super) use inner::{micro_kernel_f32_sse42, micro_kernel_f64_sse42};

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    /// Scalar GEMM reference: C = alpha * A * B + beta * C.
    ///
    /// Layout: A is MR×K (packed column-major panels), B is K×NR (packed),
    /// C is a column-major matrix with `c_stride` elements between columns.
    #[allow(dead_code)]
    fn gemm_scalar_ref<T>(
        mr: usize,
        nr: usize,
        k: usize,
        alpha: T,
        a: &[T],
        b: &[T],
        beta: T,
        c: &mut [T],
        c_stride: usize,
    ) where
        T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + Default,
    {
        let mut acc = vec![vec![T::default(); nr]; mr];
        for p in 0..k {
            for i in 0..mr {
                for j in 0..nr {
                    acc[i][j] = acc[i][j] + a[p * mr + i] * b[p * nr + j];
                }
            }
        }
        for j in 0..nr {
            for i in 0..mr {
                let idx = i + j * c_stride;
                c[idx] = alpha * acc[i][j] + beta * c[idx];
            }
        }
    }

    // -----------------------------------------------------------------------
    // f64 kernel tests
    // -----------------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    mod f64_tests {
        use super::super::inner::micro_kernel_f64_sse42;
        use super::*;

        const MR: usize = 4;
        const NR: usize = 2;

        /// Build a simple test problem and compare SSE4.2 kernel to scalar.
        fn run_f64_case(k: usize, alpha: f64, beta: f64) {
            if !is_x86_feature_detected!("sse4.2") {
                return; // skip on CPUs without SSE4.2
            }

            let a: Vec<f64> = (0..k * MR).map(|i| (i as f64) * 0.1 + 0.5).collect();
            let b: Vec<f64> = (0..k * NR).map(|i| (i as f64) * 0.2 - 0.3).collect();

            let mut c_ref = vec![0.5_f64; MR * NR]; // column-major, stride = MR
            let mut c_sse = c_ref.clone();

            gemm_scalar_ref(MR, NR, k, alpha, &a, &b, beta, &mut c_ref, MR);

            unsafe {
                micro_kernel_f64_sse42(
                    k,
                    alpha,
                    a.as_ptr(),
                    b.as_ptr(),
                    beta,
                    c_sse.as_mut_ptr(),
                    MR,
                );
            }

            for (idx, (r, s)) in c_ref.iter().zip(c_sse.iter()).enumerate() {
                let diff = (r - s).abs();
                let tol = r.abs() * 1e-12 + 1e-12;
                assert!(
                    diff < tol,
                    "f64 SSE4.2 mismatch at index {idx}: ref={r}, sse42={s}, diff={diff}"
                );
            }
        }

        #[test]
        fn test_f64_sse42_k1() {
            run_f64_case(1, 1.0, 0.0);
        }

        #[test]
        fn test_f64_sse42_k4() {
            run_f64_case(4, 1.0, 0.0);
        }

        #[test]
        fn test_f64_sse42_k7() {
            run_f64_case(7, 1.0, 0.0);
        }

        #[test]
        fn test_f64_sse42_alpha() {
            run_f64_case(8, 2.5, 0.0);
        }

        #[test]
        fn test_f64_sse42_beta_one() {
            run_f64_case(6, 1.0, 1.0);
        }

        #[test]
        fn test_f64_sse42_beta_general() {
            run_f64_case(5, 0.7, 0.3);
        }

        #[test]
        fn test_f64_sse42_k_odd() {
            run_f64_case(3, 1.0, 1.0);
        }

        #[test]
        fn test_f64_sse42_k_zero() {
            run_f64_case(0, 1.0, 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // f32 kernel tests
    // -----------------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    mod f32_tests {
        use super::super::inner::micro_kernel_f32_sse42;
        use super::*;

        const MR: usize = 4;
        const NR: usize = 4;

        fn run_f32_case(k: usize, alpha: f32, beta: f32) {
            if !is_x86_feature_detected!("sse4.2") {
                return;
            }

            let a: Vec<f32> = (0..k * MR).map(|i| (i as f32) * 0.1 + 0.5).collect();
            let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.2 - 0.3).collect();

            let mut c_ref = vec![0.5_f32; MR * NR];
            let mut c_sse = c_ref.clone();

            gemm_scalar_ref(MR, NR, k, alpha, &a, &b, beta, &mut c_ref, MR);

            unsafe {
                micro_kernel_f32_sse42(
                    k,
                    alpha,
                    a.as_ptr(),
                    b.as_ptr(),
                    beta,
                    c_sse.as_mut_ptr(),
                    MR,
                );
            }

            for (idx, (r, s)) in c_ref.iter().zip(c_sse.iter()).enumerate() {
                let diff = (r - s).abs();
                let tol = r.abs() * 1e-5 + 1e-6;
                assert!(
                    diff < tol,
                    "f32 SSE4.2 mismatch at index {idx}: ref={r}, sse42={s}, diff={diff}"
                );
            }
        }

        #[test]
        fn test_f32_sse42_k1() {
            run_f32_case(1, 1.0, 0.0);
        }

        #[test]
        fn test_f32_sse42_k4() {
            run_f32_case(4, 1.0, 0.0);
        }

        #[test]
        fn test_f32_sse42_k7() {
            run_f32_case(7, 1.0, 0.0);
        }

        #[test]
        fn test_f32_sse42_alpha() {
            run_f32_case(8, 2.5, 0.0);
        }

        #[test]
        fn test_f32_sse42_beta_one() {
            run_f32_case(6, 1.0, 1.0);
        }

        #[test]
        fn test_f32_sse42_beta_general() {
            run_f32_case(5, 0.7, 0.3);
        }

        #[test]
        fn test_f32_sse42_k_odd() {
            run_f32_case(3, 1.0, 1.0);
        }

        #[test]
        fn test_f32_sse42_k_zero() {
            run_f32_case(0, 1.0, 1.0);
        }
    }
}
