//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::simd::SimdScalar;

use super::types::{F32x16, F32x8, F64x4, F64x8};

// Test-only switch that forces the scalar fallback path so the (otherwise dead
// on a SIMD-capable CI host) `else` branches are actually executed by the
// regression tests. `thread_local` keeps the override isolated to the setting
// thread so the parallel test runner cannot cross-contaminate.
#[cfg(test)]
thread_local! {
    static FORCE_SCALAR_FALLBACK: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[cfg(test)]
fn set_force_scalar_fallback(value: bool) {
    FORCE_SCALAR_FALLBACK.with(|flag| flag.set(value));
}

#[cfg(test)]
#[inline]
pub(super) fn force_scalar_fallback() -> bool {
    FORCE_SCALAR_FALLBACK.with(|flag| flag.get())
}

/// True when the CPU can execute the AVX2 + FMA instructions used by the
/// 256-bit register tier.
///
/// Uses OS-assisted runtime detection under `std`; under `no_std` there is no
/// runtime detector, so it degrades to the compile-time `target_feature` flags
/// (a `const` the optimizer can fold), which keeps the `no_std` build sound --
/// if the crate was not built with AVX2/FMA enabled the scalar fallback is
/// selected and no AVX instruction is emitted on the reachable path.
#[inline]
pub(super) fn has_avx2_fma() -> bool {
    #[cfg(test)]
    if force_scalar_fallback() {
        return false;
    }
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2") && cfg!(target_feature = "fma")
    }
}

/// True when the CPU can execute the AVX-512F instructions used by the 512-bit
/// floating-point register tier. See [`has_avx2_fma`] for the std/no_std split.
#[inline]
pub(super) fn has_avx512f() -> bool {
    #[cfg(test)]
    if force_scalar_fallback() {
        return false;
    }
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512f")
    }
}

/// True when the CPU supports AVX-512BW (byte/word integer ops).
#[inline]
pub(super) fn has_avx512bw() -> bool {
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512bw")
    }
}

/// True when the CPU supports AVX-512VNNI (vector neural-network dot products).
#[inline]
pub(super) fn has_avx512vnni() -> bool {
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512vnni")
    }
}

/// True when the CPU supports AVX-512VBMI (vector byte manipulation).
#[inline]
pub(super) fn has_avx512vbmi() -> bool {
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512vbmi")
    }
}

/// True when the CPU supports AVX-512DQ (doubleword/quadword ops).
#[inline]
pub(super) fn has_avx512dq() -> bool {
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx512dq")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512dq")
    }
}

/// True when the CPU supports AVX-512VL (vector-length extensions).
#[inline]
pub(super) fn has_avx512vl() -> bool {
    #[cfg(feature = "std")]
    {
        is_x86_feature_detected!("avx512vl")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512vl")
    }
}

/// Cold, never-inlined panic path for an out-of-range SIMD lane index.
///
/// `SimdRegister::extract` / `insert` are *safe*, infallible fns, but a valid
/// return value does not exist for `index >= LANES`. The previous code indexed
/// a fixed-size array (`arr[index]`), which panics with an opaque message in
/// both debug and release builds. Turning the out-of-range case into an
/// explicit, documented panic (like slice indexing) keeps the hot, in-range
/// path branch-predictable, avoids duplicating the panic string into every lane
/// accessor, and never resorts to `unreachable_unchecked()` (which would be UB
/// reachable from safe code).
#[cold]
#[inline(never)]
pub(super) fn lane_index_out_of_range(index: usize, lanes: usize) -> ! {
    panic!("SIMD lane index {index} out of range (register has {lanes} lanes)");
}

/// Broadcasts `value` into every lane without touching a SIMD instruction.
#[inline]
pub(super) unsafe fn scalar_splat<R: Copy, S: Copy, const N: usize>(value: S) -> R {
    let arr = [value; N];
    core::mem::transmute_copy(&arr)
}

/// Loads `N` lanes of `S` from `ptr` (unaligned) without a SIMD instruction.
///
/// # Safety
/// `ptr` must be valid for reading `N` contiguous `S` values, in addition to
/// the module-level representation contract on `R`.
#[inline]
pub(super) unsafe fn scalar_load<R: Copy, S: Copy, const N: usize>(ptr: *const S) -> R {
    let arr: [S; N] = core::array::from_fn(|i| unsafe { ptr.add(i).read_unaligned() });
    core::mem::transmute_copy(&arr)
}

/// Stores the `N` lanes of `value` to `ptr` (unaligned) without a SIMD store.
///
/// # Safety
/// `ptr` must be valid for writing `N` contiguous `S` values, in addition to
/// the module-level representation contract on `R`.
#[inline]
pub(super) unsafe fn scalar_store<R: Copy, S: Copy, const N: usize>(value: R, ptr: *mut S) {
    let arr: [S; N] = core::mem::transmute_copy(&value);
    for i in 0..N {
        unsafe { ptr.add(i).write_unaligned(arr[i]) };
    }
}

/// Applies `f` lane-wise to two registers without a SIMD instruction.
#[inline]
pub(super) unsafe fn scalar_binop<R: Copy, S: Copy, const N: usize>(a: R, b: R, f: impl Fn(S, S) -> S) -> R {
    let aa: [S; N] = core::mem::transmute_copy(&a);
    let bb: [S; N] = core::mem::transmute_copy(&b);
    let rr: [S; N] = core::array::from_fn(|i| f(aa[i], bb[i]));
    core::mem::transmute_copy(&rr)
}

/// Applies `f` lane-wise to three registers (fused-op fallback) scalar-only.
#[inline]
pub(super) unsafe fn scalar_ternop<R: Copy, S: Copy, const N: usize>(
    a: R,
    b: R,
    c: R,
    f: impl Fn(S, S, S) -> S,
) -> R {
    let aa: [S; N] = core::mem::transmute_copy(&a);
    let bb: [S; N] = core::mem::transmute_copy(&b);
    let cc: [S; N] = core::mem::transmute_copy(&c);
    let rr: [S; N] = core::array::from_fn(|i| f(aa[i], bb[i], cc[i]));
    core::mem::transmute_copy(&rr)
}

/// Folds the lanes of a register with `f` (horizontal reduction) scalar-only.
///
/// The lanes are folded left-to-right; for `+` this can differ from a SIMD
/// tree reduction in the last ULP, and for `max`/`min` the caller-supplied
/// closure defines the NaN policy. This only runs when the feature is absent.
#[inline]
pub(super) unsafe fn scalar_reduce<R: Copy, S: Copy, const N: usize>(a: R, f: impl Fn(S, S) -> S) -> S {
    let aa: [S; N] = core::mem::transmute_copy(&a);
    let mut acc = aa[0];
    for i in 1..N {
        acc = f(acc, aa[i]);
    }
    acc
}

impl SimdScalar for f64 {
    type Simd256 = F64x4;
    type Simd512 = F64x8;
}

impl SimdScalar for f32 {
    type Simd256 = F32x8;
    type Simd512 = F32x16;
}

#[cfg(test)]
pub(super) mod tests {
    // `super::*` provides the feature predicates, the scalar-fallback helpers and
    // `set_force_scalar_fallback`. The register types, the intrinsic vector types
    // (e.g. `__mmask8`) and the `SimdRegister`/`SimdMask` traits live elsewhere and
    // are only needed by the tests, so import them here rather than widening the
    // production imports of `functions`.
    use super::super::types::*;
    use super::*;
    use crate::simd::{SimdMask, SimdRegister};
    use core::arch::x86_64::*;

    // SSE4.2 tests (always available on x86_64)
    #[test]
    fn test_f64x2_sse_basic() {
        let a = F64x2Sse::splat(2.0);
        let b = F64x2Sse::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(1), 5.0);

        let prod = a.mul(b);
        assert_eq!(prod.extract(0), 6.0);

        // Test emulated FMA
        let c = F64x2Sse::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f64x2_sse_reduce() {
        unsafe {
            let data = [1.0f64, 2.0];
            let v = F64x2Sse::load_unaligned(data.as_ptr());
            assert_eq!(v.reduce_sum(), 3.0);
            assert_eq!(v.reduce_max(), 2.0);
            assert_eq!(v.reduce_min(), 1.0);
        }
    }

    #[test]
    fn test_f32x4_sse_basic() {
        let a = F32x4Sse::splat(2.0);
        let b = F32x4Sse::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);

        let fma = a.mul_add(b, F32x4Sse::splat(1.0));
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f32x4_sse_reduce() {
        unsafe {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let v = F32x4Sse::load_unaligned(data.as_ptr());
            assert_eq!(v.reduce_sum(), 10.0);
            assert_eq!(v.reduce_max(), 4.0);
            assert_eq!(v.reduce_min(), 1.0);
        }
    }

    // AVX2 tests
    #[test]
    fn test_f64x4_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = F64x4::splat(2.0);
        let b = F64x4::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(1), 5.0);
        assert_eq!(sum.extract(2), 5.0);
        assert_eq!(sum.extract(3), 5.0);

        let prod = a.mul(b);
        assert_eq!(prod.extract(0), 6.0);

        // Test FMA
        let c = F64x4::splat(1.0);
        let fma = a.mul_add(b, c); // 2*3 + 1 = 7
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_f64x4_reduce() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            #[repr(C, align(32))]
            struct Aligned([f64; 4]);
            let data = Aligned([1.0f64, 2.0, 3.0, 4.0]);
            let v = F64x4::load_aligned(data.0.as_ptr());
            assert_eq!(v.reduce_sum(), 10.0);
            assert_eq!(v.reduce_max(), 4.0);
            assert_eq!(v.reduce_min(), 1.0);
        }
    }

    #[test]
    fn test_f32x8_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = F32x8::splat(2.0);
        let b = F32x8::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);

        let fma = a.mul_add(b, F32x8::splat(1.0));
        assert_eq!(fma.extract(0), 7.0);
    }

    #[test]
    fn test_load_store() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        unsafe {
            let src = [1.0f64, 2.0, 3.0, 4.0];
            let mut dst = [0.0f64; 4];

            let v = F64x4::load_unaligned(src.as_ptr());
            v.store_unaligned(dst.as_mut_ptr());

            assert_eq!(src, dst);
        }
    }

    // AVX-512BW tests
    #[test]
    fn test_i16x32_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        // Test using fallback implementations (transmute-based)
        let a = I16x32::splat(2);
        let b = I16x32::splat(3);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5);
        assert_eq!(sum.extract(15), 5);
        assert_eq!(sum.extract(31), 5);

        let prod = a.mullo(b);
        assert_eq!(prod.extract(0), 6);

        // Test reduce_add
        let ones = I16x32::splat(1);
        assert_eq!(ones.reduce_add(), 32);

        // Test abs
        let neg = I16x32::splat(-5);
        let abs = neg.abs();
        assert_eq!(abs.extract(0), 5);
    }

    #[test]
    fn test_i8x64_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let a = I8x64::splat(2);
        let b = I8x64::splat(3);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5);
        assert_eq!(sum.extract(63), 5);

        // Test abs
        let neg = I8x64::splat(-5);
        let abs = neg.abs();
        assert_eq!(abs.extract(0), 5);

        // Test reduce_add
        let ones = I8x64::splat(1);
        assert_eq!(ones.reduce_add(), 64);
    }

    #[test]
    fn test_u8x64_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let a = U8x64::splat(200);
        let b = U8x64::splat(100);

        let min = a.min(b);
        assert_eq!(min.extract(0), 100);

        let max = a.max(b);
        assert_eq!(max.extract(0), 200);

        // Test saturating add (should saturate at 255)
        let sat_add = a.adds(b);
        assert_eq!(sat_add.extract(0), 255);

        // Test reduce_add
        let ones = U8x64::splat(1);
        assert_eq!(ones.reduce_add(), 64);
    }

    // AVX-512VNNI tests
    #[test]
    fn test_i32x16_basic() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let a = I32x16::splat(2);
        let b = I32x16::splat(3);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5);
        assert_eq!(sum.extract(15), 5);

        let prod = a.mullo(b);
        assert_eq!(prod.extract(0), 6);
    }

    #[test]
    fn test_vnni_dpbusd_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        // Test the fallback implementation
        let acc = I32x16::zero();

        // Create test vectors: 4 elements of u8 and i8 per i32 output lane
        let a_data: [u8; 64] = [1; 64];
        let b_data: [i8; 64] = [2; 64];

        let a = unsafe { U8x64::load_unaligned(a_data.as_ptr()) };
        let b = unsafe { I8x64::load_unaligned(b_data.as_ptr()) };

        let result = acc.dpbusd_fallback(a, b);

        // Each lane should be: 1*2 + 1*2 + 1*2 + 1*2 = 8
        assert_eq!(result.extract(0), 8);
        assert_eq!(result.extract(15), 8);
    }

    #[test]
    fn test_vnni_dpwssd_fallback() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let acc = I32x16::zero();

        // Create test vectors: 2 elements of i16 per i32 output lane
        let a_data: [i16; 32] = [3; 32];
        let b_data: [i16; 32] = [4; 32];

        let a = unsafe { I16x32::load_unaligned(a_data.as_ptr()) };
        let b = unsafe { I16x32::load_unaligned(b_data.as_ptr()) };

        let result = acc.dpwssd_fallback(a, b);

        // Each lane should be: 3*4 + 3*4 = 24
        assert_eq!(result.extract(0), 24);
        assert_eq!(result.extract(15), 24);
    }

    #[test]
    fn test_avx512_feature_detection() {
        // Just test that feature detection doesn't panic
        let _bw = Avx512Features::has_avx512bw();
        let _vnni = Avx512Features::has_avx512vnni();
        let _vbmi = Avx512Features::has_avx512vbmi();
        let _dq = Avx512Features::has_avx512dq();
        let _vl = Avx512Features::has_avx512vl();
        let _full = Avx512Features::has_full_avx512();

        println!(
            "AVX-512 features: BW={}, VNNI={}, DQ={}, VL={}, Full={}",
            _bw, _vnni, _dq, _vl, _full
        );
    }

    // =========================================================================
    // Regression tests for the soundness / SSE2 / cold-panic fixes
    // =========================================================================

    /// #3: the SSE 128-bit `reduce_sum` must give the correct horizontal sum
    /// using only SSE2 instructions (no SSE3 `haddpd`/`haddps`).
    #[test]
    fn test_sse_reduce_sum_sse2_only() {
        unsafe {
            let d64 = [1.5f64, -2.5];
            let v = F64x2Sse::load_unaligned(d64.as_ptr());
            assert_eq!(v.reduce_sum(), -1.0);

            let d32 = [1.0f32, 2.0, 3.0, 4.0];
            let w = F32x4Sse::load_unaligned(d32.as_ptr());
            assert_eq!(w.reduce_sum(), 10.0);

            let d32b = [0.5f32, 0.25, 0.125, 0.0625];
            let wb = F32x4Sse::load_unaligned(d32b.as_ptr());
            assert_eq!(wb.reduce_sum(), 0.9375);
        }
    }

    /// #4: `extract`/`insert` must address every in-range lane correctly (a
    /// stale bound would corrupt a lane) and must panic (not UB) out of range.
    #[test]
    fn test_extract_insert_all_lanes() {
        let base = F32x4Sse::splat(0.0);
        let mut v = base;
        for i in 0..F32x4Sse::LANES {
            v = v.insert(i, i as f32 + 1.0);
        }
        for i in 0..F32x4Sse::LANES {
            assert_eq!(v.extract(i), i as f32 + 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_extract_out_of_range_panics() {
        let v = F64x2Sse::splat(1.0);
        let _ = v.extract(2);
    }

    /// #1: on a SIMD-capable host the intrinsic path is taken; forcing the
    /// scalar fallback must reproduce the *same* results for exact inputs. This
    /// actually executes the otherwise-dead fallback branches (add/mul/fma/
    /// reduce) for every feature-gated tier.
    #[test]
    fn test_avx2_fallback_matches_intrinsics() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let a_lanes = [1.0f64, 2.0, 3.0, 4.0];
        let b_lanes = [0.5f64, 1.5, 2.5, 3.5];
        let c_lanes = [10.0f64, 20.0, 30.0, 40.0];
        unsafe {
            let a = F64x4::load_unaligned(a_lanes.as_ptr());
            let b = F64x4::load_unaligned(b_lanes.as_ptr());
            let c = F64x4::load_unaligned(c_lanes.as_ptr());

            let simd_add = a.add(b);
            let simd_mul = a.mul(b);
            let simd_fma = a.mul_add(b, c);
            let simd_sum = a.reduce_sum();
            let simd_max = a.reduce_max();
            let simd_min = a.reduce_min();

            set_force_scalar_fallback(true);
            // The load/splat constructors must also survive the fallback path.
            let a_fb = F64x4::load_unaligned(a_lanes.as_ptr());
            let b_fb = F64x4::load_unaligned(b_lanes.as_ptr());
            let c_fb = F64x4::load_unaligned(c_lanes.as_ptr());
            assert_eq!(a_fb.add(b_fb).extract(0), simd_add.extract(0));
            assert_eq!(a_fb.add(b_fb).extract(3), simd_add.extract(3));
            assert_eq!(a_fb.mul(b_fb).extract(2), simd_mul.extract(2));
            assert_eq!(a_fb.mul_add(b_fb, c_fb).extract(1), simd_fma.extract(1));
            assert_eq!(a_fb.reduce_sum(), simd_sum);
            assert_eq!(a_fb.reduce_max(), simd_max);
            assert_eq!(a_fb.reduce_min(), simd_min);
            set_force_scalar_fallback(false);
        }
    }

    #[test]
    fn test_avx512_fallback_matches_intrinsics() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let a_lanes = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_lanes = [8.0f64, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let c_lanes = [0.5f64, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        unsafe {
            let a = F64x8::load_unaligned(a_lanes.as_ptr());
            let b = F64x8::load_unaligned(b_lanes.as_ptr());
            let c = F64x8::load_unaligned(c_lanes.as_ptr());

            let simd_add = a.add(b);
            let simd_sub = a.sub(b);
            let simd_fma = a.mul_add(b, c);
            let simd_sum = a.reduce_sum();
            let simd_max = a.reduce_max();
            let simd_min = a.reduce_min();

            let mut store_simd = [0.0f64; 8];
            a.store_unaligned(store_simd.as_mut_ptr());

            set_force_scalar_fallback(true);
            let a_fb = F64x8::load_unaligned(a_lanes.as_ptr());
            let b_fb = F64x8::load_unaligned(b_lanes.as_ptr());
            let c_fb = F64x8::load_unaligned(c_lanes.as_ptr());
            for i in 0..8 {
                assert_eq!(a_fb.add(b_fb).extract(i), simd_add.extract(i));
                assert_eq!(a_fb.sub(b_fb).extract(i), simd_sub.extract(i));
                assert_eq!(a_fb.mul_add(b_fb, c_fb).extract(i), simd_fma.extract(i));
            }
            assert_eq!(a_fb.reduce_sum(), simd_sum);
            assert_eq!(a_fb.reduce_max(), simd_max);
            assert_eq!(a_fb.reduce_min(), simd_min);

            let mut store_fb = [0.0f64; 8];
            a_fb.store_unaligned(store_fb.as_mut_ptr());
            assert_eq!(store_simd, store_fb);
            set_force_scalar_fallback(false);
        }
    }

    /// #5 (and #1): the AVX-512 `blend` and masked load/store must be correct
    /// on both the intrinsic and the fallback path.
    #[test]
    fn test_avx512_mask_ops_fallback_matches() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let a = F64x8::splat(1.0);
        let b = F64x8::splat(2.0);
        // Select `a` on even lanes, `b` on odd lanes.
        let mask: __mmask8 = 0b0101_0101;
        let simd_blend = <F64x8 as SimdMask>::blend(mask, a, b);

        set_force_scalar_fallback(true);
        let fb_blend = <F64x8 as SimdMask>::blend(mask, a, b);
        set_force_scalar_fallback(false);

        for i in 0..8 {
            assert_eq!(simd_blend.extract(i), fb_blend.extract(i));
            let expected = if (mask >> i) & 1 == 1 { 1.0 } else { 2.0 };
            assert_eq!(simd_blend.extract(i), expected);
        }
    }

    /// FMA fused-op fallbacks must implement the right algebraic identities.
    #[test]
    fn test_avx2_fma_variants_fallback() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        set_force_scalar_fallback(true);
        let s = F64x4::splat(2.0);
        let a = F64x4::splat(3.0);
        let b = F64x4::splat(4.0);
        assert_eq!(s.mul_add(a, b).extract(0), 10.0); // 2*3 + 4
        assert_eq!(s.mul_sub(a, b).extract(0), 2.0); // 2*3 - 4
        assert_eq!(s.neg_mul_add(a, b).extract(0), -2.0); // -(2*3) + 4
        set_force_scalar_fallback(false);
    }
}

