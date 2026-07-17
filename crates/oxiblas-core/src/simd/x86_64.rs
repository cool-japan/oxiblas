//! x86_64 SIMD implementations using SSE4.2, AVX2, and AVX512.
//!
//! This module provides SIMD register types and operations for x86_64
//! processors. It includes:
//! - SSE4.2 (128-bit): F64x2Sse, F32x4Sse
//! - AVX2 (256-bit): F64x4, F32x8
//! - AVX512 (512-bit): F64x8, F32x16

// Note: This module is only included when target_arch = "x86_64" (see simd.rs)

// Allow these clippy lints for SIMD code:
// - should_implement_trait: We use add/sub/mul/neg methods on trait implementations
// - missing_transmute_annotations: Transmutes in SIMD are clear from context
// - incompatible_msrv: AVX-512 intrinsics require newer Rust but we gate with runtime detection
// - needless_range_loop: Index-based loops are clearer for SIMD element access patterns
#![allow(clippy::should_implement_trait)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::needless_range_loop)]

use crate::simd::{SimdMask, SimdRegister, SimdScalar};
use core::arch::x86_64::*;

// =============================================================================
// Runtime feature detection and soundness scaffolding
// =============================================================================
//
// The 256-bit (AVX2 + FMA) and 512-bit (AVX-512F) register tiers execute
// instructions that are *not* part of the x86-64 baseline. Constructing or
// operating on those registers on a CPU that lacks the feature is undefined
// behavior. The `SimdRegister` trait is deliberately kept *safe* (matching the
// scalar / NEON / WASM tiers), so soundness cannot be delegated to the caller
// via an `unsafe` constructor. Instead, every safe entry point that would emit
// a feature-gated instruction routes through the `has_*` predicates below and
// falls back to a portable scalar implementation when the feature is absent.
//
// This is the "safe factory that checks the feature before touching the
// intrinsic" model: the branch is on a value that `is_x86_feature_detected!`
// caches after the first call (or a compile-time constant under `no_std`), so
// it is exhaustively branch-predicted and -- with `target_feature` compiled in
// -- folds away entirely, while guaranteeing that no unsupported instruction
// can ever be reached from safe code.

// Test-only switch that forces the scalar fallback path.
//
// On a SIMD-capable CI host the `else` branches would otherwise be dead code;
// flipping this per-thread lets the regression tests execute and validate the
// fallback implementations against the intrinsic path. `thread_local` (not a
// global atomic) keeps the override isolated to the setting thread so the
// parallel test runner cannot cross-contaminate.
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
fn force_scalar_fallback() -> bool {
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
fn has_avx2_fma() -> bool {
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
fn has_avx512f() -> bool {
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
fn has_avx512bw() -> bool {
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
fn has_avx512vnni() -> bool {
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
fn has_avx512vbmi() -> bool {
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
fn has_avx512dq() -> bool {
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
fn has_avx512vl() -> bool {
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
fn lane_index_out_of_range(index: usize, lanes: usize) -> ! {
    panic!("SIMD lane index {index} out of range (register has {lanes} lanes)");
}

// -----------------------------------------------------------------------------
// Generic scalar fallbacks
// -----------------------------------------------------------------------------
//
// # Safety contract for the helpers below
//
// Every helper reinterprets a register `R` as `[S; N]` (and back) with
// `transmute_copy`. Callers must uphold that `R` is a `#[repr(transparent)]`
// wrapper over a SIMD vector whose in-memory representation is exactly `N`
// contiguous `S` lanes, i.e. `size_of::<R>() == N * size_of::<S>()` and every
// bit pattern is a valid `S` (all lanes here are `f32`/`f64`, for which this
// holds). `transmute_copy` performs an unaligned read when the destination has
// stricter alignment, so no alignment invariant is imposed on `R`.

/// Broadcasts `value` into every lane without touching a SIMD instruction.
#[inline]
unsafe fn scalar_splat<R: Copy, S: Copy, const N: usize>(value: S) -> R {
    let arr = [value; N];
    core::mem::transmute_copy(&arr)
}

/// Loads `N` lanes of `S` from `ptr` (unaligned) without a SIMD instruction.
///
/// # Safety
/// `ptr` must be valid for reading `N` contiguous `S` values, in addition to
/// the module-level representation contract on `R`.
#[inline]
unsafe fn scalar_load<R: Copy, S: Copy, const N: usize>(ptr: *const S) -> R {
    let arr: [S; N] = core::array::from_fn(|i| unsafe { ptr.add(i).read_unaligned() });
    core::mem::transmute_copy(&arr)
}

/// Stores the `N` lanes of `value` to `ptr` (unaligned) without a SIMD store.
///
/// # Safety
/// `ptr` must be valid for writing `N` contiguous `S` values, in addition to
/// the module-level representation contract on `R`.
#[inline]
unsafe fn scalar_store<R: Copy, S: Copy, const N: usize>(value: R, ptr: *mut S) {
    let arr: [S; N] = core::mem::transmute_copy(&value);
    for i in 0..N {
        unsafe { ptr.add(i).write_unaligned(arr[i]) };
    }
}

/// Applies `f` lane-wise to two registers without a SIMD instruction.
#[inline]
unsafe fn scalar_binop<R: Copy, S: Copy, const N: usize>(a: R, b: R, f: impl Fn(S, S) -> S) -> R {
    let aa: [S; N] = core::mem::transmute_copy(&a);
    let bb: [S; N] = core::mem::transmute_copy(&b);
    let rr: [S; N] = core::array::from_fn(|i| f(aa[i], bb[i]));
    core::mem::transmute_copy(&rr)
}

/// Applies `f` lane-wise to three registers (fused-op fallback) scalar-only.
#[inline]
unsafe fn scalar_ternop<R: Copy, S: Copy, const N: usize>(
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
unsafe fn scalar_reduce<R: Copy, S: Copy, const N: usize>(a: R, f: impl Fn(S, S) -> S) -> S {
    let aa: [S; N] = core::mem::transmute_copy(&a);
    let mut acc = aa[0];
    for i in 1..N {
        acc = f(acc, aa[i]);
    }
    acc
}

// =============================================================================
// SSE4.2 (128-bit) implementations
// =============================================================================

/// 128-bit SIMD register for f64 (2 lanes) using SSE.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x2Sse(__m128d);

impl SimdRegister for F64x2Sse {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        unsafe { F64x2Sse(_mm_setzero_pd()) }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        unsafe { F64x2Sse(_mm_set1_pd(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        F64x2Sse(_mm_load_pd(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        F64x2Sse(_mm_loadu_pd(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        _mm_store_pd(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        _mm_storeu_pd(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_add_pd(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_sub_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_mul_pd(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F64x2Sse(_mm_div_pd(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // SSE doesn't have native FMA, emulate it
        // If FMA is available, use it
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).add(b)
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fmsub_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).sub(b)
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F64x2Sse(_mm_fnmadd_pd(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            b.sub(self.mul(a))
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        // SSE2-only: the 128-bit tier is granted on bare SSE2, but `haddpd` is
        // an SSE3 instruction. Bring the high lane down with `unpckhpd` (SSE2)
        // and add the two low doubles with `addsd` (SSE2).
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0); // [a1, a1]
            let sum = _mm_add_sd(self.0, high); // low lane = a0 + a1
            _mm_cvtsd_f64(sum)
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0);
            let max = _mm_max_pd(self.0, high);
            _mm_cvtsd_f64(max)
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        unsafe {
            let high = _mm_unpackhi_pd(self.0, self.0);
            let min = _mm_min_pd(self.0, high);
            _mm_cvtsd_f64(min)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        let arr: [f64; 2] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        let mut arr: [f64; 2] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F64x2Sse(unsafe { core::mem::transmute(arr) })
    }
}

/// 128-bit SIMD register for f32 (4 lanes) using SSE.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x4Sse(__m128);

impl SimdRegister for F32x4Sse {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        unsafe { F32x4Sse(_mm_setzero_ps()) }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        unsafe { F32x4Sse(_mm_set1_ps(value)) }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        F32x4Sse(_mm_load_ps(ptr))
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        F32x4Sse(_mm_loadu_ps(ptr))
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        _mm_store_ps(ptr, self.0);
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        _mm_storeu_ps(ptr, self.0);
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_add_ps(self.0, other.0)) }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_sub_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_mul_ps(self.0, other.0)) }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        unsafe { F32x4Sse(_mm_div_ps(self.0, other.0)) }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fmadd_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).add(b)
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fmsub_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.mul(a).sub(b)
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fma")]
        unsafe {
            F32x4Sse(_mm_fnmadd_ps(self.0, a.0, b.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            b.sub(self.mul(a))
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        // SSE2-only: `haddps` is SSE3. Reduce [a0,a1,a2,a3] using `movhlps`
        // (SSE) to fold the upper pair onto the lower pair, then a shuffle +
        // scalar add (SSE/SSE2) to combine the remaining two partial sums.
        unsafe {
            let high = _mm_movehl_ps(self.0, self.0); // [a2, a3, a2, a3]
            let sum1 = _mm_add_ps(self.0, high); // lane0 = a0+a2, lane1 = a1+a3
            let shuf = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01); // lane0 = a1+a3
            let sum2 = _mm_add_ss(sum1, shuf); // lane0 = (a0+a2)+(a1+a3)
            _mm_cvtss_f32(sum2)
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        unsafe {
            let shuffled = _mm_shuffle_ps(self.0, self.0, 0b10_11_00_01);
            let max1 = _mm_max_ps(self.0, shuffled);
            let shuffled2 = _mm_shuffle_ps(max1, max1, 0b00_00_10_10);
            let max2 = _mm_max_ps(max1, shuffled2);
            _mm_cvtss_f32(max2)
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        unsafe {
            let shuffled = _mm_shuffle_ps(self.0, self.0, 0b10_11_00_01);
            let min1 = _mm_min_ps(self.0, shuffled);
            let shuffled2 = _mm_shuffle_ps(min1, min1, 0b00_00_10_10);
            let min2 = _mm_min_ps(min1, shuffled2);
            _mm_cvtss_f32(min2)
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        let arr: [f32; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let mut arr: [f32; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F32x4Sse(unsafe { core::mem::transmute(arr) })
    }
}

/// 128-bit SIMD register type alias for compatibility.
pub type Simd128F64 = F64x2Sse;
/// 128-bit SIMD register type alias for compatibility.
pub type Simd128F32 = F32x4Sse;

// =============================================================================
// AVX2 (256-bit) implementations
// =============================================================================

/// 256-bit SIMD register for f64 (4 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x4(__m256d);

impl SimdRegister for F64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_setzero_pd()) }
        } else {
            unsafe { scalar_splat::<F64x4, f64, 4>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_set1_pd(value)) }
        } else {
            unsafe { scalar_splat::<F64x4, f64, 4>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        if has_avx2_fma() {
            F64x4(_mm256_load_pd(ptr))
        } else {
            scalar_load::<F64x4, f64, 4>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        if has_avx2_fma() {
            F64x4(_mm256_loadu_pd(ptr))
        } else {
            scalar_load::<F64x4, f64, 4>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        if has_avx2_fma() {
            _mm256_store_pd(ptr, self.0);
        } else {
            scalar_store::<F64x4, f64, 4>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        if has_avx2_fma() {
            _mm256_storeu_pd(ptr, self.0);
        } else {
            scalar_store::<F64x4, f64, 4>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_add_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_sub_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_mul_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_div_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x4, f64, 4>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // FMA: self * a + b. The scalar fallback uses two roundings (mul then
        // add), matching the SSE non-FMA path; `core`'s single-rounding
        // `f64::mul_add` is `std`-only, so it cannot be used on the no_std path.
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_fmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x4, f64, 4>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        // FMA: self * a - b
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_fmsub_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x4, f64, 4>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        // FMA: -(self * a) + b = b - self * a
        if has_avx2_fma() {
            unsafe { F64x4(_mm256_fnmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x4, f64, 4>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        if has_avx2_fma() {
            unsafe {
                // Horizontal add: [a0+a1, a2+a3, a0+a1, a2+a3]
                let sum1 = _mm256_hadd_pd(self.0, self.0);
                // Extract high 128 bits and add to low 128 bits
                let high = _mm256_extractf128_pd(sum1, 1);
                let low = _mm256_castpd256_pd128(sum1);
                let sum2 = _mm_add_pd(low, high);
                _mm_cvtsd_f64(sum2)
            }
        } else {
            unsafe { scalar_reduce::<F64x4, f64, 4>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        if has_avx2_fma() {
            unsafe {
                // Compare and take max of pairs
                let high = _mm256_extractf128_pd(self.0, 1);
                let low = _mm256_castpd256_pd128(self.0);
                let max1 = _mm_max_pd(low, high);
                // Shuffle and compare again
                let max2 = _mm_unpackhi_pd(max1, max1);
                let max3 = _mm_max_pd(max1, max2);
                _mm_cvtsd_f64(max3)
            }
        } else {
            unsafe { scalar_reduce::<F64x4, f64, 4>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        if has_avx2_fma() {
            unsafe {
                let high = _mm256_extractf128_pd(self.0, 1);
                let low = _mm256_castpd256_pd128(self.0);
                let min1 = _mm_min_pd(low, high);
                let min2 = _mm_unpackhi_pd(min1, min1);
                let min3 = _mm_min_pd(min1, min2);
                _mm_cvtsd_f64(min3)
            }
        } else {
            unsafe { scalar_reduce::<F64x4, f64, 4>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        let arr: [f64; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        let mut arr: [f64; 4] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F64x4(unsafe { core::mem::transmute(arr) })
    }
}

/// 256-bit SIMD register for f32 (8 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x8(__m256);

impl SimdRegister for F32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_setzero_ps()) }
        } else {
            unsafe { scalar_splat::<F32x8, f32, 8>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_set1_ps(value)) }
        } else {
            unsafe { scalar_splat::<F32x8, f32, 8>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        if has_avx2_fma() {
            F32x8(_mm256_load_ps(ptr))
        } else {
            scalar_load::<F32x8, f32, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        if has_avx2_fma() {
            F32x8(_mm256_loadu_ps(ptr))
        } else {
            scalar_load::<F32x8, f32, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        if has_avx2_fma() {
            _mm256_store_ps(ptr, self.0);
        } else {
            scalar_store::<F32x8, f32, 8>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        if has_avx2_fma() {
            _mm256_storeu_ps(ptr, self.0);
        } else {
            scalar_store::<F32x8, f32, 8>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_add_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_sub_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_mul_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_div_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x8, f32, 8>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_fmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x8, f32, 8>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_fmsub_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x8, f32, 8>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        if has_avx2_fma() {
            unsafe { F32x8(_mm256_fnmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x8, f32, 8>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        if has_avx2_fma() {
            unsafe {
                // Horizontal add pairs
                let sum1 = _mm256_hadd_ps(self.0, self.0);
                let sum2 = _mm256_hadd_ps(sum1, sum1);
                // Extract and add high/low halves
                let high = _mm256_extractf128_ps(sum2, 1);
                let low = _mm256_castps256_ps128(sum2);
                let sum3 = _mm_add_ps(low, high);
                _mm_cvtss_f32(sum3)
            }
        } else {
            unsafe { scalar_reduce::<F32x8, f32, 8>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        if has_avx2_fma() {
            unsafe {
                let high = _mm256_extractf128_ps(self.0, 1);
                let low = _mm256_castps256_ps128(self.0);
                let max1 = _mm_max_ps(low, high);
                // Shuffle and compare
                let max2 = _mm_shuffle_ps(max1, max1, 0b10_11_00_01);
                let max3 = _mm_max_ps(max1, max2);
                let max4 = _mm_shuffle_ps(max3, max3, 0b00_00_10_10);
                let max5 = _mm_max_ps(max3, max4);
                _mm_cvtss_f32(max5)
            }
        } else {
            unsafe { scalar_reduce::<F32x8, f32, 8>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        if has_avx2_fma() {
            unsafe {
                let high = _mm256_extractf128_ps(self.0, 1);
                let low = _mm256_castps256_ps128(self.0);
                let min1 = _mm_min_ps(low, high);
                let min2 = _mm_shuffle_ps(min1, min1, 0b10_11_00_01);
                let min3 = _mm_min_ps(min1, min2);
                let min4 = _mm_shuffle_ps(min3, min3, 0b00_00_10_10);
                let min5 = _mm_min_ps(min3, min4);
                _mm_cvtss_f32(min5)
            }
        } else {
            unsafe { scalar_reduce::<F32x8, f32, 8>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        let arr: [f32; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let mut arr: [f32; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F32x8(unsafe { core::mem::transmute(arr) })
    }
}

// =============================================================================
// AVX512 (512-bit) implementations
// =============================================================================

/// 512-bit SIMD register for f64 (8 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64x8(__m512d);

impl SimdRegister for F64x8 {
    type Scalar = f64;
    const LANES: usize = 8;

    #[inline]
    fn zero() -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_setzero_pd()) }
        } else {
            unsafe { scalar_splat::<F64x8, f64, 8>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f64) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_set1_pd(value)) }
        } else {
            unsafe { scalar_splat::<F64x8, f64, 8>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        if has_avx512f() {
            F64x8(_mm512_load_pd(ptr))
        } else {
            scalar_load::<F64x8, f64, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        if has_avx512f() {
            F64x8(_mm512_loadu_pd(ptr))
        } else {
            scalar_load::<F64x8, f64, 8>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        if has_avx512f() {
            _mm512_store_pd(ptr, self.0);
        } else {
            scalar_store::<F64x8, f64, 8>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        if has_avx512f() {
            _mm512_storeu_pd(ptr, self.0);
        } else {
            scalar_store::<F64x8, f64, 8>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_add_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_sub_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_mul_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_div_pd(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F64x8, f64, 8>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_fmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x8, f64, 8>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_fmsub_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x8, f64, 8>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F64x8(_mm512_fnmadd_pd(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F64x8, f64, 8>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        if has_avx512f() {
            unsafe { _mm512_reduce_add_pd(self.0) }
        } else {
            unsafe { scalar_reduce::<F64x8, f64, 8>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        if has_avx512f() {
            unsafe { _mm512_reduce_max_pd(self.0) }
        } else {
            unsafe { scalar_reduce::<F64x8, f64, 8>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        if has_avx512f() {
            unsafe { _mm512_reduce_min_pd(self.0) }
        } else {
            unsafe { scalar_reduce::<F64x8, f64, 8>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f64 {
        let arr: [f64; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f64) -> Self {
        let mut arr: [f64; 8] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F64x8(unsafe { core::mem::transmute(arr) })
    }
}

/// 512-bit SIMD register for f32 (16 lanes).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x16(__m512);

impl SimdRegister for F32x16 {
    type Scalar = f32;
    const LANES: usize = 16;

    #[inline]
    fn zero() -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_setzero_ps()) }
        } else {
            unsafe { scalar_splat::<F32x16, f32, 16>(0.0) }
        }
    }

    #[inline]
    fn splat(value: f32) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_set1_ps(value)) }
        } else {
            unsafe { scalar_splat::<F32x16, f32, 16>(value) }
        }
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        if has_avx512f() {
            F32x16(_mm512_load_ps(ptr))
        } else {
            scalar_load::<F32x16, f32, 16>(ptr)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        if has_avx512f() {
            F32x16(_mm512_loadu_ps(ptr))
        } else {
            scalar_load::<F32x16, f32, 16>(ptr)
        }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        if has_avx512f() {
            _mm512_store_ps(ptr, self.0);
        } else {
            scalar_store::<F32x16, f32, 16>(self, ptr);
        }
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        if has_avx512f() {
            _mm512_storeu_ps(ptr, self.0);
        } else {
            scalar_store::<F32x16, f32, 16>(self, ptr);
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_add_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x + y) }
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_sub_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x - y) }
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_mul_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x * y) }
        }
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_div_ps(self.0, other.0)) }
        } else {
            unsafe { scalar_binop::<F32x16, f32, 16>(self, other, |x, y| x / y) }
        }
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_fmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x16, f32, 16>(self, a, b, |s, av, bv| s * av + bv) }
        }
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_fmsub_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x16, f32, 16>(self, a, b, |s, av, bv| s * av - bv) }
        }
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_fnmadd_ps(self.0, a.0, b.0)) }
        } else {
            unsafe { scalar_ternop::<F32x16, f32, 16>(self, a, b, |s, av, bv| bv - s * av) }
        }
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        if has_avx512f() {
            unsafe { _mm512_reduce_add_ps(self.0) }
        } else {
            unsafe { scalar_reduce::<F32x16, f32, 16>(self, |x, y| x + y) }
        }
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        if has_avx512f() {
            unsafe { _mm512_reduce_max_ps(self.0) }
        } else {
            unsafe { scalar_reduce::<F32x16, f32, 16>(self, |x, y| if x >= y { x } else { y }) }
        }
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        if has_avx512f() {
            unsafe { _mm512_reduce_min_ps(self.0) }
        } else {
            unsafe { scalar_reduce::<F32x16, f32, 16>(self, |x, y| if x <= y { x } else { y }) }
        }
    }

    #[inline]
    fn extract(self, index: usize) -> f32 {
        let arr: [f32; 16] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
    }

    #[inline]
    fn insert(self, index: usize, value: f32) -> Self {
        let mut arr: [f32; 16] = unsafe { core::mem::transmute(self.0) };
        match arr.get_mut(index) {
            Some(slot) => *slot = value,
            None => lane_index_out_of_range(index, Self::LANES),
        }
        F32x16(unsafe { core::mem::transmute(arr) })
    }
}

// =============================================================================
// SimdScalar implementations
// =============================================================================

impl SimdScalar for f64 {
    type Simd256 = F64x4;
    type Simd512 = F64x8;
}

impl SimdScalar for f32 {
    type Simd256 = F32x8;
    type Simd512 = F32x16;
}

// =============================================================================
// Masked operations for AVX512
// =============================================================================

impl SimdMask for F64x8 {
    type Mask = __mmask8;

    #[inline]
    fn mask_from_bools(bools: &[bool]) -> Self::Mask {
        let mut mask: u8 = 0;
        for (i, &b) in bools.iter().take(8).enumerate() {
            if b {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline]
    unsafe fn load_masked(ptr: *const f64, mask: Self::Mask, default: Self) -> Self {
        if has_avx512f() {
            F64x8(_mm512_mask_loadu_pd(default.0, mask, ptr))
        } else {
            let dd: [f64; 8] = core::mem::transmute(default.0);
            let rr: [f64; 8] = core::array::from_fn(|i| {
                if (mask >> i) & 1 == 1 {
                    unsafe { ptr.add(i).read_unaligned() }
                } else {
                    dd[i]
                }
            });
            F64x8(core::mem::transmute(rr))
        }
    }

    #[inline]
    unsafe fn store_masked(self, ptr: *mut f64, mask: Self::Mask) {
        if has_avx512f() {
            _mm512_mask_storeu_pd(ptr, mask, self.0);
        } else {
            let vv: [f64; 8] = core::mem::transmute(self.0);
            for i in 0..8 {
                if (mask >> i) & 1 == 1 {
                    ptr.add(i).write_unaligned(vv[i]);
                }
            }
        }
    }

    #[inline]
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self {
        if has_avx512f() {
            // `_mm512_mask_blend_pd(k, x, y)` yields `k[i] ? y[i] : x[i]`; passing
            // `(mask, b, a)` therefore selects `a` where the mask bit is set.
            unsafe { F64x8(_mm512_mask_blend_pd(mask, b.0, a.0)) }
        } else {
            unsafe {
                let aa: [f64; 8] = core::mem::transmute(a.0);
                let bb: [f64; 8] = core::mem::transmute(b.0);
                let rr: [f64; 8] =
                    core::array::from_fn(|i| if (mask >> i) & 1 == 1 { aa[i] } else { bb[i] });
                F64x8(core::mem::transmute(rr))
            }
        }
    }
}

impl SimdMask for F32x16 {
    type Mask = __mmask16;

    #[inline]
    fn mask_from_bools(bools: &[bool]) -> Self::Mask {
        let mut mask: u16 = 0;
        for (i, &b) in bools.iter().take(16).enumerate() {
            if b {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline]
    unsafe fn load_masked(ptr: *const f32, mask: Self::Mask, default: Self) -> Self {
        if has_avx512f() {
            F32x16(_mm512_mask_loadu_ps(default.0, mask, ptr))
        } else {
            let dd: [f32; 16] = core::mem::transmute(default.0);
            let rr: [f32; 16] = core::array::from_fn(|i| {
                if (mask >> i) & 1 == 1 {
                    unsafe { ptr.add(i).read_unaligned() }
                } else {
                    dd[i]
                }
            });
            F32x16(core::mem::transmute(rr))
        }
    }

    #[inline]
    unsafe fn store_masked(self, ptr: *mut f32, mask: Self::Mask) {
        if has_avx512f() {
            _mm512_mask_storeu_ps(ptr, mask, self.0);
        } else {
            let vv: [f32; 16] = core::mem::transmute(self.0);
            for i in 0..16 {
                if (mask >> i) & 1 == 1 {
                    ptr.add(i).write_unaligned(vv[i]);
                }
            }
        }
    }

    #[inline]
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self {
        if has_avx512f() {
            unsafe { F32x16(_mm512_mask_blend_ps(mask, b.0, a.0)) }
        } else {
            unsafe {
                let aa: [f32; 16] = core::mem::transmute(a.0);
                let bb: [f32; 16] = core::mem::transmute(b.0);
                let rr: [f32; 16] =
                    core::array::from_fn(|i| if (mask >> i) & 1 == 1 { aa[i] } else { bb[i] });
                F32x16(core::mem::transmute(rr))
            }
        }
    }
}

// =============================================================================
// AVX-512BW (Byte/Word) implementations
// =============================================================================

/// 512-bit SIMD register for i16 (32 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I16x32(__m512i);

impl I16x32 {
    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { I16x32(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: i16) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn splat_impl(value: i16) -> __m512i {
            _mm512_set1_epi16(value)
        }
        unsafe { I16x32(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 32 valid i16 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn load_unaligned(ptr: *const i16) -> Self {
        I16x32(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 32 valid writable i16 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn store_unaligned(self, ptr: *mut i16) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi16(a, b)
        }
        unsafe { I16x32(add_impl(self.0, other.0)) }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn sub_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_sub_epi16(a, b)
        }
        unsafe { I16x32(sub_impl(self.0, other.0)) }
    }

    /// Element-wise multiplication (low 16 bits of each 32-bit product).
    #[inline]
    pub fn mullo(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn mullo_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_mullo_epi16(a, b)
        }
        unsafe { I16x32(mullo_impl(self.0, other.0)) }
    }

    /// Saturating addition.
    #[inline]
    pub fn adds(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn adds_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_adds_epi16(a, b)
        }
        unsafe { I16x32(adds_impl(self.0, other.0)) }
    }

    /// Saturating subtraction.
    #[inline]
    pub fn subs(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn subs_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_subs_epi16(a, b)
        }
        unsafe { I16x32(subs_impl(self.0, other.0)) }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn min_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_min_epi16(a, b)
        }
        unsafe { I16x32(min_impl(self.0, other.0)) }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn max_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_max_epi16(a, b)
        }
        unsafe { I16x32(max_impl(self.0, other.0)) }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn abs_impl(a: __m512i) -> __m512i {
            _mm512_abs_epi16(a)
        }
        unsafe { I16x32(abs_impl(self.0)) }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> i32 {
        // Sum in i32 to avoid overflow
        unsafe {
            let arr: [i16; 32] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as i32).sum()
        }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i16 {
        let arr: [i16; 32] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 32),
        }
    }
}

/// 512-bit SIMD register for i8 (64 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I8x64(__m512i);

impl I8x64 {
    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { I8x64(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: i8) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn splat_impl(value: i8) -> __m512i {
            _mm512_set1_epi8(value)
        }
        unsafe { I8x64(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid i8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn load_unaligned(ptr: *const i8) -> Self {
        I8x64(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid writable i8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn store_unaligned(self, ptr: *mut i8) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi8(a, b)
        }
        unsafe { I8x64(add_impl(self.0, other.0)) }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn sub_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_sub_epi8(a, b)
        }
        unsafe { I8x64(sub_impl(self.0, other.0)) }
    }

    /// Saturating addition.
    #[inline]
    pub fn adds(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn adds_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_adds_epi8(a, b)
        }
        unsafe { I8x64(adds_impl(self.0, other.0)) }
    }

    /// Saturating subtraction.
    #[inline]
    pub fn subs(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn subs_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_subs_epi8(a, b)
        }
        unsafe { I8x64(subs_impl(self.0, other.0)) }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn min_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_min_epi8(a, b)
        }
        unsafe { I8x64(min_impl(self.0, other.0)) }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn max_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_max_epi8(a, b)
        }
        unsafe { I8x64(max_impl(self.0, other.0)) }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn abs_impl(a: __m512i) -> __m512i {
            _mm512_abs_epi8(a)
        }
        unsafe { I8x64(abs_impl(self.0)) }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> i32 {
        unsafe {
            let arr: [i8; 64] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as i32).sum()
        }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i8 {
        let arr: [i8; 64] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 64),
        }
    }
}

/// 512-bit SIMD register for u8 (64 lanes) using AVX-512BW.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct U8x64(__m512i);

impl U8x64 {
    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { U8x64(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: u8) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn splat_impl(value: u8) -> __m512i {
            _mm512_set1_epi8(value as i8)
        }
        unsafe { U8x64(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid u8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn load_unaligned(ptr: *const u8) -> Self {
        U8x64(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 64 valid writable u8 elements.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub unsafe fn store_unaligned(self, ptr: *mut u8) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi8(a, b)
        }
        unsafe { U8x64(add_impl(self.0, other.0)) }
    }

    /// Saturating addition.
    #[inline]
    pub fn adds(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn adds_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_adds_epu8(a, b)
        }
        unsafe { U8x64(adds_impl(self.0, other.0)) }
    }

    /// Saturating subtraction.
    #[inline]
    pub fn subs(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn subs_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_subs_epu8(a, b)
        }
        unsafe { U8x64(subs_impl(self.0, other.0)) }
    }

    /// Element-wise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn min_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_min_epu8(a, b)
        }
        unsafe { U8x64(min_impl(self.0, other.0)) }
    }

    /// Element-wise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[target_feature(enable = "avx512bw")]
        unsafe fn max_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_max_epu8(a, b)
        }
        unsafe { U8x64(max_impl(self.0, other.0)) }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> u32 {
        unsafe {
            let arr: [u8; 64] = core::mem::transmute(self.0);
            arr.iter().map(|&x| x as u32).sum()
        }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> u8 {
        let arr: [u8; 64] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 64),
        }
    }
}

// =============================================================================
// AVX-512VNNI (Vector Neural Network Instructions)
// =============================================================================

/// AVX-512VNNI operations for neural network acceleration.
///
/// These operations are essential for quantized neural network inference.
pub struct Avx512Vnni;

impl Avx512Vnni {
    /// Checks if AVX-512VNNI is supported at runtime.
    #[inline]
    pub fn is_supported() -> bool {
        has_avx512vnni()
    }

    /// Dot product of 4-element vectors of u8 and i8, accumulated to i32.
    ///
    /// This performs: `dst[i] = src[i] + sum(a[i*4+j] * b[i*4+j])` for j in 0..4
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpbusd(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(src, a, b)
    }

    /// Dot product of 4-element u8*i8 vectors with saturation.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpbusds(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpbusds_epi32(src, a, b)
    }

    /// Dot product of 2-element i16 vectors accumulated to i32.
    ///
    /// This performs: `dst[i] = src[i] + sum(a[i*2+j] * b[i*2+j])` for j in 0..2
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpwssd(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpwssd_epi32(src, a, b)
    }

    /// Dot product of 2-element i16 vectors with saturation.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    #[target_feature(enable = "avx512vnni")]
    pub unsafe fn vpdpwssds(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_dpwssds_epi32(src, a, b)
    }
}

/// 512-bit integer vector for VNNI operations.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct I32x16(__m512i);

impl I32x16 {
    /// Number of lanes.
    pub const LANES: usize = 16;

    /// Creates a register with all lanes set to zero.
    #[inline]
    pub fn zero() -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn zero_impl() -> __m512i {
            _mm512_setzero_si512()
        }
        unsafe { I32x16(zero_impl()) }
    }

    /// Creates a register with all lanes set to the same value.
    #[inline]
    pub fn splat(value: i32) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn splat_impl(value: i32) -> __m512i {
            _mm512_set1_epi32(value)
        }
        unsafe { I32x16(splat_impl(value)) }
    }

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 16 valid i32 elements.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn load_unaligned(ptr: *const i32) -> Self {
        I32x16(_mm512_loadu_si512(ptr as *const __m512i))
    }

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least 16 valid writable i32 elements.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn store_unaligned(self, ptr: *mut i32) {
        _mm512_storeu_si512(ptr as *mut __m512i, self.0);
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn add_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi32(a, b)
        }
        unsafe { I32x16(add_impl(self.0, other.0)) }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn sub_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_sub_epi32(a, b)
        }
        unsafe { I32x16(sub_impl(self.0, other.0)) }
    }

    /// Element-wise multiplication (low 32 bits).
    #[inline]
    pub fn mullo(self, other: Self) -> Self {
        #[target_feature(enable = "avx512f")]
        unsafe fn mullo_impl(a: __m512i, b: __m512i) -> __m512i {
            _mm512_mullo_epi32(a, b)
        }
        unsafe { I32x16(mullo_impl(self.0, other.0)) }
    }

    /// VNNI dot product: u8 * i8 -> i32 accumulation.
    ///
    /// Computes 4-element dot products and adds to accumulator.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    pub unsafe fn dpbusd(self, a: U8x64, b: I8x64) -> Self {
        if Avx512Vnni::is_supported() {
            I32x16(Avx512Vnni::vpdpbusd(self.0, a.0, b.0))
        } else {
            // Fallback implementation
            self.dpbusd_fallback(a, b)
        }
    }

    /// Fallback implementation for VNNI dpbusd.
    #[inline]
    fn dpbusd_fallback(self, a: U8x64, b: I8x64) -> Self {
        unsafe {
            let a_arr: [u8; 64] = core::mem::transmute(a.0);
            let b_arr: [i8; 64] = core::mem::transmute(b.0);
            let mut result: [i32; 16] = core::mem::transmute(self.0);

            for i in 0..16 {
                let base = i * 4;
                for j in 0..4 {
                    result[i] += (a_arr[base + j] as i32) * (b_arr[base + j] as i32);
                }
            }
            I32x16(core::mem::transmute(result))
        }
    }

    /// VNNI dot product: i16 * i16 -> i32 accumulation.
    ///
    /// Computes 2-element dot products and adds to accumulator.
    ///
    /// # Safety
    /// Requires AVX-512VNNI support.
    #[inline]
    pub unsafe fn dpwssd(self, a: I16x32, b: I16x32) -> Self {
        if Avx512Vnni::is_supported() {
            I32x16(Avx512Vnni::vpdpwssd(self.0, a.0, b.0))
        } else {
            self.dpwssd_fallback(a, b)
        }
    }

    /// Fallback implementation for VNNI dpwssd.
    #[inline]
    fn dpwssd_fallback(self, a: I16x32, b: I16x32) -> Self {
        unsafe {
            let a_arr: [i16; 32] = core::mem::transmute(a.0);
            let b_arr: [i16; 32] = core::mem::transmute(b.0);
            let mut result: [i32; 16] = core::mem::transmute(self.0);

            for i in 0..16 {
                let base = i * 2;
                for j in 0..2 {
                    result[i] += (a_arr[base + j] as i32) * (b_arr[base + j] as i32);
                }
            }
            I32x16(core::mem::transmute(result))
        }
    }

    /// Horizontal sum of all lanes.
    #[inline]
    pub fn reduce_add(self) -> i32 {
        #[target_feature(enable = "avx512f")]
        unsafe fn reduce_impl(v: __m512i) -> i32 {
            _mm512_reduce_add_epi32(v)
        }
        unsafe { reduce_impl(self.0) }
    }

    /// Extracts a single lane.
    #[inline]
    pub fn extract(self, index: usize) -> i32 {
        let arr: [i32; 16] = unsafe { core::mem::transmute(self.0) };
        match arr.get(index) {
            Some(&value) => value,
            None => lane_index_out_of_range(index, 16),
        }
    }

    /// Get the raw __m512i register.
    #[inline]
    pub fn raw(self) -> __m512i {
        self.0
    }

    /// Create from raw __m512i register.
    #[inline]
    pub fn from_raw(v: __m512i) -> Self {
        I32x16(v)
    }
}

/// Feature detection for AVX-512 extensions.
pub struct Avx512Features;

impl Avx512Features {
    /// Check if AVX-512BW (Byte/Word) is supported.
    #[inline]
    pub fn has_avx512bw() -> bool {
        has_avx512bw()
    }

    /// Check if AVX-512VNNI (Vector Neural Network) is supported.
    #[inline]
    pub fn has_avx512vnni() -> bool {
        has_avx512vnni()
    }

    /// Check if AVX-512VBMI (Vector Byte Manipulation) is supported.
    #[inline]
    pub fn has_avx512vbmi() -> bool {
        has_avx512vbmi()
    }

    /// Check if AVX-512DQ (Doubleword and Quadword) is supported.
    #[inline]
    pub fn has_avx512dq() -> bool {
        has_avx512dq()
    }

    /// Check if AVX-512VL (Vector Length Extensions) is supported.
    #[inline]
    pub fn has_avx512vl() -> bool {
        has_avx512vl()
    }

    /// Check if all AVX-512 extensions needed for BLAS acceleration are supported.
    #[inline]
    pub fn has_full_avx512() -> bool {
        has_avx512f() && has_avx512bw() && has_avx512dq()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
