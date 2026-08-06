//! SIMD abstraction layer for OxiBLAS.
//!
//! This module provides a unified, general-purpose interface over
//! architecture-specific SIMD intrinsics from `core::arch`. It supports:
//! - x86_64: AVX2 (256-bit), AVX512F (512-bit), SSE4.2 (128-bit)
//! - AArch64: NEON (128-bit), 256-bit emulated
//! - WASM32: SIMD128 (128-bit), 256-bit emulated
//! - Scalar fallback for unsupported platforms
//!
//! The design uses runtime feature detection to dispatch to the best
//! available implementation.
//!
//! # Integration status (read this before assuming it is on the hot path)
//!
//! This is a *general-purpose* SIMD toolkit that is available for downstream
//! use, but it is **not** the code that powers OxiBLAS's tuned compute kernels.
//! To avoid advertising an integration that does not exist:
//!
//! - The BLAS/LAPACK compute kernels in `oxiblas-blas` each perform their own
//!   `is_x86_feature_detected!`-guarded dispatch and carry their own
//!   hand-written intrinsics. As of this writing that includes
//!   `level1::{dot, axpy, nrm2}`, `level2::gemv`, and
//!   `level3::{gemm, gemm_kernel, gemm_packing, gemm_small, gemm_kernel_sse42}`.
//!   None of them call the [`SimdRegister`]/[`SimdScalar`] register API below.
//! - What *is* consumed outside this module's own tests is the lightweight
//!   capability layer: [`SimdLevel`] and [`detect_simd_level`] /
//!   [`detect_simd_level_raw`], which `crate::tuning` uses for cache/block-size
//!   heuristics and which the crate re-exports.
//! - The register-level API ([`SimdRegister`], [`SimdScalar`], the concrete
//!   register types, and the [`complex`], [`dispatch`], [`multiver`], and
//!   [`scalar`] submodules) is currently exercised only by this crate's unit
//!   tests and the `simd` benchmark. It is correct and covered, but no kernel
//!   is wired to it yet.
//!
//! A downstream Level-1 loop would adopt this layer by pairing [`SimdChunks`]
//! (for head/body/tail splitting) with a [`SimdRegister`] type, as the
//! `test_abstraction_level1_dot_matches_scalar` regression test demonstrates.
//!
//! # Safety: the target-feature contract of the wide registers
//!
//! The 128-bit SSE register methods are sound on x86_64 because SSE2 is part of
//! the x86_64 baseline (the FMA path is selected at compile time via
//! `#[cfg(target_feature = "fma")]`). The **256-bit (AVX2) and 512-bit
//! (AVX-512) register methods are declared as safe `fn`s but emit AVX2/AVX-512
//! instructions without an internal runtime feature guard**. Executing those
//! instructions on a CPU that lacks the feature is undefined behavior, so a
//! caller MUST confirm the feature is present (e.g. via
//! `is_x86_feature_detected!("avx2")`) before constructing or operating on
//! those register types. This unguarded contract is the reason the tuned
//! kernels — and this module's own tests — gate every wide-register use behind
//! runtime detection rather than calling the abstraction blindly.
//!
//! # Complex SIMD
//!
//! The `complex` submodule provides SIMD types for complex numbers in
//! interleaved format `[re0, im0, re1, im1, ...]`.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "wasm32")]
pub mod wasm32;

pub mod complex;
pub mod dispatch;
pub mod multiver;
pub mod scalar;

use crate::scalar::{Field, Real, Scalar};

/// SIMD capability level detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD, scalar operations only
    Scalar,
    /// 128-bit SIMD (SSE2 on x86, NEON on ARM)
    Simd128,
    /// 256-bit SIMD (AVX2 on x86)
    Simd256,
    /// 512-bit SIMD (AVX512F on x86)
    Simd512,
}

impl SimdLevel {
    /// Returns the number of lanes for a given scalar type.
    #[inline]
    pub const fn lanes<T: Scalar>(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Simd128 => 16 / core::mem::size_of::<T>(),
            SimdLevel::Simd256 => 32 / core::mem::size_of::<T>(),
            SimdLevel::Simd512 => 64 / core::mem::size_of::<T>(),
        }
    }

    /// Returns the register width in bytes.
    #[inline]
    pub const fn width_bytes(self) -> usize {
        match self {
            SimdLevel::Scalar => 8, // Treat as 64-bit for alignment
            SimdLevel::Simd128 => 16,
            SimdLevel::Simd256 => 32,
            SimdLevel::Simd512 => 64,
        }
    }
}

/// Detects the best available SIMD level at runtime.
///
/// This function respects the following feature flags:
/// - `force-scalar`: Always returns `SimdLevel::Scalar` (useful for debugging)
/// - `max-simd-128`: Limits maximum to `SimdLevel::Simd128`
/// - `max-simd-256`: Limits maximum to `SimdLevel::Simd256`
#[inline]
pub fn detect_simd_level() -> SimdLevel {
    // Feature flag: force scalar operations (useful for debugging)
    #[cfg(feature = "force-scalar")]
    {
        SimdLevel::Scalar
    }

    #[cfg(not(feature = "force-scalar"))]
    {
        let detected = detect_simd_level_raw();

        // Apply maximum SIMD level limits from feature flags
        #[cfg(feature = "max-simd-128")]
        {
            return if detected > SimdLevel::Simd128 {
                SimdLevel::Simd128
            } else {
                detected
            };
        }

        #[cfg(feature = "max-simd-256")]
        #[cfg(not(feature = "max-simd-128"))]
        {
            return if detected > SimdLevel::Simd256 {
                SimdLevel::Simd256
            } else {
                detected
            };
        }

        #[cfg(not(any(feature = "max-simd-128", feature = "max-simd-256")))]
        {
            detected
        }
    }
}

/// Raw SIMD level detection without feature flag limits.
///
/// This is the internal detection function that returns the actual
/// hardware SIMD capability.
#[inline]
pub fn detect_simd_level_raw() -> SimdLevel {
    // On x86_64 with std, use runtime feature detection
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::Simd512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            SimdLevel::Simd256
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Simd128
        } else {
            SimdLevel::Scalar
        }
    }

    // On x86_64 without std, use compile-time target features only
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    {
        #[cfg(target_feature = "avx512f")]
        {
            SimdLevel::Simd512
        }
        #[cfg(all(
            target_feature = "avx2",
            target_feature = "fma",
            not(target_feature = "avx512f")
        ))]
        {
            SimdLevel::Simd256
        }
        #[cfg(all(
            target_feature = "sse2",
            not(target_feature = "avx2"),
            not(target_feature = "avx512f")
        ))]
        {
            SimdLevel::Simd128
        }
        #[cfg(not(any(
            target_feature = "sse2",
            target_feature = "avx2",
            target_feature = "avx512f"
        )))]
        {
            SimdLevel::Scalar
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on AArch64
        SimdLevel::Simd128
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM SIMD128 when simd128 feature is enabled
        #[cfg(target_feature = "simd128")]
        {
            SimdLevel::Simd128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            SimdLevel::Scalar
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        SimdLevel::Scalar
    }
}

/// Trait for SIMD-capable scalar types.
///
/// This trait provides the interface for types that can be vectorized
/// using SIMD operations.
pub trait SimdScalar: Field {
    /// The 256-bit SIMD register type for this scalar (e.g., AVX2 on x86-64).
    type Simd256: SimdRegister<Scalar = Self>;
    /// The 512-bit SIMD register type for this scalar (e.g., AVX-512 on x86-64).
    type Simd512: SimdRegister<Scalar = Self>;

    /// Number of elements that fit in a 256-bit register.
    const LANES_256: usize = 32 / core::mem::size_of::<Self>();

    /// Number of elements that fit in a 512-bit register.
    const LANES_512: usize = 64 / core::mem::size_of::<Self>();
}

/// Trait for SIMD register types.
///
/// This provides a unified interface for SIMD operations across
/// different architectures and vector widths.
pub trait SimdRegister: Copy + Clone + Send + Sync {
    /// The scalar type this register holds.
    type Scalar: SimdScalar;

    /// Number of lanes in this register.
    const LANES: usize;

    /// Creates a register with all lanes set to zero.
    fn zero() -> Self;

    /// Creates a register with all lanes set to the same value.
    fn splat(value: Self::Scalar) -> Self;

    /// Loads from an aligned pointer.
    ///
    /// # Safety
    /// The pointer must be aligned to the register width and point to
    /// at least LANES valid elements.
    unsafe fn load_aligned(ptr: *const Self::Scalar) -> Self;

    /// Loads from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least LANES valid elements.
    unsafe fn load_unaligned(ptr: *const Self::Scalar) -> Self;

    /// Stores to an aligned pointer.
    ///
    /// # Safety
    /// The pointer must be aligned to the register width and point to
    /// at least LANES valid writable elements.
    unsafe fn store_aligned(self, ptr: *mut Self::Scalar);

    /// Stores to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must point to at least LANES valid writable elements.
    unsafe fn store_unaligned(self, ptr: *mut Self::Scalar);

    /// Element-wise addition.
    fn add(self, other: Self) -> Self;

    /// Element-wise subtraction.
    fn sub(self, other: Self) -> Self;

    /// Element-wise multiplication.
    fn mul(self, other: Self) -> Self;

    /// Element-wise division.
    fn div(self, other: Self) -> Self;

    /// Fused multiply-add: self * a + b
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Fused multiply-subtract: self * a - b
    fn mul_sub(self, a: Self, b: Self) -> Self;

    /// Fused negative multiply-add: -(self * a) + b = b - self * a
    fn neg_mul_add(self, a: Self, b: Self) -> Self;

    /// Horizontal sum of all lanes.
    fn reduce_sum(self) -> Self::Scalar;

    /// Horizontal maximum of all lanes (for real types).
    fn reduce_max(self) -> Self::Scalar
    where
        Self::Scalar: Real;

    /// Horizontal minimum of all lanes (for real types).
    fn reduce_min(self) -> Self::Scalar
    where
        Self::Scalar: Real;

    /// Extracts a single lane.
    fn extract(self, index: usize) -> Self::Scalar;

    /// Inserts a value into a single lane.
    fn insert(self, index: usize, value: Self::Scalar) -> Self;
}

/// Extension trait for masked SIMD operations.
pub trait SimdMask: SimdRegister {
    /// The mask type for this register.
    type Mask: Copy + Clone;

    /// Creates a mask from a boolean array.
    fn mask_from_bools(bools: &[bool]) -> Self::Mask;

    /// Masked load: only loads elements where mask is true.
    ///
    /// # Safety
    /// For lanes where mask is true, the corresponding pointer element must be valid.
    unsafe fn load_masked(ptr: *const Self::Scalar, mask: Self::Mask, default: Self) -> Self;

    /// Masked store: only stores elements where mask is true.
    ///
    /// # Safety
    /// For lanes where mask is true, the corresponding pointer element must be valid and writable.
    unsafe fn store_masked(self, ptr: *mut Self::Scalar, mask: Self::Mask);

    /// Blends two registers based on mask: if mask\[i\] then a\[i\] else b\[i\].
    fn blend(mask: Self::Mask, a: Self, b: Self) -> Self;
}

/// Helper struct for iterating over SIMD chunks with proper head/body/tail handling.
#[derive(Debug, Clone, Copy)]
pub struct SimdChunks {
    /// Total number of elements.
    pub len: usize,
    /// Number of lanes per SIMD register.
    pub lanes: usize,
    /// Index where head (unaligned prefix) ends.
    pub head_end: usize,
    /// Index where body (aligned middle) ends.
    pub body_end: usize,
}

impl SimdChunks {
    /// Creates a new chunk iterator for the given length and alignment.
    #[inline]
    pub fn new<T: Scalar>(ptr: *const T, len: usize, level: SimdLevel) -> Self {
        let lanes = level.lanes::<T>();
        let align = level.width_bytes();

        if lanes <= 1 || len < lanes * 2 {
            // Not worth SIMD, treat everything as head
            return SimdChunks {
                len,
                lanes,
                head_end: len,
                body_end: len,
            };
        }

        let addr = ptr as usize;
        let misalign = addr % align;

        let head_end = if misalign == 0 {
            0
        } else {
            let elements_to_align = (align - misalign) / core::mem::size_of::<T>();
            elements_to_align.min(len)
        };

        let remaining = len - head_end;
        let full_vectors = remaining / lanes;
        let body_end = head_end + full_vectors * lanes;

        SimdChunks {
            len,
            lanes,
            head_end,
            body_end,
        }
    }

    /// Returns the number of head elements (before aligned body).
    #[inline]
    pub fn head_len(&self) -> usize {
        self.head_end
    }

    /// Returns the number of body elements (aligned middle).
    #[inline]
    pub fn body_len(&self) -> usize {
        self.body_end - self.head_end
    }

    /// Returns the number of tail elements (after aligned body).
    #[inline]
    pub fn tail_len(&self) -> usize {
        self.len - self.body_end
    }

    /// Returns the number of full SIMD vectors in the body.
    #[inline]
    pub fn body_vectors(&self) -> usize {
        self.body_len() / self.lanes
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_detect_simd_level() {
        let level = detect_simd_level();
        #[cfg(feature = "std")]
        println!("Detected SIMD level: {:?}", level);

        // When force-scalar is enabled, should always be Scalar
        #[cfg(feature = "force-scalar")]
        {
            assert_eq!(level, SimdLevel::Scalar);
            // But raw detection should still show hardware capability
            let raw = detect_simd_level_raw();
            println!("Raw hardware SIMD level: {:?}", raw);
        }

        // Without force-scalar, should detect hardware SIMD
        #[cfg(not(feature = "force-scalar"))]
        {
            #[cfg(target_arch = "x86_64")]
            assert!(level >= SimdLevel::Simd128);

            #[cfg(target_arch = "aarch64")]
            assert_eq!(level, SimdLevel::Simd128);
        }
    }

    #[test]
    fn test_simd_level_lanes() {
        assert_eq!(SimdLevel::Simd256.lanes::<f64>(), 4);
        assert_eq!(SimdLevel::Simd256.lanes::<f32>(), 8);
        assert_eq!(SimdLevel::Simd512.lanes::<f64>(), 8);
        assert_eq!(SimdLevel::Simd512.lanes::<f32>(), 16);
    }

    #[test]
    fn test_simd_chunks() {
        // Create a pointer with known alignment
        let data: Vec<f64> = vec![0.0; 100];
        let ptr = data.as_ptr();

        let chunks = SimdChunks::new(ptr, 100, SimdLevel::Simd256);
        #[cfg(feature = "std")]
        println!(
            "Chunks: head_end={}, body_end={}",
            chunks.head_end, chunks.body_end
        );

        // Verify that head + body + tail = len
        assert_eq!(
            chunks.head_len() + chunks.body_len() + chunks.tail_len(),
            100
        );
    }

    // =============================================================================
    // Comprehensive SIMD correctness tests
    // =============================================================================

    /// Test scalar fallback FMA accuracy.
    #[test]
    fn test_scalar_fma_accuracy() {
        use crate::simd::scalar::ScalarF64;

        let a = ScalarF64::splat(1.0 + 1e-15);
        let b = ScalarF64::splat(1.0 + 1e-15);
        let c = ScalarF64::splat(-(1.0 + 2e-15));

        // FMA should preserve more precision than separate mul+add
        let fma_result = a.mul_add(b, c);
        let mul_add_result = a.mul(b).add(c);

        // Both should be very small but may differ slightly
        assert!(fma_result.0.abs() < 1e-14);
        assert!(mul_add_result.0.abs() < 1e-14);
    }

    /// Test load/store roundtrip.
    #[test]
    fn test_load_store_roundtrip() {
        use crate::simd::scalar::ScalarF64;

        let values = [42.0f64, 1.5, -3.5, 1000.0];

        for &val in &values {
            let v = ScalarF64::splat(val);
            assert_eq!(v.reduce_sum(), val);
            assert_eq!(v.extract(0), val);
        }
    }

    /// Test arithmetic identities.
    #[test]
    fn test_arithmetic_identities() {
        use crate::simd::scalar::{ScalarF32, ScalarF64};

        // Test with f64
        let a = ScalarF64::splat(5.0);
        let zero = ScalarF64::zero();
        let one = ScalarF64::splat(1.0);

        // a + 0 = a
        assert_eq!(a.add(zero).0, 5.0);
        // a - 0 = a
        assert_eq!(a.sub(zero).0, 5.0);
        // a * 1 = a
        assert_eq!(a.mul(one).0, 5.0);
        // a / 1 = a
        assert_eq!(a.div(one).0, 5.0);
        // a * 0 = 0
        assert_eq!(a.mul(zero).0, 0.0);

        // Test with f32
        let a32 = ScalarF32::splat(5.0);
        let zero32 = ScalarF32::zero();
        let one32 = ScalarF32::splat(1.0);

        assert_eq!(a32.add(zero32).0, 5.0);
        assert_eq!(a32.mul(one32).0, 5.0);
    }

    /// Test reduction operations.
    #[test]
    fn test_reductions() {
        use crate::simd::scalar::{ScalarF32, ScalarF64};

        // For scalar types, all reductions return the same value
        let a = ScalarF64::splat(42.0);
        assert_eq!(a.reduce_sum(), 42.0);
        assert_eq!(a.reduce_max(), 42.0);
        assert_eq!(a.reduce_min(), 42.0);

        let b = ScalarF32::splat(-3.5);
        assert_eq!(b.reduce_sum(), -3.5);
        assert_eq!(b.reduce_max(), -3.5);
        assert_eq!(b.reduce_min(), -3.5);
    }

    /// Test negative value handling.
    #[test]
    fn test_negative_values() {
        use crate::simd::scalar::ScalarF64;

        let neg = ScalarF64::splat(-5.0);
        let pos = ScalarF64::splat(3.0);

        // -5 + 3 = -2
        assert_eq!(neg.add(pos).0, -2.0);
        // -5 * 3 = -15
        assert_eq!(neg.mul(pos).0, -15.0);
        // -5 - 3 = -8
        assert_eq!(neg.sub(pos).0, -8.0);
    }

    /// Test FMA variants.
    #[test]
    fn test_fma_variants() {
        use crate::simd::scalar::ScalarF64;

        let a = ScalarF64::splat(2.0);
        let b = ScalarF64::splat(3.0);
        let c = ScalarF64::splat(4.0);

        // mul_add: a * b + c = 2 * 3 + 4 = 10
        assert_eq!(a.mul_add(b, c).0, 10.0);

        // mul_sub: a * b - c = 2 * 3 - 4 = 2
        assert_eq!(a.mul_sub(b, c).0, 2.0);

        // neg_mul_add: -(a * b) + c = -6 + 4 = -2
        assert_eq!(a.neg_mul_add(b, c).0, -2.0);
    }

    /// Test insert/extract operations.
    #[test]
    fn test_insert_extract() {
        use crate::simd::scalar::ScalarF64;

        let a = ScalarF64::splat(1.0);
        let b = a.insert(0, 42.0);
        assert_eq!(b.extract(0), 42.0);
    }

    /// Platform-specific tests for native SIMD.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_aarch64_simd_correctness() {
        use crate::simd::aarch64::{F32x4, F64x2, F64x4};

        // Test F64x2
        let a = F64x2::splat(2.0);
        let b = F64x2::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.extract(0), 5.0);
        assert_eq!(sum.extract(1), 5.0);

        let fma = a.mul_add(b, F64x2::splat(1.0));
        assert_eq!(fma.extract(0), 7.0); // 2*3 + 1

        // Test F64x4 (emulated)
        let c = F64x4::splat(2.0);
        let d = F64x4::splat(3.0);

        assert_eq!(c.add(d).reduce_sum(), 20.0); // 4 * 5.0

        // Test F32x4
        let e = F32x4::splat(2.0);
        let f = F32x4::splat(3.0);

        assert_eq!(e.add(f).reduce_sum(), 20.0); // 4 * 5.0
    }

    /// The SIMD body loop relies on [`SimdChunks`] to hand it a pointer that is
    /// aligned to the register width so a subsequent `load_aligned` is sound.
    /// This locks in that contract across every possible starting misalignment:
    /// the head must land the body on a width-aligned address, and
    /// head + body + tail must partition the length with a whole number of
    /// vectors in the body. A regression in the chunk arithmetic (e.g. the
    /// `(align - misalign) / size_of` head computation) would fault a real
    /// `load_aligned`; asserting it here turns that latent segfault into a clean
    /// test failure. Nothing else currently drives `SimdChunks` past its length
    /// bookkeeping.
    #[test]
    fn test_simd_chunks_alignment_contract() {
        let level = SimdLevel::Simd256;
        let lanes = level.lanes::<f64>();
        let align = level.width_bytes();
        // Vec<f64> is at least 8-byte aligned, so offsetting the base by 0..8
        // elements sweeps every residue class modulo the 32-byte register width.
        let backing: Vec<f64> = vec![0.0; 256];

        for start in 0..8usize {
            let ptr = unsafe { backing.as_ptr().add(start) };
            let len = 200;
            let chunks = SimdChunks::new(ptr, len, level);

            assert_eq!(
                chunks.head_len() + chunks.body_len() + chunks.tail_len(),
                len,
                "partition must be exact for start offset {start}"
            );
            assert_eq!(
                chunks.body_len() % lanes,
                0,
                "body must be a whole number of vectors for start offset {start}"
            );

            let body_ptr = unsafe { ptr.add(chunks.head_len()) };
            assert_eq!(
                (body_ptr as usize) % align,
                0,
                "body pointer not {align}-byte aligned for start offset {start}"
            );
        }
    }

    /// End-to-end proof that a Level-1 dot product built on the abstraction
    /// ([`SimdChunks`]-style splitting plus a native [`SimdRegister`]) reproduces
    /// the scalar result, including special-value propagation. This is exactly
    /// the property any future BLAS wiring of this layer would depend on, and no
    /// other test exercises the chunked load / FMA-accumulate / reduce path
    /// together on a real wide register.
    ///
    /// # Safety
    /// The 256-bit register methods emit AVX2/FMA instructions with no internal
    /// runtime guard (see the module-level docs), so the whole body is gated
    /// behind `is_x86_feature_detected!`; on hardware without AVX2+FMA the check
    /// is skipped rather than risking undefined behavior.
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn test_abstraction_level1_dot_matches_scalar() {
        use crate::simd::SimdScalar;

        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            // No sound way to run the wide-register path; the scalar-fallback
            // arithmetic is already covered by the other tests in this module.
            return;
        }

        // Dot product through the abstraction: FMA-accumulate full vectors, then
        // finish the ragged tail with scalars — the canonical Level-1 shape.
        fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
            assert_eq!(a.len(), b.len());
            type Reg = <f64 as SimdScalar>::Simd256;
            let lanes = Reg::LANES;
            let len = a.len();
            let full = len / lanes * lanes;

            let mut acc = Reg::zero();
            let mut i = 0;
            while i < full {
                // SAFETY: `i + lanes <= full <= len`, so both loads read `lanes`
                // in-bounds elements; the caller confirmed AVX2+FMA is present.
                let va = unsafe { Reg::load_unaligned(a.as_ptr().add(i)) };
                let vb = unsafe { Reg::load_unaligned(b.as_ptr().add(i)) };
                acc = va.mul_add(vb, acc);
                i += lanes;
            }
            let mut sum = acc.reduce_sum();
            while i < len {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }

        // Length deliberately not a multiple of 4 to force the scalar tail.
        let len = 103usize;
        let a: Vec<f64> = (0..len).map(|k| (k as f64) * 0.5 - 7.0).collect();
        let b: Vec<f64> = (0..len).map(|k| 1.0 / (k as f64 + 1.0)).collect();

        let reference: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let got = simd_dot(&a, &b);

        // FMA accumulation can differ from separate mul+add in the final ULPs,
        // so compare with a tight relative tolerance rather than bit-for-bit.
        let tol = 1e-12 * reference.abs().max(1.0);
        assert!(
            (got - reference).abs() <= tol,
            "simd dot {got} vs scalar {reference} (tol {tol})"
        );

        // A NaN in the vectorized body must propagate through load/FMA/reduce.
        let mut a_nan = a.clone();
        a_nan[50] = f64::NAN;
        assert!(simd_dot(&a_nan, &b).is_nan(), "NaN did not propagate");

        // +Inf against a positive weight must yield an infinite result (no
        // silent saturation to a finite value).
        let mut a_inf = a.clone();
        a_inf[10] = f64::INFINITY;
        assert!(simd_dot(&a_inf, &b).is_infinite(), "Inf did not propagate");
    }
}
