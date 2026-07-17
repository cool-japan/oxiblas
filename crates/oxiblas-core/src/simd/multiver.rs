//! Runtime function multi-versioning infrastructure for OxiBLAS.
//!
//! This module is a thin façade over [`crate::simd::dispatch`], which is the
//! single source of truth for CPU SIMD capability detection and kernel
//! selection.  It provides:
//!
//! - Extended [`SimdCapabilityInfo`] with the field names required by the
//!   multi-versioning layer (`has_avx512f`, `has_avx2`, `has_sse42`, `has_fma`,
//!   `has_neon`, `cache_line_bytes`, `vector_width_bytes`, `has_simd128`) and a
//!   `detect()` that returns `&'static Self` when `std` is enabled.  Its values
//!   are copied verbatim from [`dispatch::SimdCapabilities`] so the two can
//!   never diverge.
//! - The [`simd_dispatch_caps!`] macro for named-arm dispatch.
//!
//! [`SimdDispatcher`], [`KernelSelector`] and [`GemmKernelKind`] are **re-exported
//! unchanged** from [`crate::simd::dispatch`].  An earlier revision of this
//! module defined its own copies of those three types, but they had drifted out
//! of sync — the copies here silently dropped the SSE4.2 kernel tier, so an
//! SSE4.2-only x86-64 CPU was misclassified as scalar.  Re-exporting the
//! dispatch definitions makes that class of divergence structurally impossible.
//!
//! # no_std
//!
//! Under `no_std`, `SimdCapabilityInfo::detect()` returns a freshly derived
//! value on every call because `OnceLock` is unavailable.  The value is still
//! copied from the dispatch layer, which itself derives from compile-time
//! `cfg!(target_feature = …)` constants that the compiler constant-folds away.

#[cfg(feature = "std")]
use std::sync::OnceLock;

use crate::simd::dispatch::{SimdCapabilities as LegacyCaps, SimdLevel as LegacyLevel};

// ---------------------------------------------------------------------------
// SimdCapabilityInfo — enhanced capability struct
// ---------------------------------------------------------------------------

/// Extended CPU SIMD capability description with the field naming convention
/// required by the multi-versioning layer.
///
/// On `std`-enabled builds, [`SimdCapabilityInfo::detect`] returns a
/// `&'static Self` reference that is initialised exactly once per process
/// (via [`OnceLock`]).  On `no_std` builds it returns a fresh `Self` value
/// computed from compile-time `cfg!` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilityInfo {
    // ------------------------------------------------------------------
    // x86-64
    // ------------------------------------------------------------------
    /// SSE4.2 support (x86-64 only).
    pub has_sse42: bool,
    /// AVX support (x86-64 only).
    pub has_avx: bool,
    /// AVX2 support (x86-64 only).
    pub has_avx2: bool,
    /// FMA (Fused Multiply-Add) support (x86-64 only).
    pub has_fma: bool,
    /// AVX-512 Foundation support (x86-64 only).
    pub has_avx512f: bool,
    /// AVX-512 Byte & Word instructions (x86-64 only).
    pub has_avx512bw: bool,
    /// AVX-512 Vector Length extensions (x86-64 only).
    pub has_avx512vl: bool,

    // ------------------------------------------------------------------
    // ARM
    // ------------------------------------------------------------------
    /// NEON support.  Always `true` on AArch64.
    pub has_neon: bool,
    /// SVE (Scalable Vector Extension) support.
    pub has_sve: bool,

    // ------------------------------------------------------------------
    // WebAssembly
    // ------------------------------------------------------------------
    /// WebAssembly `simd128` (128-bit) support (wasm32 target only).
    pub has_simd128: bool,

    // ------------------------------------------------------------------
    // Memory topology
    // ------------------------------------------------------------------
    /// Bytes in a single cache line (typically 64 on modern CPUs).
    pub cache_line_bytes: usize,
    /// Width of the widest supported SIMD vector register in bytes.
    pub vector_width_bytes: usize,
}

impl SimdCapabilityInfo {
    // ------------------------------------------------------------------
    // Public entry-points
    // ------------------------------------------------------------------

    /// Detect (or derive) capabilities for the current CPU and return a
    /// reference to the process-wide cached value.
    ///
    /// The first call performs runtime detection and stores the result.
    /// Subsequent calls return the same `&'static` reference.
    #[cfg(feature = "std")]
    #[inline]
    pub fn detect() -> &'static Self {
        static INFO: OnceLock<SimdCapabilityInfo> = OnceLock::new();
        INFO.get_or_init(Self::compute)
    }

    /// Derive capabilities from compile-time target features (no_std path).
    ///
    /// Returns a fresh value on every call.  The computation consists only of
    /// `cfg!` evaluations which the compiler constant-folds.
    #[cfg(not(feature = "std"))]
    #[inline]
    pub fn detect() -> Self {
        Self::compute()
    }

    // ------------------------------------------------------------------
    // Internal construction
    // ------------------------------------------------------------------

    fn compute() -> Self {
        let legacy = simd_caps();
        Self::from_legacy(legacy)
    }

    /// Build from the authoritative [`dispatch::SimdCapabilities`].
    ///
    /// Every field is copied verbatim.  The dispatch layer already performs
    /// accurate runtime SSE4.2 detection *and* applies the `force-scalar` /
    /// `max-simd-128` / `max-simd-256` cargo-feature gating, so re-deriving any
    /// value here (as an earlier revision did — it wrongly claimed "the legacy
    /// struct only tracks SSE3" and recomputed the width, defeating the
    /// feature gating) would only reintroduce divergence.
    #[cfg(feature = "std")]
    fn from_legacy(legacy: &LegacyCaps) -> Self {
        Self::mirror(legacy)
    }

    /// Build from the authoritative [`dispatch::SimdCapabilities`] (no_std path,
    /// passed by value).
    #[cfg(not(feature = "std"))]
    fn from_legacy(legacy: LegacyCaps) -> Self {
        Self::mirror(&legacy)
    }

    /// Field-by-field copy from [`dispatch::SimdCapabilities`].  This is the
    /// only construction path for [`SimdCapabilityInfo`], guaranteeing it stays
    /// a faithful view of the single source of truth.
    #[inline]
    fn mirror(legacy: &LegacyCaps) -> Self {
        Self {
            has_sse42: legacy.has_sse42,
            has_avx: legacy.has_avx,
            has_avx2: legacy.has_avx2,
            has_fma: legacy.has_fma,
            has_avx512f: legacy.has_avx512f,
            has_avx512bw: legacy.has_avx512bw,
            has_avx512vl: legacy.has_avx512vl,
            has_neon: legacy.has_neon,
            has_sve: legacy.has_sve,
            has_simd128: legacy.has_simd128,
            cache_line_bytes: legacy.cache_line_bytes,
            vector_width_bytes: legacy.vector_width_bytes,
        }
    }

    // ------------------------------------------------------------------
    // Capability queries
    // ------------------------------------------------------------------

    /// Returns `true` when AVX-512F, BW, and VL are all present.
    #[inline]
    pub fn has_avx512_full(&self) -> bool {
        self.has_avx512f && self.has_avx512bw && self.has_avx512vl
    }

    /// Returns `true` when both AVX2 and FMA are present.
    #[inline]
    pub fn has_avx2_fma(&self) -> bool {
        self.has_avx2 && self.has_fma
    }

    /// Number of `f64` elements that fit in the widest supported SIMD register.
    ///
    /// For AVX-512 this is 8; for AVX2 / NEON-128 this is 4 / 2; for scalar 1.
    #[inline]
    pub fn f64_simd_width(&self) -> usize {
        self.vector_width_bytes / core::mem::size_of::<f64>()
    }

    /// Number of `f32` elements that fit in the widest supported SIMD register.
    ///
    /// Always twice `f64_simd_width()`.
    #[inline]
    pub fn f32_simd_width(&self) -> usize {
        self.vector_width_bytes / core::mem::size_of::<f32>()
    }

    /// Returns the [`LegacyLevel`] that best summarises these capabilities.
    ///
    /// The tier ordering (SVE before NEON, plus the wasm `simd128` tier) mirrors
    /// [`dispatch::SimdCapabilities::optimal_level`] exactly.
    #[inline]
    pub fn optimal_level(&self) -> LegacyLevel {
        if self.has_avx512_full() {
            LegacyLevel::Avx512
        } else if self.has_avx2_fma() {
            LegacyLevel::Avx2
        } else if self.has_avx {
            LegacyLevel::Avx
        } else if self.has_sse42 {
            LegacyLevel::Sse42
        } else if self.has_sve {
            LegacyLevel::Sve
        } else if self.has_neon {
            LegacyLevel::Neon
        } else if self.has_simd128 {
            LegacyLevel::Simd128
        } else {
            LegacyLevel::Scalar
        }
    }
}

// ---------------------------------------------------------------------------
// simd_dispatch_caps! macro
// ---------------------------------------------------------------------------

/// Dispatch to the best available SIMD implementation, selecting among five
/// named arms in priority order: `avx512`, `avx2`, `sse42`, `neon`, `scalar`.
///
/// The first argument must be a [`SimdCapabilityInfo`] reference (or value);
/// on `std`-enabled builds use [`SimdCapabilityInfo::detect()`].
///
/// # Priority
///
/// 1. `avx512`  — AVX-512F + BW + VL
/// 2. `avx2`    — AVX2 + FMA (256-bit)
/// 3. `sse42`   — SSE4.2 (128-bit)
/// 4. `neon`    — AArch64 NEON
/// 5. `scalar`  — portable fallback
///
/// # Example
///
/// ```rust
/// use oxiblas_core::simd::multiver::{SimdCapabilityInfo, simd_dispatch_caps};
///
/// # #[cfg(feature = "std")]
/// let result: &str = simd_dispatch_caps!(
///     SimdCapabilityInfo::detect(),
///     avx512  => "avx512",
///     avx2    => "avx2",
///     sse42   => "sse42",
///     neon    => "neon",
///     scalar  => "scalar",
/// );
/// ```
#[macro_export]
macro_rules! simd_dispatch_caps {
    (
        $caps:expr,
        avx512  => $avx512:expr,
        avx2    => $avx2:expr,
        sse42   => $sse42:expr,
        neon    => $neon:expr,
        scalar  => $scalar:expr $(,)?
    ) => {{
        let _caps = $caps;
        if _caps.has_avx512_full() {
            $avx512
        } else if _caps.has_avx2_fma() {
            $avx2
        } else if _caps.has_sse42 {
            $sse42
        } else if _caps.has_neon {
            $neon
        } else {
            $scalar
        }
    }};
}

// Make the macro accessible as `crate::simd::multiver::simd_dispatch_caps`.
pub use simd_dispatch_caps;

// ---------------------------------------------------------------------------
// Re-exports from dispatch.rs (the single source of truth)
// ---------------------------------------------------------------------------
//
// `SimdDispatcher`, `KernelSelector` and `GemmKernelKind` are re-exported
// verbatim from `dispatch` rather than redefined here.  The dispatch versions
// carry the full x86-64 tier ladder — crucially including the SSE4.2 tier that
// the former local copies silently dropped — so re-exporting them makes the two
// modules physically the same types and eliminates any possibility of drift.

pub use crate::simd::dispatch::{
    GemmKernelKind, KernelSelector, SimdCapabilities, SimdDispatcher, SimdLevel, has_avx2_fma,
    has_avx512, has_neon, optimal_simd_level, simd_caps,
};

// ---------------------------------------------------------------------------
// Tests (>= 10 tests as required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Detection does not panic, cache_line_bytes is sane
    // ------------------------------------------------------------------
    #[test]
    fn test_detect_does_not_panic_and_cache_line_sane() {
        let caps = SimdCapabilityInfo::detect();
        // Cache line must be at least 8 bytes and a power of two.
        assert!(caps.cache_line_bytes >= 8);
        assert!(caps.cache_line_bytes.is_power_of_two());
        // vector_width_bytes must also be a power of two.
        assert!(caps.vector_width_bytes.is_power_of_two());
    }

    // ------------------------------------------------------------------
    // 2. AArch64: NEON always present
    // ------------------------------------------------------------------
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_aarch64_neon_always_true() {
        let caps = SimdCapabilityInfo::detect();
        assert!(caps.has_neon, "NEON is mandatory on AArch64");
        assert!(!caps.has_avx2, "AVX2 must not appear on AArch64");
        assert!(!caps.has_avx512f, "AVX-512 must not appear on AArch64");
    }

    // ------------------------------------------------------------------
    // 3. x86-64: flag hierarchy must be self-consistent
    // ------------------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_64_flag_hierarchy() {
        let caps = SimdCapabilityInfo::detect();
        assert!(!caps.has_neon, "NEON must not appear on x86-64");
        // AVX2 implies AVX (architectural requirement).
        if caps.has_avx2 {
            assert!(caps.has_avx, "AVX2 requires AVX");
        }
        // AVX-512 implies SSE4.2 on all known shipping CPUs.
        if caps.has_avx512f {
            assert!(caps.has_sse42, "AVX-512 implies SSE4.2");
        }
    }

    // ------------------------------------------------------------------
    // 4. f64/f32 simd widths are derived from vector_width_bytes
    // ------------------------------------------------------------------
    #[test]
    fn test_simd_width_derivation() {
        let caps = SimdCapabilityInfo::detect();
        assert_eq!(
            caps.f64_simd_width(),
            caps.vector_width_bytes / core::mem::size_of::<f64>()
        );
        assert_eq!(
            caps.f32_simd_width(),
            caps.vector_width_bytes / core::mem::size_of::<f32>()
        );
        // f32 must fit twice as many elements as f64 in the same register.
        assert_eq!(caps.f32_simd_width(), caps.f64_simd_width() * 2);
    }

    // ------------------------------------------------------------------
    // 5. vector_width_bytes agrees with reported capabilities
    // ------------------------------------------------------------------
    #[test]
    fn test_vector_width_matches_capability_tier() {
        let caps = SimdCapabilityInfo::detect();
        if caps.has_avx512f {
            assert_eq!(caps.vector_width_bytes, 64);
        } else if caps.has_avx2 {
            assert_eq!(caps.vector_width_bytes, 32);
        }
    }

    // ------------------------------------------------------------------
    // 6. simd_caps() is idempotent — same pointer on repeated calls (std)
    // ------------------------------------------------------------------
    #[cfg(feature = "std")]
    #[test]
    fn test_simd_caps_stable_pointer() {
        let a = simd_caps();
        let b = simd_caps();
        assert!(
            core::ptr::eq(a, b),
            "simd_caps() must return a stable &'static"
        );
    }

    // ------------------------------------------------------------------
    // 7. SimdCapabilityInfo::detect() is stable pointer on std builds
    // ------------------------------------------------------------------
    #[cfg(feature = "std")]
    #[test]
    fn test_capability_info_stable_pointer() {
        let a = SimdCapabilityInfo::detect();
        let b = SimdCapabilityInfo::detect();
        assert!(
            core::ptr::eq(a, b),
            "detect() must return a stable &'static"
        );
    }

    // ------------------------------------------------------------------
    // 8. optimal_level() is consistent with the capability flags
    // ------------------------------------------------------------------
    #[test]
    fn test_optimal_level_consistent_with_flags() {
        let caps = SimdCapabilityInfo::detect();
        let level = caps.optimal_level();
        match level {
            LegacyLevel::Avx512 => assert!(caps.has_avx512_full()),
            LegacyLevel::Avx2 => {
                assert!(!caps.has_avx512_full());
                assert!(caps.has_avx2_fma());
            }
            LegacyLevel::Avx => {
                assert!(!caps.has_avx512_full());
                assert!(!caps.has_avx2_fma());
                assert!(caps.has_avx);
            }
            LegacyLevel::Sse42 => {
                assert!(!caps.has_avx);
                assert!(caps.has_sse42);
            }
            LegacyLevel::Sve => {
                // SVE now wins the SVE/NEON tie, so has_neon may also be set.
                assert!(caps.has_sve);
            }
            LegacyLevel::Neon => {
                assert!(caps.has_neon);
                assert!(!caps.has_avx);
                assert!(!caps.has_sve);
            }
            LegacyLevel::Simd128 => {
                assert!(caps.has_simd128);
                assert!(!caps.has_neon);
            }
            LegacyLevel::Scalar => {
                assert!(!caps.has_avx);
                assert!(!caps.has_neon);
                assert!(!caps.has_sve);
                assert!(!caps.has_simd128);
            }
        }
    }

    // ------------------------------------------------------------------
    // 9. KernelSelector::select() returns valid GemmKernelKind values
    // ------------------------------------------------------------------
    #[test]
    fn test_kernel_selector_valid_kinds() {
        #[cfg(feature = "std")]
        let sel = *KernelSelector::select();
        #[cfg(not(feature = "std"))]
        let sel = KernelSelector::select();

        assert!(matches!(
            sel.gemm_f64_kernel,
            GemmKernelKind::Avx512
                | GemmKernelKind::Avx2
                | GemmKernelKind::Sse42
                | GemmKernelKind::Neon
                | GemmKernelKind::Scalar
        ));
        assert!(matches!(
            sel.gemm_f32_kernel,
            GemmKernelKind::Avx512
                | GemmKernelKind::Avx2
                | GemmKernelKind::Sse42
                | GemmKernelKind::Neon
                | GemmKernelKind::Scalar
        ));
    }

    // ------------------------------------------------------------------
    // 10. KernelSelector agrees with SimdCapabilityInfo on which tier to use
    // ------------------------------------------------------------------
    #[test]
    fn test_kernel_selector_agrees_with_capability_info() {
        let caps = SimdCapabilityInfo::detect();

        #[cfg(feature = "std")]
        let sel = *KernelSelector::select();
        #[cfg(not(feature = "std"))]
        let sel = KernelSelector::select();

        if caps.has_avx512_full() {
            assert_eq!(sel.gemm_f64_kernel, GemmKernelKind::Avx512);
            assert_eq!(sel.gemm_f32_kernel, GemmKernelKind::Avx512);
        } else if caps.has_avx2_fma() {
            assert_eq!(sel.gemm_f64_kernel, GemmKernelKind::Avx2);
            assert_eq!(sel.gemm_f32_kernel, GemmKernelKind::Avx2);
        } else if caps.has_sse42 {
            // The SSE4.2 tier must be honored — it used to be dropped to scalar.
            assert_eq!(sel.gemm_f64_kernel, GemmKernelKind::Sse42);
            assert_eq!(sel.gemm_f32_kernel, GemmKernelKind::Sse42);
        } else if caps.has_neon {
            assert_eq!(sel.gemm_f64_kernel, GemmKernelKind::Neon);
            assert_eq!(sel.gemm_f32_kernel, GemmKernelKind::Neon);
        } else {
            assert_eq!(sel.gemm_f64_kernel, GemmKernelKind::Scalar);
            assert_eq!(sel.gemm_f32_kernel, GemmKernelKind::Scalar);
        }
    }

    // ------------------------------------------------------------------
    // 11. simd_dispatch_caps! macro selects a branch consistent with caps
    // ------------------------------------------------------------------
    #[test]
    fn test_simd_dispatch_caps_macro_branch_selection() {
        #[cfg(feature = "std")]
        let caps = SimdCapabilityInfo::detect();
        #[cfg(not(feature = "std"))]
        let caps = SimdCapabilityInfo::detect();

        let chosen: u32 = simd_dispatch_caps!(
            &caps,
            avx512  => 512u32,
            avx2    => 256u32,
            sse42   => 128u32,
            neon    => 1000u32,
            scalar  => 1u32,
        );

        // Verify the chosen branch is consistent with the flags.
        if caps.has_avx512_full() {
            assert_eq!(chosen, 512);
        } else if caps.has_avx2_fma() {
            assert_eq!(chosen, 256);
        } else if caps.has_sse42 {
            assert_eq!(chosen, 128);
        } else if caps.has_neon {
            assert_eq!(chosen, 1000);
        } else {
            assert_eq!(chosen, 1);
        }
    }

    // ------------------------------------------------------------------
    // 12. SimdDispatcher trait: reference implementation is correct
    // ------------------------------------------------------------------
    struct DotProduct<'a> {
        x: &'a [f64],
        y: &'a [f64],
    }

    impl SimdDispatcher for DotProduct<'_> {
        type Output = f64;

        fn dispatch_avx512(&self) -> f64 {
            // Use scalar path for portability in the test.
            self.dispatch_scalar()
        }

        fn dispatch_avx2(&self) -> f64 {
            self.dispatch_scalar()
        }

        fn dispatch_neon(&self) -> f64 {
            self.dispatch_scalar()
        }

        fn dispatch_scalar(&self) -> f64 {
            self.x.iter().zip(self.y.iter()).map(|(a, b)| a * b).sum()
        }
    }

    #[test]
    fn test_simd_dispatcher_trait_correctness() {
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let y = [5.0_f64, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = DotProduct { x: &x, y: &y }.dispatch();
        assert!((result - 70.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // 13. GemmKernelKind::name() returns non-empty strings
    // ------------------------------------------------------------------
    #[test]
    fn test_gemm_kernel_kind_names_non_empty() {
        for kind in [
            GemmKernelKind::Avx512,
            GemmKernelKind::Avx2,
            GemmKernelKind::Sse42,
            GemmKernelKind::Neon,
            GemmKernelKind::Scalar,
        ] {
            assert!(!kind.name().is_empty());
        }
    }

    // ------------------------------------------------------------------
    // 14. GemmKernelKind::is_simd() is false only for Scalar
    // ------------------------------------------------------------------
    #[test]
    fn test_gemm_kernel_kind_is_simd() {
        assert!(GemmKernelKind::Avx512.is_simd());
        assert!(GemmKernelKind::Avx2.is_simd());
        assert!(GemmKernelKind::Sse42.is_simd());
        assert!(GemmKernelKind::Neon.is_simd());
        assert!(!GemmKernelKind::Scalar.is_simd());
    }

    // ------------------------------------------------------------------
    // 15. has_avx512/has_avx2_fma/has_neon helpers agree with detect()
    // ------------------------------------------------------------------
    #[test]
    fn test_free_helper_fns_agree_with_detect() {
        let caps = SimdCapabilityInfo::detect();
        // The helpers delegate to the legacy simd_caps() which is consistent
        // with detect().  We verify they do not contradict each other.
        if caps.has_avx512_full() {
            assert!(has_avx512());
        }
        if caps.has_avx2_fma() {
            assert!(has_avx2_fma());
        }
        if caps.has_neon {
            assert!(has_neon());
        }
    }

    // ------------------------------------------------------------------
    // 16. SimdDispatcher: dispatch_scalar used as ground truth
    // ------------------------------------------------------------------
    #[test]
    fn test_dispatcher_scalar_ground_truth() {
        let x = [0.0_f64; 0];
        let y = [0.0_f64; 0];
        let result = DotProduct { x: &x, y: &y }.dispatch();
        assert_eq!(result, 0.0, "empty dot product must be zero");
    }

    // ------------------------------------------------------------------
    // 17. Finding 5: multiver's shared types ARE dispatch's types.
    //     Proving the type identity makes the "SSE4.2 tier silently dropped"
    //     class of divergence structurally impossible: there is only one
    //     GemmKernelKind / KernelSelector / SimdDispatcher in the crate now.
    // ------------------------------------------------------------------
    #[test]
    fn test_multiver_delegates_to_dispatch_single_source() {
        use core::any::TypeId;
        assert_eq!(
            TypeId::of::<KernelSelector>(),
            TypeId::of::<crate::simd::dispatch::KernelSelector>(),
            "multiver::KernelSelector must be dispatch::KernelSelector"
        );
        assert_eq!(
            TypeId::of::<GemmKernelKind>(),
            TypeId::of::<crate::simd::dispatch::GemmKernelKind>(),
            "multiver::GemmKernelKind must be dispatch::GemmKernelKind"
        );
        // The SSE4.2 tier that the old local copies dropped now exists.
        assert_eq!(GemmKernelKind::Sse42.name(), "SSE4.2");
    }

    // ------------------------------------------------------------------
    // 18. Finding 1 (multiver view): SimdCapabilityInfo mirrors the gated
    //     dispatch capabilities field-for-field — never a wider/looser view.
    // ------------------------------------------------------------------
    #[test]
    fn test_capability_info_mirrors_dispatch_exactly() {
        let info = SimdCapabilityInfo::detect();
        let legacy = simd_caps();
        assert_eq!(info.has_sse42, legacy.has_sse42);
        assert_eq!(info.has_avx2, legacy.has_avx2);
        assert_eq!(info.has_avx512f, legacy.has_avx512f);
        assert_eq!(info.has_neon, legacy.has_neon);
        assert_eq!(info.has_sve, legacy.has_sve);
        assert_eq!(info.has_simd128, legacy.has_simd128);
        assert_eq!(info.vector_width_bytes, legacy.vector_width_bytes);
        assert_eq!(info.cache_line_bytes, legacy.cache_line_bytes);
        assert_eq!(info.optimal_level(), legacy.optimal_level());
    }
}
