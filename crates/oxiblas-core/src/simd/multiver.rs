//! Runtime function multi-versioning infrastructure for OxiBLAS.
//!
//! This module builds on top of [`crate::simd::dispatch`] to provide:
//!
//! - Extended [`SimdCapabilityInfo`] with the field names required by the task
//!   specification (`has_avx512f`, `has_avx2`, `has_sse42`, `has_fma`,
//!   `has_neon`, `cache_line_bytes`, `vector_width_bytes`) and a `detect()`
//!   that returns `&'static Self` when `std` is enabled.
//! - [`simd_dispatch_caps!`] macro for named-arm dispatch.
//! - [`SimdDispatcher`] trait for multi-versioned computations.
//! - [`KernelSelector`] / [`GemmKernelKind`] for startup kernel selection.
//!
//! # no_std
//!
//! Under `no_std`, `SimdCapabilityInfo::detect()` returns a freshly derived
//! value on every call because `OnceLock` is unavailable.  All fields are set
//! from compile-time `cfg!(target_feature = …)` constants, which the compiler
//! constant-folds away.

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

    /// Build from the legacy [`SimdCapabilities`] already present in
    /// `dispatch.rs`.  This ensures the two structs never diverge in their
    /// detection logic.
    #[cfg(feature = "std")]
    fn from_legacy(legacy: &LegacyCaps) -> Self {
        // SSE4.2 detection: the legacy struct only tracks SSE3.  We check for
        // SSE4.2 directly when std is available so that runtime detection is
        // accurate.
        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        let has_sse42 = is_x86_feature_detected!("sse4.2");
        #[cfg(not(target_arch = "x86_64"))]
        let has_sse42 = false;

        let has_avx512f = legacy.has_avx512f;
        let has_avx2 = legacy.has_avx2;

        let vector_width_bytes = if has_avx512f {
            64
        } else if has_avx2 {
            32
        } else if has_sse42 || legacy.has_neon {
            16
        } else {
            8
        };

        Self {
            has_sse42,
            has_avx: legacy.has_avx,
            has_avx2,
            has_fma: legacy.has_fma,
            has_avx512f,
            has_avx512bw: legacy.has_avx512bw,
            has_avx512vl: legacy.has_avx512vl,
            has_neon: legacy.has_neon,
            has_sve: legacy.has_sve,
            cache_line_bytes: 64,
            vector_width_bytes,
        }
    }

    /// Build from the legacy [`SimdCapabilities`] (no_std path, passed by
    /// value).
    #[cfg(not(feature = "std"))]
    fn from_legacy(legacy: LegacyCaps) -> Self {
        let has_sse42 = cfg!(target_feature = "sse4.2");
        let has_avx512f = legacy.has_avx512f;
        let has_avx2 = legacy.has_avx2;

        let vector_width_bytes: usize = if has_avx512f {
            64
        } else if has_avx2 {
            32
        } else if has_sse42 || legacy.has_neon {
            16
        } else {
            8
        };

        Self {
            has_sse42,
            has_avx: legacy.has_avx,
            has_avx2,
            has_fma: legacy.has_fma,
            has_avx512f,
            has_avx512bw: legacy.has_avx512bw,
            has_avx512vl: legacy.has_avx512vl,
            has_neon: legacy.has_neon,
            has_sve: legacy.has_sve,
            cache_line_bytes: 64,
            vector_width_bytes,
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
        } else if self.has_neon {
            LegacyLevel::Neon
        } else if self.has_sve {
            LegacyLevel::Sve
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
// SimdDispatcher trait
// ---------------------------------------------------------------------------

/// Trait for types that provide architecture-specialised implementations of a
/// single computation via function multi-versioning.
///
/// Implement the four required methods and call [`SimdDispatcher::dispatch`]
/// to have the runtime select the fastest available path automatically.
///
/// # Design note
///
/// The trait uses `&self` receivers so that the dispatch object can carry all
/// input data as fields, keeping call-sites clean.
///
/// # Example
///
/// ```rust
/// use oxiblas_core::simd::multiver::SimdDispatcher;
///
/// struct ScalarSum<'a>(&'a [f64]);
///
/// impl SimdDispatcher for ScalarSum<'_> {
///     type Output = f64;
///     fn dispatch_avx512(&self) -> f64 { self.dispatch_scalar() }
///     fn dispatch_avx2(&self)   -> f64 { self.dispatch_scalar() }
///     fn dispatch_neon(&self)   -> f64 { self.dispatch_scalar() }
///     fn dispatch_scalar(&self) -> f64 { self.0.iter().copied().sum() }
/// }
///
/// assert_eq!(ScalarSum(&[1.0, 2.0, 3.0]).dispatch(), 6.0);
/// ```
pub trait SimdDispatcher {
    /// The type returned by the computation.
    type Output;

    /// AVX-512F+BW+VL specialised implementation.
    fn dispatch_avx512(&self) -> Self::Output;

    /// AVX2 + FMA specialised implementation.
    fn dispatch_avx2(&self) -> Self::Output;

    /// NEON (AArch64) specialised implementation.
    fn dispatch_neon(&self) -> Self::Output;

    /// Portable scalar fallback.
    fn dispatch_scalar(&self) -> Self::Output;

    /// Select and call the best available implementation for the current CPU.
    ///
    /// The selection is based on [`SimdCapabilityInfo::detect`], which caches
    /// the result in a process-wide static (std builds) or recomputes from
    /// compile-time flags (no_std builds).
    fn dispatch(&self) -> Self::Output {
        #[cfg(feature = "std")]
        let caps = SimdCapabilityInfo::detect();
        #[cfg(not(feature = "std"))]
        let caps = SimdCapabilityInfo::detect();

        if caps.has_avx512_full() {
            self.dispatch_avx512()
        } else if caps.has_avx2_fma() {
            self.dispatch_avx2()
        } else if caps.has_neon {
            self.dispatch_neon()
        } else {
            self.dispatch_scalar()
        }
    }
}

// ---------------------------------------------------------------------------
// GemmKernelKind
// ---------------------------------------------------------------------------

/// Identifies which microkernel variant is used for GEMM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKernelKind {
    /// AVX-512 microkernel (512-bit registers, x86-64).
    Avx512,
    /// AVX2 + FMA microkernel (256-bit registers, x86-64).
    Avx2,
    /// NEON microkernel (128-bit registers, AArch64).
    Neon,
    /// Portable scalar microkernel (fallback for all targets).
    Scalar,
}

impl GemmKernelKind {
    /// Human-readable name of this kernel kind.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            GemmKernelKind::Avx512 => "AVX-512",
            GemmKernelKind::Avx2 => "AVX2+FMA",
            GemmKernelKind::Neon => "NEON",
            GemmKernelKind::Scalar => "scalar",
        }
    }

    /// Returns `true` when this kind uses SIMD (i.e., is not `Scalar`).
    #[inline]
    pub const fn is_simd(self) -> bool {
        !matches!(self, GemmKernelKind::Scalar)
    }
}

// ---------------------------------------------------------------------------
// KernelSelector
// ---------------------------------------------------------------------------

/// Selects the optimal GEMM microkernel for `f64` and `f32` based on the CPU
/// capabilities detected at runtime (or compile-time on no_std).
///
/// Call [`KernelSelector::select`] once at startup to obtain the globally
/// cached selector; subsequent calls return the same reference (std) or a
/// freshly computed identical value (no_std).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSelector {
    /// Best microkernel kind for double-precision (f64) GEMM.
    pub gemm_f64_kernel: GemmKernelKind,
    /// Best microkernel kind for single-precision (f32) GEMM.
    pub gemm_f32_kernel: GemmKernelKind,
}

impl KernelSelector {
    fn from_caps(caps: &SimdCapabilityInfo) -> Self {
        let kind = if caps.has_avx512_full() {
            GemmKernelKind::Avx512
        } else if caps.has_avx2_fma() {
            GemmKernelKind::Avx2
        } else if caps.has_neon {
            GemmKernelKind::Neon
        } else {
            GemmKernelKind::Scalar
        };

        Self {
            gemm_f64_kernel: kind,
            gemm_f32_kernel: kind,
        }
    }

    /// Returns a reference to the globally cached [`KernelSelector`].
    ///
    /// The first call performs detection; all subsequent calls return the same
    /// `&'static` reference.
    #[cfg(feature = "std")]
    pub fn select() -> &'static Self {
        static KERNEL_SEL: OnceLock<KernelSelector> = OnceLock::new();
        KERNEL_SEL.get_or_init(|| Self::from_caps(SimdCapabilityInfo::detect()))
    }

    /// Recomputes the [`KernelSelector`] from compile-time target features
    /// (no_std path).
    #[cfg(not(feature = "std"))]
    pub fn select() -> Self {
        Self::from_caps(&SimdCapabilityInfo::detect())
    }
}

// ---------------------------------------------------------------------------
// Convenience re-exports from dispatch.rs
// ---------------------------------------------------------------------------

pub use crate::simd::dispatch::{
    SimdCapabilities, SimdLevel, has_avx2_fma, has_avx512, has_neon, optimal_simd_level, simd_caps,
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
            LegacyLevel::Neon => {
                assert!(caps.has_neon);
                assert!(!caps.has_avx);
            }
            LegacyLevel::Sve => {
                assert!(caps.has_sve);
                assert!(!caps.has_neon);
            }
            LegacyLevel::Scalar => {
                assert!(!caps.has_avx);
                assert!(!caps.has_neon);
                assert!(!caps.has_sve);
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
                | GemmKernelKind::Neon
                | GemmKernelKind::Scalar
        ));
        assert!(matches!(
            sel.gemm_f32_kernel,
            GemmKernelKind::Avx512
                | GemmKernelKind::Avx2
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
}
