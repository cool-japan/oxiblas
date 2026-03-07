//! Runtime SIMD dispatch system with function multi-versioning.
//!
//! Provides centralized CPU feature detection and caching for optimal
//! performance, plus infrastructure for dispatching to the best available
//! microkernel implementation at runtime.
//!
//! # Architecture
//!
//! - [`SimdCapabilities`]: Describes which SIMD extensions are available.
//! - [`simd_caps`]: Returns a `&'static SimdCapabilities` (std) or a freshly
//!   computed value (no_std) for the current CPU.
//! - [`simd_dispatch!`]: Macro that selects the best implementation branch
//!   based on detected capabilities.
//! - [`SimdDispatcher`]: Trait for types that provide multi-versioned
//!   implementations of a computation.
//! - [`KernelSelector`]: Chooses the optimal GEMM microkernel kind at startup.
//!
//! # no_std note
//!
//! When the `std` feature is disabled, `SimdCapabilities::detect()` returns a
//! freshly computed value derived from compile-time `target_feature` flags on
//! every call (no caching), and `simd_caps()` returns that value by value.
//! This is consistent with the no_std contract: no heap, no globals.

#[cfg(feature = "std")]
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// SimdCapabilities
// ---------------------------------------------------------------------------

/// CPU SIMD capabilities detected (or derived from compile-time flags) at
/// startup.
///
/// On x86-64 with `std` enabled, detection uses the `is_x86_feature_detected!`
/// macro which reads CPUID and is cached by the standard library.  On targets
/// that do not have `std`, detection falls back to compile-time
/// `target_feature` constants, which are set by RUSTFLAGS / `.cargo/config.toml`.
///
/// The struct is intentionally kept `Copy` so callers on no_std targets can
/// store it cheaply on the stack without worrying about ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    // ------------------------------------------------------------------
    // x86-64 fields
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
    // ARM fields
    // ------------------------------------------------------------------
    /// NEON support.  Always `true` on AArch64; may be `true` on 32-bit ARM.
    pub has_neon: bool,
    /// SVE (Scalable Vector Extension) support.
    pub has_sve: bool,

    // ------------------------------------------------------------------
    // Memory topology
    // ------------------------------------------------------------------
    /// Size of a single cache line in bytes (typically 64).
    pub cache_line_bytes: usize,
    /// Width of the widest supported SIMD vector in bytes.
    pub vector_width_bytes: usize,
}

impl SimdCapabilities {
    // ------------------------------------------------------------------
    // Construction helpers
    // ------------------------------------------------------------------

    /// Compute the [`SimdCapabilities`] for the current CPU.
    ///
    /// When compiled with `std`, the result is cached in a [`OnceLock`] and
    /// this function returns `&'static Self`; the detection work is only done
    /// once per process.
    ///
    /// When compiled **without** `std`, this returns `Self` by value (derived
    /// from compile-time `cfg!(target_feature = …)` flags).
    #[cfg(feature = "std")]
    pub fn detect() -> &'static Self {
        simd_caps()
    }

    /// Compute the [`SimdCapabilities`] for the current CPU (no_std path).
    ///
    /// Returns a fresh value on every call because `OnceLock` is not available
    /// without the standard library.
    #[cfg(not(feature = "std"))]
    pub fn detect() -> Self {
        simd_caps()
    }

    // ------------------------------------------------------------------
    // Internal constructor used by simd_caps_compute()
    // ------------------------------------------------------------------

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn compute() -> Self {
        let has_avx512f = is_x86_feature_detected!("avx512f");
        let has_avx512bw = is_x86_feature_detected!("avx512bw");
        let has_avx512vl = is_x86_feature_detected!("avx512vl");
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_fma = is_x86_feature_detected!("fma");
        let has_avx = is_x86_feature_detected!("avx");
        let has_sse42 = is_x86_feature_detected!("sse4.2");

        let vector_width_bytes = if has_avx512f {
            64
        } else if has_avx2 {
            32
        } else if has_sse42 {
            16
        } else {
            8
        };

        Self {
            has_sse42,
            has_avx,
            has_avx2,
            has_fma,
            has_avx512f,
            has_avx512bw,
            has_avx512vl,
            has_neon: false,
            has_sve: false,
            cache_line_bytes: 64,
            vector_width_bytes,
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    fn compute() -> Self {
        let has_avx512f = cfg!(target_feature = "avx512f");
        let has_avx512bw = cfg!(target_feature = "avx512bw");
        let has_avx512vl = cfg!(target_feature = "avx512vl");
        let has_avx2 = cfg!(target_feature = "avx2");
        let has_fma = cfg!(target_feature = "fma");
        let has_avx = cfg!(target_feature = "avx");
        let has_sse42 = cfg!(target_feature = "sse4.2");

        let vector_width_bytes: usize = if has_avx512f {
            64
        } else if has_avx2 {
            32
        } else if has_sse42 {
            16
        } else {
            8
        };

        Self {
            has_sse42,
            has_avx,
            has_avx2,
            has_fma,
            has_avx512f,
            has_avx512bw,
            has_avx512vl,
            has_neon: false,
            has_sve: false,
            cache_line_bytes: 64,
            vector_width_bytes,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn compute() -> Self {
        // NEON is mandatory on AArch64 per the architecture specification.
        let has_sve = cfg!(target_feature = "sve");
        Self {
            has_sse42: false,
            has_avx: false,
            has_avx2: false,
            has_fma: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_neon: true,
            has_sve,
            cache_line_bytes: 64,
            vector_width_bytes: 16,
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn compute() -> Self {
        Self {
            has_sse42: false,
            has_avx: false,
            has_avx2: false,
            has_fma: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_neon: false,
            has_sve: false,
            cache_line_bytes: 64,
            vector_width_bytes: 8,
        }
    }

    // ------------------------------------------------------------------
    // Capability queries
    // ------------------------------------------------------------------

    /// Returns `true` when AVX-512F, BW, and VL are all available — a common
    /// prerequisite for most practical AVX-512 code paths.
    #[inline]
    pub fn has_avx512_full(&self) -> bool {
        self.has_avx512f && self.has_avx512bw && self.has_avx512vl
    }

    /// Returns `true` when both AVX2 and FMA are available — the usual
    /// prerequisite for 256-bit fused multiply-add kernels.
    #[inline]
    pub fn has_avx2_fma(&self) -> bool {
        self.has_avx2 && self.has_fma
    }

    /// Returns the number of `f64` elements that fit in the widest supported
    /// SIMD vector register.
    ///
    /// | ISA             | f64 elements |
    /// |-----------------|-------------|
    /// | AVX-512         | 8           |
    /// | AVX2 / NEON-256 | 4           |
    /// | SSE4.2 / NEON   | 2           |
    /// | scalar          | 1           |
    #[inline]
    pub fn f64_simd_width(&self) -> usize {
        self.vector_width_bytes / core::mem::size_of::<f64>()
    }

    /// Returns the number of `f32` elements that fit in the widest supported
    /// SIMD vector register.
    #[inline]
    pub fn f32_simd_width(&self) -> usize {
        self.vector_width_bytes / core::mem::size_of::<f32>()
    }

    /// Returns the [`SimdLevel`] that best summarises the capabilities.
    #[inline]
    pub fn optimal_level(&self) -> SimdLevel {
        if self.has_avx512_full() {
            SimdLevel::Avx512
        } else if self.has_avx2_fma() {
            SimdLevel::Avx2
        } else if self.has_avx {
            SimdLevel::Avx
        } else if self.has_sse42 {
            SimdLevel::Sse42
        } else if self.has_neon {
            SimdLevel::Neon
        } else if self.has_sve {
            SimdLevel::Sve
        } else {
            SimdLevel::Scalar
        }
    }
}

// ---------------------------------------------------------------------------
// SimdLevel  (kept distinct from the coarser SimdLevel in simd.rs)
// ---------------------------------------------------------------------------

/// Fine-grained SIMD instruction set level, used by the dispatch layer.
///
/// The ordering (`Ord` derived) is meaningful only within the x86-64 family;
/// ARM levels intentionally have large discriminants so that comparisons across
/// families are not accidentally relied upon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// Scalar (no SIMD).
    Scalar = 0,
    /// SSE4.2 (128-bit, x86-64).
    Sse42 = 1,
    /// AVX (256-bit, x86-64, no FMA).
    Avx = 2,
    /// AVX2 with FMA (256-bit, x86-64).
    Avx2 = 3,
    /// AVX-512F+BW+VL (512-bit, x86-64).
    Avx512 = 4,
    /// NEON (128-bit, AArch64).
    Neon = 10,
    /// SVE (scalable, AArch64).
    Sve = 11,
}

impl SimdLevel {
    /// Human-readable name of the level.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Sse42 => "SSE4.2",
            SimdLevel::Avx => "AVX",
            SimdLevel::Avx2 => "AVX2+FMA",
            SimdLevel::Avx512 => "AVX-512",
            SimdLevel::Neon => "NEON",
            SimdLevel::Sve => "SVE",
        }
    }

    /// Number of `f64` elements in a vector of this width.
    #[inline]
    pub const fn f64_width(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse42 => 2,
            SimdLevel::Avx | SimdLevel::Avx2 => 4,
            SimdLevel::Avx512 => 8,
            SimdLevel::Neon => 2,
            SimdLevel::Sve => 2, // conservative; actual width is dynamic
        }
    }

    /// Number of `f32` elements in a vector of this width.
    #[inline]
    pub const fn f32_width(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse42 => 4,
            SimdLevel::Avx | SimdLevel::Avx2 => 8,
            SimdLevel::Avx512 => 16,
            SimdLevel::Neon => 4,
            SimdLevel::Sve => 4, // conservative; actual width is dynamic
        }
    }
}

// ---------------------------------------------------------------------------
// Cached global (std path)
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
static SIMD_CAPS: OnceLock<SimdCapabilities> = OnceLock::new();

/// Returns a reference to the globally cached [`SimdCapabilities`].
///
/// Detection is performed exactly once and the result is stored in a process-
/// wide static.  All subsequent calls return the same reference.
#[cfg(feature = "std")]
#[inline]
pub fn simd_caps() -> &'static SimdCapabilities {
    SIMD_CAPS.get_or_init(SimdCapabilities::compute)
}

/// Returns the [`SimdCapabilities`] derived from compile-time target features.
///
/// Because `OnceLock` requires `std`, this no_std variant recomputes the value
/// on every call.  The computation is a chain of `cfg!` evaluations and is
/// expected to be fully inlined / constant-folded by the compiler.
#[cfg(not(feature = "std"))]
#[inline]
pub fn simd_caps() -> SimdCapabilities {
    SimdCapabilities::compute()
}

/// Returns the optimal [`SimdLevel`] for the current CPU.
#[inline]
pub fn optimal_simd_level() -> SimdLevel {
    #[cfg(feature = "std")]
    {
        simd_caps().optimal_level()
    }
    #[cfg(not(feature = "std"))]
    {
        simd_caps().optimal_level()
    }
}

/// Returns `true` when AVX-512F+BW+VL are all available.
#[inline]
pub fn has_avx512() -> bool {
    #[cfg(feature = "std")]
    {
        simd_caps().has_avx512_full()
    }
    #[cfg(not(feature = "std"))]
    {
        simd_caps().has_avx512_full()
    }
}

/// Returns `true` when AVX2 and FMA are both available.
#[inline]
pub fn has_avx2_fma() -> bool {
    #[cfg(feature = "std")]
    {
        simd_caps().has_avx2_fma()
    }
    #[cfg(not(feature = "std"))]
    {
        simd_caps().has_avx2_fma()
    }
}

/// Returns `true` when NEON is available.
#[inline]
pub fn has_neon() -> bool {
    #[cfg(feature = "std")]
    {
        simd_caps().has_neon
    }
    #[cfg(not(feature = "std"))]
    {
        simd_caps().has_neon
    }
}

// ---------------------------------------------------------------------------
// simd_dispatch! macro
// ---------------------------------------------------------------------------

/// Dispatch to the best available SIMD implementation for the current CPU.
///
/// The macro evaluates the provided [`SimdCapabilities`] reference (or value)
/// and selects **exactly one** branch, in priority order:
///
/// 1. `avx512`  — AVX-512F+BW+VL
/// 2. `avx2`    — AVX2 + FMA (256-bit)
/// 3. `sse42`   — SSE4.2 (128-bit)
/// 4. `neon`    — AArch64 NEON
/// 5. `scalar`  — Portable scalar fallback
///
/// # Example
///
/// ```rust,ignore
/// use oxiblas_core::simd::dispatch::{simd_caps, simd_dispatch};
///
/// let result = simd_dispatch!(
///     simd_caps(),
///     avx512  => compute_avx512(&a, &b),
///     avx2    => compute_avx2(&a, &b),
///     sse42   => compute_sse42(&a, &b),
///     neon    => compute_neon(&a, &b),
///     scalar  => compute_scalar(&a, &b),
/// );
/// ```
#[macro_export]
macro_rules! simd_dispatch {
    (
        $caps:expr,
        avx512  => $avx512:expr,
        avx2    => $avx2:expr,
        sse42   => $sse42:expr,
        neon    => $neon:expr,
        scalar  => $scalar:expr $(,)?
    ) => {{
        let caps = $caps;
        if caps.has_avx512_full() {
            $avx512
        } else if caps.has_avx2_fma() {
            $avx2
        } else if caps.has_sse42 {
            $sse42
        } else if caps.has_neon {
            $neon
        } else {
            $scalar
        }
    }};
}

// Re-export so the macro is accessible from `oxiblas_core::simd::dispatch`.
pub use simd_dispatch;

// ---------------------------------------------------------------------------
// SimdDispatcher trait
// ---------------------------------------------------------------------------

/// Trait for types that provide architecture-specialised implementations of a
/// single computation.
///
/// Implement the four dispatch variants and call [`SimdDispatcher::dispatch`]
/// to let the runtime choose the best one automatically.
///
/// # Example
///
/// ```rust,ignore
/// struct DotProductF64<'a> {
///     x: &'a [f64],
///     y: &'a [f64],
/// }
///
/// impl SimdDispatcher for DotProductF64<'_> {
///     type Output = f64;
///     fn dispatch_avx512(&self) -> f64 { /* AVX-512 kernel */ unimplemented!() }
///     fn dispatch_avx2(&self)   -> f64 { /* AVX2+FMA kernel */ unimplemented!() }
///     fn dispatch_neon(&self)   -> f64 { /* NEON kernel */ unimplemented!() }
///     fn dispatch_scalar(&self) -> f64 { self.x.iter().zip(self.y).map(|(a,b)| a*b).sum() }
/// }
///
/// let result = DotProductF64 { x: &[1.0, 2.0], y: &[3.0, 4.0] }.dispatch();
/// assert_eq!(result, 11.0);
/// ```
pub trait SimdDispatcher {
    /// The return type of the computation.
    type Output;

    /// AVX-512F+BW+VL specialised implementation.
    fn dispatch_avx512(&self) -> Self::Output;

    /// AVX2 + FMA specialised implementation.
    fn dispatch_avx2(&self) -> Self::Output;

    /// NEON (AArch64) specialised implementation.
    fn dispatch_neon(&self) -> Self::Output;

    /// Portable scalar fallback implementation.
    fn dispatch_scalar(&self) -> Self::Output;

    /// SSE4.2 (x86-64, 128-bit) specialised implementation.
    ///
    /// Defaults to [`dispatch_scalar`][SimdDispatcher::dispatch_scalar] when
    /// not overridden, so implementors only need to override it when an
    /// SSE4.2-specific path exists.
    fn dispatch_sse42(&self) -> Self::Output {
        self.dispatch_scalar()
    }

    /// Select and call the best available implementation.
    ///
    /// The selection is based on [`SimdCapabilities::detect`] which caches the
    /// result in a process-wide static (when `std` is enabled).
    fn dispatch(&self) -> Self::Output {
        #[cfg(feature = "std")]
        let caps = SimdCapabilities::detect();
        #[cfg(not(feature = "std"))]
        let caps = SimdCapabilities::detect();

        if caps.has_avx512_full() {
            self.dispatch_avx512()
        } else if caps.has_avx2_fma() {
            self.dispatch_avx2()
        } else if caps.has_sse42 {
            self.dispatch_sse42()
        } else if caps.has_neon {
            self.dispatch_neon()
        } else {
            self.dispatch_scalar()
        }
    }
}

// ---------------------------------------------------------------------------
// KernelSelector  /  GemmKernelKind
// ---------------------------------------------------------------------------

/// Identifies the microkernel variant chosen for GEMM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKernelKind {
    /// AVX-512 microkernel (x86-64, 512-bit registers).
    Avx512,
    /// AVX2 + FMA microkernel (x86-64, 256-bit registers).
    Avx2,
    /// SSE4.2 microkernel (x86-64, 128-bit registers).
    Sse42,
    /// NEON microkernel (AArch64, 128-bit registers).
    Neon,
    /// Portable scalar microkernel (fallback).
    Scalar,
}

impl GemmKernelKind {
    /// Returns a human-readable name for the kind.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            GemmKernelKind::Avx512 => "AVX-512",
            GemmKernelKind::Avx2 => "AVX2+FMA",
            GemmKernelKind::Sse42 => "SSE4.2",
            GemmKernelKind::Neon => "NEON",
            GemmKernelKind::Scalar => "scalar",
        }
    }
}

/// Selects the optimal GEMM microkernel for each floating-point type based on
/// the CPU capabilities detected at runtime.
///
/// The selection is performed once and stored in a process-wide static (on
/// std-enabled builds).  Call [`KernelSelector::select`] to obtain a shared
/// reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSelector {
    /// Best microkernel kind for double-precision GEMM.
    pub gemm_f64_kernel: GemmKernelKind,
    /// Best microkernel kind for single-precision GEMM.
    pub gemm_f32_kernel: GemmKernelKind,
}

impl KernelSelector {
    /// Choose kernel kinds from a [`SimdCapabilities`] snapshot.
    fn from_caps(caps: &SimdCapabilities) -> Self {
        let kind = if caps.has_avx512_full() {
            GemmKernelKind::Avx512
        } else if caps.has_avx2_fma() {
            GemmKernelKind::Avx2
        } else if caps.has_sse42 {
            GemmKernelKind::Sse42
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
    /// Detection is performed exactly once per process.
    #[cfg(feature = "std")]
    pub fn select() -> &'static Self {
        static KERNEL_SEL: OnceLock<KernelSelector> = OnceLock::new();
        KERNEL_SEL.get_or_init(|| Self::from_caps(simd_caps()))
    }

    /// Recomputes the [`KernelSelector`] from compile-time features (no_std).
    #[cfg(not(feature = "std"))]
    pub fn select() -> Self {
        Self::from_caps(&simd_caps())
    }
}

// ---------------------------------------------------------------------------
// Debugging helper (std only)
// ---------------------------------------------------------------------------

/// Prints a formatted summary of the detected SIMD capabilities to stdout.
///
/// Useful for diagnostics and quick sanity checks.  Only available when the
/// `std` feature is enabled.
#[cfg(feature = "std")]
pub fn print_capabilities() {
    let caps = simd_caps();
    let level = caps.optimal_level();

    println!("=== OxiBLAS SIMD Capabilities ===");
    println!("Optimal level   : {}", level.name());
    println!("Cache line      : {} bytes", caps.cache_line_bytes);
    println!("Vector width    : {} bytes", caps.vector_width_bytes);
    println!("f64 SIMD width  : {} elements", caps.f64_simd_width());
    println!("f32 SIMD width  : {} elements", caps.f32_simd_width());

    #[cfg(target_arch = "x86_64")]
    {
        println!("x86-64 Features:");
        println!("  SSE4.2     : {}", caps.has_sse42);
        println!("  AVX        : {}", caps.has_avx);
        println!("  AVX2       : {}", caps.has_avx2);
        println!("  FMA        : {}", caps.has_fma);
        println!("  AVX-512F   : {}", caps.has_avx512f);
        println!("  AVX-512BW  : {}", caps.has_avx512bw);
        println!("  AVX-512VL  : {}", caps.has_avx512vl);
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("AArch64 Features:");
        println!("  NEON       : {}", caps.has_neon);
        println!("  SVE        : {}", caps.has_sve);
    }

    let sel = KernelSelector::select();
    println!("Kernel Selection:");
    println!("  GEMM f64   : {}", sel.gemm_f64_kernel.name());
    println!("  GEMM f32   : {}", sel.gemm_f32_kernel.name());
    println!("==================================");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Detection smoke-test: at least one SIMD feature or scalar path
    // ------------------------------------------------------------------
    #[test]
    fn test_capabilities_detection_does_not_panic() {
        // Must not panic on any supported target.
        let caps = SimdCapabilities::detect();
        // Cache line should be a sensible power-of-two value.
        assert!(caps.cache_line_bytes >= 8);
        assert!(caps.cache_line_bytes.is_power_of_two());
    }

    // ------------------------------------------------------------------
    // 2. AArch64: NEON must always be true
    // ------------------------------------------------------------------
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_aarch64_neon_always_present() {
        let caps = SimdCapabilities::detect();
        assert!(caps.has_neon, "NEON is mandatory on AArch64");
        // x86-64 flags must be absent
        assert!(!caps.has_avx2);
        assert!(!caps.has_avx512f);
    }

    // ------------------------------------------------------------------
    // 3. x86-64: flags are self-consistent
    // ------------------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_64_flag_consistency() {
        let caps = SimdCapabilities::detect();
        // NEON must not be set on x86-64.
        assert!(!caps.has_neon);
        // If AVX2 is present, AVX must be present too (architectural requirement).
        if caps.has_avx2 {
            assert!(caps.has_avx, "AVX2 implies AVX");
        }
        // If AVX-512F is present, SSE4.2 should be present (all x86-64 CPUs
        // with AVX-512 also support SSE4.2).
        if caps.has_avx512f {
            assert!(caps.has_sse42, "AVX-512 implies SSE4.2");
        }
    }

    // ------------------------------------------------------------------
    // 4. Vector width matches capabilities
    // ------------------------------------------------------------------
    #[test]
    fn test_vector_width_consistent_with_capabilities() {
        let caps = SimdCapabilities::detect();
        if caps.has_avx512f {
            assert_eq!(caps.vector_width_bytes, 64);
        } else if caps.has_avx2 {
            assert_eq!(caps.vector_width_bytes, 32);
        }
    }

    // ------------------------------------------------------------------
    // 5. f64 / f32 SIMD widths are derived from vector_width_bytes
    // ------------------------------------------------------------------
    #[test]
    fn test_simd_widths_derived_correctly() {
        let caps = SimdCapabilities::detect();
        assert_eq!(
            caps.f64_simd_width(),
            caps.vector_width_bytes / core::mem::size_of::<f64>()
        );
        assert_eq!(
            caps.f32_simd_width(),
            caps.vector_width_bytes / core::mem::size_of::<f32>()
        );
        // f32 always holds twice as many elements as f64 in the same register.
        assert_eq!(caps.f32_simd_width(), caps.f64_simd_width() * 2);
    }

    // ------------------------------------------------------------------
    // 6. SimdLevel ordering is meaningful within x86-64 family
    // ------------------------------------------------------------------
    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 > SimdLevel::Avx);
        assert!(SimdLevel::Avx > SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 > SimdLevel::Scalar);
    }

    // ------------------------------------------------------------------
    // 7. SimdLevel vector widths are consistent
    // ------------------------------------------------------------------
    #[test]
    fn test_simd_level_widths() {
        // Scalar holds exactly 1 element per lane.
        assert_eq!(SimdLevel::Scalar.f64_width(), 1);
        assert_eq!(SimdLevel::Scalar.f32_width(), 1);
        // Each wider ISA must hold more elements.
        assert_eq!(SimdLevel::Sse42.f64_width(), 2);
        assert_eq!(SimdLevel::Sse42.f32_width(), 4);
        assert_eq!(SimdLevel::Avx2.f64_width(), 4);
        assert_eq!(SimdLevel::Avx2.f32_width(), 8);
        assert_eq!(SimdLevel::Avx512.f64_width(), 8);
        assert_eq!(SimdLevel::Avx512.f32_width(), 16);
        assert_eq!(SimdLevel::Neon.f64_width(), 2);
        assert_eq!(SimdLevel::Neon.f32_width(), 4);
    }

    // ------------------------------------------------------------------
    // 8. simd_caps() is idempotent (same result on repeated calls)
    // ------------------------------------------------------------------
    #[cfg(feature = "std")]
    #[test]
    fn test_simd_caps_cached_identity() {
        let a = simd_caps();
        let b = simd_caps();
        // Must be the same static reference.
        assert!(
            core::ptr::eq(a, b),
            "simd_caps() must return a stable &'static"
        );
    }

    // ------------------------------------------------------------------
    // 9. optimal_level() returns a variant that matches the caps flags
    // ------------------------------------------------------------------
    #[test]
    fn test_optimal_level_consistent_with_flags() {
        let caps = SimdCapabilities::detect();
        let level = caps.optimal_level();
        match level {
            SimdLevel::Avx512 => assert!(caps.has_avx512_full()),
            SimdLevel::Avx2 => {
                assert!(!caps.has_avx512_full());
                assert!(caps.has_avx2_fma());
            }
            SimdLevel::Avx => {
                assert!(!caps.has_avx512_full());
                assert!(!caps.has_avx2_fma());
                assert!(caps.has_avx);
            }
            SimdLevel::Sse42 => {
                assert!(!caps.has_avx);
                assert!(caps.has_sse42);
            }
            SimdLevel::Neon => {
                assert!(caps.has_neon);
                assert!(!caps.has_avx);
            }
            SimdLevel::Sve => {
                assert!(caps.has_sve);
                assert!(!caps.has_neon);
            }
            SimdLevel::Scalar => {
                assert!(!caps.has_sse42);
                assert!(!caps.has_avx);
                assert!(!caps.has_neon);
                assert!(!caps.has_sve);
            }
        }
    }

    // ------------------------------------------------------------------
    // 10. KernelSelector::select() does not panic and returns valid kinds
    // ------------------------------------------------------------------
    #[test]
    fn test_kernel_selector_valid_kinds() {
        #[cfg(feature = "std")]
        let sel = *KernelSelector::select();
        #[cfg(not(feature = "std"))]
        let sel = KernelSelector::select();

        // Kind must be one of the defined variants.
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
    // 11. KernelSelector and optimal_level agree on AVX-512
    // ------------------------------------------------------------------
    #[test]
    fn test_kernel_selector_matches_optimal_level() {
        let caps = SimdCapabilities::detect();

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
    // 12. simd_dispatch! macro selects a branch without panicking
    // ------------------------------------------------------------------
    #[test]
    fn test_simd_dispatch_macro_selects_branch() {
        #[cfg(feature = "std")]
        let caps = simd_caps();
        #[cfg(not(feature = "std"))]
        let caps = &simd_caps();

        let result: u32 = simd_dispatch!(
            caps,
            avx512  => 512u32,
            avx2    => 256u32,
            sse42   => 128u32,
            neon    => 1000u32,
            scalar  => 1u32,
        );

        // The result must be consistent with what optimal_level() says.
        // Note: simd_dispatch! routes AVX (no FMA) to the avx2 branch because
        // the macro checks has_avx2_fma() not has_avx, so AVX-only maps to sse42.
        let expected: u32 = match caps.optimal_level() {
            SimdLevel::Avx512 => 512,
            SimdLevel::Avx2 => 256,
            SimdLevel::Avx | SimdLevel::Sse42 => 128,
            SimdLevel::Neon | SimdLevel::Sve => 1000,
            SimdLevel::Scalar => 1,
        };
        assert_eq!(result, expected);
    }

    // ------------------------------------------------------------------
    // 13. SimdDispatcher trait: scalar reference implementation
    // ------------------------------------------------------------------
    struct ScalarDot<'a> {
        x: &'a [f64],
        y: &'a [f64],
    }

    impl SimdDispatcher for ScalarDot<'_> {
        type Output = f64;

        fn dispatch_avx512(&self) -> f64 {
            // In production this would use AVX-512 intrinsics; here we
            // delegate to scalar so the test compiles on all platforms.
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
        let dot = ScalarDot { x: &x, y: &y };

        // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = dot.dispatch();
        assert!((result - 70.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // 14. print_capabilities does not panic (std only)
    // ------------------------------------------------------------------
    #[cfg(feature = "std")]
    #[test]
    fn test_print_capabilities_does_not_panic() {
        print_capabilities();
    }

    // ------------------------------------------------------------------
    // 15. GemmKernelKind names are non-empty strings
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
    // 16. has_avx512() / has_avx2_fma() / has_neon() helpers agree with caps
    // ------------------------------------------------------------------
    #[test]
    fn test_helper_functions_agree_with_caps() {
        let caps = SimdCapabilities::detect();
        assert_eq!(has_avx512(), caps.has_avx512_full());
        assert_eq!(has_avx2_fma(), caps.has_avx2_fma());
        assert_eq!(has_neon(), caps.has_neon);
    }
}
