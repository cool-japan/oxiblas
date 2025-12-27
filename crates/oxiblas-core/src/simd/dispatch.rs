//! Runtime SIMD dispatch system
//!
//! Provides centralized CPU feature detection and caching for optimal performance.
//! Supports x86-64 (SSE3, AVX2, FMA, AVX-512) and ARM (NEON, SVE).

use std::sync::OnceLock;

/// CPU SIMD capabilities detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    // x86-64 features
    /// SSE3 support (x86-64)
    pub sse3: bool,
    /// AVX support (x86-64)
    pub avx: bool,
    /// AVX2 support (x86-64)
    pub avx2: bool,
    /// FMA (Fused Multiply-Add) support (x86-64)
    pub fma: bool,
    /// AVX-512 Foundation support (x86-64)
    pub avx512f: bool,
    /// AVX-512 Byte & Word support (x86-64)
    pub avx512bw: bool,
    /// AVX-512 Vector Length extensions (x86-64)
    pub avx512vl: bool,

    // ARM features
    /// NEON support (ARM/AArch64)
    pub neon: bool,
    /// SVE (Scalable Vector Extension) support (ARM)
    pub sve: bool,
}

impl SimdCapabilities {
    /// Detect CPU capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse3: is_x86_feature_detected!("sse3"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                neon: false,
                sve: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse3: false,
                avx: false,
                avx2: false,
                fma: false,
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                neon: true, // NEON is mandatory on aarch64
                sve: {
                    #[cfg(target_feature = "sve")]
                    {
                        true
                    }
                    #[cfg(not(target_feature = "sve"))]
                    {
                        false
                    }
                },
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse3: false,
                avx: false,
                avx2: false,
                fma: false,
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                neon: false,
                sve: false,
            }
        }
    }

    /// Check if AVX-512 is fully supported (F+BW+VL required for most ops)
    #[inline]
    pub fn has_avx512_full(&self) -> bool {
        self.avx512f && self.avx512bw && self.avx512vl
    }

    /// Check if AVX2 with FMA is supported
    #[inline]
    pub fn has_avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }

    /// Get optimal SIMD level for the current CPU
    #[inline]
    pub fn optimal_level(&self) -> SimdLevel {
        if self.has_avx512_full() {
            SimdLevel::Avx512
        } else if self.has_avx2_fma() {
            SimdLevel::Avx2
        } else if self.avx {
            SimdLevel::Avx
        } else if self.sse3 {
            SimdLevel::Sse3
        } else if self.neon {
            SimdLevel::Neon
        } else if self.sve {
            SimdLevel::Sve
        } else {
            SimdLevel::Scalar
        }
    }
}

/// SIMD instruction set level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// Scalar (no SIMD)
    Scalar = 0,
    /// SSE3 (128-bit, x86-64)
    Sse3 = 1,
    /// AVX (256-bit, x86-64)
    Avx = 2,
    /// AVX2 with FMA (256-bit, x86-64)
    Avx2 = 3,
    /// AVX-512 (512-bit, x86-64)
    Avx512 = 4,
    /// NEON (128-bit, ARM)
    Neon = 10,
    /// SVE (variable width, ARM)
    Sve = 11,
}

impl SimdLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Sse3 => "SSE3",
            SimdLevel::Avx => "AVX",
            SimdLevel::Avx2 => "AVX2+FMA",
            SimdLevel::Avx512 => "AVX-512",
            SimdLevel::Neon => "NEON",
            SimdLevel::Sve => "SVE",
        }
    }

    /// Get vector width in elements for f64
    pub fn f64_width(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse3 => 2,                  // 128-bit / 64-bit
            SimdLevel::Avx | SimdLevel::Avx2 => 4, // 256-bit / 64-bit
            SimdLevel::Avx512 => 8,                // 512-bit / 64-bit
            SimdLevel::Neon => 2,                  // 128-bit / 64-bit
            SimdLevel::Sve => 2,                   // Variable, conservative estimate
        }
    }

    /// Get vector width in elements for f32
    pub fn f32_width(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse3 => 4,                  // 128-bit / 32-bit
            SimdLevel::Avx | SimdLevel::Avx2 => 8, // 256-bit / 32-bit
            SimdLevel::Avx512 => 16,               // 512-bit / 32-bit
            SimdLevel::Neon => 4,                  // 128-bit / 32-bit
            SimdLevel::Sve => 4,                   // Variable, conservative estimate
        }
    }
}

/// Cached SIMD capabilities (initialized once)
static SIMD_CAPS: OnceLock<SimdCapabilities> = OnceLock::new();

/// Get cached SIMD capabilities
#[inline]
pub fn simd_caps() -> &'static SimdCapabilities {
    SIMD_CAPS.get_or_init(SimdCapabilities::detect)
}

/// Get optimal SIMD level for current CPU
#[inline]
pub fn optimal_simd_level() -> SimdLevel {
    simd_caps().optimal_level()
}

/// Check if AVX-512 is available
#[inline]
pub fn has_avx512() -> bool {
    simd_caps().has_avx512_full()
}

/// Check if AVX2 with FMA is available
#[inline]
pub fn has_avx2_fma() -> bool {
    simd_caps().has_avx2_fma()
}

/// Check if NEON is available
#[inline]
pub fn has_neon() -> bool {
    simd_caps().neon
}

/// Print detected SIMD capabilities (for debugging/info)
pub fn print_capabilities() {
    let caps = simd_caps();
    let level = caps.optimal_level();

    println!("=== OxiBLAS SIMD Capabilities ===");
    println!("Optimal Level: {}", level.name());

    #[cfg(target_arch = "x86_64")]
    {
        println!("x86-64 Features:");
        println!("  SSE3:      {}", caps.sse3);
        println!("  AVX:       {}", caps.avx);
        println!("  AVX2:      {}", caps.avx2);
        println!("  FMA:       {}", caps.fma);
        println!("  AVX-512F:  {}", caps.avx512f);
        println!("  AVX-512BW: {}", caps.avx512bw);
        println!("  AVX-512VL: {}", caps.avx512vl);
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("ARM Features:");
        println!("  NEON: {}", caps.neon);
        println!("  SVE:  {}", caps.sve);
    }

    println!("Vector Widths:");
    println!("  f64: {} elements", level.f64_width());
    println!("  f32: {} elements", level.f32_width());
    println!("==================================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_detection() {
        let caps = SimdCapabilities::detect();

        // Should detect at least one capability
        let has_any = caps.sse3 || caps.avx || caps.avx2 || caps.neon || caps.sve;

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert!(has_any, "Should detect SIMD capabilities on x86_64/aarch64");
        }
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 > SimdLevel::Avx);
        assert!(SimdLevel::Avx > SimdLevel::Sse3);
        assert!(SimdLevel::Sse3 > SimdLevel::Scalar);
    }

    #[test]
    fn test_vector_widths() {
        assert_eq!(SimdLevel::Scalar.f64_width(), 1);
        assert_eq!(SimdLevel::Avx2.f64_width(), 4);
        assert_eq!(SimdLevel::Avx512.f64_width(), 8);

        assert_eq!(SimdLevel::Scalar.f32_width(), 1);
        assert_eq!(SimdLevel::Avx2.f32_width(), 8);
        assert_eq!(SimdLevel::Avx512.f32_width(), 16);
    }

    #[test]
    fn test_cached_capabilities() {
        let caps1 = simd_caps();
        let caps2 = simd_caps();

        // Should be same reference (cached)
        assert_eq!(caps1 as *const _, caps2 as *const _);
    }

    #[test]
    fn test_optimal_level() {
        let level = optimal_simd_level();
        println!("Detected optimal SIMD level: {}", level.name());

        // Should not be uninitialized
        assert!(matches!(
            level,
            SimdLevel::Scalar
                | SimdLevel::Sse3
                | SimdLevel::Avx
                | SimdLevel::Avx2
                | SimdLevel::Avx512
                | SimdLevel::Neon
                | SimdLevel::Sve
        ));
    }

    #[test]
    fn test_print_capabilities() {
        // Just ensure it doesn't panic
        print_capabilities();
    }
}
