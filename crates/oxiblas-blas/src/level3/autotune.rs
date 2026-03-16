//! Auto-tuning module for GEMM block sizes.
//!
//! This module provides automatic detection of optimal block sizes
//! based on CPU cache hierarchy and matrix dimensions.

use crate::level3::gemm_kernel::MicroKernelShape;
use std::sync::OnceLock;

/// CPU cache information.
#[derive(Debug, Clone, Copy)]
pub struct CacheInfo {
    /// L1 data cache size in bytes (typically 32-128 KB).
    pub l1d: usize,
    /// L2 cache size in bytes (typically 256 KB - 1 MB).
    pub l2: usize,
    /// L3 cache size in bytes (typically 4-32 MB, may be shared).
    pub l3: usize,
    /// Cache line size in bytes (typically 64 bytes).
    pub line_size: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        // Conservative defaults for modern CPUs
        Self {
            l1d: 32 * 1024,      // 32 KB
            l2: 256 * 1024,      // 256 KB
            l3: 8 * 1024 * 1024, // 8 MB
            line_size: 64,
        }
    }
}

/// Cached CPU information.
static CPU_CACHE_INFO: OnceLock<CacheInfo> = OnceLock::new();

/// Gets CPU cache information.
///
/// This function detects CPU cache sizes at runtime when possible,
/// or falls back to conservative defaults.
pub fn get_cache_info() -> CacheInfo {
    *CPU_CACHE_INFO.get_or_init(detect_cache_info)
}

/// Detects CPU cache information at runtime.
fn detect_cache_info() -> CacheInfo {
    // Try platform-specific detection
    #[cfg(target_os = "macos")]
    {
        if let Some(info) = detect_cache_macos() {
            return info;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(info) = detect_cache_linux() {
            return info;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(info) = detect_cache_x86() {
            return info;
        }
    }

    // Default values for different architectures
    #[cfg(target_arch = "aarch64")]
    {
        // Apple Silicon defaults (M1/M2/M3 have large caches)
        return CacheInfo {
            l1d: 128 * 1024,      // 128 KB performance cores
            l2: 4 * 1024 * 1024,  // 4 MB L2 per cluster
            l3: 24 * 1024 * 1024, // ~24 MB system level cache (varies)
            line_size: 128,       // Apple Silicon uses 128-byte lines
        };
    }

    #[allow(unreachable_code)]
    CacheInfo::default()
}

/// Detects cache info on macOS using sysctl.
#[cfg(target_os = "macos")]
fn detect_cache_macos() -> Option<CacheInfo> {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(["-n", "hw.l1dcachesize"])
        .output()
        .ok()?;
    let l1d_str = String::from_utf8_lossy(&output.stdout);
    let l1d: usize = l1d_str.trim().parse().ok()?;

    let output = Command::new("sysctl")
        .args(["-n", "hw.l2cachesize"])
        .output()
        .ok()?;
    let l2_str = String::from_utf8_lossy(&output.stdout);
    let l2: usize = l2_str.trim().parse().ok()?;

    // L3 may not exist on all Macs (Apple Silicon has SLC instead)
    let l3 = Command::new("sysctl")
        .args(["-n", "hw.l3cachesize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(l2 * 8); // Estimate L3 as 8x L2

    // Cache line size
    let line_size = Command::new("sysctl")
        .args(["-n", "hw.cachelinesize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(64);

    Some(CacheInfo {
        l1d,
        l2,
        l3,
        line_size,
    })
}

/// Detects cache info on Linux using /sys filesystem.
#[cfg(target_os = "linux")]
fn detect_cache_linux() -> Option<CacheInfo> {
    use std::fs;

    // Try to read from /sys/devices/system/cpu/cpu0/cache/
    let base = "/sys/devices/system/cpu/cpu0/cache";

    let mut l1d = 0;
    let mut l2 = 0;
    let mut l3 = 0;
    let mut line_size = 64;

    // Iterate through cache indices (index0, index1, etc.)
    for i in 0..4 {
        let index_path = format!("{}/index{}", base, i);
        if !std::path::Path::new(&index_path).exists() {
            break;
        }

        // Read cache type
        let type_path = format!("{}/type", index_path);
        let cache_type = fs::read_to_string(&type_path).ok()?;
        let cache_type = cache_type.trim();

        // Read cache level
        let level_path = format!("{}/level", index_path);
        let level: u32 = fs::read_to_string(&level_path).ok()?.trim().parse().ok()?;

        // Read cache size
        let size_path = format!("{}/size", index_path);
        let size_str = fs::read_to_string(&size_path).ok()?;
        let size_str = size_str.trim();

        // Parse size (e.g., "32K", "256K", "8192K")
        let size = parse_size_linux(size_str)?;

        // Read line size
        if let Ok(ls_str) = fs::read_to_string(format!("{}/coherency_line_size", index_path)) {
            if let Ok(ls) = ls_str.trim().parse::<usize>() {
                line_size = ls;
            }
        }

        match (level, cache_type) {
            (1, "Data") | (1, "Unified") => l1d = size,
            (2, _) => l2 = size,
            (3, _) => l3 = size,
            _ => {}
        }
    }

    if l1d > 0 && l2 > 0 {
        Some(CacheInfo {
            l1d,
            l2,
            l3: if l3 > 0 { l3 } else { l2 * 8 },
            line_size,
        })
    } else {
        None
    }
}

/// Parses Linux cache size string (e.g., "32K", "256K", "8192K").
#[cfg(target_os = "linux")]
fn parse_size_linux(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix('K') {
        stripped.parse::<usize>().ok().map(|n| n * 1024)
    } else if let Some(stripped) = s.strip_suffix('M') {
        stripped.parse::<usize>().ok().map(|n| n * 1024 * 1024)
    } else {
        s.parse().ok()
    }
}

/// Detects cache info using x86 CPUID instruction.
#[cfg(target_arch = "x86_64")]
fn detect_cache_x86() -> Option<CacheInfo> {
    use core::arch::x86_64::__cpuid;

    unsafe {
        // Check if CPUID is available and supports extended function 0x80000006
        let basic = __cpuid(0);
        let max_basic = basic.eax;

        // Try Intel deterministic cache parameters (leaf 4)
        if max_basic >= 4 {
            let mut l1d = 0;
            let mut l2 = 0;
            let mut l3 = 0;
            let mut line_size = 64;

            for subleaf in 0..16 {
                let result = __cpuid_count(4, subleaf);

                // Cache type (bits 0-4 of EAX)
                let cache_type = result.eax & 0x1F;
                if cache_type == 0 {
                    break; // No more caches
                }

                // Cache level (bits 5-7 of EAX)
                let level = (result.eax >> 5) & 0x7;

                // Line size = EBX[11:0] + 1
                let line = (result.ebx & 0xFFF) + 1;
                // Partitions = EBX[21:12] + 1
                let partitions = ((result.ebx >> 12) & 0x3FF) + 1;
                // Ways = EBX[31:22] + 1
                let ways = ((result.ebx >> 22) & 0x3FF) + 1;
                // Sets = ECX + 1
                let sets = result.ecx + 1;

                let size = (line * partitions * ways * sets) as usize;

                // cache_type: 1 = data, 2 = instruction, 3 = unified
                match (level, cache_type) {
                    (1, 1) | (1, 3) => {
                        l1d = size;
                        line_size = line as usize;
                    }
                    (2, _) => l2 = size,
                    (3, _) => l3 = size,
                    _ => {}
                }
            }

            if l1d > 0 && l2 > 0 {
                return Some(CacheInfo {
                    l1d,
                    l2,
                    l3: if l3 > 0 { l3 } else { l2 * 8 },
                    line_size,
                });
            }
        }

        // Fallback to AMD extended CPUID (leaf 0x80000006)
        let ext = __cpuid(0x8000_0000);
        if ext.eax >= 0x8000_0006 {
            let cache = __cpuid(0x8000_0006);

            // L2 size in KB is in bits 31:16 of ECX
            let l2_kb = (cache.ecx >> 16) as usize;
            // L2 line size is in bits 7:0 of ECX
            let l2_line = (cache.ecx & 0xFF) as usize;

            // L3 size in 512KB units is in bits 31:18 of EDX
            let l3_512kb = (cache.edx >> 18) as usize;

            if l2_kb > 0 {
                return Some(CacheInfo {
                    l1d: 32 * 1024, // Assume 32KB L1
                    l2: l2_kb * 1024,
                    l3: if l3_512kb > 0 {
                        l3_512kb * 512 * 1024
                    } else {
                        l2_kb * 1024 * 8
                    },
                    line_size: if l2_line > 0 { l2_line } else { 64 },
                });
            }
        }

        None
    }
}

/// x86 CPUID with subleaf support.
#[cfg(target_arch = "x86_64")]
unsafe fn __cpuid_count(leaf: u32, subleaf: u32) -> core::arch::x86_64::CpuidResult {
    core::arch::x86_64::__cpuid_count(leaf, subleaf)
}

/// Auto-tuned blocking parameters for GEMM.
#[derive(Debug, Clone, Copy)]
pub struct AutoTunedBlocking {
    /// Block size for rows of A (MC).
    pub mc: usize,
    /// Block size for the K dimension (KC).
    pub kc: usize,
    /// Block size for columns of B (NC).
    pub nc: usize,
}

/// Computes optimal blocking parameters based on cache sizes and matrix dimensions.
///
/// # Algorithm
///
/// The BLIS-style GEMM algorithm has three nested loops with blocks:
/// - NC (outermost): columns of B, should fit B panel in L3
/// - KC (middle): K dimension, should fit packed B in L3, packed A in L2
/// - MC (innermost): rows of A, should fit packed A in L1/L2
///
/// Memory requirements for packing:
/// - `pack_b`: KC × NC elements → should fit in L3
/// - `pack_a`: MC × KC elements → should fit in L2 (ideally L1 for micro-panel)
///
/// For optimal performance:
/// - KC × NR × sizeof(T) should fit in L1 (B micro-panel)
/// - MC × KC × sizeof(T) should fit in L2 (A macro-panel)
/// - KC × NC × sizeof(T) should fit in L3 (B macro-panel)
///
/// # Arguments
///
/// * `m` - Number of rows in C
/// * `k` - Inner dimension
/// * `n` - Number of columns in C
/// * `elem_size` - Size of each element in bytes
/// * `shape` - Micro-kernel shape (MR × NR)
#[must_use]
pub fn compute_blocking(
    m: usize,
    k: usize,
    n: usize,
    elem_size: usize,
    shape: &MicroKernelShape,
) -> AutoTunedBlocking {
    let cache = get_cache_info();
    let mr = shape.mr;
    let nr = shape.nr;
    let line_size = cache.line_size;

    // Improved cache utilization targets based on empirical testing:
    // - L1: 80% for B micro-panel (critical path, keep hot)
    // - L2: Variable based on size (smaller caches need higher utilization)
    // - L3: 60% for B macro-panel (largest, less pressure)
    let l1_target = cache.l1d * 4 / 5; // 80%

    // L2 target: Use higher percentage for smaller caches to maximize utilization
    // For 256KB L2, we want to use 75% to ensure packed A fits well
    let l2_target = if cache.l2 <= 256 * 1024 {
        cache.l2 * 3 / 4 // 75% for small L2 (256KB or less)
    } else if cache.l2 <= 512 * 1024 {
        cache.l2 * 7 / 10 // 70% for medium-small L2
    } else {
        cache.l2 * 7 / 10 // 70% for larger L2
    };

    let l3_target = cache.l3 * 6 / 10; // 60%

    // Compute KC: B micro-panel (KC × NR) should fit in L1
    // KC × NR × elem_size ≤ L1_target
    // KC ≤ L1_target / (NR × elem_size)
    let kc_from_l1 = l1_target / (nr * elem_size);

    // For larger caches (Apple Silicon L2 = 4-16 MB), we can afford larger KC
    // to reduce the number of packing operations. The key insight is that
    // larger KC means fewer K-loop iterations, which reduces packing overhead.
    //
    // Empirical observation: KC in range 256-512 for f64, 512-1024 for f32
    // works well on modern CPUs with large L2 caches.
    //
    // Platform-specific tuning:
    // - Apple Silicon (aarch64): Very large L2 (4-16 MB), prefer large KC
    // - Intel/AMD x86_64: Varies widely (128KB-1MB typical), need adaptive tuning
    let kc_target = if cache.l2 >= 4 * 1024 * 1024 {
        // Large L2 (>= 4 MB): can use larger KC to reduce packing overhead
        // KC=448 for f64 gives 17% fewer K-loop iterations vs 384
        if elem_size >= 8 { 448 } else { 896 }
    } else if cache.l2 >= 1024 * 1024 {
        // Medium L2 (1-4 MB): typical for newer Intel/AMD desktop CPUs
        if elem_size >= 8 { 256 } else { 512 }
    } else if cache.l2 >= 512 * 1024 {
        // 512KB L2: typical for mid-range Intel/AMD CPUs
        // Optimize for better L2 utilization with moderate KC
        if elem_size >= 8 { 192 } else { 384 }
    } else if cache.l2 >= 256 * 1024 {
        // 256KB L2: typical for older Intel Xeon (e.g., E5-2600 v3/v4)
        // Use smaller KC to fit packed panels in L2
        // For f64: KC=256 uses 256*8=2KB per row of B, KC=192 uses 1.5KB
        // With MR=8, pack_a needs MC*KC*8 bytes in L2
        // Optimize for MC*KC*8 ≤ 180KB (70% of 256KB)
        // If KC=192: MC ≤ 180KB/(192*8) = 117 → rounds to 112 (multiple of 8)
        // This gives better L2 fit than KC=128
        if elem_size >= 8 { 192 } else { 320 }
    } else if cache.l2 >= 128 * 1024 {
        // 128KB L2: small cache, minimize KC
        if elem_size >= 8 { 96 } else { 192 }
    } else {
        // Very small L2 (< 128KB): fallback to conservative values
        if elem_size >= 8 { 64 } else { 128 }
    };

    // KC should not exceed L1 constraint but should aim for target
    let kc_initial = kc_from_l1.min(kc_target).max(64);

    // Compute MC: pack_a (MC × KC) should fit in L2
    // MC × KC × elem_size ≤ L2_target
    // MC ≤ L2_target / (KC × elem_size)
    let mc_from_l2 = l2_target / (kc_initial * elem_size);

    // MC should be a multiple of MR and cache-line aligned
    let elems_per_line = line_size / elem_size;
    let mc_aligned = round_up_to_multiple(mc_from_l2, mr.max(elems_per_line));

    // Refine KC based on actual MC to maximize pack_a utilization of L2
    // pack_a = MC × KC should fit in L2
    let kc_from_l2 = l2_target / (mc_aligned * elem_size);

    // Final KC: balance L1 micro-panel and L2 macro-panel constraints
    // Prefer larger KC (reduces packing overhead) but respect L1 constraint
    let kc = kc_from_l1.min(kc_from_l2).min(kc_target).max(64);

    // Re-compute MC with final KC, aiming for maximum utilization
    let mc_raw = l2_target / (kc * elem_size);
    let mc = round_up_to_multiple(mc_raw, mr.max(elems_per_line));

    // Compute NC: pack_b (KC × NC) should fit in L3
    // NC ≤ L3_target / (KC × elem_size)
    let nc_from_l3 = l3_target / (kc * elem_size);

    // NC should be a multiple of NR and cache-line aligned
    let nc = round_up_to_multiple(nc_from_l3, nr.max(elems_per_line));

    // Clamp to actual matrix dimensions for small matrices
    let mc_final = mc.min(m).max(mr);
    let kc_final = kc.min(k).max(1);
    let nc_final = nc.min(n).max(nr);

    // Ensure MR/NR alignment (final pass)
    let mc_final = round_down_to_multiple(mc_final, mr);
    let nc_final = round_down_to_multiple(nc_final, nr);

    AutoTunedBlocking {
        mc: if mc_final == 0 { mr } else { mc_final },
        kc: kc_final,
        nc: if nc_final == 0 { nr } else { nc_final },
    }
}

/// Rounds up `n` to the nearest multiple of `m`.
#[inline]
const fn round_up_to_multiple(n: usize, m: usize) -> usize {
    if m == 0 {
        return n;
    }
    ((n + m - 1) / m) * m
}

/// Rounds down `n` to the nearest multiple of `m`.
#[inline]
const fn round_down_to_multiple(n: usize, m: usize) -> usize {
    if m == 0 {
        return n;
    }
    (n / m) * m
}

/// Computes blocking optimized for specific matrix aspect ratios.
///
/// This function handles edge cases better than the general algorithm:
/// - Very tall-thin matrices (m >> k, m >> n)
/// - Very short-wide matrices (n >> m, n >> k)
/// - Inner-product dominated (k >> m, k >> n)
/// - Small matrices (use reduced blocking to minimize packing overhead)
#[must_use]
pub fn compute_blocking_adaptive(
    m: usize,
    k: usize,
    n: usize,
    elem_size: usize,
    shape: &MicroKernelShape,
) -> AutoTunedBlocking {
    let mr = shape.mr;
    let nr = shape.nr;
    let cache = get_cache_info();
    let line_size = cache.line_size;
    let elems_per_line = (line_size / elem_size).max(1);

    // For small matrices, use minimal blocking to reduce packing overhead
    let total_flops = 2 * m * n * k;
    if total_flops < 100_000 {
        // Very small: just use minimal blocking
        return AutoTunedBlocking {
            mc: round_down_to_multiple(m.min(64), mr).max(mr),
            kc: k.clamp(1, 128),
            nc: round_down_to_multiple(n.min(64), nr).max(nr),
        };
    }

    // Start with general tuning
    let base = compute_blocking(m, k, n, elem_size, shape);

    // Compute aspect ratios
    let m_f = m as f64;
    let k_f = k as f64;
    let n_f = n as f64;

    // Tall-thin (m >> k, m >> n): prioritize MC to process more rows per iteration
    // The key is to maximize reuse of packed B data across many row blocks
    if m_f > k_f * 4.0 && m_f > n_f * 4.0 {
        let mc_boost = round_down_to_multiple((base.mc * 3 / 2).min(m), mr.max(elems_per_line));
        let kc_reduced = (base.kc * 2 / 3).max(64);
        let nc_reduced = round_down_to_multiple(base.nc * 2 / 3, nr);
        return AutoTunedBlocking {
            mc: mc_boost.max(mr),
            kc: kc_reduced.min(k).max(1),
            nc: nc_reduced.max(nr),
        };
    }

    // Short-wide (n >> m, n >> k): prioritize NC to process more columns per iteration
    // The key is to maximize reuse of packed A data across many column blocks
    if n_f > m_f * 4.0 && n_f > k_f * 4.0 {
        let mc_reduced = round_down_to_multiple(base.mc * 2 / 3, mr);
        let nc_boost = round_down_to_multiple((base.nc * 3 / 2).min(n), nr.max(elems_per_line));
        return AutoTunedBlocking {
            mc: mc_reduced.max(mr),
            kc: base.kc.min(k).max(1),
            nc: nc_boost.max(nr),
        };
    }

    // Inner-product dominated (k >> m, k >> n): prioritize KC to amortize packing
    // Larger KC means fewer K-loop iterations = less packing overhead
    if k_f > m_f * 4.0 && k_f > n_f * 4.0 {
        let kc_boost = (base.kc * 2).min(k).min(1024); // Cap at 1024 for memory
        let mc_reduced = round_down_to_multiple(base.mc * 2 / 3, mr);
        let nc_reduced = round_down_to_multiple(base.nc * 2 / 3, nr);
        return AutoTunedBlocking {
            mc: mc_reduced.max(mr),
            kc: kc_boost.max(64),
            nc: nc_reduced.max(nr),
        };
    }

    // Panel-panel (m >> k and n >> k): small K, large M and N
    // Reduce KC to match K, but keep MC and NC large
    if m_f > k_f * 4.0 && n_f > k_f * 4.0 && k < base.kc {
        return AutoTunedBlocking {
            mc: base.mc.min(m).max(mr),
            kc: k,
            nc: base.nc.min(n).max(nr),
        };
    }

    base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_detection() {
        let info = get_cache_info();
        println!("Detected cache info: {:?}", info);

        // Sanity checks
        assert!(info.l1d >= 16 * 1024, "L1D should be at least 16KB");
        assert!(info.l2 >= info.l1d, "L2 should be >= L1D");
        assert!(info.l3 >= info.l2, "L3 should be >= L2");
        assert!(
            info.line_size >= 32,
            "Line size should be at least 32 bytes"
        );
    }

    #[test]
    fn test_blocking_computation() {
        let shape = MicroKernelShape { mr: 8, nr: 8 };

        // Small matrix
        let small = compute_blocking(32, 32, 32, 8, &shape);
        println!("Small (32x32): {:?}", small);
        assert!(small.mc >= shape.mr);
        assert!(small.nc >= shape.nr);

        // Medium matrix
        let medium = compute_blocking(256, 256, 256, 8, &shape);
        println!("Medium (256x256): {:?}", medium);
        assert!(medium.mc >= small.mc);

        // Large matrix
        let large = compute_blocking(2048, 2048, 2048, 8, &shape);
        println!("Large (2048x2048): {:?}", large);
        assert!(large.mc > 0);
        assert!(large.kc > 0);
        assert!(large.nc > 0);
    }

    #[test]
    fn test_adaptive_blocking() {
        let shape = MicroKernelShape { mr: 8, nr: 8 };

        // Tall-thin matrix
        let tall = compute_blocking_adaptive(4096, 64, 64, 8, &shape);
        println!("Tall-thin (4096x64x64): {:?}", tall);

        // Short-wide matrix
        let wide = compute_blocking_adaptive(64, 64, 4096, 8, &shape);
        println!("Short-wide (64x64x4096): {:?}", wide);

        // Inner-product dominated
        let inner = compute_blocking_adaptive(64, 4096, 64, 8, &shape);
        println!("Inner-product (64x4096x64): {:?}", inner);

        // Balanced
        let balanced = compute_blocking_adaptive(512, 512, 512, 8, &shape);
        println!("Balanced (512x512x512): {:?}", balanced);
    }

    #[test]
    fn test_f32_vs_f64_blocking() {
        let shape = MicroKernelShape { mr: 8, nr: 8 };

        let f64_block = compute_blocking(1024, 1024, 1024, 8, &shape);
        let f32_block = compute_blocking(1024, 1024, 1024, 4, &shape);

        println!("f64 blocking: {:?}", f64_block);
        println!("f32 blocking: {:?}", f32_block);

        // f32 should allow larger KC since elements are smaller
        // (not always true depending on cache constraints)
        assert!(f32_block.kc > 0);
        assert!(f64_block.kc > 0);
    }

    #[test]
    fn test_blocking_alignment() {
        let shape = MicroKernelShape { mr: 16, nr: 6 };

        let block = compute_blocking(1000, 1000, 1000, 8, &shape);

        // MC should be multiple of MR
        assert_eq!(block.mc % shape.mr, 0, "MC should be aligned to MR");

        // NC should be multiple of NR
        assert_eq!(block.nc % shape.nr, 0, "NC should be aligned to NR");
    }

    #[test]
    fn test_edge_case_tiny_matrix() {
        let shape = MicroKernelShape { mr: 8, nr: 8 };

        // Very tiny matrix
        let tiny = compute_blocking(4, 4, 4, 8, &shape);
        println!("Tiny (4x4): {:?}", tiny);

        // Should still have valid blocking
        assert!(tiny.mc >= shape.mr || tiny.mc >= 4);
        assert!(tiny.kc >= 1);
        assert!(tiny.nc >= shape.nr || tiny.nc >= 4);
    }
}
