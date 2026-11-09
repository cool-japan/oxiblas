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
        // Fallback for every aarch64 target where cache detection fails or
        // does not exist (on macOS: only when neither the `hw.perflevel0.*`
        // nor the `hw.*` L1D or L2 key can be read, see `macos_cache_info`).
        // The values
        // are Apple Silicon sizes (M1/M2/M3 have large caches next to the
        // generic default of 32 KiB L1D / 256 KiB L2) but they mix core types:
        // on an Apple M3 the performance (P) cores have 128 KiB L1D and 16 MiB
        // L2, the efficiency (E) cores 64 KiB and 4 MiB. The L1D below is the
        // P-core size, the L2 the E-core size.
        return CacheInfo {
            l1d: 128 * 1024,      // 128 KiB: P-core L1D (E cores: 64 KiB)
            l2: 4 * 1024 * 1024,  // 4 MiB: E-core L2 (P cores: 16 MiB on M3)
            l3: 24 * 1024 * 1024, // ~24 MB system level cache (varies)
            line_size: 128,       // Apple Silicon uses 128-byte lines
        };
    }

    #[allow(unreachable_code)]
    CacheInfo::default()
}

/// Detects cache info on macOS using sysctl.
///
/// See [`macos_cache_info`] for which keys are read and how they combine.
#[cfg(target_os = "macos")]
fn detect_cache_macos() -> Option<CacheInfo> {
    macos_cache_info(sysctl_usize)
}

/// Reads a positive integer sysctl value with `sysctl -n <key>`.
///
/// A key the kernel does not know makes `sysctl` exit with status 1 and print
/// nothing; some known keys (`hw.l3cachesize` on an Apple M3) exit with
/// status 0 and print an empty value. Both, like a zero or non-numeric value,
/// read as `None`.
#[cfg(target_os = "macos")]
fn sysctl_usize(key: &str) -> Option<usize> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value: usize = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    (value > 0).then_some(value)
}

/// Builds the cache description of a Mac from sysctl lookups (`read` returns
/// `None` for a missing or empty key), falling back key by key.
///
/// On Apple Silicon the plain `hw.*` keys describe the *efficiency* cores (an
/// Apple M3 reports 64 KiB L1D and a 4 MiB L2), while the GEMM threads that
/// matter run on the performance cores, described by the `hw.perflevel0.*`
/// keys. Intel Macs have no `perflevel*` keys at all, so each value falls back
/// to its `hw.*` key separately:
///
/// - L1D: `hw.perflevel0.l1dcachesize`, else `hw.l1dcachesize` (required).
/// - L2: the per-core share of the performance cluster's L2,
///   `hw.perflevel0.l2cachesize / hw.perflevel0.cpusperl2` (an M3: 16 MiB
///   shared by 4 cores, 4 MiB each), else `hw.l2cachesize` (required). The
///   per-core share is what one thread of a parallel GEMM can count on.
/// - L3: `hw.perflevel0.l3cachesize`, else `hw.l3cachesize`, else 8 × L2
///   (Apple Silicon has a system-level cache but no L3 key; on an M3 this is
///   32 MiB, the same estimate as with the efficiency-core L2).
/// - Line size: `hw.cachelinesize`, else 64 bytes.
///
/// On an Apple M3 the result differs from the plain `hw.*` keys only in the
/// L1D (128 KiB instead of 64 KiB). The GEMM block sizes derived from it are
/// the same either way, because the L2 and `kc` targets bind before the L1D
/// does (see `test_m3_perflevel_detection_keeps_blocking`).
#[cfg(any(target_os = "macos", test))]
fn macos_cache_info(read: impl Fn(&str) -> Option<usize>) -> Option<CacheInfo> {
    let l1d = read("hw.perflevel0.l1dcachesize").or_else(|| read("hw.l1dcachesize"))?;
    let l2 = match (
        read("hw.perflevel0.l2cachesize"),
        read("hw.perflevel0.cpusperl2"),
    ) {
        (Some(cluster), Some(cores)) if cores > 0 => cluster / cores,
        (Some(cluster), _) => cluster,
        (None, _) => read("hw.l2cachesize")?,
    };
    let l3 = read("hw.perflevel0.l3cachesize")
        .or_else(|| read("hw.l3cachesize"))
        .unwrap_or(l2.saturating_mul(8));
    let line_size = read("hw.cachelinesize").unwrap_or(64);
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
        if let Ok(ls_str) = fs::read_to_string(format!("{}/coherency_line_size", index_path))
            && let Ok(ls) = ls_str.trim().parse::<usize>()
        {
            line_size = ls;
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

    // SAFETY: the block only executes CPUID, which every x86_64 CPU
    // implements, and touches no memory. Leaf 4 is queried only after leaf 0
    // reports a maximum basic leaf of at least 4, and leaf 0x8000_0006 only
    // after leaf 0x8000_0000 reports it, so no unsupported leaf is requested.
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
/// # Fitting to the dimensions
///
/// The cache model yields dimension-free *targets*, which are then fitted to
/// `m`, `k`, `n`: a dimension that fits its target is a single block of
/// exactly that size, and a larger one is split into the fewest near-equal
/// blocks (multiples of MR for MC, of NR for NC and of 8 for KC). The packed volume of A is `m × k × ⌈n / NC⌉`, so an avoidable extra NC
/// block re-packs the whole of A.
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
    fit_to_dims(cache_targets(elem_size, shape), m, k, n, shape)
}

/// Dimension-free MC/KC/NC targets derived from the cache hierarchy.
///
/// See [`compute_blocking`] for the cache model. The cache-line alignment
/// (`elems_per_line`) only shapes these targets; the final block sizes come
/// from [`fit_to_dims`].
fn cache_targets(elem_size: usize, shape: &MicroKernelShape) -> AutoTunedBlocking {
    cache_targets_for(&get_cache_info(), elem_size, shape)
}

/// [`cache_targets`] for an explicit cache description.
fn cache_targets_for(
    cache: &CacheInfo,
    elem_size: usize,
    shape: &MicroKernelShape,
) -> AutoTunedBlocking {
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

    AutoTunedBlocking { mc, kc, nc }
}

/// Fits dimension-free block targets to the actual GEMM dimensions with
/// [`balanced_block`] (units: MR for MC, [`KC_BLOCK_UNIT`] for KC, NR for NC).
///
/// Clamping a target to its dimension and *then* rounding it down to a
/// multiple of MR/NR used to split dimensions that fit in one block: `n = 256`
/// with `NR = 6` gave `NC = 252`, i.e. blocks `[252, 4]`, and the 4-column tail
/// re-packed all of A for a single micro-panel of work.
fn fit_to_dims(
    targets: AutoTunedBlocking,
    m: usize,
    k: usize,
    n: usize,
    shape: &MicroKernelShape,
) -> AutoTunedBlocking {
    AutoTunedBlocking {
        mc: balanced_block(m, targets.mc, shape.mr),
        kc: balanced_block(k, targets.kc, KC_BLOCK_UNIT),
        nc: balanced_block(n, targets.nc, shape.nr),
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

/// Granularity of a KC (inner-dimension) block whenever `k` has to be split.
///
/// A multiple of 8 keeps every packed micro-panel stride (`KC × NR ×
/// size_of::<T>()`) a multiple of 64 bytes and matches the 2×/4× k-loop
/// unrolling of the micro-kernels.
pub(crate) const KC_BLOCK_UNIT: usize = 8;

/// Splits `dim` into the fewest blocks of at most `target` elements and
/// returns the (balanced) block size.
///
/// - `dim <= target`: one block of exactly `dim`. A block that fits is never
///   split by a rounding step.
/// - Otherwise, with `t = max(round_down(target, unit), unit)`: if `dim <= t`
///   (only possible when `target < unit`) one block of `dim`; else the result
///   is `round_up(⌈dim / nb⌉, unit)` with `nb = ⌈dim / t⌉`. It is a multiple of
///   `unit`, at most `t`, yields exactly `nb` blocks, and the padding of the
///   last block is below `nb × unit`.
/// - `dim == 0` yields `unit`, so no blocked loop can ever `step_by(0)`.
///
/// The result is idempotent: feeding it back as `target` returns it unchanged.
#[must_use]
pub(crate) const fn balanced_block(dim: usize, target: usize, unit: usize) -> usize {
    let unit = if unit == 0 { 1 } else { unit };
    if dim == 0 {
        return unit;
    }
    if dim <= target {
        return dim;
    }
    let rounded = round_down_to_multiple(target, unit);
    let t = if rounded < unit { unit } else { rounded };
    if dim <= t {
        return dim;
    }
    let blocks = dim.div_ceil(t);
    round_up_to_multiple(dim.div_ceil(blocks), unit)
}

/// Computes blocking optimized for specific matrix aspect ratios.
///
/// This function handles edge cases better than the general algorithm:
/// - Very tall-thin matrices (m >> k, m >> n)
/// - Very short-wide matrices (n >> m, n >> k)
/// - Inner-product dominated (k >> m, k >> n)
/// - Small matrices (use reduced blocking to minimize packing overhead)
///
/// Every adjustment scales the dimension-free cache *targets*; the result is
/// then fitted to the dimensions exactly like [`compute_blocking`]. (Scaling
/// values that were already clamped to a dimension used to re-split it, e.g.
/// tall-thin 4096×512×256 got NC = 168, blocks `[168, 88]`.)
#[must_use]
pub fn compute_blocking_adaptive(
    m: usize,
    k: usize,
    n: usize,
    elem_size: usize,
    shape: &MicroKernelShape,
) -> AutoTunedBlocking {
    // For small matrices, use minimal blocking to reduce packing overhead
    let total_flops = 2 * m * n * k;
    if total_flops < 100_000 {
        // Very small: just use minimal blocking
        let targets = AutoTunedBlocking {
            mc: 64,
            kc: 128,
            nc: 64,
        };
        return fit_to_dims(targets, m, k, n, shape);
    }

    // Start with the general cache targets
    let base = cache_targets(elem_size, shape);

    // Compute aspect ratios
    let m_f = m as f64;
    let k_f = k as f64;
    let n_f = n as f64;

    let targets = if m_f > k_f * 4.0 && m_f > n_f * 4.0 {
        // Tall-thin (m >> k, m >> n): prioritize MC to process more rows per iteration
        // The key is to maximize reuse of packed B data across many row blocks
        AutoTunedBlocking {
            mc: base.mc * 3 / 2,
            kc: (base.kc * 2 / 3).max(64),
            nc: base.nc * 2 / 3,
        }
    } else if n_f > m_f * 4.0 && n_f > k_f * 4.0 {
        // Short-wide (n >> m, n >> k): prioritize NC to process more columns per iteration
        // The key is to maximize reuse of packed A data across many column blocks
        AutoTunedBlocking {
            mc: base.mc * 2 / 3,
            kc: base.kc,
            nc: base.nc * 3 / 2,
        }
    } else if k_f > m_f * 4.0 && k_f > n_f * 4.0 {
        // Inner-product dominated (k >> m, k >> n): prioritize KC to amortize packing
        // Larger KC means fewer K-loop iterations = less packing overhead
        AutoTunedBlocking {
            mc: base.mc * 2 / 3,
            kc: (base.kc * 2).min(1024), // Cap at 1024 for memory
            nc: base.nc * 2 / 3,
        }
    } else if m_f > k_f * 4.0 && n_f > k_f * 4.0 && k < base.kc {
        // Panel-panel (m >> k and n >> k): small K, large M and N
        // The whole of K forms one block; MC and NC stay large
        AutoTunedBlocking {
            mc: base.mc,
            kc: k,
            nc: base.nc,
        }
    } else {
        base
    };

    fit_to_dims(targets, m, k, n, shape)
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

        // A dimension that is split uses MR/NR-multiples; one that fits its
        // target is a single block of exactly the dimension.
        assert!(block.mc <= 1000 && block.nc <= 1000, "{block:?}");
        assert!(
            block.mc == 1000 || block.mc.is_multiple_of(shape.mr),
            "MC should be the full dimension or aligned to MR: {block:?}"
        );
        assert!(
            block.nc == 1000 || block.nc.is_multiple_of(shape.nr),
            "NC should be the full dimension or aligned to NR: {block:?}"
        );
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

    #[test]
    fn test_issue_4_balanced_block_invariants() {
        let dims = [
            1usize, 2, 5, 7, 8, 63, 64, 65, 255, 256, 257, 500, 511, 512, 1000, 1023, 2048, 2050,
            4096, 4099,
        ];
        let targets = [
            0usize, 1, 3, 6, 8, 16, 64, 100, 168, 252, 256, 441, 448, 832, 2046, 5712,
        ];
        for &dim in &dims {
            for &target in &targets {
                for &unit in &[1usize, 4, 6, 8, 16] {
                    let b = balanced_block(dim, target, unit);
                    let ctx = format!("dim={dim} target={target} unit={unit} -> {b}");
                    assert!(b >= 1 && b <= dim, "{ctx}");
                    let t = (target / unit * unit).max(unit);
                    let blocks = dim.div_ceil(b);
                    if dim <= target || dim <= t {
                        // I1: a dimension that fits is one block of exactly dim.
                        assert_eq!(b, dim, "{ctx}");
                    } else {
                        // I2: unit-aligned, within the rounded target, and the
                        // minimum number of blocks.
                        assert_eq!(b % unit, 0, "{ctx}");
                        assert!(b <= t, "{ctx}");
                        assert_eq!(blocks, dim.div_ceil(t), "{ctx}");
                    }
                    // I3: total padding below one unit per block.
                    assert!(blocks * b - dim < blocks * unit, "{ctx}");
                    // I4: idempotent.
                    assert_eq!(balanced_block(dim, b, unit), b, "{ctx}");
                }
            }
        }
        // An empty dimension never yields a zero step.
        assert_eq!(balanced_block(0, 128, 8), 8);
        assert_eq!(balanced_block(0, 0, 0), 1);
        // Reporter-style examples (independent of the cache model).
        assert_eq!(balanced_block(256, 5712, 6), 256);
        assert_eq!(balanced_block(500, 448, 8), 256); // [256, 244], not [448, 52]
        assert_eq!(balanced_block(2050, 2046, 6), 1026); // [1026, 1024], not [2046, 4]
    }

    /// I5 helper: `b` is a valid block for `dim` with granularity `unit`.
    fn check_block(b: usize, dim: usize, unit: usize, ctx: &str) {
        assert!(b >= 1 && b <= dim.max(1), "{ctx}: {b} vs dim {dim}");
        assert!(
            b == dim || b.is_multiple_of(unit),
            "{ctx}: {b} vs unit {unit}"
        );
        let blocks = dim.div_ceil(b);
        assert!(blocks * b - dim < blocks * unit, "{ctx}: padding");
    }

    #[test]
    fn test_issue_4_blocking_sweep_never_splits_needlessly() {
        let dims = [
            1usize, 5, 63, 64, 255, 256, 257, 511, 512, 1000, 1023, 2048, 4096,
        ];
        for &(mr, nr) in &[(4usize, 4usize), (8, 6), (8, 8), (16, 6), (16, 16)] {
            let shape = MicroKernelShape { mr, nr };
            for &elem in &[4usize, 8] {
                let t = cache_targets(elem, &shape);
                for &m in &dims {
                    for &k in &dims {
                        for &n in &dims {
                            let ctx = format!("{mr}x{nr} elem={elem} ({m},{k},{n})");
                            let plain = compute_blocking(m, k, n, elem, &shape);
                            let adaptive = compute_blocking_adaptive(m, k, n, elem, &shape);
                            for blk in [plain, adaptive] {
                                check_block(blk.mc, m, mr, &ctx);
                                check_block(blk.kc, k, KC_BLOCK_UNIT, &ctx);
                                check_block(blk.nc, n, nr, &ctx);
                            }
                            // A dimension within its cache target is one block.
                            assert!(m > t.mc || plain.mc == m, "{ctx}: {plain:?}");
                            assert!(k > t.kc || plain.kc == k, "{ctx}: {plain:?}");
                            assert!(n > t.nc || plain.nc == n, "{ctx}: {plain:?}");
                        }
                    }
                }
            }
        }
    }

    /// A sysctl lookup backed by a fixed key/value table (`None` = missing or
    /// empty key).
    fn fake_sysctl(table: &'static [(&'static str, usize)]) -> impl Fn(&str) -> Option<usize> {
        move |key| {
            table
                .iter()
                .find(|(k, _)| *k == key)
                .map(|&(_, value)| value)
        }
    }

    /// The keys an Apple M3 reports (`hw.l3cachesize` and
    /// `hw.perflevel0.l3cachesize` print an empty value, so they are absent).
    const M3_SYSCTL: &[(&str, usize)] = &[
        ("hw.l1dcachesize", 64 * 1024),
        ("hw.l2cachesize", 4 * 1024 * 1024),
        ("hw.cachelinesize", 128),
        ("hw.perflevel0.l1dcachesize", 128 * 1024),
        ("hw.perflevel0.l2cachesize", 16 * 1024 * 1024),
        ("hw.perflevel0.cpusperl2", 4),
        ("hw.perflevel1.l1dcachesize", 64 * 1024),
        ("hw.perflevel1.l2cachesize", 4 * 1024 * 1024),
        ("hw.perflevel1.cpusperl2", 4),
    ];

    #[test]
    fn test_macos_cache_info_m3_uses_performance_cores() {
        let info = macos_cache_info(fake_sysctl(M3_SYSCTL)).expect("M3 keys");
        assert_eq!(info.l1d, 128 * 1024);
        // 16 MiB shared by 4 performance cores: 4 MiB per core.
        assert_eq!(info.l2, 4 * 1024 * 1024);
        assert_eq!(info.l3, 32 * 1024 * 1024);
        assert_eq!(info.line_size, 128);
    }

    #[test]
    fn test_macos_cache_info_intel_falls_back_per_key() {
        // Intel Macs have no perflevel keys; each value uses its hw.* key.
        const INTEL: &[(&str, usize)] = &[
            ("hw.l1dcachesize", 32 * 1024),
            ("hw.l2cachesize", 256 * 1024),
            ("hw.l3cachesize", 12 * 1024 * 1024),
            ("hw.cachelinesize", 64),
        ];
        let info = macos_cache_info(fake_sysctl(INTEL)).expect("Intel keys");
        assert_eq!(
            (info.l1d, info.l2, info.l3, info.line_size),
            (32 * 1024, 256 * 1024, 12 * 1024 * 1024, 64)
        );

        // Partial perflevel information: the L2 without a core count is used
        // whole, a missing L3 is estimated, a missing line size defaults.
        const PARTIAL: &[(&str, usize)] = &[
            ("hw.l1dcachesize", 48 * 1024),
            ("hw.perflevel0.l2cachesize", 2 * 1024 * 1024),
        ];
        let info = macos_cache_info(fake_sysctl(PARTIAL)).expect("partial keys");
        assert_eq!(
            (info.l1d, info.l2, info.l3, info.line_size),
            (48 * 1024, 2 * 1024 * 1024, 16 * 1024 * 1024, 64)
        );
    }

    #[test]
    fn test_macos_cache_info_requires_l1_and_l2() {
        const NO_L2: &[(&str, usize)] = &[("hw.l1dcachesize", 64 * 1024)];
        assert!(macos_cache_info(fake_sysctl(NO_L2)).is_none());
        const NO_L1: &[(&str, usize)] = &[("hw.perflevel0.l2cachesize", 1 << 24)];
        assert!(macos_cache_info(fake_sysctl(NO_L1)).is_none());
    }

    #[test]
    fn test_m3_perflevel_detection_keeps_blocking() {
        // On an M3 the efficiency-core keys (the previous detection) and the
        // performance-core keys differ only in the L1D, which never binds:
        // every GEMM block target, hence every blocking, is unchanged.
        let old = CacheInfo {
            l1d: 64 * 1024,
            l2: 4 * 1024 * 1024,
            l3: 32 * 1024 * 1024,
            line_size: 128,
        };
        let new = macos_cache_info(fake_sysctl(M3_SYSCTL)).expect("M3 keys");
        // The micro-kernels an aarch64 Mac runs: NEON f64 8x6 and f32 8x8,
        // the scalar 4x4 (`force-scalar`), and the complex 2x2 kernels
        // (Complex32 is 8 bytes, Complex64 16).
        for &(mr, nr, elem) in &[
            (8usize, 6usize, 8usize),
            (8, 8, 4),
            (4, 4, 8),
            (4, 4, 4),
            (2, 2, 8),
            (2, 2, 16),
        ] {
            let shape = MicroKernelShape { mr, nr };
            let a = cache_targets_for(&old, elem, &shape);
            let b = cache_targets_for(&new, elem, &shape);
            assert_eq!(
                (a.mc, a.kc, a.nc),
                (b.mc, b.kc, b.nc),
                "{mr}x{nr} elem={elem}"
            );
        }
    }

    #[test]
    fn test_issue_4_reporter_shapes_keep_n_in_one_block() {
        // f64 on an 8x6 micro-kernel: 256 is not a multiple of NR = 6.
        // Reporter's cases: NC was 252 (blocks [252, 4]) for 1024x1024x256
        // and 168 (blocks [168, 88]) for the tall-thin 4096x512x256.
        let shape = MicroKernelShape { mr: 8, nr: 6 };
        for &(m, k, n) in &[(1024usize, 1024usize, 256usize), (4096, 512, 256)] {
            let blk = compute_blocking_adaptive(m, k, n, 8, &shape);
            assert_eq!(blk.nc, n, "({m},{k},{n}) adaptive: {blk:?}");
            let plain = compute_blocking(m, k, n, 8, &shape);
            assert_eq!(plain.nc, n, "({m},{k},{n}) plain: {plain:?}");
        }
    }
}
