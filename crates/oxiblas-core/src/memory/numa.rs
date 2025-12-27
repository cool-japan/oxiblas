//! Memory management utilities for OxiBLAS.
//!
//! This module provides:
//! - Aligned memory allocation
//! - Stack-based temporary allocation (StackReq pattern)
//! - Cache-aware data layout utilities
//! - Prefetch hints for cache optimization
//! - Memory pool for temporary allocations
//! - Custom allocator support via the `Alloc` trait

use core::alloc::Layout;
use core::ptr::NonNull;
use std::alloc::{alloc, alloc_zeroed};

// =============================================================================
// NUMA-aware utilities
// =============================================================================

/// NUMA (Non-Uniform Memory Access) topology information.
///
/// NUMA awareness is crucial for optimal performance on multi-socket systems
/// where memory access latency varies based on which CPU socket is accessing
/// which memory bank.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes in the system.
    pub num_nodes: usize,
    /// CPUs per NUMA node (approximate, may vary).
    pub cpus_per_node: usize,
    /// Total number of CPUs (logical processors).
    pub total_cpus: usize,
}

impl NumaTopology {
    /// Detects the NUMA topology of the current system.
    ///
    /// On non-NUMA systems, returns a topology with 1 node.
    pub fn detect() -> Self {
        let total_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Try to detect NUMA configuration
        #[cfg(target_os = "linux")]
        {
            if let Ok(num_nodes) = Self::detect_linux_numa_nodes() {
                return NumaTopology {
                    num_nodes,
                    cpus_per_node: total_cpus.saturating_div(num_nodes.max(1)),
                    total_cpus,
                };
            }
        }

        // Default: assume single NUMA node
        NumaTopology {
            num_nodes: 1,
            cpus_per_node: total_cpus,
            total_cpus,
        }
    }

    /// Returns true if the system has multiple NUMA nodes.
    #[inline]
    pub fn is_numa_system(&self) -> bool {
        self.num_nodes > 1
    }

    /// Gets the NUMA node ID for a given CPU.
    ///
    /// This is a heuristic based on typical CPU-to-node mappings.
    #[inline]
    pub fn cpu_to_node(&self, cpu_id: usize) -> usize {
        if self.num_nodes <= 1 {
            0
        } else {
            // Simple heuristic: divide CPUs evenly across nodes
            cpu_id.saturating_div(self.cpus_per_node.max(1)) % self.num_nodes
        }
    }

    /// Gets the range of CPUs on a given NUMA node.
    pub fn node_cpu_range(&self, node_id: usize) -> (usize, usize) {
        let start = node_id * self.cpus_per_node;
        let end = ((node_id + 1) * self.cpus_per_node).min(self.total_cpus);
        (start, end)
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_numa_nodes() -> Result<usize, std::io::Error> {
        use std::fs;

        // Count directories in /sys/devices/system/node/
        let node_path = std::path::Path::new("/sys/devices/system/node");
        if !node_path.exists() {
            return Ok(1);
        }

        let mut count = 0;
        for entry in fs::read_dir(node_path)? {
            let entry = entry?;
            let name = entry.file_name();
            if let Some(name_str) = name.to_str() {
                if name_str.starts_with("node") {
                    count += 1;
                }
            }
        }

        Ok(count.max(1))
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

/// NUMA-aware work distribution hint.
///
/// This struct helps distribute work across NUMA nodes to maximize memory locality.
#[derive(Debug, Clone, Copy)]
pub struct NumaWorkHint {
    /// The NUMA node this work should ideally run on.
    pub preferred_node: usize,
    /// Start index of the work range.
    pub range_start: usize,
    /// End index of the work range (exclusive).
    pub range_end: usize,
}

/// Distributes work across NUMA nodes for optimal memory locality.
///
/// Given a total work size and NUMA topology, returns hints for how to
/// distribute the work to maximize local memory access.
///
/// # Arguments
/// * `total_size` - Total number of work items
/// * `topology` - NUMA topology of the system
///
/// # Returns
/// A vector of work hints, one per NUMA node.
pub fn numa_distribute_work(total_size: usize, topology: &NumaTopology) -> Vec<NumaWorkHint> {
    let num_nodes = topology.num_nodes;
    if num_nodes <= 1 {
        return vec![NumaWorkHint {
            preferred_node: 0,
            range_start: 0,
            range_end: total_size,
        }];
    }

    let base_chunk = total_size / num_nodes;
    let remainder = total_size % num_nodes;

    let mut hints = Vec::with_capacity(num_nodes);
    let mut start = 0;

    for node in 0..num_nodes {
        // Distribute remainder evenly among first nodes
        let chunk_size = base_chunk + if node < remainder { 1 } else { 0 };
        let end = start + chunk_size;

        hints.push(NumaWorkHint {
            preferred_node: node,
            range_start: start,
            range_end: end,
        });

        start = end;
    }

    hints
}

/// Memory interleaving strategy for NUMA systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaInterleavingStrategy {
    /// First-touch policy: Memory is allocated on the node where it's first accessed.
    FirstTouch,
    /// Interleave pages across all nodes (round-robin).
    Interleave,
    /// Prefer allocation on a specific node.
    PreferNode(usize),
    /// Bind allocation to a specific node (strict).
    BindNode(usize),
}

/// NUMA memory allocation hint.
///
/// Provides hints for optimal memory allocation on NUMA systems.
pub struct NumaAllocHint {
    /// The interleaving strategy to use.
    pub strategy: NumaInterleavingStrategy,
    /// Preferred page size (0 for default).
    pub page_size: usize,
    /// Whether to pre-fault pages (touch all pages after allocation).
    pub prefault: bool,
}

impl Default for NumaAllocHint {
    fn default() -> Self {
        NumaAllocHint {
            strategy: NumaInterleavingStrategy::FirstTouch,
            page_size: 0,
            prefault: false,
        }
    }
}

impl NumaAllocHint {
    /// Creates a hint for first-touch allocation (default).
    pub fn first_touch() -> Self {
        Self::default()
    }

    /// Creates a hint for interleaved allocation.
    pub fn interleaved() -> Self {
        NumaAllocHint {
            strategy: NumaInterleavingStrategy::Interleave,
            ..Self::default()
        }
    }

    /// Creates a hint for allocation on a specific node.
    pub fn on_node(node: usize) -> Self {
        NumaAllocHint {
            strategy: NumaInterleavingStrategy::PreferNode(node),
            ..Self::default()
        }
    }

    /// Creates a hint for bound allocation on a specific node.
    pub fn bind_node(node: usize) -> Self {
        NumaAllocHint {
            strategy: NumaInterleavingStrategy::BindNode(node),
            ..Self::default()
        }
    }

    /// Enables pre-faulting of pages.
    pub fn with_prefault(mut self) -> Self {
        self.prefault = true;
        self
    }
}

/// Allocates memory with NUMA awareness.
///
/// This function allocates memory and optionally applies NUMA hints.
/// On systems without NUMA support, falls back to standard allocation.
///
/// # Arguments
/// * `layout` - Memory layout to allocate
/// * `hint` - NUMA allocation hint
///
/// # Safety
/// The returned pointer must be deallocated with `dealloc` using the same layout.
pub unsafe fn numa_alloc(layout: Layout, hint: &NumaAllocHint) -> Option<NonNull<u8>> {
    // Allocate memory using standard allocator
    let ptr = alloc(layout);
    if ptr.is_null() {
        return None;
    }

    // Apply NUMA policy if available
    #[cfg(target_os = "linux")]
    {
        apply_linux_numa_policy(ptr, layout.size(), hint);
    }

    // Pre-fault pages if requested
    if hint.prefault {
        prefault_pages(ptr, layout.size());
    }

    NonNull::new(ptr)
}

/// Allocates zeroed memory with NUMA awareness.
///
/// # Safety
///
/// The caller must ensure that:
/// - The layout's size and alignment are valid (non-zero alignment, size doesn't overflow)
/// - The returned memory must be properly deallocated when no longer needed
pub unsafe fn numa_alloc_zeroed(layout: Layout, hint: &NumaAllocHint) -> Option<NonNull<u8>> {
    let ptr = alloc_zeroed(layout);
    if ptr.is_null() {
        return None;
    }

    #[cfg(target_os = "linux")]
    {
        apply_linux_numa_policy(ptr, layout.size(), hint);
    }

    // Suppress warning on non-Linux platforms
    #[cfg(not(target_os = "linux"))]
    let _ = hint;

    // Note: zeroing already touches all pages, so prefault is implicit

    NonNull::new(ptr)
}

/// Pre-faults (touches) all pages in a memory region.
///
/// This ensures all pages are physically allocated and mapped.
fn prefault_pages(ptr: *mut u8, size: usize) {
    const PAGE_SIZE: usize = 4096;
    let mut offset = 0;

    while offset < size {
        unsafe {
            // Write a byte to fault the page
            core::ptr::write_volatile(ptr.add(offset), 0);
        }
        offset += PAGE_SIZE;
    }
}

#[cfg(target_os = "linux")]
fn apply_linux_numa_policy(ptr: *mut u8, size: usize, hint: &NumaAllocHint) {
    use std::os::raw::c_int;

    // NUMA policy constants (from numaif.h)
    const _MPOL_DEFAULT: c_int = 0;
    const MPOL_PREFERRED: c_int = 1;
    const MPOL_BIND: c_int = 2;
    const MPOL_INTERLEAVE: c_int = 3;

    unsafe extern "C" {
        fn mbind(
            addr: *mut u8,
            len: usize,
            mode: c_int,
            nodemask: *const usize,
            maxnode: usize,
            flags: u32,
        ) -> c_int;
    }

    let (mode, node_mask, max_node) = match hint.strategy {
        NumaInterleavingStrategy::FirstTouch => return, // Default behavior
        NumaInterleavingStrategy::Interleave => {
            // Set all nodes in mask
            let mask: usize = !0;
            (MPOL_INTERLEAVE, mask, 64)
        }
        NumaInterleavingStrategy::PreferNode(node) => {
            let mask: usize = 1 << node;
            (MPOL_PREFERRED, mask, node + 1)
        }
        NumaInterleavingStrategy::BindNode(node) => {
            let mask: usize = 1 << node;
            (MPOL_BIND, mask, node + 1)
        }
    };

    unsafe {
        // Ignore errors - NUMA may not be available
        let _ = mbind(ptr, size, mode, &node_mask, max_node, 0);
    }
}

/// Gets the current system's page size.
pub fn get_page_size() -> usize {
    #[cfg(unix)]
    {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }
    #[cfg(not(unix))]
    {
        4096 // Common default
    }
}

/// Gets the system's huge page size (if supported).
///
/// Returns `None` if huge pages are not supported or cannot be determined.
pub fn get_huge_page_size() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("Hugepagesize:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_distribute_work() {
        let topo = NumaTopology {
            num_nodes: 4,
            cpus_per_node: 8,
            total_cpus: 32,
        };

        let hints = numa_distribute_work(1000, &topo);
        assert_eq!(hints.len(), 4);

        // Verify all work is covered
        let total: usize = hints.iter().map(|h| h.range_end - h.range_start).sum();
        assert_eq!(total, 1000);

        // Verify ranges are contiguous
        for i in 1..hints.len() {
            assert_eq!(hints[i].range_start, hints[i - 1].range_end);
        }
    }

    #[test]
    fn test_numa_distribute_work_single_node() {
        let topo = NumaTopology {
            num_nodes: 1,
            cpus_per_node: 8,
            total_cpus: 8,
        };

        let hints = numa_distribute_work(500, &topo);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].range_start, 0);
        assert_eq!(hints[0].range_end, 500);
    }

    #[test]
    fn test_numa_alloc_hint_builders() {
        let hint = NumaAllocHint::first_touch();
        assert_eq!(hint.strategy, NumaInterleavingStrategy::FirstTouch);
        assert!(!hint.prefault);

        let hint = NumaAllocHint::interleaved();
        assert_eq!(hint.strategy, NumaInterleavingStrategy::Interleave);

        let hint = NumaAllocHint::on_node(2);
        assert_eq!(hint.strategy, NumaInterleavingStrategy::PreferNode(2));

        let hint = NumaAllocHint::bind_node(1).with_prefault();
        assert_eq!(hint.strategy, NumaInterleavingStrategy::BindNode(1));
        assert!(hint.prefault);
    }

    #[test]
    fn test_get_page_size() {
        let page_size = get_page_size();

        // Page size should be a power of 2
        assert!(page_size.is_power_of_two());

        // Common page sizes
        assert!(page_size >= 4096);
        assert!(page_size <= 65536);
    }
}
