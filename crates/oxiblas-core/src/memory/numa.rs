//! Memory management utilities for OxiBLAS.
//!
//! This module provides:
//! - Aligned memory allocation
//! - Stack-based temporary allocation (StackReq pattern)
//! - Cache-aware data layout utilities
//! - Prefetch hints for cache optimization
//! - Memory pool for temporary allocations
//! - Custom allocator support via the `Alloc` trait
//!
//! Note: NUMA-aware allocation requires the `std` feature.

use core::alloc::Layout;
use core::ptr::NonNull;
use std::alloc::{alloc, alloc_zeroed, dealloc};

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
            if let Some(name_str) = name.to_str()
                && name_str.starts_with("node")
            {
                count += 1;
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

/// Returns `true` when a strategy needs an explicit kernel NUMA policy
/// (i.e. anything other than the kernel-default first-touch behaviour).
#[inline]
fn requires_mbind(strategy: NumaInterleavingStrategy) -> bool {
    !matches!(strategy, NumaInterleavingStrategy::FirstTouch)
}

/// Rounds `value` up to the next multiple of `page`.
///
/// `page` must be a non-zero power of two (guaranteed by [`get_page_size`]).
/// Returns `None` on overflow rather than wrapping to a bogus small value.
#[inline]
fn round_up_to(value: usize, page: usize) -> Option<usize> {
    debug_assert!(page.is_power_of_two() && page != 0);
    value.checked_add(page - 1).map(|v| v & !(page - 1))
}

/// Computes the *effective* layout used for a NUMA allocation.
///
/// `mbind(2)` operates at page granularity and requires a page-aligned start
/// address, so any allocation that will carry an explicit NUMA policy must be
/// page-aligned and sized to a whole number of pages. This transform is
/// deterministic in `(layout, hint)`, so [`numa_alloc`] and [`numa_dealloc`]
/// derive the identical layout and stay allocator-consistent.
///
/// Zero-sized and first-touch allocations pass through unchanged.
fn effective_layout(layout: Layout, hint: &NumaAllocHint) -> Option<Layout> {
    if layout.size() == 0 || !requires_mbind(hint.strategy) {
        return Some(layout);
    }
    let page = get_page_size();
    if page == 0 || !page.is_power_of_two() {
        // Can't reason about pages; skip the bump (binding then becomes a
        // no-op, which the best-effort contract already tolerates).
        return Some(layout);
    }
    let align = layout.align().max(page);
    let size = round_up_to(layout.size(), page)?;
    Layout::from_size_align(size, align).ok()
}

/// Allocates memory with NUMA awareness.
///
/// The allocation is placed according to `hint`. For any strategy other than
/// first-touch the region is page-aligned and page-sized so that the kernel
/// NUMA policy (applied via [`bind_memory_policy`]) provably covers exactly the
/// allocation. On systems without NUMA support the binding is a best-effort
/// no-op and this degrades to a plain (still page-aligned) allocation.
///
/// # Arguments
/// * `layout` - Memory layout to allocate
/// * `hint` - NUMA allocation hint
///
/// # Safety
/// The returned pointer must be released with [`numa_dealloc`] using the **same**
/// `layout` and `hint`; deallocating with a different layout/hint (e.g. the raw
/// [`std::alloc::dealloc`]) is undefined behaviour because the real allocation
/// may have been page-aligned and rounded up.
pub unsafe fn numa_alloc(layout: Layout, hint: &NumaAllocHint) -> Option<NonNull<u8>> {
    // Zero-sized allocations must never reach the global allocator: calling
    // `alloc` with a zero-size layout is undefined behaviour. Mirror the
    // `GlobalAlloc`/ZST convention and hand back a dangling-but-aligned pointer
    // (a valid, non-null, correctly-aligned address that is never dereferenced).
    if layout.size() == 0 {
        return NonNull::new(core::ptr::without_provenance_mut::<u8>(layout.align()));
    }

    let effective = effective_layout(layout, hint)?;
    // Safety: `effective` has non-zero size (>= layout.size() > 0) and a valid
    // power-of-two alignment.
    let ptr = unsafe { alloc(effective) };
    if ptr.is_null() {
        return None;
    }

    if requires_mbind(hint.strategy) {
        // Best-effort NUMA binding. A failure here (no NUMA hardware, running
        // inside a container, a seccomp filter blocking `mbind(2)`, ...) does
        // NOT invalidate the allocation, so we deliberately keep the memory
        // rather than break the documented non-NUMA fallback. The errno is not
        // silently lost: `bind_memory_policy` materialises it into an
        // `io::Error` that callers needing certainty can observe directly.
        if let Err(_bind_err) = unsafe { bind_memory_policy(ptr, effective.size(), hint) } {
            // Intentionally ignored (see rationale above).
        }
    }

    // Pre-fault pages if requested (faults them in under the freshly-applied
    // policy).
    if hint.prefault {
        prefault_pages(ptr, effective.size());
    }

    NonNull::new(ptr)
}

/// Allocates zeroed memory with NUMA awareness.
///
/// Note on placement: the binding is applied *after* zero-initialisation. If
/// the underlying allocator eagerly faults pages while zeroing, the policy only
/// affects pages that are not yet resident (`mbind` without `MPOL_MF_MOVE` does
/// not migrate present pages). When strict placement of zeroed memory is
/// required, prefer [`numa_alloc`] followed by explicit zeroing so the zeroing
/// writes fault pages in under the policy.
///
/// # Safety
///
/// The returned pointer must be released with [`numa_dealloc`] using the **same**
/// `layout` and `hint` (see [`numa_alloc`]).
pub unsafe fn numa_alloc_zeroed(layout: Layout, hint: &NumaAllocHint) -> Option<NonNull<u8>> {
    // Zero-sized layouts: see `numa_alloc`.
    if layout.size() == 0 {
        return NonNull::new(core::ptr::without_provenance_mut::<u8>(layout.align()));
    }

    let effective = effective_layout(layout, hint)?;
    // Safety: `effective` has non-zero size and a valid power-of-two alignment.
    let ptr = unsafe { alloc_zeroed(effective) };
    if ptr.is_null() {
        return None;
    }

    if requires_mbind(hint.strategy) {
        // Best-effort binding; see `numa_alloc` for why failures are tolerated.
        if let Err(_bind_err) = unsafe { bind_memory_policy(ptr, effective.size(), hint) } {
            // Intentionally ignored.
        }
    }

    // Zeroing already touches all pages, so prefault is implicit.

    NonNull::new(ptr)
}

/// Releases memory obtained from [`numa_alloc`] / [`numa_alloc_zeroed`].
///
/// # Safety
///
/// * `ptr` must have been returned by [`numa_alloc`] or [`numa_alloc_zeroed`]
///   for exactly this `layout` and `hint`.
/// * After this call `ptr` must not be used.
pub unsafe fn numa_dealloc(ptr: NonNull<u8>, layout: Layout, hint: &NumaAllocHint) {
    if layout.size() == 0 {
        // ZST: `numa_alloc` returned a dangling pointer without touching the
        // allocator, so there is nothing to free.
        return;
    }
    if let Some(effective) = effective_layout(layout, hint) {
        // Safety: `effective` is byte-for-byte the layout used to allocate this
        // pointer (identical `layout` + `hint` yield the identical transform).
        unsafe { dealloc(ptr.as_ptr(), effective) };
    }
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

/// Applies a NUMA memory policy to a page-aligned region via `mbind(2)`.
///
/// This is the honest, error-returning primitive underpinning the NUMA
/// allocators. Unlike a fire-and-forget syscall it does not swallow the errno:
/// on any failure it returns the corresponding [`std::io::Error`]. The higher
/// level [`numa_alloc`] treats binding as best-effort, but callers that need
/// certainty can invoke this directly and inspect the result.
///
/// First-touch always succeeds as a no-op. On non-Linux targets this is a
/// no-op returning `Ok(())` (no `mbind` equivalent exists).
///
/// # Safety
///
/// `ptr` must point to a caller-owned mapping of at least `len` bytes. For any
/// strategy other than first-touch `ptr` must additionally be page-aligned
/// (`mbind` returns `EINVAL` otherwise, which this function surfaces as an
/// error). Passing a range the caller does not own could alter the NUMA policy
/// of unrelated memory.
pub unsafe fn bind_memory_policy(
    ptr: *mut u8,
    len: usize,
    hint: &NumaAllocHint,
) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux_bind_memory_policy(ptr, len, hint)
    }
    #[cfg(not(target_os = "linux"))]
    {
        // No `mbind` equivalent: first-touch is the only meaningful policy and
        // it is already the platform default, so this is a genuine no-op.
        let _ = (ptr, len, hint);
        Ok(())
    }
}

/// Parses a Linux `cpulist`/`nodelist` string (e.g. `"0-1,4"`) into node ids.
///
/// Malformed fragments are skipped rather than panicking; the caller decides
/// what an empty result means.
#[cfg(target_os = "linux")]
fn parse_node_list(text: &str) -> Vec<usize> {
    let mut nodes = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) =
                (start.trim().parse::<usize>(), end.trim().parse::<usize>())
            {
                for node in start..=end {
                    nodes.push(node);
                }
            }
        } else if let Ok(node) = part.parse::<usize>() {
            nodes.push(node);
        }
    }
    nodes
}

/// Reads the set of online NUMA nodes from
/// `/sys/devices/system/node/online`. Returns `None` if the file is absent or
/// unreadable (e.g. a kernel built without `CONFIG_NUMA`, or a restricted
/// container), or if it lists no nodes.
#[cfg(target_os = "linux")]
fn read_online_nodes() -> Option<Vec<usize>> {
    let content = std::fs::read_to_string("/sys/devices/system/node/online").ok()?;
    let nodes = parse_node_list(&content);
    if nodes.is_empty() { None } else { Some(nodes) }
}

/// Upper bound on NUMA node ids we are willing to encode. Matches the common
/// kernel `MAX_NUMNODES` ceiling and guards against building an absurdly large
/// bitmask from a bogus node id.
#[cfg(target_os = "linux")]
const MAX_NUMA_NODES: usize = 1024;

/// Validates a requested node id: it must be within [`MAX_NUMA_NODES`] and,
/// when the online set is known, actually online.
#[cfg(target_os = "linux")]
fn validate_numa_node(node: usize) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    if node >= MAX_NUMA_NODES {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "NUMA node id exceeds supported maximum",
        ));
    }
    if let Some(online) = read_online_nodes()
        && !online.contains(&node)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "requested NUMA node is not online",
        ));
    }
    Ok(())
}

/// Builds the `mbind` node bitmask (array of `unsigned long`) plus the
/// `maxnode` argument for the given node ids.
///
/// `maxnode` follows the libnuma convention of `highest_node + 2`: the kernel
/// decrements it internally (`--maxnode` in `get_nodes`), so this makes the
/// kernel read exactly the words we provide and keep the top set bit, avoiding
/// the classic final-partial-word off-by-one. Correct for both 64-bit
/// (`c_ulong` = 64 bits) and 32-bit (`c_ulong` = 32 bits) Linux.
#[cfg(target_os = "linux")]
fn build_node_mask(nodes: &[usize]) -> (Vec<libc::c_ulong>, usize) {
    let bits_per_word = core::mem::size_of::<libc::c_ulong>() * 8;
    let highest = nodes.iter().copied().max().unwrap_or(0);
    let num_words = highest / bits_per_word + 1;
    let mut mask: Vec<libc::c_ulong> = vec![0; num_words];
    for &node in nodes {
        let word = node / bits_per_word;
        let bit = node % bits_per_word;
        let bit_mask: libc::c_ulong = 1 << bit;
        mask[word] |= bit_mask;
    }
    (mask, highest + 2)
}

/// Linux implementation of [`bind_memory_policy`].
#[cfg(target_os = "linux")]
fn linux_bind_memory_policy(ptr: *mut u8, len: usize, hint: &NumaAllocHint) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};

    // NUMA policy modes (from numaif.h).
    const MPOL_PREFERRED: libc::c_int = 1;
    const MPOL_BIND: libc::c_int = 2;
    const MPOL_INTERLEAVE: libc::c_int = 3;

    let (mode, nodes): (libc::c_int, Vec<usize>) = match hint.strategy {
        // First-touch is the kernel default; nothing to bind.
        NumaInterleavingStrategy::FirstTouch => return Ok(()),
        NumaInterleavingStrategy::Interleave => {
            // A full !0 mask (the previous behaviour) names offline nodes and
            // is rejected with EINVAL. Restrict interleaving to actually-online
            // nodes so the policy is accepted and effective.
            let online = read_online_nodes().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unsupported,
                    "cannot determine online NUMA nodes for interleaving",
                )
            })?;
            (MPOL_INTERLEAVE, online)
        }
        NumaInterleavingStrategy::PreferNode(node) => {
            validate_numa_node(node)?;
            (MPOL_PREFERRED, vec![node])
        }
        NumaInterleavingStrategy::BindNode(node) => {
            validate_numa_node(node)?;
            (MPOL_BIND, vec![node])
        }
    };

    let page = get_page_size();
    if page == 0 || !page.is_power_of_two() {
        return Err(Error::other("invalid system page size"));
    }
    // `mbind` requires a page-aligned start address (the kernel rounds the
    // length up to a page multiple itself, but we do it explicitly so the
    // policy provably spans the whole region).
    if (ptr as usize) % page != 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "mbind requires a page-aligned address",
        ));
    }
    let len_aligned = round_up_to(len, page)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "length overflow"))?;
    if len_aligned == 0 {
        return Ok(());
    }

    let (mask, maxnode) = build_node_mask(&nodes);

    // Safety: `libc::syscall` forwards to the C `syscall(2)` wrapper, which
    // sets `errno` and returns -1 on failure. `libc::SYS_mbind` is the
    // arch-correct syscall number the libc crate provides for the target (a
    // `c_long` on every supported architecture). `ptr` refers to a
    // caller-owned, page-aligned mapping of at least `len_aligned` bytes, and
    // `mask`/`maxnode` describe a valid nodemask.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            ptr as *mut libc::c_void,
            len_aligned as libc::c_ulong,
            mode,
            mask.as_ptr(),
            maxnode as libc::c_ulong,
            0 as libc::c_uint,
        )
    };
    if ret < 0 {
        Err(Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Gets the current system's page size in bytes.
///
/// Falls back to the near-universal 4 KiB default when the size cannot be
/// determined.
pub fn get_page_size() -> usize {
    #[cfg(unix)]
    {
        // `sysconf` returns -1 (as `c_long`) on error and leaves `errno` set.
        // Casting that negative value straight to `usize` would yield an
        // enormous bogus "page size" (~1.8e19 on 64-bit), which then poisons
        // every page-alignment computation. Check the sign first and fall back
        // to 4 KiB — the smallest and by far most common page size, and a valid
        // power of two — when the query fails or reports a non-positive size.
        let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if raw > 0 { raw as usize } else { 4096 }
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
                    if parts.len() >= 2
                        && let Ok(kb) = parts[1].parse::<usize>()
                    {
                        return Some(kb * 1024);
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

// =============================================================================
// NumaAllocator - typed allocator with NUMA affinity
// =============================================================================

/// A typed allocator that places memory on a preferred NUMA node.
///
/// `NumaAllocator<T>` wraps allocation of `T`-typed data with a
/// `NumaAllocHint` so that the resulting memory respects the chosen
/// NUMA interleaving strategy.  On non-NUMA platforms (or when the
/// kernel call is unavailable) it silently falls back to the standard
/// global allocator.
///
/// # Example
///
/// ```rust
/// use oxiblas_core::memory::numa::{NumaAllocator, NumaAllocHint};
///
/// let alloc: NumaAllocator<f64> = NumaAllocator::new(NumaAllocHint::on_node(0));
/// let ptr = alloc.allocate(16).expect("allocation failed");
/// unsafe { alloc.deallocate(ptr, 16); }
/// ```
pub struct NumaAllocator<T> {
    hint: NumaAllocHint,
    _marker: core::marker::PhantomData<T>,
}

impl<T> NumaAllocator<T> {
    /// Creates a new `NumaAllocator` with the given hint.
    #[inline]
    pub fn new(hint: NumaAllocHint) -> Self {
        NumaAllocator {
            hint,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a `NumaAllocator` that prefers the given NUMA node.
    #[inline]
    pub fn on_node(node: usize) -> Self {
        Self::new(NumaAllocHint::on_node(node))
    }

    /// Creates a `NumaAllocator` that interleaves across all nodes.
    #[inline]
    pub fn interleaved() -> Self {
        Self::new(NumaAllocHint::interleaved())
    }

    /// Creates a `NumaAllocator` with first-touch policy (the default).
    #[inline]
    pub fn first_touch() -> Self {
        Self::new(NumaAllocHint::first_touch())
    }

    /// Allocates `count` elements of type `T`.
    ///
    /// Returns `None` on allocation failure or if `count` is zero.
    pub fn allocate(&self, count: usize) -> Option<NonNull<T>> {
        if count == 0 {
            return None;
        }
        let layout = Layout::array::<T>(count).ok()?;
        // Safety: layout is valid (non-zero size, proper alignment).
        let raw = unsafe { numa_alloc(layout, &self.hint) }?;
        Some(raw.cast::<T>())
    }

    /// Allocates `count` zero-initialised elements of type `T`.
    ///
    /// Returns `None` on allocation failure or if `count` is zero.
    pub fn allocate_zeroed(&self, count: usize) -> Option<NonNull<T>> {
        if count == 0 {
            return None;
        }
        let layout = Layout::array::<T>(count).ok()?;
        // Safety: layout is valid.
        let raw = unsafe { numa_alloc_zeroed(layout, &self.hint) }?;
        Some(raw.cast::<T>())
    }

    /// Deallocates a pointer previously returned by `allocate` or
    /// `allocate_zeroed` for `count` elements.
    ///
    /// # Safety
    ///
    /// * `ptr` must have been returned by this allocator for exactly `count`
    ///   elements.
    /// * After this call `ptr` must not be used.
    pub unsafe fn deallocate(&self, ptr: NonNull<T>, count: usize) {
        if count == 0 {
            return;
        }
        if let Ok(layout) = Layout::array::<T>(count) {
            // Must mirror `numa_alloc`'s effective layout: route through
            // `numa_dealloc` with the same hint rather than the raw allocator.
            unsafe { numa_dealloc(ptr.cast::<u8>(), layout, &self.hint) };
        }
    }
}

// =============================================================================
// NumaVec - Vec-like container with NUMA-aware allocation
// =============================================================================

/// A `Vec`-like container whose backing storage is allocated with a
/// `NumaAllocHint`, enabling preferred-node or interleaved placement.
///
/// On non-NUMA systems the allocation transparently falls back to the
/// standard global allocator, so code written against `NumaVec` is
/// portable.
///
/// # Example
///
/// ```rust
/// use oxiblas_core::memory::numa::{NumaVec, NumaAllocHint};
///
/// let mut v: NumaVec<f64> = NumaVec::with_hint_and_capacity(
///     NumaAllocHint::on_node(0), 128,
/// ).expect("allocation failed");
/// v.push(1.0).expect("push failed");
/// assert_eq!(v.len(), 1);
/// ```
pub struct NumaVec<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    hint: NumaAllocHint,
}

// Safety: NumaVec owns its data and the allocator is platform-level.
unsafe impl<T: Send> Send for NumaVec<T> {}
unsafe impl<T: Sync> Sync for NumaVec<T> {}

impl<T> NumaVec<T> {
    /// Creates an empty `NumaVec` with first-touch allocation policy.
    pub fn new() -> Self {
        NumaVec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            hint: NumaAllocHint::first_touch(),
        }
    }

    /// Creates an empty `NumaVec` with the given allocation hint.
    pub fn with_hint(hint: NumaAllocHint) -> Self {
        NumaVec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            hint,
        }
    }

    /// Allocates a `NumaVec` with at least `capacity` elements reserved.
    ///
    /// Returns an error string on allocation failure.
    pub fn with_hint_and_capacity(
        hint: NumaAllocHint,
        capacity: usize,
    ) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Ok(Self::with_hint(hint));
        }
        let layout = Layout::array::<T>(capacity).map_err(|_| "layout overflow")?;
        // Safety: layout is valid with non-zero size.
        let raw = unsafe { numa_alloc(layout, &hint) }.ok_or("allocation failed")?;
        Ok(NumaVec {
            ptr: raw.cast::<T>(),
            len: 0,
            cap: capacity,
            hint,
        })
    }

    /// Returns the number of elements currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the currently allocated capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns a raw pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns a shared slice over the stored elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // Safety: ptr is valid for `len` initialised elements.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice over the stored elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Safety: ptr is valid for `len` initialised elements.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Appends an element to the end.
    ///
    /// Returns an error string if reallocation is needed and fails.
    pub fn push(&mut self, value: T) -> Result<(), &'static str> {
        if self.len == self.cap {
            self.grow()?;
        }
        // Safety: ptr + len is within the allocation and uninitialised.
        unsafe { core::ptr::write(self.ptr.as_ptr().add(self.len), value) };
        self.len += 1;
        Ok(())
    }

    /// Removes and returns the last element, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // Safety: the element at `len` is initialised and we are taking
        // ownership.
        Some(unsafe { core::ptr::read(self.ptr.as_ptr().add(self.len)) })
    }

    /// Reserves space for at least `additional` more elements.
    ///
    /// Returns an error string if allocation fails.
    pub fn reserve(&mut self, additional: usize) -> Result<(), &'static str> {
        let required = self
            .len
            .checked_add(additional)
            .ok_or("capacity overflow")?;
        if required <= self.cap {
            return Ok(());
        }
        self.realloc(required)
    }

    /// Grows the internal buffer using an exponential strategy.
    fn grow(&mut self) -> Result<(), &'static str> {
        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap.checked_mul(2).ok_or("capacity overflow")?
        };
        self.realloc(new_cap)
    }

    fn realloc(&mut self, new_cap: usize) -> Result<(), &'static str> {
        let new_layout = Layout::array::<T>(new_cap).map_err(|_| "layout overflow")?;
        // Safety: new_layout is valid.
        let new_raw = unsafe { numa_alloc(new_layout, &self.hint) }.ok_or("allocation failed")?;
        let new_ptr = new_raw.cast::<T>();

        if self.cap > 0 {
            // Safety: both src and dst are valid, non-overlapping for `len`.
            unsafe {
                core::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
            // Free old allocation (via the NUMA-aware deallocator so the
            // effective layout matches the one `numa_alloc` used).
            if let Ok(old_layout) = Layout::array::<T>(self.cap) {
                unsafe { numa_dealloc(self.ptr.cast::<u8>(), old_layout, &self.hint) };
            }
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }
}

impl<T> Default for NumaVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for NumaVec<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // Drop initialised elements.
        for i in 0..self.len {
            unsafe { core::ptr::drop_in_place(self.ptr.as_ptr().add(i)) };
        }
        // Deallocate backing store (matching `numa_alloc`'s effective layout).
        if let Ok(layout) = Layout::array::<T>(self.cap) {
            unsafe { numa_dealloc(self.ptr.cast::<u8>(), layout, &self.hint) };
        }
    }
}

impl<T> core::ops::Deref for NumaVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> core::ops::DerefMut for NumaVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// =============================================================================
// MatNuma - Matrix type with NUMA-aware allocation
// =============================================================================

/// A dense, row-major matrix whose backing storage is allocated with a
/// `NumaAllocHint`.
///
/// Element `(row, col)` is stored at index `row * num_cols + col`.
///
/// # Example
///
/// ```rust
/// use oxiblas_core::memory::numa::{MatNuma, NumaAllocHint};
///
/// let mut mat: MatNuma<f64> = MatNuma::zeros(
///     4, 4, NumaAllocHint::on_node(0),
/// ).expect("allocation failed");
/// *mat.get_mut(1, 2).unwrap() = 3.14;
/// assert!((mat.get(1, 2).unwrap() - 3.14).abs() < 1e-12);
/// ```
pub struct MatNuma<T> {
    data: NumaVec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Copy + Default> MatNuma<T> {
    /// Allocates a zero-initialised `rows × cols` matrix on the preferred
    /// NUMA node described by `hint`.
    ///
    /// Elements are value-initialised using `T::default()`.
    pub fn zeros(rows: usize, cols: usize, hint: NumaAllocHint) -> Result<Self, &'static str> {
        let total = rows.checked_mul(cols).ok_or("dimension overflow")?;
        let mut data = NumaVec::with_hint_and_capacity(hint, total)?;
        for _ in 0..total {
            data.push(T::default()).map_err(|_| "push failed")?;
        }
        Ok(MatNuma { data, rows, cols })
    }
}

impl<T> MatNuma<T> {
    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns `true` when the matrix has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Returns a shared reference to the element at `(row, col)`, or `None`
    /// if the indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let idx = row * self.cols + col;
        self.data.as_slice().get(idx)
    }

    /// Returns a mutable reference to the element at `(row, col)`, or `None`
    /// if the indices are out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let idx = row * self.cols + col;
        self.data.as_mut_slice().get_mut(idx)
    }

    /// Returns a flat shared slice of all elements in row-major order.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns a flat mutable slice of all elements in row-major order.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns a shared slice for the given row, or `None` if out of bounds.
    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        self.data.as_slice().get(start..start + self.cols)
    }

    /// Returns a mutable slice for the given row, or `None` if out of bounds.
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [T]> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        self.data.as_mut_slice().get_mut(start..start + self.cols)
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

    // ----- NumaAllocator tests -----------------------------------------------

    #[test]
    fn test_numa_allocator_alloc_dealloc() {
        let alloc: NumaAllocator<f64> = NumaAllocator::first_touch();
        let count = 64usize;
        let ptr = alloc.allocate(count).expect("allocation must succeed");
        // Write and read back to confirm memory is accessible.
        unsafe {
            for i in 0..count {
                core::ptr::write(ptr.as_ptr().add(i), i as f64);
            }
            for i in 0..count {
                assert!((core::ptr::read(ptr.as_ptr().add(i)) - i as f64).abs() < f64::EPSILON);
            }
            alloc.deallocate(ptr, count);
        }
    }

    #[test]
    fn test_numa_allocator_zeroed() {
        let alloc: NumaAllocator<u64> = NumaAllocator::first_touch();
        let count = 32usize;
        let ptr = alloc
            .allocate_zeroed(count)
            .expect("zeroed allocation must succeed");
        unsafe {
            for i in 0..count {
                assert_eq!(core::ptr::read(ptr.as_ptr().add(i)), 0u64);
            }
            alloc.deallocate(ptr, count);
        }
    }

    #[test]
    fn test_numa_allocator_on_node() {
        // Node 0 is always valid; fallback on non-NUMA is fine.
        let alloc: NumaAllocator<f32> = NumaAllocator::on_node(0);
        let ptr = alloc.allocate(16).expect("allocation must succeed");
        unsafe { alloc.deallocate(ptr, 16) };
    }

    #[test]
    fn test_numa_allocator_zero_count_returns_none() {
        let alloc: NumaAllocator<f64> = NumaAllocator::first_touch();
        assert!(alloc.allocate(0).is_none());
        assert!(alloc.allocate_zeroed(0).is_none());
    }

    // ----- NumaVec tests -----------------------------------------------------

    #[test]
    fn test_numa_vec_push_pop() {
        let mut v: NumaVec<i32> = NumaVec::new();
        assert!(v.is_empty());

        for i in 0..100i32 {
            v.push(i).expect("push must succeed");
        }
        assert_eq!(v.len(), 100);

        for i in (0..100i32).rev() {
            assert_eq!(v.pop(), Some(i));
        }
        assert!(v.is_empty());
    }

    #[test]
    fn test_numa_vec_slice_access() {
        let mut v: NumaVec<f64> = NumaVec::new();
        for i in 0..10 {
            v.push(i as f64).expect("push");
        }
        let s = v.as_slice();
        assert_eq!(s.len(), 10);
        for (i, &x) in s.iter().enumerate() {
            assert!((x - i as f64).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_numa_vec_with_hint_and_capacity() {
        let v = NumaVec::<f64>::with_hint_and_capacity(NumaAllocHint::first_touch(), 128)
            .expect("alloc");
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 128);
    }

    #[test]
    fn test_numa_vec_reserve() {
        let mut v: NumaVec<u8> = NumaVec::new();
        v.reserve(256).expect("reserve");
        assert!(v.capacity() >= 256);
    }

    #[test]
    fn test_numa_vec_interleaved() {
        let mut v = NumaVec::<u32>::with_hint(NumaAllocHint::interleaved());
        for i in 0..50u32 {
            v.push(i).expect("push");
        }
        assert_eq!(v.len(), 50);
        assert_eq!(v[0], 0);
        assert_eq!(v[49], 49);
    }

    // ----- MatNuma tests -----------------------------------------------------

    #[test]
    fn test_mat_numa_zeros() {
        let mat: MatNuma<f64> = MatNuma::zeros(4, 4, NumaAllocHint::first_touch()).expect("zeros");
        assert_eq!(mat.nrows(), 4);
        assert_eq!(mat.ncols(), 4);
        assert_eq!(mat.len(), 16);
        for &v in mat.as_slice() {
            assert_eq!(v, 0.0f64);
        }
    }

    #[test]
    fn test_mat_numa_get_set() {
        let mut mat: MatNuma<f64> =
            MatNuma::zeros(3, 5, NumaAllocHint::first_touch()).expect("zeros");
        *mat.get_mut(1, 3).expect("valid index") = 42.0;
        assert!((mat.get(1, 3).expect("valid index") - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mat_numa_out_of_bounds() {
        let mat: MatNuma<f32> = MatNuma::zeros(2, 2, NumaAllocHint::first_touch()).expect("zeros");
        assert!(mat.get(2, 0).is_none());
        assert!(mat.get(0, 2).is_none());
    }

    #[test]
    fn test_mat_numa_row_slice() {
        let mut mat: MatNuma<i32> =
            MatNuma::zeros(3, 4, NumaAllocHint::first_touch()).expect("zeros");
        if let Some(row) = mat.row_mut(1) {
            for (i, v) in row.iter_mut().enumerate() {
                *v = i as i32 * 10;
            }
        }
        let row = mat.row(1).expect("valid row");
        assert_eq!(row, &[0, 10, 20, 30]);
    }

    #[test]
    fn test_mat_numa_on_node_zero() {
        // Node 0 is always valid; on non-NUMA systems falls back silently.
        let mat: MatNuma<f64> = MatNuma::zeros(8, 8, NumaAllocHint::on_node(0)).expect("zeros");
        assert_eq!(mat.len(), 64);
    }

    #[test]
    fn test_mat_numa_fallback_non_numa() {
        // Interleaved hint on non-NUMA falls back to standard alloc.
        let mat: MatNuma<f32> =
            MatNuma::zeros(16, 16, NumaAllocHint::interleaved()).expect("zeros");
        assert_eq!(mat.len(), 256);
        for &v in mat.as_slice() {
            assert_eq!(v, 0.0f32);
        }
    }

    // ----- page-alignment arithmetic (finding: mbind alignment) --------------

    #[test]
    fn test_round_up_to_page_math() {
        assert_eq!(round_up_to(0, 4096), Some(0));
        assert_eq!(round_up_to(1, 4096), Some(4096));
        assert_eq!(round_up_to(4095, 4096), Some(4096));
        assert_eq!(round_up_to(4096, 4096), Some(4096));
        assert_eq!(round_up_to(4097, 4096), Some(8192));
        assert_eq!(round_up_to(12288, 4096), Some(12288));
        // A non-4096 (but power-of-two) page size.
        assert_eq!(round_up_to(1, 16384), Some(16384));
        assert_eq!(round_up_to(16384, 16384), Some(16384));
        // Overflow must yield None, never wrap to a bogus small value.
        assert_eq!(round_up_to(usize::MAX, 4096), None);
        assert_eq!(round_up_to(usize::MAX - 10, 4096), None);
    }

    #[test]
    fn test_effective_layout_first_touch_unchanged() {
        let layout = Layout::from_size_align(100, 8).expect("valid layout");
        let eff = effective_layout(layout, &NumaAllocHint::first_touch()).expect("some layout");
        assert_eq!(eff.size(), 100);
        assert_eq!(eff.align(), 8);
    }

    #[test]
    fn test_effective_layout_bind_is_page_aligned() {
        let page = get_page_size();
        let layout = Layout::from_size_align(100, 8).expect("valid layout");
        let eff = effective_layout(layout, &NumaAllocHint::bind_node(0)).expect("some layout");
        assert!(eff.align() >= page);
        assert_eq!(eff.size() % page, 0);
        assert!(eff.size() >= 100);

        // Zero-size stays zero-size even for a binding strategy.
        let zero = Layout::from_size_align(0, 8).expect("valid layout");
        let eff_zero = effective_layout(zero, &NumaAllocHint::bind_node(0)).expect("some layout");
        assert_eq!(eff_zero.size(), 0);
    }

    // ----- zero-size / ZST handling (finding: ZST alloc UB) ------------------

    #[test]
    fn test_numa_alloc_zero_size_layout_is_dangling_not_ub() {
        // A zero-size layout must NOT reach the global allocator (UB); it must
        // return a valid, aligned, non-null dangling pointer instead.
        let layout = Layout::from_size_align(0, 16).expect("valid layout");
        let hint = NumaAllocHint::first_touch();
        let ptr = unsafe { numa_alloc(layout, &hint) }.expect("zero-size alloc -> dangling ptr");
        assert_eq!(ptr.as_ptr() as usize % 16, 0);
        // Deallocation of a dangling ZST pointer must be a no-op, not a crash.
        unsafe { numa_dealloc(ptr, layout, &hint) };

        // Same for the zeroed variant.
        let ptr2 = unsafe { numa_alloc_zeroed(layout, &hint) }.expect("zero-size zeroed -> ptr");
        assert_eq!(ptr2.as_ptr() as usize % 16, 0);
        unsafe { numa_dealloc(ptr2, layout, &hint) };
    }

    #[test]
    fn test_numa_allocator_zst_count() {
        // A zero-sized T with non-zero count yields a zero-size layout: must not
        // hit the allocator, must round-trip cleanly.
        let alloc: NumaAllocator<()> = NumaAllocator::first_touch();
        let ptr = alloc.allocate(8).expect("zst alloc -> dangling ptr");
        unsafe { alloc.deallocate(ptr, 8) };
    }

    #[test]
    fn test_numa_vec_zst() {
        let mut v: NumaVec<()> = NumaVec::new();
        for _ in 0..10 {
            v.push(()).expect("push zst");
        }
        assert_eq!(v.len(), 10);
        for _ in 0..10 {
            assert_eq!(v.pop(), Some(()));
        }
        assert!(v.is_empty());
    }

    // ----- bind_memory_policy error propagation (finding: swallowed errno) ---

    #[test]
    fn test_bind_memory_policy_first_touch_is_noop_ok() {
        let page = get_page_size();
        let layout = Layout::from_size_align(page, page).expect("valid layout");
        let hint = NumaAllocHint::first_touch();
        let ptr = unsafe { numa_alloc(layout, &hint) }.expect("alloc");
        let res = unsafe { bind_memory_policy(ptr.as_ptr(), layout.size(), &hint) };
        assert!(res.is_ok(), "first-touch binding must be a successful no-op");
        unsafe { numa_dealloc(ptr, layout, &hint) };
    }

    #[test]
    fn test_numa_alloc_bind_roundtrip_memory_valid() {
        // The page-aligned binding path must yield fully usable memory whether
        // or not the kernel actually applied the policy, and it must be
        // page-aligned.
        let count = 1000usize;
        let alloc: NumaAllocator<f64> = NumaAllocator::on_node(0);
        let ptr = alloc.allocate(count).expect("alloc");
        assert_eq!(ptr.as_ptr() as usize % get_page_size(), 0);
        unsafe {
            for i in 0..count {
                core::ptr::write(ptr.as_ptr().add(i), i as f64);
            }
            for i in 0..count {
                assert!((core::ptr::read(ptr.as_ptr().add(i)) - i as f64).abs() < f64::EPSILON);
            }
            alloc.deallocate(ptr, count);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_bind_memory_policy_rejects_invalid_node() {
        // Binding to a node id beyond the supported maximum must return an
        // error (never silently "succeed" doing nothing).
        let page = get_page_size();
        let layout = Layout::from_size_align(page, page).expect("valid layout");
        let ft = NumaAllocHint::first_touch();
        let ptr = unsafe { numa_alloc(layout, &ft) }.expect("alloc");
        let hint = NumaAllocHint::bind_node(9999);
        let res = unsafe { bind_memory_policy(ptr.as_ptr(), layout.size(), &hint) };
        assert!(res.is_err(), "binding to an invalid node must error");
        unsafe { numa_dealloc(ptr, layout, &ft) };
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_bind_memory_policy_rejects_unaligned_address() {
        // A non-page-aligned start address must be rejected with an error
        // rather than issuing an mbind that the kernel would EINVAL silently.
        let page = get_page_size();
        let layout = Layout::from_size_align(page * 2, page).expect("valid layout");
        let ft = NumaAllocHint::first_touch();
        let ptr = unsafe { numa_alloc(layout, &ft) }.expect("alloc");
        let unaligned = unsafe { ptr.as_ptr().add(1) };
        let hint = NumaAllocHint::bind_node(0);
        let res = unsafe { bind_memory_policy(unaligned, page, &hint) };
        assert!(res.is_err(), "unaligned address must be rejected");
        unsafe { numa_dealloc(ptr, layout, &ft) };
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_node_list() {
        assert_eq!(parse_node_list("0"), vec![0]);
        assert_eq!(parse_node_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_node_list("0,2-3"), vec![0, 2, 3]);
        assert_eq!(parse_node_list("0-1\n"), vec![0, 1]);
        assert_eq!(parse_node_list(" 1 , 4 "), vec![1, 4]);
        assert!(parse_node_list("").is_empty());
        assert!(parse_node_list("garbage").is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_build_node_mask() {
        let bits_per_word = core::mem::size_of::<libc::c_ulong>() * 8;
        let one: libc::c_ulong = 1;

        let (mask, maxnode) = build_node_mask(&[0]);
        assert_eq!(mask, vec![1]);
        assert_eq!(maxnode, 2);

        let (mask, maxnode) = build_node_mask(&[0, 1]);
        assert_eq!(mask, vec![0b11]);
        assert_eq!(maxnode, 3);

        // Highest bit inside the first word.
        let top = bits_per_word - 1;
        let (mask, maxnode) = build_node_mask(&[top]);
        assert_eq!(mask.len(), 1);
        assert_eq!(mask[0], one << top);
        assert_eq!(maxnode, top + 2);

        // A node that lands in the second word.
        let (mask, maxnode) = build_node_mask(&[bits_per_word]);
        assert_eq!(mask.len(), 2);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[1], 1);
        assert_eq!(maxnode, bits_per_word + 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_validate_numa_node_bounds() {
        // Beyond the supported maximum is always rejected.
        assert!(validate_numa_node(MAX_NUMA_NODES).is_err());
        assert!(validate_numa_node(MAX_NUMA_NODES + 1).is_err());
    }
}
