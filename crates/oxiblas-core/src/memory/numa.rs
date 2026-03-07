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

    // SYS_mbind syscall number on x86_64
    const SYS_MBIND: libc::c_long = 237;

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
        // Use syscall directly to avoid libnuma dependency
        let _ = libc::syscall(
            SYS_MBIND,
            ptr as *mut libc::c_void,
            size,
            mode,
            &node_mask as *const usize,
            max_node,
            0u32,
        );
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
            unsafe { dealloc(ptr.cast::<u8>().as_ptr(), layout) };
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
            // Free old allocation.
            if let Ok(old_layout) = Layout::array::<T>(self.cap) {
                unsafe { dealloc(self.ptr.cast::<u8>().as_ptr(), old_layout) };
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
        // Deallocate backing store.
        if let Ok(layout) = Layout::array::<T>(self.cap) {
            unsafe { dealloc(self.ptr.cast::<u8>().as_ptr(), layout) };
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
}
