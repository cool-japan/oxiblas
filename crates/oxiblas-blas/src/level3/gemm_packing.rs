//! Optimized packing strategies for GEMM.
//!
//! This module provides enhanced packing functions with:
//! - Cache-line aware packing
//! - Software prefetching hints in [`pack_a_optimized`], issued on x86_64
//!   only (elsewhere the hint is a no-op and the hardware prefetchers follow
//!   the column streams)
//! - Optimized loop unrolling
//! - Streaming-friendly memory access patterns
//!
//! ## Packing Layout
//!
//! For A panel (MR × KC blocks):
//! ```text
//! A_packed = [ A00 A01 ... A0(KC-1) ]  <- MR elements per column
//!            [ A10 A11 ... A1(KC-1) ]
//!            [ ... ]
//! ```
//!
//! For B panel (KC × NR blocks):
//! ```text
//! B_packed = [ B00 B01 ... B0(NR-1) ]  <- NR elements per row
//!            [ B10 B11 ... B1(NR-1) ]
//!            [ ... ]
//! ```

use crate::level3::gemm_kernel::GemmKernel;
use core::cell::Cell;
use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

// =============================================================================
// Per-thread packing buffers
// =============================================================================

/// Which packing buffer a per-thread cache slot holds.
///
/// Every element type with a micro-kernel in this crate (`f32`, `f64`,
/// `Complex32`, `Complex64`) keeps one cached `A~` and one cached `B~` buffer
/// per thread, so back-to-back GEMMs on a thread reuse their packing memory
/// instead of allocating and zero-filling it on every call. The parallel
/// driver takes `B` on the calling thread (the shared panel) and `A` on each
/// worker (one per tile task).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackSlot {
    /// The packed `op(A)` macro-panel (`MC × KC`).
    A = 0,
    /// The packed `op(B)` macro-panel (`KC × NC`).
    B = 1,
}

/// Most packing-buffer bytes one thread keeps cached between GEMM calls, over
/// all element types and both slots together.
///
/// Policy:
/// - A buffer is moved *out of* its slot for the whole call (`Cell::take`) and
///   put back when the call returns, so no borrow is held while the GEMM runs.
///   A nested GEMM on the same thread (rayon work stealing while the outer one
///   waits in a join) finds the slot empty and allocates its own buffer.
/// - A cached buffer only grows. Growing frees the old allocation and
///   allocates a zero-filled one, so every element handed out is initialized;
///   a reused buffer is not cleared again (the packers overwrite every element
///   they hand to the micro-kernel, including the zero padding of edge panels).
/// - When putting a buffer back would raise the thread's cached total above
///   this limit, the buffer is freed instead. A single large GEMM therefore
///   never pins more than this much memory per thread after it returns (at
///   most 8 MiB × the number of threads that ran GEMMs).
///
/// This is the default of each thread's limit. The drivers read the limit in
/// force through `pack_cache_limit`. Only in-crate tests (including the A/B
/// harness) set other values per thread, with a setter that exists only in
/// test builds; nothing else changes it.
pub const PACK_CACHE_MAX_BYTES: usize = 8 << 20;

thread_local! {
    /// Bytes currently parked in this thread's packing-buffer slots.
    static PACK_CACHE_BYTES: Cell<usize> = const { Cell::new(0) };
    /// This thread's cache limit in bytes: [`PACK_CACHE_MAX_BYTES`] unless a
    /// test set another one.
    static PACK_CACHE_LIMIT: Cell<usize> = const { Cell::new(PACK_CACHE_MAX_BYTES) };
}

#[cfg(test)]
thread_local! {
    /// Number of buffers [`PackBuffer::take`] allocated on this thread (test
    /// hook; a loan served from the cache does not count).
    static PACK_CACHE_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// The most packing-buffer bytes this thread keeps cached between GEMM calls:
/// [`PACK_CACHE_MAX_BYTES`] unless a test changed it for this thread.
///
/// The issue #23 B-block cap (`autotune::capped_nc`) reads the same value, so
/// the room it leaves beside the A buffer always matches the limit the cache
/// enforces.
#[inline]
pub(crate) fn pack_cache_limit() -> usize {
    PACK_CACHE_LIMIT
        .try_with(Cell::get)
        .unwrap_or(PACK_CACHE_MAX_BYTES)
}

/// Restores a thread's previous packing-cache limit when dropped (see
/// [`set_pack_cache_limit`]).
///
/// The guard is neither `Send` nor `Sync`: it restores the limit of the
/// thread that drops it, so it must be dropped on the thread that created it.
#[cfg(test)]
#[must_use = "the previous limit is restored when the guard drops"]
pub(crate) struct PackCacheLimitGuard {
    previous: usize,
    /// Keeps the guard on its thread (`*const ()` is neither `Send` nor
    /// `Sync`).
    _not_send: core::marker::PhantomData<*const ()>,
}

#[cfg(test)]
impl Drop for PackCacheLimitGuard {
    fn drop(&mut self) {
        let _ = PACK_CACHE_LIMIT.try_with(|limit| limit.set(self.previous));
    }
}

/// Sets the calling thread's packing-cache limit to `bytes` until the
/// returned guard drops (test and A/B hook).
///
/// The limit is per thread: to change it for the workers of a rayon pool,
/// call this on every worker (for example with `ThreadPool::broadcast`) and
/// drop the guards on those same workers when the run is over (the guard is
/// not `Send`, so it cannot be handed back to the caller). Lowering the
/// limit evicts nothing at once; a buffer parked above the new limit is freed
/// the next time it is put back.
#[cfg(test)]
pub(crate) fn set_pack_cache_limit(bytes: usize) -> PackCacheLimitGuard {
    let previous = PACK_CACHE_LIMIT
        .try_with(|limit| limit.replace(bytes))
        .unwrap_or(PACK_CACHE_MAX_BYTES);
    PackCacheLimitGuard {
        previous,
        _not_send: core::marker::PhantomData,
    }
}

/// Heap bytes owned by `buf`.
#[inline]
fn pack_bytes<T>(buf: &AlignedVec<T>) -> usize {
    buf.capacity() * core::mem::size_of::<T>()
}

/// Moves this thread's cached buffer for `slot` out of the cache (an empty
/// buffer if there is none).
fn take_cached<T: GemmKernel>(slot: PackSlot) -> AlignedVec<T> {
    let buf = T::with_pack_cache(slot, Cell::take).unwrap_or_default();
    let bytes = pack_bytes(&buf);
    if bytes > 0 {
        let _ = PACK_CACHE_BYTES.try_with(|parked| parked.set(parked.get().saturating_sub(bytes)));
    }
    buf
}

/// Returns `buf` to this thread's cache slot, or frees it when the type keeps
/// no cache or the size policy of [`PACK_CACHE_MAX_BYTES`] (with this
/// thread's [`pack_cache_limit`]) does not allow it.
fn put_cached<T: GemmKernel>(slot: PackSlot, buf: AlignedVec<T>) {
    let bytes = pack_bytes(&buf);
    if bytes == 0 {
        return;
    }
    let Ok(parked) = PACK_CACHE_BYTES.try_with(Cell::get) else {
        return;
    };
    if parked.saturating_add(bytes) > pack_cache_limit() {
        return;
    }
    let mut stored = false;
    let previous = T::with_pack_cache(slot, |cell| {
        stored = true;
        cell.replace(buf)
    });
    if stored {
        // `previous` is normally empty; a nested GEMM on this thread may have
        // parked a buffer meanwhile, which `buf` replaces (and which is freed
        // when `previous` drops here).
        let freed = previous.as_ref().map_or(0, pack_bytes);
        let _ = PACK_CACHE_BYTES.try_with(|total| {
            total.set(total.get().saturating_add(bytes).saturating_sub(freed));
        });
    }
}

/// A packing buffer on loan from the calling thread's cache.
///
/// Dereferences to an [`AlignedVec`] of at least the requested length whose
/// elements are all initialized. Dropping the loan returns the buffer to the
/// cache (see [`PACK_CACHE_MAX_BYTES`] for the policy).
pub(crate) struct PackBuffer<T: GemmKernel> {
    buf: AlignedVec<T>,
    slot: PackSlot,
}

impl<T: GemmKernel + bytemuck::Zeroable> PackBuffer<T> {
    /// Borrows this thread's buffer for `slot`, growing it to at least `len`
    /// elements.
    pub(crate) fn take(slot: PackSlot, len: usize) -> Self {
        let mut buf = take_cached::<T>(slot);
        if buf.len() < len {
            // Free the old allocation before making the zero-filled one, so a
            // grow never holds both.
            drop(core::mem::take(&mut buf));
            buf = AlignedVec::zeros(len);
            #[cfg(test)]
            let _ = PACK_CACHE_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
        Self { buf, slot }
    }
}

impl<T: GemmKernel> core::ops::Deref for PackBuffer<T> {
    type Target = AlignedVec<T>;
    #[inline]
    fn deref(&self) -> &AlignedVec<T> {
        &self.buf
    }
}

impl<T: GemmKernel> core::ops::DerefMut for PackBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut AlignedVec<T> {
        &mut self.buf
    }
}

impl<T: GemmKernel> Drop for PackBuffer<T> {
    fn drop(&mut self) {
        put_cached(self.slot, core::mem::take(&mut self.buf));
    }
}

/// Bytes cached in this thread's packing slots (test hook).
#[cfg(test)]
pub(crate) fn pack_cache_bytes() -> usize {
    PACK_CACHE_BYTES.try_with(Cell::get).unwrap_or(0)
}

/// Number of packing buffers this thread has allocated so far (test hook; a
/// loan served from the cache allocates nothing).
#[cfg(test)]
pub(crate) fn pack_cache_allocations() -> usize {
    PACK_CACHE_ALLOCATIONS.try_with(Cell::get).unwrap_or(0)
}

/// Capacity, in elements, of this thread's cached `T` buffer for `slot`
/// (test hook; 0 when the slot is empty or the type keeps no cache).
#[cfg(test)]
pub(crate) fn pack_cache_capacity<T: GemmKernel>(slot: PackSlot) -> usize {
    T::with_pack_cache(slot, |cell| {
        let buf = cell.take();
        let cap = buf.capacity();
        cell.set(buf);
        cap
    })
    .unwrap_or(0)
}

/// How far ahead [`pack_a_optimized`] prefetches, in source columns (one
/// prefetch every four columns), and the default of
/// [`PackingConfig::prefetch_distance`].
///
/// Only x86_64 issues the prefetches: [`prefetch_read`] is a no-op on
/// aarch64, where the hardware prefetchers track the column streams, so the
/// aarch64 value only feeds `PackingConfig`.
#[cfg(target_arch = "aarch64")]
const PREFETCH_DISTANCE: usize = 6; // source columns ahead

#[cfg(not(target_arch = "aarch64"))]
const PREFETCH_DISTANCE: usize = 4; // source columns ahead

/// Returns whether rows `start..start + count` fit in `0..len`.
#[inline]
fn range_fits(start: usize, count: usize, len: usize) -> bool {
    start <= len && count <= len - start
}

/// Packs a panel of A into `MR`-row micro-panels.
///
/// Rows `row_start..row_start + nrows` and columns `col_start..col_start +
/// ncols` of `a` become `⌈nrows / mr⌉` micro-panels of `mr × ncols` elements:
/// micro-panel `b` holds, column after column, the `mr` elements
/// `a[row_start + b·mr + ii, col_start + p]` (`ii < mr`), zero-padded past
/// `nrows`. That is the layout the micro-kernels read; every element of the
/// `⌈nrows / mr⌉ · mr · ncols` prefix of `pack` is written.
///
/// For plain column-major storage (`a.col_stride() == 1`) each column of a
/// micro-panel is `mr` contiguous elements, copied with a compile-time `mr`
/// for every micro-kernel height in this crate (2, 4, 6, 8, 16) so the copy is
/// a few vector loads and stores rather than a `memcpy` call; other `mr` take
/// a generic loop. Views whose rows are not adjacent in memory (for example a
/// [`MatRef::transpose`] view) are read element by element, with the same
/// layout.
///
/// # Arguments
///
/// * `a` - Source matrix
/// * `row_start` - Starting row in A
/// * `col_start` - Starting column in A
/// * `nrows` - Number of rows to pack
/// * `ncols` - Number of columns to pack
/// * `pack` - Destination packed buffer
/// * `mr` - Micro-kernel row block size
///
/// # Panics
///
/// Panics if `mr == 0`, if the region lies outside `a`, or if `pack` holds
/// fewer than `⌈nrows / mr⌉ · mr · ncols` elements.
#[inline]
pub fn pack_a_optimized<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    mr: usize,
) {
    assert!(mr > 0, "pack_a_optimized: mr must be positive");
    assert!(
        range_fits(row_start, nrows, a.nrows()) && range_fits(col_start, ncols, a.ncols()),
        "pack_a_optimized: region outside the matrix"
    );
    let len = nrows.div_ceil(mr) * mr * ncols;
    assert!(pack.len() >= len, "pack_a_optimized: buffer too small");
    if len == 0 {
        return;
    }
    if a.col_stride() != 1 {
        pack_a_strided(
            a,
            row_start,
            col_start,
            nrows,
            ncols,
            &mut pack.as_mut_slice()[..len],
            mr,
        );
        return;
    }

    let ld = a.row_stride();
    let dst = pack.as_mut_ptr();
    // SAFETY: with unit row stride, element `(row_start + i, col_start + p)` of
    // `a` lives at `row_start + i + (col_start + p) * ld` from `a.as_ptr()`, and
    // the asserts above keep every `i < nrows`, `p < ncols` inside `a`; `src`
    // is that element for `i = p = 0` (in bounds because `len > 0`). The
    // packers write exactly the first `len <= pack.len()` elements of `dst`,
    // a buffer distinct from `a`.
    unsafe {
        let src = a.as_ptr().add(row_start + col_start * ld);
        match mr {
            2 => pack_a_panels::<T, 2>(src, ld, nrows, ncols, dst),
            4 => pack_a_panels::<T, 4>(src, ld, nrows, ncols, dst),
            6 => pack_a_panels::<T, 6>(src, ld, nrows, ncols, dst),
            8 => pack_a_panels::<T, 8>(src, ld, nrows, ncols, dst),
            16 => pack_a_panels::<T, 16>(src, ld, nrows, ncols, dst),
            _ => pack_a_panels_dyn(src, ld, nrows, ncols, dst, mr),
        }
    }
}

/// [`pack_a_optimized`] for a compile-time micro-panel height `MR`.
///
/// # Safety
///
/// `src` points at the top-left element of an `nrows × ncols` block with unit
/// row stride and leading dimension `ld`, readable throughout, and `dst` is
/// writable for `⌈nrows / MR⌉ · MR · ncols` elements and does not overlap it.
#[inline(always)]
unsafe fn pack_a_panels<T: Field, const MR: usize>(
    src: *const T,
    ld: usize,
    nrows: usize,
    ncols: usize,
    dst: *mut T,
) {
    let full = nrows / MR;
    for blk in 0..full {
        let s = src.add(blk * MR);
        let d = dst.add(blk * MR * ncols);
        for p in 0..ncols {
            if p.is_multiple_of(4) && p + PREFETCH_DISTANCE < ncols {
                prefetch_read(s.add((p + PREFETCH_DISTANCE) * ld));
            }
            // Compile-time length: a few vector moves, no `memcpy` call.
            core::ptr::copy_nonoverlapping(s.add(p * ld), d.add(p * MR), MR);
        }
    }
    let rem = nrows - full * MR;
    if rem > 0 {
        let s = src.add(full * MR);
        let d = dst.add(full * MR * ncols);
        for p in 0..ncols {
            let col = s.add(p * ld);
            let out = d.add(p * MR);
            for ii in 0..rem {
                *out.add(ii) = *col.add(ii);
            }
            for ii in rem..MR {
                *out.add(ii) = T::zero();
            }
        }
    }
}

/// [`pack_a_optimized`] for a run-time micro-panel height.
///
/// # Safety
///
/// As for [`pack_a_panels`] with `MR = mr`.
unsafe fn pack_a_panels_dyn<T: Field>(
    src: *const T,
    ld: usize,
    nrows: usize,
    ncols: usize,
    dst: *mut T,
    mr: usize,
) {
    let mut idx = 0;
    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);
        for p in 0..ncols {
            let col = src.add(i + p * ld);
            core::ptr::copy_nonoverlapping(col, dst.add(idx), ib);
            for ii in ib..mr {
                *dst.add(idx + ii) = T::zero();
            }
            idx += mr;
        }
    }
}

/// Element-wise [`pack_a_optimized`] for views whose rows are not adjacent in
/// memory; `out` is exactly the packed prefix.
fn pack_a_strided<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    out: &mut [T],
    mr: usize,
) {
    let mut chunks = out.chunks_exact_mut(mr);
    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);
        for p in 0..ncols {
            let Some(chunk) = chunks.next() else {
                return;
            };
            for (ii, slot) in chunk.iter_mut().enumerate() {
                *slot = if ii < ib {
                    a[(row_start + i + ii, col_start + p)]
                } else {
                    T::zero()
                };
            }
        }
    }
}

/// Packs a panel of B into `NR`-column micro-panels.
///
/// Rows `row_start..row_start + nrows` and columns `col_start..col_start +
/// ncols` of `b` become `⌈ncols / nr⌉` micro-panels of `nrows × nr` elements,
/// row-major inside a micro-panel: micro-panel `b` holds, row after row, the
/// `nr` elements `b[row_start + p, col_start + b·nr + jj]` (`jj < nr`),
/// zero-padded past `ncols`. Every element of the `⌈ncols / nr⌉ · nr · nrows`
/// prefix of `pack` is written.
///
/// For plain column-major storage (`b.col_stride() == 1`) the gather runs with
/// a compile-time `nr` for every micro-kernel width in this crate (2, 4, 6, 8,
/// 16); on aarch64 the f64 `nr = 6` and f32 `nr = 8` panels are transposed in
/// NEON registers. Other `nr` take a generic loop, and views whose rows are not
/// adjacent in memory are read element by element, all with the same layout.
///
/// # Arguments
///
/// * `b` - Source matrix
/// * `row_start` - Starting row in B
/// * `col_start` - Starting column in B
/// * `nrows` - Number of rows to pack
/// * `ncols` - Number of columns to pack
/// * `pack` - Destination packed buffer
/// * `nr` - Micro-kernel column block size
///
/// # Panics
///
/// Panics if `nr == 0`, if the region lies outside `b`, or if `pack` holds
/// fewer than `⌈ncols / nr⌉ · nr · nrows` elements.
#[inline]
pub fn pack_b_optimized<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    assert!(nr > 0, "pack_b_optimized: nr must be positive");
    assert!(
        range_fits(row_start, nrows, b.nrows()) && range_fits(col_start, ncols, b.ncols()),
        "pack_b_optimized: region outside the matrix"
    );
    let len = ncols.div_ceil(nr) * nr * nrows;
    assert!(pack.len() >= len, "pack_b_optimized: buffer too small");
    if len == 0 {
        return;
    }
    if b.col_stride() != 1 {
        pack_b_strided(
            b,
            row_start,
            col_start,
            nrows,
            ncols,
            &mut pack.as_mut_slice()[..len],
            nr,
        );
        return;
    }

    let ld = b.row_stride();
    let dst = pack.as_mut_ptr();
    // SAFETY: as in `pack_a_optimized`: unit row stride puts element
    // `(row_start + p, col_start + j)` at `row_start + p + (col_start + j) * ld`,
    // inside `b` for every `p < nrows`, `j < ncols` by the asserts; exactly the
    // first `len <= pack.len()` elements of the distinct buffer `dst` are
    // written.
    unsafe {
        let src = b.as_ptr().add(row_start + col_start * ld);
        match nr {
            2 => pack_b_panels::<T, 2>(src, ld, nrows, ncols, dst),
            4 => pack_b_panels::<T, 4>(src, ld, nrows, ncols, dst),
            6 => pack_b_panels::<T, 6>(src, ld, nrows, ncols, dst),
            8 => pack_b_panels::<T, 8>(src, ld, nrows, ncols, dst),
            16 => pack_b_panels::<T, 16>(src, ld, nrows, ncols, dst),
            _ => pack_b_panels_dyn(src, ld, nrows, ncols, dst, nr),
        }
    }
}

/// [`pack_b_optimized`] for a compile-time micro-panel width `NR`.
///
/// # Safety
///
/// `src` points at the top-left element of an `nrows × ncols` block with unit
/// row stride and leading dimension `ld`, readable throughout, and `dst` is
/// writable for `⌈ncols / NR⌉ · NR · nrows` elements and does not overlap it.
#[inline(always)]
unsafe fn pack_b_panels<T: Field, const NR: usize>(
    src: *const T,
    ld: usize,
    nrows: usize,
    ncols: usize,
    dst: *mut T,
) {
    let full = ncols / NR;
    for blk in 0..full {
        pack_b_full_panel::<T, NR>(src.add(blk * NR * ld), ld, nrows, dst.add(blk * NR * nrows));
    }
    let rem = ncols - full * NR;
    if rem > 0 {
        let s = src.add(full * NR * ld);
        let d = dst.add(full * NR * nrows);
        for p in 0..nrows {
            let out = d.add(p * NR);
            for jj in 0..rem {
                *out.add(jj) = *s.add(p + jj * ld);
            }
            for jj in rem..NR {
                *out.add(jj) = T::zero();
            }
        }
    }
}

/// Packs one full `nrows × NR` micro-panel of B.
///
/// # Safety
///
/// `s` points at `NR` columns of `nrows` contiguous elements each, `ld`
/// elements apart, and `d` is writable for `NR · nrows` elements that do not
/// overlap them.
#[inline(always)]
unsafe fn pack_b_full_panel<T: Field, const NR: usize>(
    s: *const T,
    ld: usize,
    nrows: usize,
    d: *mut T,
) {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        use core::any::TypeId;
        if NR == 6 && TypeId::of::<T>() == TypeId::of::<f64>() {
            // `T` is `f64`, so the casts only restate the element type.
            pack_b_panel_f64x6_neon(s.cast::<f64>(), ld, nrows, d.cast::<f64>());
            return;
        }
        if NR == 8 && TypeId::of::<T>() == TypeId::of::<f32>() {
            // `T` is `f32`, as above.
            pack_b_panel_f32x8_neon(s.cast::<f32>(), ld, nrows, d.cast::<f32>());
            return;
        }
    }
    let cols: [*const T; NR] = core::array::from_fn(|jj| s.wrapping_add(jj * ld));
    for p in 0..nrows {
        let out = d.add(p * NR);
        for (jj, col) in cols.iter().enumerate() {
            *out.add(jj) = *col.add(p);
        }
    }
}

/// Packs one full `nrows × 6` f64 micro-panel of B, two rows at a time:
/// each column contributes a 2-row vector and `zip1`/`zip2` turn three column
/// pairs into the two packed rows.
///
/// # Safety
///
/// As for [`pack_b_full_panel`] with `NR = 6`.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[inline(always)]
unsafe fn pack_b_panel_f64x6_neon(s: *const f64, ld: usize, nrows: usize, d: *mut f64) {
    use core::arch::aarch64::{vld1q_f64, vst1q_f64, vzip1q_f64, vzip2q_f64};
    let c: [*const f64; 6] = core::array::from_fn(|jj| s.wrapping_add(jj * ld));
    let mut p = 0;
    while p + 2 <= nrows {
        let v0 = vld1q_f64(c[0].add(p));
        let v1 = vld1q_f64(c[1].add(p));
        let v2 = vld1q_f64(c[2].add(p));
        let v3 = vld1q_f64(c[3].add(p));
        let v4 = vld1q_f64(c[4].add(p));
        let v5 = vld1q_f64(c[5].add(p));
        let out = d.add(p * 6);
        vst1q_f64(out, vzip1q_f64(v0, v1));
        vst1q_f64(out.add(2), vzip1q_f64(v2, v3));
        vst1q_f64(out.add(4), vzip1q_f64(v4, v5));
        vst1q_f64(out.add(6), vzip2q_f64(v0, v1));
        vst1q_f64(out.add(8), vzip2q_f64(v2, v3));
        vst1q_f64(out.add(10), vzip2q_f64(v4, v5));
        p += 2;
    }
    if p < nrows {
        let out = d.add(p * 6);
        for (jj, col) in c.iter().enumerate() {
            *out.add(jj) = *col.add(p);
        }
    }
}

/// Packs one full `nrows × 8` f32 micro-panel of B, four rows at a time: each
/// column contributes a 4-row vector, and two 4 × 4 register transposes turn
/// the eight columns into four packed rows.
///
/// # Safety
///
/// As for [`pack_b_full_panel`] with `NR = 8`.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[inline(always)]
unsafe fn pack_b_panel_f32x8_neon(s: *const f32, ld: usize, nrows: usize, d: *mut f32) {
    use core::arch::aarch64::{
        float32x4_t, vld1q_f32, vreinterpretq_f32_f64, vreinterpretq_f64_f32, vst1q_f32,
        vtrn1q_f32, vtrn1q_f64, vtrn2q_f32, vtrn2q_f64,
    };

    /// Transposes the 4 × 4 block whose columns are `a`, `b`, `c`, `e` into
    /// its four rows.
    #[inline(always)]
    unsafe fn transpose4(
        a: float32x4_t,
        b: float32x4_t,
        c: float32x4_t,
        e: float32x4_t,
    ) -> [float32x4_t; 4] {
        let t0 = vreinterpretq_f64_f32(vtrn1q_f32(a, b)); // a0 b0 a2 b2
        let t1 = vreinterpretq_f64_f32(vtrn2q_f32(a, b)); // a1 b1 a3 b3
        let t2 = vreinterpretq_f64_f32(vtrn1q_f32(c, e)); // c0 e0 c2 e2
        let t3 = vreinterpretq_f64_f32(vtrn2q_f32(c, e)); // c1 e1 c3 e3
        [
            vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)), // a0 b0 c0 e0
            vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)), // a1 b1 c1 e1
            vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)), // a2 b2 c2 e2
            vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)), // a3 b3 c3 e3
        ]
    }

    let c: [*const f32; 8] = core::array::from_fn(|jj| s.wrapping_add(jj * ld));
    let mut p = 0;
    while p + 4 <= nrows {
        let lo = transpose4(
            vld1q_f32(c[0].add(p)),
            vld1q_f32(c[1].add(p)),
            vld1q_f32(c[2].add(p)),
            vld1q_f32(c[3].add(p)),
        );
        let hi = transpose4(
            vld1q_f32(c[4].add(p)),
            vld1q_f32(c[5].add(p)),
            vld1q_f32(c[6].add(p)),
            vld1q_f32(c[7].add(p)),
        );
        let out = d.add(p * 8);
        for (r, (&left, &right)) in lo.iter().zip(hi.iter()).enumerate() {
            vst1q_f32(out.add(r * 8), left);
            vst1q_f32(out.add(r * 8 + 4), right);
        }
        p += 4;
    }
    while p < nrows {
        let out = d.add(p * 8);
        for (jj, col) in c.iter().enumerate() {
            *out.add(jj) = *col.add(p);
        }
        p += 1;
    }
}

/// [`pack_b_optimized`] for a run-time micro-panel width.
///
/// # Safety
///
/// As for [`pack_b_panels`] with `NR = nr`.
unsafe fn pack_b_panels_dyn<T: Field>(
    src: *const T,
    ld: usize,
    nrows: usize,
    ncols: usize,
    dst: *mut T,
    nr: usize,
) {
    let mut idx = 0;
    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);
        let s = src.add(j * ld);
        for p in 0..nrows {
            for jj in 0..jb {
                *dst.add(idx + jj) = *s.add(p + jj * ld);
            }
            for jj in jb..nr {
                *dst.add(idx + jj) = T::zero();
            }
            idx += nr;
        }
    }
}

/// Element-wise [`pack_b_optimized`] for views whose rows are not adjacent in
/// memory; `out` is exactly the packed prefix.
fn pack_b_strided<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    out: &mut [T],
    nr: usize,
) {
    let mut chunks = out.chunks_exact_mut(nr);
    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);
        for p in 0..nrows {
            let Some(chunk) = chunks.next() else {
                return;
            };
            for (jj, slot) in chunk.iter_mut().enumerate() {
                *slot = if jj < jb {
                    b[(row_start + p, col_start + j + jj)]
                } else {
                    T::zero()
                };
            }
        }
    }
}

/// Packs an A panel, taking a flat-copy fast path when the source is a single
/// fully-contiguous micro-panel.
///
/// This crate stores matrices in column-major order: element `(i, j)` lives at
/// `ptr + i + j * row_stride`, so consecutive rows within a column are always
/// one element apart and `row_stride` is the leading dimension. A matrix is
/// therefore *fully contiguous* (no padding between columns) exactly when
/// `row_stride == nrows` — not `row_stride == ncols`, which the previous check
/// used and which is wrong for any non-square matrix.
///
/// [`pack_a_optimized`] re-tiles the panel into `mr`-row sub-panels, so its
/// packed order coincides with plain column-major order **only** when the whole
/// panel is a single `mr`-row block (`nrows == mr`). For a taller panel a flat
/// `memcpy` would emit the wrong layout, so the fast path additionally requires
/// `nrows == mr`; every other shape delegates to the general path.
#[inline]
pub fn pack_a_contiguous<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    mr: usize,
) {
    // Column-major contiguity: the leading dimension equals the row count only
    // when columns sit back-to-back with no inter-column padding.
    let row_stride = a.row_stride();
    let is_contiguous = row_stride == a.nrows() && a.col_stride() == 1;

    if is_contiguous
        && row_start == 0
        && col_start == 0
        && nrows == a.nrows()
        && ncols == a.ncols()
        && nrows == mr
    {
        // The whole matrix is contiguous AND a single micro-row-block, so the
        // packed layout is exactly column-major and one bulk copy reproduces it.
        assert!(
            pack.len() >= nrows * ncols,
            "pack_a_contiguous: buffer too small"
        );
        let src = a.as_ptr();
        let dst = pack.as_mut_ptr();
        // SAFETY: unit row stride and `row_stride == nrows` mean the
        // `nrows * ncols` elements are stored contiguously with no gaps, `src`
        // points at the first of them (row_start == col_start == 0), and `dst`
        // is a distinct buffer of at least that many elements (assert above);
        // the ranges do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, nrows * ncols);
        }
    } else {
        // General path: correct for any stride, padding, or multi-block panel.
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Packs a B panel using non-temporal (streaming) stores when the CPU supports
/// them, falling back to ordinary cache-allocating stores otherwise.
///
/// The packed B buffer is written once here and then read exactly once by the
/// micro-kernel, which makes it a textbook candidate for non-temporal stores:
/// routing these writes through the cache would evict the packed-A panel and the
/// live C tile that the kernel actually reuses. On AVX hardware we assemble each
/// 32-byte lane and emit it with `_mm256_stream_si256` (`VMOVNTDQ`), bypassing
/// the cache hierarchy; an `sfence` before returning makes those non-temporal
/// writes visible to the subsequent (ordinary) reads.
///
/// The store is issued through the integer lane intrinsic so it is agnostic to
/// `T`'s element type — only the raw bytes matter. The produced layout is
/// byte-for-byte identical to [`pack_b_optimized`] on every path: streaming only
/// changes *how* the bytes reach memory, never their values or their order.
///
/// (The stores are 256-bit, which needs only AVX. A 512-bit variant using
/// `_mm512_stream_si512`, stable since Rust 1.89, has not been written.)
#[inline]
pub fn pack_b_streaming<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // `_mm256_stream_si256` needs a 32-byte-aligned destination and moves a
        // full 32-byte lane at a time. `AlignedVec`'s base is aligned to
        // `DEFAULT_ALIGN` (>= 64 on x86_64), but we still verify at runtime so the
        // SAFETY contract holds for any buffer. Requiring `size_of::<T>()` to
        // divide 32 guarantees every lane boundary lands on a 32-byte-aligned
        // element offset.
        let elem_size = core::mem::size_of::<T>();
        // The streaming path addresses B with unit row stride and writes the
        // whole packed prefix unchecked, so its arguments are checked here as
        // `pack_b_optimized` checks its own.
        assert!(nr > 0, "pack_b_streaming: nr must be positive");
        assert!(
            range_fits(row_start, nrows, b.nrows()) && range_fits(col_start, ncols, b.ncols()),
            "pack_b_streaming: region outside the matrix"
        );
        assert!(
            pack.len() >= ncols.div_ceil(nr) * nr * nrows,
            "pack_b_streaming: buffer too small"
        );
        if is_x86_feature_detected!("avx")
            && b.col_stride() == 1
            && elem_size != 0
            && 32 % elem_size == 0
            && (pack.as_ptr() as usize).is_multiple_of(32)
        {
            // SAFETY: AVX is present, the destination base is 32-byte aligned,
            // and the element size divides the 32-byte lane, so every
            // streaming-store target is 32-byte aligned; the asserts above keep
            // every read inside `b` (unit row stride) and every write inside
            // `pack`.
            unsafe {
                pack_b_streaming_avx(b, row_start, col_start, nrows, ncols, pack, nr);
            }
            return;
        }
    }

    // Portable fallback: ordinary temporal stores, identical output.
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

/// AVX (256-bit) non-temporal streaming backend for [`pack_b_streaming`].
///
/// Traverses `B` in the exact linear order [`pack_b_optimized`] writes, buffers
/// a full 32-byte lane, then emits it with `_mm256_stream_si256`. Any trailing
/// partial lane uses ordinary stores; a final `sfence` orders all non-temporal
/// writes before the buffer is read back.
///
/// # Safety
///
/// The caller must guarantee that AVX is available, that `pack`'s base pointer is
/// 32-byte aligned, and that `size_of::<T>()` divides 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn pack_b_streaming_avx<T: Field>(
    b: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<T>,
    nr: usize,
) {
    use std::arch::x86_64::*;

    // Elements per 32-byte non-temporal lane. `size_of::<T>()` divides 32 (caller
    // contract), so this is exact and `1 <= lane <= 32`.
    let lane = 32 / core::mem::size_of::<T>();

    let row_stride = b.row_stride();
    let base_ptr = b.as_ptr();
    let dst = pack.as_mut_ptr();

    // Staging holds one full lane. Sized to the widest possible lane (32, for a
    // hypothetical 1-byte element) so it is always large enough regardless of T;
    // only the first `lane` elements are ever touched per flush.
    let mut staging = [T::zero(); 32];
    let mut buf_len = 0usize; // elements currently staged
    let mut written = 0usize; // elements already streamed to `dst`

    for j in (0..ncols).step_by(nr) {
        let jb = nr.min(ncols - j);
        let col_base = col_start + j;
        for p in 0..nrows {
            let row_base = base_ptr.add(row_start + p + col_base * row_stride);
            for jj in 0..nr {
                // Real element inside the block, zero padding beyond `jb`.
                staging[buf_len] = if jj < jb {
                    *row_base.add(jj * row_stride)
                } else {
                    T::zero()
                };
                buf_len += 1;

                if buf_len == lane {
                    // `written` is a multiple of `lane`, so `written * size_of`
                    // is a multiple of 32 and the store target stays 32-aligned.
                    let lane_vec = _mm256_loadu_si256(staging.as_ptr().cast::<__m256i>());
                    _mm256_stream_si256(dst.add(written).cast::<__m256i>(), lane_vec);
                    written += lane;
                    buf_len = 0;
                }
            }
        }
    }

    // Trailing partial lane (fewer than `lane` elements left): ordinary stores.
    for k in 0..buf_len {
        *dst.add(written + k) = staging[k];
    }

    // Order the non-temporal stores before the packed buffer is read back.
    _mm_sfence();
}

/// Prefetch data for reading.
///
/// Uses architecture-specific prefetch instructions when available.
/// On unsupported platforms, this is a no-op.
#[inline(always)]
#[allow(unused_variables)]
unsafe fn prefetch_read<T>(ptr: *const T) {
    // Use intrinsics when available, otherwise no-op
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    // Note: aarch64 prefetch intrinsics are unstable, so we skip them for now.
    // The compiler and hardware prefetchers generally handle prefetching well
    // on modern ARM processors.
}

/// Packing configuration for different matrix shapes.
///
/// Descriptive only: no packing routine in this crate takes a
/// `PackingConfig`; the packers choose their prefetching, unrolling and
/// streaming themselves.
#[derive(Debug, Clone, Copy)]
pub struct PackingConfig {
    /// Use streaming stores (for large matrices).
    pub use_streaming: bool,
    /// Prefetch distance in source columns of the packed block (not elements
    /// or cache lines), the unit of the fixed distance [`pack_a_optimized`]
    /// uses; the default is that distance.
    pub prefetch_distance: usize,
    /// Whether to use 4-way unrolling.
    pub use_unrolling: bool,
}

impl Default for PackingConfig {
    fn default() -> Self {
        Self {
            use_streaming: false,
            prefetch_distance: PREFETCH_DISTANCE,
            use_unrolling: true,
        }
    }
}

impl PackingConfig {
    /// Create config optimized for large matrices.
    #[must_use]
    pub const fn for_large_matrix() -> Self {
        Self {
            use_streaming: true,
            prefetch_distance: 8,
            use_unrolling: true,
        }
    }

    /// Create config optimized for small matrices.
    #[must_use]
    pub const fn for_small_matrix() -> Self {
        Self {
            use_streaming: false,
            prefetch_distance: 2,
            use_unrolling: false,
        }
    }
}

// =============================================================================
// SIMD-Optimized Packing Functions
// =============================================================================

/// Checks the arguments of the SIMD `pack_a` wrappers as [`pack_a_optimized`]
/// does, so their unchecked fast paths stay in bounds.
///
/// Returns `false` when the rows of `a` are not adjacent in memory (the fast
/// paths assume unit row stride); the wrappers then use `pack_a_optimized`.
///
/// # Panics
///
/// Panics if `mr == 0`, if the region lies outside `a`, or if `pack` is too
/// small for it.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn simd_pack_a_args_ok<T: Field>(
    a: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &AlignedVec<T>,
    mr: usize,
) -> bool {
    assert!(mr > 0, "pack_a: mr must be positive");
    assert!(
        range_fits(row_start, nrows, a.nrows()) && range_fits(col_start, ncols, a.ncols()),
        "pack_a: region outside the matrix"
    );
    assert!(
        pack.len() >= nrows.div_ceil(mr) * mr * ncols,
        "pack_a: buffer too small"
    );
    a.col_stride() == 1
}

/// SIMD-optimized `pack_a` for f64.
///
/// This crate's column-major storage keeps every column contiguous — element
/// `(i, j)` is at `ptr + i + j * row_stride`, so successive rows are one element
/// apart. Packing an `mr`-row block therefore copies `mr` *contiguous* elements
/// per column, which the AVX2 fast path vectorizes for **any** `row_stride`. No
/// special "row_stride == 1" layout is required (and indeed `row_stride == 1`
/// never holds for a genuine multi-row column-major matrix); we only need `mr`
/// wide enough for a vector and AVX2 present at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pack_a_simd_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    // A full AVX2 register holds 4 f64; below that the scalar path is faster.
    if mr >= 4
        && simd_pack_a_args_ok(a, row_start, col_start, nrows, ncols, pack, mr)
        && is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 is available at runtime, and the checks above keep
        // every element read inside `a` (unit row stride) and every write
        // inside `pack`.
        unsafe {
            pack_a_simd_contiguous_f64(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        // Fall back to optimized scalar path.
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Pack contiguous column data using SIMD for f64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_a_simd_contiguous_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    use std::arch::x86_64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            // Full block - use AVX2 for copies
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using AVX2 (4 doubles at a time)
                let mut j = 0;
                while j + 4 <= mr {
                    let v = _mm256_loadu_pd(src.add(j));
                    _mm256_storeu_pd(dst_ptr.add(j), v);
                    j += 4;
                }

                // Handle remaining elements
                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            // Partial block - scalar with zero padding
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// SIMD-optimized `pack_a` for f32.
///
/// See [`pack_a_simd_f64`] for why any `row_stride` is supported: columns are
/// always contiguous in this crate's column-major storage, so the AVX2 path
/// copies `mr` contiguous elements per column. A full AVX2 register holds 8 f32.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn pack_a_simd_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    if mr >= 8
        && simd_pack_a_args_ok(a, row_start, col_start, nrows, ncols, pack, mr)
        && is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 is available at runtime, and the checks above keep
        // every element read inside `a` (unit row stride) and every write
        // inside `pack`.
        unsafe {
            pack_a_simd_contiguous_f32(a, row_start, col_start, nrows, ncols, pack, mr);
        }
        return;
    }
    pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
}

/// Pack contiguous column data using SIMD for f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_a_simd_contiguous_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    use std::arch::x86_64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using AVX2 (8 floats at a time)
                let mut j = 0;
                while j + 8 <= mr {
                    let v = _mm256_loadu_ps(src.add(j));
                    _mm256_storeu_ps(dst_ptr.add(j), v);
                    j += 8;
                }

                // Handle remaining elements
                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// `pack_b` for f64: delegates to [`pack_b_optimized`], which gathers with a
/// compile-time `nr` for every kernel width (4, 6, 8, 16 on x86_64) and checks
/// its arguments. (This used to be a separate, unchecked 8-way unrolled loop
/// producing the same layout.)
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn pack_b_simd_f64(
    b: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    nr: usize,
) {
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

/// `pack_b` for f32: delegates to [`pack_b_optimized`]; see
/// [`pack_b_simd_f64`].
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn pack_b_simd_f32(
    b: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    nr: usize,
) {
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

// =============================================================================
// NEON-Optimized Packing for ARM (aarch64)
// =============================================================================

/// SIMD-optimized `pack_a` for f64 on ARM NEON.
///
/// Columns are always contiguous in this crate's column-major storage (element
/// `(i, j)` is at `ptr + i + j * row_stride`), so the NEON path copies `mr`
/// contiguous elements per column for **any** `row_stride`. A NEON 128-bit
/// register holds 2 f64.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_a_simd_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    if mr >= 2 && simd_pack_a_args_ok(a, row_start, col_start, nrows, ncols, pack, mr) {
        // SAFETY: NEON is part of the aarch64 baseline, and the checks above
        // keep every element read inside `a` (unit row stride) and every
        // write inside `pack`.
        unsafe {
            pack_a_neon_contiguous_f64(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Pack contiguous column data using NEON for f64.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pack_a_neon_contiguous_f64(
    a: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    mr: usize,
) {
    use std::arch::aarch64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using NEON (2 doubles at a time)
                let mut j = 0;
                while j + 2 <= mr {
                    let v = vld1q_f64(src.add(j));
                    vst1q_f64(dst_ptr.add(j), v);
                    j += 2;
                }

                // Handle remaining
                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// SIMD-optimized `pack_a` for f32 on ARM NEON.
///
/// See [`pack_a_simd_f64`] (aarch64) for why any `row_stride` is supported:
/// columns are always contiguous, so the NEON path copies `mr` contiguous
/// elements per column. A NEON 128-bit register holds 4 f32.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_a_simd_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    if mr >= 4 && simd_pack_a_args_ok(a, row_start, col_start, nrows, ncols, pack, mr) {
        // SAFETY: NEON is part of the aarch64 baseline, and the checks above
        // keep every element read inside `a` (unit row stride) and every
        // write inside `pack`.
        unsafe {
            pack_a_neon_contiguous_f32(a, row_start, col_start, nrows, ncols, pack, mr);
        }
    } else {
        pack_a_optimized(a, row_start, col_start, nrows, ncols, pack, mr);
    }
}

/// Pack contiguous column data using NEON for f32.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pack_a_neon_contiguous_f32(
    a: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    mr: usize,
) {
    use std::arch::aarch64::*;

    let base_ptr = a.as_ptr();
    let row_stride = a.row_stride();
    let dst = pack.as_mut_ptr();
    let mut idx = 0;

    for i in (0..nrows).step_by(mr) {
        let ib = mr.min(nrows - i);

        if ib == mr {
            for p in 0..ncols {
                let src = base_ptr.add(row_start + i + (col_start + p) * row_stride);
                let dst_ptr = dst.add(idx);

                // Copy mr elements using NEON (4 floats at a time)
                let mut j = 0;
                while j + 4 <= mr {
                    let v = vld1q_f32(src.add(j));
                    vst1q_f32(dst_ptr.add(j), v);
                    j += 4;
                }

                while j < mr {
                    *dst_ptr.add(j) = *src.add(j);
                    j += 1;
                }

                idx += mr;
            }
        } else {
            for p in 0..ncols {
                for ii in 0..ib {
                    *dst.add(idx) =
                        *base_ptr.add(row_start + i + ii + (col_start + p) * row_stride);
                    idx += 1;
                }
                for _ in ib..mr {
                    *dst.add(idx) = 0.0;
                    idx += 1;
                }
            }
        }
    }
}

/// `pack_b` for f64 on ARM: delegates to [`pack_b_optimized`], which
/// transposes `nr = 6` panels (the NEON f64 kernel's width) in NEON registers
/// and gathers other widths with a compile-time `nr`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_b_simd_f64(
    b: &MatRef<'_, f64>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f64>,
    nr: usize,
) {
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

/// `pack_b` for f32 on ARM: delegates to [`pack_b_optimized`], which
/// transposes `nr = 8` panels (the NEON f32 kernel's width) in NEON registers.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn pack_b_simd_f32(
    b: &MatRef<'_, f32>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    pack: &mut AlignedVec<f32>,
    nr: usize,
) {
    pack_b_optimized(b, row_start, col_start, nrows, ncols, pack, nr);
}

#[cfg(test)]
#[path = "gemm_packing_tests.rs"]
mod tests;
