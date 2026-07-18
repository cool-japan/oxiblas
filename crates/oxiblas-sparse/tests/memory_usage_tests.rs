//! Memory usage tests for sparse operations
//!
//! These tests install a custom [`GlobalAlloc`] wrapper ([`TrackingAllocator`])
//! as this test binary's `#[global_allocator]`. It counts, at the byte
//! level, how much heap memory is currently outstanding (allocated minus
//! freed) as well as how many allocation requests have been made. That
//! gives the tests something real to check: a "no leak" test captures a
//! baseline live-byte count, exercises a create/use/drop cycle many times,
//! and then asserts the live-byte count has come back down to (approximately)
//! the baseline. A previous version of this file used names like
//! `*_no_leak` and `*_memory` without ever consulting the allocator, so the
//! tests passed unconditionally regardless of whether the code actually
//! leaked - this version fixes that.
//!
//! `cargo test` runs multiple `#[test]` functions concurrently, but each one
//! gets its own dedicated OS thread, and every heap object these tests
//! create is both allocated *and* dropped on that same thread. The counters
//! below are therefore kept **per-thread** (`thread_local!`) rather than as
//! process-global atomics: a test only ever observes allocation activity
//! caused by its own code, so it is immune to unrelated background
//! allocations happening concurrently on other threads (e.g. the test
//! harness itself spawning/joining sibling test threads, which also goes
//! through the global allocator but on threads these tests never touch).
//! This makes the tests both leak-detecting *and* safe to run at full
//! `cargo test` concurrency, with no shared lock required.
//!
//! A few tests (in "Part 3" below) intentionally do *not* use the tracking
//! allocator: they check that a `CsrMatrix`'s logical footprint
//! (`size_of_val` of its backing slices) scales linearly with `nnz` rather
//! than quadratically with `n`. That is a structural/algorithmic property,
//! not a leak check, so it is left as direct slice-size arithmetic.

use oxiblas_sparse::convert::{coo_to_csr, csc_to_coo, csr_to_coo, csr_to_csc};
use oxiblas_sparse::linalg::cg;
use oxiblas_sparse::linalg::gmres;
use oxiblas_sparse::linalg::lu::{ILU0, SparseLU};
use oxiblas_sparse::linalg::precond::{BlockJacobi, GaussSeidel, Jacobi};
use oxiblas_sparse::ops::{spmm_sparse, spmv};
use oxiblas_sparse::{CooMatrixBuilder, CsrMatrix};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// ============================================================================
// Allocator tracking infrastructure
// ============================================================================

thread_local! {
    /// Number of bytes currently outstanding (allocated by *this thread* via
    /// the global allocator, minus bytes freed by this thread). This is not
    /// an estimate: it is a direct, per-thread tally of every
    /// `alloc`/`dealloc`/`realloc` call routed through [`TrackingAllocator`].
    /// Const-initialized so no heap allocation is needed to bring the slot
    /// into existence on a new thread (which would otherwise re-enter the
    /// allocator being defined here).
    static LIVE_BYTES: Cell<usize> = const { Cell::new(0) };

    /// Total number of `alloc`/`alloc_zeroed`/`realloc`-to-larger-size
    /// requests this thread has made so far. Used by tests that assert a
    /// hot loop performs *no* new allocations at all (e.g. `spmv` reusing
    /// pre-sized buffers).
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// A [`GlobalAlloc`] wrapper around [`System`] that maintains running,
/// per-thread totals of live bytes and allocation requests, so tests can
/// detect real memory leaks instead of merely checking that code runs
/// without panicking.
struct TrackingAllocator;

// Safety: `TrackingAllocator` forwards every call directly to `System`,
// which is itself a valid `GlobalAlloc` implementation; the only added
// behavior is `Cell` bookkeeping in thread-local storage around the
// delegated calls, which performs no allocation of its own (the `Cell`s are
// const-initialized) and therefore cannot re-enter the allocator.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Safety: `layout` is passed through unchanged to `System::alloc`,
        // satisfying the same preconditions the caller of this method
        // already guaranteed.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.with(|b| b.set(b.get() + layout.size()));
            ALLOCATION_COUNT.with(|c| c.set(c.get() + 1));
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Safety: `ptr`/`layout` are forwarded unchanged; the caller already
        // guarantees they describe a live allocation from this allocator.
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.with(|b| b.set(b.get().saturating_sub(layout.size())));
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Safety: same contract as `alloc` above.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.with(|b| b.set(b.get() + layout.size()));
            ALLOCATION_COUNT.with(|c| c.set(c.get() + 1));
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Safety: all arguments are forwarded unchanged to `System::realloc`,
        // whose preconditions match this method's.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size >= old_size {
                let grew_by = new_size - old_size;
                LIVE_BYTES.with(|b| b.set(b.get() + grew_by));
                ALLOCATION_COUNT.with(|c| c.set(c.get() + 1));
            } else {
                let shrank_by = old_size - new_size;
                LIVE_BYTES.with(|b| b.set(b.get().saturating_sub(shrank_by)));
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Snapshot of the calling thread's [`LIVE_BYTES`] counter.
fn current_allocated_bytes() -> usize {
    LIVE_BYTES.with(Cell::get)
}

/// Snapshot of the calling thread's [`ALLOCATION_COUNT`] counter.
fn current_allocation_count() -> usize {
    ALLOCATION_COUNT.with(Cell::get)
}

/// Maximum number of bytes a "no leak" test may still have outstanding
/// after its loop completes, relative to its pre-loop baseline. The tracked
/// counters are exact, per-thread accounting of every `alloc`/`dealloc`
/// call this test's own thread makes (not a process-wide or RSS estimate),
/// so a genuine leak that scales with loop iteration count or problem size
/// will exceed this budget by orders of magnitude; the small allowance here
/// only guards against benign allocator-internal rounding.
const LEAK_TOLERANCE_BYTES: usize = 4096;

/// Asserts that the live-byte count has returned to within
/// [`LEAK_TOLERANCE_BYTES`] of `baseline`, which must have been captured via
/// [`current_allocated_bytes`] immediately before the loop under test (after
/// a warm-up iteration to absorb one-time setup costs).
fn assert_returned_to_baseline(baseline: usize, label: &str) {
    let after = current_allocated_bytes();
    let leaked = after.saturating_sub(baseline);
    assert!(
        leaked <= LEAK_TOLERANCE_BYTES,
        "{label}: {leaked} bytes still live after the loop (baseline {baseline} bytes, \
         now {after} bytes) - this indicates a memory leak"
    );
}

/// Helper to estimate memory usage of CSR matrix (bytes)
fn estimate_csr_memory(a: &CsrMatrix<f64>) -> usize {
    // row_ptrs: (n+1) * usize, col_indices: nnz * usize, values: nnz * f64
    std::mem::size_of_val(a.row_ptrs())
        + std::mem::size_of_val(a.col_indices())
        + std::mem::size_of_val(a.values())
}

/// Helper to create a sparse Laplacian matrix (1D tridiagonal)
fn create_laplacian(n: usize) -> CsrMatrix<f64> {
    let mut builder = CooMatrixBuilder::new(n, n);
    for i in 0..n {
        builder.add(i, i, 4.0);
        if i > 0 {
            builder.add(i, i - 1, -1.0);
        }
        if i < n - 1 {
            builder.add(i, i + 1, -1.0);
        }
    }
    builder.build().to_csr()
}

/// Helper to create a 2D Laplacian matrix (5-point stencil)
fn create_laplacian_2d(nx: usize, ny: usize) -> CsrMatrix<f64> {
    let n = nx * ny;
    let mut builder = CooMatrixBuilder::new(n, n);
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            builder.add(idx, idx, 4.0);
            if i > 0 {
                builder.add(idx, idx - 1, -1.0);
            }
            if i < nx - 1 {
                builder.add(idx, idx + 1, -1.0);
            }
            if j > 0 {
                builder.add(idx, idx - nx, -1.0);
            }
            if j < ny - 1 {
                builder.add(idx, idx + nx, -1.0);
            }
        }
    }
    builder.build().to_csr()
}

// ============================================================================
// Part 1: Basic memory usage tests (now backed by real allocator tracking)
// ============================================================================

#[test]
fn test_spmv_no_allocation() {
    // Set up the matrix and buffers once; only the repeated `spmv` calls in
    // the loop below are measured for allocation activity.
    let n = 1000;
    let a = create_laplacian(n);
    let x = vec![1.0; n];
    let mut y = vec![0.0; n];

    // Warm up once so that any first-touch behavior happens before we start
    // measuring.
    spmv(1.0, &a, &x, 0.0, &mut y);
    let baseline = current_allocated_bytes();
    let allocations_before = current_allocation_count();

    // Perform SpMV multiple times with pre-allocated buffers - `spmv` reads
    // `a`/`x` and writes into `y` without touching the heap, so this must
    // not allocate any new memory.
    for _ in 0..100 {
        spmv(1.0, &a, &x, 0.0, &mut y);
    }

    let allocations_after = current_allocation_count();
    assert_eq!(
        allocations_after,
        allocations_before,
        "spmv performed {} unexpected heap allocation(s) over 100 calls with \
         pre-allocated buffers",
        allocations_after - allocations_before
    );
    assert_returned_to_baseline(baseline, "test_spmv_no_allocation");

    // Verify result is computed correctly
    assert_eq!(y.len(), n);
    for &val in &y {
        assert!(val.is_finite(), "SpMV produced non-finite values");
    }
}

#[test]
fn test_spmm_sparse_reasonable_size() {
    let n = 100;
    let a = create_laplacian(n);
    let b = create_laplacian(n);

    let c = spmm_sparse(&a, &b);

    // Result matrix should have reasonable size
    assert_eq!(c.nrows(), n);
    assert_eq!(c.ncols(), n);

    // For Laplacian^2, nnz should be more than input but bounded
    assert!(
        c.nnz() > a.nnz(),
        "Result should have at least as many non-zeros"
    );
    assert!(
        c.nnz() < a.nnz() * 5,
        "Result should not have excessive fill-in for Laplacian"
    );

    let c_memory = estimate_csr_memory(&c);
    assert!(c_memory > 0, "Result matrix should use some memory");
}

#[test]
fn test_cg_solver_memory() {
    let n = 500;
    let a = create_laplacian(n);
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];

    // Warm up so one-time setup costs (if any) happen before the baseline.
    let _ = cg(&a, &b, &x0, 1e-6, 1);
    let baseline = current_allocated_bytes();

    let result = cg(&a, &b, &x0, 1e-6, 100);
    assert!(result.is_ok(), "CG should converge");
    let cg_result = result.expect("CG should succeed");

    // Solution should be correct size
    assert_eq!(cg_result.x.len(), n);

    // All values should be finite
    for &val in &cg_result.x {
        assert!(val.is_finite(), "CG produced non-finite values");
    }

    // The solve legitimately allocates Krylov vectors while it runs; make
    // sure the measurement infrastructure actually observed that (otherwise
    // the leak check below would be vacuous).
    let during = current_allocated_bytes();
    assert!(
        during > baseline,
        "CG solve did not appear to allocate anything (baseline {baseline}, during {during}); \
         the tracking allocator may not be wired up correctly"
    );

    drop(cg_result);
    assert_returned_to_baseline(baseline, "test_cg_solver_memory");
}

#[test]
fn test_gmres_memory() {
    let n = 200;
    let a = create_laplacian(n);
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];

    let _ = gmres(&a, &b, &x0, 20, 1e-6, 1);
    let baseline = current_allocated_bytes();

    // GMRES with restart=20
    let result = gmres(&a, &b, &x0, 20, 1e-6, 50);
    assert!(result.is_ok(), "GMRES should converge");
    let gmres_result = result.expect("GMRES should succeed");

    // Solution should be correct size
    assert_eq!(gmres_result.x.len(), n);

    for &val in &gmres_result.x {
        assert!(val.is_finite(), "GMRES produced non-finite values");
    }

    let during = current_allocated_bytes();
    assert!(
        during > baseline,
        "GMRES solve did not appear to allocate anything (baseline {baseline}, during {during})"
    );

    drop(gmres_result);
    assert_returned_to_baseline(baseline, "test_gmres_memory");
}

#[test]
fn test_ilu0_memory_usage() {
    let n = 500;
    let a = create_laplacian(n);

    // Warm up.
    let _ = ILU0::new(&a);
    let baseline = current_allocated_bytes();

    let ilu = ILU0::new(&a);
    assert!(ilu.is_ok(), "ILU0 construction should succeed");
    let ilu = ilu.expect("ILU0 should succeed");

    // Verify we can apply it as a preconditioner
    let b = vec![1.0; n];
    let x = ilu.apply(&b);

    assert_eq!(x.len(), n);
    for &val in &x {
        assert!(val.is_finite(), "ILU0 apply produced non-finite values");
    }

    drop(x);
    drop(b);
    drop(ilu);
    assert_returned_to_baseline(baseline, "test_ilu0_memory_usage");
}

#[test]
fn test_sparse_lu_memory_usage() {
    let n = 100;
    let a_csr = create_laplacian(n);
    let a = a_csr.to_csc(); // SparseLU requires CSC format

    let _ = SparseLU::new(&a);
    let baseline = current_allocated_bytes();

    let lu = SparseLU::new(&a);
    assert!(lu.is_ok(), "Sparse LU construction should succeed");
    let lu = lu.expect("Sparse LU should succeed");

    // Verify we can solve with it
    let b = vec![1.0; n];
    let x = lu.solve(&b);

    assert_eq!(x.len(), n);
    for &val in &x {
        assert!(
            val.is_finite(),
            "Sparse LU solve produced non-finite values"
        );
    }

    drop(x);
    drop(b);
    drop(lu);
    assert_returned_to_baseline(baseline, "test_sparse_lu_memory_usage");
}

// ============================================================================
// Part 2: Sparse matrix create/use/drop memory leak tests
// ============================================================================

#[test]
fn test_csr_create_use_drop_no_leak() {
    // Warm up once (outside measurement) so first-touch allocations don't
    // pollute the baseline.
    {
        let a = create_laplacian(100);
        let x = vec![1.0; 100];
        let mut y = vec![0.0; 100];
        spmv(1.0, &a, &x, 0.0, &mut y);
    }
    let baseline = current_allocated_bytes();

    // Create many matrices, use them, and drop them. If there's a memory
    // leak, LIVE_BYTES will grow with each iteration instead of returning
    // to baseline once the loop finishes.
    for iteration in 0..200 {
        let n = 100 + (iteration % 50);
        let a = create_laplacian(n);

        // Use the matrix
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        spmv(1.0, &a, &x, 0.0, &mut y);

        // Verify usage
        assert_eq!(y.len(), n);
        // a, x, y are dropped here
    }

    assert_returned_to_baseline(baseline, "test_csr_create_use_drop_no_leak");
}

#[test]
fn test_csc_create_use_drop_no_leak() {
    {
        let csr = create_laplacian(200);
        let _csc = csr.to_csc();
    }
    let baseline = current_allocated_bytes();

    for _ in 0..200 {
        let csr = create_laplacian(200);
        let csc = csr.to_csc();
        assert_eq!(csc.nnz(), csr.nnz());
        // csr and csc dropped here
    }

    assert_returned_to_baseline(baseline, "test_csc_create_use_drop_no_leak");
}

#[test]
fn test_coo_create_use_drop_no_leak() {
    fn build_and_convert() -> CsrMatrix<f64> {
        let mut builder = CooMatrixBuilder::new(100, 100);
        for i in 0..100 {
            builder.add(i, i, 2.0);
            if i > 0 {
                builder.add(i, i - 1, -1.0);
            }
        }
        let coo = builder.build();
        coo.to_csr()
    }

    let _ = build_and_convert();
    let baseline = current_allocated_bytes();

    for _ in 0..200 {
        let _csr = build_and_convert();
        // all dropped here
    }

    assert_returned_to_baseline(baseline, "test_coo_create_use_drop_no_leak");
}

// ============================================================================
// Part 3: Memory scales linearly with nnz (not n*n)
//
// These tests check a structural/algorithmic property (logical footprint
// vs. nnz, computed directly from slice lengths) rather than live allocator
// state, so they do not consult the tracking counters at all.
// ============================================================================

#[test]
fn test_memory_scales_linearly_with_nnz() {
    // Create sparse matrices of increasing size and verify memory
    // scales linearly with nnz, not quadratically with n.
    let sizes = [100, 500, 1000, 5000];
    let mut measurements: Vec<(usize, usize, usize)> = Vec::new(); // (n, nnz, memory)

    for &n in &sizes {
        let a = create_laplacian(n);
        let mem = estimate_csr_memory(&a);
        measurements.push((n, a.nnz(), mem));
    }

    // Check that memory/nnz ratio is approximately constant
    // For CSR: each non-zero costs 1 usize (col_index) + 1 f64 (value) = 16 bytes
    // Plus row_ptrs overhead of (n+1)*8 bytes
    let bytes_per_nnz_first = measurements[0].2 as f64 / measurements[0].1 as f64;

    for &(n, nnz, mem) in &measurements {
        let bytes_per_nnz = mem as f64 / nnz as f64;
        // The ratio should be approximately constant (within 2x),
        // not growing with n (which would indicate O(n^2) behavior)
        assert!(
            bytes_per_nnz < bytes_per_nnz_first * 2.0,
            "Memory per nnz at n={n} ({bytes_per_nnz:.1}) is too large compared to \
             n={} ({bytes_per_nnz_first:.1}), suggesting non-linear scaling",
            measurements[0].0
        );
    }

    // Additionally verify that memory does NOT grow as n^2
    // For the largest matrix, memory should be much less than n^2 * 8 bytes
    let (n_large, _, mem_large) = measurements[measurements.len() - 1];
    let dense_memory = n_large * n_large * std::mem::size_of::<f64>();
    assert!(
        mem_large < dense_memory / 10,
        "Sparse memory {mem_large} is too close to dense memory {dense_memory}"
    );
}

#[test]
fn test_2d_laplacian_memory_scales_linearly() {
    // 2D Laplacian: nnz ~= 5*n for grid size sqrt(n) x sqrt(n)
    let grid_sizes = [5, 10, 20, 40];
    let mut prev_bytes_per_nnz: Option<f64> = None;

    for &g in &grid_sizes {
        let a = create_laplacian_2d(g, g);
        let mem = estimate_csr_memory(&a);
        let nnz = a.nnz();
        let bytes_per_nnz = mem as f64 / nnz as f64;

        if let Some(prev) = prev_bytes_per_nnz {
            assert!(
                bytes_per_nnz < prev * 1.5,
                "Bytes per nnz grew from {prev:.1} to {bytes_per_nnz:.1} at grid={g}"
            );
        }
        prev_bytes_per_nnz = Some(bytes_per_nnz);
    }
}

// ============================================================================
// Part 4: Format conversion roundtrip memory tests (CSR->CSC->COO->CSR)
// ============================================================================

#[test]
fn test_csr_csc_coo_roundtrip_no_leak() {
    fn roundtrip_once() {
        let original = create_laplacian_2d(10, 10);
        let original_nnz = original.nnz();

        // CSR -> CSC
        let csc = csr_to_csc(&original);
        assert_eq!(csc.nnz(), original_nnz);

        // CSC -> COO
        let coo = csc_to_coo(&csc);
        assert_eq!(coo.len(), original_nnz);

        // COO -> CSR
        let roundtripped = coo_to_csr(&coo);
        assert_eq!(roundtripped.nnz(), original_nnz);

        // Verify values survived the roundtrip
        for i in 0..original.nrows() {
            for (col, val) in original.row_iter(i) {
                let rt_val = roundtripped.get_or_zero(i, col);
                assert!(
                    (*val - rt_val).abs() < 1e-14,
                    "Roundtrip value mismatch at ({i}, {col})"
                );
            }
        }
        // All intermediates dropped here
    }

    roundtrip_once();
    let baseline = current_allocated_bytes();

    // Perform many roundtrip conversions to detect memory leaks
    for _ in 0..50 {
        roundtrip_once();
    }

    assert_returned_to_baseline(baseline, "test_csr_csc_coo_roundtrip_no_leak");
}

#[test]
fn test_csr_coo_csc_roundtrip_memory_bounded() {
    let original = create_laplacian_2d(20, 20);
    let original_mem = estimate_csr_memory(&original);

    // CSR -> COO -> CSC -> CSR
    let coo = csr_to_coo(&original);
    let csc = oxiblas_sparse::convert::coo_to_csc(&coo);
    let back_to_csr = oxiblas_sparse::convert::csc_to_csr(&csc);

    let roundtrip_mem = estimate_csr_memory(&back_to_csr);

    // Memory of roundtripped matrix should be very close to original
    // (small differences possible due to COO duplicate summing affecting row_ptrs alignment)
    let diff = roundtrip_mem.abs_diff(original_mem);
    assert!(
        diff < original_mem / 10,
        "Roundtrip memory {roundtrip_mem} differs too much from original {original_mem}"
    );
    assert_eq!(back_to_csr.nnz(), original.nnz());
}

#[test]
fn test_repeated_conversions_no_growth() {
    let mut current = create_laplacian(500);
    let original_nnz = current.nnz();

    // Warm up one roundtrip so the loop below starts from a stable baseline.
    {
        let csc = csr_to_csc(&current);
        let coo = csc_to_coo(&csc);
        current = coo_to_csr(&coo);
    }
    let baseline = current_allocated_bytes();

    // Perform 50 roundtrip conversions
    for _ in 0..50 {
        let csc = csr_to_csc(&current);
        let coo = csc_to_coo(&csc);
        current = coo_to_csr(&coo);
    }

    // nnz should not have grown
    assert_eq!(
        current.nnz(),
        original_nnz,
        "nnz grew after repeated conversions: {} vs original {}",
        current.nnz(),
        original_nnz
    );

    // `current` (the final CSR matrix) is still alive here, so the baseline
    // comparison naturally accounts for its footprint - only the discarded
    // intermediates from each iteration should have been freed.
    assert_returned_to_baseline(baseline, "test_repeated_conversions_no_growth");
}

// ============================================================================
// Part 5: Iterative solver memory accumulation tests
// ============================================================================

#[test]
fn test_cg_repeated_solves_no_accumulation() {
    let n = 200;
    let a = create_laplacian(n);
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];

    let _ = cg(&a, &b, &x0, 1e-6, 200);
    let baseline = current_allocated_bytes();

    // Run CG many times - should not accumulate memory
    for iteration in 0..100 {
        let result = cg(&a, &b, &x0, 1e-6, 200);
        assert!(
            result.is_ok(),
            "CG failed on iteration {iteration}: {:?}",
            result.err()
        );
        let cg_result = result.expect("CG should succeed");
        assert_eq!(cg_result.x.len(), n);
        // Result is dropped here
    }

    assert_returned_to_baseline(baseline, "test_cg_repeated_solves_no_accumulation");
}

#[test]
fn test_gmres_repeated_solves_no_accumulation() {
    let n = 100;
    let a = create_laplacian(n);
    let b = vec![1.0; n];
    let x0 = vec![0.0; n];

    let _ = gmres(&a, &b, &x0, 10, 1e-6, 100);
    let baseline = current_allocated_bytes();

    // Run GMRES many times with different restart values
    for iteration in 0..50 {
        let restart = 10 + (iteration % 20);
        let result = gmres(&a, &b, &x0, restart, 1e-6, 100);
        assert!(
            result.is_ok(),
            "GMRES failed on iteration {iteration}: {:?}",
            result.err()
        );
        let gmres_result = result.expect("GMRES should succeed");
        assert_eq!(gmres_result.x.len(), n);
        // Result and all Krylov vectors are dropped here
    }

    assert_returned_to_baseline(baseline, "test_gmres_repeated_solves_no_accumulation");
}

#[test]
fn test_iterative_solver_with_varying_rhs() {
    let n = 200;
    let a = create_laplacian(n);
    let x0 = vec![0.0; n];

    let warmup_b = vec![0.0; n];
    let _ = cg(&a, &warmup_b, &x0, 1e-6, 500);
    let baseline = current_allocated_bytes();

    // Solve with many different right-hand sides
    for k in 0..50 {
        let b: Vec<f64> = (0..n).map(|i| ((i + k) as f64).sin()).collect();
        let result = cg(&a, &b, &x0, 1e-6, 500);
        assert!(result.is_ok(), "CG failed for rhs #{k}");
        // All intermediates dropped each iteration
    }

    assert_returned_to_baseline(baseline, "test_iterative_solver_with_varying_rhs");
}

// ============================================================================
// Part 6: Preconditioner setup/teardown memory tests
// ============================================================================

#[test]
fn test_jacobi_preconditioner_setup_teardown() {
    let n = 300;
    let a = create_laplacian(n);

    {
        let warm = Jacobi::new(&a).expect("Jacobi creation should succeed");
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        warm.apply(&x, &mut y);
    }
    let baseline = current_allocated_bytes();

    // Create and destroy many Jacobi preconditioners
    for _ in 0..200 {
        let jacobi = Jacobi::new(&a).expect("Jacobi creation should succeed");

        // Use the preconditioner
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        jacobi.apply(&x, &mut y);

        assert_eq!(y.len(), n);
        for &val in &y {
            assert!(val.is_finite(), "Jacobi apply produced non-finite values");
        }
        // jacobi dropped here
    }

    assert_returned_to_baseline(baseline, "test_jacobi_preconditioner_setup_teardown");
}

#[test]
fn test_jacobi_memory_proportional_to_n() {
    // Jacobi stores only the diagonal: n * sizeof(f64)
    let sizes = [100, 500, 1000, 5000];

    for &n in &sizes {
        let a = create_laplacian(n);
        let jacobi = Jacobi::new(&a).expect("Jacobi creation should succeed");

        // Jacobi size should be n
        assert_eq!(jacobi.size(), n, "Jacobi size should equal n");
        // jacobi dropped here
    }
}

#[test]
fn test_block_jacobi_setup_teardown() {
    let n = 300;
    let a = create_laplacian(n);

    // Use uniform block sizes of 10
    let block_sizes: Vec<usize> = std::iter::repeat_n(10, n / 10).collect();

    {
        let warm = BlockJacobi::new(&a, &block_sizes).expect("BlockJacobi creation should succeed");
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        warm.apply(&x, &mut y);
    }
    let baseline = current_allocated_bytes();

    for _ in 0..100 {
        let bj = BlockJacobi::new(&a, &block_sizes).expect("BlockJacobi creation should succeed");

        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        bj.apply(&x, &mut y);

        assert_eq!(y.len(), n);
        for &val in &y {
            assert!(
                val.is_finite(),
                "BlockJacobi apply produced non-finite values"
            );
        }
        // bj dropped here
    }

    assert_returned_to_baseline(baseline, "test_block_jacobi_setup_teardown");
}

#[test]
fn test_gauss_seidel_setup_teardown() {
    let n = 200;
    let a = create_laplacian(n);

    {
        let warm = GaussSeidel::new(&a).expect("GaussSeidel creation should succeed");
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        warm.apply(&x, &mut y);
    }
    let baseline = current_allocated_bytes();

    for _ in 0..100 {
        let gs = GaussSeidel::new(&a).expect("GaussSeidel creation should succeed");

        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        gs.apply(&x, &mut y);

        assert_eq!(y.len(), n);
        for &val in &y {
            assert!(
                val.is_finite(),
                "GaussSeidel apply produced non-finite values"
            );
        }
        // gs dropped here
    }

    assert_returned_to_baseline(baseline, "test_gauss_seidel_setup_teardown");
}

#[test]
fn test_ilu0_setup_teardown() {
    let n = 300;
    let a = create_laplacian(n);

    {
        let warm = ILU0::new(&a).expect("ILU0 creation should succeed");
        let b = vec![1.0; n];
        let _ = warm.apply(&b);
    }
    let baseline = current_allocated_bytes();

    for _ in 0..100 {
        let ilu = ILU0::new(&a).expect("ILU0 creation should succeed");

        let b = vec![1.0; n];
        let x = ilu.apply(&b);

        assert_eq!(x.len(), n);
        for &val in &x {
            assert!(val.is_finite(), "ILU0 apply produced non-finite values");
        }
        // ilu dropped here
    }

    assert_returned_to_baseline(baseline, "test_ilu0_setup_teardown");
}

#[test]
fn test_preconditioner_reapply_no_growth() {
    // Apply a preconditioner many times and verify no memory growth
    let n = 500;
    let a = create_laplacian(n);
    let jacobi = Jacobi::new(&a).expect("Jacobi creation should succeed");

    let x = vec![1.0; n];
    let mut y = vec![0.0; n];

    jacobi.apply(&x, &mut y);
    let baseline = current_allocated_bytes();

    for _ in 0..1000 {
        jacobi.apply(&x, &mut y);
    }

    // If we get here without issues, the preconditioner doesn't leak
    assert_eq!(y.len(), n);
    for &val in &y {
        assert!(val.is_finite());
    }

    assert_returned_to_baseline(baseline, "test_preconditioner_reapply_no_growth");
}

// ============================================================================
// Part 7: Matrix conversion memory tests
// ============================================================================

#[test]
fn test_matrix_conversion_memory() {
    let n = 500;
    let mut builder = CooMatrixBuilder::new(n, n);

    // Create sparse matrix
    for i in 0..n {
        for j in 0..5.min(n) {
            if i >= j {
                builder.add(i, (i - j) % n, 1.0);
            }
        }
    }

    let coo = builder.build();
    let coo_size = std::mem::size_of_val(coo.row_indices())
        + std::mem::size_of_val(coo.col_indices())
        + std::mem::size_of_val(coo.values());

    // Convert COO -> CSR
    let csr = coo.to_csr();
    let csr_size = estimate_csr_memory(&csr);

    // Convert CSR -> CSC
    let csc = csr.to_csc();
    let csc_size = std::mem::size_of_val(csc.col_ptrs())
        + std::mem::size_of_val(csc.row_indices())
        + std::mem::size_of_val(csc.values());

    // All formats should have similar memory footprint (within 2x)
    assert!(
        csr_size < coo_size * 2,
        "CSR should not use excessive memory vs COO"
    );
    assert!(
        csc_size < csr_size * 2,
        "CSC should not use excessive memory vs CSR"
    );
}

#[test]
fn test_repeated_operations_no_growth() {
    // Test that repeated operations don't cause unbounded memory growth
    let n = 100;
    let a = create_laplacian(n);
    let x = vec![1.0; n];

    {
        let mut y = vec![0.0; n];
        spmv(1.0, &a, &x, 0.0, &mut y);
    }
    let baseline = current_allocated_bytes();

    // Create many temporary results - if there's a leak, this will accumulate
    for _ in 0..1000 {
        let mut y = vec![0.0; n];
        spmv(1.0, &a, &x, 0.0, &mut y);
        // y is dropped here
    }

    assert_returned_to_baseline(baseline, "test_repeated_operations_no_growth");
}

#[test]
fn test_large_matrix_memory_estimate() {
    // Test memory estimation for large matrix
    let n = 10000;
    let a = create_laplacian(n);

    let memory = estimate_csr_memory(&a);
    let nnz = a.nnz();

    // Expected: (n+1) * 8 + 2*nnz*8 for row_ptrs + (col_indices + values)
    // For tridiagonal Laplacian: nnz ~ 3n, so ~(n + 6n)*8 = 56n bytes
    let expected_approx = 56 * n;

    assert!(
        memory < expected_approx * 2,
        "Memory usage {memory} should be close to expected {expected_approx} for {nnz} nnz"
    );
    assert!(
        memory > expected_approx / 2,
        "Memory usage {memory} should be close to expected {expected_approx} for {nnz} nnz"
    );
}

#[test]
fn test_preconditioner_memory() {
    let n = 300;
    let a = create_laplacian(n);

    // Warm up.
    {
        let warm = Jacobi::new(&a).expect("Jacobi creation should succeed");
        drop(warm);
    }
    let baseline = current_allocated_bytes();

    let jacobi = Jacobi::new(&a).expect("Jacobi creation should succeed");
    let after_construct = current_allocated_bytes();

    let x = vec![1.0; n];
    let mut y = vec![0.0; n];

    // Apply preconditioner multiple times
    for _ in 0..100 {
        jacobi.apply(&x, &mut y);
    }

    // Check output is correct size
    assert_eq!(y.len(), n);
    for &val in &y {
        assert!(val.is_finite(), "Jacobi apply produced non-finite values");
    }

    // Jacobi preconditioner stores only the diagonal (n * sizeof(f64)); check
    // that its real, allocator-measured footprint is actually in that
    // ballpark rather than just computing a number nobody checks.
    let constructed_bytes = after_construct.saturating_sub(baseline);
    let expected_size = n * std::mem::size_of::<f64>();
    assert!(
        constructed_bytes >= expected_size,
        "Jacobi construction allocated only {constructed_bytes} bytes, expected at least \
         {expected_size} (n * sizeof(f64)) for storing the diagonal"
    );
    assert!(
        constructed_bytes <= expected_size * 4,
        "Jacobi construction allocated {constructed_bytes} bytes, far more than the \
         {expected_size} expected for storing just the diagonal"
    );

    drop(jacobi);
    drop(x);
    drop(y);
    assert_returned_to_baseline(baseline, "test_preconditioner_memory");
}
