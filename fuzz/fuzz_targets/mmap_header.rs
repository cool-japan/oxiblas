#![no_main]

//! Fuzzes `oxiblas_matrix::mmap::MmapMat::<f64>::open`, the `.oxiblas`
//! memory-mapped-matrix header parser.
//!
//! `Header::from_bytes` / `Header::validate` / `Header::validate_layout`
//! were hardened (checked arithmetic, explicit length checks against the
//! real mapping size) so that a truncated or adversarially-crafted header
//! cannot make `MmapMat::open` hand back a struct whose recorded
//! `nrows`/`ncols`/`row_stride` claim more elements than the backing mmap
//! actually holds — the failure mode being guarded against is a
//! bounds-panic-free out-of-bounds read from later, fully safe accessor
//! calls. The property under test: no input file, however malformed,
//! should let `open` succeed with an unsound layout, or panic while
//! rejecting it.
//!
//! This target round-trips fuzz input through an actual temp file (mmap
//! needs a real file descriptor); the file is created fresh per-input and
//! removed afterward so repeated runs cannot accumulate state.
//!
//! Run with (from the `fuzz/` directory, nightly toolchain required):
//! `cargo +nightly fuzz run mmap_header`

use libfuzzer_sys::fuzz_target;
use oxiblas_matrix::mmap::MmapMat;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

fuzz_target!(|data: &[u8]| {
    // Unique-per-input, unique-per-process file name under the OS temp
    // directory (never a hardcoded path) so parallel fuzzer workers cannot
    // collide with each other or with a prior run's leftovers.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "oxiblas-fuzz-mmap-header-{}-{id}.oxiblas",
        std::process::id()
    ));

    {
        let Ok(mut file) = std::fs::File::create(&path) else {
            return;
        };
        // A write failure just means this input can't be exercised this
        // run; nothing to assert either way.
        let _ = file.write_all(data);
    }

    // Every outcome here is fine except a panic: `Ok` means the header
    // passed validation (and every accessor below must then stay in
    // bounds), `Err` means malformed input was correctly rejected.
    if let Ok(mmat) = MmapMat::<f64>::open(&path) {
        let nrows = mmat.nrows();
        let ncols = mmat.ncols();
        // Touch the corners a caller would naturally read first; if
        // `validate_layout` ever regresses to trust unchecked header
        // dimensions, this is what turns the bug into an observable crash
        // instead of silent OOB.
        if nrows > 0 && ncols > 0 {
            let _ = mmat[(0, 0)];
            let _ = mmat[(nrows - 1, ncols - 1)];
        }
    }

    let _ = std::fs::remove_file(&path);
});
