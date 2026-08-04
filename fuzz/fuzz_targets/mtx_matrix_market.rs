#![no_main]

//! Fuzzes `oxiblas_sparse::read_matrix_market_str`, the Matrix Market (.mtx)
//! parser used to ingest SuiteSparse-style sparse matrices.
//!
//! This is the workspace's primary untrusted-input parser: `nnz`, `nrows`,
//! and `ncols` all come straight out of the file's header/size line and, at
//! audit time, drove unbounded `Vec::with_capacity` reservations before the
//! COOLJAPAN hygiene pass added `nnz <= nrows*ncols` (coordinate format) and
//! checked-multiplication (array format) validation. The property under
//! test is simply: no input, however malformed, should panic, abort
//! (capacity-overflow / OOM), or hang — a well-formed `Result::Err` for
//! malformed input is always fine.
//!
//! Run with (from the `fuzz/` directory, nightly toolchain required):
//! `cargo +nightly fuzz run mtx_matrix_market`

use libfuzzer_sys::fuzz_target;
use oxiblas_sparse::read_matrix_market_str;

fuzz_target!(|data: &[u8]| {
    // Matrix Market is a text format; invalid UTF-8 should be rejected by
    // the parser like any other malformed input, not by the harness, so
    // fall back to lossy conversion rather than skipping the input.
    let text = String::from_utf8_lossy(data);

    let _ = read_matrix_market_str::<f64>(&text);
    let _ = read_matrix_market_str::<f32>(&text);
});
