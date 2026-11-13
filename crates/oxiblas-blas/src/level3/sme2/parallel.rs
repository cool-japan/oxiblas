//! The parallel split of the SME2 GEMM driver (Wave 3).
//!
//! What will live here: the split of C into one part per SME unit (along its
//! longer side, at 64-row or panel-width column boundaries), each part run by
//! the sequential driver of `driver.rs` on a rayon worker with its own
//! packing buffers and its own readiness check; a part whose thread is not
//! ready is computed by the NEON blocked driver with the same KC partition,
//! which gives the same bits. Its tests live in `tests_gemm.rs`.
