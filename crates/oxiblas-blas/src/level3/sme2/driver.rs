//! The SME2 GEMM driver (Wave 3, milestone 1: plain packers).
//!
//! What will live here: `try_gemm`, the one entry point of the backend. It
//! decides once, on the calling thread, whether a product runs on SME2
//! (element type, run-time detection, the caller's KC step, operand
//! normalisation, the size class, thread readiness), and then walks the
//! depth, column and row blocks like the NEON blocked driver, with the
//! caller's KC partition, the plain packers of `pack.rs` and one kernel call
//! per block. It returns `true` when C holds the complete result and `false`
//! when C is untouched. The GEMM hook that calls it lands in Wave 6. Its
//! tests live in `tests_gemm.rs` and `tests_stress.rs`.
