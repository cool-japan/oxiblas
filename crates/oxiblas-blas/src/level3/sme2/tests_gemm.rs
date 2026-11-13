//! Tests of the SME2 GEMM driver of `driver.rs` and `parallel.rs` (Wave 3):
//! every accepted call against the NEON blocked result bit for bit, the
//! declining classes with C untouched, explicit KC, views, thread counts and
//! part counts, and the counters that prove SME2 ran.
