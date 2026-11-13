//! Stress tests of the SME2 GEMM driver (Wave 3): long alternations of SME2
//! products with NEON and scalar floating-point work on the same thread and
//! on several threads, with the streaming state checked after every call.
