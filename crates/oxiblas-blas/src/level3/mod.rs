//! Level 3 BLAS operations (matrix-matrix).
//!
//! This module provides operations that operate on matrices:
//!
//! - [`gemm::gemm`]: General matrix-matrix multiplication
//! - [`gemmt::gemmt`]: General matrix-matrix with triangular result update
//! - [`symm::symm`]: Symmetric matrix-matrix multiplication
//! - [`hemm::hemm`]: Hermitian matrix-matrix multiplication
//! - [`trmm::trmm`]: Triangular matrix-matrix multiplication
//! - [`trsm::trsm`]: Triangular solve (matrix)
//! - [`syrk::syrk`]: Symmetric rank-k update
//! - [`herk::herk`]: Hermitian rank-k update
//! - [`syr2k::syr2k`]: Symmetric rank-2k update
//! - [`her2k::her2k`]: Hermitian rank-2k update
//! - [`strassen`]: Strassen's algorithm for very large matrices
//! - [`batched`]: Batched operations for ML workloads (batch GEMM, GEMV, AXPY)
//!
//! [`sme2_active()`] reports whether the SME2 backend (the opt-in `sme2`
//! Cargo feature on `aarch64`) is active for calls made from the current
//! thread; in every other build it returns `false`.

pub mod autotune;
pub mod batched;
pub mod complex_gemm;
pub mod gemm;
#[cfg(test)]
mod gemm_ab_tests;
pub mod gemm_cache_oblivious;
pub mod gemm_kernel;
pub mod gemm_kernel_sse42;
pub mod gemm_packing;
#[cfg(feature = "parallel")]
mod gemm_parallel;
mod gemm_plan;
pub mod gemm_small;
mod gemm_thin;
pub mod gemm_winograd;
pub mod gemmt;
pub mod hemm;
pub mod her2k;
pub mod herk;
#[cfg(feature = "runtime-tuning")]
pub mod runtime_autotune;
#[cfg(all(feature = "sme2", target_arch = "aarch64"))]
pub(crate) mod sme2;
pub mod strassen;
pub mod symm;
pub mod syr2k;
pub mod syrk;
mod tri_gemm;
pub mod trmm;
pub mod trsm;

pub use autotune::{
    AutoTunedBlocking, CacheInfo, compute_blocking, compute_blocking_adaptive, get_cache_info,
};
pub use batched::{
    BatchedError, Transpose as BatchedTranspose, axpy_batched, gemm_batched, gemm_strided_batched,
    gemv_batched,
};
#[cfg(feature = "parallel")]
pub use batched::{
    axpy_batched_parallel, gemm_batched_parallel, gemm_strided_batched_parallel,
    gemv_batched_parallel,
};
pub use complex_gemm::{gemm3m_c32, gemm3m_c64};
pub use gemm::{
    GemmBlocking, gemm, gemm_asymmetric, gemm_asymmetric_with_par, gemm_auto, gemm_auto_with_par,
    gemm_with_blocking, gemm_with_par,
};
pub use gemm_cache_oblivious::{gemm_cache_oblivious, gemm_cache_oblivious_with_threshold};
pub use gemm_kernel::{GemmKernel, MicroKernelShape};
pub use gemm_packing::{
    PackingConfig, pack_a_contiguous, pack_a_optimized, pack_b_optimized, pack_b_streaming,
};
pub use gemm_small::{SMALL_THRESHOLD, gemm_small};
pub use gemm_winograd::{gemm_winograd, gemm_winograd_blocked};
pub use gemmt::{GemmtError, gemmt, gemmt_new, gemmt_symmetric};
pub use hemm::{HemmError, hemm, hemm_c32, hemm_c64, hemm_new};
pub use her2k::{Her2kError, her2k, her2k_new};
pub use herk::{HerkError, herk, herk_new};
#[cfg(feature = "parallel")]
pub use strassen::gemm_strassen_parallel;
pub use strassen::{
    STRASSEN_THRESHOLD, gemm_strassen, gemm_strassen_with_par, should_use_strassen,
};
pub use symm::{SymmError, symm, symm_new};
pub use syr2k::{Syr2kError, syr2k, syr2k_new};
pub use syrk::{SyrkError, syrk, syrk_new};
pub use trmm::{TrmmDiag, TrmmError, TrmmSide, TrmmTrans, TrmmUplo, trmm, trmm_in_place};
pub use trsm::{
    Diag, Side, Trans, Uplo, trsm, trsm_c32, trsm_c32_in_place, trsm_c64, trsm_c64_in_place,
    trsm_in_place, trsm_in_place_with_par, trsm_with_par,
};

/// Returns `true` when the SME2 backend is active for calls made from the
/// current thread.
///
/// The function exists in every build. Without the opt-in `sme2` Cargo
/// feature, or on any architecture other than `aarch64`, it always returns
/// `false`. With the feature on `aarch64` it is `true` only when the build
/// does not cap SIMD (`force-scalar`, `max-simd-128` and `max-simd-256` keep
/// the backend off), the `OXIBLAS_DISABLE_SME2` environment variable is unset,
/// empty or `0` (read once, at the first call), the operating system reports
/// SME2 with a 512-bit streaming vector length, and the calling thread is
/// not in streaming mode, holds no live ZA state and has no pending lazy ZA
/// save. A host where the detection cannot run (for example a sandbox that
/// forbids spawning `sysctl` on macOS) leaves the backend off.
///
/// A `true` result covers `f32`; `f64` additionally needs FEAT_SME_F64F64.
pub fn sme2_active() -> bool {
    #[cfg(all(feature = "sme2", target_arch = "aarch64"))]
    {
        sme2::sme2_active()
    }
    #[cfg(not(all(feature = "sme2", target_arch = "aarch64")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sme2_active_exists_in_every_build() {
        let active = super::sme2_active();
        // A build that caps SIMD keeps the backend off even with the feature.
        if cfg!(any(
            feature = "force-scalar",
            feature = "max-simd-128",
            feature = "max-simd-256"
        )) {
            assert!(!active, "a SIMD-capped build never runs SME2");
        }
        #[cfg(not(all(feature = "sme2", target_arch = "aarch64")))]
        assert!(!active, "no SME2 backend in this build");
        // Asking again from the same thread gives the same answer.
        assert_eq!(super::sme2_active(), active);
    }
}
