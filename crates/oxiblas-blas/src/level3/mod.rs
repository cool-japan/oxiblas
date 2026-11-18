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
mod gemm_policy;
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
/// SME2 with a 512-bit streaming vector length, the first-use checks of the
/// SME2 kernels passed on this CPU (they run once per process, at the first
/// call of this function or at the first GEMM or GEMV call the backend could
/// serve, whichever comes first), and the calling thread is not in streaming
/// mode, holds no live ZA state, has no pending lazy ZA save and still has a
/// 512-bit streaming vector length (Linux lets a thread change it). A host
/// where the detection cannot run (for example a sandbox that forbids
/// spawning `sysctl` on macOS) leaves the backend off.
///
/// A `true` result covers `f32`; `f64` additionally needs FEAT_SME_F64F64.
///
/// While the backend is active, [`gemm::gemm`] and the other GEMM entry
/// points run the products that a build without it computes on the blocked
/// route (more than 32³ multiply-adds) on SME2, and [`gemv`](fn@crate::level2::gemv)
/// and [`gemv_with_par`](fn@crate::level2::gemv_with_par) run GEMVs past
/// a size crossover on SME2, as do the GEMM entry points for the products
/// with one row or one column that they compute as one GEMV (the thin
/// products the default GEMM policy of aarch64 sends to its GEMV route:
/// those of its measured scope, with both operands stored column-major;
/// a thin product with a transposed or strided view as an operand takes
/// the blocked route above 32³, where the SME2 GEMM driver declines thin
/// products, and the small routes below). The
/// results equal those of a build without the backend bit for bit, except
/// the sign and payload of NaN results and the floating-point flags.
///
/// This function waits for another thread that runs the detection or the
/// first-use checks, and returns their answer, on any calling thread; it
/// runs the checks itself only on a thread that is ready for SME code (see
/// above: outside streaming mode, no live ZA state, no pending lazy ZA
/// save, a 512-bit streaming vector length; a thread outside SME code
/// whose vector length was changed is not ready). The GEMM and GEMV calls
/// never wait for another thread (such a call runs on NEON, with the same
/// result). When this function returns, no
/// detection or check that was in progress when it was called is still in
/// progress. In a build that does not cap SIMD the detection is then
/// settled. Where it found SME2 that the backend can use (a 512-bit
/// streaming vector length, not turned off by `OXIBLAS_DISABLE_SME2`), the
/// checks are settled too, unless they have not run yet (only possible when
/// it was called from a thread that is not ready, for example one inside
/// SME code); where it found none, the checks never run. A build that caps
/// SIMD runs neither the detection nor the checks.
///
/// In a child forked while another thread was inside the detection or the
/// checks, before they stored their result, the GEMM and GEMV calls never
/// use SME2 and never wait, and this function would wait forever. A program
/// that forks should therefore call this function once before it forks,
/// from a thread that is ready for SME code: when it returns, the detection
/// and the checks are settled as far as this process runs them at all (see
/// above), so the child inherits their results and never waits for them,
/// whatever other threads do. Called from a thread that is not ready, it
/// may return before the checks have run; the fork is then safe only when
/// no other thread is in a GEMM, GEMV or `sme2_active` call, or starts
/// one, between this call and the fork (the usual rule for forking a
/// multi-threaded process).
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
