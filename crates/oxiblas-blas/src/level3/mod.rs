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

pub mod autotune;
pub mod complex_gemm;
pub mod gemm;
pub mod gemm_cache_oblivious;
pub mod gemm_kernel;
pub mod gemm_packing;
pub mod gemm_small;
pub mod gemm_winograd;
pub mod gemmt;
pub mod hemm;
pub mod her2k;
pub mod herk;
pub mod strassen;
pub mod symm;
pub mod syr2k;
pub mod syrk;
pub mod trmm;
pub mod trsm;

pub use autotune::{
    AutoTunedBlocking, CacheInfo, compute_blocking, compute_blocking_adaptive, get_cache_info,
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
    trsm_in_place,
};
