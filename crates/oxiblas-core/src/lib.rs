//! OxiBLAS Core - Foundational types and traits for OxiBLAS.
//!
//! This crate provides the core infrastructure for OxiBLAS:
//!
//! - **Scalar traits**: `Scalar`, `Real`, `ComplexScalar`, `Field` for numeric types
//! - **SIMD abstraction**: Custom SIMD layer via `core::arch` intrinsics
//! - **Memory management**: Aligned allocation, stack-based temporaries
//! - **Parallelization**: Work partitioning and parallel execution
//!
//! # Supported Types
//!
//! - `f32`, `f64`: Real floating-point numbers
//! - `Complex32`, `Complex64`: Complex numbers (via `num-complex`)
//!
//! # SIMD Support
//!
//! The SIMD abstraction automatically detects and uses the best available
//! instruction set:
//!
//! - **x86_64**: AVX2 (256-bit), AVX512F (512-bit)
//! - **AArch64**: NEON (128-bit), with 256-bit emulation
//! - **Fallback**: Scalar operations for unsupported platforms
//!
//! # Example
//!
//! ```
//! use oxiblas_core::scalar::{Scalar, Field};
//! use oxiblas_core::simd::detect_simd_level;
//!
//! // Check SIMD capability
//! let level = detect_simd_level();
//! println!("SIMD level: {:?}", level);
//!
//! // Use scalar traits
//! let x: f64 = 3.0;
//! let y: f64 = 4.0;
//! assert_eq!(x.abs_sq() + y.abs_sq(), 25.0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

pub mod blocking;
pub mod memory;
pub mod parallel;
pub mod scalar;
pub mod simd;
pub mod tuning;

// Re-exports for convenience
pub use blocking::{
    BASE_CASE_THRESHOLD, BlockRange, BlockVisitor, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, RecursiveTask,
    cache_oblivious_traverse, factorization_panel_width, gemm_block_sizes, morton_decode,
    morton_index, trsm_block_size,
};
pub use memory::{
    AlignedPool, AlignedVec, Alloc, CACHE_LINE_SIZE, DEFAULT_ALIGN, Global, MemStack, MemoryPool,
    NumaAllocHint, NumaInterleavingStrategy, NumaTopology, NumaWorkHint, PrefetchDistance,
    PrefetchLocality, StackReq, get_huge_page_size, get_page_size, numa_alloc, numa_alloc_zeroed,
    numa_distribute_work, prefetch_read, prefetch_read_range, prefetch_write, prefetch_write_range,
};
#[cfg(feature = "parallel")]
pub use parallel::{CustomRayonPool, RayonGlobalPool};
pub use parallel::{
    Par, ParThreshold, PoolScope, SequentialPool, ThreadPool, WorkRange, default_pool,
    for_each_indexed, for_each_range, map_reduce, partition_work, with_default_pool,
};
pub use scalar::{
    C32, C64, ComplexExt, ComplexScalar, ExtendedPrecision, Field, HasFastFma, I32, I64, KBKSum,
    KahanSum, Real, Scalar, ScalarBatch, ScalarClass, ScalarClassify, SimdCompatible, ToComplex,
    UnrollHints, c32, c64, from_polar, from_polar32, imag, imag_unit, imag_unit32, imag32,
    pairwise_sum, real, real32,
};
pub use simd::{
    SimdChunks, SimdLevel, SimdRegister, SimdScalar, detect_simd_level, detect_simd_level_raw,
};
pub use tuning::{AutoTuner, TuningCache, TuningConfig};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::blocking::{
        BlockRange, BlockVisitor, RecursiveTask, cache_oblivious_traverse, gemm_block_sizes,
    };
    pub use crate::memory::{
        AlignedPool, AlignedVec, MemStack, MemoryPool, NumaAllocHint, NumaTopology,
        PrefetchLocality, StackReq, numa_distribute_work, prefetch_read, prefetch_write,
    };
    pub use crate::parallel::{Par, ParThreshold};
    pub use crate::scalar::{
        C32, C64, ComplexExt, ComplexScalar, ExtendedPrecision, Field, HasFastFma, I32, I64,
        KBKSum, KahanSum, Real, Scalar, ScalarBatch, ScalarClass, ScalarClassify, SimdCompatible,
        ToComplex, UnrollHints, c32, c64, imag, pairwise_sum, real,
    };
    pub use crate::simd::{SimdChunks, SimdLevel, SimdRegister, SimdScalar, detect_simd_level};
    pub use crate::tuning::{AutoTuner, TuningCache, TuningConfig};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {:?}", level);

        // When force-scalar is enabled, should be Scalar
        #[cfg(feature = "force-scalar")]
        assert_eq!(level, SimdLevel::Scalar);

        // Without force-scalar, should detect hardware SIMD on common platforms
        #[cfg(not(feature = "force-scalar"))]
        {
            #[cfg(target_arch = "x86_64")]
            assert!(level >= SimdLevel::Simd128);

            #[cfg(target_arch = "aarch64")]
            assert!(level >= SimdLevel::Simd128);
        }
    }

    #[test]
    fn test_scalar_traits() {
        use num_complex::Complex64;

        let x: f64 = -3.0;
        assert_eq!(x.abs(), 3.0);
        assert_eq!(x.conj(), -3.0);
        assert!(f64::is_real());

        let z = Complex64::new(3.0, 4.0);
        assert!((z.abs() - 5.0).abs() < 1e-10);
        assert!(!Complex64::is_real());
    }

    #[test]
    fn test_aligned_alloc() {
        let vec: AlignedVec<f64> = AlignedVec::zeros(100);
        assert_eq!(vec.len(), 100);

        // Check alignment
        let ptr = vec.as_ptr();
        assert_eq!(ptr as usize % DEFAULT_ALIGN, 0);
    }
}
