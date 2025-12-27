//! `OxiBLAS` BLAS - Pure Rust BLAS implementation.
//!
//! This crate provides BLAS (Basic Linear Algebra Subprograms) operations
//! implemented in pure Rust with SIMD optimization.
//!
//! # BLAS Levels
//!
//! - **Level 1**: Vector-vector operations (dot, axpy, nrm2, etc.)
//! - **Level 2**: Matrix-vector operations (gemv, trmv, etc.)
//! - **Level 3**: Matrix-matrix operations (gemm, trmm, etc.)
//!
//! # Example
//!
//! ```
//! use oxiblas_blas::level3::gemm;
//! use oxiblas_matrix::Mat;
//!
//! // Create matrices
//! let a: Mat<f64> = Mat::filled(100, 50, 1.0);
//! let b: Mat<f64> = Mat::filled(50, 80, 2.0);
//! let mut c: Mat<f64> = Mat::zeros(100, 80);
//!
//! // GEMM: C = A * B
//! gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
// Stylistic choices for BLAS library
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// BLAS uses Self vs typename interchangeably
#![allow(clippy::use_self)]
// Constants defined close to usage is clearer for BLAS
#![allow(clippy::items_after_statements)]
// Technical terms (OxiBLAS, GEMM, etc.) don't need backticks
#![allow(clippy::doc_markdown)]
// BLAS functions have well-known semantics
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
// Not critical for performance library
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
// Sometimes makes match arms more explicit
#![allow(clippy::match_same_arms)]
// Explicit lifetimes can be clearer
#![allow(clippy::needless_lifetimes)]
// Index-based loops common in BLAS
#![allow(clippy::needless_range_loop)]
// Not all functions need const
#![allow(clippy::missing_const_for_fn)]
// API consistency with Result/Option
#![allow(clippy::unnecessary_wraps)]
// Raw pointer casting common in SIMD code
#![allow(clippy::ptr_as_ptr)]
// SIMD code uses transmute
#![allow(clippy::transmute_ptr_to_ref)]
// Casting in BLAS is intentional
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
// CBLAS extern functions have well-known semantics
#![allow(clippy::missing_safety_doc)]
// Manual assign clearer for BLAS code
#![allow(clippy::assign_op_pattern)]
// Transmute in SIMD code is intentional
#![allow(clippy::transmute_undefined_repr)]
#![allow(clippy::missing_transmute_annotations)]
// SIMD kernels benefit from inline(always)
#![allow(clippy::inline_always)]
// Some refs are cfg-gated
#![allow(clippy::needless_pass_by_ref_mut)]
// Small types passed by value intentionally
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::needless_pass_by_value)]
// Sometimes makes conditional code clearer
#![allow(clippy::if_same_then_else)]
#![allow(clippy::branches_sharing_code)]
// Strict float comparison sometimes needed
#![allow(clippy::float_cmp)]
// Older Rust compatible code
#![allow(clippy::manual_div_ceil)]
// Manual copy for specific layouts
#![allow(clippy::manual_memcpy)]
// Unused variable patterns are intentional
#![allow(clippy::no_effect_underscore_binding)]

pub mod accuracy;
pub mod cblas;
pub mod complex_interleaved;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod ndtensor;
pub mod tensor;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::level1::{asum, axpy, copy, dot, iamax, iamin, nrm2, nrm2_sq, scal, swap};
    pub use crate::level2::{
        DiagKind,
        GemvTrans,
        HerError,
        HerUplo,
        SyrError,
        SyrUplo,
        TriangularMode,
        TriangularSide,
        TrmvError,
        TrmvOp,
        TrmvUplo,
        gemv,
        gemv_simple,
        ger,
        gerc,
        her,
        her_new,
        // Symmetric/Hermitian rank-1 updates
        syr,
        syr_new,
        // Triangular matrix-vector multiply
        trmv,
        trmv_alloc,
        trsv,
        trsv_in_place,
    };
    pub use crate::level3::{
        Diag,
        GemmBlocking,
        GemmKernel,
        Her2kError,
        HerkError,
        Side,
        Syr2kError,
        SyrkError,
        Trans,
        TrmmDiag,
        TrmmError,
        TrmmSide,
        TrmmTrans,
        TrmmUplo,
        Uplo,
        gemm,
        gemm_with_par,
        her2k,
        her2k_new,
        herk,
        herk_new,
        syr2k,
        syr2k_new,
        syrk,
        syrk_new,
        // Triangular matrix-matrix multiply
        trmm,
        trmm_in_place,
        trsm,
    };
    pub use crate::ndtensor::{NdTensor, NdTensorError, Order};
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_gemm_correctness() {
        // Test with known values
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let b: Mat<f64> = Mat::from_rows(&[&[9.0, 8.0, 7.0], &[6.0, 5.0, 4.0], &[3.0, 2.0, 1.0]]);

        let mut c: Mat<f64> = Mat::zeros(3, 3);

        level3::gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Expected result:
        // C[0,0] = 1*9 + 2*6 + 3*3 = 9 + 12 + 9 = 30
        // C[0,1] = 1*8 + 2*5 + 3*2 = 8 + 10 + 6 = 24
        // C[0,2] = 1*7 + 2*4 + 3*1 = 7 + 8 + 3 = 18
        // C[1,0] = 4*9 + 5*6 + 6*3 = 36 + 30 + 18 = 84
        // etc.

        assert!((c[(0, 0)] - 30.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 24.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 18.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 84.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm_non_square() {
        let a: Mat<f64> = Mat::filled(10, 20, 1.0);
        let b: Mat<f64> = Mat::filled(20, 15, 1.0);
        let mut c: Mat<f64> = Mat::zeros(10, 15);

        level3::gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Each element should be 20 (sum of 20 ones)
        for i in 0..10 {
            for j in 0..15 {
                assert!((c[(i, j)] - 20.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_large() {
        let n = 128;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        level3::gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Each element should be n
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - n as f64).abs() < 1e-8,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    n
                );
            }
        }
    }
}
