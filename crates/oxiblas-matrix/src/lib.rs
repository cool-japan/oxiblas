//! OxiBLAS Matrix - Matrix types and views for OxiBLAS.
//!
//! This crate provides the core matrix types for OxiBLAS:
//!
//! - [`Mat<T>`]: Owned, heap-allocated matrix with column-major storage
//! - [`CowMat<T>`]: Copy-on-Write matrix for efficient sharing
//! - [`MatRef<'a, T>`]: Immutable view into a matrix
//! - [`MatMut<'a, T>`]: Mutable view with reborrow semantics
//!
//! # Specialized Matrix Types
//!
//! - [`packed::PackedMat<T>`]: Packed triangular/symmetric storage
//! - [`banded::BandedMat<T>`]: Banded matrix storage
//! - [`triangular::TriangularMat<T>`]: Triangular matrix with packed storage
//! - [`symmetric::SymmetricMat<T>`]: Symmetric matrix with packed storage
//! - [`symmetric::HermitianMat<T>`]: Hermitian matrix with packed storage
//!
//! # Memory Layout
//!
//! All matrices use column-major (Fortran) storage order. This means elements
//! within a column are contiguous in memory. The storage is aligned for
//! efficient SIMD operations.
//!
//! # Views and Reborrows
//!
//! The view types (`MatRef`, `MatMut`) do not own their data. They can
//! represent submatrices, transposed views, or any strided data.
//!
//! `MatMut` uses the reborrow pattern to prevent aliasing:
//! - `rb()` creates an immutable reborrow
//! - `rb_mut()` creates a mutable reborrow with a shorter lifetime
//!
//! # Example
//!
//! ```
//! use oxiblas_matrix::{Mat, MatRef, MatMut};
//!
//! // Create a matrix
//! let mut m: Mat<f64> = Mat::zeros(4, 4);
//!
//! // Modify through a mutable view
//! {
//!     let mut view = m.as_mut();
//!     view[(0, 0)] = 1.0;
//!     view[(1, 1)] = 2.0;
//!     view[(2, 2)] = 3.0;
//!     view[(3, 3)] = 4.0;
//! }
//!
//! // Read through an immutable view
//! let view = m.as_ref();
//! assert_eq!(view.diagonal()[2], 3.0);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::must_use_candidate)]
// Prefetch functions use raw pointer arithmetic but don't dereference
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// Loop index variables are common in matrix operations
#![allow(clippy::needless_range_loop)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Core matrix types
pub mod mat;
pub mod mat_mut;
pub mod mat_ref;

// Specialized storage formats
pub mod banded;
pub mod cow;
pub mod lazy;
pub mod ops;
pub mod packed;
pub mod prefetch;
pub mod symmetric;
pub mod triangular;

// Memory-mapped matrices (requires mmap feature)
#[cfg(feature = "mmap")]
pub mod mmap;

// nalgebra interoperability (requires nalgebra feature)
#[cfg(feature = "nalgebra")]
pub mod nalgebra_compat;

// Re-exports - allocator support
pub use oxiblas_core::memory::{Alloc, Global};

// Re-exports - core types
pub use cow::CowMat;
pub use mat::Mat;
pub use mat_mut::MatMut;
pub use mat_ref::{DiagRef, MatRef, TransposeRef};

// Re-exports - specialized types
pub use banded::{BandedMat, BandedMut, BandedRef, SymmetricBandedMat};
pub use packed::{PackedMat, PackedMut, PackedRef, TriangularKind};
pub use prefetch::{
    CACHE_LINE_SIZE, MatrixPrefetcher, PREFETCH_DISTANCE_BYTES, PREFETCH_DISTANCE_LINES,
    PrefetchLocality, prefetch_block, prefetch_column, prefetch_range_read, prefetch_range_write,
    prefetch_read, prefetch_write,
};
pub use symmetric::{
    HermitianMat, SymmetricMat, SymmetricMut, SymmetricRef, SymmetricView, SymmetricViewMut,
};
pub use triangular::{
    DiagonalKind, TriangularMat, TriangularMut, TriangularRef, TriangularView, TriangularViewMut,
};

// Re-exports - lazy evaluation
pub use lazy::{
    ComplexExpr, ComplexScalar, Expr, ExprAdd, ExprConj, ExprFma, ExprGemm, ExprHermitian,
    ExprLeaf, ExprMul, ExprNeg, ExprScale, ExprSub, ExprTranspose, LazyExt, fma, gemm,
};

// Re-exports - memory-mapped matrices
#[cfg(feature = "mmap")]
pub use mmap::{
    MmapBuilder, MmapError, MmapMat, MmapMatMut, read_dimensions, write_mat as write_mmap,
};

// Re-exports - nalgebra conversions
#[cfg(feature = "nalgebra")]
pub use nalgebra_compat::{
    DMatrixOxiblasExt, MatNalgebraExt, dmatrix_to_mat, dmatrix_view_to_mat, dvector_to_mat,
    mat_ref_to_dmatrix, mat_to_dmatrix, mat_to_dvector,
};

/// Prelude module for convenient imports.
pub mod prelude {
    // Allocator support
    pub use crate::{Alloc, Global};

    // Core types
    pub use crate::cow::CowMat;
    pub use crate::mat::Mat;
    pub use crate::mat_mut::MatMut;
    pub use crate::mat_ref::{DiagRef, MatRef, TransposeRef};

    // Specialized storage
    pub use crate::banded::{BandedMat, SymmetricBandedMat};
    pub use crate::packed::{PackedMat, TriangularKind};
    pub use crate::symmetric::{HermitianMat, SymmetricMat};
    pub use crate::triangular::{DiagonalKind, TriangularMat};

    // Lazy evaluation
    pub use crate::lazy::{ComplexExpr, Expr, LazyExt, fma as lazy_fma, gemm as lazy_gemm};

    // Operations
    pub use crate::ops;

    // Performance utilities (only MatrixPrefetcher is unique to oxiblas-matrix)
    // PrefetchLocality and prefetch_* are in oxiblas-core
    pub use crate::prefetch::MatrixPrefetcher;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_workflow() {
        // Create a 3x3 matrix
        let mut m: Mat<f64> = Mat::zeros(3, 3);

        // Fill diagonal
        for i in 0..3 {
            m[(i, i)] = (i + 1) as f64;
        }

        // Check via view
        let view = m.as_ref();
        assert_eq!(view[(0, 0)], 1.0);
        assert_eq!(view[(1, 1)], 2.0);
        assert_eq!(view[(2, 2)], 3.0);

        // Modify via mutable view
        {
            let mut view = m.as_mut();
            view[(0, 1)] = 10.0;
        }

        assert_eq!(m[(0, 1)], 10.0);
    }

    #[test]
    fn test_submatrix_views() {
        let m: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
        ]);

        // Get a 2x2 submatrix from the center
        let sub = m.as_ref().submatrix(1, 1, 2, 2);
        assert_eq!(sub[(0, 0)], 6.0);
        assert_eq!(sub[(0, 1)], 7.0);
        assert_eq!(sub[(1, 0)], 10.0);
        assert_eq!(sub[(1, 1)], 11.0);
    }

    #[test]
    fn test_complex_matrix() {
        use num_complex::Complex64;

        // Use filled() instead of zeros() since Complex64 doesn't implement bytemuck::Zeroable
        let mut m: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        m[(0, 0)] = Complex64::new(1.0, 2.0);
        m[(1, 1)] = Complex64::new(3.0, 4.0);

        assert_eq!(m[(0, 0)].re, 1.0);
        assert_eq!(m[(0, 0)].im, 2.0);
        assert_eq!(m[(1, 1)].re, 3.0);
        assert_eq!(m[(1, 1)].im, 4.0);
    }
}
