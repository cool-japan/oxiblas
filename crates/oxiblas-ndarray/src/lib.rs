//! OxiBLAS ndarray Integration
//!
//! This crate provides seamless integration between OxiBLAS and the `ndarray` crate,
//! allowing you to use OxiBLAS BLAS and LAPACK operations directly on ndarray types.
//!
//! # Features
//!
//! - **Conversions**: Efficient conversion between ndarray and OxiBLAS matrix types
//! - **BLAS Operations**: Level 1-3 BLAS operations (dot, gemv, gemm, etc.)
//! - **LAPACK Operations**: Decompositions (LU, QR, SVD, EVD, Cholesky)
//! - **Linear Solve**: Direct and iterative solvers
//!
//! # Quick Start
//!
//! ```
//! use ndarray::{array, Array2};
//! use oxiblas_ndarray::prelude::*;
//!
//! // Matrix multiplication
//! let a = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j + 1) as f64);
//! let b = Array2::from_shape_fn((3, 2), |(i, j)| (i * 2 + j + 1) as f64);
//! let c = matmul(&a, &b);
//! assert_eq!(c.dim(), (2, 2));
//!
//! // Matrix-vector multiplication
//! let x = array![1.0f64, 2.0, 3.0];
//! let y = matvec(&a, &x);
//! assert_eq!(y.len(), 2);
//!
//! // Dot product
//! let v1 = array![1.0f64, 2.0, 3.0];
//! let v2 = array![4.0f64, 5.0, 6.0];
//! let d = dot_ndarray(&v1, &v2);
//! assert!((d - 32.0).abs() < 1e-10);
//! ```
//!
//! # LAPACK Operations
//!
//! ```
//! use ndarray::array;
//! use oxiblas_ndarray::prelude::*;
//!
//! // Solve linear system
//! let a = array![[2.0f64, 1.0], [1.0, 3.0]];
//! let b = array![5.0f64, 7.0];
//! let x = solve_ndarray(&a, &b).unwrap();
//!
//! // LU decomposition
//! let lu = lu_ndarray(&a).unwrap();
//! let det = lu.det();  // Determinant
//!
//! // QR decomposition
//! let qr = qr_ndarray(&a).unwrap();
//!
//! // SVD
//! let svd = svd_ndarray(&a).unwrap();
//!
//! // Symmetric eigenvalue decomposition
//! let evd = eig_symmetric(&a).unwrap();
//! ```
//!
//! # Memory Layout
//!
//! OxiBLAS uses column-major (Fortran) order internally. This crate handles
//! both row-major and column-major ndarray layouts:
//!
//! - **Column-major arrays**: Zero-copy or minimal-copy operations
//! - **Row-major arrays**: Automatic conversion (with copy) when needed
//!
//! For best performance, use column-major arrays when possible:
//!
//! ```
//! use ndarray::{Array2, ShapeBuilder};
//! use oxiblas_ndarray::prelude::*;
//!
//! // Create column-major array (preferred)
//! let a: Array2<f64> = zeros_f(100, 100);
//! assert!(is_column_major(&a));
//!
//! // Or convert existing row-major array
//! let row_major = Array2::<f64>::zeros((100, 100));
//! let col_major = to_column_major(&row_major);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
// Loop index variables are common in matrix operations
#![allow(clippy::needless_range_loop)]
// Bounds in two places for clarity
#![allow(clippy::multiple_bound_locations)]

pub mod blas;
pub mod conversions;
pub mod lapack;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "sparse")]
pub mod sparse;

// Re-export core types from ndarray for convenience
pub use ndarray::{
    Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMutD, IxDyn,
};

// Re-export complex types for convenience
pub use num_complex::{Complex32, Complex64};

/// Prelude module for convenient imports.
///
/// Import all commonly used functions with:
/// ```
/// use oxiblas_ndarray::prelude::*;
/// ```
pub mod prelude {
    // Conversions (Array2)
    pub use crate::conversions::{
        array_view_mut_to_mat_mut, array_view_to_mat_ref, array_view_to_mat_ref_or_transposed,
        array_view1_as_slice, array_view1_as_slice_mut, array1_to_vec, array2_into_mat,
        array2_to_mat, filled_f, is_column_major, is_row_major, mat_ref_to_array2, mat_to_array2,
        mat_to_array2_c, slice_to_array1, to_column_major, zeros_f,
    };

    // Conversions (ArrayD - Dynamic dimension)
    pub use crate::conversions::{
        array_view_mutd_to_mat_mut, array_viewd_to_mat_ref, array_viewd_to_mat_ref_or_transposed,
        array2_to_arrayd, arrayd_into_mat, arrayd_to_array2, arrayd_to_mat, mat_ref_to_arrayd,
        mat_to_arrayd,
    };

    // BLAS Level 1
    pub use crate::blas::{
        asum_ndarray, axpy_ndarray, dot_ndarray, dot_view, nrm2_ndarray, scal_ndarray,
    };

    // BLAS Level 1 - Complex
    pub use crate::blas::{
        asum_c32_ndarray, asum_c64_ndarray, axpy_c32_ndarray, axpy_c64_ndarray, dotc_c32_ndarray,
        dotc_c64_ndarray, dotu_c32_ndarray, dotu_c64_ndarray, nrm2_c32_ndarray, nrm2_c64_ndarray,
        scal_c32_ndarray, scal_c64_ndarray,
    };

    // BLAS Level 2
    pub use crate::blas::{Transpose, gemv_ndarray, matvec, matvec_t};

    // BLAS Level 3
    pub use crate::blas::{gemm_ndarray, matmul, matmul_c, matmul_into};

    // Matrix norms
    pub use crate::blas::{frobenius_norm, norm_1, norm_inf, norm_max};

    // Matrix norms - Complex
    pub use crate::blas::{
        frobenius_norm_c32, frobenius_norm_c64, norm_1_c32, norm_1_c64, norm_inf_c32, norm_inf_c64,
        norm_max_c32, norm_max_c64,
    };

    // Utilities
    pub use crate::blas::{eye, eye_f, trace, transpose};

    // Utilities - Complex
    pub use crate::blas::{
        conj_transpose_c32, conj_transpose_c64, eye_c32, eye_c64, trace_c32, trace_c64,
    };

    // LAPACK decompositions
    pub use crate::lapack::{
        CholeskyResult, LuResult, QrResult, SvdResult, SymEvdResult, cholesky_ndarray,
        eig_symmetric, eigvals_symmetric, lu_ndarray, qr_ndarray, svd_ndarray, svd_truncated,
    };

    // LAPACK - Randomized SVD
    pub use crate::lapack::{
        RandomizedSvdResult, low_rank_approx_ndarray, rsvd_ndarray, rsvd_power_ndarray,
    };

    // LAPACK - Schur decomposition
    pub use crate::lapack::{SchurResult, schur_ndarray};

    // LAPACK - General eigenvalue decomposition
    pub use crate::lapack::{Eigenvalue, GeneralEvdResult, eig_ndarray, eigvals_ndarray};

    // LAPACK - Tridiagonal solvers
    pub use crate::lapack::{
        tridiag_solve_multiple_ndarray, tridiag_solve_ndarray, tridiag_solve_spd_ndarray,
    };

    // LAPACK solvers
    pub use crate::lapack::{lstsq_ndarray, solve_multiple_ndarray, solve_ndarray};

    // Matrix operations
    pub use crate::lapack::{cond_ndarray, det_ndarray, inv_ndarray, pinv_ndarray, rank_ndarray};

    // Error types
    pub use crate::lapack::{LapackError, LapackResult};

    // Parallel BLAS operations
    #[cfg(feature = "parallel")]
    pub use crate::parallel::{gemm_par_ndarray, matmul_par};

    // Sparse integration
    #[cfg(feature = "sparse")]
    pub use crate::sparse::{
        SparseNdarrayError, array2_to_csc, array2_to_csr, csc_to_array2, csr_to_array2,
        sparse_solve_ndarray, sparse_solve_ndarray_with_options, spmv_full_ndarray, spmv_ndarray,
    };
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_full_workflow() {
        // Create matrices
        let a = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j + 1) as f64);
        let b = Array2::from_shape_fn((3, 3), |(i, j)| ((i + j) % 3 + 1) as f64);

        // Matrix multiplication
        let c = matmul(&a, &b);
        assert_eq!(c.dim(), (3, 3));

        // Matrix-vector multiplication
        let x = array![1.0f64, 2.0, 3.0];
        let y = matvec(&a, &x);
        assert_eq!(y.len(), 3);

        // Norms
        let fnorm = frobenius_norm(&a);
        assert!(fnorm > 0.0);

        // Solve linear system
        let symmetric = array![[4.0f64, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]];
        let rhs = array![5.0f64, 6.0, 5.0];
        let solution = solve_ndarray(&symmetric, &rhs).unwrap();
        assert_eq!(solution.len(), 3);

        // Verify solution
        let residual = matvec(&symmetric, &solution);
        for i in 0..3 {
            assert!((residual[i] - rhs[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decomposition_workflow() {
        let a = array![[4.0f64, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]];

        // LU decomposition
        let lu = lu_ndarray(&a).unwrap();
        let det = lu.det();
        assert!(det.abs() > 1e-10); // Non-singular

        // QR decomposition
        let qr = qr_ndarray(&a).unwrap();
        assert_eq!(qr.q.dim().0, 3);
        assert_eq!(qr.r.dim().1, 3);

        // SVD
        let svd = svd_ndarray(&a).unwrap();
        assert_eq!(svd.s.len(), 3);

        // Symmetric EVD (a is symmetric)
        let evd = eig_symmetric(&a).unwrap();
        assert_eq!(evd.eigenvalues.len(), 3);

        // Cholesky (a is positive definite)
        let chol = cholesky_ndarray(&a).unwrap();
        assert_eq!(chol.l.dim(), (3, 3));
    }

    #[test]
    fn test_blas_operations() {
        // Level 1
        let x = array![1.0f64, 2.0, 3.0];
        let y = array![4.0f64, 5.0, 6.0];

        let d = dot_ndarray(&x, &y);
        assert!((d - 32.0).abs() < 1e-10);

        let n = nrm2_ndarray(&x);
        assert!((n - 14.0f64.sqrt()).abs() < 1e-10);

        let s = asum_ndarray(&x);
        assert!((s - 6.0).abs() < 1e-10);

        // Level 2
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let v = array![1.0f64, 1.0];
        let result = matvec(&a, &v);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);

        // Level 3
        let b = array![[1.0f64, 0.0], [0.0, 1.0]];
        let c = matmul(&a, &b);
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_column_major_efficiency() {
        // Column-major arrays should be detected correctly
        let col_major: Array2<f64> = zeros_f(10, 10);
        assert!(is_column_major(&col_major));

        let row_major: Array2<f64> = Array2::zeros((10, 10));
        assert!(!is_column_major(&row_major));
        assert!(is_row_major(&row_major));

        // Conversion should work
        let converted = to_column_major(&row_major);
        assert!(is_column_major(&converted));
    }

    #[test]
    fn test_identity_operations() {
        let id: Array2<f64> = eye(3);

        // Trace of identity = n
        let tr = trace(&id);
        assert!((tr - 3.0).abs() < 1e-15);

        // Frobenius norm of identity = sqrt(n)
        let fnorm = frobenius_norm(&id);
        assert!((fnorm - 3.0f64.sqrt()).abs() < 1e-15);

        // Determinant of identity = 1
        let det = det_ndarray(&id).unwrap();
        assert!((det - 1.0).abs() < 1e-10);

        // Inverse of identity = identity
        let inv = inv_ndarray(&id).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-10);
            }
        }

        // Condition number of identity = 1
        let cond = cond_ndarray(&id).unwrap();
        assert!((cond - 1.0).abs() < 1e-10);

        // Rank of identity = n
        let r = rank_ndarray(&id).unwrap();
        assert_eq!(r, 3);
    }

    #[test]
    fn test_large_matrices() {
        let n = 100;
        let a: Array2<f64> = zeros_f(n, n);
        let mut a = a.mapv(|_| 0.0);

        // Create a well-conditioned matrix
        for i in 0..n {
            a[[i, i]] = 10.0;
            if i > 0 {
                a[[i, i - 1]] = 1.0;
            }
            if i < n - 1 {
                a[[i, i + 1]] = 1.0;
            }
        }

        // Test matrix-vector multiplication
        let x: ndarray::Array1<f64> = ndarray::Array1::ones(n);
        let y = matvec(&a, &x);
        assert_eq!(y.len(), n);

        // Test matrix multiplication
        let id: Array2<f64> = eye(n);
        let c = matmul(&a, &id);
        for i in 0..n {
            for j in 0..n {
                assert!((c[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_numerical_accuracy() {
        // Test with Hilbert matrix (ill-conditioned)
        let n = 5;
        let mut h: Array2<f64> = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] = 1.0 / ((i + j + 1) as f64);
            }
        }

        // SVD should still work
        let svd = svd_ndarray(&h).unwrap();
        assert_eq!(svd.s.len(), n);

        // All singular values should be positive
        for s in svd.s.iter() {
            assert!(*s > 0.0);
        }

        // Condition number should be large (ill-conditioned)
        let cond = cond_ndarray(&h).unwrap();
        assert!(cond > 1000.0);
    }

    #[test]
    fn test_arrayd_integration() {
        use ndarray::{ArrayD, IxDyn};

        // Create a 2D ArrayD
        let arr_d = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] * 4 + idx[1]) as f64);

        // Convert to Mat and back
        let mat = arrayd_to_mat(&arr_d);
        assert_eq!(mat.shape(), (3, 4));

        let recovered = mat_to_arrayd(&mat);
        assert_eq!(recovered.shape(), &[3, 4]);

        // Verify values
        for i in 0..3 {
            for j in 0..4 {
                assert!((arr_d[[i, j].as_ref()] - recovered[[i, j].as_ref()]).abs() < 1e-15);
            }
        }

        // Convert to Array2 for BLAS operations
        let arr_2 = arrayd_to_array2(&arr_d);
        let fnorm = frobenius_norm(&arr_2);
        assert!(fnorm > 0.0);

        // Convert back to ArrayD
        let result_d = array2_to_arrayd(&arr_2);
        assert_eq!(result_d.shape(), &[3, 4]);
    }

    #[test]
    fn test_arrayd_matrix_operations() {
        use ndarray::{ArrayD, IxDyn};

        // Create two ArrayD matrices
        let a = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] + idx[1] + 1) as f64);
        let b = ArrayD::from_shape_fn(IxDyn(&[4, 2]), |idx| (idx[0] * idx[1] + 1) as f64);

        // Convert to Array2 for multiplication
        let a2 = arrayd_to_array2(&a);
        let b2 = arrayd_to_array2(&b);

        let c2 = matmul(&a2, &b2);
        assert_eq!(c2.dim(), (3, 2));

        // Convert result back to ArrayD
        let c_d = array2_to_arrayd(&c2);
        assert_eq!(c_d.shape(), &[3, 2]);
    }
}
