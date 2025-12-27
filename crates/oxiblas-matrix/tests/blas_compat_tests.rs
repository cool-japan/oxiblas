//! BLAS layout compatibility tests.
//!
//! These tests verify that oxiblas-matrix types are compatible with BLAS conventions:
//! - Column-major storage order
//! - Proper leading dimension (lda) handling
//! - Correct stride patterns
//! - Packed storage formats match BLAS expectations

use oxiblas_matrix::{
    Mat,
    banded::BandedMat,
    packed::{PackedMat, TriangularKind},
};

// ============================================================================
// Column-major layout tests
// ============================================================================

#[test]
fn test_column_major_storage_order() {
    // In column-major (Fortran) order:
    // A = [1 3]   is stored as [1, 2, 3, 4]
    //     [2 4]

    let m = Mat::from_rows(&[&[1.0, 3.0], &[2.0, 4.0]]);

    // Access elements to verify column-major order
    assert_eq!(m[(0, 0)], 1.0); // Row 0, Col 0
    assert_eq!(m[(1, 0)], 2.0); // Row 1, Col 0
    assert_eq!(m[(0, 1)], 3.0); // Row 0, Col 1
    assert_eq!(m[(1, 1)], 4.0); // Row 1, Col 1

    // Get raw pointer access
    let ptr = m.as_ptr();
    unsafe {
        // In column-major, consecutive elements in a column are contiguous
        // Column 0: elements at indices 0, 1
        // Column 1: elements at indices lda, lda+1
        let lda = m.row_stride();

        assert_eq!(*ptr.add(0), 1.0);
        assert_eq!(*ptr.add(1), 2.0);
        assert_eq!(*ptr.add(lda), 3.0);
        assert_eq!(*ptr.add(lda + 1), 4.0);
    }
}

#[test]
fn test_leading_dimension_contiguous_columns() {
    let m: Mat<f64> = Mat::zeros(4, 3);
    let view = m.as_ref();

    // Leading dimension (lda) should be >= nrows
    assert!(view.row_stride() >= view.nrows());

    // Each column should be contiguous in memory
    // Column j starts at ptr + j * lda
    let lda = view.row_stride();
    let ptr = view.as_ptr();

    for j in 0..view.ncols() {
        for i in 0..view.nrows() {
            let expected_offset = j * lda + i;
            let actual_ptr = view.ptr_at(i, j);
            unsafe {
                assert_eq!(actual_ptr, ptr.add(expected_offset));
            }
        }
    }
}

#[test]
fn test_submatrix_stride_preservation() {
    let mut m: Mat<f64> = Mat::zeros(6, 6);

    // Fill with known values
    for i in 0..6 {
        for j in 0..6 {
            m[(i, j)] = (i * 10 + j) as f64;
        }
    }

    // Get a 3x3 submatrix starting at (1, 2)
    let sub = m.as_ref().submatrix(1, 2, 3, 3);

    // Verify submatrix values
    assert_eq!(sub[(0, 0)], 12.0); // m[1,2]
    assert_eq!(sub[(1, 0)], 22.0); // m[2,2]
    assert_eq!(sub[(2, 0)], 32.0); // m[3,2]
    assert_eq!(sub[(0, 1)], 13.0); // m[1,3]
    assert_eq!(sub[(0, 2)], 14.0); // m[1,4]

    // Submatrix should have same row_stride as parent
    assert_eq!(sub.row_stride(), m.row_stride());
}

#[test]
fn test_transpose_view_stride() {
    let m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

    let t = m.as_ref().transpose();

    // Transposed view should swap row and column counts
    assert_eq!(t.nrows(), m.ncols());
    assert_eq!(t.ncols(), m.nrows());

    // Verify transposed access
    assert_eq!(t[(0, 0)], 1.0);
    assert_eq!(t[(0, 1)], 4.0);
    assert_eq!(t[(1, 0)], 2.0);
    assert_eq!(t[(1, 1)], 5.0);
    assert_eq!(t[(2, 0)], 3.0);
    assert_eq!(t[(2, 1)], 6.0);
}

// ============================================================================
// Packed storage format tests (BLAS compatible)
// ============================================================================

#[test]
fn test_packed_upper_blas_format() {
    // BLAS upper triangular packed format (column-major):
    // For 3x3 matrix A:
    // A = [a b d]    packed as [a, b, c, d, e, f]
    //     [  c e]    indices:   0  1  2  3  4  5
    //     [    f]

    let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Upper);

    // Set elements
    p.set(0, 0, 1.0); // a
    p.set(0, 1, 2.0); // b
    p.set(1, 1, 3.0); // c
    p.set(0, 2, 4.0); // d
    p.set(1, 2, 5.0); // e
    p.set(2, 2, 6.0); // f

    // Verify packed storage follows BLAS convention
    let storage = p.as_slice();
    assert_eq!(storage[0], 1.0); // a: (0,0)
    assert_eq!(storage[1], 2.0); // b: (0,1)
    assert_eq!(storage[2], 3.0); // c: (1,1)
    assert_eq!(storage[3], 4.0); // d: (0,2)
    assert_eq!(storage[4], 5.0); // e: (1,2)
    assert_eq!(storage[5], 6.0); // f: (2,2)
}

#[test]
fn test_packed_lower_blas_format() {
    // BLAS lower triangular packed format (column-major):
    // For 3x3 matrix A:
    // A = [a    ]    packed as [a, b, c, d, e, f]
    //     [b d  ]    indices:   0  1  2  3  4  5
    //     [c e f]

    let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Lower);

    // Set elements
    p.set(0, 0, 1.0); // a
    p.set(1, 0, 2.0); // b
    p.set(2, 0, 3.0); // c
    p.set(1, 1, 4.0); // d
    p.set(2, 1, 5.0); // e
    p.set(2, 2, 6.0); // f

    // Verify packed storage follows BLAS convention
    let storage = p.as_slice();
    assert_eq!(storage[0], 1.0); // a: (0,0)
    assert_eq!(storage[1], 2.0); // b: (1,0)
    assert_eq!(storage[2], 3.0); // c: (2,0)
    assert_eq!(storage[3], 4.0); // d: (1,1)
    assert_eq!(storage[4], 5.0); // e: (2,1)
    assert_eq!(storage[5], 6.0); // f: (2,2)
}

#[test]
fn test_packed_storage_size() {
    // Packed storage size should be n*(n+1)/2
    for n in 1..=10 {
        let upper: PackedMat<f64> = PackedMat::zeros(n, TriangularKind::Upper);
        let lower: PackedMat<f64> = PackedMat::zeros(n, TriangularKind::Lower);

        let expected = n * (n + 1) / 2;
        assert_eq!(
            upper.as_slice().len(),
            expected,
            "Upper packed size wrong for n={}",
            n
        );
        assert_eq!(
            lower.as_slice().len(),
            expected,
            "Lower packed size wrong for n={}",
            n
        );
    }
}

// ============================================================================
// Banded storage format tests (BLAS compatible)
// ============================================================================

#[test]
fn test_banded_storage_format() {
    // BLAS banded storage: AB(ku+i-j, j) = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl)
    // Storage layout: (kl+ku+1) rows, n columns

    let n = 5;
    let kl = 1; // 1 subdiagonal
    let ku = 2; // 2 superdiagonals

    let mut b: BandedMat<f64> = BandedMat::zeros(n, n, kl, ku);

    // Set some elements
    b.set(0, 0, 11.0);
    b.set(0, 1, 12.0);
    b.set(0, 2, 13.0);
    b.set(1, 0, 21.0);
    b.set(1, 1, 22.0);
    b.set(1, 2, 23.0);
    b.set(1, 3, 24.0);
    b.set(2, 1, 32.0);
    b.set(2, 2, 33.0);
    b.set(2, 3, 34.0);
    b.set(2, 4, 35.0);

    // Verify storage size
    assert_eq!(b.as_slice().len(), (kl + ku + 1) * n);

    // Verify we can read back the values
    assert_eq!(b.get(0, 0), Some(&11.0));
    assert_eq!(b.get(0, 1), Some(&12.0));
    assert_eq!(b.get(1, 1), Some(&22.0));
    assert_eq!(b.get(2, 2), Some(&33.0));

    // Elements outside band should return None
    assert_eq!(b.get(3, 0), None);
    assert_eq!(b.get(0, 3), None);
}

#[test]
fn test_banded_diagonal_access() {
    let n = 5;
    let kl = 1;
    let ku = 1;

    let mut b: BandedMat<f64> = BandedMat::zeros(n, n, kl, ku);

    // Set main diagonal
    for i in 0..n {
        b.set(i, i, (i + 1) as f64 * 10.0);
    }

    // Set subdiagonal
    for i in 1..n {
        b.set(i, i - 1, (i + 1) as f64);
    }

    // Set superdiagonal
    for i in 0..n - 1 {
        b.set(i, i + 1, ((i + 1) * 100) as f64);
    }

    // Verify diagonals
    for i in 0..n {
        assert_eq!(b.get(i, i), Some(&((i + 1) as f64 * 10.0)));
    }
    for i in 1..n {
        assert_eq!(b.get(i, i - 1), Some(&((i + 1) as f64)));
    }
    for i in 0..n - 1 {
        assert_eq!(b.get(i, i + 1), Some(&(((i + 1) * 100) as f64)));
    }
}

// ============================================================================
// BLAS operation compatibility tests
// ============================================================================

#[test]
fn test_gemm_layout_requirements() {
    // GEMM requires: C := alpha*A*B + beta*C
    // All matrices should be column-major with proper leading dimensions

    let m = 4;
    let n = 3;
    let k = 5;

    let a: Mat<f64> = Mat::zeros(m, k);
    let b: Mat<f64> = Mat::zeros(k, n);
    let c: Mat<f64> = Mat::zeros(m, n);

    // Verify leading dimensions
    let lda = a.row_stride();
    let ldb = b.row_stride();
    let ldc = c.row_stride();

    assert!(lda >= m, "lda must be >= m");
    assert!(ldb >= k, "ldb must be >= k");
    assert!(ldc >= m, "ldc must be >= m");
}

#[test]
fn test_pointer_arithmetic_for_blas() {
    let m: Mat<f64> = Mat::zeros(4, 4);
    let view = m.as_ref();

    let ptr = view.as_ptr();
    let lda = view.row_stride();

    // BLAS-style pointer arithmetic for accessing A(i,j)
    // A(i,j) = *(ptr + i + j*lda)
    for i in 0..4 {
        for j in 0..4 {
            let blas_ptr = unsafe { ptr.add(i + j * lda) };
            let view_ptr = view.ptr_at(i, j);
            assert_eq!(blas_ptr, view_ptr, "Pointer mismatch at ({}, {})", i, j);
        }
    }
}

#[test]
fn test_column_extraction_for_blas() {
    let mut m: Mat<f64> = Mat::zeros(4, 4);

    // Fill with test data
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = (i * 10 + j) as f64;
        }
    }

    // Extract column j: starts at ptr + j*lda, with stride 1
    let view = m.as_ref();
    let lda = view.row_stride();

    for j in 0..4 {
        let col = view.col(j);

        // Column should point to correct location
        let expected_ptr = unsafe { view.as_ptr().add(j * lda) };
        assert_eq!(col.as_ptr(), expected_ptr);

        // Verify column contents using MatRef indexing
        for i in 0..4 {
            assert_eq!(col[(i, 0)], (i * 10 + j) as f64);
        }
    }
}

#[test]
fn test_row_extraction_for_blas() {
    let mut m: Mat<f64> = Mat::zeros(4, 4);

    // Fill with test data
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = (i * 10 + j) as f64;
        }
    }

    // In column-major, row extraction has stride = lda
    let view = m.as_ref();
    let _lda = view.row_stride();

    for i in 0..4 {
        let row = view.row(i);

        // Row should point to correct location
        let expected_ptr = unsafe { view.as_ptr().add(i) };
        assert_eq!(row.as_ptr(), expected_ptr);

        // Verify row contents using MatRef indexing
        for j in 0..4 {
            assert_eq!(row[(0, j)], (i * 10 + j) as f64);
        }
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_single_element_matrix() {
    let m: Mat<f64> = Mat::filled(1, 1, 42.0);

    // Should still have valid layout
    assert!(m.row_stride() >= 1);
    assert_eq!(m[(0, 0)], 42.0);

    // Pointer access should work
    unsafe {
        assert_eq!(*m.as_ptr(), 42.0);
    }
}

#[test]
fn test_single_row_matrix() {
    let m: Mat<f64> = Mat::zeros(1, 10);

    let view = m.as_ref();
    assert_eq!(view.nrows(), 1);
    assert_eq!(view.ncols(), 10);

    // Single-row matrix is essentially a row vector
    // Each "column" has one element
    assert!(view.row_stride() >= 1);
}

#[test]
fn test_single_column_matrix() {
    let m: Mat<f64> = Mat::zeros(10, 1);

    let view = m.as_ref();
    assert_eq!(view.nrows(), 10);
    assert_eq!(view.ncols(), 1);

    // Single-column matrix is essentially a column vector
    // The column should be contiguous
    assert!(view.row_stride() >= 10);
}
