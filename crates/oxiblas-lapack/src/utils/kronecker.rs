//! Kronecker Product and Related Operations.
//!
//! Provides:
//! - Kronecker product (tensor product of matrices)
//! - Khatri-Rao product (column-wise Kronecker product)
//! - Kronecker sum
//! - Vec operation (stacking columns)

use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Computes the Kronecker product A ⊗ B.
///
/// If A is m×n and B is p×q, then A ⊗ B is mp×nq.
///
/// # Arguments
///
/// * `a` - First matrix (m × n)
/// * `b` - Second matrix (p × q)
///
/// # Returns
///
/// The Kronecker product (mp × nq)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::kron;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[0.0f64, 5.0],
///     &[6.0, 7.0],
/// ]);
///
/// let c = kron(a.as_ref(), b.as_ref());
///
/// // Result is 4×4 matrix
/// assert_eq!(c.nrows(), 4);
/// assert_eq!(c.ncols(), 4);
///
/// // c[0,0] = a[0,0] * b[0,0] = 1 * 0 = 0
/// assert!((c[(0, 0)] - 0.0).abs() < 1e-10);
/// // c[0,1] = a[0,0] * b[0,1] = 1 * 5 = 5
/// assert!((c[(0, 1)] - 5.0).abs() < 1e-10);
/// ```
pub fn kron<T: Scalar + Clone + Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let p = b.nrows();
    let q = b.ncols();

    let mut result = Mat::zeros(m * p, n * q);

    for i in 0..m {
        for j in 0..n {
            let aij = a[(i, j)];
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k, j * q + l)] = aij * b[(k, l)];
                }
            }
        }
    }

    result
}

/// Computes the Khatri-Rao product of A and B.
///
/// The Khatri-Rao product is the column-wise Kronecker product.
/// If A is m×n and B is p×n (same number of columns), then
/// the result is (mp)×n.
///
/// # Arguments
///
/// * `a` - First matrix (m × n)
/// * `b` - Second matrix (p × n)
///
/// # Returns
///
/// The Khatri-Rao product (mp × n)
///
/// # Panics
///
/// Panics if the matrices have different numbers of columns.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::khatri_rao;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[5.0f64, 6.0],
///     &[7.0, 8.0],
/// ]);
///
/// let c = khatri_rao(a.as_ref(), b.as_ref());
///
/// // Result is 4×2 matrix
/// assert_eq!(c.nrows(), 4);
/// assert_eq!(c.ncols(), 2);
/// ```
pub fn khatri_rao<T: Scalar + Clone + Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Mat<T> {
    assert_eq!(
        a.ncols(),
        b.ncols(),
        "Matrices must have the same number of columns for Khatri-Rao product"
    );

    let m = a.nrows();
    let p = b.nrows();
    let n = a.ncols();

    let mut result = Mat::zeros(m * p, n);

    for j in 0..n {
        for i in 0..m {
            let aij = a[(i, j)];
            for k in 0..p {
                result[(i * p + k, j)] = aij * b[(k, j)];
            }
        }
    }

    result
}

/// Computes the Kronecker sum A ⊕ B.
///
/// The Kronecker sum is defined as:
/// A ⊕ B = A ⊗ I_p + I_m ⊗ B
///
/// where A is m×m, B is p×p, and I_k is the k×k identity matrix.
///
/// # Arguments
///
/// * `a` - First square matrix (m × m)
/// * `b` - Second square matrix (p × p)
///
/// # Returns
///
/// The Kronecker sum (mp × mp)
///
/// # Panics
///
/// Panics if either matrix is not square.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::kron_sum;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let b = Mat::from_rows(&[
///     &[5.0f64, 6.0],
///     &[7.0, 8.0],
/// ]);
///
/// let c = kron_sum(a.as_ref(), b.as_ref());
///
/// // Result is 4×4 matrix
/// assert_eq!(c.nrows(), 4);
/// assert_eq!(c.ncols(), 4);
/// ```
pub fn kron_sum<T: Scalar + Clone + Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Mat<T> {
    assert_eq!(a.nrows(), a.ncols(), "First matrix must be square");
    assert_eq!(b.nrows(), b.ncols(), "Second matrix must be square");

    let m = a.nrows();
    let p = b.nrows();
    let n = m * p;

    // Create identity matrices
    let eye_m = Mat::<T>::eye(m);
    let eye_p = Mat::<T>::eye(p);

    // Compute A ⊗ I_p
    let a_kron_ip = kron(a, eye_p.as_ref());

    // Compute I_m ⊗ B
    let im_kron_b = kron(eye_m.as_ref(), b);

    // Sum the two
    let mut result = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            result[(i, j)] = a_kron_ip[(i, j)] + im_kron_b[(i, j)];
        }
    }

    result
}

/// Vectorizes a matrix by stacking its columns.
///
/// Also known as the vec operation.
///
/// # Arguments
///
/// * `a` - Matrix (m × n)
///
/// # Returns
///
/// A column vector (mn × 1)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::vec_mat;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let v = vec_mat(a.as_ref());
///
/// // Result is [1, 3, 2, 4]^T (column-major order)
/// assert_eq!(v.nrows(), 4);
/// assert_eq!(v.ncols(), 1);
/// assert!((v[(0, 0)] - 1.0).abs() < 1e-10);
/// assert!((v[(1, 0)] - 3.0).abs() < 1e-10);
/// assert!((v[(2, 0)] - 2.0).abs() < 1e-10);
/// assert!((v[(3, 0)] - 4.0).abs() < 1e-10);
/// ```
pub fn vec_mat<T: Scalar + Clone + bytemuck::Zeroable>(a: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();

    let mut result = Mat::zeros(m * n, 1);

    for j in 0..n {
        for i in 0..m {
            result[(j * m + i, 0)] = a[(i, j)];
        }
    }

    result
}

/// Reshapes a vector back into a matrix.
///
/// Inverse of vec_mat.
///
/// # Arguments
///
/// * `v` - Column vector (mn × 1)
/// * `m` - Number of rows in result
/// * `n` - Number of columns in result
///
/// # Returns
///
/// A matrix (m × n)
///
/// # Panics
///
/// Panics if v.nrows() != m * n.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::{vec_mat, unvec};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let v = vec_mat(a.as_ref());
/// let b = unvec(v.as_ref(), 2, 2);
///
/// // b should equal a
/// for i in 0..2 {
///     for j in 0..2 {
///         assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-10);
///     }
/// }
/// ```
pub fn unvec<T: Scalar + Clone + bytemuck::Zeroable>(
    v: MatRef<'_, T>,
    m: usize,
    n: usize,
) -> Mat<T> {
    assert_eq!(v.nrows(), m * n, "Vector length must equal m*n");
    assert_eq!(v.ncols(), 1, "Input must be a column vector");

    let mut result = Mat::zeros(m, n);

    for j in 0..n {
        for i in 0..m {
            result[(i, j)] = v[(j * m + i, 0)];
        }
    }

    result
}

/// Computes the commutation matrix K_{m,n}.
///
/// The commutation matrix K_{m,n} is an (mn)×(mn) matrix such that
/// K_{m,n} * vec(A) = vec(A^T) for any m×n matrix A.
///
/// # Arguments
///
/// * `m` - Number of rows in the matrix being permuted
/// * `n` - Number of columns in the matrix being permuted
///
/// # Returns
///
/// The (mn)×(mn) commutation matrix
///
/// # Example
///
/// ```
/// use oxiblas_lapack::utils::{commutation_matrix, vec_mat};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
///
/// let k = commutation_matrix::<f64>(2, 2);
/// let va = vec_mat(a.as_ref());
///
/// // K * vec(A) should equal vec(A^T)
/// ```
pub fn commutation_matrix<T: Scalar + Clone + Field + bytemuck::Zeroable>(
    m: usize,
    n: usize,
) -> Mat<T> {
    let mn = m * n;
    let mut result = Mat::zeros(mn, mn);

    for i in 0..m {
        for j in 0..n {
            let idx1 = j * m + i; // Column-major index in vec(A)
            let idx2 = i * n + j; // Column-major index in vec(A^T)
            result[(idx2, idx1)] = T::one();
        }
    }

    result
}

/// Computes the duplication matrix D_n.
///
/// The duplication matrix D_n is an (n^2)×(n(n+1)/2) matrix such that
/// D_n * vech(A) = vec(A) for any symmetric n×n matrix A.
///
/// Here, vech(A) is the half-vectorization containing the lower triangular part.
///
/// # Arguments
///
/// * `n` - Size of the symmetric matrix
///
/// # Returns
///
/// The (n²)×(n(n+1)/2) duplication matrix
pub fn duplication_matrix<T: Scalar + Clone + Field + bytemuck::Zeroable>(n: usize) -> Mat<T> {
    let n_sq = n * n;
    let n_half = n * (n + 1) / 2;

    let mut result = Mat::zeros(n_sq, n_half);

    let mut col = 0;
    for j in 0..n {
        for i in j..n {
            // Position in vec(A) for (i, j)
            let row1 = j * n + i;
            result[(row1, col)] = T::one();

            // Position in vec(A) for (j, i) if i != j
            if i != j {
                let row2 = i * n + j;
                result[(row2, col)] = T::one();
            }

            col += 1;
        }
    }

    result
}

/// Computes the elimination matrix L_n.
///
/// The elimination matrix L_n is an (n(n+1)/2)×(n²) matrix such that
/// L_n * vec(A) = vech(A) for any symmetric n×n matrix A.
///
/// # Arguments
///
/// * `n` - Size of the symmetric matrix
///
/// # Returns
///
/// The (n(n+1)/2)×(n²) elimination matrix
pub fn elimination_matrix<T: Scalar + Clone + Field + bytemuck::Zeroable>(n: usize) -> Mat<T> {
    let n_sq = n * n;
    let n_half = n * (n + 1) / 2;

    let mut result = Mat::zeros(n_half, n_sq);

    let mut row = 0;
    for j in 0..n {
        for i in j..n {
            // Position in vec(A) for (i, j)
            let col = j * n + i;
            result[(row, col)] = T::one();
            row += 1;
        }
    }

    result
}

/// Applies Kronecker-matrix-vector product (A ⊗ B) * x efficiently.
///
/// Instead of forming the full Kronecker product, this computes
/// vec(B * X * A^T) where x = vec(X).
///
/// # Arguments
///
/// * `a` - First matrix (m × n)
/// * `b` - Second matrix (p × q)
/// * `x` - Vector (nq × 1)
///
/// # Returns
///
/// The result (mp × 1)
pub fn kron_vec<T: Scalar + Clone + Field + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    x: &[T],
) -> Vec<T> {
    let m = a.nrows();
    let n = a.ncols();
    let p = b.nrows();
    let q = b.ncols();

    assert_eq!(x.len(), n * q, "Vector length must match A.ncols * B.ncols");

    // Reshape x into X (q × n), compute B * X * A^T, vectorize
    // First compute Y = X * A^T (q × m)
    let mut y = vec![T::zero(); q * m];
    for i in 0..q {
        for j in 0..m {
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + x[k * q + i] * a[(j, k)]; // A^T[k,j] = A[j,k]
            }
            y[j * q + i] = sum;
        }
    }

    // Then compute Z = B * Y (p × m)
    let mut result = vec![T::zero(); p * m];
    for i in 0..p {
        for j in 0..m {
            let mut sum = T::zero();
            for k in 0..q {
                sum = sum + b[(i, k)] * y[j * q + k];
            }
            result[j * p + i] = sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kron_identity() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let c = kron(a.as_ref(), b.as_ref());

        // A ⊗ I_2 should give block diagonal with A elements
        assert_eq!(c.nrows(), 4);
        assert_eq!(c.ncols(), 4);

        // Check some elements
        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 0.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 2.0).abs() < 1e-10);
        assert!((c[(0, 3)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kron_2x2() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[0.0f64, 5.0], &[6.0, 7.0]]);

        let c = kron(a.as_ref(), b.as_ref());

        // Expected result:
        // [1*[0,5;6,7]  2*[0,5;6,7]]
        // [3*[0,5;6,7]  4*[0,5;6,7]]
        assert_eq!(c.nrows(), 4);
        assert_eq!(c.ncols(), 4);

        // Top-left block: 1 * B
        assert!((c[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 5.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 6.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 7.0).abs() < 1e-10);

        // Top-right block: 2 * B
        assert!((c[(0, 2)] - 0.0).abs() < 1e-10);
        assert!((c[(0, 3)] - 10.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 12.0).abs() < 1e-10);
        assert!((c[(1, 3)] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_kron_rectangular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[2.0]]);

        let c = kron(a.as_ref(), b.as_ref());

        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 3);

        assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((c[(0, 2)] - 3.0).abs() < 1e-10);
        assert!((c[(1, 2)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_khatri_rao() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);

        let c = khatri_rao(a.as_ref(), b.as_ref());

        assert_eq!(c.nrows(), 4);
        assert_eq!(c.ncols(), 2);

        // First column: kron(a[:,0], b[:,0])
        assert!((c[(0, 0)] - 5.0).abs() < 1e-10); // 1*5
        assert!((c[(1, 0)] - 7.0).abs() < 1e-10); // 1*7
        assert!((c[(2, 0)] - 15.0).abs() < 1e-10); // 3*5
        assert!((c[(3, 0)] - 21.0).abs() < 1e-10); // 3*7

        // Second column: kron(a[:,1], b[:,1])
        assert!((c[(0, 1)] - 12.0).abs() < 1e-10); // 2*6
        assert!((c[(1, 1)] - 16.0).abs() < 1e-10); // 2*8
        assert!((c[(2, 1)] - 24.0).abs() < 1e-10); // 4*6
        assert!((c[(3, 1)] - 32.0).abs() < 1e-10); // 4*8
    }

    #[test]
    fn test_kron_sum() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);
        let b = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let c = kron_sum(a.as_ref(), b.as_ref());

        assert_eq!(c.nrows(), 4);
        assert_eq!(c.ncols(), 4);

        // For diagonal A and B, the diagonal of A ⊕ B should be
        // [a1+b1, a1+b2, a2+b1, a2+b2] = [4, 5, 5, 6]
        assert!((c[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 5.0).abs() < 1e-10);
        assert!((c[(2, 2)] - 5.0).abs() < 1e-10);
        assert!((c[(3, 3)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_unvec() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let v = vec_mat(a.as_ref());

        // Column-major: [1, 3, 2, 4]
        assert!((v[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((v[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((v[(2, 0)] - 2.0).abs() < 1e-10);
        assert!((v[(3, 0)] - 4.0).abs() < 1e-10);

        let b = unvec(v.as_ref(), 2, 2);

        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - b[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_commutation_matrix() {
        let k = commutation_matrix::<f64>(2, 3);

        // K_{2,3} is 6x6
        assert_eq!(k.nrows(), 6);
        assert_eq!(k.ncols(), 6);

        // Check that K is a permutation matrix
        for i in 0..6 {
            let mut row_sum = 0.0;
            for j in 0..6 {
                row_sum += k[(i, j)];
            }
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_duplication_matrix() {
        let d = duplication_matrix::<f64>(2);

        // D_2 is 4x3
        assert_eq!(d.nrows(), 4);
        assert_eq!(d.ncols(), 3);
    }

    #[test]
    fn test_elimination_matrix() {
        let l = elimination_matrix::<f64>(2);

        // L_2 is 3x4
        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 4);
    }

    #[test]
    fn test_kron_vec() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);
        let x = vec![1.0f64, 0.0, 0.0, 1.0]; // vec(I_2)

        let y = kron_vec(a.as_ref(), b.as_ref(), &x);

        // Compare with explicit Kronecker product
        let k = kron(a.as_ref(), b.as_ref());
        for i in 0..4 {
            let mut expected = 0.0;
            for j in 0..4 {
                expected += k[(i, j)] * x[j];
            }
            assert!((y[i] - expected).abs() < 1e-10);
        }
    }
}
