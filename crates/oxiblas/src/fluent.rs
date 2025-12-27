//! Fluent API for chained matrix operations.
//!
//! This module provides extension traits that enable method chaining for
//! common linear algebra operations. Instead of calling functions with
//! explicit output parameters, you can chain operations together:
//!
//! # Example
//!
//! ```
//! use oxiblas::prelude::*;
//! use oxiblas::fluent::MatrixOps;
//!
//! // Create matrices
//! let a: Mat<f64> = Mat::from_rows(&[
//!     &[1.0, 2.0],
//!     &[3.0, 4.0],
//! ]);
//! let b: Mat<f64> = Mat::from_rows(&[
//!     &[5.0, 6.0],
//!     &[7.0, 8.0],
//! ]);
//!
//! // Fluent API: chain operations
//! let c = a.as_ref().matmul(&b);
//!
//! // Equivalent to:
//! // let mut c = Mat::zeros(2, 2);
//! // gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
//!
//! assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
//! ```
//!
//! # Available Operations
//!
//! ## Matrix-Matrix Operations
//! - [`MatrixOps::matmul`] - Matrix multiplication (GEMM)
//! - [`MatrixOps::add_scaled`] - Add scaled matrix (αA + βB)
//!
//! ## Matrix Properties
//! - [`MatrixOps::transpose`] - Create transpose view
//! - [`MatrixOps::frobenius_norm`] - Frobenius norm
//! - [`MatrixOps::trace`] - Matrix trace
//!
//! ## Decompositions (returning Result)
//! - [`MatrixOps::lu`] - LU decomposition
//! - [`MatrixOps::qr`] - QR decomposition
//! - [`MatrixOps::cholesky`] - Cholesky decomposition
//! - [`MatrixOps::svd`] - Singular value decomposition
//!
//! ## Solvers
//! - [`MatrixOps::solve`] - Solve Ax = B
//! - [`MatrixOps::lstsq`] - Least squares solution

use oxiblas_blas::level1;
use oxiblas_blas::level3::{self, GemmKernel};
use oxiblas_core::{Field, Real, Scalar};
use oxiblas_lapack::{cholesky::Cholesky, lu::Lu, qr::Qr, solve, svd::Svd};
use oxiblas_matrix::{Mat, MatMut, MatRef, TransposeRef};

/// Extension trait for fluent matrix operations on immutable references.
///
/// This trait provides method chaining for common matrix operations.
/// Operations that produce a new matrix return `Mat<T>`, while
/// decompositions return `Result<Decomposition, Error>`.
pub trait MatrixOps<'a, T: Scalar> {
    /// Returns a transpose view of the matrix.
    ///
    /// This is a zero-copy operation that returns a view with swapped strides.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas::prelude::*;
    /// use oxiblas::fluent::MatrixOps;
    ///
    /// let m: Mat<f64> = Mat::from_rows(&[
    ///     &[1.0, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    /// ]);
    ///
    /// let t = m.as_ref().transpose();
    /// assert_eq!(t.nrows(), 3);
    /// assert_eq!(t.ncols(), 2);
    /// ```
    fn transpose(&self) -> TransposeRef<'a, T>;

    /// Computes the matrix product C = A * B.
    ///
    /// Returns a new owned matrix containing the result.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are incompatible (self.ncols != other.nrows).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas::prelude::*;
    /// use oxiblas::fluent::MatrixOps;
    ///
    /// let a: Mat<f64> = Mat::from_rows(&[
    ///     &[1.0, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    /// let b: Mat<f64> = Mat::from_rows(&[
    ///     &[5.0, 6.0],
    ///     &[7.0, 8.0],
    /// ]);
    ///
    /// let c = a.as_ref().matmul(&b);
    /// assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
    /// ```
    fn matmul(&self, other: &Mat<T>) -> Mat<T>
    where
        T: Field + GemmKernel + bytemuck::Zeroable;

    /// Computes C = α * A * B.
    ///
    /// Returns a new owned matrix containing the scaled product.
    fn matmul_scaled(&self, alpha: T, other: &Mat<T>) -> Mat<T>
    where
        T: Field + GemmKernel + bytemuck::Zeroable;

    /// Computes C = α * self + β * other.
    ///
    /// Returns a new owned matrix containing the result.
    ///
    /// # Panics
    ///
    /// Panics if dimensions don't match.
    fn add_scaled<'b>(&self, alpha: T, beta: T, other: MatRef<'b, T>) -> Mat<T>
    where
        T: bytemuck::Zeroable;

    /// Computes the Frobenius norm ||A||_F = sqrt(sum(a_ij^2)).
    fn frobenius_norm(&self) -> T::Real
    where
        T: Real;

    /// Computes the trace (sum of diagonal elements).
    fn trace(&self) -> T;

    /// Computes the 1-norm (maximum column sum).
    fn norm_1(&self) -> T::Real
    where
        T: Real;

    /// Computes the infinity norm (maximum row sum).
    fn norm_inf(&self) -> T::Real
    where
        T: Real;

    /// Computes the LU decomposition with partial pivoting.
    ///
    /// Returns `Result<Lu<T>, LuError>`.
    fn lu(&self) -> Result<Lu<T>, oxiblas_lapack::lu::LuError>
    where
        T: Field + bytemuck::Zeroable;

    /// Computes the QR decomposition.
    ///
    /// Returns `Result<Qr<T>, QrError>`.
    fn qr(&self) -> Result<Qr<T>, oxiblas_lapack::qr::QrError>
    where
        T: Field + Real + bytemuck::Zeroable;

    /// Computes the Cholesky decomposition (for SPD matrices).
    ///
    /// Returns `Result<Cholesky<T>, CholeskyError>`.
    fn cholesky(&self) -> Result<Cholesky<T>, oxiblas_lapack::cholesky::CholeskyError>
    where
        T: Field + Real + bytemuck::Zeroable;

    /// Computes the singular value decomposition.
    ///
    /// Returns `Result<Svd<T>, SvdError>`.
    fn svd(&self) -> Result<Svd<T>, oxiblas_lapack::svd::SvdError>
    where
        T: Field + Real + bytemuck::Zeroable;

    /// Solves the linear system Ax = B.
    ///
    /// Uses LU decomposition internally.
    fn solve<'b>(&self, b: MatRef<'b, T>) -> Result<Mat<T>, oxiblas_lapack::solve::SolveError>
    where
        T: Field + Real + bytemuck::Zeroable;

    /// Computes the least squares solution to Ax = B.
    ///
    /// Uses QR decomposition internally.
    fn lstsq<'b>(
        &self,
        b: MatRef<'b, T>,
    ) -> Result<oxiblas_lapack::solve::LeastSquaresResult<T>, oxiblas_lapack::solve::LstSqError>
    where
        T: Field + Real + bytemuck::Zeroable;
}

impl<'a, T: Scalar> MatrixOps<'a, T> for MatRef<'a, T> {
    fn transpose(&self) -> TransposeRef<'a, T> {
        MatRef::transpose(self)
    }

    fn matmul(&self, other: &Mat<T>) -> Mat<T>
    where
        T: Field + GemmKernel + bytemuck::Zeroable,
    {
        assert_eq!(
            self.ncols(),
            other.nrows(),
            "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
            self.nrows(),
            self.ncols(),
            other.nrows(),
            other.ncols()
        );
        let mut c = Mat::zeros(self.nrows(), other.ncols());
        level3::gemm(T::one(), *self, other.as_ref(), T::zero(), c.as_mut());
        c
    }

    fn matmul_scaled(&self, alpha: T, other: &Mat<T>) -> Mat<T>
    where
        T: Field + GemmKernel + bytemuck::Zeroable,
    {
        assert_eq!(
            self.ncols(),
            other.nrows(),
            "Matrix dimensions incompatible for multiplication"
        );
        let mut c = Mat::zeros(self.nrows(), other.ncols());
        level3::gemm(alpha, *self, other.as_ref(), T::zero(), c.as_mut());
        c
    }

    fn add_scaled<'b>(&self, alpha: T, beta: T, other: MatRef<'b, T>) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        assert_eq!(
            (self.nrows(), self.ncols()),
            (other.nrows(), other.ncols()),
            "Matrix dimensions must match for addition"
        );
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                result[(i, j)] = alpha * (*self)[(i, j)] + beta * other[(i, j)];
            }
        }
        result
    }

    fn frobenius_norm(&self) -> T::Real
    where
        T: Real,
    {
        let mut sum = T::Real::zero();
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = (*self)[(i, j)];
                sum += val * val;
            }
        }
        Real::sqrt(sum)
    }

    fn trace(&self) -> T {
        let n = self.nrows().min(self.ncols());
        let mut sum = T::zero();
        for i in 0..n {
            sum += (*self)[(i, i)];
        }
        sum
    }

    fn norm_1(&self) -> T::Real
    where
        T: Real,
    {
        let mut max_col_sum = T::Real::zero();
        for j in 0..self.ncols() {
            let mut col_sum = T::Real::zero();
            for i in 0..self.nrows() {
                col_sum += Scalar::abs((*self)[(i, j)]);
            }
            if col_sum > max_col_sum {
                max_col_sum = col_sum;
            }
        }
        max_col_sum
    }

    fn norm_inf(&self) -> T::Real
    where
        T: Real,
    {
        let mut max_row_sum = T::Real::zero();
        for i in 0..self.nrows() {
            let mut row_sum = T::Real::zero();
            for j in 0..self.ncols() {
                row_sum += Scalar::abs((*self)[(i, j)]);
            }
            if row_sum > max_row_sum {
                max_row_sum = row_sum;
            }
        }
        max_row_sum
    }

    fn lu(&self) -> Result<Lu<T>, oxiblas_lapack::lu::LuError>
    where
        T: Field + bytemuck::Zeroable,
    {
        Lu::compute(*self)
    }

    fn qr(&self) -> Result<Qr<T>, oxiblas_lapack::qr::QrError>
    where
        T: Field + Real + bytemuck::Zeroable,
    {
        Qr::compute(*self)
    }

    fn cholesky(&self) -> Result<Cholesky<T>, oxiblas_lapack::cholesky::CholeskyError>
    where
        T: Field + Real + bytemuck::Zeroable,
    {
        Cholesky::compute(*self)
    }

    fn svd(&self) -> Result<Svd<T>, oxiblas_lapack::svd::SvdError>
    where
        T: Field + Real + bytemuck::Zeroable,
    {
        Svd::compute(*self)
    }

    fn solve<'b>(&self, b: MatRef<'b, T>) -> Result<Mat<T>, oxiblas_lapack::solve::SolveError>
    where
        T: Field + Real + bytemuck::Zeroable,
    {
        solve::solve(*self, b)
    }

    fn lstsq<'b>(
        &self,
        b: MatRef<'b, T>,
    ) -> Result<oxiblas_lapack::solve::LeastSquaresResult<T>, oxiblas_lapack::solve::LstSqError>
    where
        T: Field + Real + bytemuck::Zeroable,
    {
        oxiblas_lapack::solve::lstsq(*self, b)
    }
}

/// Extension trait for fluent matrix operations on mutable references.
///
/// This trait provides in-place operations that modify the matrix directly.
pub trait MatrixOpsMut<T: Scalar> {
    /// Scales the matrix in place: self = α * self.
    fn scale(&mut self, alpha: T);

    /// Adds a scaled matrix in place: self = self + α * other.
    fn add_assign<'b>(&mut self, alpha: T, other: MatRef<'b, T>);

    /// Sets self = α * A * B + β * self.
    ///
    /// This is the GEMM operation performed in place.
    fn set_to_gemm<'a, 'b>(&mut self, alpha: T, a: MatRef<'a, T>, b: MatRef<'b, T>, beta: T)
    where
        T: Field + GemmKernel;

    /// Performs matrix-vector multiplication: y = α * A * x + β * y.
    fn gemv_inplace<'b>(&mut self, alpha: T, a: MatRef<'b, T>, x: &[T], beta: T)
    where
        T: Field;

    /// Transposes the matrix in place (must be square).
    fn transpose_inplace(&mut self)
    where
        T: Clone;
}

impl<T: Scalar + bytemuck::Zeroable> MatrixOpsMut<T> for MatMut<'_, T> {
    fn scale(&mut self, alpha: T) {
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                let val = (*self)[(i, j)];
                (*self)[(i, j)] = alpha * val;
            }
        }
    }

    fn add_assign<'b>(&mut self, alpha: T, other: MatRef<'b, T>) {
        assert_eq!(
            (self.nrows(), self.ncols()),
            (other.nrows(), other.ncols()),
            "Matrix dimensions must match"
        );
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                (*self)[(i, j)] += alpha * other[(i, j)];
            }
        }
    }

    fn set_to_gemm<'a, 'b>(&mut self, alpha: T, a: MatRef<'a, T>, b: MatRef<'b, T>, beta: T)
    where
        T: Field + GemmKernel,
    {
        level3::gemm(alpha, a, b, beta, self.rb_mut());
    }

    fn gemv_inplace<'b>(&mut self, alpha: T, a: MatRef<'b, T>, x: &[T], beta: T)
    where
        T: Field,
    {
        assert_eq!(self.ncols(), 1, "gemv_inplace requires column vector");
        assert_eq!(a.ncols(), x.len(), "Incompatible dimensions for gemv");
        assert_eq!(a.nrows(), self.nrows(), "Incompatible dimensions for gemv");

        // Simple implementation for now
        for i in 0..self.nrows() {
            let mut sum = T::zero();
            for k in 0..a.ncols() {
                sum += a[(i, k)] * x[k];
            }
            (*self)[(i, 0)] = alpha * sum + beta * (*self)[(i, 0)];
        }
    }

    fn transpose_inplace(&mut self)
    where
        T: Clone,
    {
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "In-place transpose requires square matrix"
        );
        let n = self.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let tmp = (*self)[(i, j)];
                (*self)[(i, j)] = (*self)[(j, i)];
                (*self)[(j, i)] = tmp;
            }
        }
    }
}

/// Extension trait for vector operations.
pub trait VectorOps<T: Scalar> {
    /// Computes the dot product of two vectors.
    fn dot(&self, other: &[T]) -> T
    where
        T: Field;

    /// Computes the Euclidean norm (L2 norm).
    fn norm2(&self) -> T::Real
    where
        T: Real;

    /// Scales the vector in place: x = α * x.
    fn scale(&mut self, alpha: T)
    where
        T: Field;

    /// Adds a scaled vector: y = α * x + y.
    fn axpy(&mut self, alpha: T, x: &[T])
    where
        T: Field;
}

impl<T: Scalar> VectorOps<T> for [T] {
    fn dot(&self, other: &[T]) -> T
    where
        T: Field,
    {
        assert_eq!(self.len(), other.len(), "Vector lengths must match");
        level1::dot(self, other)
    }

    fn norm2(&self) -> T::Real
    where
        T: Real,
    {
        level1::nrm2(self)
    }

    fn scale(&mut self, alpha: T)
    where
        T: Field,
    {
        level1::scal(alpha, self);
    }

    fn axpy(&mut self, alpha: T, x: &[T])
    where
        T: Field,
    {
        assert_eq!(self.len(), x.len(), "Vector lengths must match");
        level1::axpy(alpha, x, self);
    }
}

impl<T: Scalar> VectorOps<T> for Vec<T> {
    fn dot(&self, other: &[T]) -> T
    where
        T: Field,
    {
        self.as_slice().dot(other)
    }

    fn norm2(&self) -> T::Real
    where
        T: Real,
    {
        self.as_slice().norm2()
    }

    fn scale(&mut self, alpha: T)
    where
        T: Field,
    {
        self.as_mut_slice().scale(alpha);
    }

    fn axpy(&mut self, alpha: T, x: &[T])
    where
        T: Field,
    {
        self.as_mut_slice().axpy(alpha, x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Mat;

    #[test]
    fn test_matmul_fluent() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        let c = a.as_ref().matmul(&b);

        assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_fluent() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let t = m.as_ref().transpose();

        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 4.0);
    }

    #[test]
    fn test_add_scaled_fluent() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        // C = 2*A + 3*B
        let c = a.as_ref().add_scaled(2.0, 3.0, b.as_ref());

        assert!((c[(0, 0)] - 17.0).abs() < 1e-10); // 2*1 + 3*5 = 17
        assert!((c[(0, 1)] - 22.0).abs() < 1e-10); // 2*2 + 3*6 = 22
    }

    #[test]
    fn test_norms_fluent() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let frob = m.as_ref().frobenius_norm();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((frob - expected).abs() < 1e-10);

        let tr = m.as_ref().trace();
        assert!((tr - 5.0).abs() < 1e-10); // 1 + 4 = 5

        let n1 = m.as_ref().norm_1();
        assert!((n1 - 6.0).abs() < 1e-10); // max(1+3, 2+4) = 6

        let ninf = m.as_ref().norm_inf();
        assert!((ninf - 7.0).abs() < 1e-10); // max(1+2, 3+4) = 7
    }

    #[test]
    fn test_lu_fluent() {
        let m: Mat<f64> = Mat::from_rows(&[&[4.0, 3.0], &[6.0, 3.0]]);

        let lu = m.as_ref().lu().expect("LU should succeed");
        assert_eq!(lu.size(), 2);
    }

    #[test]
    fn test_qr_fluent() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = m.as_ref().qr().expect("QR should succeed");
        assert_eq!(qr.nrows(), 3);
        assert_eq!(qr.ncols(), 2);
    }

    #[test]
    fn test_solve_fluent() {
        // Solve Ax = b where A = [[4, 1], [1, 3]], b = [[1], [2]]
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[1.0], &[2.0]]);

        let x = a.as_ref().solve(b.as_ref()).expect("Solve should succeed");

        // Verify: A*x ≈ b
        let ax = a.as_ref().matmul(&x);
        assert!((ax[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((ax[(1, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_fluent() {
        // SPD matrix
        let m: Mat<f64> = Mat::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);

        let chol = m.as_ref().cholesky().expect("Cholesky should succeed");
        assert_eq!(chol.size(), 2);
    }

    #[test]
    fn test_svd_fluent() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = m.as_ref().svd().expect("SVD should succeed");
        assert_eq!(svd.singular_values().len(), 2);
    }

    #[test]
    fn test_matrix_ops_mut_scale() {
        let mut m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        m.as_mut().scale(2.0);

        assert!((m[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((m[(1, 1)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_ops_mut_add_assign() {
        let mut m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let other: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        m.as_mut().add_assign(2.0, other.as_ref());

        assert!((m[(0, 0)] - 11.0).abs() < 1e-10); // 1 + 2*5 = 11
        assert!((m[(1, 1)] - 20.0).abs() < 1e-10); // 4 + 2*8 = 20
    }

    #[test]
    fn test_matrix_ops_mut_transpose_inplace() {
        let mut m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        m.as_mut().transpose_inplace();

        assert!((m[(0, 1)] - 3.0).abs() < 1e-10);
        assert!((m[(1, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_ops_dot() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y = vec![5.0_f64, 6.0, 7.0, 8.0];

        let dot = x.dot(&y);

        assert!((dot - 70.0).abs() < 1e-10); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_vector_ops_norm2() {
        let x = vec![3.0_f64, 4.0];

        let norm = x.norm2();

        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_ops_axpy() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y = vec![4.0_f64, 5.0, 6.0];

        y.axpy(2.0, &x); // y = 2*x + y

        assert!((y[0] - 6.0).abs() < 1e-10); // 2*1 + 4 = 6
        assert!((y[1] - 9.0).abs() < 1e-10); // 2*2 + 5 = 9
        assert!((y[2] - 12.0).abs() < 1e-10); // 2*3 + 6 = 12
    }
}
