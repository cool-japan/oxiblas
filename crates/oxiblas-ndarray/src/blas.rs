//! BLAS operations on ndarray types.
//!
//! This module provides BLAS Level 1, 2, and 3 operations directly on
//! ndarray types, using OxiBLAS as the backend.

use crate::conversions::array2_to_mat;
use ndarray::{Array1, Array2, ArrayView1, ShapeBuilder};
use num_complex::{Complex32, Complex64};
use oxiblas_blas::level1::{asum, axpy, dot, dotc_c32, dotc_c64, dotu_c32, dotu_c64, nrm2, scal};
use oxiblas_blas::level2::{GemvTrans, gemv as blas_gemv};
use oxiblas_blas::level3::{GemmKernel, gemm as blas_gemm};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;

// =============================================================================
// BLAS Level 1: Vector-Vector Operations
// =============================================================================

/// Computes the dot product of two 1D arrays.
///
/// # Arguments
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
/// The dot product x·y
///
/// # Panics
/// Panics if vectors have different lengths.
pub fn dot_ndarray<T: Field>(x: &Array1<T>, y: &Array1<T>) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    // Try to get contiguous slices for efficient computation
    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice()) {
        dot(x_slice, y_slice)
    } else {
        // Non-contiguous: convert to contiguous first
        let x_vec: Vec<T> = x.iter().cloned().collect();
        let y_vec: Vec<T> = y.iter().cloned().collect();
        dot(&x_vec, &y_vec)
    }
}

/// Computes the dot product of two array views.
pub fn dot_view<T: Field>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice()) {
        dot(x_slice, y_slice)
    } else {
        let x_vec: Vec<T> = x.iter().cloned().collect();
        let y_vec: Vec<T> = y.iter().cloned().collect();
        dot(&x_vec, &y_vec)
    }
}

// =============================================================================
// Complex Dot Products
// =============================================================================

/// Computes the conjugate dot product of two Complex64 vectors (ZDOTC).
///
/// x^H · y = Σ conj(x\[i\]) * y\[i\]
///
/// This is the standard inner product for complex vector spaces.
///
/// # Arguments
/// * `x` - First complex vector (will be conjugated)
/// * `y` - Second complex vector
///
/// # Returns
/// The conjugate dot product
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use oxiblas_ndarray::blas::dotc_c64_ndarray;
/// use ndarray::array;
/// use num_complex::Complex64;
///
/// let x = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
/// let y = array![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
/// let result = dotc_c64_ndarray(&x, &y);
/// // conj(1+2i)*(5+6i) + conj(3+4i)*(7+8i) = (1-2i)(5+6i) + (3-4i)(7+8i)
/// // = (5+12) + (6-10)i + (21+32) + (24-28)i = 70 - 8i
/// assert!((result.re - 70.0).abs() < 1e-10);
/// assert!((result.im - (-8.0)).abs() < 1e-10);
/// ```
pub fn dotc_c64_ndarray(x: &Array1<Complex64>, y: &Array1<Complex64>) -> Complex64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice()) {
        dotc_c64(x_slice, y_slice)
    } else {
        let x_vec: Vec<Complex64> = x.iter().copied().collect();
        let y_vec: Vec<Complex64> = y.iter().copied().collect();
        dotc_c64(&x_vec, &y_vec)
    }
}

/// Computes the conjugate dot product of two Complex32 vectors (CDOTC).
///
/// x^H · y = Σ conj(x\[i\]) * y\[i\]
///
/// # Panics
/// Panics if vectors have different lengths.
pub fn dotc_c32_ndarray(x: &Array1<Complex32>, y: &Array1<Complex32>) -> Complex32 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice()) {
        dotc_c32(x_slice, y_slice)
    } else {
        let x_vec: Vec<Complex32> = x.iter().copied().collect();
        let y_vec: Vec<Complex32> = y.iter().copied().collect();
        dotc_c32(&x_vec, &y_vec)
    }
}

/// Computes the unconjugated dot product of two Complex64 vectors (ZDOTU).
///
/// x · y = Σ x\[i\] * y\[i\]
///
/// Note: This is the bilinear form, not the standard inner product.
/// For the standard inner product (sesquilinear), use `dotc_c64_ndarray`.
///
/// # Panics
/// Panics if vectors have different lengths.
pub fn dotu_c64_ndarray(x: &Array1<Complex64>, y: &Array1<Complex64>) -> Complex64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice()) {
        dotu_c64(x_slice, y_slice)
    } else {
        let x_vec: Vec<Complex64> = x.iter().copied().collect();
        let y_vec: Vec<Complex64> = y.iter().copied().collect();
        dotu_c64(&x_vec, &y_vec)
    }
}

/// Computes the unconjugated dot product of two Complex32 vectors (CDOTU).
///
/// x · y = Σ x\[i\] * y\[i\]
///
/// # Panics
/// Panics if vectors have different lengths.
pub fn dotu_c32_ndarray(x: &Array1<Complex32>, y: &Array1<Complex32>) -> Complex32 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice()) {
        dotu_c32(x_slice, y_slice)
    } else {
        let x_vec: Vec<Complex32> = x.iter().copied().collect();
        let y_vec: Vec<Complex32> = y.iter().copied().collect();
        dotu_c32(&x_vec, &y_vec)
    }
}

/// Computes the Euclidean norm of a Complex64 vector.
///
/// ||x||_2 = sqrt(Σ |x\[i\]|²) = sqrt(Σ (x\[i\].re² + x\[i\].im²))
///
/// This is equivalent to sqrt(x^H · x).
pub fn nrm2_c64_ndarray(x: &Array1<Complex64>) -> f64 {
    let mut sum = 0.0f64;
    for xi in x.iter() {
        sum += xi.norm_sqr();
    }
    sum.sqrt()
}

/// Computes the Euclidean norm of a Complex32 vector.
///
/// ||x||_2 = sqrt(Σ |x\[i\]|²)
pub fn nrm2_c32_ndarray(x: &Array1<Complex32>) -> f32 {
    let mut sum = 0.0f32;
    for xi in x.iter() {
        sum += xi.norm_sqr();
    }
    sum.sqrt()
}

/// Computes the L1 norm of a Complex64 vector (sum of absolute values).
///
/// ||x||_1 = Σ |x\[i\]|
pub fn asum_c64_ndarray(x: &Array1<Complex64>) -> f64 {
    let mut sum = 0.0f64;
    for xi in x.iter() {
        sum += xi.norm();
    }
    sum
}

/// Computes the L1 norm of a Complex32 vector (sum of absolute values).
///
/// ||x||_1 = Σ |x\[i\]|
pub fn asum_c32_ndarray(x: &Array1<Complex32>) -> f32 {
    let mut sum = 0.0f32;
    for xi in x.iter() {
        sum += xi.norm();
    }
    sum
}

/// Computes the Euclidean (L2) norm of a vector.
///
/// ||x||_2 = sqrt(sum(x_i^2))
pub fn nrm2_ndarray<T: Field + oxiblas_core::scalar::Real>(x: &Array1<T>) -> T {
    if let Some(slice) = x.as_slice() {
        nrm2(slice)
    } else {
        let vec: Vec<T> = x.iter().cloned().collect();
        nrm2(&vec)
    }
}

/// Computes the L1 norm (sum of absolute values) of a vector.
///
/// ||x||_1 = sum(|x_i|)
pub fn asum_ndarray<T: Field + oxiblas_core::scalar::Real>(x: &Array1<T>) -> T {
    if let Some(slice) = x.as_slice() {
        asum(slice)
    } else {
        let vec: Vec<T> = x.iter().cloned().collect();
        asum(&vec)
    }
}

/// Computes y = α·x + y (AXPY operation).
///
/// # Arguments
/// * `alpha` - Scalar multiplier
/// * `x` - Input vector
/// * `y` - Output vector (modified in place)
pub fn axpy_ndarray<T: Field>(alpha: T, x: &Array1<T>, y: &mut Array1<T>) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    if let (Some(x_slice), Some(y_slice)) = (x.as_slice(), y.as_slice_mut()) {
        axpy(alpha, x_slice, y_slice);
    } else {
        // Non-contiguous: element-wise
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi = alpha * (*xi) + *yi;
        }
    }
}

/// Scales a vector: x = α·x
pub fn scal_ndarray<T: Field>(alpha: T, x: &mut Array1<T>) {
    if let Some(slice) = x.as_slice_mut() {
        scal(alpha, slice);
    } else {
        for xi in x.iter_mut() {
            *xi = alpha * (*xi);
        }
    }
}

// =============================================================================
// BLAS Level 2: Matrix-Vector Operations
// =============================================================================

/// Transpose options for matrix-vector operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// No transpose
    NoTrans,
    /// Transpose
    Trans,
    /// Conjugate transpose (for complex types)
    ConjTrans,
}

impl From<Transpose> for GemvTrans {
    fn from(t: Transpose) -> Self {
        match t {
            Transpose::NoTrans => GemvTrans::NoTrans,
            Transpose::Trans => GemvTrans::Trans,
            Transpose::ConjTrans => GemvTrans::ConjTrans,
        }
    }
}

/// General matrix-vector multiplication: y = α·op(A)·x + β·y
///
/// # Arguments
/// * `trans` - Whether to transpose A
/// * `alpha` - Scalar multiplier for A·x
/// * `a` - The matrix (m×n)
/// * `x` - Input vector
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (modified in place)
///
/// # Panics
/// Panics if dimensions don't match.
pub fn gemv_ndarray<T: Field + Clone>(
    trans: Transpose,
    alpha: T,
    a: &Array2<T>,
    x: &Array1<T>,
    beta: T,
    y: &mut Array1<T>,
) where
    T: bytemuck::Zeroable,
{
    let a_mat = array2_to_mat(a);
    let (m, n) = a.dim();

    // Determine expected dimensions
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };

    assert_eq!(x.len(), x_len, "x dimension mismatch");
    assert_eq!(y.len(), y_len, "y dimension mismatch");

    // Convert vectors to slices
    let x_vec: Vec<T> = x.iter().cloned().collect();

    if let Some(y_slice) = y.as_slice_mut() {
        blas_gemv(trans.into(), alpha, a_mat.as_ref(), &x_vec, beta, y_slice);
    } else {
        let mut y_vec: Vec<T> = y.iter().cloned().collect();
        blas_gemv(
            trans.into(),
            alpha,
            a_mat.as_ref(),
            &x_vec,
            beta,
            &mut y_vec,
        );
        for (yi, val) in y.iter_mut().zip(y_vec.into_iter()) {
            *yi = val;
        }
    }
}

/// Matrix-vector multiplication: y = A·x
///
/// Simplified version of gemv with alpha=1, beta=0.
pub fn matvec<T: Field + Clone>(a: &Array2<T>, x: &Array1<T>) -> Array1<T>
where
    T: bytemuck::Zeroable,
{
    let (m, _n) = a.dim();
    let mut y = Array1::zeros(m);
    gemv_ndarray(Transpose::NoTrans, T::one(), a, x, T::zero(), &mut y);
    y
}

/// Transposed matrix-vector multiplication: y = A^T·x
pub fn matvec_t<T: Field + Clone>(a: &Array2<T>, x: &Array1<T>) -> Array1<T>
where
    T: bytemuck::Zeroable,
{
    let (_m, n) = a.dim();
    let mut y = Array1::zeros(n);
    gemv_ndarray(Transpose::Trans, T::one(), a, x, T::zero(), &mut y);
    y
}

// =============================================================================
// BLAS Level 3: Matrix-Matrix Operations
// =============================================================================

/// General matrix-matrix multiplication: C = α·A·B + β·C
///
/// # Arguments
/// * `alpha` - Scalar multiplier for A·B
/// * `a` - Left matrix (m×k)
/// * `b` - Right matrix (k×n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m×n), modified in place
///
/// # Panics
/// Panics if matrix dimensions are incompatible.
pub fn gemm_ndarray<T: Field + GemmKernel>(
    alpha: T,
    a: &Array2<T>,
    b: &Array2<T>,
    beta: T,
    c: &mut Array2<T>,
) where
    T: bytemuck::Zeroable + Clone,
{
    let a_mat = array2_to_mat(a);
    let b_mat = array2_to_mat(b);

    let (m, n) = c.dim();
    let mut c_mat: Mat<T> = Mat::zeros(m, n);

    // Copy existing C values if beta != 0
    if beta != T::zero() {
        for i in 0..m {
            for j in 0..n {
                c_mat[(i, j)] = c[[i, j]];
            }
        }
    }

    blas_gemm(alpha, a_mat.as_ref(), b_mat.as_ref(), beta, c_mat.as_mut());

    // Copy result back
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = c_mat[(i, j)];
        }
    }
}

/// Matrix multiplication: C = A·B
///
/// Simplified version that allocates a new output matrix.
pub fn matmul<T: Field + GemmKernel>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: bytemuck::Zeroable + Clone,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);

    let a_mat = array2_to_mat(a);
    let b_mat = array2_to_mat(b);
    let mut c_mat: Mat<T> = Mat::zeros(m, n);

    blas_gemm(
        T::one(),
        a_mat.as_ref(),
        b_mat.as_ref(),
        T::zero(),
        c_mat.as_mut(),
    );

    // Create output in column-major order for efficiency
    Array2::from_shape_fn((m, n).f(), |(i, j)| c_mat[(i, j)])
}

/// Matrix multiplication returning row-major output.
pub fn matmul_c<T: Field + GemmKernel>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: bytemuck::Zeroable + Clone,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k1, k2, "Inner dimensions must match");

    let a_mat = array2_to_mat(a);
    let b_mat = array2_to_mat(b);
    let mut c_mat: Mat<T> = Mat::zeros(m, n);

    blas_gemm(
        T::one(),
        a_mat.as_ref(),
        b_mat.as_ref(),
        T::zero(),
        c_mat.as_mut(),
    );

    // Row-major output
    Array2::from_shape_fn((m, n), |(i, j)| c_mat[(i, j)])
}

/// In-place matrix multiplication: C = A·B (C is reallocated)
pub fn matmul_into<T: Field + GemmKernel>(a: &Array2<T>, b: &Array2<T>, c: &mut Array2<T>)
where
    T: bytemuck::Zeroable + Clone,
{
    gemm_ndarray(T::one(), a, b, T::zero(), c);
}

// =============================================================================
// Matrix Norms
// =============================================================================

/// Computes the Frobenius norm of a matrix.
///
/// ||A||_F = sqrt(sum(a_ij^2))
pub fn frobenius_norm<T: Field + oxiblas_core::scalar::Real>(a: &Array2<T>) -> T {
    let mut sum = T::zero();
    for val in a.iter() {
        sum += (*val) * (*val);
    }
    oxiblas_core::scalar::Real::sqrt(sum)
}

/// Computes the 1-norm (maximum column sum) of a matrix.
///
/// For real types where `Real = T`.
pub fn norm_1(a: &Array2<f64>) -> f64 {
    let (nrows, ncols) = a.dim();
    let mut max_sum = 0.0f64;

    for j in 0..ncols {
        let mut col_sum = 0.0f64;
        for i in 0..nrows {
            col_sum += a[[i, j]].abs();
        }
        if col_sum > max_sum {
            max_sum = col_sum;
        }
    }

    max_sum
}

/// Computes the infinity-norm (maximum row sum) of a matrix.
///
/// For real types where `Real = T`.
pub fn norm_inf(a: &Array2<f64>) -> f64 {
    let (nrows, ncols) = a.dim();
    let mut max_sum = 0.0f64;

    for i in 0..nrows {
        let mut row_sum = 0.0f64;
        for j in 0..ncols {
            row_sum += a[[i, j]].abs();
        }
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }

    max_sum
}

/// Computes the maximum absolute element of a matrix.
///
/// For real types where `Real = T`.
pub fn norm_max(a: &Array2<f64>) -> f64 {
    let mut max_val = 0.0f64;
    for val in a.iter() {
        let abs_val = val.abs();
        if abs_val > max_val {
            max_val = abs_val;
        }
    }
    max_val
}

// =============================================================================
// Additional Operations
// =============================================================================

/// Computes the trace of a square matrix.
pub fn trace<T: Field>(a: &Array2<T>) -> T {
    let (nrows, ncols) = a.dim();
    assert_eq!(nrows, ncols, "Matrix must be square for trace");

    let mut sum = T::zero();
    for i in 0..nrows {
        sum += a[[i, i]];
    }
    sum
}

/// Transposes a matrix.
pub fn transpose<T: Clone>(a: &Array2<T>) -> Array2<T> {
    a.t().to_owned()
}

/// Creates an identity matrix.
pub fn eye<T: Field>(n: usize) -> Array2<T>
where
    T: Clone,
{
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        result[[i, i]] = T::one();
    }
    result
}

/// Creates an identity matrix in column-major order.
pub fn eye_f<T: Field>(n: usize) -> Array2<T>
where
    T: Clone,
{
    let mut result: Array2<T> = Array2::from_shape_fn((n, n).f(), |_| T::zero());
    for i in 0..n {
        result[[i, i]] = T::one();
    }
    result
}

// =============================================================================
// Complex Matrix Operations
// =============================================================================

/// Computes the Hermitian (conjugate) transpose of a Complex64 matrix.
///
/// Returns A^H where (A^H)\[i,j\] = conj(A\[j,i\])
///
/// # Example
/// ```
/// use oxiblas_ndarray::blas::conj_transpose_c64;
/// use ndarray::array;
/// use num_complex::Complex64;
///
/// let a = array![
///     [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
///     [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]
/// ];
/// let ah = conj_transpose_c64(&a);
/// // ah[0,0] = conj(a[0,0]) = 1 - 2i
/// assert!((ah[[0, 0]].re - 1.0).abs() < 1e-10);
/// assert!((ah[[0, 0]].im - (-2.0)).abs() < 1e-10);
/// // ah[0,1] = conj(a[1,0]) = 5 - 6i
/// assert!((ah[[0, 1]].re - 5.0).abs() < 1e-10);
/// assert!((ah[[0, 1]].im - (-6.0)).abs() < 1e-10);
/// ```
pub fn conj_transpose_c64(a: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    Array2::from_shape_fn((n, m), |(i, j)| a[[j, i]].conj())
}

/// Computes the Hermitian (conjugate) transpose of a Complex32 matrix.
///
/// Returns A^H where (A^H)\[i,j\] = conj(A\[j,i\])
pub fn conj_transpose_c32(a: &Array2<Complex32>) -> Array2<Complex32> {
    let (m, n) = a.dim();
    Array2::from_shape_fn((n, m), |(i, j)| a[[j, i]].conj())
}

/// Computes the Frobenius norm of a Complex64 matrix.
///
/// ||A||_F = sqrt(Σ |a\[i,j\]|²) = sqrt(Σ (a\[i,j\].re² + a\[i,j\].im²))
///
/// This is equivalent to sqrt(trace(A^H * A)).
pub fn frobenius_norm_c64(a: &Array2<Complex64>) -> f64 {
    let mut sum = 0.0f64;
    for val in a.iter() {
        sum += val.norm_sqr();
    }
    sum.sqrt()
}

/// Computes the Frobenius norm of a Complex32 matrix.
///
/// ||A||_F = sqrt(Σ |a\[i,j\]|²)
pub fn frobenius_norm_c32(a: &Array2<Complex32>) -> f32 {
    let mut sum = 0.0f32;
    for val in a.iter() {
        sum += val.norm_sqr();
    }
    sum.sqrt()
}

/// Computes the 1-norm (maximum column sum of absolute values) of a Complex64 matrix.
///
/// ||A||_1 = max_j Σ_i |a\[i,j\]|
pub fn norm_1_c64(a: &Array2<Complex64>) -> f64 {
    let (nrows, ncols) = a.dim();
    let mut max_sum = 0.0f64;

    for j in 0..ncols {
        let mut col_sum = 0.0f64;
        for i in 0..nrows {
            col_sum += a[[i, j]].norm();
        }
        if col_sum > max_sum {
            max_sum = col_sum;
        }
    }

    max_sum
}

/// Computes the 1-norm of a Complex32 matrix.
pub fn norm_1_c32(a: &Array2<Complex32>) -> f32 {
    let (nrows, ncols) = a.dim();
    let mut max_sum = 0.0f32;

    for j in 0..ncols {
        let mut col_sum = 0.0f32;
        for i in 0..nrows {
            col_sum += a[[i, j]].norm();
        }
        if col_sum > max_sum {
            max_sum = col_sum;
        }
    }

    max_sum
}

/// Computes the infinity-norm (maximum row sum of absolute values) of a Complex64 matrix.
///
/// ||A||_∞ = max_i Σ_j |a\[i,j\]|
pub fn norm_inf_c64(a: &Array2<Complex64>) -> f64 {
    let (nrows, ncols) = a.dim();
    let mut max_sum = 0.0f64;

    for i in 0..nrows {
        let mut row_sum = 0.0f64;
        for j in 0..ncols {
            row_sum += a[[i, j]].norm();
        }
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }

    max_sum
}

/// Computes the infinity-norm of a Complex32 matrix.
pub fn norm_inf_c32(a: &Array2<Complex32>) -> f32 {
    let (nrows, ncols) = a.dim();
    let mut max_sum = 0.0f32;

    for i in 0..nrows {
        let mut row_sum = 0.0f32;
        for j in 0..ncols {
            row_sum += a[[i, j]].norm();
        }
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }

    max_sum
}

/// Computes the maximum absolute element of a Complex64 matrix.
///
/// max |a\[i,j\]|
pub fn norm_max_c64(a: &Array2<Complex64>) -> f64 {
    let mut max_val = 0.0f64;
    for val in a.iter() {
        let abs_val = val.norm();
        if abs_val > max_val {
            max_val = abs_val;
        }
    }
    max_val
}

/// Computes the maximum absolute element of a Complex32 matrix.
pub fn norm_max_c32(a: &Array2<Complex32>) -> f32 {
    let mut max_val = 0.0f32;
    for val in a.iter() {
        let abs_val = val.norm();
        if abs_val > max_val {
            max_val = abs_val;
        }
    }
    max_val
}

/// Computes the trace of a Complex64 square matrix.
pub fn trace_c64(a: &Array2<Complex64>) -> Complex64 {
    let (nrows, ncols) = a.dim();
    assert_eq!(nrows, ncols, "Matrix must be square for trace");

    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..nrows {
        sum += a[[i, i]];
    }
    sum
}

/// Computes the trace of a Complex32 square matrix.
pub fn trace_c32(a: &Array2<Complex32>) -> Complex32 {
    let (nrows, ncols) = a.dim();
    assert_eq!(nrows, ncols, "Matrix must be square for trace");

    let mut sum = Complex32::new(0.0, 0.0);
    for i in 0..nrows {
        sum += a[[i, i]];
    }
    sum
}

/// Scales a Complex64 vector: x = α·x
pub fn scal_c64_ndarray(alpha: Complex64, x: &mut Array1<Complex64>) {
    for xi in x.iter_mut() {
        *xi = alpha * (*xi);
    }
}

/// Scales a Complex32 vector: x = α·x
pub fn scal_c32_ndarray(alpha: Complex32, x: &mut Array1<Complex32>) {
    for xi in x.iter_mut() {
        *xi = alpha * (*xi);
    }
}

/// AXPY operation for Complex64: y = α·x + y
pub fn axpy_c64_ndarray(alpha: Complex64, x: &Array1<Complex64>, y: &mut Array1<Complex64>) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha * (*xi) + *yi;
    }
}

/// AXPY operation for Complex32: y = α·x + y
pub fn axpy_c32_ndarray(alpha: Complex32, x: &Array1<Complex32>, y: &mut Array1<Complex32>) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha * (*xi) + *yi;
    }
}

/// Creates a Complex64 identity matrix.
pub fn eye_c64(n: usize) -> Array2<Complex64> {
    let mut result: Array2<Complex64> = Array2::from_elem((n, n), Complex64::new(0.0, 0.0));
    for i in 0..n {
        result[[i, i]] = Complex64::new(1.0, 0.0);
    }
    result
}

/// Creates a Complex32 identity matrix.
pub fn eye_c32(n: usize) -> Array2<Complex32> {
    let mut result: Array2<Complex32> = Array2::from_elem((n, n), Complex32::new(0.0, 0.0));
    for i in 0..n {
        result[[i, i]] = Complex32::new(1.0, 0.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dot_ndarray() {
        let x = array![1.0f64, 2.0, 3.0];
        let y = array![4.0f64, 5.0, 6.0];
        let d = dot_ndarray(&x, &y);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_ndarray() {
        let x = array![3.0f64, 4.0];
        let norm = nrm2_ndarray(&x);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_ndarray() {
        let x = array![-1.0f64, 2.0, -3.0];
        let sum = asum_ndarray(&x);
        assert!((sum - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_ndarray() {
        let x = array![1.0f64, 2.0, 3.0];
        let mut y = array![4.0f64, 5.0, 6.0];
        axpy_ndarray(2.0, &x, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 9.0).abs() < 1e-10);
        assert!((y[2] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_scal_ndarray() {
        let mut x = array![1.0f64, 2.0, 3.0];
        scal_ndarray(2.0, &mut x);
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
        assert!((x[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_notrans() {
        let a = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j + 1) as f64);
        let x = array![1.0f64, 1.0, 1.0];
        let mut y = array![0.0f64, 0.0];

        gemv_ndarray(Transpose::NoTrans, 1.0, &a, &x, 0.0, &mut y);

        // y[0] = 1 + 2 + 3 = 6
        // y[1] = 4 + 5 + 6 = 15
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_trans() {
        let a = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j + 1) as f64);
        let x = array![1.0f64, 1.0];
        let mut y = array![0.0f64, 0.0, 0.0];

        gemv_ndarray(Transpose::Trans, 1.0, &a, &x, 0.0, &mut y);

        // y[0] = 1 + 4 = 5
        // y[1] = 2 + 5 = 7
        // y[2] = 3 + 6 = 9
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
        assert!((y[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_matvec() {
        let a = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j + 1) as f64);
        let x = array![1.0f64, 2.0, 3.0];
        let y = matvec(&a, &x);

        // y[0] = 1*1 + 2*2 + 3*3 = 14
        // y[1] = 4*1 + 5*2 + 6*3 = 32
        assert!((y[0] - 14.0).abs() < 1e-10);
        assert!((y[1] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul() {
        let a = Array2::from_shape_fn((2, 3), |_| 1.0f64);
        let b = Array2::from_shape_fn((3, 2), |_| 2.0f64);
        let c = matmul(&a, &b);

        assert_eq!(c.dim(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - 6.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gemm_ndarray() {
        let a = Array2::from_shape_fn((2, 3), |_| 1.0f64);
        let b = Array2::from_shape_fn((3, 2), |_| 2.0f64);
        let mut c = Array2::from_shape_fn((2, 2), |_| 1.0f64);

        gemm_ndarray(1.0, &a, &b, 1.0, &mut c);

        // C = 1 * A * B + 1 * C = 6 + 1 = 7
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - 7.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_frobenius_norm() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let norm = frobenius_norm(&a);
        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert!((norm - 30.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_norm_1() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let norm = norm_1(&a);
        // max(1+3, 2+4) = max(4, 6) = 6
        assert!((norm - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_inf() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let norm = norm_inf(&a);
        // max(1+2, 3+4) = max(3, 7) = 7
        assert!((norm - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let a = array![[1.0f64, 2.0], [3.0, 4.0]];
        let tr = trace(&a);
        assert!((tr - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eye() {
        let id: Array2<f64> = eye(3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((id[[i, j]] - 1.0).abs() < 1e-15);
                } else {
                    assert!(id[[i, j]].abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_transpose() {
        let a = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let at = transpose(&a);
        assert_eq!(at.dim(), (3, 2));
        assert!((at[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((at[[2, 1]] - 6.0).abs() < 1e-15);
    }

    // =========================================================================
    // Complex Number Tests
    // =========================================================================

    #[test]
    fn test_dotc_c64_ndarray() {
        // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
        // conj(x) * y = (1-2i)(5+6i) + (3-4i)(7+8i)
        //             = (5+12) + (6-10)i + (21+32) + (24-28)i
        //             = 70 - 8i
        let x = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = array![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let result = dotc_c64_ndarray(&x, &y);
        assert!((result.re - 70.0).abs() < 1e-10);
        assert!((result.im - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dotc_c32_ndarray() {
        let x = array![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = array![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

        let result = dotc_c32_ndarray(&x, &y);
        assert!((result.re - 70.0).abs() < 1e-5);
        assert!((result.im - (-8.0)).abs() < 1e-5);
    }

    #[test]
    fn test_dotu_c64_ndarray() {
        // x = [1+2i, 3+4i], y = [5+6i, 7+8i]
        // x * y = (1+2i)(5+6i) + (3+4i)(7+8i)
        //       = (5-12) + (6+10)i + (21-32) + (24+28)i
        //       = -18 + 68i
        let x = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = array![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let result = dotu_c64_ndarray(&x, &y);
        assert!((result.re - (-18.0)).abs() < 1e-10);
        assert!((result.im - 68.0).abs() < 1e-10);
    }

    #[test]
    fn test_dotu_c32_ndarray() {
        let x = array![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = array![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

        let result = dotu_c32_ndarray(&x, &y);
        assert!((result.re - (-18.0)).abs() < 1e-5);
        assert!((result.im - 68.0).abs() < 1e-5);
    }

    #[test]
    fn test_dotc_c64_self_inner_product() {
        // x^H * x should be real and equal to ||x||^2
        let x = array![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0)
        ];

        let result = dotc_c64_ndarray(&x, &x);

        // Should be purely real
        assert!(result.im.abs() < 1e-10);

        // Should equal sum of |x_i|^2 = (1+4) + (9+16) + (25+36) = 5 + 25 + 61 = 91
        assert!((result.re - 91.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_c64_ndarray() {
        // ||x||_2 = sqrt(sum(|x_i|^2))
        let x = array![Complex64::new(3.0, 4.0)]; // |3+4i| = 5
        let norm = nrm2_c64_ndarray(&x);
        assert!((norm - 5.0).abs() < 1e-10);

        let x = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let norm = nrm2_c64_ndarray(&x);
        // sqrt(1 + 1) = sqrt(2)
        assert!((norm - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_c32_ndarray() {
        let x = array![Complex32::new(3.0, 4.0)];
        let norm = nrm2_c32_ndarray(&x);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_asum_c64_ndarray() {
        // sum of |x_i|
        let x = array![Complex64::new(3.0, 4.0), Complex64::new(5.0, 12.0)];
        // |3+4i| = 5, |5+12i| = 13
        let sum = asum_c64_ndarray(&x);
        assert!((sum - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_c32_ndarray() {
        let x = array![Complex32::new(3.0, 4.0), Complex32::new(5.0, 12.0)];
        let sum = asum_c32_ndarray(&x);
        assert!((sum - 18.0).abs() < 1e-5);
    }

    #[test]
    fn test_conj_transpose_c64() {
        let a = array![
            [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]
        ];

        let ah = conj_transpose_c64(&a);
        assert_eq!(ah.dim(), (2, 2));

        // ah[0,0] = conj(a[0,0]) = 1-2i
        assert!((ah[[0, 0]].re - 1.0).abs() < 1e-10);
        assert!((ah[[0, 0]].im - (-2.0)).abs() < 1e-10);

        // ah[0,1] = conj(a[1,0]) = 5-6i
        assert!((ah[[0, 1]].re - 5.0).abs() < 1e-10);
        assert!((ah[[0, 1]].im - (-6.0)).abs() < 1e-10);

        // ah[1,0] = conj(a[0,1]) = 3-4i
        assert!((ah[[1, 0]].re - 3.0).abs() < 1e-10);
        assert!((ah[[1, 0]].im - (-4.0)).abs() < 1e-10);

        // ah[1,1] = conj(a[1,1]) = 7-8i
        assert!((ah[[1, 1]].re - 7.0).abs() < 1e-10);
        assert!((ah[[1, 1]].im - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_conj_transpose_c64_rectangular() {
        let a = array![
            [
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0)
            ],
            [
                Complex64::new(4.0, 4.0),
                Complex64::new(5.0, 5.0),
                Complex64::new(6.0, 6.0)
            ]
        ];

        let ah = conj_transpose_c64(&a);
        assert_eq!(ah.dim(), (3, 2));

        // ah[2,1] = conj(a[1,2]) = 6-6i
        assert!((ah[[2, 1]].re - 6.0).abs() < 1e-10);
        assert!((ah[[2, 1]].im - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm_c64() {
        let a = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)]
        ];
        // |1|^2 + |i|^2 + |i|^2 + |1|^2 = 1 + 1 + 1 + 1 = 4
        let norm = frobenius_norm_c64(&a);
        assert!((norm - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm_c32() {
        let a = array![
            [Complex32::new(3.0, 4.0)] // |3+4i| = 5, |3+4i|^2 = 25
        ];
        let norm = frobenius_norm_c32(&a);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_1_c64() {
        let a = array![
            [Complex64::new(3.0, 4.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(5.0, 12.0)]
        ];
        // col 0: |3+4i| + |0| = 5 + 0 = 5
        // col 1: |i| + |5+12i| = 1 + 13 = 14
        let norm = norm_1_c64(&a);
        assert!((norm - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_inf_c64() {
        let a = array![
            [Complex64::new(3.0, 4.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(5.0, 12.0)]
        ];
        // row 0: |3+4i| + |i| = 5 + 1 = 6
        // row 1: |0| + |5+12i| = 0 + 13 = 13
        let norm = norm_inf_c64(&a);
        assert!((norm - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_max_c64() {
        let a = array![
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 4.0)],
            [Complex64::new(5.0, 12.0), Complex64::new(0.0, 1.0)]
        ];
        // max(|1|, |3+4i|, |5+12i|, |i|) = max(1, 5, 13, 1) = 13
        let max = norm_max_c64(&a);
        assert!((max - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_c64() {
        let a = array![
            [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]
        ];
        // trace = (1+2i) + (7+8i) = 8 + 10i
        let tr = trace_c64(&a);
        assert!((tr.re - 8.0).abs() < 1e-10);
        assert!((tr.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_c32() {
        let a = array![
            [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
            [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)]
        ];
        let tr = trace_c32(&a);
        assert!((tr.re - 8.0).abs() < 1e-5);
        assert!((tr.im - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_scal_c64_ndarray() {
        let mut x = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let alpha = Complex64::new(2.0, 0.0);
        scal_c64_ndarray(alpha, &mut x);

        assert!((x[0].re - 2.0).abs() < 1e-10);
        assert!((x[0].im - 4.0).abs() < 1e-10);
        assert!((x[1].re - 6.0).abs() < 1e-10);
        assert!((x[1].im - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_scal_c64_ndarray_complex_alpha() {
        let mut x = array![Complex64::new(1.0, 0.0)];
        let alpha = Complex64::new(0.0, 1.0); // i
        scal_c64_ndarray(alpha, &mut x);

        // i * 1 = i
        assert!((x[0].re - 0.0).abs() < 1e-10);
        assert!((x[0].im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_c64_ndarray() {
        let x = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let mut y = array![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
        let alpha = Complex64::new(2.0, 0.0);

        axpy_c64_ndarray(alpha, &x, &mut y);

        // y = 2*x + y = 2*(1+2i) + (5+6i) = (2+4i) + (5+6i) = 7+10i
        assert!((y[0].re - 7.0).abs() < 1e-10);
        assert!((y[0].im - 10.0).abs() < 1e-10);

        // y = 2*(3+4i) + (7+8i) = (6+8i) + (7+8i) = 13+16i
        assert!((y[1].re - 13.0).abs() < 1e-10);
        assert!((y[1].im - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_c32_ndarray() {
        let x = array![Complex32::new(1.0, 2.0)];
        let mut y = array![Complex32::new(3.0, 4.0)];
        let alpha = Complex32::new(0.0, 1.0); // i

        axpy_c32_ndarray(alpha, &x, &mut y);

        // y = i*(1+2i) + (3+4i) = (i - 2) + (3+4i) = (1) + (5i)
        assert!((y[0].re - 1.0).abs() < 1e-5);
        assert!((y[0].im - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_eye_c64() {
        let id = eye_c64(3);
        assert_eq!(id.dim(), (3, 3));

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((id[[i, j]].re - 1.0).abs() < 1e-10);
                    assert!(id[[i, j]].im.abs() < 1e-10);
                } else {
                    assert!(id[[i, j]].re.abs() < 1e-10);
                    assert!(id[[i, j]].im.abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_eye_c32() {
        let id = eye_c32(2);
        assert_eq!(id.dim(), (2, 2));
        assert!((id[[0, 0]].re - 1.0).abs() < 1e-5);
        assert!((id[[1, 1]].re - 1.0).abs() < 1e-5);
        assert!(id[[0, 1]].re.abs() < 1e-5);
        assert!(id[[1, 0]].re.abs() < 1e-5);
    }

    #[test]
    fn test_dotc_c64_large() {
        // Test with larger arrays to verify SIMD path
        let n = 1000;
        let x: Array1<Complex64> =
            Array1::from_shape_fn(n, |i| Complex64::new(i as f64, (i as f64) * 0.5));
        let y: Array1<Complex64> =
            Array1::from_shape_fn(n, |i| Complex64::new(1.0, 0.1 * i as f64));

        let result = dotc_c64_ndarray(&x, &y);

        // Verify against manual computation
        let expected: Complex64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi.conj() * yi).sum();
        assert!((result.re - expected.re).abs() < 1e-6);
        assert!((result.im - expected.im).abs() < 1e-6);
    }

    #[test]
    fn test_dotu_c64_large() {
        let n = 1000;
        let x: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
            Complex64::new((i % 100) as f64, ((i + 50) % 100) as f64)
        });
        let y: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
            Complex64::new(((i + 25) % 100) as f64, ((i + 75) % 100) as f64)
        });

        let result = dotu_c64_ndarray(&x, &y);

        let expected: Complex64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        assert!((result.re - expected.re).abs() < 1e-6);
        assert!((result.im - expected.im).abs() < 1e-6);
    }

    #[test]
    fn test_hermitian_property() {
        // For a Hermitian matrix A = A^H, verify property holds
        let a = array![
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            [Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)]
        ];

        let ah = conj_transpose_c64(&a);

        // A should equal A^H for Hermitian matrix
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[[i, j]].re - ah[[i, j]].re).abs() < 1e-10);
                assert!((a[[i, j]].im - ah[[i, j]].im).abs() < 1e-10);
            }
        }
    }
}
