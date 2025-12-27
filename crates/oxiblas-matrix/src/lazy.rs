//! Lazy evaluation for matrix operations.
//!
//! This module provides expression types that defer computation until explicitly evaluated.
//! This enables operation fusion and optimization for chained matrix operations.
//!
//! # Expression Types
//!
//! - [`ExprAdd`]: Element-wise addition
//! - [`ExprSub`]: Element-wise subtraction
//! - [`ExprNeg`]: Element-wise negation
//! - [`ExprScale`]: Scalar multiplication
//! - [`ExprMul`]: Matrix-matrix multiplication
//! - [`ExprTranspose`]: Matrix transpose
//! - [`ExprConj`]: Complex conjugate
//! - [`ExprHermitian`]: Conjugate transpose
//!
//! # Example
//!
//! ```
//! use oxiblas_matrix::{Mat, lazy::*};
//!
//! let a: Mat<f64> = Mat::from_rows(&[
//!     &[1.0, 2.0],
//!     &[3.0, 4.0],
//! ]);
//! let b: Mat<f64> = Mat::from_rows(&[
//!     &[5.0, 6.0],
//!     &[7.0, 8.0],
//! ]);
//!
//! // Build an expression tree (no computation yet)
//! let expr = a.as_ref().lazy() + b.as_ref().lazy();
//!
//! // Evaluate the expression
//! let result: Mat<f64> = expr.eval();
//! assert_eq!(result[(0, 0)], 6.0); // 1 + 5
//! assert_eq!(result[(1, 1)], 12.0); // 4 + 8
//! ```
//!
//! # Optimization
//!
//! Lazy evaluation enables several optimizations:
//!
//! 1. **Fusion**: Multiple element-wise operations can be fused into a single pass
//! 2. **Transpose elimination**: `(A^T)^T` can be simplified to `A`
//! 3. **Scale accumulation**: `a * (b * A)` becomes `(a*b) * A`
//! 4. **Memory efficiency**: Intermediate matrices are not allocated

use crate::mat::Mat;
use crate::mat_ref::MatRef;
use core::marker::PhantomData;
use num_complex::Complex;
use num_traits::Zero;
use oxiblas_core::scalar::Scalar;

// =============================================================================
// Expr trait - Base trait for lazy expressions
// =============================================================================

/// Trait for lazy matrix expressions.
///
/// All expression types implement this trait, providing common methods
/// for querying dimensions and evaluating the expression.
pub trait Expr: Sized {
    /// The element type of the expression.
    type Elem: Scalar + bytemuck::Zeroable + Zero;

    /// Returns the number of rows in the result.
    fn nrows(&self) -> usize;

    /// Returns the number of columns in the result.
    fn ncols(&self) -> usize;

    /// Returns the shape as (nrows, ncols).
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Evaluates the expression and returns the result as an owned matrix.
    fn eval(&self) -> Mat<Self::Elem>;

    /// Evaluates the expression into an existing matrix.
    fn eval_into(&self, target: &mut Mat<Self::Elem>);

    /// Wraps this expression in a transpose expression.
    fn t(self) -> ExprTranspose<Self> {
        ExprTranspose { inner: self }
    }

    /// Scales this expression by a scalar.
    fn scale(self, alpha: Self::Elem) -> ExprScale<Self> {
        ExprScale { inner: self, alpha }
    }

    /// Adds another expression to this one.
    fn add<E: Expr<Elem = Self::Elem>>(self, other: E) -> ExprAdd<Self, E> {
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Matrix dimensions must match for addition"
        );
        ExprAdd {
            lhs: self,
            rhs: other,
        }
    }

    /// Subtracts another expression from this one.
    fn sub<E: Expr<Elem = Self::Elem>>(self, other: E) -> ExprSub<Self, E> {
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Matrix dimensions must match for subtraction"
        );
        ExprSub {
            lhs: self,
            rhs: other,
        }
    }

    /// Negates this expression.
    fn neg(self) -> ExprNeg<Self> {
        ExprNeg { inner: self }
    }

    /// Matrix multiplies this expression with another.
    fn matmul<E: Expr<Elem = Self::Elem>>(self, other: E) -> ExprMul<Self, E> {
        debug_assert_eq!(
            self.ncols(),
            other.nrows(),
            "Matrix dimensions must be compatible for multiplication"
        );
        ExprMul {
            lhs: self,
            rhs: other,
        }
    }
}

/// Extension trait for complex expressions.
pub trait ComplexExpr: Expr
where
    Self::Elem: ComplexScalar,
{
    /// Returns the complex conjugate of this expression.
    fn conj(self) -> ExprConj<Self> {
        ExprConj { inner: self }
    }

    /// Returns the conjugate transpose (Hermitian) of this expression.
    fn h(self) -> ExprHermitian<Self> {
        ExprHermitian { inner: self }
    }
}

// Blanket implementation for complex expressions
impl<E: Expr> ComplexExpr for E where E::Elem: ComplexScalar {}

/// Marker trait for complex scalar types.
pub trait ComplexScalar: Scalar + bytemuck::Zeroable + Zero {
    /// Returns the complex conjugate.
    fn conj(&self) -> Self;
}

impl ComplexScalar for Complex<f32> {
    fn conj(&self) -> Self {
        Complex::conj(self)
    }
}

impl ComplexScalar for Complex<f64> {
    fn conj(&self) -> Self {
        Complex::conj(self)
    }
}

// Real numbers are their own conjugate
impl ComplexScalar for f32 {
    fn conj(&self) -> Self {
        *self
    }
}

impl ComplexScalar for f64 {
    fn conj(&self) -> Self {
        *self
    }
}

// =============================================================================
// ExprLeaf - Wraps a MatRef as a lazy expression
// =============================================================================

/// A leaf expression wrapping a matrix reference.
#[derive(Clone, Copy)]
pub struct ExprLeaf<'a, T: Scalar + bytemuck::Zeroable + Zero> {
    mat: MatRef<'a, T>,
}

impl<'a, T: Scalar + bytemuck::Zeroable + Zero> ExprLeaf<'a, T> {
    /// Creates a new leaf expression from a matrix reference.
    pub fn new(mat: MatRef<'a, T>) -> Self {
        Self { mat }
    }
}

impl<'a, T: Scalar + bytemuck::Zeroable + Zero> Expr for ExprLeaf<'a, T> {
    type Elem = T;

    fn nrows(&self) -> usize {
        self.mat.nrows()
    }

    fn ncols(&self) -> usize {
        self.mat.ncols()
    }

    fn eval(&self) -> Mat<T> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<T>) {
        target.copy_from(&self.mat);
    }
}

/// Trait extension for MatRef to create lazy expressions.
pub trait LazyExt<'a, T: Scalar + bytemuck::Zeroable + Zero> {
    /// Creates a lazy expression from this matrix reference.
    fn lazy(self) -> ExprLeaf<'a, T>;
}

impl<'a, T: Scalar + bytemuck::Zeroable + Zero> LazyExt<'a, T> for MatRef<'a, T> {
    fn lazy(self) -> ExprLeaf<'a, T> {
        ExprLeaf::new(self)
    }
}

// =============================================================================
// ExprTranspose - Transpose expression
// =============================================================================

/// A lazy transpose expression.
pub struct ExprTranspose<E: Expr> {
    inner: E,
}

impl<E: Expr> Expr for ExprTranspose<E> {
    type Elem = E::Elem;

    fn nrows(&self) -> usize {
        self.inner.ncols()
    }

    fn ncols(&self) -> usize {
        self.inner.nrows()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let inner = self.inner.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = inner[(j, i)];
            }
        }
    }
}

// Double transpose optimization: (A^T)^T = A
impl<E: Expr> ExprTranspose<ExprTranspose<E>> {
    /// Eliminates double transpose.
    pub fn simplify(self) -> E {
        self.inner.inner
    }
}

// =============================================================================
// ExprScale - Scalar multiplication expression
// =============================================================================

/// A lazy scalar multiplication expression.
pub struct ExprScale<E: Expr> {
    inner: E,
    alpha: E::Elem,
}

impl<E: Expr> Expr for ExprScale<E> {
    type Elem = E::Elem;

    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let inner = self.inner.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = inner[(i, j)] * self.alpha;
            }
        }
    }
}

// Double scale optimization: a * (b * A) = (a*b) * A
impl<E: Expr> ExprScale<ExprScale<E>> {
    /// Combines nested scales.
    pub fn simplify(self) -> ExprScale<E> {
        ExprScale {
            inner: self.inner.inner,
            alpha: self.alpha * self.inner.alpha,
        }
    }
}

// =============================================================================
// ExprNeg - Negation expression
// =============================================================================

/// A lazy negation expression.
pub struct ExprNeg<E: Expr> {
    inner: E,
}

impl<E: Expr> Expr for ExprNeg<E> {
    type Elem = E::Elem;

    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let inner = self.inner.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = E::Elem::zero() - inner[(i, j)];
            }
        }
    }
}

// Double negation optimization: --A = A
impl<E: Expr> ExprNeg<ExprNeg<E>> {
    /// Eliminates double negation.
    pub fn simplify(self) -> E {
        self.inner.inner
    }
}

// =============================================================================
// ExprAdd - Addition expression
// =============================================================================

/// A lazy addition expression.
pub struct ExprAdd<L: Expr, R: Expr<Elem = L::Elem>> {
    lhs: L,
    rhs: R,
}

impl<L: Expr, R: Expr<Elem = L::Elem>> Expr for ExprAdd<L, R> {
    type Elem = L::Elem;

    fn nrows(&self) -> usize {
        self.lhs.nrows()
    }

    fn ncols(&self) -> usize {
        self.lhs.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let lhs = self.lhs.eval();
        let rhs = self.rhs.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = lhs[(i, j)] + rhs[(i, j)];
            }
        }
    }
}

// =============================================================================
// ExprSub - Subtraction expression
// =============================================================================

/// A lazy subtraction expression.
pub struct ExprSub<L: Expr, R: Expr<Elem = L::Elem>> {
    lhs: L,
    rhs: R,
}

impl<L: Expr, R: Expr<Elem = L::Elem>> Expr for ExprSub<L, R> {
    type Elem = L::Elem;

    fn nrows(&self) -> usize {
        self.lhs.nrows()
    }

    fn ncols(&self) -> usize {
        self.lhs.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let lhs = self.lhs.eval();
        let rhs = self.rhs.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = lhs[(i, j)] - rhs[(i, j)];
            }
        }
    }
}

// =============================================================================
// ExprMul - Matrix multiplication expression
// =============================================================================

/// A lazy matrix multiplication expression.
pub struct ExprMul<L: Expr, R: Expr<Elem = L::Elem>> {
    lhs: L,
    rhs: R,
}

impl<L: Expr, R: Expr<Elem = L::Elem>> Expr for ExprMul<L, R> {
    type Elem = L::Elem;

    fn nrows(&self) -> usize {
        self.lhs.nrows()
    }

    fn ncols(&self) -> usize {
        self.rhs.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let lhs = self.lhs.eval();
        let rhs = self.rhs.eval();
        let k = self.lhs.ncols();

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let mut sum = L::Elem::zero();
                for kk in 0..k {
                    sum += lhs[(i, kk)] * rhs[(kk, j)];
                }
                target[(i, j)] = sum;
            }
        }
    }
}

// =============================================================================
// ExprConj - Complex conjugate expression
// =============================================================================

/// A lazy complex conjugate expression.
pub struct ExprConj<E: Expr>
where
    E::Elem: ComplexScalar,
{
    inner: E,
}

impl<E: Expr> Expr for ExprConj<E>
where
    E::Elem: ComplexScalar,
{
    type Elem = E::Elem;

    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let inner = self.inner.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = inner[(i, j)].conj();
            }
        }
    }
}

// Double conjugate optimization: conj(conj(A)) = A
impl<E: Expr> ExprConj<ExprConj<E>>
where
    E::Elem: ComplexScalar,
{
    /// Eliminates double conjugation.
    pub fn simplify(self) -> E {
        self.inner.inner
    }
}

// =============================================================================
// ExprHermitian - Conjugate transpose expression
// =============================================================================

/// A lazy conjugate transpose (Hermitian) expression.
pub struct ExprHermitian<E: Expr>
where
    E::Elem: ComplexScalar,
{
    inner: E,
}

impl<E: Expr> Expr for ExprHermitian<E>
where
    E::Elem: ComplexScalar,
{
    type Elem = E::Elem;

    fn nrows(&self) -> usize {
        self.inner.ncols()
    }

    fn ncols(&self) -> usize {
        self.inner.nrows()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let inner = self.inner.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = inner[(j, i)].conj();
            }
        }
    }
}

// Double Hermitian optimization: (A^H)^H = A
impl<E: Expr> ExprHermitian<ExprHermitian<E>>
where
    E::Elem: ComplexScalar,
{
    /// Eliminates double conjugate transpose.
    pub fn simplify(self) -> E {
        self.inner.inner
    }
}

// =============================================================================
// Operator overloading for expressions
// =============================================================================

impl<'a, T: Scalar + bytemuck::Zeroable + Zero, R: Expr<Elem = T>> core::ops::Add<R>
    for ExprLeaf<'a, T>
{
    type Output = ExprAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        Expr::add(self, rhs)
    }
}

impl<'a, T: Scalar + bytemuck::Zeroable + Zero, R: Expr<Elem = T>> core::ops::Sub<R>
    for ExprLeaf<'a, T>
{
    type Output = ExprSub<Self, R>;

    fn sub(self, rhs: R) -> Self::Output {
        Expr::sub(self, rhs)
    }
}

impl<'a, T: Scalar + bytemuck::Zeroable + Zero> core::ops::Neg for ExprLeaf<'a, T> {
    type Output = ExprNeg<Self>;

    fn neg(self) -> Self::Output {
        Expr::neg(self)
    }
}

// Add for ExprAdd
impl<L1, R1, L2: Expr<Elem = L1::Elem>> core::ops::Add<L2> for ExprAdd<L1, R1>
where
    L1: Expr,
    R1: Expr<Elem = L1::Elem>,
{
    type Output = ExprAdd<Self, L2>;

    fn add(self, rhs: L2) -> Self::Output {
        Expr::add(self, rhs)
    }
}

// Sub for ExprAdd
impl<L1, R1, L2: Expr<Elem = L1::Elem>> core::ops::Sub<L2> for ExprAdd<L1, R1>
where
    L1: Expr,
    R1: Expr<Elem = L1::Elem>,
{
    type Output = ExprSub<Self, L2>;

    fn sub(self, rhs: L2) -> Self::Output {
        Expr::sub(self, rhs)
    }
}

// Neg for ExprAdd
impl<L1, R1> core::ops::Neg for ExprAdd<L1, R1>
where
    L1: Expr,
    R1: Expr<Elem = L1::Elem>,
{
    type Output = ExprNeg<Self>;

    fn neg(self) -> Self::Output {
        Expr::neg(self)
    }
}

// Add for ExprScale
impl<E, R: Expr<Elem = E::Elem>> core::ops::Add<R> for ExprScale<E>
where
    E: Expr,
{
    type Output = ExprAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        Expr::add(self, rhs)
    }
}

// Sub for ExprScale
impl<E, R: Expr<Elem = E::Elem>> core::ops::Sub<R> for ExprScale<E>
where
    E: Expr,
{
    type Output = ExprSub<Self, R>;

    fn sub(self, rhs: R) -> Self::Output {
        Expr::sub(self, rhs)
    }
}

// Neg for ExprScale
impl<E> core::ops::Neg for ExprScale<E>
where
    E: Expr,
{
    type Output = ExprNeg<Self>;

    fn neg(self) -> Self::Output {
        Expr::neg(self)
    }
}

// Add for ExprTranspose
impl<E, R: Expr<Elem = E::Elem>> core::ops::Add<R> for ExprTranspose<E>
where
    E: Expr,
{
    type Output = ExprAdd<Self, R>;

    fn add(self, rhs: R) -> Self::Output {
        Expr::add(self, rhs)
    }
}

// Sub for ExprTranspose
impl<E, R: Expr<Elem = E::Elem>> core::ops::Sub<R> for ExprTranspose<E>
where
    E: Expr,
{
    type Output = ExprSub<Self, R>;

    fn sub(self, rhs: R) -> Self::Output {
        Expr::sub(self, rhs)
    }
}

// Neg for ExprTranspose
impl<E> core::ops::Neg for ExprTranspose<E>
where
    E: Expr,
{
    type Output = ExprNeg<Self>;

    fn neg(self) -> Self::Output {
        Expr::neg(self)
    }
}

// =============================================================================
// FusedExpr - Optimized fused operations
// =============================================================================

/// Fused multiply-add expression: alpha * A + beta * B
pub struct ExprFma<L: Expr, R: Expr<Elem = L::Elem>> {
    lhs: L,
    rhs: R,
    alpha: L::Elem,
    beta: L::Elem,
}

impl<L: Expr, R: Expr<Elem = L::Elem>> ExprFma<L, R> {
    /// Creates a new fused multiply-add expression.
    pub fn new(lhs: L, rhs: R, alpha: L::Elem, beta: L::Elem) -> Self {
        debug_assert_eq!(lhs.shape(), rhs.shape());
        Self {
            lhs,
            rhs,
            alpha,
            beta,
        }
    }
}

impl<L: Expr, R: Expr<Elem = L::Elem>> Expr for ExprFma<L, R> {
    type Elem = L::Elem;

    fn nrows(&self) -> usize {
        self.lhs.nrows()
    }

    fn ncols(&self) -> usize {
        self.lhs.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let lhs = self.lhs.eval();
        let rhs = self.rhs.eval();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                target[(i, j)] = self.alpha * lhs[(i, j)] + self.beta * rhs[(i, j)];
            }
        }
    }
}

/// GEMM expression: alpha * A * B + beta * C
pub struct ExprGemm<A: Expr, B: Expr<Elem = A::Elem>, C: Expr<Elem = A::Elem>> {
    a: A,
    b: B,
    c: C,
    alpha: A::Elem,
    beta: A::Elem,
    _marker: PhantomData<A::Elem>,
}

impl<A: Expr, B: Expr<Elem = A::Elem>, C: Expr<Elem = A::Elem>> ExprGemm<A, B, C> {
    /// Creates a new GEMM expression.
    pub fn new(a: A, b: B, c: C, alpha: A::Elem, beta: A::Elem) -> Self {
        debug_assert_eq!(a.ncols(), b.nrows());
        debug_assert_eq!(a.nrows(), c.nrows());
        debug_assert_eq!(b.ncols(), c.ncols());
        Self {
            a,
            b,
            c,
            alpha,
            beta,
            _marker: PhantomData,
        }
    }
}

impl<A: Expr, B: Expr<Elem = A::Elem>, C: Expr<Elem = A::Elem>> Expr for ExprGemm<A, B, C> {
    type Elem = A::Elem;

    fn nrows(&self) -> usize {
        self.a.nrows()
    }

    fn ncols(&self) -> usize {
        self.b.ncols()
    }

    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        let a = self.a.eval();
        let b = self.b.eval();
        let c = self.c.eval();
        let k = self.a.ncols();

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let mut sum = Self::Elem::zero();
                for kk in 0..k {
                    sum += a[(i, kk)] * b[(kk, j)];
                }
                target[(i, j)] = self.alpha * sum + self.beta * c[(i, j)];
            }
        }
    }
}

// =============================================================================
// Expression builder helpers
// =============================================================================

/// Creates a fused multiply-add expression: alpha * A + beta * B
pub fn fma<L: Expr, R: Expr<Elem = L::Elem>>(
    alpha: L::Elem,
    a: L,
    beta: L::Elem,
    b: R,
) -> ExprFma<L, R> {
    ExprFma::new(a, b, alpha, beta)
}

/// Creates a GEMM expression: alpha * A * B + beta * C
pub fn gemm<A: Expr, B: Expr<Elem = A::Elem>, C: Expr<Elem = A::Elem>>(
    alpha: A::Elem,
    a: A,
    b: B,
    beta: A::Elem,
    c: C,
) -> ExprGemm<A, B, C> {
    ExprGemm::new(a, b, c, alpha, beta)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_leaf() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expr = a.as_ref().lazy();
        let result = expr.eval();

        assert_eq!(result[(0, 0)], 1.0);
        assert_eq!(result[(1, 1)], 4.0);
    }

    #[test]
    fn test_lazy_add() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        let expr = a.as_ref().lazy() + b.as_ref().lazy();
        let result = expr.eval();

        assert_eq!(result[(0, 0)], 6.0); // 1 + 5
        assert_eq!(result[(0, 1)], 8.0); // 2 + 6
        assert_eq!(result[(1, 0)], 10.0); // 3 + 7
        assert_eq!(result[(1, 1)], 12.0); // 4 + 8
    }

    #[test]
    fn test_lazy_sub() {
        let a: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expr = a.as_ref().lazy() - b.as_ref().lazy();
        let result = expr.eval();

        assert_eq!(result[(0, 0)], 4.0); // 5 - 1
        assert_eq!(result[(0, 1)], 4.0); // 6 - 2
        assert_eq!(result[(1, 0)], 4.0); // 7 - 3
        assert_eq!(result[(1, 1)], 4.0); // 8 - 4
    }

    #[test]
    fn test_lazy_neg() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expr = -a.as_ref().lazy();
        let result = expr.eval();

        assert_eq!(result[(0, 0)], -1.0);
        assert_eq!(result[(1, 1)], -4.0);
    }

    #[test]
    fn test_lazy_scale() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expr = a.as_ref().lazy().scale(2.0);
        let result = expr.eval();

        assert_eq!(result[(0, 0)], 2.0);
        assert_eq!(result[(0, 1)], 4.0);
        assert_eq!(result[(1, 0)], 6.0);
        assert_eq!(result[(1, 1)], 8.0);
    }

    #[test]
    fn test_lazy_transpose() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let expr = a.as_ref().lazy().t();
        let result = expr.eval();

        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result[(0, 0)], 1.0);
        assert_eq!(result[(1, 0)], 2.0);
        assert_eq!(result[(2, 0)], 3.0);
        assert_eq!(result[(0, 1)], 4.0);
        assert_eq!(result[(1, 1)], 5.0);
        assert_eq!(result[(2, 1)], 6.0);
    }

    #[test]
    fn test_lazy_matmul() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        let expr = a.as_ref().lazy().matmul(b.as_ref().lazy());
        let result = expr.eval();

        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        assert_eq!(result[(0, 0)], 19.0);
        assert_eq!(result[(0, 1)], 22.0);
        assert_eq!(result[(1, 0)], 43.0);
        assert_eq!(result[(1, 1)], 50.0);
    }

    #[test]
    fn test_lazy_chained() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c: Mat<f64> = Mat::from_rows(&[&[1.0, 1.0], &[1.0, 1.0]]);

        // (A + B) - C
        let expr = (a.as_ref().lazy() + b.as_ref().lazy()) - c.as_ref().lazy();
        let result = expr.eval();

        assert_eq!(result[(0, 0)], 5.0); // 1 + 5 - 1
        assert_eq!(result[(0, 1)], 7.0); // 2 + 6 - 1
        assert_eq!(result[(1, 0)], 9.0); // 3 + 7 - 1
        assert_eq!(result[(1, 1)], 11.0); // 4 + 8 - 1
    }

    #[test]
    fn test_lazy_complex_conj() {
        use num_complex::Complex64;

        let a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        let mut a = a;
        a[(0, 0)] = Complex64::new(1.0, 2.0);
        a[(0, 1)] = Complex64::new(3.0, 4.0);
        a[(1, 0)] = Complex64::new(5.0, 6.0);
        a[(1, 1)] = Complex64::new(7.0, 8.0);

        let expr = a.as_ref().lazy().conj();
        let result = expr.eval();

        assert_eq!(result[(0, 0)], Complex64::new(1.0, -2.0));
        assert_eq!(result[(0, 1)], Complex64::new(3.0, -4.0));
        assert_eq!(result[(1, 0)], Complex64::new(5.0, -6.0));
        assert_eq!(result[(1, 1)], Complex64::new(7.0, -8.0));
    }

    #[test]
    fn test_lazy_hermitian() {
        use num_complex::Complex64;

        let mut a: Mat<Complex64> = Mat::filled(2, 2, Complex64::new(0.0, 0.0));
        a[(0, 0)] = Complex64::new(1.0, 2.0);
        a[(0, 1)] = Complex64::new(3.0, 4.0);
        a[(1, 0)] = Complex64::new(5.0, 6.0);
        a[(1, 1)] = Complex64::new(7.0, 8.0);

        let expr = a.as_ref().lazy().h();
        let result = expr.eval();

        // Transpose + conjugate
        assert_eq!(result.shape(), (2, 2));
        assert_eq!(result[(0, 0)], Complex64::new(1.0, -2.0));
        assert_eq!(result[(1, 0)], Complex64::new(3.0, -4.0));
        assert_eq!(result[(0, 1)], Complex64::new(5.0, -6.0));
        assert_eq!(result[(1, 1)], Complex64::new(7.0, -8.0));
    }

    #[test]
    fn test_double_transpose_simplify() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expr = a.as_ref().lazy().t().t();
        let simplified = expr.simplify();
        let result = simplified.eval();

        assert_eq!(result[(0, 0)], 1.0);
        assert_eq!(result[(1, 1)], 4.0);
    }

    #[test]
    fn test_double_scale_simplify() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let expr = a.as_ref().lazy().scale(2.0).scale(3.0);
        let simplified = expr.simplify();
        let result = simplified.eval();

        // 2.0 * 3.0 = 6.0
        assert_eq!(result[(0, 0)], 6.0);
        assert_eq!(result[(1, 1)], 24.0);
    }

    #[test]
    fn test_fma() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        let expr = fma(2.0, a.as_ref().lazy(), 3.0, b.as_ref().lazy());
        let result = expr.eval();

        // 2*A + 3*B
        assert_eq!(result[(0, 0)], 2.0 * 1.0 + 3.0 * 5.0); // 17
        assert_eq!(result[(0, 1)], 2.0 * 2.0 + 3.0 * 6.0); // 22
        assert_eq!(result[(1, 0)], 2.0 * 3.0 + 3.0 * 7.0); // 27
        assert_eq!(result[(1, 1)], 2.0 * 4.0 + 3.0 * 8.0); // 32
    }

    #[test]
    fn test_gemm() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let c: Mat<f64> = Mat::from_rows(&[&[10.0, 10.0], &[10.0, 10.0]]);

        let expr = gemm(
            2.0,
            a.as_ref().lazy(),
            b.as_ref().lazy(),
            1.0,
            c.as_ref().lazy(),
        );
        let result = expr.eval();

        // 2 * A * I + 1 * C = 2 * A + C
        assert_eq!(result[(0, 0)], 2.0 * 1.0 + 10.0); // 12
        assert_eq!(result[(0, 1)], 2.0 * 2.0 + 10.0); // 14
        assert_eq!(result[(1, 0)], 2.0 * 3.0 + 10.0); // 16
        assert_eq!(result[(1, 1)], 2.0 * 4.0 + 10.0); // 18
    }

    #[test]
    fn test_eval_into() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        let expr = a.as_ref().lazy() + b.as_ref().lazy();
        let mut result: Mat<f64> = Mat::zeros(2, 2);
        expr.eval_into(&mut result);

        assert_eq!(result[(0, 0)], 6.0);
        assert_eq!(result[(1, 1)], 12.0);
    }
}
