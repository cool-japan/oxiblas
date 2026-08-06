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
//! # Evaluation model
//!
//! Every expression node implements [`Expr::eval_elem`], which computes a single
//! `(row, col)` element of the result by walking the expression tree: leaves read
//! one matrix element, and element-wise nodes combine their children's elements.
//! The default [`Expr::eval_into`] drives this accessor over a caller-provided
//! destination, so an entire chain of element-wise operations is **fused into a
//! single pass with no intermediate `Mat` allocations** — `eval_into` writes each
//! result element exactly once, directly into the destination buffer.
//!
//! This genuine fusion applies to the element-wise family: [`ExprLeaf`],
//! [`ExprAdd`], [`ExprSub`], [`ExprNeg`], [`ExprScale`], [`ExprFma`],
//! [`ExprTranspose`], [`ExprConj`], and [`ExprHermitian`]. For any expression
//! built solely from these nodes, `expr.eval_into(&mut target)` performs zero heap
//! allocations (verified by a counting-allocator regression test).
//!
//! Matrix products ([`ExprMul`], [`ExprGemm`]) are the deliberate exception: their
//! `eval_into` materializes their operands once into temporary buffers and runs a
//! cache-friendly triple loop, so they *do* allocate. When a product appears as a
//! sub-expression inside an element-wise expression it is instead evaluated lazily
//! through [`Expr::eval_elem`] (allocation-free, `O(k)` per output element); if
//! such a product's own operands are themselves products, the inner products are
//! recomputed per access. To avoid that recomputation, evaluate the product first
//! with [`Expr::eval`] and wrap the result as a leaf via [`LazyExt::lazy`].
//!
//! # Algebraic simplifications
//!
//! - **Transpose elimination**: `(A^T)^T` simplifies to `A` (see [`ExprTranspose::simplify`]).
//! - **Scale accumulation**: `a * (b * A)` simplifies to `(a*b) * A` (see [`ExprScale::simplify`]).
//! - **Double negation / conjugation**: `--A`, `conj(conj(A))`, and `(A^H)^H` all simplify away.

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

    /// Computes a single element `(row, col)` of the result lazily.
    ///
    /// This is the fusion primitive: element-wise nodes implement it by combining
    /// their children's elements, so an element-wise expression tree can be
    /// evaluated one output element at a time without materializing any
    /// intermediate matrix. It is cheap (`O(1)`) for leaves and element-wise
    /// nodes, and `O(k)` for a single matrix product. Composing matrix products
    /// through this accessor recomputes shared inner products (see the module
    /// docs), which is why [`ExprMul`]/[`ExprGemm`] materialize their operands in
    /// [`Expr::eval_into`] instead of relying on this method when evaluated on
    /// their own.
    ///
    /// # Panics
    ///
    /// May panic if `row >= self.nrows()` or `col >= self.ncols()`.
    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem;

    /// Evaluates the expression and returns the result as an owned matrix.
    ///
    /// The returned matrix is the output, not an intermediate: for element-wise
    /// expressions the only allocation is this result buffer.
    fn eval(&self) -> Mat<Self::Elem> {
        let mut result = Mat::zeros(self.nrows(), self.ncols());
        self.eval_into(&mut result);
        result
    }

    /// Evaluates the expression into an existing matrix.
    ///
    /// The default implementation is a fused, allocation-free pass: it writes each
    /// result element directly into `target` via [`Expr::eval_elem`]. Matrix
    /// products override this to materialize their operands once for cache
    /// efficiency.
    ///
    /// # Panics
    ///
    /// Panics if `target.shape() != self.shape()`. This is an unconditional check
    /// (active in release builds) so a mismatched destination can never silently
    /// truncate the result.
    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        assert_eq!(
            target.shape(),
            self.shape(),
            "eval_into: target shape must match expression shape"
        );
        let (nrows, ncols) = self.shape();
        // Column-major iteration matches `Mat`'s storage order.
        for col in 0..ncols {
            for row in 0..nrows {
                target[(row, col)] = self.eval_elem(row, col);
            }
        }
    }

    /// Wraps this expression in a transpose expression.
    fn t(self) -> ExprTranspose<Self> {
        ExprTranspose { inner: self }
    }

    /// Scales this expression by a scalar.
    fn scale(self, alpha: Self::Elem) -> ExprScale<Self> {
        ExprScale { inner: self, alpha }
    }

    /// Adds another expression to this one.
    ///
    /// # Panics
    ///
    /// Panics if the two expressions do not have identical shapes. This is an
    /// unconditional check (active in release builds).
    fn add<E: Expr<Elem = Self::Elem>>(self, other: E) -> ExprAdd<Self, E> {
        assert_eq!(
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
    ///
    /// # Panics
    ///
    /// Panics if the two expressions do not have identical shapes. This is an
    /// unconditional check (active in release builds).
    fn sub<E: Expr<Elem = Self::Elem>>(self, other: E) -> ExprSub<Self, E> {
        assert_eq!(
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
    ///
    /// # Panics
    ///
    /// Panics if `self.ncols() != other.nrows()`. This is an unconditional check
    /// (active in release builds).
    fn matmul<E: Expr<Elem = Self::Elem>>(self, other: E) -> ExprMul<Self, E> {
        assert_eq!(
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

    fn eval_elem(&self, row: usize, col: usize) -> T {
        self.mat[(row, col)]
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        // Transpose swaps indices: element (row, col) of A^T is (col, row) of A.
        self.inner.eval_elem(col, row)
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        self.inner.eval_elem(row, col) * self.alpha
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        // Kept as `zero() - x` (rather than `-x`) to preserve the exact IEEE-754
        // sign behaviour of the previous implementation for signed zeros.
        Self::Elem::zero() - self.inner.eval_elem(row, col)
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        self.lhs.eval_elem(row, col) + self.rhs.eval_elem(row, col)
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        self.lhs.eval_elem(row, col) - self.rhs.eval_elem(row, col)
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        // Lazy per-element dot product (used when this product is a sub-expression
        // of an element-wise expression). Reads operands element-by-element, so it
        // allocates nothing; see the module docs for the recomputation caveat when
        // operands are themselves products.
        let k = self.lhs.ncols();
        let mut sum = Self::Elem::zero();
        for kk in 0..k {
            sum += self.lhs.eval_elem(row, kk) * self.rhs.eval_elem(kk, col);
        }
        sum
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        assert_eq!(
            target.shape(),
            self.shape(),
            "eval_into: target shape must match expression shape"
        );
        // Materialize operands once into contiguous buffers: this keeps the hot
        // triple loop cache-friendly and evaluates any product-valued operand a
        // single time (no per-element recomputation).
        let lhs = self.lhs.eval();
        let rhs = self.rhs.eval();
        let k = self.lhs.ncols();
        let (nrows, ncols) = self.shape();

        for col in 0..ncols {
            for row in 0..nrows {
                let mut sum = Self::Elem::zero();
                for kk in 0..k {
                    sum += lhs[(row, kk)] * rhs[(kk, col)];
                }
                target[(row, col)] = sum;
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        ComplexScalar::conj(&self.inner.eval_elem(row, col))
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        // Conjugate transpose swaps indices and conjugates.
        ComplexScalar::conj(&self.inner.eval_elem(col, row))
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
    ///
    /// # Panics
    ///
    /// Panics if `lhs.shape() != rhs.shape()`. This is an unconditional check
    /// (active in release builds).
    pub fn new(lhs: L, rhs: R, alpha: L::Elem, beta: L::Elem) -> Self {
        assert_eq!(
            lhs.shape(),
            rhs.shape(),
            "Matrix dimensions must match for fused multiply-add"
        );
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        // Genuine fusion: the whole `alpha*A + beta*B` element is produced in one
        // step from the operands' elements, with no intermediate matrices.
        self.alpha * self.lhs.eval_elem(row, col) + self.beta * self.rhs.eval_elem(row, col)
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
    ///
    /// # Panics
    ///
    /// Panics if the operand shapes are not compatible for `alpha * A * B + beta *
    /// C` (i.e. `A.ncols() != B.nrows()`, `A.nrows() != C.nrows()`, or `B.ncols()
    /// != C.ncols()`). These are unconditional checks (active in release builds).
    pub fn new(a: A, b: B, c: C, alpha: A::Elem, beta: A::Elem) -> Self {
        assert_eq!(
            a.ncols(),
            b.nrows(),
            "Matrix dimensions must be compatible for multiplication (A.ncols == B.nrows)"
        );
        assert_eq!(
            a.nrows(),
            c.nrows(),
            "Matrix dimensions must match for accumulation (A.nrows == C.nrows)"
        );
        assert_eq!(
            b.ncols(),
            c.ncols(),
            "Matrix dimensions must match for accumulation (B.ncols == C.ncols)"
        );
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

    fn eval_elem(&self, row: usize, col: usize) -> Self::Elem {
        // Lazy per-element evaluation (used when this GEMM is a sub-expression of
        // an element-wise expression). See module docs for the recomputation
        // caveat when the product operands are themselves products.
        let k = self.a.ncols();
        let mut sum = Self::Elem::zero();
        for kk in 0..k {
            sum += self.a.eval_elem(row, kk) * self.b.eval_elem(kk, col);
        }
        self.alpha * sum + self.beta * self.c.eval_elem(row, col)
    }

    fn eval_into(&self, target: &mut Mat<Self::Elem>) {
        assert_eq!(
            target.shape(),
            self.shape(),
            "eval_into: target shape must match expression shape"
        );
        // Materialize operands once (see `ExprMul::eval_into`).
        let a = self.a.eval();
        let b = self.b.eval();
        let c = self.c.eval();
        let k = self.a.ncols();
        let (nrows, ncols) = self.shape();

        for col in 0..ncols {
            for row in 0..nrows {
                let mut sum = Self::Elem::zero();
                for kk in 0..k {
                    sum += a[(row, kk)] * b[(kk, col)];
                }
                target[(row, col)] = self.alpha * sum + self.beta * c[(row, col)];
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

    // -------------------------------------------------------------------------
    // Allocation-counting harness.
    //
    // A thread-local counting allocator wraps the system allocator. Counting is
    // opt-in per thread (`ALLOC_ACTIVE`), so it observes only the allocations
    // performed by the code under measurement on the current test thread and is
    // immune to allocations that the test runner performs on other threads in
    // parallel. Const-initialized thread locals are never lazily heap-allocated,
    // so touching them inside the allocator cannot recurse.
    // -------------------------------------------------------------------------
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
        static ALLOC_ACTIVE: Cell<bool> = const { Cell::new(false) };
    }

    struct CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ALLOC_ACTIVE.with(Cell::get) {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if ALLOC_ACTIVE.with(Cell::get) {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
            System.alloc_zeroed(layout)
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if ALLOC_ACTIVE.with(Cell::get) {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
            System.realloc(ptr, layout, new_size)
        }
    }

    #[global_allocator]
    static COUNTING_GLOBAL: CountingAllocator = CountingAllocator;

    /// Runs `body`, returning its result together with the number of heap
    /// allocations it performed on the current thread.
    fn measure_allocations<R>(body: impl FnOnce() -> R) -> (R, u64) {
        ALLOC_COUNT.with(|c| c.set(0));
        ALLOC_ACTIVE.with(|a| a.set(true));
        let result = body();
        ALLOC_ACTIVE.with(|a| a.set(false));
        let count = ALLOC_COUNT.with(Cell::get);
        (result, count)
    }

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

    // -------------------------------------------------------------------------
    // Regression tests for finding #1 / #2: shape guards must be active in
    // release builds (previously `debug_assert`, so mismatches silently produced
    // truncated results). Tests are built without debug-assertions in release, so
    // `#[should_panic]` here proves the checks are unconditional.
    // -------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "Matrix dimensions must match for addition")]
    fn test_add_shape_mismatch_panics() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0]]); // 1x3
        // Construction itself must reject the mismatch.
        let _ = a.as_ref().lazy().add(b.as_ref().lazy());
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions must match for subtraction")]
    fn test_sub_shape_mismatch_panics() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0]]); // 1x3
        let _ = a.as_ref().lazy().sub(b.as_ref().lazy());
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions must be compatible for multiplication")]
    fn test_matmul_shape_mismatch_panics() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2 (ncols = 2)
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0]]); // 1x3 (nrows = 1)
        let _ = a.as_ref().lazy().matmul(b.as_ref().lazy());
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions must match for fused multiply-add")]
    fn test_fma_shape_mismatch_panics() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0]]); // 1x3
        let _ = fma(2.0, a.as_ref().lazy(), 3.0, b.as_ref().lazy());
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions must be compatible for multiplication")]
    fn test_gemm_shape_mismatch_panics() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2 (ncols = 2)
        let b: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0]]); // 1x3 (nrows = 1)
        let c: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]); // 2x3
        let _ = gemm(
            1.0,
            a.as_ref().lazy(),
            b.as_ref().lazy(),
            1.0,
            c.as_ref().lazy(),
        );
    }

    #[test]
    #[should_panic(expected = "eval_into: target shape must match expression shape")]
    fn test_eval_into_shape_mismatch_panics() {
        // A correctly-shaped expression written into a wrong-shaped target must
        // panic instead of silently iterating only the smaller shape.
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]); // 2x2
        let expr = a.as_ref().lazy() + b.as_ref().lazy();
        let mut target: Mat<f64> = Mat::zeros(3, 3);
        expr.eval_into(&mut target);
    }

    #[test]
    #[should_panic(expected = "eval_into: target shape must match expression shape")]
    fn test_matmul_eval_into_shape_mismatch_panics() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]); // 2x2
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]); // 2x2
        let expr = a.as_ref().lazy().matmul(b.as_ref().lazy()); // 2x2
        let mut target: Mat<f64> = Mat::zeros(2, 3);
        expr.eval_into(&mut target);
    }

    // -------------------------------------------------------------------------
    // Regression tests for finding #3: fused element-wise `eval_into` must not
    // perform any intermediate heap allocation.
    // -------------------------------------------------------------------------

    #[test]
    fn test_fma_zero_intermediate_allocation() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let mut target: Mat<f64> = Mat::zeros(2, 2);

        // Warm up the thread-local counting slots so their (lazy on some targets)
        // initialization is never attributed to the measured region.
        let _ = measure_allocations(|| ());

        let expr = fma(2.0, a.as_ref().lazy(), 3.0, b.as_ref().lazy());
        let ((), allocations) = measure_allocations(|| {
            expr.eval_into(&mut target);
        });

        assert_eq!(
            allocations, 0,
            "fused FMA eval_into must not perform any intermediate heap allocation"
        );
        // Correctness must be preserved (2*A + 3*B).
        assert_eq!(target[(0, 0)], 2.0 * 1.0 + 3.0 * 5.0);
        assert_eq!(target[(0, 1)], 2.0 * 2.0 + 3.0 * 6.0);
        assert_eq!(target[(1, 0)], 2.0 * 3.0 + 3.0 * 7.0);
        assert_eq!(target[(1, 1)], 2.0 * 4.0 + 3.0 * 8.0);
    }

    #[test]
    fn test_elementwise_chain_zero_intermediate_allocation() {
        // A deeper element-wise tree: ((2*A + 3*B) - C)^T, all fused.
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c: Mat<f64> = Mat::from_rows(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let mut target: Mat<f64> = Mat::zeros(2, 2);

        let _ = measure_allocations(|| ());

        // `ExprFma` has no `-` operator overload, so use the `Expr::sub` method.
        let expr = fma(2.0, a.as_ref().lazy(), 3.0, b.as_ref().lazy())
            .sub(c.as_ref().lazy())
            .t();
        let ((), allocations) = measure_allocations(|| {
            expr.eval_into(&mut target);
        });

        assert_eq!(
            allocations, 0,
            "fused element-wise chain eval_into must not allocate"
        );

        // Reference values: M = (2*A + 3*B) - C, result = M^T.
        // M = [[2*1+3*5-1, 2*2+3*6-1], [2*3+3*7-1, 2*4+3*8-1]]
        //   = [[16, 21], [26, 31]]
        // M^T = [[16, 26], [21, 31]]
        assert_eq!(target[(0, 0)], 16.0);
        assert_eq!(target[(1, 0)], 21.0);
        assert_eq!(target[(0, 1)], 26.0);
        assert_eq!(target[(1, 1)], 31.0);
    }

    #[test]
    fn test_counting_allocator_detects_allocation() {
        // Sanity check the harness: an operation that *does* allocate is counted,
        // so a zero count in the fusion tests is meaningful.
        let _ = measure_allocations(|| ());
        let (v, allocations) = measure_allocations(|| {
            let mut v: Vec<u8> = Vec::with_capacity(4096);
            v.push(1);
            v
        });
        assert_eq!(v.len(), 1);
        assert!(
            allocations >= 1,
            "counting allocator must observe a real allocation"
        );
    }
}
