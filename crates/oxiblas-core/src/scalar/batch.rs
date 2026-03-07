//! Batch operations, SIMD compatibility, classification, and summation algorithms.

use num_complex::{Complex32, Complex64};

use super::traits::Scalar;

#[cfg(feature = "f16")]
use half::f16;

#[cfg(feature = "f128")]
use super::extended::QuadFloat;

// =============================================================================
// Scalar trait specialization for performance
// =============================================================================

/// Marker trait for types with hardware FMA (fused multiply-add) support.
///
/// Types implementing this trait have efficient hardware FMA instructions,
/// enabling optimized implementations of algorithms like dot products and
/// matrix multiplications.
pub trait HasFastFma: Scalar {}

impl HasFastFma for f32 {}
impl HasFastFma for f64 {}
impl HasFastFma for Complex32 {}
impl HasFastFma for Complex64 {}

/// Marker trait for types that can be efficiently vectorized with SIMD.
///
/// This trait indicates that the type has a natural mapping to SIMD registers
/// and operations.
pub trait SimdCompatible: Scalar {
    /// The preferred SIMD width (number of elements) for this type.
    const SIMD_WIDTH: usize;

    /// Returns true if SIMD operations are beneficial for the given length.
    #[inline]
    fn use_simd_for(len: usize) -> bool {
        len >= Self::SIMD_WIDTH * 2
    }
}

impl SimdCompatible for f32 {
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 8; // AVX2: 256-bit / 32-bit = 8

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 4; // NEON: 128-bit / 32-bit = 4

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 4;
}

impl SimdCompatible for f64 {
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 4; // AVX2: 256-bit / 64-bit = 4

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 2; // NEON: 128-bit / 64-bit = 2

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 2;
}

impl SimdCompatible for Complex32 {
    // Complex types have half the SIMD width due to doubled storage
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 4;

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 2;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 2;
}

impl SimdCompatible for Complex64 {
    #[cfg(target_arch = "x86_64")]
    const SIMD_WIDTH: usize = 2;

    #[cfg(target_arch = "aarch64")]
    const SIMD_WIDTH: usize = 1;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const SIMD_WIDTH: usize = 1;
}

/// Batch operations on scalar arrays for performance-critical code.
///
/// This trait provides optimized implementations of common operations on
/// contiguous arrays of scalars, leveraging SIMD where available.
pub trait ScalarBatch: Scalar + SimdCompatible {
    /// Computes the dot product of two slices.
    ///
    /// # Safety
    /// Both slices must have the same length.
    fn dot_batch(x: &[Self], y: &[Self]) -> Self;

    /// Computes the sum of all elements.
    fn sum_batch(x: &[Self]) -> Self;

    /// Computes the sum of absolute values (L1 norm).
    fn asum_batch(x: &[Self]) -> Self::Real;

    /// Finds the index of the element with maximum absolute value.
    fn iamax_batch(x: &[Self]) -> usize;

    /// Scales a vector: x = alpha * x
    fn scale_batch(alpha: Self, x: &mut [Self]);

    /// AXPY operation: y = alpha * x + y
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]);

    /// Fused multiply-add on arrays: `z[i] = a[i] * b[i] + c[i]`
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]);
}

impl ScalarBatch for f32 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = 0.0f32;
        for i in 0..x.len() {
            sum = x[i].mul_add(y[i], sum);
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|&v| v.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] = alpha.mul_add(x[i], y[i]);
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

impl ScalarBatch for f64 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = 0.0f64;
        for i in 0..x.len() {
            sum = x[i].mul_add(y[i], sum);
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|&v| v.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] = alpha.mul_add(x[i], y[i]);
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

impl ScalarBatch for Complex32 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = Complex32::new(0.0, 0.0);
        for i in 0..x.len() {
            sum += x[i] * y[i];
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|z| z.re.abs() + z.im.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                (a.re.abs() + a.im.abs())
                    .partial_cmp(&(b.re.abs() + b.im.abs()))
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] += alpha * x[i];
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

impl ScalarBatch for Complex64 {
    #[inline]
    fn dot_batch(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..x.len() {
            sum += x[i] * y[i];
        }
        sum
    }

    #[inline]
    fn sum_batch(x: &[Self]) -> Self {
        x.iter().copied().sum()
    }

    #[inline]
    fn asum_batch(x: &[Self]) -> Self::Real {
        x.iter().map(|z| z.re.abs() + z.im.abs()).sum()
    }

    #[inline]
    fn iamax_batch(x: &[Self]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                (a.re.abs() + a.im.abs())
                    .partial_cmp(&(b.re.abs() + b.im.abs()))
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[inline]
    fn scale_batch(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    #[inline]
    fn axpy_batch(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            y[i] += alpha * x[i];
        }
    }

    #[inline]
    fn fma_batch(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

/// Type-level scalar classification for compile-time dispatch.
///
/// This enum enables algorithms to specialize at compile time based on
/// the scalar type's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarClass {
    /// Single-precision real (f32)
    RealF32,
    /// Double-precision real (f64)
    RealF64,
    /// Single-precision complex
    ComplexF32,
    /// Double-precision complex
    ComplexF64,
    /// Half-precision real (f16)
    RealF16,
    /// Quad-precision real (f128)
    RealF128,
    /// Unknown/other type
    Other,
}

/// Trait for compile-time scalar classification.
pub trait ScalarClassify: Scalar {
    /// The compile-time class of this scalar type.
    const CLASS: ScalarClass;

    /// Returns the precision level (1 = lowest, 4 = highest).
    const PRECISION_LEVEL: u8;

    /// Returns the storage size in bytes.
    const STORAGE_BYTES: usize = core::mem::size_of::<Self>();
}

impl ScalarClassify for f32 {
    const CLASS: ScalarClass = ScalarClass::RealF32;
    const PRECISION_LEVEL: u8 = 2;
}

impl ScalarClassify for f64 {
    const CLASS: ScalarClass = ScalarClass::RealF64;
    const PRECISION_LEVEL: u8 = 3;
}

impl ScalarClassify for Complex32 {
    const CLASS: ScalarClass = ScalarClass::ComplexF32;
    const PRECISION_LEVEL: u8 = 2;
}

impl ScalarClassify for Complex64 {
    const CLASS: ScalarClass = ScalarClass::ComplexF64;
    const PRECISION_LEVEL: u8 = 3;
}

#[cfg(feature = "f16")]
impl ScalarClassify for f16 {
    const CLASS: ScalarClass = ScalarClass::RealF16;
    const PRECISION_LEVEL: u8 = 1;
}

#[cfg(feature = "f128")]
impl ScalarClassify for QuadFloat {
    const CLASS: ScalarClass = ScalarClass::RealF128;
    const PRECISION_LEVEL: u8 = 4;
}

/// Unrolling hints for vectorized loops.
///
/// These constants help the compiler make better unrolling decisions
/// for different scalar types.
pub trait UnrollHints: Scalar {
    /// Recommended unroll factor for tight loops.
    const UNROLL_FACTOR: usize;

    /// Recommended chunk size for blocked algorithms.
    const BLOCK_SIZE: usize;

    /// Whether to prefer streaming stores (for large writes).
    const PREFER_STREAMING: bool;
}

impl UnrollHints for f32 {
    const UNROLL_FACTOR: usize = 8;
    const BLOCK_SIZE: usize = 64;
    const PREFER_STREAMING: bool = true;
}

impl UnrollHints for f64 {
    const UNROLL_FACTOR: usize = 4;
    const BLOCK_SIZE: usize = 32;
    const PREFER_STREAMING: bool = true;
}

impl UnrollHints for Complex32 {
    const UNROLL_FACTOR: usize = 4;
    const BLOCK_SIZE: usize = 32;
    const PREFER_STREAMING: bool = true;
}

impl UnrollHints for Complex64 {
    const UNROLL_FACTOR: usize = 2;
    const BLOCK_SIZE: usize = 16;
    const PREFER_STREAMING: bool = true;
}

/// Extended precision accumulation support.
///
/// For algorithms requiring higher precision during intermediate calculations,
/// this trait provides access to an extended precision accumulator type.
pub trait ExtendedPrecision: Scalar {
    /// The type used for extended precision accumulation.
    type Accumulator: Scalar;

    /// Converts a value to the accumulator type.
    fn to_accumulator(self) -> Self::Accumulator;

    /// Converts from the accumulator type back to this type.
    fn from_accumulator(acc: Self::Accumulator) -> Self;
}

impl ExtendedPrecision for f32 {
    type Accumulator = f64;

    #[inline]
    fn to_accumulator(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_accumulator(acc: f64) -> f32 {
        acc as f32
    }
}

impl ExtendedPrecision for f64 {
    // For f64, we use the same type (or could use f128 if available)
    type Accumulator = f64;

    #[inline]
    fn to_accumulator(self) -> f64 {
        self
    }

    #[inline]
    fn from_accumulator(acc: f64) -> f64 {
        acc
    }
}

impl ExtendedPrecision for Complex32 {
    type Accumulator = Complex64;

    #[inline]
    fn to_accumulator(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }

    #[inline]
    fn from_accumulator(acc: Complex64) -> Complex32 {
        Complex32::new(acc.re as f32, acc.im as f32)
    }
}

impl ExtendedPrecision for Complex64 {
    type Accumulator = Complex64;

    #[inline]
    fn to_accumulator(self) -> Complex64 {
        self
    }

    #[inline]
    fn from_accumulator(acc: Complex64) -> Complex64 {
        acc
    }
}

// =============================================================================
// Summation algorithms
// =============================================================================

/// Kahan summation for improved accuracy.
///
/// Uses compensated summation to reduce floating-point errors.
#[derive(Debug, Clone, Copy)]
pub struct KahanSum<T: Scalar> {
    sum: T,
    compensation: T,
}

impl<T: Scalar> Default for KahanSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> KahanSum<T> {
    /// Creates a new Kahan sum accumulator initialized to zero.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            compensation: T::zero(),
        }
    }

    /// Adds a value to the sum with compensation.
    #[inline]
    pub fn add(&mut self, value: T) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Returns the current sum.
    #[inline]
    pub fn sum(self) -> T {
        self.sum
    }
}

/// Pairwise summation for reduced error accumulation.
///
/// Recursively splits the array and sums pairs, reducing error from O(n) to O(log n).
#[inline]
pub fn pairwise_sum<T: Scalar>(values: &[T]) -> T {
    const THRESHOLD: usize = 32;

    if values.is_empty() {
        return T::zero();
    }
    if values.len() <= THRESHOLD {
        return values.iter().copied().fold(T::zero(), |acc, x| acc + x);
    }

    let mid = values.len() / 2;
    pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
}

/// Kahan-Babuska-Klein summation (improved compensated summation).
///
/// Provides even better error bounds than standard Kahan summation.
#[derive(Debug, Clone, Copy)]
pub struct KBKSum<T: Scalar> {
    sum: T,
    cs: T,
    ccs: T,
}

impl<T: Scalar> Default for KBKSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> KBKSum<T> {
    /// Creates a new KBK sum accumulator.
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: T::zero(),
            cs: T::zero(),
            ccs: T::zero(),
        }
    }

    /// Adds a value with double compensation.
    #[inline]
    pub fn add(&mut self, value: T) {
        let t = self.sum + value;
        let c = if Scalar::abs(self.sum) >= Scalar::abs(value) {
            (self.sum - t) + value
        } else {
            (value - t) + self.sum
        };
        self.sum = t;

        let t2 = self.cs + c;
        let cc = if Scalar::abs(self.cs) >= Scalar::abs(c) {
            (self.cs - t2) + c
        } else {
            (c - t2) + self.cs
        };
        self.cs = t2;
        self.ccs += cc;
    }

    /// Returns the compensated sum.
    #[inline]
    pub fn sum(self) -> T {
        self.sum + self.cs + self.ccs
    }
}
