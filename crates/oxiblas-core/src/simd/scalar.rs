//! Scalar fallback implementations for platforms without SIMD support.
//!
//! These types provide a consistent interface when SIMD is not available,
//! allowing the same code to work on any platform.

use crate::simd::SimdRegister;

/// Scalar "register" for f64 - processes one element at a time.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ScalarF64(pub f64);

impl SimdRegister for ScalarF64 {
    type Scalar = f64;
    const LANES: usize = 1;

    #[inline]
    fn zero() -> Self {
        ScalarF64(0.0)
    }

    #[inline]
    fn splat(value: f64) -> Self {
        ScalarF64(value)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self {
        unsafe { ScalarF64(*ptr) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        unsafe { ScalarF64(*ptr) }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f64) {
        unsafe { *ptr = self.0 };
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f64) {
        unsafe { *ptr = self.0 };
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        ScalarF64(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        ScalarF64(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        ScalarF64(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        ScalarF64(self.0 / other.0)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        ScalarF64(self.0.mul_add(a.0, b.0))
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        ScalarF64(self.0.mul_add(a.0, -b.0))
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        ScalarF64((-self.0).mul_add(a.0, b.0))
    }

    #[inline]
    fn reduce_sum(self) -> f64 {
        self.0
    }

    #[inline]
    fn reduce_max(self) -> f64 {
        self.0
    }

    #[inline]
    fn reduce_min(self) -> f64 {
        self.0
    }

    #[inline]
    fn extract(self, _index: usize) -> f64 {
        self.0
    }

    #[inline]
    fn insert(self, _index: usize, value: f64) -> Self {
        ScalarF64(value)
    }
}

/// Scalar "register" for f32 - processes one element at a time.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ScalarF32(pub f32);

impl SimdRegister for ScalarF32 {
    type Scalar = f32;
    const LANES: usize = 1;

    #[inline]
    fn zero() -> Self {
        ScalarF32(0.0)
    }

    #[inline]
    fn splat(value: f32) -> Self {
        ScalarF32(value)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        unsafe { ScalarF32(*ptr) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        unsafe { ScalarF32(*ptr) }
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { *ptr = self.0 };
    }

    #[inline]
    unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe { *ptr = self.0 };
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        ScalarF32(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        ScalarF32(self.0 - other.0)
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        ScalarF32(self.0 * other.0)
    }

    #[inline]
    fn div(self, other: Self) -> Self {
        ScalarF32(self.0 / other.0)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        ScalarF32(self.0.mul_add(a.0, b.0))
    }

    #[inline]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        ScalarF32(self.0.mul_add(a.0, -b.0))
    }

    #[inline]
    fn neg_mul_add(self, a: Self, b: Self) -> Self {
        ScalarF32((-self.0).mul_add(a.0, b.0))
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        self.0
    }

    #[inline]
    fn reduce_max(self) -> f32 {
        self.0
    }

    #[inline]
    fn reduce_min(self) -> f32 {
        self.0
    }

    #[inline]
    fn extract(self, _index: usize) -> f32 {
        self.0
    }

    #[inline]
    fn insert(self, _index: usize, value: f32) -> Self {
        ScalarF32(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_f64_basic() {
        let a = ScalarF64::splat(2.0);
        let b = ScalarF64::splat(3.0);

        assert_eq!(a.add(b).0, 5.0);
        assert_eq!(a.mul(b).0, 6.0);
        assert_eq!(a.sub(b).0, -1.0);

        // FMA
        let c = ScalarF64::splat(1.0);
        assert_eq!(a.mul_add(b, c).0, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_scalar_f32_basic() {
        let a = ScalarF32::splat(2.0);
        let b = ScalarF32::splat(3.0);

        assert_eq!(a.add(b).0, 5.0);
        assert_eq!(a.mul(b).0, 6.0);
    }

    #[test]
    fn test_scalar_load_store() {
        unsafe {
            let src = 42.0f64;
            let v = ScalarF64::load_unaligned(&src);
            assert_eq!(v.0, 42.0);

            let mut dst = 0.0f64;
            v.store_unaligned(&mut dst);
            assert_eq!(dst, 42.0);
        }
    }
}
