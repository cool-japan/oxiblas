//! Portable, architecture-independent SIMD register emulation.
//!
//! These registers store their lanes in a plain `[T; N]` array and implement
//! every [`SimdRegister`] operation with ordinary scalar arithmetic, one lane
//! at a time. They are the generic backend used for [`SimdScalar`] on every
//! target that has no dedicated intrinsic backend in this crate (that is,
//! everything other than x86_64, AArch64 and wasm32 — e.g. riscv64,
//! powerpc64, s390x, loongarch64, 32-bit ARM, i686, ...).
//!
//! The lane counts match the nominal register widths (`F64x4`/`F32x8` for
//! 256 bits, `F64x8`/`F32x16` for 512 bits) so that
//! [`SimdScalar::LANES_256`] / [`SimdScalar::LANES_512`] agree with the
//! register's [`SimdRegister::LANES`]. The compiler's auto-vectorizer is free
//! to lower the fixed-size lane loops to whatever vector unit the target has.
//!
//! The types are compiled on every architecture (not only on the fallback
//! ones) so they can be unit-tested against the native SIMD backends on the
//! host.

use crate::simd::SimdRegister;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
use crate::simd::SimdScalar;
// Inherent `f32`/`f64` float methods (`mul_add`, `sqrt`, ...) live in `std`,
// so they are called through `num_traits::Float`: with `std` that forwards to
// the inherent method (bit-identical), without it to `libm`.
use num_traits::Float;

/// Cold path for out-of-range lane indices, mirroring the native backends.
#[cold]
#[inline(never)]
#[track_caller]
fn lane_index_out_of_range(index: usize, lanes: usize) -> ! {
    panic!("SIMD lane index {index} out of range (register has {lanes} lanes)");
}

macro_rules! portable_register {
    ($(#[$meta:meta])* $name:ident, $scalar:ty, $lanes:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(transparent)]
        pub struct $name(pub [$scalar; $lanes]);

        impl $name {
            /// Applies `f` lane-wise to `self` and `other`.
            #[inline(always)]
            fn zip_with(self, other: Self, f: impl Fn($scalar, $scalar) -> $scalar) -> Self {
                let mut out = self.0;
                for (dst, src) in out.iter_mut().zip(other.0.iter()) {
                    *dst = f(*dst, *src);
                }
                $name(out)
            }

            /// Applies `f` lane-wise to `self`, `a` and `b`.
            #[inline(always)]
            fn zip3_with(
                self,
                a: Self,
                b: Self,
                f: impl Fn($scalar, $scalar, $scalar) -> $scalar,
            ) -> Self {
                let mut out = self.0;
                for ((dst, x), y) in out.iter_mut().zip(a.0.iter()).zip(b.0.iter()) {
                    *dst = f(*dst, *x, *y);
                }
                $name(out)
            }

            /// Returns the lanes as an array.
            #[inline]
            pub const fn to_array(self) -> [$scalar; $lanes] {
                self.0
            }
        }

        impl SimdRegister for $name {
            type Scalar = $scalar;
            const LANES: usize = $lanes;

            #[inline]
            fn zero() -> Self {
                $name([0.0; $lanes])
            }

            #[inline]
            fn splat(value: $scalar) -> Self {
                $name([value; $lanes])
            }

            #[inline]
            unsafe fn load_aligned(ptr: *const $scalar) -> Self {
                // SAFETY: the caller guarantees `LANES` readable elements.
                // Alignment beyond that of the scalar is not required here.
                unsafe { Self::load_unaligned(ptr) }
            }

            #[inline]
            unsafe fn load_unaligned(ptr: *const $scalar) -> Self {
                // SAFETY: the caller guarantees `ptr` points to `LANES`
                // readable elements; `read_unaligned` imposes no alignment.
                $name(unsafe { core::ptr::read_unaligned(ptr.cast::<[$scalar; $lanes]>()) })
            }

            #[inline]
            unsafe fn store_aligned(self, ptr: *mut $scalar) {
                // SAFETY: forwarded caller contract (`LANES` writable elements).
                unsafe { self.store_unaligned(ptr) }
            }

            #[inline]
            unsafe fn store_unaligned(self, ptr: *mut $scalar) {
                // SAFETY: the caller guarantees `ptr` points to `LANES`
                // writable elements; `write_unaligned` imposes no alignment.
                unsafe { core::ptr::write_unaligned(ptr.cast::<[$scalar; $lanes]>(), self.0) }
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                self.zip_with(other, |x, y| x + y)
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                self.zip_with(other, |x, y| x - y)
            }

            #[inline]
            fn mul(self, other: Self) -> Self {
                self.zip_with(other, |x, y| x * y)
            }

            #[inline]
            fn div(self, other: Self) -> Self {
                self.zip_with(other, |x, y| x / y)
            }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                self.zip3_with(a, b, |s, x, y| Float::mul_add(s, x, y))
            }

            #[inline]
            fn mul_sub(self, a: Self, b: Self) -> Self {
                self.zip3_with(a, b, |s, x, y| Float::mul_add(s, x, -y))
            }

            #[inline]
            fn neg_mul_add(self, a: Self, b: Self) -> Self {
                self.zip3_with(a, b, |s, x, y| Float::mul_add(-s, x, y))
            }

            #[inline]
            fn reduce_sum(self) -> $scalar {
                // Pairwise (tree) reduction, matching the shape of the native
                // horizontal adds more closely than a left fold.
                let mut buf = self.0;
                let mut width = $lanes;
                while width > 1 {
                    let half = width / 2;
                    let (lo, hi) = buf[..width].split_at_mut(half);
                    for (l, h) in lo.iter_mut().zip(hi.iter()) {
                        *l += *h;
                    }
                    width = half;
                }
                buf[0]
            }

            #[inline]
            fn reduce_max(self) -> $scalar {
                let mut acc = self.0[0];
                for &v in &self.0[1..] {
                    if v > acc {
                        acc = v;
                    }
                }
                acc
            }

            #[inline]
            fn reduce_min(self) -> $scalar {
                let mut acc = self.0[0];
                for &v in &self.0[1..] {
                    if v < acc {
                        acc = v;
                    }
                }
                acc
            }

            #[inline]
            fn extract(self, index: usize) -> $scalar {
                match self.0.get(index) {
                    Some(&v) => v,
                    None => lane_index_out_of_range(index, $lanes),
                }
            }

            #[inline]
            fn insert(self, index: usize, value: $scalar) -> Self {
                let mut out = self.0;
                match out.get_mut(index) {
                    Some(slot) => *slot = value,
                    None => lane_index_out_of_range(index, $lanes),
                }
                $name(out)
            }
        }
    };
}

portable_register!(
    /// Portable 256-bit register emulation for `f64` (4 lanes).
    PortableF64x4,
    f64,
    4
);
portable_register!(
    /// Portable 512-bit register emulation for `f64` (8 lanes).
    PortableF64x8,
    f64,
    8
);
portable_register!(
    /// Portable 256-bit register emulation for `f32` (8 lanes).
    PortableF32x8,
    f32,
    8
);
portable_register!(
    /// Portable 512-bit register emulation for `f32` (16 lanes).
    PortableF32x16,
    f32,
    16
);

// Generic fallback: every architecture without a dedicated intrinsic backend
// uses the portable registers. The cfg is the exact complement of the native
// backends' gates in `simd.rs`, so adding a new native backend only requires
// extending this list.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
impl SimdScalar for f64 {
    type Simd256 = PortableF64x4;
    type Simd512 = PortableF64x8;
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
impl SimdScalar for f32 {
    type Simd256 = PortableF32x8;
    type Simd512 = PortableF32x16;
}

#[cfg(test)]
#[path = "portable_tests.rs"]
mod tests;
