//! BLAS Level 1 - Vector Operations.
//!
//! This module provides basic vector operations:
//!
//! - `dot` - Inner product of two vectors
//! - `dotc` - Conjugate dot product (for complex vectors)
//! - `axpy` - y = α·x + y
//! - `scal` - x = α·x
//! - `copy` - Copy a vector
//! - `swap` - Swap two vectors
//! - `nrm2` - Euclidean (L2) norm
//! - `asum` - Sum of absolute values (L1 norm)
//! - `iamax` - Index of maximum absolute value
//! - `rotg` - Generate a Givens plane rotation
//! - `rot` - Apply a plane rotation
//! - `rotmg` - Generate a modified Givens rotation
//! - `rotm` - Apply a modified Givens rotation
//!
//! # Example
//!
//! ```
//! use oxiblas_blas::level1::{dot, axpy, nrm2};
//!
//! let x = [1.0f64, 2.0, 3.0];
//! let mut y = [4.0f64, 5.0, 6.0];
//!
//! // Dot product: x·y = 1*4 + 2*5 + 3*6 = 32
//! let d = dot(&x, &y);
//! assert!((d - 32.0).abs() < 1e-10);
//!
//! // axpy: y = 2*x + y = [6, 9, 12]
//! axpy(2.0, &x, &mut y);
//! assert!((y[0] - 6.0).abs() < 1e-10);
//!
//! // L2 norm: ||x|| = sqrt(1 + 4 + 9) = sqrt(14)
//! let norm = nrm2(&x);
//! assert!((norm - 14.0f64.sqrt()).abs() < 1e-10);
//! ```

mod asum;
mod axpy;
mod copy;
mod dot;
mod dot_extended;
mod iamax;
mod nrm2;
pub mod parallel;
mod rot;
mod scal;

pub use asum::asum;
pub use axpy::{axpy, axpy_f32, axpy_f64};
pub use copy::copy;
pub use dot::{dot, dot_f32, dot_f64, dotc, dotc_c32, dotc_c64, dotu_c32, dotu_c64};
pub use dot_extended::{dot_kahan, dot_pairwise, dsdot, sdsdot};
pub use iamax::{iamax, iamin};
pub use nrm2::{nrm2, nrm2_f32, nrm2_f64, nrm2_sq};
pub use parallel::{asum_par, axpy_par, dot_par, nrm2_par, scal_par};
pub use rot::{RotgResult, RotmParams, rot, rotg, rotm, rotmg};
pub use scal::scal;

/// Swaps two vectors.
///
/// x ↔ y
pub fn swap<T: Copy>(x: &mut [T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");
    for i in 0..x.len() {
        core::mem::swap(&mut x[i], &mut y[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap() {
        let mut x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];

        swap(&mut x, &mut y);

        assert_eq!(x, [4.0, 5.0, 6.0]);
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_level1_workflow() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let mut y = [0.5, 1.0, 1.5, 2.0];

        // y = 2*x + y
        axpy(2.0, &x, &mut y);

        // Expected: [2.5, 5.0, 7.5, 10.0]
        assert!((y[0] - 2.5).abs() < 1e-10);
        assert!((y[1] - 5.0).abs() < 1e-10);
        assert!((y[2] - 7.5).abs() < 1e-10);
        assert!((y[3] - 10.0).abs() < 1e-10);

        // Dot product
        let d = dot(&x, &y);
        // x·y = 1*2.5 + 2*5 + 3*7.5 + 4*10 = 2.5 + 10 + 22.5 + 40 = 75
        assert!((d - 75.0).abs() < 1e-10);

        // Norms
        let n2 = nrm2(&x);
        assert!((n2 - 30.0f64.sqrt()).abs() < 1e-10);

        let n1 = asum(&x);
        assert!((n1 - 10.0).abs() < 1e-10);

        // Max index
        let idx = iamax(&y);
        assert_eq!(idx, 3); // y[3] = 10.0 is largest
    }

    // =============================================================================
    // Memory Bounds Tests
    // =============================================================================

    #[test]
    fn test_empty_vector_operations() {
        // All level 1 operations should handle empty vectors gracefully
        let empty: [f64; 0] = [];
        let mut empty_mut: [f64; 0] = [];

        // dot
        assert_eq!(dot(&empty, &empty), 0.0);

        // axpy (should be no-op)
        axpy(2.0, &empty, &mut empty_mut);

        // nrm2
        assert_eq!(nrm2(&empty), 0.0);

        // asum
        assert_eq!(asum(&empty), 0.0);

        // scal (should be no-op)
        scal(2.0, &mut empty_mut);

        // copy (should be no-op)
        copy(&empty, &mut empty_mut);
    }

    #[test]
    fn test_single_element_operations() {
        let x = [5.0f64];
        let mut y = [3.0f64];

        // dot
        assert!((dot(&x, &y) - 15.0).abs() < 1e-10);

        // axpy
        axpy(2.0, &x, &mut y);
        assert!((y[0] - 13.0).abs() < 1e-10);

        // nrm2
        assert!((nrm2(&x) - 5.0).abs() < 1e-10);

        // asum
        assert!((asum(&x) - 5.0).abs() < 1e-10);

        // iamax
        assert_eq!(iamax(&x), 0);
    }

    #[test]
    fn test_odd_length_vectors() {
        // Test vectors with odd lengths (not divisible by SIMD widths)
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 7 elements
        let mut y = [1.0f64; 7];

        // dot
        let d = dot(&x, &y);
        assert!((d - 28.0).abs() < 1e-10); // 1+2+3+4+5+6+7 = 28

        // axpy
        axpy(1.0, &x, &mut y);
        assert!((y[6] - 8.0).abs() < 1e-10);

        // nrm2
        let norm = nrm2(&x);
        assert!((norm - 140.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_prime_length_vectors() {
        // Test with prime length that won't align with any common SIMD width
        let n = 17; // Prime number
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![1.0; n];

        // dot = 1 + 2 + ... + 17 = 17*18/2 = 153
        assert!((dot(&x, &y) - 153.0).abs() < 1e-10);

        // nrm2 = sqrt(1 + 4 + 9 + ... + 289)
        let expected_sq: f64 = (1..=n).map(|i| (i * i) as f64).sum();
        assert!((nrm2(&x) - expected_sq.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_large_vectors_no_overflow() {
        // Test with large vectors to ensure no buffer overflows
        let n = 10001; // Odd, prime-like
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![0.0; n];

        // axpy
        axpy(2.0, &x, &mut y);
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[n - 1] - 2.0).abs() < 1e-10);

        // dot
        assert!((dot(&x, &x) - n as f64).abs() < 1e-6);

        // nrm2
        assert!((nrm2(&x) - (n as f64).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_swap_empty() {
        let mut x: [f64; 0] = [];
        let mut y: [f64; 0] = [];
        swap(&mut x, &mut y);
        // Should complete without panic
    }

    #[test]
    fn test_copy_bounds() {
        // Ensure copy doesn't access out of bounds
        let src = [1.0, 2.0, 3.0];
        let mut dst = [0.0; 3];
        copy(&src, &mut dst);
        assert_eq!(dst, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scal_bounds() {
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0];
        scal(2.0, &mut x);
        assert_eq!(x, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_iamax_bounds() {
        // Edge case: all same values
        let x = [5.0, 5.0, 5.0];
        assert_eq!(iamax(&x), 0); // First occurrence

        // Edge case: negative max
        let x = [-10.0, -5.0, -1.0];
        assert_eq!(iamax(&x), 0); // |-10| is largest

        // Empty
        let empty: [f64; 0] = [];
        assert_eq!(iamax(&empty), 0);
    }

    #[test]
    fn test_iamin_bounds() {
        // Edge case: all same values
        let x = [5.0, 5.0, 5.0];
        assert_eq!(iamin(&x), 0);

        // Edge case: zero
        let x = [1.0, 0.0, 2.0];
        assert_eq!(iamin(&x), 1);

        // Empty
        let empty: [f64; 0] = [];
        assert_eq!(iamin(&empty), 0);
    }

    #[test]
    fn test_rot_bounds() {
        let mut x = [1.0f64, 2.0];
        let mut y = [3.0f64, 4.0];
        let c = 0.6f64;
        let s = 0.8f64;
        rot(c, s, &mut x, &mut y);
        // x' = c*x + s*y = 0.6*1 + 0.8*3 = 3.0
        // y' = c*y - s*x = 0.6*3 - 0.8*1 = 1.0
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((y[0] - 1.0).abs() < 1e-10);
    }
}
