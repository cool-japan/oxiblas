//! COPY: y = x
//!
//! Copies a vector to another vector.

/// Copies vector x to vector y: y = x
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::copy;
///
/// let x = [1.0, 2.0, 3.0];
/// let mut y = [0.0, 0.0, 0.0];
///
/// copy(&x, &mut y);
///
/// assert_eq!(y, [1.0, 2.0, 3.0]);
/// ```
pub fn copy<T: Copy>(x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    y.copy_from_slice(x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_basic() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0; 4];

        copy(&x, &mut y);

        assert_eq!(y, x);
    }

    #[test]
    fn test_copy_f32() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32; 3];

        copy(&x, &mut y);

        assert_eq!(y, x);
    }

    #[test]
    fn test_copy_empty() {
        let x: [f64; 0] = [];
        let mut y: [f64; 0] = [];

        copy(&x, &mut y);
        // Should not panic
    }

    #[test]
    fn test_copy_single() {
        let x = [42.0];
        let mut y = [0.0];

        copy(&x, &mut y);

        assert_eq!(y[0], 42.0);
    }
}
