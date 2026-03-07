//! Tridiagonal eigenvalue decomposition using bisection and inverse iteration.
//!
//! This module provides alternative algorithms for computing eigenvalues and
//! eigenvectors of symmetric tridiagonal matrices:
//!
//! - **Bisection**: Uses Sturm sequences to count eigenvalues and bisection to
//!   locate them to arbitrary precision. Efficient for computing a subset of
//!   eigenvalues (e.g., a specific range or indices).
//!
//! - **Inverse Iteration**: Computes eigenvectors for given eigenvalues using
//!   inverse iteration. Combined with bisection, this provides an alternative
//!   to the implicit QR algorithm.
//!
//! # When to Use
//!
//! - Use bisection when you need only a subset of eigenvalues
//! - Use bisection + inverse iteration when you need specific eigenvalue/eigenvector pairs
//! - Use the QR algorithm (SymmetricEvd) when you need all eigenvalues/eigenvectors
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::{TridiagEvd, EigenvalueSelector};
//!
//! // Compute eigenvalues 2-5 (0-indexed) of a symmetric tridiagonal matrix
//! let diagonal = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let off_diagonal = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
//!
//! let evd = TridiagEvd::compute_selected(
//!     &diagonal,
//!     &off_diagonal,
//!     EigenvalueSelector::IndexRange { low: 2, high: 5 },
//! ).unwrap();
//!
//! // Get eigenvalues in the computed range
//! let eigenvalues = evd.eigenvalues();
//! assert_eq!(eigenvalues.len(), 4); // indices 2, 3, 4, 5
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

/// Maximum bisection iterations (module-level constant).
const MAX_BISECTION_ITER: usize = 1000;

/// Maximum inverse iteration iterations (module-level constant).
const MAX_INVERSE_ITER: usize = 100;

/// Error type for tridiagonal eigenvalue decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TridiagEvdError {
    /// Empty input.
    EmptyInput,
    /// Dimension mismatch between diagonal and off-diagonal.
    DimensionMismatch,
    /// Invalid index range.
    InvalidIndexRange,
    /// Invalid value range.
    InvalidValueRange,
    /// Bisection did not converge.
    BisectionNotConverged,
    /// Inverse iteration did not converge.
    InverseIterationNotConverged,
}

impl core::fmt::Display for TridiagEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Empty input"),
            Self::DimensionMismatch => write!(f, "Off-diagonal must have length n-1"),
            Self::InvalidIndexRange => write!(f, "Invalid index range"),
            Self::InvalidValueRange => write!(f, "Invalid value range (low >= high)"),
            Self::BisectionNotConverged => write!(f, "Bisection did not converge"),
            Self::InverseIterationNotConverged => write!(f, "Inverse iteration did not converge"),
        }
    }
}

impl std::error::Error for TridiagEvdError {}

/// Selector for which eigenvalues to compute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EigenvalueSelector<T> {
    /// Compute all eigenvalues.
    All,
    /// Compute eigenvalues in a value range [low, high].
    ValueRange {
        /// Lower bound of the value range.
        low: T,
        /// Upper bound of the value range.
        high: T,
    },
    /// Compute eigenvalues with indices in [low, high] (0-indexed, ascending order).
    IndexRange {
        /// Lower index (0-indexed).
        low: usize,
        /// Upper index (0-indexed, inclusive).
        high: usize,
    },
}

/// Tridiagonal eigenvalue decomposition result.
#[derive(Debug, Clone)]
pub struct TridiagEvd<T: Scalar> {
    /// Computed eigenvalues (sorted in ascending order).
    eigenvalues: Vec<T>,
    /// Eigenvectors (columns correspond to eigenvalues).
    eigenvectors: Option<Mat<T>>,
    /// Original matrix dimension.
    n: usize,
    /// Index offset for the computed eigenvalues.
    index_offset: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> TridiagEvd<T> {
    /// Compute eigenvalues and eigenvectors using bisection and inverse iteration.
    ///
    /// This computes all eigenvalues and eigenvectors of a symmetric tridiagonal
    /// matrix defined by its diagonal and off-diagonal elements.
    ///
    /// # Arguments
    ///
    /// * `diagonal` - Main diagonal elements (length n)
    /// * `off_diagonal` - Off-diagonal elements (length n-1)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::TridiagEvd;
    ///
    /// let diag = vec![2.0, 3.0, 4.0];
    /// let off_diag = vec![1.0, 1.0];
    ///
    /// let evd = TridiagEvd::compute(&diag, &off_diag).unwrap();
    /// assert_eq!(evd.eigenvalues().len(), 3);
    /// ```
    pub fn compute(diagonal: &[T], off_diagonal: &[T]) -> Result<Self, TridiagEvdError> {
        Self::compute_selected(diagonal, off_diagonal, EigenvalueSelector::All)
    }

    /// Compute eigenvalues only (no eigenvectors).
    ///
    /// More efficient when eigenvectors are not needed.
    pub fn eigenvalues_only(diagonal: &[T], off_diagonal: &[T]) -> Result<Self, TridiagEvdError> {
        Self::compute_eigenvalues_impl(diagonal, off_diagonal, EigenvalueSelector::All)
    }

    /// Compute selected eigenvalues and their eigenvectors.
    ///
    /// # Arguments
    ///
    /// * `diagonal` - Main diagonal elements (length n)
    /// * `off_diagonal` - Off-diagonal elements (length n-1)
    /// * `selector` - Which eigenvalues to compute
    pub fn compute_selected(
        diagonal: &[T],
        off_diagonal: &[T],
        selector: EigenvalueSelector<T>,
    ) -> Result<Self, TridiagEvdError> {
        let n = diagonal.len();

        if n == 0 {
            return Err(TridiagEvdError::EmptyInput);
        }

        if off_diagonal.len() != n.saturating_sub(1) {
            return Err(TridiagEvdError::DimensionMismatch);
        }

        // Handle trivial case
        if n == 1 {
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues: vec![diagonal[0]],
                eigenvectors: Some(eigenvectors),
                n,
                index_offset: 0,
            });
        }

        // Compute eigenvalues using bisection
        let mut result = Self::compute_eigenvalues_impl(diagonal, off_diagonal, selector)?;

        // Compute eigenvectors using inverse iteration
        let eigenvectors = inverse_iteration(
            diagonal,
            off_diagonal,
            &result.eigenvalues,
            MAX_INVERSE_ITER,
        )?;

        result.eigenvectors = Some(eigenvectors);
        Ok(result)
    }

    /// Compute eigenvalues only (internal implementation).
    fn compute_eigenvalues_impl(
        diagonal: &[T],
        off_diagonal: &[T],
        selector: EigenvalueSelector<T>,
    ) -> Result<Self, TridiagEvdError> {
        let n = diagonal.len();

        if n == 0 {
            return Err(TridiagEvdError::EmptyInput);
        }

        if off_diagonal.len() != n.saturating_sub(1) {
            return Err(TridiagEvdError::DimensionMismatch);
        }

        // Handle trivial case
        if n == 1 {
            return Ok(Self {
                eigenvalues: vec![diagonal[0]],
                eigenvectors: None,
                n,
                index_offset: 0,
            });
        }

        // Compute Gershgorin bounds for eigenvalue range
        let (glow, ghigh) = gershgorin_bounds(diagonal, off_diagonal);

        // Determine which eigenvalues to compute
        let (index_low, index_high, value_low, value_high) = match selector {
            EigenvalueSelector::All => (0, n - 1, glow, ghigh),
            EigenvalueSelector::ValueRange { low, high } => {
                if low >= high {
                    return Err(TridiagEvdError::InvalidValueRange);
                }
                // Count eigenvalues to determine index range
                let count_below_low = sturm_count_stable(diagonal, off_diagonal, low);
                let count_below_high = sturm_count_stable(diagonal, off_diagonal, high);
                if count_below_high == count_below_low {
                    // No eigenvalues in range
                    return Ok(Self {
                        eigenvalues: Vec::new(),
                        eigenvectors: None,
                        n,
                        index_offset: count_below_low,
                    });
                }
                (count_below_low, count_below_high - 1, low, high)
            }
            EigenvalueSelector::IndexRange { low, high } => {
                if low > high || high >= n {
                    return Err(TridiagEvdError::InvalidIndexRange);
                }
                (low, high, glow, ghigh)
            }
        };

        // Compute eigenvalues using bisection
        let eigenvalues = bisection_eigenvalues(
            diagonal,
            off_diagonal,
            index_low,
            index_high,
            value_low,
            value_high,
            MAX_BISECTION_ITER,
        )?;

        Ok(Self {
            eigenvalues,
            eigenvectors: None,
            n,
            index_offset: index_low,
        })
    }

    /// Returns the computed eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix (columns correspond to eigenvalues).
    ///
    /// Returns `None` if eigenvectors were not computed.
    pub fn eigenvectors(&self) -> Option<&Mat<T>> {
        self.eigenvectors.as_ref()
    }

    /// Returns the original matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the number of computed eigenvalues.
    pub fn num_eigenvalues(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Returns the index offset for the first computed eigenvalue.
    ///
    /// For example, if computing eigenvalues 5-10, this returns 5.
    pub fn index_offset(&self) -> usize {
        self.index_offset
    }
}

/// Compute Gershgorin bounds for a symmetric tridiagonal matrix.
///
/// Returns (lower_bound, upper_bound) that contain all eigenvalues.
fn gershgorin_bounds<T: Field + Real>(diagonal: &[T], off_diagonal: &[T]) -> (T, T) {
    let n = diagonal.len();

    if n == 0 {
        return (T::zero(), T::zero());
    }

    if n == 1 {
        return (diagonal[0], diagonal[0]);
    }

    // First row
    let mut min = diagonal[0] - Scalar::abs(off_diagonal[0]);
    let mut max = diagonal[0] + Scalar::abs(off_diagonal[0]);

    // Middle rows
    for i in 1..(n - 1) {
        let radius = Scalar::abs(off_diagonal[i - 1]) + Scalar::abs(off_diagonal[i]);
        let low = diagonal[i] - radius;
        let high = diagonal[i] + radius;
        if low < min {
            min = low;
        }
        if high > max {
            max = high;
        }
    }

    // Last row
    let last_low = diagonal[n - 1] - Scalar::abs(off_diagonal[n - 2]);
    let last_high = diagonal[n - 1] + Scalar::abs(off_diagonal[n - 2]);
    if last_low < min {
        min = last_low;
    }
    if last_high > max {
        max = last_high;
    }

    // Add small margin
    let margin = (max - min) * T::from_f64(0.01).unwrap_or(T::zero());
    (min - margin, max + margin)
}

/// Sturm count using the division-based recurrence (more stable).
fn sturm_count_stable<T: Field + Real>(diagonal: &[T], off_diagonal: &[T], x: T) -> usize {
    let n = diagonal.len();
    if n == 0 {
        return 0;
    }

    // Use the recurrence: d_i = a_i - x - e_{i-1}^2 / d_{i-1}
    // Count the number of sign agreements between d_i and d_{i+1}
    // Number of eigenvalues <= x equals number of negative d_i

    let mut count = 0;
    let eps = <T as Scalar>::epsilon();

    // d_0 = a_0 - x
    let mut d = diagonal[0] - x;
    if d < T::zero() {
        count += 1;
    } else if d == T::zero() {
        // Perturb to avoid division by zero
        d = -eps;
        count += 1;
    }

    for i in 1..n {
        let e_sq = off_diagonal[i - 1] * off_diagonal[i - 1];

        // d_i = a_i - x - e_{i-1}^2 / d_{i-1}
        if Scalar::abs(d) < eps {
            // d is very small, treat as signed epsilon
            d = if d >= T::zero() { eps } else { -eps };
        }

        d = (diagonal[i] - x) - e_sq / d;

        if d < T::zero() {
            count += 1;
        } else if d == T::zero() {
            // At an eigenvalue, count it and perturb
            d = -eps;
            count += 1;
        }
    }

    count
}

/// Compute eigenvalues in the given index range using bisection.
fn bisection_eigenvalues<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    index_low: usize,
    index_high: usize,
    value_low: T,
    value_high: T,
    max_iter: usize,
) -> Result<Vec<T>, TridiagEvdError> {
    let num_eigenvalues = index_high - index_low + 1;
    let mut eigenvalues = Vec::with_capacity(num_eigenvalues);

    let eps = <T as Scalar>::epsilon();
    let two = T::one() + T::one();

    // Compute each eigenvalue individually
    for target_index in index_low..=index_high {
        // Find eigenvalue with the given index using bisection
        let mut lo = value_low;
        let mut hi = value_high;

        for _iter in 0..max_iter {
            // Check convergence
            let tol = eps * (Scalar::abs(lo) + Scalar::abs(hi));
            if hi - lo <= tol {
                break;
            }

            let mid = (lo + hi) / two;
            let count = sturm_count_stable(diagonal, off_diagonal, mid);

            if count <= target_index {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        eigenvalues.push((lo + hi) / two);
    }

    Ok(eigenvalues)
}

/// Compute eigenvectors using inverse iteration.
fn inverse_iteration<T: Field + Real + bytemuck::Zeroable>(
    diagonal: &[T],
    off_diagonal: &[T],
    eigenvalues: &[T],
    max_iter: usize,
) -> Result<Mat<T>, TridiagEvdError> {
    let n = diagonal.len();
    let k = eigenvalues.len();

    if k == 0 {
        return Ok(Mat::zeros(n, 0));
    }

    let mut eigenvectors = Mat::zeros(n, k);
    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    for (j, &lambda) in eigenvalues.iter().enumerate() {
        // Initialize with random-ish vector
        let mut v = vec![T::one(); n];
        for i in 0..n {
            // Simple deterministic "random" initialization
            v[i] = T::from_f64(1.0 + 0.1 * (i as f64 % 7.0)).unwrap_or(T::one());
        }

        // Shift the matrix slightly to avoid singularity
        // (T - lambda*I) is singular, so we use (T - (lambda + shift)*I)
        let shift = eps * (T::one() + Scalar::abs(lambda));

        // Build the shifted tridiagonal matrix
        let mut diag_shifted: Vec<T> = diagonal.iter().map(|&d| d - lambda - shift).collect();

        // Apply inverse iteration
        for _iter in 0..max_iter {
            // Solve (T - lambda*I) * y = v using Thomas algorithm
            let y = solve_tridiagonal(&diag_shifted, off_diagonal, &v)?;

            // Normalize
            let norm = vector_norm(&y);
            if norm < tol {
                // Vector collapsed, re-initialize
                for i in 0..n {
                    v[i] = T::from_f64(1.0 + 0.2 * ((i + j) as f64 % 11.0)).unwrap_or(T::one());
                }
                // Adjust shift
                diag_shifted = diagonal
                    .iter()
                    .map(|&d| d - lambda - shift * T::from_f64(2.0).unwrap_or_else(T::zero))
                    .collect();
                continue;
            }

            for i in 0..n {
                v[i] = y[i] / norm;
            }

            // Check convergence by comparing with previous iteration
            // For simplicity, we just iterate a fixed number of times
        }

        // Orthogonalize against previous eigenvectors (Gram-Schmidt)
        for jj in 0..j {
            let mut dot = T::zero();
            for i in 0..n {
                dot = dot + eigenvectors[(i, jj)] * v[i];
            }
            for i in 0..n {
                v[i] = v[i] - dot * eigenvectors[(i, jj)];
            }
        }

        // Re-normalize after orthogonalization
        let norm = vector_norm(&v);
        if norm > tol {
            for i in 0..n {
                eigenvectors[(i, j)] = v[i] / norm;
            }
        } else {
            // Failed to find eigenvector, use last iteration result
            for i in 0..n {
                eigenvectors[(i, j)] = v[i];
            }
        }
    }

    Ok(eigenvectors)
}

/// Solve tridiagonal system using Thomas algorithm.
fn solve_tridiagonal<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    rhs: &[T],
) -> Result<Vec<T>, TridiagEvdError> {
    let n = diagonal.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        if Scalar::abs(diagonal[0]) < <T as Scalar>::epsilon() {
            return Err(TridiagEvdError::InverseIterationNotConverged);
        }
        return Ok(vec![rhs[0] / diagonal[0]]);
    }

    // Forward elimination
    let mut c = vec![T::zero(); n - 1];
    let mut d = vec![T::zero(); n];

    c[0] = off_diagonal[0] / diagonal[0];
    d[0] = rhs[0] / diagonal[0];

    for i in 1..(n - 1) {
        let denom = diagonal[i] - off_diagonal[i - 1] * c[i - 1];
        if Scalar::abs(denom) < <T as Scalar>::epsilon() {
            // Singular or near-singular, use small value
            let sign = if denom >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            let denom = sign * <T as Scalar>::epsilon();
            c[i] = off_diagonal[i] / denom;
            d[i] = (rhs[i] - off_diagonal[i - 1] * d[i - 1]) / denom;
        } else {
            c[i] = off_diagonal[i] / denom;
            d[i] = (rhs[i] - off_diagonal[i - 1] * d[i - 1]) / denom;
        }
    }

    // Last element
    let denom = diagonal[n - 1] - off_diagonal[n - 2] * c[n - 2];
    if Scalar::abs(denom) < <T as Scalar>::epsilon() {
        let sign = if denom >= T::zero() {
            T::one()
        } else {
            -T::one()
        };
        d[n - 1] =
            (rhs[n - 1] - off_diagonal[n - 2] * d[n - 2]) / (sign * <T as Scalar>::epsilon());
    } else {
        d[n - 1] = (rhs[n - 1] - off_diagonal[n - 2] * d[n - 2]) / denom;
    }

    // Back substitution
    let mut x = vec![T::zero(); n];
    x[n - 1] = d[n - 1];

    for i in (0..(n - 1)).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }

    Ok(x)
}

/// Compute vector 2-norm.
fn vector_norm<T: Field + Real>(v: &[T]) -> T {
    let mut sum = T::zero();
    for &x in v {
        sum = sum + x * x;
    }
    Real::sqrt(sum)
}

/// Public function to count eigenvalues less than or equal to x.
///
/// Uses Sturm sequence for efficient counting.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements
/// * `x` - Value to compare against
///
/// # Returns
///
/// Number of eigenvalues ≤ x
pub fn count_eigenvalues<T: Field + Real>(diagonal: &[T], off_diagonal: &[T], x: T) -> usize {
    sturm_count_stable(diagonal, off_diagonal, x)
}

/// Compute eigenvalue bounds using Gershgorin circles.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements
///
/// # Returns
///
/// (lower_bound, upper_bound) containing all eigenvalues
pub fn eigenvalue_bounds<T: Field + Real>(diagonal: &[T], off_diagonal: &[T]) -> (T, T) {
    gershgorin_bounds(diagonal, off_diagonal)
}

/// Compute eigenvalues in a specified value range.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements
/// * `low` - Lower bound of value range
/// * `high` - Upper bound of value range
///
/// # Returns
///
/// Vector of eigenvalues in [low, high], sorted in ascending order
pub fn eigenvalues_in_range<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    low: T,
    high: T,
) -> Result<Vec<T>, TridiagEvdError> {
    let n = diagonal.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    if low >= high {
        return Err(TridiagEvdError::InvalidValueRange);
    }

    let count_low = sturm_count_stable(diagonal, off_diagonal, low);
    let count_high = sturm_count_stable(diagonal, off_diagonal, high);

    if count_high == count_low {
        return Ok(Vec::new());
    }

    bisection_eigenvalues(
        diagonal,
        off_diagonal,
        count_low,
        count_high - 1,
        low,
        high,
        MAX_BISECTION_ITER,
    )
}

/// Compute eigenvalues by index range.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements
/// * `index_low` - First eigenvalue index (0-indexed)
/// * `index_high` - Last eigenvalue index (0-indexed, inclusive)
///
/// # Returns
///
/// Vector of eigenvalues with indices in [index_low, index_high]
pub fn eigenvalues_by_index<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    index_low: usize,
    index_high: usize,
) -> Result<Vec<T>, TridiagEvdError> {
    let n = diagonal.len();

    if n == 0 {
        return Err(TridiagEvdError::EmptyInput);
    }

    if index_low > index_high || index_high >= n {
        return Err(TridiagEvdError::InvalidIndexRange);
    }

    let (low, high) = gershgorin_bounds(diagonal, off_diagonal);

    bisection_eigenvalues(
        diagonal,
        off_diagonal,
        index_low,
        index_high,
        low,
        high,
        MAX_BISECTION_ITER,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_gershgorin_bounds() {
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![0.5, 0.5];

        let (lo, hi) = gershgorin_bounds(&diag, &off_diag);

        // All eigenvalues should be in [lo, hi]
        assert!(lo < 1.0);
        assert!(hi > 3.0);
    }

    #[test]
    fn test_sturm_count_2x2() {
        // Matrix [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        assert_eq!(sturm_count_stable(&diag, &off_diag, 0.5), 0);
        assert_eq!(sturm_count_stable(&diag, &off_diag, 1.5), 1);
        assert_eq!(sturm_count_stable(&diag, &off_diag, 2.5), 1);
        assert_eq!(sturm_count_stable(&diag, &off_diag, 3.5), 2);
    }

    #[test]
    fn test_bisection_2x2() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        let evd = TridiagEvd::eigenvalues_only(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 2);
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_bisection_3x3() {
        // Diagonal matrix
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![0.0, 0.0];

        let evd = TridiagEvd::eigenvalues_only(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 3);
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_inverse_iteration_2x2() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        let evd = TridiagEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();
        let vecs = evd.eigenvectors().unwrap();

        // Check eigenvalues
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));

        // Check eigenvector orthogonality
        let mut dot = 0.0;
        for i in 0..2 {
            dot += vecs[(i, 0)] * vecs[(i, 1)];
        }
        assert!(approx_eq(dot, 0.0, 1e-8));

        // Check normalization
        for j in 0..2 {
            let mut norm = 0.0;
            for i in 0..2 {
                norm += vecs[(i, j)] * vecs[(i, j)];
            }
            assert!(approx_eq(norm, 1.0, 1e-8));
        }
    }

    #[test]
    fn test_selected_by_index() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let off_diag = vec![0.0, 0.0, 0.0, 0.0];

        // Compute only eigenvalues 1-3
        let evd = TridiagEvd::compute_selected(
            &diag,
            &off_diag,
            EigenvalueSelector::IndexRange { low: 1, high: 3 },
        )
        .unwrap();

        let eigs = evd.eigenvalues();
        assert_eq!(eigs.len(), 3);
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 4.0, 1e-10));
    }

    #[test]
    fn test_selected_by_value() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let off_diag = vec![0.0, 0.0, 0.0, 0.0];

        // Compute eigenvalues in [1.5, 3.5]
        let evd = TridiagEvd::compute_selected(
            &diag,
            &off_diag,
            EigenvalueSelector::ValueRange {
                low: 1.5,
                high: 3.5,
            },
        )
        .unwrap();

        let eigs = evd.eigenvalues();
        assert_eq!(eigs.len(), 2);
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_eigenvalues_in_range() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let off_diag = vec![0.0, 0.0, 0.0, 0.0];

        let eigs = eigenvalues_in_range(&diag, &off_diag, 2.5, 4.5).unwrap();
        assert_eq!(eigs.len(), 2);
        assert!(approx_eq(eigs[0], 3.0, 1e-10));
        assert!(approx_eq(eigs[1], 4.0, 1e-10));
    }

    #[test]
    fn test_eigenvalues_by_index() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let off_diag = vec![0.0, 0.0, 0.0, 0.0];

        let eigs = eigenvalues_by_index(&diag, &off_diag, 2, 4).unwrap();
        assert_eq!(eigs.len(), 3);
        assert!(approx_eq(eigs[0], 3.0, 1e-10));
        assert!(approx_eq(eigs[1], 4.0, 1e-10));
        assert!(approx_eq(eigs[2], 5.0, 1e-10));
    }

    #[test]
    fn test_count_eigenvalues() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        assert_eq!(count_eigenvalues(&diag, &off_diag, 0.0), 0);
        assert_eq!(count_eigenvalues(&diag, &off_diag, 1.0), 1);
        assert_eq!(count_eigenvalues(&diag, &off_diag, 2.0), 1);
        assert_eq!(count_eigenvalues(&diag, &off_diag, 3.0), 2);
    }

    #[test]
    fn test_single_element() {
        let diag = vec![5.0];
        let off_diag: Vec<f64> = vec![];

        let evd = TridiagEvd::compute(&diag, &off_diag).unwrap();
        assert_eq!(evd.eigenvalues().len(), 1);
        assert!(approx_eq(evd.eigenvalues()[0], 5.0, 1e-10));
    }

    #[test]
    fn test_larger_matrix() {
        // Symmetric tridiagonal with known structure
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let off_diag: Vec<f64> = vec![0.0; n - 1];

        let evd = TridiagEvd::eigenvalues_only(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), n);
        for (i, &e) in eigs.iter().enumerate() {
            assert!(approx_eq(e, (i + 1) as f64, 1e-10));
        }
    }

    #[test]
    fn test_f32() {
        let diag = vec![2.0f32, 2.0];
        let off_diag = vec![1.0f32];

        let evd = TridiagEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_repeated_eigenvalues() {
        // Matrix with repeated eigenvalue
        let diag = vec![2.0, 2.0, 2.0];
        let off_diag = vec![0.0, 0.0];

        let evd = TridiagEvd::eigenvalues_only(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        assert_eq!(eigs.len(), 3);
        for &e in eigs {
            assert!(approx_eq(e, 2.0, 1e-10));
        }
    }

    #[test]
    fn test_non_trivial_tridiagonal() {
        // Non-trivial tridiagonal matrix
        let diag = vec![4.0, 3.0, 2.0, 1.0];
        let off_diag = vec![1.0, 2.0, 1.0];

        let evd = TridiagEvd::compute(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();
        let vecs = evd.eigenvectors().unwrap();

        // Verify eigenvalue equation: T * v = lambda * v
        for (j, &lambda) in eigs.iter().enumerate() {
            for i in 0..4 {
                let mut tv_i = diag[i] * vecs[(i, j)];
                if i > 0 {
                    tv_i += off_diag[i - 1] * vecs[(i - 1, j)];
                }
                if i < 3 {
                    tv_i += off_diag[i] * vecs[(i + 1, j)];
                }

                let lambda_v_i = lambda * vecs[(i, j)];
                assert!(
                    approx_eq(tv_i, lambda_v_i, 1e-6),
                    "T*v[{},{}] = {}, lambda*v = {}, diff = {}",
                    i,
                    j,
                    tv_i,
                    lambda_v_i,
                    (tv_i - lambda_v_i).abs()
                );
            }
        }
    }

    #[test]
    fn test_negative_eigenvalues() {
        // Matrix with negative eigenvalues
        let diag = vec![-2.0, -2.0];
        let off_diag = vec![1.0];

        let evd = TridiagEvd::eigenvalues_only(&diag, &off_diag).unwrap();
        let eigs = evd.eigenvalues();

        // Eigenvalues: -3 and -1
        assert!(approx_eq(eigs[0], -3.0, 1e-10));
        assert!(approx_eq(eigs[1], -1.0, 1e-10));
    }

    #[test]
    fn test_empty_range() {
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![0.0, 0.0];

        // No eigenvalues in [4, 5]
        let evd = TridiagEvd::compute_selected(
            &diag,
            &off_diag,
            EigenvalueSelector::ValueRange {
                low: 4.0,
                high: 5.0,
            },
        )
        .unwrap();

        assert_eq!(evd.eigenvalues().len(), 0);
    }

    #[test]
    fn test_error_conditions() {
        // Empty input
        let empty: Vec<f64> = vec![];
        assert!(matches!(
            TridiagEvd::compute(&empty, &empty),
            Err(TridiagEvdError::EmptyInput)
        ));

        // Dimension mismatch
        let diag = vec![1.0, 2.0, 3.0];
        let off_diag = vec![1.0]; // Should be length 2
        assert!(matches!(
            TridiagEvd::compute(&diag, &off_diag),
            Err(TridiagEvdError::DimensionMismatch)
        ));

        // Invalid index range
        let off_diag = vec![1.0, 1.0];
        assert!(matches!(
            TridiagEvd::compute_selected(
                &diag,
                &off_diag,
                EigenvalueSelector::IndexRange { low: 5, high: 6 }
            ),
            Err(TridiagEvdError::InvalidIndexRange)
        ));

        // Invalid value range
        assert!(matches!(
            TridiagEvd::compute_selected(
                &diag,
                &off_diag,
                EigenvalueSelector::ValueRange {
                    low: 5.0,
                    high: 1.0
                }
            ),
            Err(TridiagEvdError::InvalidValueRange)
        ));
    }
}
