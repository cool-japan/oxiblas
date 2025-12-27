//! Polynomial preconditioners (Neumann series, Chebyshev).

use super::types::PreconditionerError;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Configuration for polynomial preconditioners.
pub struct PolynomialConfig<T: Scalar> {
    /// Polynomial degree (default: 3).
    pub degree: usize,
    /// Estimated minimum eigenvalue (default: auto-estimate).
    pub lambda_min: Option<T>,
    /// Estimated maximum eigenvalue (default: auto-estimate).
    pub lambda_max: Option<T>,
}

impl<T: Scalar> Default for PolynomialConfig<T> {
    fn default() -> Self {
        Self {
            degree: 3,
            lambda_min: None,
            lambda_max: None,
        }
    }
}

/// Type of polynomial preconditioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolynomialType {
    /// Neumann series: p(A) = I + (I - D^{-1}A) + (I - D^{-1}A)^2 + ...
    /// Good for diagonally dominant matrices.
    Neumann,
    /// Chebyshev polynomial: optimal polynomial for given eigenvalue bounds.
    /// Requires eigenvalue estimates.
    #[default]
    Chebyshev,
}

/// Polynomial preconditioner.
///
/// Polynomial preconditioners approximate M^{-1} ≈ p(A) where p is a polynomial.
/// They are matrix-free (only require matrix-vector products) and are useful
/// when factorization-based preconditioners are too expensive.
///
/// # Supported Types
///
/// - **Neumann**: Uses truncated Neumann series, good for diagonally dominant matrices.
/// - **Chebyshev**: Uses Chebyshev polynomials for optimal approximation within given eigenvalue bounds.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::precond::{Polynomial, PolynomialConfig, PolynomialType};
///
/// let config = PolynomialConfig {
///     degree: 5,
///     lambda_min: Some(0.1),
///     lambda_max: Some(2.0),
/// };
/// let poly = Polynomial::new(&matrix, PolynomialType::Chebyshev, config)?;
/// let mut z = vec![0.0; n];
/// poly.apply(&r, &mut z);
/// ```
#[derive(Debug, Clone)]
pub struct Polynomial<T: Scalar> {
    /// Reference to the matrix (for matrix-vector products).
    matrix: CsrMatrix<T>,
    /// Inverse diagonal for scaling.
    diag_inv: Vec<T>,
    /// Polynomial type.
    poly_type: PolynomialType,
    /// Polynomial degree.
    degree: usize,
    /// Minimum eigenvalue estimate.
    lambda_min: T,
    /// Maximum eigenvalue estimate.
    lambda_max: T,
    /// Chebyshev center: c = (lambda_max + lambda_min) / 2.
    center: T,
    /// Chebyshev half-width: d = (lambda_max - lambda_min) / 2.
    half_width: T,
}

impl<T: Scalar<Real = T> + Clone + Field + PartialOrd> Polynomial<T> {
    /// Create a new polynomial preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to precondition.
    /// * `poly_type` - Type of polynomial to use.
    /// * `config` - Configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if matrix is not square or has zero diagonal elements.
    pub fn new(
        a: &CsrMatrix<T>,
        poly_type: PolynomialType,
        config: PolynomialConfig<T>,
    ) -> Result<Self, PreconditionerError> {
        if a.nrows() != a.ncols() {
            return Err(PreconditionerError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        let n = a.nrows();
        let mut diag_inv = vec![T::zero(); n];

        // Extract diagonal
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            let mut found = false;
            for k in start..end {
                if a.col_indices()[k] == i {
                    let diag_val = a.values()[k].clone();
                    if Scalar::abs(diag_val.clone()) < T::from_f64(1e-14).unwrap_or(T::zero()) {
                        return Err(PreconditionerError::ZeroDiagonal(i));
                    }
                    diag_inv[i] = T::one() / diag_val;
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(PreconditionerError::ZeroDiagonal(i));
            }
        }

        // Estimate eigenvalues if not provided
        let (lambda_min, lambda_max) = Self::estimate_eigenvalues(a, &diag_inv, &config);

        // Compute Chebyshev parameters
        let two = T::from_f64(2.0).unwrap_or(T::one() + T::one());
        let center = (lambda_max.clone() + lambda_min.clone()) / two.clone();
        let half_width = (lambda_max.clone() - lambda_min.clone()) / two;

        Ok(Self {
            matrix: a.clone(),
            diag_inv,
            poly_type,
            degree: config.degree,
            lambda_min,
            lambda_max,
            center,
            half_width,
        })
    }

    /// Estimate eigenvalue bounds using Gershgorin circles.
    fn estimate_eigenvalues(
        a: &CsrMatrix<T>,
        diag_inv: &[T],
        config: &PolynomialConfig<T>,
    ) -> (T, T) {
        // Use provided values if available
        if let (Some(min), Some(max)) = (config.lambda_min.clone(), config.lambda_max.clone()) {
            return (min, max);
        }

        let n = a.nrows();

        // Use Gershgorin theorem on the scaled matrix D^{-1}A
        // For symmetric positive definite matrices, eigenvalues of D^{-1}A are in (0, 2)
        // for diagonally dominant matrices.

        let mut gershgorin_min = T::from_f64(f64::MAX).unwrap_or(T::one());
        let mut gershgorin_max = T::zero();

        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];

            // Center of Gershgorin circle (scaled diagonal)
            let mut center = T::zero();
            let mut radius = T::zero();

            for k in start..end {
                let j = a.col_indices()[k];
                let val = a.values()[k].clone() * diag_inv[i].clone();

                if j == i {
                    center = val;
                } else {
                    radius = radius + Scalar::abs(val);
                }
            }

            // Gershgorin bounds
            let lower = center.clone() - radius.clone();
            let upper = center + radius;

            if lower < gershgorin_min {
                gershgorin_min = lower;
            }
            if upper > gershgorin_max {
                gershgorin_max = upper;
            }
        }

        // Ensure positive bounds for stability
        if gershgorin_min <= T::zero() {
            gershgorin_min = T::from_f64(0.1).unwrap_or(T::one());
        }
        if gershgorin_max <= gershgorin_min.clone() {
            gershgorin_max = T::from_f64(2.0).unwrap_or(T::one() + T::one());
        }

        // Override with provided values if specified
        let lambda_min = config.lambda_min.clone().unwrap_or(gershgorin_min);
        let lambda_max = config.lambda_max.clone().unwrap_or(gershgorin_max);

        (lambda_min, lambda_max)
    }

    /// Apply the polynomial preconditioner: solve M z = r for z.
    ///
    /// # Panics
    ///
    /// Panics if r and z have different lengths or don't match the matrix size.
    pub fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.matrix.nrows();
        assert_eq!(r.len(), n, "r length must match matrix size");
        assert_eq!(z.len(), n, "z length must match matrix size");

        match self.poly_type {
            PolynomialType::Neumann => self.apply_neumann(r, z),
            PolynomialType::Chebyshev => self.apply_chebyshev(r, z),
        }
    }

    /// Apply Neumann series preconditioner.
    ///
    /// Computes z = (I + (I - D^{-1}A) + (I - D^{-1}A)^2 + ...) * D^{-1} * r
    fn apply_neumann(&self, r: &[T], z: &mut [T]) {
        let n = self.matrix.nrows();

        // Start with z = D^{-1} * r
        for i in 0..n {
            z[i] = self.diag_inv[i].clone() * r[i].clone();
        }

        if self.degree == 0 {
            return;
        }

        // w = (I - D^{-1}A) * z_0 = z_0 - D^{-1}*A*z_0
        let mut w = vec![T::zero(); n];
        let mut temp = vec![T::zero(); n];

        // Compute sum = z_0 + (I - D^{-1}A)*z_0 + (I - D^{-1}A)^2*z_0 + ...
        // Let v_k = (I - D^{-1}A)^k * z_0
        // sum = z_0 + v_1 + v_2 + ...

        // v_0 = z_0 (already computed)
        let mut v = z.to_vec();

        for _ in 0..self.degree {
            // Compute temp = A * v
            self.spmv(&v, &mut temp);

            // Compute w = D^{-1} * temp
            for i in 0..n {
                w[i] = self.diag_inv[i].clone() * temp[i].clone();
            }

            // v = v - w = (I - D^{-1}A) * old_v
            for i in 0..n {
                v[i] = v[i].clone() - w[i].clone();
            }

            // Add to sum
            for i in 0..n {
                z[i] = z[i].clone() + v[i].clone();
            }
        }
    }

    /// Apply Chebyshev polynomial preconditioner.
    ///
    /// Uses the three-term Chebyshev recurrence for optimal polynomial approximation.
    fn apply_chebyshev(&self, r: &[T], z: &mut [T]) {
        let n = self.matrix.nrows();

        if self.degree == 0 {
            // Degree 0: just use diagonal scaling
            for i in 0..n {
                z[i] = self.diag_inv[i].clone() * r[i].clone();
            }
            return;
        }

        let two = T::from_f64(2.0).unwrap_or(T::one() + T::one());
        let one = T::one();

        // Chebyshev parameters
        // θ = 1/c where c is the center
        let theta = one.clone() / self.center.clone();
        // δ = d/c where d is the half-width
        let delta = self.half_width.clone() / self.center.clone();

        // z_0 = θ * D^{-1} * r
        for i in 0..n {
            z[i] = theta.clone() * self.diag_inv[i].clone() * r[i].clone();
        }

        if self.degree == 1 {
            return;
        }

        // Work arrays
        let mut z_prev = vec![T::zero(); n]; // z_{k-1}
        let mut z_curr = z.to_vec(); // z_k
        let mut temp = vec![T::zero(); n]; // A * z_k
        let mut w = vec![T::zero(); n]; // D^{-1} * A * z_k

        // Initial sigma
        let mut sigma_prev = one.clone();
        let mut sigma = delta.clone();

        for _k in 1..self.degree {
            // Compute temp = A * z_curr
            self.spmv(&z_curr, &mut temp);

            // Compute w = D^{-1} * temp
            for i in 0..n {
                w[i] = self.diag_inv[i].clone() * temp[i].clone();
            }

            // Update sigma
            let sigma_next = two.clone() * delta.clone() * sigma.clone() - sigma_prev.clone();

            // Chebyshev coefficient
            let rho = sigma.clone() / sigma_next.clone();

            // z_{k+1} = rho * (2 * θ * (D^{-1}*r - D^{-1}*A*z_k) + 2*σ*z_k) - rho * σ_{k-1}/σ * z_{k-1}
            // Simplified: z_{k+1} = 2*ρ*θ*D^{-1}*r - 2*ρ*θ*w + 2*ρ*δ*z_k - (ρ*σ_{k-1}/σ)*z_{k-1}

            let coeff1 = two.clone() * rho.clone() * theta.clone(); // coefficient for D^{-1}*r
            let coeff2 = two.clone() * rho.clone() * delta.clone(); // coefficient for z_k
            let coeff3 = rho.clone() * sigma_prev.clone() / sigma.clone(); // coefficient for z_{k-1}

            // z_{k+1} = coeff1 * D^{-1}*r - coeff1 * w + coeff2 * z_k - coeff3 * z_{k-1}
            for i in 0..n {
                let dr = self.diag_inv[i].clone() * r[i].clone();
                z[i] = coeff1.clone() * dr - coeff1.clone() * w[i].clone()
                    + coeff2.clone() * z_curr[i].clone()
                    - coeff3.clone() * z_prev[i].clone();
            }

            // Shift for next iteration
            for i in 0..n {
                z_prev[i] = z_curr[i].clone();
                z_curr[i] = z[i].clone();
            }

            sigma_prev = sigma;
            sigma = sigma_next;
        }
    }

    /// Sparse matrix-vector multiplication: y = A * x.
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let n = self.matrix.nrows();

        for i in 0..n {
            let start = self.matrix.row_ptrs()[i];
            let end = self.matrix.row_ptrs()[i + 1];

            let mut sum = T::zero();
            for k in start..end {
                let j = self.matrix.col_indices()[k];
                sum = sum + self.matrix.values()[k].clone() * x[j].clone();
            }
            y[i] = sum;
        }
    }

    /// Returns the polynomial degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the polynomial type.
    pub fn poly_type(&self) -> PolynomialType {
        self.poly_type
    }

    /// Returns the estimated eigenvalue bounds.
    pub fn eigenvalue_bounds(&self) -> (T, T) {
        (self.lambda_min.clone(), self.lambda_max.clone())
    }
}
