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
/// ```
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::precond::{Polynomial, PolynomialConfig, PolynomialType};
///
/// // Diagonally dominant tridiagonal matrix [[4,1,0],[1,4,1],[0,1,4]].
/// let matrix = CsrMatrix::new(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0],
/// )
/// .unwrap();
///
/// let config = PolynomialConfig {
///     degree: 5,
///     lambda_min: Some(0.1),
///     lambda_max: Some(6.0),
/// };
/// let poly = Polynomial::new(&matrix, PolynomialType::Chebyshev, config).unwrap();
/// let r = vec![1.0, 1.0, 1.0];
/// let mut z = vec![0.0; 3];
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
    /// Returns error if matrix is not square or has zero diagonal elements
    /// ([`PreconditionerError::ZeroDiagonal`] for a missing or exactly zero
    /// diagonal entry). Only an exact zero is rejected, and the eigenvalue
    /// bounds refer to the scale-free `D⁻¹A`, so `s·A` builds whenever `A`
    /// does and its preconditioner is exactly `1/s` times that of `A` for a
    /// power-of-two `s`.
    ///
    /// For [`PolynomialType::Chebyshev`], the interval `[λmin, λmax]` (from
    /// `config`, with Gershgorin estimates for a bound that is not given) must
    /// satisfy `0 < λmin <= λmax < ∞`, with a finite center `(λmax + λmin)/2`;
    /// otherwise [`PreconditionerError::InvalidMatrix`] is returned (the
    /// variant the constructor already uses for invalid input). Bounds outside
    /// this range make the Chebyshev recurrence divide by zero or produce
    /// non-finite coefficients (`λmin + λmax = 0` gives a zero center,
    /// `λmin > λmax` a negative half-width, NaN or `±∞` propagate), and the
    /// preconditioner is only positive definite for a positive interval. The
    /// degenerate interval `λmin = λmax > 0` is valid: one step is then exact
    /// for a single eigenvalue. [`PolynomialType::Neumann`] does not use the
    /// bounds and does not validate them.
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
                    if diag_val == T::zero() {
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

        if poly_type == PolynomialType::Chebyshev {
            Self::validate_chebyshev_bounds(&lambda_min, &lambda_max, &center, &half_width)?;
        }

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

    /// Checks the Chebyshev interval `[λmin, λmax]` and its derived center and
    /// half-width: `0 < λmin <= λmax`, all four finite (see [`Self::new`]).
    fn validate_chebyshev_bounds(
        lambda_min: &T,
        lambda_max: &T,
        center: &T,
        half_width: &T,
    ) -> Result<(), PreconditionerError> {
        let finite = |v: &T| v.clone() - v.clone() == T::zero();
        let problem = if !finite(lambda_min) || !finite(lambda_max) {
            Some("bounds must be finite")
        } else if !(*lambda_min > T::zero()) {
            Some("lambda_min must be positive (the interval must not contain 0)")
        } else if !(*lambda_max >= *lambda_min) {
            Some("lambda_max must not be smaller than lambda_min")
        } else if !finite(center) || !finite(half_width) {
            Some("the center (lambda_max + lambda_min) / 2 must be finite")
        } else {
            None
        };
        match problem {
            Some(reason) => Err(PreconditionerError::InvalidMatrix(format!(
                "invalid Chebyshev eigenvalue bounds [{lambda_min}, {lambda_max}] of D^-1 A: {reason}"
            ))),
            None => Ok(()),
        }
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

        // Ensure positive bounds for stability (also for an empty matrix,
        // which has no Gershgorin discs)
        if n == 0 || gershgorin_min <= T::zero() {
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
    /// Runs `degree` steps of the Chebyshev iteration (Saad, *Iterative
    /// Methods for Sparse Linear Systems*, Algorithm 12.1) on
    /// `D⁻¹A z = D⁻¹r` from `z = 0`, for the interval `[λmin, λmax]` assumed to
    /// contain the spectrum of `D⁻¹A`. With `c = (λmax + λmin)/2` and
    /// `d = (λmax − λmin)/2`, `k` steps leave the residual `R_k(D⁻¹A) D⁻¹r`,
    /// where `R_k(λ) = T_k((c − λ)/d) / T_k(c/d)` is the polynomial with
    /// `R_k(0) = 1` that is smallest in maximum norm on the interval:
    /// `|R_k| <= 1/T_k(c/d) < 1` there, so the preconditioner is positive
    /// definite on that interval. Degree 0 is plain diagonal scaling and
    /// degree 1 is `D⁻¹r / c`; a zero-width interval stops after one step,
    /// which is then exact.
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

        // θ = 1/c, so the first step is z_1 = θ·D⁻¹r; it is also the first
        // update direction.
        let theta = one.clone() / self.center.clone();
        let mut direction = vec![T::zero(); n];
        for i in 0..n {
            z[i] = theta.clone() * self.diag_inv[i].clone() * r[i].clone();
            direction[i] = z[i].clone();
        }

        if self.degree == 1 || self.half_width == T::zero() {
            return;
        }

        // σ₁ = c/d and ρ₀ = 1/σ₁; then ρ_k = 1/(2σ₁ − ρ_{k−1}) and
        //   dir_k = ρ_k·ρ_{k−1}·dir_{k−1} + (2ρ_k/d)·r_k,
        //   z_{k+1} = z_k + dir_k,
        // with the residual r_k = D⁻¹r − D⁻¹A·z_k.
        let sigma1 = self.center.clone() / self.half_width.clone();
        let mut rho_prev = one.clone() / sigma1.clone();
        let mut temp = vec![T::zero(); n]; // A * z_k

        for _k in 1..self.degree {
            self.spmv(z, &mut temp);

            let rho = one.clone() / (two.clone() * sigma1.clone() - rho_prev.clone());
            let direction_coeff = rho.clone() * rho_prev.clone();
            let residual_coeff = two.clone() * rho.clone() / self.half_width.clone();

            for i in 0..n {
                let residual = self.diag_inv[i].clone() * r[i].clone()
                    - self.diag_inv[i].clone() * temp[i].clone();
                direction[i] = direction_coeff.clone() * direction[i].clone()
                    + residual_coeff.clone() * residual;
                z[i] = z[i].clone() + direction[i].clone();
            }

            rho_prev = rho;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tridiagonal matrix with unit diagonal and off-diagonal `off` (so
    /// `D⁻¹A = A`), whose eigenvalues are `1 + 2·off·cos(jπ/(n+1))`.
    fn unit_diagonal_tridiagonal(n: usize, off: f64) -> CsrMatrix<f64> {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];
        for i in 0..n {
            if i > 0 {
                values.push(off);
                col_indices.push(i - 1);
            }
            values.push(1.0);
            col_indices.push(i);
            if i + 1 < n {
                values.push(off);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
    }

    fn norm2(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Chebyshev polynomial `T_k(x)` for `x >= 1`.
    fn chebyshev_t(k: usize, x: f64) -> f64 {
        (k as f64 * x.acosh()).cosh()
    }

    /// `degree` Chebyshev steps for an interval containing the spectrum of
    /// `D⁻¹A` leave the residual `R_k(D⁻¹A) D⁻¹r` with
    /// `max |R_k| = 1 / T_k(c/d)` on the interval, so
    /// `‖D⁻¹(r − A z)‖ <= ‖D⁻¹r‖ / T_k(c/d)`, and the preconditioner is
    /// positive definite (`rᵀz > 0`). The recurrence used to grow the
    /// normalisation with `2·(d/c)` instead of `2·(c/d)`, which breaks both.
    #[test]
    fn test_chebyshev_residual_meets_chebyshev_bound() {
        let n = 40;
        // Eigenvalues 1 - 0.9 cos(jπ/41) lie in (0.1, 1.9).
        let a = unit_diagonal_tridiagonal(n, -0.45);
        let (lambda_min, lambda_max) = (0.1, 1.9);
        let (c, d) = (
            (lambda_max + lambda_min) / 2.0,
            (lambda_max - lambda_min) / 2.0,
        );
        for degree in 1..=12 {
            let config = PolynomialConfig {
                degree,
                lambda_min: Some(lambda_min),
                lambda_max: Some(lambda_max),
            };
            let poly = Polynomial::new(&a, PolynomialType::Chebyshev, config).unwrap();
            let bound = 1.0 / chebyshev_t(degree, c / d);
            for seed in 0..4u64 {
                let r: Vec<f64> = (0..n)
                    .map(|i| {
                        (((i as u64 + 1) * (seed + 3) * 2_654_435_761) % 1000) as f64 / 500.0 - 1.0
                    })
                    .collect();
                let mut z = vec![0.0; n];
                poly.apply(&r, &mut z);
                let mut az = vec![0.0; n];
                poly.spmv(&z, &mut az);
                let residual: Vec<f64> = r.iter().zip(&az).map(|(ri, ai)| ri - ai).collect();
                let ratio = norm2(&residual) / norm2(&r);
                assert!(
                    ratio <= bound * (1.0 + 1e-10),
                    "degree {degree}, seed {seed}: residual ratio {ratio} exceeds the Chebyshev bound {bound}"
                );
                let rz: f64 = r.iter().zip(&z).map(|(ri, zi)| ri * zi).sum();
                assert!(rz > 0.0, "degree {degree}, seed {seed}: r^T z = {rz}");
            }
        }
    }

    /// One eigenvalue (`λmin = λmax = c`): `z = D⁻¹r / c` is exact and the
    /// zero half-width must not produce NaN.
    #[test]
    fn test_chebyshev_degenerate_interval() {
        let a = unit_diagonal_tridiagonal(5, 0.0);
        let config = PolynomialConfig {
            degree: 4,
            lambda_min: Some(1.0),
            lambda_max: Some(1.0),
        };
        let poly = Polynomial::new(&a, PolynomialType::Chebyshev, config).unwrap();
        let r = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let mut z = vec![0.0; 5];
        poly.apply(&r, &mut z);
        assert_eq!(z, r);
    }

    fn chebyshev_with_bounds(
        lambda_min: Option<f64>,
        lambda_max: Option<f64>,
    ) -> Result<Polynomial<f64>, PreconditionerError> {
        let a = unit_diagonal_tridiagonal(6, -0.25);
        let config = PolynomialConfig {
            degree: 4,
            lambda_min,
            lambda_max,
        };
        Polynomial::new(&a, PolynomialType::Chebyshev, config)
    }

    /// Bounds for which the Chebyshev recurrence divides by zero or produces
    /// non-finite coefficients are rejected by the constructor: a zero center
    /// (`λmin + λmax = 0`), a reversed interval, a non-positive interval (the
    /// Chebyshev preconditioner of an SPD `D⁻¹A` needs `0 < λmin`) and
    /// non-finite bounds. They used to be accepted, and `apply` returned
    /// `Inf`/`NaN`.
    #[test]
    fn test_chebyshev_invalid_bounds_rejected() {
        let invalid: [(f64, f64); 11] = [
            (-1.0, 1.0),
            (2.0, 1.0),
            (1.0, 0.5),
            (0.0, 2.0),
            (-0.5, 2.0),
            (-3.0, -1.0),
            (f64::NAN, 2.0),
            (0.1, f64::NAN),
            (0.1, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (f64::MAX, f64::MAX),
        ];
        for (lambda_min, lambda_max) in invalid {
            let result = chebyshev_with_bounds(Some(lambda_min), Some(lambda_max));
            assert!(
                matches!(result, Err(PreconditionerError::InvalidMatrix(_))),
                "bounds [{lambda_min}, {lambda_max}] must be rejected"
            );
        }
        // A provided bound that is inconsistent with the estimated other one.
        assert!(matches!(
            chebyshev_with_bounds(Some(5.0), None),
            Err(PreconditionerError::InvalidMatrix(_))
        ));
        assert!(matches!(
            chebyshev_with_bounds(None, Some(-1.0)),
            Err(PreconditionerError::InvalidMatrix(_))
        ));
    }

    /// Valid bounds (including the degenerate `λmin = λmax > 0`) build, and
    /// `apply` stays finite.
    #[test]
    fn test_chebyshev_valid_bounds_give_finite_output() {
        let r = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        for (lambda_min, lambda_max) in [
            (Some(0.1), Some(1.9)),
            (Some(1.0), Some(1.0)),
            (Some(1e-300), Some(1e300)),
            (None, None),
            (Some(0.2), None),
            (None, Some(3.0)),
        ] {
            let poly = chebyshev_with_bounds(lambda_min, lambda_max)
                .unwrap_or_else(|e| panic!("bounds {lambda_min:?}, {lambda_max:?} rejected: {e}"));
            let mut z = vec![0.0; r.len()];
            poly.apply(&r, &mut z);
            assert!(
                z.iter().all(|v| v.is_finite()),
                "bounds {lambda_min:?}, {lambda_max:?}: z = {z:?}"
            );
        }
    }

    /// The Neumann series ignores the eigenvalue bounds, so it does not
    /// validate them.
    #[test]
    fn test_neumann_ignores_bounds() {
        let a = unit_diagonal_tridiagonal(4, -0.25);
        let config = PolynomialConfig {
            degree: 3,
            lambda_min: Some(-1.0),
            lambda_max: Some(1.0),
        };
        let poly = Polynomial::new(&a, PolynomialType::Neumann, config).unwrap();
        let mut z = vec![0.0; 4];
        poly.apply(&[1.0, 2.0, 3.0, 4.0], &mut z);
        assert!(z.iter().all(|v| v.is_finite()));
    }
}
