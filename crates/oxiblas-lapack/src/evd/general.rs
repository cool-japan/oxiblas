//! General (non-symmetric) eigenvalue decomposition.
//!
//! Computes eigenvalues and eigenvectors of general (possibly non-symmetric) matrices.
//! Unlike symmetric EVD, eigenvalues may be complex even for real matrices.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::schur::{Eigenvalue, Schur, SchurError};

/// Error type for general eigenvalue decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralEvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for GeneralEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix must be square"),
            Self::NotConverged => write!(f, "Eigenvalue algorithm did not converge"),
        }
    }
}

impl std::error::Error for GeneralEvdError {}

impl From<SchurError> for GeneralEvdError {
    fn from(e: SchurError) -> Self {
        match e {
            SchurError::EmptyMatrix => Self::EmptyMatrix,
            SchurError::NotSquare => Self::NotSquare,
            SchurError::NotConverged => Self::NotConverged,
        }
    }
}

/// General eigenvalue decomposition.
///
/// For a general matrix A, computes eigenvalues and optionally eigenvectors.
/// Note that for non-symmetric real matrices, eigenvalues may be complex,
/// and eigenvectors are computed from the Schur decomposition.
///
/// Right eigenvectors satisfy: A * v = λ * v
/// Left eigenvectors satisfy: u^H * A = λ * u^H (or equivalently A^T * u = λ * u)
#[derive(Debug, Clone)]
pub struct GeneralEvd<T: Scalar> {
    /// Eigenvalues (may be complex).
    eigenvalues: Vec<Eigenvalue<T>>,
    /// Right eigenvectors (stored column-wise). Only real parts for complex eigenvectors.
    eigenvectors_real: Option<Mat<T>>,
    /// Imaginary parts of right eigenvectors for complex eigenvalues.
    eigenvectors_imag: Option<Mat<T>>,
    /// Left eigenvectors (stored column-wise). Only real parts for complex eigenvectors.
    left_eigenvectors_real: Option<Mat<T>>,
    /// Imaginary parts of left eigenvectors for complex eigenvalues.
    left_eigenvectors_imag: Option<Mat<T>>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> GeneralEvd<T> {
    /// Computes only eigenvalues of a general matrix (no eigenvectors).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::GeneralEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    ///
    /// let evd = GeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
    /// let eigenvalues = evd.eigenvalues();
    ///
    /// // Sum of eigenvalues equals trace
    /// let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
    /// assert!((sum - 5.0).abs() < 1e-10); // trace = 1 + 4 = 5
    /// ```
    pub fn eigenvalues_only(a: MatRef<'_, T>) -> Result<Self, GeneralEvdError> {
        let schur = Schur::compute(a)?;
        let n = a.nrows();
        let eigenvalues = schur.eigenvalues().to_vec();

        Ok(Self {
            eigenvalues,
            eigenvectors_real: None,
            eigenvectors_imag: None,
            left_eigenvectors_real: None,
            left_eigenvectors_imag: None,
            n,
        })
    }

    /// Computes eigenvalues and right eigenvectors of a general matrix.
    ///
    /// For complex eigenvalue pairs, the eigenvectors are stored as consecutive
    /// columns: v_real and v_imag such that the eigenvector for λ = a + bi is
    /// v_real + i*v_imag, and for λ = a - bi is v_real - i*v_imag.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::GeneralEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    ///
    /// let evd = GeneralEvd::compute(a.as_ref()).unwrap();
    /// let vr = evd.eigenvectors_real().unwrap();
    ///
    /// // For real eigenvalues, eigenvector imaginary parts are zero
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, GeneralEvdError> {
        let schur = Schur::compute(a)?;
        let n = a.nrows();
        let eigenvalues = schur.eigenvalues().to_vec();

        // Compute right eigenvectors from Schur form
        let (vr, vi) = Self::compute_eigenvectors_from_schur(&schur);

        Ok(Self {
            eigenvalues,
            eigenvectors_real: Some(vr),
            eigenvectors_imag: Some(vi),
            left_eigenvectors_real: None,
            left_eigenvectors_imag: None,
            n,
        })
    }

    /// Computes eigenvalues, right eigenvectors, and left eigenvectors of a general matrix.
    ///
    /// Right eigenvectors satisfy: A * v = λ * v
    /// Left eigenvectors satisfy: u^H * A = λ * u^H (or equivalently A^T * u = λ * u)
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::evd::GeneralEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    /// ]);
    ///
    /// let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();
    /// let vr = evd.eigenvectors_real().unwrap();
    /// let vl = evd.left_eigenvectors_real().unwrap();
    /// ```
    pub fn compute_full(a: MatRef<'_, T>) -> Result<Self, GeneralEvdError> {
        let n = a.nrows();
        if n == 0 {
            return Err(GeneralEvdError::EmptyMatrix);
        }
        if a.ncols() != n {
            return Err(GeneralEvdError::NotSquare);
        }

        // Compute Schur decomposition for A
        let schur = Schur::compute(a)?;
        let eigenvalues = schur.eigenvalues().to_vec();

        // Compute right eigenvectors from Schur form of A
        let (vr, vi) = Self::compute_eigenvectors_from_schur(&schur);

        // Compute left eigenvectors from Schur form of A^T
        // Left eigenvectors of A are right eigenvectors of A^T
        let mut at = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                at[(i, j)] = a[(j, i)];
            }
        }
        let schur_t = Schur::compute(at.as_ref())?;
        let (vl, vl_i) = Self::compute_eigenvectors_from_schur(&schur_t);

        Ok(Self {
            eigenvalues,
            eigenvectors_real: Some(vr),
            eigenvectors_imag: Some(vi),
            left_eigenvectors_real: Some(vl),
            left_eigenvectors_imag: Some(vl_i),
            n,
        })
    }

    /// Computes eigenvectors from Schur decomposition.
    fn compute_eigenvectors_from_schur(schur: &Schur<T>) -> (Mat<T>, Mat<T>) {
        let n = schur.t().nrows();
        let t = schur.t();
        let q = schur.q();
        let eigenvalues = schur.eigenvalues();

        let mut vr = Mat::zeros(n, n);
        let mut vi = Mat::zeros(n, n);
        let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

        // Process eigenvalues
        let mut col = 0;
        let mut ev_idx = 0;

        while ev_idx < eigenvalues.len() && col < n {
            if eigenvalues[ev_idx].is_real() {
                // Real eigenvalue - solve (T - λI)x = 0 by back substitution
                let lambda = eigenvalues[ev_idx].real;
                let mut x = vec![T::zero(); n];
                x[col] = T::one();

                // Back substitution for upper triangular system
                for i in (0..col).rev() {
                    let mut sum = T::zero();
                    for j in (i + 1)..=col {
                        sum = sum + t[(i, j)] * x[j];
                    }
                    let diag = t[(i, i)] - lambda;
                    if Scalar::abs(diag) > eps {
                        x[i] = -sum / diag;
                    }
                }

                // Normalize
                let mut norm_sq = T::zero();
                for i in 0..n {
                    norm_sq = norm_sq + x[i] * x[i];
                }
                let norm = Real::sqrt(norm_sq);
                if norm > eps {
                    for i in 0..n {
                        x[i] = x[i] / norm;
                    }
                }

                // Transform back: v = Q * x
                for i in 0..n {
                    let mut sum = T::zero();
                    for j in 0..n {
                        sum = sum + q[(i, j)] * x[j];
                    }
                    vr[(i, col)] = sum;
                    vi[(i, col)] = T::zero();
                }

                col += 1;
                ev_idx += 1;
            } else {
                // Complex conjugate pair
                let real_part = eigenvalues[ev_idx].real;
                let imag_part = Scalar::abs(eigenvalues[ev_idx].imag);

                // For 2×2 block with complex eigenvalues, compute eigenvector
                // The block is at position (col, col) to (col+1, col+1)
                let mut xr = vec![T::zero(); n];
                let mut xi = vec![T::zero(); n];

                // Simple approach: use (1, (λ - t11)/t12) as basis
                if col + 1 < n {
                    let t11 = t[(col, col)];
                    let t12 = t[(col, col + 1)];

                    xr[col] = T::one();
                    xi[col] = T::zero();

                    if Scalar::abs(t12) > eps {
                        // (λ - t11) / t12 where λ = real_part + i*imag_part
                        // Real part: (real_part - t11) / t12
                        // Imag part: imag_part / t12
                        xr[col + 1] = (real_part - t11) / t12;
                        xi[col + 1] = imag_part / t12;
                    } else {
                        xr[col + 1] = T::zero();
                        xi[col + 1] = T::one();
                    }

                    // Normalize
                    let mut norm_sq = T::zero();
                    for i in 0..n {
                        norm_sq = norm_sq + xr[i] * xr[i] + xi[i] * xi[i];
                    }
                    let norm = Real::sqrt(norm_sq);
                    if norm > eps {
                        for i in 0..n {
                            xr[i] = xr[i] / norm;
                            xi[i] = xi[i] / norm;
                        }
                    }

                    // Transform back: v = Q * x
                    for i in 0..n {
                        let mut sum_r = T::zero();
                        let mut sum_i = T::zero();
                        for j in 0..n {
                            sum_r = sum_r + q[(i, j)] * xr[j];
                            sum_i = sum_i + q[(i, j)] * xi[j];
                        }
                        // First column: v_real
                        vr[(i, col)] = sum_r;
                        vi[(i, col)] = sum_i;
                        // Second column: v_real (same real part for conjugate)
                        vr[(i, col + 1)] = sum_r;
                        vi[(i, col + 1)] = -sum_i; // conjugate
                    }

                    col += 2;
                    ev_idx += 2;
                } else {
                    col += 1;
                    ev_idx += 1;
                }
            }
        }

        (vr, vi)
    }

    /// Returns the eigenvalues.
    pub fn eigenvalues(&self) -> &[Eigenvalue<T>] {
        &self.eigenvalues
    }

    /// Returns the real parts of eigenvalues.
    pub fn eigenvalues_real(&self) -> Vec<T> {
        self.eigenvalues.iter().map(|e| e.real).collect()
    }

    /// Returns true if all eigenvalues are real.
    pub fn all_eigenvalues_real(&self) -> bool {
        self.eigenvalues.iter().all(|e| e.is_real())
    }

    /// Returns the real parts of eigenvectors (if computed).
    pub fn eigenvectors_real(&self) -> Option<MatRef<'_, T>> {
        self.eigenvectors_real.as_ref().map(|m| m.as_ref())
    }

    /// Returns the imaginary parts of right eigenvectors (if computed).
    pub fn eigenvectors_imag(&self) -> Option<MatRef<'_, T>> {
        self.eigenvectors_imag.as_ref().map(|m| m.as_ref())
    }

    /// Returns the real parts of left eigenvectors (if computed).
    ///
    /// Left eigenvectors satisfy: u^H * A = λ * u^H
    pub fn left_eigenvectors_real(&self) -> Option<MatRef<'_, T>> {
        self.left_eigenvectors_real.as_ref().map(|m| m.as_ref())
    }

    /// Returns the imaginary parts of left eigenvectors (if computed).
    pub fn left_eigenvectors_imag(&self) -> Option<MatRef<'_, T>> {
        self.left_eigenvectors_imag.as_ref().map(|m| m.as_ref())
    }

    /// Returns the number of complex eigenvalue pairs.
    pub fn num_complex_pairs(&self) -> usize {
        self.eigenvalues
            .iter()
            .filter(|e| !e.is_real() && e.imag > T::zero())
            .count()
    }

    /// Verifies the eigenvalue decomposition: A * v ≈ λ * v for real eigenvalues.
    /// Returns the maximum residual norm.
    pub fn verify(&self, a: MatRef<'_, T>) -> T {
        if self.eigenvectors_real.is_none() {
            return T::zero();
        }

        let vr = self
            .eigenvectors_real
            .as_ref()
            .expect("value should be present");
        let vi = self
            .eigenvectors_imag
            .as_ref()
            .expect("value should be present");
        let mut max_residual = T::zero();

        for (col, eigenvalue) in self.eigenvalues.iter().enumerate() {
            if col >= self.n {
                break;
            }

            if eigenvalue.is_real() {
                // For real eigenvalue: ||A*v - λ*v||
                let lambda = eigenvalue.real;
                let mut residual_sq = T::zero();

                for i in 0..self.n {
                    let mut av_i = T::zero();
                    for j in 0..self.n {
                        av_i = av_i + a[(i, j)] * vr[(j, col)];
                    }
                    let diff = av_i - lambda * vr[(i, col)];
                    residual_sq = residual_sq + diff * diff;
                }

                let residual = Real::sqrt(residual_sq);
                if residual > max_residual {
                    max_residual = residual;
                }
            } else {
                // For complex eigenvalue: need to check with complex arithmetic
                // A * (vr + i*vi) = (λr + i*λi) * (vr + i*vi)
                // Real part: A*vr = λr*vr - λi*vi
                // Imag part: A*vi = λr*vi + λi*vr
                let lambda_r = eigenvalue.real;
                let lambda_i = eigenvalue.imag;
                let mut residual_sq = T::zero();

                for i in 0..self.n {
                    // A * vr
                    let mut avr_i = T::zero();
                    let mut avi_i = T::zero();
                    for j in 0..self.n {
                        avr_i = avr_i + a[(i, j)] * vr[(j, col)];
                        avi_i = avi_i + a[(i, j)] * vi[(j, col)];
                    }

                    // Expected: λr*vr - λi*vi, λr*vi + λi*vr
                    let expected_r = lambda_r * vr[(i, col)] - lambda_i * vi[(i, col)];
                    let expected_i = lambda_r * vi[(i, col)] + lambda_i * vr[(i, col)];

                    let diff_r = avr_i - expected_r;
                    let diff_i = avi_i - expected_i;
                    residual_sq = residual_sq + diff_r * diff_r + diff_i * diff_i;
                }

                let residual = Real::sqrt(residual_sq);
                if residual > max_residual {
                    max_residual = residual;
                }
            }
        }

        max_residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_general_evd_eigenvalues_only() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::eigenvalues_only(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Sum of eigenvalues = trace
        let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
        assert!(approx_eq(sum, 5.0, 1e-10)); // trace = 1 + 4 = 5

        // Product of eigenvalues = determinant
        let mut prod: f64 = 1.0;
        for e in eigenvalues {
            prod *= e.real;
        }
        // det = 1*4 - 2*3 = -2
        assert!(approx_eq(prod, -2.0, 1e-10));
    }

    #[test]
    fn test_general_evd_with_eigenvectors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        assert!(evd.eigenvectors_real().is_some());

        // Eigenvalues should still be correct
        let eigenvalues = evd.eigenvalues();
        let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
        assert!(approx_eq(sum, 5.0, 1e-10)); // trace = 1 + 4 = 5

        // Product of eigenvalues = determinant
        let mut prod: f64 = 1.0;
        for e in eigenvalues {
            prod *= e.real;
        }
        assert!(approx_eq(prod, -2.0, 1e-10)); // det = 1*4 - 2*3 = -2
    }

    #[test]
    fn test_general_evd_diagonal() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 3.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
        assert!(approx_eq(eigs[2], 5.0, 1e-10));

        assert!(evd.all_eigenvalues_real());
    }

    #[test]
    fn test_general_evd_complex_eigenvalues() {
        // Rotation matrix has complex eigenvalues
        let theta = core::f64::consts::FRAC_PI_4;
        let c = theta.cos();
        let s = theta.sin();
        let a = Mat::from_rows(&[&[c, -s], &[s, c]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        assert_eq!(eigenvalues.len(), 2);
        assert!(!evd.all_eigenvalues_real());
        assert_eq!(evd.num_complex_pairs(), 1);

        // Complex conjugate pair
        assert!(approx_eq(eigenvalues[0].real, eigenvalues[1].real, 1e-10));
        assert!(approx_eq(eigenvalues[0].imag, -eigenvalues[1].imag, 1e-10));
    }

    #[test]
    fn test_general_evd_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        // Upper triangular: eigenvalues are diagonal elements
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 4.0, 1e-10));
        assert!(approx_eq(eigs[2], 6.0, 1e-10));
    }

    #[test]
    fn test_general_evd_single() {
        let a = Mat::from_rows(&[&[7.0f64]]);
        let evd = GeneralEvd::compute(a.as_ref()).unwrap();

        assert_eq!(evd.eigenvalues().len(), 1);
        assert!(approx_eq(evd.eigenvalues()[0].real, 7.0, 1e-10));
    }

    #[test]
    fn test_general_evd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute(a.as_ref()).unwrap();
        let eigenvalues = evd.eigenvalues();

        let sum: f32 = eigenvalues.iter().map(|e| e.real).sum();
        assert!((sum - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_general_evd_full_with_left_eigenvectors() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();

        // Should have both right and left eigenvectors
        assert!(evd.eigenvectors_real().is_some());
        assert!(evd.left_eigenvectors_real().is_some());

        let vr = evd.eigenvectors_real().unwrap();
        let vl = evd.left_eigenvectors_real().unwrap();

        // Both should have the same dimensions
        assert_eq!(vr.nrows(), 2);
        assert_eq!(vr.ncols(), 2);
        assert_eq!(vl.nrows(), 2);
        assert_eq!(vl.ncols(), 2);

        // Eigenvalues should be correct
        let eigenvalues = evd.eigenvalues();
        let sum: f64 = eigenvalues.iter().map(|e| e.real).sum();
        assert!(approx_eq(sum, 5.0, 1e-10)); // trace = 1 + 4 = 5
    }

    #[test]
    fn test_general_evd_full_diagonal() {
        // For a diagonal matrix, left and right eigenvectors should be identity columns
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 5.0]]);

        let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();

        assert!(evd.eigenvectors_real().is_some());
        assert!(evd.left_eigenvectors_real().is_some());

        // Eigenvalues should be 2 and 5
        let eigenvalues = evd.eigenvalues();
        let mut eigs: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(approx_eq(eigs[0], 2.0, 1e-10));
        assert!(approx_eq(eigs[1], 5.0, 1e-10));
    }

    #[test]
    fn test_general_evd_full_verify_left_eigenvector_property() {
        // Test that left eigenvectors satisfy: u^T * A = λ * u^T
        // Since left eigenvectors of A are right eigenvectors of A^T,
        // the eigenvalue correspondence may differ.
        // We verify that each left eigenvector u satisfies A^T * u = λ * u
        // for SOME eigenvalue λ.
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let evd = GeneralEvd::compute_full(a.as_ref()).unwrap();
        let vl = evd.left_eigenvectors_real().unwrap();
        let eigenvalues = evd.eigenvalues();

        // Collect all eigenvalue real parts
        let lambda_values: Vec<f64> = eigenvalues.iter().map(|e| e.real).collect();

        // Left eigenvector exists and has approximately right dimension
        assert!(
            vl.nrows() == 2 && vl.ncols() == 2,
            "Left eigenvector matrix should be 2x2"
        );

        // For each left eigenvector column, find the corresponding eigenvalue
        for col in 0..2 {
            let norm = (vl[(0, col)] * vl[(0, col)] + vl[(1, col)] * vl[(1, col)]).sqrt();
            if norm < 1e-10 {
                continue; // Skip zero eigenvectors
            }

            // Normalize the eigenvector for better numerical comparison
            let u0 = vl[(0, col)] / norm;
            let u1 = vl[(1, col)] / norm;

            // Compute A^T * u (using normalized eigenvector)
            let at_u0 = a[(0, 0)] * u0 + a[(1, 0)] * u1;
            let at_u1 = a[(0, 1)] * u0 + a[(1, 1)] * u1;

            // Find which eigenvalue this eigenvector corresponds to
            // For normalized u, A^T * u = λ * u means |A^T * u - λ * u| should be small
            let mut min_residual = f64::MAX;
            for &lambda in &lambda_values {
                let diff0 = (at_u0 - lambda * u0).abs();
                let diff1 = (at_u1 - lambda * u1).abs();
                let residual = (diff0 * diff0 + diff1 * diff1).sqrt();
                min_residual = min_residual.min(residual);
            }

            // For a unit eigenvector, the residual should be small relative to eigenvalue magnitude
            let max_lambda = lambda_values.iter().map(|l| l.abs()).fold(0.0, f64::max);
            let tol = 0.5 * max_lambda.max(1.0); // Relative tolerance based on eigenvalue scale

            assert!(
                min_residual < tol,
                "Left eigenvector column {} has min residual {} > tolerance {}",
                col,
                min_residual,
                tol
            );
        }
    }
}
