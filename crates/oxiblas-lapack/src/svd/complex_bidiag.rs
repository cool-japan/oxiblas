//! Direct complex bidiagonal reduction using complex Householder reflectors.
//!
//! Reduces a complex m x n matrix A to bidiagonal form: A = U * B * V^H
//! where U and V are unitary matrices and B is a REAL bidiagonal matrix.
//!
//! For m >= n (tall or square):
//!   B is upper bidiagonal with diagonal d[0..n] and superdiagonal e[0..n-1]
//!
//! For m < n (wide):
//!   B is lower bidiagonal with diagonal d[0..m] and subdiagonal e[0..m-1]
//!
//! Algorithm overview:
//! 1. Apply alternating left and right complex Householder reflectors to reduce
//!    A to a complex bidiagonal form (diagonal and off-diagonal are complex).
//! 2. Compute diagonal phase matrices P_U, P_V such that P_U^H * B_complex * P_V
//!    is real bidiagonal.
//! 3. Absorb phases into U and V: U_final = U_raw * P_U, V_final = V_raw * P_V.

use num_traits::Zero;
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::BidiagError;

/// Result of complex bidiagonal reduction.
///
/// Stores the Householder vectors, tau scalars, and bidiagonal entries.
/// The factorization is A = U * B * V^H where B is real bidiagonal.
#[derive(Debug, Clone)]
pub struct ComplexBidiagFactors<T: Scalar> {
    /// Working matrix with Householder vectors stored in the zeroed parts.
    work: Mat<T>,
    /// Complex diagonal elements from the Householder reduction.
    #[allow(dead_code)]
    d_complex: Vec<T>,
    /// Complex off-diagonal elements from the Householder reduction.
    #[allow(dead_code)]
    e_complex: Vec<T>,
    /// Complex Householder scalars for left reflectors (tauq).
    tauq: Vec<T>,
    /// Complex Householder scalars for right reflectors (taup).
    taup: Vec<T>,
    /// Real diagonal of the bidiagonal matrix B.
    pub d: Vec<T::Real>,
    /// Real off-diagonal of the bidiagonal matrix B.
    pub e: Vec<T::Real>,
    /// Phase diagonal for U absorption: U_final = U_raw * diag(phase_u).
    phase_u: Vec<T>,
    /// Phase diagonal for V absorption: V_final = V_raw * diag(phase_v).
    phase_v: Vec<T>,
    /// Original number of rows.
    pub m: usize,
    /// Original number of columns.
    pub n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexBidiagFactors<T>
where
    T::Real: Real,
{
    /// Computes the complex bidiagonal reduction of matrix A.
    ///
    /// A = U * B * V^H where B is real bidiagonal, U and V are unitary.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, BidiagError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(BidiagError::EmptyMatrix);
        }

        if m >= n {
            Self::compute_tall(a)
        } else {
            Self::compute_wide(a)
        }
    }

    /// Bidiagonalize a tall or square complex matrix (m >= n).
    /// Produces upper bidiagonal form.
    fn compute_tall(a: MatRef<'_, T>) -> Result<Self, BidiagError> {
        let m = a.nrows();
        let n = a.ncols();

        let mut work: Mat<T> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                work[(i, j)] = a[(i, j)];
            }
        }

        let mut tauq: Vec<T> = vec![T::zero(); n];
        let num_e = n.saturating_sub(1);
        let mut taup: Vec<T> = vec![T::zero(); num_e];
        let mut d_complex: Vec<T> = vec![T::zero(); n];
        let mut e_complex: Vec<T> = vec![T::zero(); num_e];

        for j in 0..n {
            // Left Householder: zero column j below diagonal
            let (tau, alpha) = complex_householder_column(&mut work, j, j, m);
            d_complex[j] = alpha;
            tauq[j] = tau;
            complex_apply_householder_left(&mut work, j, j, m, n, tau);

            // Right Householder: zero row j right of superdiagonal
            if j < n - 1 {
                let (tau_r, alpha_r) = complex_householder_row(&mut work, j, j + 1, n);
                e_complex[j] = alpha_r;
                taup[j] = tau_r;
                complex_apply_householder_right(&mut work, j, j + 1, m, n, tau_r);
            }
        }

        // Compute phase absorption for upper bidiagonal
        // B_real = P_U^H * B_complex * P_V, where B_complex is upper bidiagonal.
        // B_real[j,j] = conj(p_u[j]) * d_complex[j] * p_v[j]
        // B_real[j,j+1] = conj(p_u[j]) * e_complex[j] * p_v[j+1]
        let mut phase_u: Vec<T> = vec![T::one(); n];
        let mut phase_v: Vec<T> = vec![T::one(); n];
        let mut d: Vec<T::Real> = vec![T::Real::zero(); n];
        let mut e: Vec<T::Real> = vec![T::Real::zero(); num_e];

        // p_u[0] = 1
        // p_v[0]: make d[0] real => conj(1) * d_complex[0] * p_v[0] real
        // => p_v[0] = conj(d_complex[0]) / |d_complex[0]| if nonzero
        let abs_d0 = d_complex[0].abs();
        if abs_d0 > T::Real::epsilon() {
            phase_v[0] = d_complex[0].conj() / T::from_real(abs_d0);
            d[0] = abs_d0;
        }

        for j in 0..num_e {
            // Make e[j] real: conj(p_u[j]) * e_complex[j] * p_v[j+1] real
            // We know p_u[j], so product = conj(p_u[j]) * e_complex[j]
            let prod_e = phase_u[j].conj() * e_complex[j];
            let abs_e = prod_e.abs();
            if abs_e > T::Real::epsilon() {
                phase_v[j + 1] = prod_e.conj() / T::from_real(abs_e);
                e[j] = abs_e;
            }

            // Make d[j+1] real: conj(p_u[j+1]) * d_complex[j+1] * p_v[j+1] real
            let prod_d = d_complex[j + 1] * phase_v[j + 1];
            let abs_d = prod_d.abs();
            if abs_d > T::Real::epsilon() {
                phase_u[j + 1] = prod_d / T::from_real(abs_d);
                d[j + 1] = abs_d;
            }
        }

        Ok(Self {
            work,
            d_complex,
            e_complex,
            tauq,
            taup,
            d,
            e,
            phase_u,
            phase_v,
            m,
            n,
        })
    }

    /// Bidiagonalize a wide complex matrix (m < n).
    /// Produces lower bidiagonal form.
    fn compute_wide(a: MatRef<'_, T>) -> Result<Self, BidiagError> {
        let m = a.nrows();
        let n = a.ncols();

        let mut work: Mat<T> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                work[(i, j)] = a[(i, j)];
            }
        }

        let mut tauq: Vec<T> = vec![T::zero(); m];
        let mut taup: Vec<T> = vec![T::zero(); m];
        let mut d_complex: Vec<T> = vec![T::zero(); m];
        let num_e = if m > 0 { m - 1 } else { 0 };
        let mut e_complex: Vec<T> = vec![T::zero(); num_e];

        for j in 0..m {
            // Right Householder: zero row j right of diagonal
            let (tau_p, alpha_d) = complex_householder_row(&mut work, j, j, n);
            d_complex[j] = alpha_d;
            taup[j] = tau_p;
            complex_apply_householder_right(&mut work, j, j, m, n, tau_p);

            // Left Householder: zero column j below row j+1
            if j < m - 1 {
                let (tau_q, alpha_e) = complex_householder_column(&mut work, j, j + 1, m);
                e_complex[j] = alpha_e;
                tauq[j] = tau_q;
                complex_apply_householder_left(&mut work, j, j + 1, m, n, tau_q);
            }
        }

        if m > 0 {
            tauq[m - 1] = T::zero();
        }

        // Phase absorption for lower bidiagonal
        // B_complex: d on diagonal, e on SUBdiagonal (e[j] at position (j+1, j))
        // B_real[j,j] = conj(p_u[j]) * d_complex[j] * p_v[j]
        // B_real[j+1,j] = conj(p_u[j+1]) * e_complex[j] * p_v[j]
        let mut phase_u: Vec<T> = vec![T::one(); m];
        let mut phase_v: Vec<T> = vec![T::one(); m];
        let mut d: Vec<T::Real> = vec![T::Real::zero(); m];
        let mut e: Vec<T::Real> = vec![T::Real::zero(); num_e];

        // p_u[0] = 1
        // p_v[0]: make d[0] real
        let abs_d0 = if m > 0 {
            d_complex[0].abs()
        } else {
            T::Real::zero()
        };
        if abs_d0 > T::Real::epsilon() {
            phase_v[0] = d_complex[0].conj() / T::from_real(abs_d0);
            d[0] = abs_d0;
        }

        for j in 0..num_e {
            // Make e[j] at (j+1, j) real: conj(p_u[j+1]) * e_complex[j] * p_v[j]
            // p_v[j] is known. product = e_complex[j] * p_v[j]
            let prod_e = e_complex[j] * phase_v[j];
            let abs_e = prod_e.abs();
            if abs_e > T::Real::epsilon() {
                phase_u[j + 1] = prod_e / T::from_real(abs_e);
                e[j] = abs_e;
            }

            // Make d[j+1] at (j+1, j+1) real: conj(p_u[j+1]) * d_complex[j+1] * p_v[j+1]
            let prod_d = phase_u[j + 1].conj() * d_complex[j + 1];
            let abs_d = prod_d.abs();
            if abs_d > T::Real::epsilon() {
                phase_v[j + 1] = prod_d.conj() / T::from_real(abs_d);
                d[j + 1] = abs_d;
            }
        }

        Ok(Self {
            work,
            d_complex,
            e_complex,
            tauq,
            taup,
            d,
            e,
            phase_u,
            phase_v,
            m,
            n,
        })
    }

    /// Returns the real diagonal of the bidiagonal matrix B.
    pub fn diagonal(&self) -> &[T::Real] {
        &self.d
    }

    /// Returns the real off-diagonal of the bidiagonal matrix B.
    pub fn off_diagonal(&self) -> &[T::Real] {
        &self.e
    }

    /// Generates the unitary matrix U explicitly with phase absorption.
    ///
    /// For m >= n: returns m x n matrix (thin U).
    /// For m < n: returns m x m unitary matrix.
    pub fn generate_u(&self) -> Result<Mat<T>, BidiagError> {
        if self.m >= self.n {
            self.generate_u_tall()
        } else {
            self.generate_u_wide()
        }
    }

    /// Generates the unitary matrix V explicitly with phase absorption.
    ///
    /// Returns n x n unitary matrix V such that A = U * B * V^H.
    pub fn generate_v(&self) -> Result<Mat<T>, BidiagError> {
        if self.m >= self.n {
            self.generate_v_tall()
        } else {
            self.generate_v_wide()
        }
    }

    /// Generate U for tall/square case. U_raw = H_0 * H_1 * ... * H_{n-1}.
    fn generate_u_tall(&self) -> Result<Mat<T>, BidiagError> {
        let mut u: Mat<T> = Mat::zeros(self.m, self.m);
        for i in 0..self.m {
            u[(i, i)] = T::one();
        }

        // Accumulate: U = I * H_0 * H_1 * ... (apply each H_j from right)
        for j in 0..self.n {
            let tau = self.tauq[j];
            if tau.abs() > T::Real::zero() {
                // H_j = I - tau * v * v^H
                // U := U * H_j = U - tau * (U * v) * v^H
                // v[j]=1, v[j+1:] in work[j+1:m, j]
                for r in 0..self.m {
                    let mut w = u[(r, j)]; // v[j]=1
                    for i in (j + 1)..self.m {
                        w = w + u[(r, i)] * self.work[(i, j)];
                    }
                    let tw = tau * w;
                    u[(r, j)] = u[(r, j)] - tw;
                    for i in (j + 1)..self.m {
                        u[(r, i)] = u[(r, i)] - tw * self.work[(i, j)].conj();
                    }
                }
            }
        }

        // Phase absorption: U_final[:,j] = U_raw[:,j] * phase_u[j]
        let mut u_thin: Mat<T> = Mat::zeros(self.m, self.n);
        for j in 0..self.n {
            let p = self.phase_u[j];
            for i in 0..self.m {
                u_thin[(i, j)] = u[(i, j)] * p;
            }
        }

        Ok(u_thin)
    }

    /// Generate U for wide case.
    fn generate_u_wide(&self) -> Result<Mat<T>, BidiagError> {
        let mut u: Mat<T> = Mat::zeros(self.m, self.m);
        for i in 0..self.m {
            u[(i, i)] = T::one();
        }

        let num_q = self.m.saturating_sub(1);
        for j in 0..num_q {
            let tau = self.tauq[j];
            if tau.abs() > T::Real::zero() {
                let start = j + 1;
                // v[start]=1, v[start+1:] in work[start+1:m, j]
                for r in 0..self.m {
                    let mut w = u[(r, start)];
                    for i in (start + 1)..self.m {
                        w = w + u[(r, i)] * self.work[(i, j)];
                    }
                    let tw = tau * w;
                    u[(r, start)] = u[(r, start)] - tw;
                    for i in (start + 1)..self.m {
                        u[(r, i)] = u[(r, i)] - tw * self.work[(i, j)].conj();
                    }
                }
            }
        }

        // Phase absorption: U_final[:,j] = U_raw[:,j] * phase_u[j]
        for j in 0..self.m {
            let p = self.phase_u[j];
            for i in 0..self.m {
                u[(i, j)] = u[(i, j)] * p;
            }
        }

        Ok(u)
    }

    /// Generate V for tall/square case.
    fn generate_v_tall(&self) -> Result<Mat<T>, BidiagError> {
        let mut v: Mat<T> = Mat::zeros(self.n, self.n);
        for i in 0..self.n {
            v[(i, i)] = T::one();
        }

        let num_p = self.taup.len();
        for j in 0..num_p {
            let tau = self.taup[j];
            if tau.abs() > T::Real::zero() {
                let start = j + 1;
                // G_j = I - tau * w * w^H  (but stored as column Householder on conj(row))
                // Apply G_j^H from right: V := V * G_j^H = V * (I - conj(tau) * v * v^H)
                // Wait -- we need to be careful about what G_j is.
                //
                // The right Householder was constructed as: we took conj of the row,
                // applied a standard column Householder H, then G = H^H.
                // H = I - tau_h * v * v^H, G = H^H = I - conj(tau_h) * v * v^H.
                //
                // V is accumulated as V = G_0 * G_1 * ... = H_0^H * H_1^H * ...
                //
                // Apply G_j from right: V := V * G_j = V * (I - conj(tau) * v * v^H)
                //   = V - conj(tau) * (V * v) * v^H
                let tau_conj = tau.conj();
                for r in 0..self.n {
                    let mut w = v[(r, start)]; // v[start]=1
                    for i in (start + 1)..self.n {
                        w = w + v[(r, i)] * self.work[(j, i)];
                    }
                    let tw = tau_conj * w;
                    v[(r, start)] = v[(r, start)] - tw;
                    for i in (start + 1)..self.n {
                        v[(r, i)] = v[(r, i)] - tw * self.work[(j, i)].conj();
                    }
                }
            }
        }

        // Phase absorption: V_final[:,j] = V_raw[:,j] * phase_v[j]
        for j in 0..self.n {
            let p = self.phase_v[j];
            for i in 0..self.n {
                v[(i, j)] = v[(i, j)] * p;
            }
        }

        Ok(v)
    }

    /// Generate V for wide case.
    fn generate_v_wide(&self) -> Result<Mat<T>, BidiagError> {
        let mut v: Mat<T> = Mat::zeros(self.n, self.n);
        for i in 0..self.n {
            v[(i, i)] = T::one();
        }

        for j in 0..self.m {
            let tau = self.taup[j];
            if tau.abs() > T::Real::zero() {
                // Same as tall but v starts at j (not j+1)
                let tau_conj = tau.conj();
                for r in 0..self.n {
                    let mut w = v[(r, j)]; // v[j]=1
                    for i in (j + 1)..self.n {
                        w = w + v[(r, i)] * self.work[(j, i)];
                    }
                    let tw = tau_conj * w;
                    v[(r, j)] = v[(r, j)] - tw;
                    for i in (j + 1)..self.n {
                        v[(r, i)] = v[(r, i)] - tw * self.work[(j, i)].conj();
                    }
                }
            }
        }

        // Phase absorption for first m columns (the ones that matter)
        for j in 0..self.m {
            let p = self.phase_v[j];
            for i in 0..self.n {
                v[(i, j)] = v[(i, j)] * p;
            }
        }

        Ok(v)
    }
}

// =============================================================================
// Complex Householder helper functions
// =============================================================================

/// Computes a complex Householder vector for column `col`, rows `start..m`.
///
/// Constructs H = I - tau * v * v^H such that H * x = alpha * e_1
/// where x = work[start:m, col].
///
/// v[0] = 1 (implicit at row `start`), v[1:] stored in work[start+1:m, col].
///
/// Returns (tau, alpha).
fn complex_householder_column<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    col: usize,
    start: usize,
    m: usize,
) -> (T, T)
where
    T::Real: Real,
{
    let mut norm_sq = T::Real::zero();
    for i in start..m {
        norm_sq = norm_sq + work[(i, col)].abs_sq();
    }
    let norm = <T::Real as Real>::sqrt(norm_sq);

    if norm <= T::Real::zero() {
        return (T::zero(), T::zero());
    }

    let x0 = work[(start, col)];
    let x0_abs = x0.abs();

    // alpha = -sign(x0) * ||x||
    let alpha = if x0_abs > T::Real::zero() {
        let sign = T::from_real_imag(x0.real() / x0_abs, x0.imag() / x0_abs);
        T::zero() - sign * T::from_real(norm)
    } else {
        T::from_real(-norm)
    };

    // tau = (alpha - x0) / alpha
    let tau = (alpha - x0) / alpha;

    // v[i] = x[i] / (x0 - alpha) for i > start
    let denom = x0 - alpha;
    if denom.abs() > T::Real::zero() {
        let scale = T::one() / denom;
        for i in (start + 1)..m {
            work[(i, col)] = work[(i, col)] * scale;
        }
    }

    (tau, alpha)
}

/// Computes a complex Householder vector for a row segment.
///
/// Given x = work[row, start:n], finds G such that x * G = alpha * e_1^T.
///
/// Works by constructing a standard column Householder H on y = conj(x):
///   H * y = beta * e_1, then G = H^H, alpha = conj(beta).
///
/// Stores the Householder vector v in work[row, start+1:n] (these are the
/// components of v for the H on conj(x)). v[0] = 1 (implicit at col start).
///
/// Returns (tau_h, alpha) where tau_h is the tau for H (use conj(tau_h) for G).
fn complex_householder_row<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    row: usize,
    start: usize,
    n: usize,
) -> (T, T)
where
    T::Real: Real,
{
    let mut norm_sq = T::Real::zero();
    for i in start..n {
        norm_sq = norm_sq + work[(row, i)].abs_sq();
    }
    let norm = <T::Real as Real>::sqrt(norm_sq);

    if norm <= T::Real::zero() {
        return (T::zero(), T::zero());
    }

    // y[0] = conj(x[0])
    let y0 = work[(row, start)].conj();
    let y0_abs = y0.abs();

    // beta = -sign(y0) * ||y||
    let beta = if y0_abs > T::Real::zero() {
        let sign = T::from_real_imag(y0.real() / y0_abs, y0.imag() / y0_abs);
        T::zero() - sign * T::from_real(norm)
    } else {
        T::from_real(-norm)
    };

    // tau_h = (beta - y0) / beta
    let tau_h = (beta - y0) / beta;

    // v[i] = y[i] / (y0 - beta) = conj(x[i]) / (conj(x[0]) - beta)
    let denom = y0 - beta;
    if denom.abs() > T::Real::zero() {
        let scale = T::one() / denom;
        for i in (start + 1)..n {
            work[(row, i)] = work[(row, i)].conj() * scale;
        }
    }

    // alpha = conj(beta) (the scalar that x*G produces)
    let alpha = beta.conj();

    (tau_h, alpha)
}

/// Applies left Householder H = I - tau * v * v^H to trailing columns.
///
/// v starts at row `start` with v[0]=1 implicit. v[1:] in work[start+1:m, col_store].
/// Applied to columns col_store+1..n.
fn complex_apply_householder_left<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    col_store: usize,
    start: usize,
    m: usize,
    n: usize,
    tau: T,
) where
    T::Real: Real,
{
    if tau.abs() < T::Real::epsilon() {
        return;
    }

    for col in (col_store + 1)..n {
        // w = v^H * work[start:m, col]
        let mut w = work[(start, col)]; // v[start]=1
        for i in (start + 1)..m {
            w = w + work[(i, col_store)].conj() * work[(i, col)];
        }

        let tw = tau * w;
        work[(start, col)] = work[(start, col)] - tw;
        for i in (start + 1)..m {
            work[(i, col)] = work[(i, col)] - tw * work[(i, col_store)];
        }
    }
}

/// Applies right Householder G = H^H = I - conj(tau_h) * v * v^H to trailing rows.
///
/// v starts at col `start` with v[0]=1 implicit. v[1:] in work[row_store, start+1:n].
/// Applied to rows row_store+1..m.
///
/// For a row x: x * G = x - conj(tau_h) * (x * v) * v^H
/// x*v = sum_i x[i]*v[i] (no conjugate on v).
fn complex_apply_householder_right<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    row_store: usize,
    start: usize,
    m: usize,
    n: usize,
    tau_h: T,
) where
    T::Real: Real,
{
    if tau_h.abs() < T::Real::epsilon() {
        return;
    }

    let tau_conj = tau_h.conj();

    for row in (row_store + 1)..m {
        // dot = sum work[row, i] * v[i]
        let mut dot = work[(row, start)]; // v[start]=1
        for i in (start + 1)..n {
            dot = dot + work[(row, i)] * work[(row_store, i)];
        }

        let tw = tau_conj * dot;
        work[(row, start)] = work[(row, start)] - tw; // conj(v[start])=1
        for i in (start + 1)..n {
            work[(row, i)] = work[(row, i)] - tw * work[(row_store, i)].conj();
        }
    }
}

/// Convenience function to compute complex bidiagonal reduction.
/// Equivalent to LAPACK's ZGEBRD/CGEBRD.
pub fn complex_gebrd<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<ComplexBidiagFactors<T>, BidiagError>
where
    T::Real: Real,
{
    ComplexBidiagFactors::compute(a)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn assert_unitary_c64(mat: &Mat<Complex64>, label: &str, tol: f64) {
        let n = mat.nrows();
        let m = mat.ncols();
        for i in 0..m {
            for j in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + mat[(k, i)].conj() * mat[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (sum - Complex64::new(expected, 0.0)).norm();
                assert!(
                    diff < tol,
                    "{}^H * {}[{},{}] error: {} (got ({:.6},{:.6}))",
                    label,
                    label,
                    i,
                    j,
                    diff,
                    sum.re,
                    sum.im,
                );
            }
        }
    }

    fn assert_unitary_c32(mat: &Mat<Complex32>, label: &str, tol: f32) {
        let n = mat.nrows();
        let m = mat.ncols();
        for i in 0..m {
            for j in 0..m {
                let mut sum = Complex32::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + mat[(k, i)].conj() * mat[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (sum - Complex32::new(expected, 0.0)).norm();
                assert!(
                    diff < tol,
                    "{}^H * {}[{},{}] error: {}",
                    label,
                    label,
                    i,
                    j,
                    diff,
                );
            }
        }
    }

    /// Reconstruct A from U, d, e, V for upper bidiagonal (tall/square).
    fn reconstruct_upper_c64(
        u: &Mat<Complex64>,
        d: &[f64],
        e: &[f64],
        v: &Mat<Complex64>,
        m: usize,
        n: usize,
    ) -> Mat<Complex64> {
        let mut b: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..d.len() {
            b[(i, i)] = Complex64::new(d[i], 0.0);
        }
        for i in 0..e.len() {
            b[(i, i + 1)] = Complex64::new(e[i], 0.0);
        }

        let mut bvh: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + b[(i, k)] * v[(j, k)].conj();
                }
                bvh[(i, j)] = sum;
            }
        }

        let u_cols = u.ncols();
        let mut result: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..u_cols {
                    sum = sum + u[(i, k)] * bvh[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    /// Reconstruct A from U, d, e, V for lower bidiagonal (wide).
    /// A = U * B * V^H where B is m x n with lower bidiagonal in m x m block.
    fn reconstruct_lower_c64(
        u: &Mat<Complex64>,
        d: &[f64],
        e: &[f64],
        v: &Mat<Complex64>,
        m: usize,
        n: usize,
    ) -> Mat<Complex64> {
        // B is m x n with lower bidiagonal structure
        let mut b: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..d.len() {
            b[(i, i)] = Complex64::new(d[i], 0.0);
        }
        for i in 0..e.len() {
            b[(i + 1, i)] = Complex64::new(e[i], 0.0);
        }

        // B * V^H (V is n x n)
        let mut bvh: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + b[(i, k)] * v[(j, k)].conj(); // V^H[k,j] = conj(V[j,k])
                }
                bvh[(i, j)] = sum;
            }
        }

        // U * (B * V^H)
        let mut result: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    sum = sum + u[(i, k)] * bvh[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    fn check_reconstruction(
        reconstructed: &Mat<Complex64>,
        original: &Mat<Complex64>,
        tol: f64,
        label: &str,
    ) {
        let m = original.nrows();
        let n = original.ncols();
        for i in 0..m {
            for j in 0..n {
                let diff = (reconstructed[(i, j)] - original[(i, j)]).norm();
                assert!(
                    diff < tol,
                    "{} A[{},{}] error: {} (got ({:.6},{:.6}), exp ({:.6},{:.6}))",
                    label,
                    i,
                    j,
                    diff,
                    reconstructed[(i, j)].re,
                    reconstructed[(i, j)].im,
                    original[(i, j)].re,
                    original[(i, j)].im,
                );
            }
        }
    }

    #[test]
    fn test_complex_bidiag_tall_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(0.5, 0.5),
            ],
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(2.0, -2.0),
            ],
            &[
                Complex64::new(7.0, -1.0),
                Complex64::new(8.0, 0.0),
                Complex64::new(3.0, 1.0),
            ],
            &[
                Complex64::new(10.0, 0.5),
                Complex64::new(11.0, -0.5),
                Complex64::new(4.0, 0.0),
            ],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("compute");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c64(&u, "U", 1e-10);
        assert_unitary_c64(&v, "V", 1e-10);

        let rec = reconstruct_upper_c64(&u, &f.d, &f.e, &v, 4, 3);
        check_reconstruction(&rec, &a, 1e-9, "Tall");
    }

    #[test]
    fn test_complex_bidiag_square_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 2.0),
                Complex64::new(6.0, -1.0),
            ],
            &[
                Complex64::new(7.0, -0.5),
                Complex64::new(8.0, 0.0),
                Complex64::new(9.0, 1.0),
            ],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("compute");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c64(&u, "U", 1e-10);
        assert_unitary_c64(&v, "V", 1e-10);

        let rec = reconstruct_upper_c64(&u, &f.d, &f.e, &v, 3, 3);
        check_reconstruction(&rec, &a, 1e-9, "Square");
    }

    #[test]
    fn test_complex_bidiag_wide_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(4.0, 0.5),
            ],
            &[
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 1.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, -0.5),
            ],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("compute");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c64(&u, "U", 1e-10);
        assert_unitary_c64(&v, "V", 1e-10);

        let rec = reconstruct_lower_c64(&u, &f.d, &f.e, &v, 2, 4);
        check_reconstruction(&rec, &a, 1e-9, "Wide");
    }

    #[test]
    fn test_complex_bidiag_1x1_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(3.0, 4.0)]]);
        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("1x1");
        assert_eq!(f.d.len(), 1);
        assert_eq!(f.e.len(), 0);
        assert!((f.d[0] - 5.0).abs() < 1e-10, "d[0]={}", f.d[0]);
    }

    #[test]
    fn test_complex_bidiag_hermitian_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("hermitian");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c64(&u, "U", 1e-10);
        assert_unitary_c64(&v, "V", 1e-10);

        let rec = reconstruct_upper_c64(&u, &f.d, &f.e, &v, 2, 2);
        check_reconstruction(&rec, &a, 1e-10, "Hermitian");
    }

    #[test]
    fn test_complex_bidiag_identity_c64() {
        let a: Mat<Complex64> = Mat::eye(3);
        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("identity");

        for &di in &f.d {
            assert!((di - 1.0).abs() < 1e-10, "diag={}", di);
        }
        for &ei in &f.e {
            assert!(ei.abs() < 1e-10, "offdiag={}", ei);
        }
    }

    #[test]
    fn test_complex_bidiag_real_matrix_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
            &[
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
                Complex64::new(9.0, 0.0),
            ],
            &[
                Complex64::new(10.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(12.0, 0.0),
            ],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("real");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c64(&u, "U", 1e-10);
        assert_unitary_c64(&v, "V", 1e-10);

        let rec = reconstruct_upper_c64(&u, &f.d, &f.e, &v, 4, 3);
        check_reconstruction(&rec, &a, 1e-9, "Real");
    }

    #[test]
    fn test_complex_bidiag_larger_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.3),
                Complex64::new(2.0, -0.5),
                Complex64::new(3.0, 0.1),
                Complex64::new(4.0, -0.2),
            ],
            &[
                Complex64::new(5.0, 0.7),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, -0.3),
                Complex64::new(8.0, 0.4),
            ],
            &[
                Complex64::new(9.0, -0.1),
                Complex64::new(10.0, 0.6),
                Complex64::new(11.0, 0.0),
                Complex64::new(12.0, -0.5),
            ],
            &[
                Complex64::new(13.0, 0.2),
                Complex64::new(14.0, -0.4),
                Complex64::new(15.0, 0.8),
                Complex64::new(16.0, 0.0),
            ],
            &[
                Complex64::new(0.5, 1.0),
                Complex64::new(1.5, -1.0),
                Complex64::new(2.5, 0.5),
                Complex64::new(3.5, -0.5),
            ],
            &[
                Complex64::new(4.5, 0.0),
                Complex64::new(5.5, 0.3),
                Complex64::new(6.5, -0.7),
                Complex64::new(7.5, 0.1),
            ],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("larger");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c64(&u, "U", 1e-10);
        assert_unitary_c64(&v, "V", 1e-10);

        let rec = reconstruct_upper_c64(&u, &f.d, &f.e, &v, 6, 4);
        check_reconstruction(&rec, &a, 1e-8, "Larger");
    }

    #[test]
    fn test_complex_bidiag_c32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
            &[Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.5)],
            &[Complex32::new(5.0, -0.5), Complex32::new(6.0, 0.0)],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("c32");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c32(&u, "U", 1e-4);
        assert_unitary_c32(&v, "V", 1e-4);

        // Reconstruction
        let m = 3;
        let n = 2;
        let mut b: Mat<Complex32> = Mat::zeros(m, n);
        for i in 0..f.d.len() {
            b[(i, i)] = Complex32::new(f.d[i], 0.0);
        }
        for i in 0..f.e.len() {
            b[(i, i + 1)] = Complex32::new(f.e[i], 0.0);
        }

        let mut bvh: Mat<Complex32> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex32::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + b[(i, k)] * v[(j, k)].conj();
                }
                bvh[(i, j)] = sum;
            }
        }

        let mut rec: Mat<Complex32> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex32::new(0.0, 0.0);
                for k in 0..u.ncols() {
                    sum = sum + u[(i, k)] * bvh[(k, j)];
                }
                rec[(i, j)] = sum;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let diff = (rec[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-4, "C32 A[{},{}] error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_bidiag_wide_c32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[
                Complex32::new(1.0, 0.5),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, -0.5),
                Complex32::new(4.0, 0.3),
            ],
            &[
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, -0.3),
                Complex32::new(7.0, 0.0),
                Complex32::new(8.0, 0.5),
            ],
        ]);

        let f = ComplexBidiagFactors::compute(a.as_ref()).expect("wide c32");
        let u = f.generate_u().expect("u");
        let v = f.generate_v().expect("v");

        assert_unitary_c32(&u, "U", 1e-4);
        assert_unitary_c32(&v, "V", 1e-4);

        // Reconstruction: A = U * B * V^H
        let m = 2;
        let n = 4;
        let mut b: Mat<Complex32> = Mat::zeros(m, n);
        for i in 0..f.d.len() {
            b[(i, i)] = Complex32::new(f.d[i], 0.0);
        }
        for i in 0..f.e.len() {
            b[(i + 1, i)] = Complex32::new(f.e[i], 0.0);
        }

        // B * V^H
        let mut bvh: Mat<Complex32> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex32::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + b[(i, k)] * v[(j, k)].conj();
                }
                bvh[(i, j)] = sum;
            }
        }

        // U * (B * V^H)
        let mut rec: Mat<Complex32> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex32::new(0.0, 0.0);
                for k in 0..m {
                    sum = sum + u[(i, k)] * bvh[(k, j)];
                }
                rec[(i, j)] = sum;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let diff = (rec[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-4, "Wide C32 A[{},{}] error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_gebrd_convenience_c64() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.3)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.7)],
        ]);

        let f = complex_gebrd(a.as_ref()).expect("complex_gebrd");
        assert_eq!(f.d.len(), 2);
        assert_eq!(f.e.len(), 1);
    }
}
